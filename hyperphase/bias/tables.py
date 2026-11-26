"""Per-planet and per-instrument tables of the bias, the spurious redistribution and the retrieval predictions."""
import contextlib
import io
from dataclasses import replace

import numpy as np
import pandas as pd

from hyperphase._util import _normalise
from hyperphase.bands import band_of, detector_label, filter_lambda
from hyperphase.bias.detectors import _detector_cfgs, cfg_detectors, detectors, instrument_cfg, instrument_key, is_detector_set, template_cfg
from hyperphase.bias.fisher import (DEPTH_N_LAYERS, VERDICT_NO_NOISE, VERDICT_UNCONSTRAINED, _false_F_result, bias_verdict, critical_sigma_1min,
                                    critical_time, false_F_fit, fisher_depth, fit_truth, flat_prior_sigma, measurability, model_depth)
from hyperphase.bias.fits import is_railed
from hyperphase.defaults import A_B_BOUNDS, A_B_FIT_BOUNDS, F_BOUNDS
from hyperphase.model import with_planet
from hyperphase.noise import NoNoiseLevelError, rms_1min
from hyperphase.parallel import pmap
from hyperphase.planet import PlanetParams
from hyperphase.stellar import check_teff_nodes
from hyperphase.targets import hi_planets_by_name, planet_archive_value, planet_esm


############################################################################################################################################
#####   Planet lists
############################################################################################################################################

def _planet_items(planets):
    """The items of a planet list: names in any loose spelling (e.g. paper.PAPER_PLANETS), PlanetParams, or {name: PlanetParams}."""
    if isinstance(planets, (str, PlanetParams)): raise TypeError(f"planets must be a list of planets, not a single one ({planets!r}): wrap it in a list")
    return list(planets.values()) if isinstance(planets, dict) else list(planets)


def planet_list(planets):
    """The PlanetParams of a planet list (names looked up with hi_planets_by_name)."""
    return [hi_planets_by_name(p) if isinstance(p, str) else p for p in _planet_items(planets)]


def planet_names(planets):
    """The names of a planet list, without a lookup, so it also takes hypothetical planets that are not in the archive (moved_planet,
    reference_planet)."""
    return list(dict.fromkeys(p if isinstance(p, str) else p.name for p in _planet_items(planets)))


def select_planets(tab, planets, verbose=True):
    """The rows of a per-planet table for `planets`, in that order: what every per-planet plot shows. Planets the table lacks are listed in a note;
    the attrs are kept."""
    names, key = planet_names(planets), tab["name"].map(_normalise).to_numpy()
    rows, missing = [], []
    for nm in names:
        hit = np.flatnonzero(key == _normalise(nm))
        if len(hit): rows.append(hit[0])
        else: missing.append(nm)
    if missing and verbose: print(f"not in the table, so not plotted: {', '.join(missing)}")
    if not rows: raise ValueError(f"none of the planets {names} is in the table (it has {list(tab['name'])})")
    out = tab.iloc[rows].reset_index(drop=True)
    out.attrs = dict(tab.attrs)
    return out


############################################################################################################################################
#####   Per-planet and per-instrument tables
############################################################################################################################################

def drop_unsupported_hosts(cfg, pls, verbose=True):
    """The planets whose host the stellar model can represent. A host outside the NewEra cube (newera.CUBE_*) or off the stellar grid's T_eff nodes
    would raise in the worker pool and abort the whole table, so such planets are left out with a note, checked here in the parent process. A
    band-less template has no grid to check and keeps every planet."""
    keep, dropped = [], []
    for pl in pls:
        try:
            for c in (cfg if is_detector_set(cfg) else [cfg]):   # every detector's grid (a band-less template has none)
                c = with_planet(c, pl)
                if c.stellar_grid is not None: check_teff_nodes(c.stellar_grid, pl.T2_eff)
            keep.append(pl)
        except ValueError as e:
            dropped.append(f"{pl.name} ({e})")
    if dropped and verbose: print(f"no stellar spectrum for {len(dropped)} of {len(pls)} planets, left out of the table:\n  " + "\n  ".join(dropped))
    if not keep: raise ValueError("the stellar model covers none of the planets' hosts, so there is nothing to tabulate")
    return keep


def _has_noise(s1):
    """True if s1 is a noise level: {detector: ppm} for a detector set, a finite number for one band. Without one, see VERDICT_NO_NOISE."""
    return isinstance(s1, dict) or (s1 is not None and np.isfinite(s1))


def _table_noise(sigma_1min, cfg, name):
    """The checked entry of `sigma_1min` ({name: level}) for one planet: a number for one band, {detector: ppm} with every detector for a set (a
    planet lacking one has no noise level, as for a joint fit in instrument_retrievals). NaN or None counts as no noise level (VERDICT_NO_NOISE).
    An infinite level, or a non-finite per-detector level of a set, raises instead of propagating: it enters sigma_bin and so the chi^2 of the
    fit itself, which then returns a wrong bias, not just a NaN sigma."""
    s = sigma_1min.get(name, np.nan)
    if is_detector_set(cfg):
        if s is None or (not isinstance(s, dict) and np.isnan(s)): return np.nan
        if not isinstance(s, dict): raise ValueError(f"sigma_1min for {name}: a detector set needs {{detector: ppm}} (rms_1min_pandexo_sample(names, instrument) gives it), got {s!r}")
        dets = cfg_detectors(cfg)
        if any(d not in s for d in dets): return np.nan
        if not all(np.isfinite(s[d]) for d in dets): raise ValueError(f"non-finite sigma_1min for {name}: {s}")
        return {d: float(s[d]) for d in dets}
    if isinstance(s, dict): raise ValueError(f"sigma_1min for {name}: a single band needs a number, got the per-detector levels {s} (a joint fit needs cfg to be the detector set)")
    if s is not None and not np.isnan(s) and not np.isfinite(s): raise ValueError(f"non-finite sigma_1min for {name}: {s}")
    return np.nan if s is None else float(s)


def _planet_cfg(cfg, pl, s1):
    """The cfg or detector set of one planet of a per-planet table: cfg with the planet (with_planet, which rebuilds the stellar grid and band) and
    its noise level s1 (without one, cfg's placeholder noise)."""
    if not is_detector_set(cfg): return with_planet(cfg, pl, sigma_1min=float(s1) if _has_noise(s1) else cfg.sigma_1min)
    return [with_planet(c, pl, sigma_1min=s1[band_of(c)] if _has_noise(s1) else c.sigma_1min) for c in cfg]


def _noise_columns(cfg, s1):
    """The noise columns of a per-planet table row: sigma_1min (NaN for a detector set) and, for a set, sigma_1min:<label> per detector
    (detector_label)."""
    if not is_detector_set(cfg): return dict(sigma_1min=float(s1) if _has_noise(s1) else np.nan)
    return dict(sigma_1min=np.nan, **{f"sigma_1min:{detector_label(d)}": (s1[d] if _has_noise(s1) else np.nan) for d in cfg_detectors(cfg)})


def _per_detector_crit(tab, factor):
    """sigma_1min_crit:<label> = sigma_1min:<label> * factor for a joint fit's per-detector noise columns (the factor is common to the set)."""
    for col in [c for c in tab.columns if c.startswith("sigma_1min:")]:
        tab["sigma_1min_crit:" + col.split(":", 1)[1]] = tab[col] * factor
    return tab


def _planet_bias_row(cfg, A_B_true, pl, s1, redist, bounds, F_bounds, kw):
    """One row of planet_bias_table: the planet's own geometry and noise, everything else from cfg."""
    has_noise = _has_noise(s1)   # see VERDICT_NO_NOISE
    c = _planet_cfg(cfg, pl, s1)
    model = "naive_redist" if redist else "naive"
    fit = fit_truth(c, A_B_true, models=(model,), free_offset=True, bounds=bounds, F_bounds=F_bounds, fisher=has_noise, **kw)[model]
    return dict(name=pl.name, a_rs=pl.R / pl.r2, period=pl.period, T_eff=pl.T2_eff, bias_A_B=fit["bias_A_B"], sigma_A_B=fit.get("sigma_A_B", np.nan),
                F_fit=fit.get("F_fit", np.nan), sigma_F=fit.get("sigma_F", np.nan), **_noise_columns(cfg, s1), has_noise=has_noise,
                # for planet_false_F_table(fits=...): chi2 at the planet's own noise, and the fitted A_B itself, since bias_A_B + A_B_true can differ
                # from it in the last bit
                chi2=fit["chi2"] if has_noise else np.nan, A_B_fit_exact=fit["A_B_fit"])


def planet_bias_table(cfg, A_B_true, planets, sigma_1min, redist=False, quantity=None, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, AGG=None, n_jobs=-1, verbose=True, **kw):
    """Bias, Fisher sigma and critical noise for a sample of planets in one band or detector set, each at its own geometry and noise, sorted by a/Rs.
    A_B_true is the same assumed albedo for every planet. planets: see _planet_items. sigma_1min: {name: ppm}, e.g. from
    noise.rms_1min_pandexo_sample, or {name: {detector: ppm}} for a detector set, whose scalar sigma_1min and sigma_1min_crit columns are then
    NaN and the sigma_1min:<label> columns carry the levels (noise_cols). A planet without an entry, or with NaN, is fitted at
    cfg.sigma_1min (VERDICT_NO_NOISE); an infinite entry raises (_table_noise). quantity: "A_B" (default) or "F" (redist=True only), the
    parameter the bias, sigma, bias_sigma, sigma_1min_crit and verdict columns refer to; the truth has F = 0, so F_fit is the bias in F. For F,
    bias_sigma is F_fit / sigma_F (Fisher), not the significance of the spurious F: that is z_F = sqrt(dchi2_F) of planet_false_F_table
    (see bias.fisher.false_F_fit).
    Verdicts: bias_verdict. AGG: the archive aggregate path; with it the table carries each planet's ESM (esm) and radius (rp_rearth)."""
    quantity = quantity or "A_B"
    if quantity not in ("A_B", "F"): raise ValueError(f'quantity must be "A_B" or "F", got {quantity!r}')
    if quantity == "F" and not redist: raise ValueError('quantity="F" needs redist=True (only the joint fit has an F)')
    pls = planet_list(planets)
    noise = {pl.name: _table_noise(sigma_1min, cfg, pl.name) for pl in pls}   # see _table_noise
    no_noise = [nm for nm, s in noise.items() if not _has_noise(s)]
    if no_noise and verbose: print(f"no sigma_1min for {len(no_noise)} of {len(pls)} planets, fitted without one (the fitted A_B does not depend on it), so no Fisher sigma: {', '.join(no_noise)}")
    c0 = template_cfg(cfg)
    if not np.isfinite(c0.sigma_1min): raise ValueError(f"cfg.sigma_1min = {c0.sigma_1min}: the fits without a noise level run at it, so it must be finite")
    pls = drop_unsupported_hosts(cfg, pls, verbose=verbose)

    rows = pmap(_planet_bias_row, [(cfg, A_B_true, pl, noise[pl.name], redist, bounds, F_bounds, kw) for pl in pls], n_jobs)
    tab = pd.DataFrame(rows).sort_values("a_rs").reset_index(drop=True)
    tab["esm"] = [planet_esm(AGG, nm) for nm in tab["name"]] if AGG is not None else np.nan
    tab["rp_rearth"] = [planet_archive_value(AGG, nm, "pl_rade") for nm in tab["name"]] if AGG is not None else np.nan   # [R_Earth]
    tab["t_obs"] = c0.n_orbits * tab["period"] * 24.   # [h]: n_orbits full orbits, so it differs per planet
    tab["A_B_fit"] = tab["bias_A_B"] + A_B_true   # the fitted value itself

    tab.attrs.update(A_B_true=A_B_true, n_orbits=c0.n_orbits, model="naive_redist" if redist else "naive", bounds=bounds, F_bounds=F_bounds, detectors=cfg_detectors(cfg))
    # physical ranges (flat_prior_sigma), per quantity
    tab.attrs["quantity"], tab.attrs["sigma_prior"] = quantity, {"A_B": flat_prior_sigma(A_B_BOUNDS), "F": flat_prior_sigma(F_BOUNDS)}
    tab["bias"], tab["sigma"] = tab[f"bias_{quantity}" if quantity == "A_B" else "F_fit"], tab[f"sigma_{quantity}"]
    # at each planet's own sigma_1min (NaN for a detector set: per detector below)
    tab["bias_sigma"], tab["sigma_1min_crit"] = measurability(tab["bias"], tab["sigma"], tab["sigma_1min"])
    _per_detector_crit(tab, tab["bias_sigma"])
    tab["verdict"] = bias_verdict(tab["bias_sigma"], tab["sigma"], tab.attrs["sigma_prior"][quantity])
    tab.loc[~tab["has_noise"], "verdict"] = VERDICT_NO_NOISE
    tab["t_crit"] = critical_time(tab["t_obs"], tab["bias_sigma"], tab["verdict"])   # [h], see critical_time
    return tab


def noise_cols(tab):
    """The noise columns of a table to print: sigma_1min and sigma_1min_crit, or a joint fit's per-detector sigma_1min:<label> and
    sigma_1min_crit:<label> (its scalar columns are NaN). The per-instrument tables keep the scalar columns, with a joint fit's levels in
    tab.attrs["sigma_1min"]."""
    per = [c for c in tab.columns if c.startswith("sigma_1min:")]
    if not per: return [c for c in ("sigma_1min", "sigma_1min_crit") if c in tab.columns]
    return per + [c for c in tab.columns if c.startswith("sigma_1min_crit:")]


def railed_names(tab, A_B_true, bounds=None):
    """The planets of a planet_bias_table whose A_B fit railed on a bound of `bounds` (default: the table's own); the table keeps their fitted
    values."""
    bounds = tab.attrs.get("bounds", A_B_FIT_BOUNDS) if bounds is None else bounds
    return list(tab.loc[is_railed(tab["bias_A_B"] + A_B_true, bounds), "name"])


def drop_unconstrained(tab, like=None):
    """A planet_bias_table without its unconstrained planets (VERDICT_UNCONSTRAINED). like: drop those unconstrained in that table instead (e.g. the
    joint fit's), so two fits keep the same planets. The attrs are copied, since pandas does not carry them through every selection."""
    keep = (like if like is not None else tab)
    names = set(keep.loc[keep["verdict"] != VERDICT_UNCONSTRAINED, "name"])
    out = tab[tab["name"].isin(names)].reset_index(drop=True)
    out.attrs = dict(tab.attrs)
    return out


# Verdicts of the false-positive tables (_false_F_decision).
FALSE_POSITIVE, NO_FALSE_POSITIVE = "false positive", "below threshold"


def _false_F_decision(tab, z):
    """The decision columns of a false-positive table, from dchi2_F at each row's own noise (false_F_fit): z_F = sqrt(dchi2_F), the expected
    significance of the spurious F; noise_margin = z_F / z = sigma_1min_crit / sigma_1min, the factor by which the noise may be worse (> 1) or
    must be better (< 1) for a z-sigma false positive, also for a joint fit (all noise levels scaled alike); sigma_1min_crit (critical_sigma_1min;
    NaN for a joint fit, per detector in its per-planet table); t_crit [h] = t_obs (z / z_F)^2, the observing time at which the spurious F reaches
    z sigma; verdict FALSE_POSITIVE where z_F >= z, "-" on a rail."""
    tab["z_F"] = np.sqrt(tab["dchi2_F"])
    tab["noise_margin"] = tab["z_F"] / z
    tab["sigma_1min_crit"] = critical_sigma_1min(tab["dchi2_F"], tab["sigma_1min"], z=z)   # NaN for a joint fit's per-planet rows: per detector below
    _per_detector_crit(tab, tab["noise_margin"])
    tab["t_crit"] = tab["t_obs"] * (z / tab["z_F"]) ** 2
    tab["verdict"] = np.where(~np.isfinite(tab["z_F"]), "-", np.where(tab["z_F"] >= z, FALSE_POSITIVE, NO_FALSE_POSITIVE))
    tab.attrs["z"] = z
    return tab


def _bias_table_pair(fits, A_B_true, bounds, F_bounds, n_orbits):
    """Checks fits=(tab, tab_J) of the false-positive tables: the A_B-only and the joint bias table of the same sample (planet_bias_table or
    instrument_retrievals with redist=False and True). Their two fits per row are those of false_F_fit (same truth, bounds and tolerances), so
    their chi2 columns give dchi2_F without refitting. That they come from this cfg and **kw is not checked; the models, A_B_true, bounds,
    n_orbits and (by the callers) each row's noise level are, since chi2 ∝ 1 / sigma_1min^2."""
    if len(fits) != 2: raise ValueError("fits: (A_B-only table, joint table) expected")
    want = (A_B_true, tuple(bounds), tuple(F_bounds), n_orbits)
    for tab, model in zip(fits, ("naive", "naive_redist")):
        a = tab.attrs
        if a.get("model") != model: raise ValueError(f"fits: (A_B-only table, joint table) expected, got the models {[t.attrs.get('model') for t in fits]}")
        got = (a["A_B_true"], tuple(a["bounds"]), tuple(a["F_bounds"]), a["n_orbits"])
        if got != want: raise ValueError(f"fits: the {model} table has (A_B_true, bounds, F_bounds, n_orbits) = {got}, this call {want}")
        if "chi2" not in tab: raise ValueError(f"fits: the {model} table has no chi2 column: re-make it with planet_bias_table or instrument_retrievals")
    return fits


def _table_fit_pair(tabs, col, key):
    """The fits (n, r) of false_F_fit for row `key` of a _bias_table_pair, as fit_truth-like dicts, and the two table rows. The fitted A_B is
    A_B_fit_exact where the table has it (planet_bias_table rebuilds A_B_fit from the bias), so the result is bit-identical to refitting."""
    rows = []
    for tab in tabs:
        sel = tab[tab[col] == key]
        if len(sel) != 1: raise ValueError(f"fits: {key} is not in the {tab.attrs['model']} table")
        rows.append(sel.iloc[0])
    A = lambda row: float(row["A_B_fit_exact"] if "A_B_fit_exact" in row else row["A_B_fit"])
    n, r = rows
    return dict(A_B_fit=A(n), chi2=float(n["chi2"])), dict(A_B_fit=A(r), F_fit=float(r["F_fit"]), chi2=float(r["chi2"])), rows


def _same_noise(a, b):
    return bool(np.isclose(a, b, rtol=1e-9, atol=0.))


def _planet_false_F_row(cfg, A_B_true, pl, s1, bounds, F_bounds, kw):
    c = _planet_cfg(cfg, pl, s1)
    return dict(name=pl.name, a_rs=pl.R / pl.r2, period=pl.period, T_eff=pl.T2_eff, **_noise_columns(cfg, s1), **false_F_fit(c, A_B_true, bounds=bounds, F_bounds=F_bounds, **kw))


def _planet_false_F_row_from(tabs, cfg, pl, s1, bounds, F_bounds):
    """The row of _planet_false_F_row from the fits of a _bias_table_pair (no refit)."""
    n, r, rows = _table_fit_pair(tabs, "name", pl.name)
    noise = _noise_columns(cfg, s1)
    for tab, row in zip(tabs, rows):
        for col, lv in noise.items():   # the scalar level, or a set's per-detector levels (scalar NaN on both sides)
            if np.isfinite(lv) and not _same_noise(row[col], lv): raise ValueError(f"fits: {pl.name} was fitted at {col} = {row[col]} ppm in the {tab.attrs['model']} table, this call has {lv}")
    return dict(name=pl.name, a_rs=pl.R / pl.r2, period=pl.period, T_eff=pl.T2_eff, **noise, **_false_F_result(n, r, pl, template_cfg(cfg).emissivity_bol, bounds, F_bounds))


def planet_false_F_table(cfg, A_B_true, planets, sigma_1min, z=2., bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, AGG=None, n_jobs=-1, fits=None, verbose=True, **kw):
    """False-positive F for a sample of planets in one band or detector set, each at its own geometry and noise, sorted by a/Rs: the counterpart of
    planet_bias_table, with the same inputs, except that planets without a noise level are left out. z: the significance the decision refers to
    (_false_F_decision). fits=(tab, tab_J): the planet_bias_table of this sample with redist=False and True, reused instead of refitting
    (_bias_table_pair)."""
    pls = planet_list(planets)
    noise = {pl.name: _table_noise(sigma_1min, cfg, pl.name) for pl in pls}   # as in planet_bias_table
    missing = [nm for nm, s in noise.items() if not _has_noise(s)]
    if len(missing) == len(pls): raise ValueError(f"no planet has a sigma_1min entry (planets: {missing}; sigma_1min has {sorted(sigma_1min)}), so there is nothing to tabulate")
    if missing:
        if verbose: print(f"no sigma_1min for {len(missing)} of {len(pls)} planets, left out of the table: {', '.join(missing)}")
        pls = [pl for pl in pls if pl.name not in missing]
    c0 = template_cfg(cfg)
    pls = drop_unsupported_hosts(cfg, pls, verbose=verbose)
    if fits is None:
        rows = pmap(_planet_false_F_row, [(cfg, A_B_true, pl, noise[pl.name], bounds, F_bounds, kw) for pl in pls], n_jobs)
    else:
        tabs = _bias_table_pair(fits, A_B_true, bounds, F_bounds, c0.n_orbits)
        rows = [_planet_false_F_row_from(tabs, cfg, pl, noise[pl.name], bounds, F_bounds) for pl in pls]
    tab = pd.DataFrame(rows).sort_values("a_rs").reset_index(drop=True)
    tab["esm"] = [planet_esm(AGG, nm) for nm in tab["name"]] if AGG is not None else np.nan
    tab["t_obs"] = c0.n_orbits * tab["period"] * 24.   # [h]
    tab.attrs.update(A_B_true=A_B_true, n_orbits=c0.n_orbits, detectors=cfg_detectors(cfg))
    return _false_F_decision(tab, z)


def _instrument_false_F_row(cfg, A_B_true, pl, instrument, sigma_1min, bounds, F_bounds, kw):
    c = _detector_cfgs(cfg, pl, instrument, sigma_1min)
    dets = detectors(instrument)
    return dict(instrument=instrument, sigma_1min=float(sigma_1min[dets[0]]) if len(dets) == 1 else np.nan, **false_F_fit(c, A_B_true, bounds=bounds, F_bounds=F_bounds, **kw))


def _instrument_false_F_row_from(tabs, cfg, pl, instrument, sigma_1min, bounds, F_bounds):
    """The row of _instrument_false_F_row from the fits of a _bias_table_pair (no refit)."""
    dets = detectors(instrument)
    n, r, _ = _table_fit_pair(tabs, "instrument", instrument)
    for tab in tabs:
        if tab.attrs.get("planet") != pl.name: raise ValueError(f"fits: the {tab.attrs['model']} table is of {tab.attrs.get('planet')}, not {pl.name}")
        for d in dets:
            s_tab = tab.attrs["sigma_1min"].get(d, np.nan)
            if not _same_noise(s_tab, sigma_1min[d]): raise ValueError(f"fits: {d} was fitted at sigma_1min = {s_tab} ppm in the {tab.attrs['model']} table, this call has {sigma_1min[d]}")
    return dict(instrument=instrument, sigma_1min=float(sigma_1min[dets[0]]) if len(dets) == 1 else np.nan, **_false_F_result(n, r, pl, template_cfg(cfg).emissivity_bol, bounds, F_bounds))


def instrument_false_F(cfg, planet, instruments, A_B_true, z=2., sigma_1min=None, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, n_jobs=-1, fits=None, verbose=True, **kw):
    """False-positive F for one planet in several instruments: the counterpart of instrument_retrievals, with the same `instruments` entries, except
    that instruments without a noise level are left out. A joint fit's sigma_1min_crit is NaN; its noise_margin applies to all its detectors
    alike. fits=(tab, tab_J): the instrument_retrievals of this planet with redist=False and True, reused instead of refitting (_bias_table_pair)."""
    c0 = template_cfg(cfg)
    instruments = [instrument_key(ins) for ins in instruments]
    dets = list(dict.fromkeys(d for ins in instruments for d in detectors(ins)))
    if sigma_1min is None:
        sigma_1min, missing = {}, []
        for d in dets:
            try:
                with contextlib.redirect_stdout(io.StringIO()): sigma_1min[d] = rms_1min(planet, d)
            except NoNoiseLevelError:
                missing.append(d)
        if missing and verbose: print(f"{planet.name}: no noise level on {', '.join(missing)} (no PandExo mode and no CUSTOM_SIGMA_1MIN entry, or not observable: UNOBSERVABLE / fully saturated), left out")
    used = [ins for ins in instruments if all(d in sigma_1min for d in detectors(ins))]
    if not used: raise ValueError(f"{planet.name}: no noise level for any of {instruments}")
    if fits is None:
        rows = pmap(_instrument_false_F_row, [(cfg, A_B_true, planet, ins, sigma_1min, bounds, F_bounds, kw) for ins in used], n_jobs)
    else:
        tabs = _bias_table_pair(fits, A_B_true, bounds, F_bounds, c0.n_orbits)
        rows = [_instrument_false_F_row_from(tabs, cfg, planet, ins, sigma_1min, bounds, F_bounds) for ins in used]
    tab = pd.DataFrame(rows)
    tab["t_obs"] = c0.n_orbits * planet.period * 24.   # [h]
    tab.attrs.update(planet=planet.name, A_B_true=A_B_true, n_orbits=c0.n_orbits, sigma_1min={d: float(sigma_1min[d]) for d in dets if d in sigma_1min})
    return _false_F_decision(tab, z)


def _instrument_row(cfg, A_B_true, pl, instrument, sigma_1min, redist, bounds, F_bounds, kw):
    dets = detectors(instrument)
    has_noise = all(d in sigma_1min for d in dets)   # False only for a single detector: see VERDICT_NO_NOISE
    c = _detector_cfgs(cfg, pl, instrument, sigma_1min) if has_noise else with_planet(instrument_cfg(cfg, instrument), pl)
    model = "naive_redist" if redist else "naive"
    fit = fit_truth(c, A_B_true, models=(model,), free_offset=True, bounds=bounds, F_bounds=F_bounds, fisher=has_noise, **kw)[model]
    # a joint fit has one noise level per detector: see tab.attrs["sigma_1min"]
    return dict(instrument=instrument, sigma_1min=float(sigma_1min[dets[0]]) if len(dets) == 1 and has_noise else np.nan,
                A_B_fit=fit["A_B_fit"], bias_A_B=fit["bias_A_B"], sigma_A_B=fit.get("sigma_A_B", np.nan), F_fit=fit.get("F_fit", np.nan), sigma_F=fit.get("sigma_F", np.nan),
                chi2=fit["chi2"] if has_noise else np.nan)   # chi2 at the detectors' own noise, for instrument_false_F(fits=...)


def instrument_retrievals(cfg, planet, instruments, A_B_true, redist=False, sigma_1min=None, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, n_jobs=-1, verbose=True, **kw):
    """One planet, several instruments: the fitted A_B and its Fisher sigma per instrument for an extended-source truth at A_B_true (the
    multi-instrument counterpart of planet_bias_table). cfg supplies everything but the band. An entry of `instruments` is a band key, or a tuple
    of keys for a joint fit (its row carries the "+"-joined key and NaN as sigma_1min). sigma_1min: {detector: ppm}; by default noise.rms_1min
    per detector. The returned attrs["sigma_1min"] can be passed back in to refit with the same levels. A single detector without a noise level is
    fitted without one (VERDICT_NO_NOISE); a joint fit lacking one is left out, since it weights its detectors by their noise. Returns a DataFrame
    in the order of `instruments`, with the verdict (bias_verdict) of each fit."""
    c0 = template_cfg(cfg)   # its instrument-level settings; each instrument brings its own bands
    instruments = [instrument_key(ins) for ins in instruments]
    dets = list(dict.fromkeys(d for ins in instruments for d in detectors(ins)))
    if sigma_1min is None:
        sigma_1min = {}
        for d in dets:
            try:
                with contextlib.redirect_stdout(io.StringIO()): sigma_1min[d] = rms_1min(planet, d)
            except NoNoiseLevelError:
                pass
    no_noise = [ins for ins in instruments if len(detectors(ins)) == 1 and ins not in sigma_1min]
    joint_out = [ins for ins in instruments if len(detectors(ins)) > 1 and not all(d in sigma_1min for d in detectors(ins))]
    if no_noise and verbose: print(f"{planet.name}: no noise level on {', '.join(no_noise)} (no PandExo mode and no CUSTOM_SIGMA_1MIN entry, or not observable: UNOBSERVABLE / fully saturated): fitted without one (the fitted A_B does not depend on it), so no Fisher sigma")
    if joint_out and verbose: print(f"{planet.name}: joint fit(s) {', '.join(joint_out)} left out: a joint fit weights its detectors by their noise levels, and not all of them have one")
    if no_noise and not np.isfinite(c0.sigma_1min): raise ValueError(f"cfg.sigma_1min = {c0.sigma_1min}: the fits without a noise level run at it, so it must be finite")
    used = [ins for ins in instruments if ins not in joint_out]
    if not used: raise ValueError(f"{planet.name}: nothing to fit among {instruments}")
    rows = pmap(_instrument_row, [(cfg, A_B_true, planet, ins, sigma_1min, redist, bounds, F_bounds, kw) for ins in used], n_jobs)
    tab = pd.DataFrame(rows)
    tab.attrs.update(planet=planet.name, A_B_true=A_B_true, model="naive_redist" if redist else "naive", sigma_prior=flat_prior_sigma(A_B_BOUNDS), n_orbits=c0.n_orbits,
                     bounds=bounds, F_bounds=F_bounds, sigma_1min={d: float(sigma_1min[d]) for d in dets if d in sigma_1min})
    bias_sigma, _ = measurability(tab["bias_A_B"], tab["sigma_A_B"], tab["sigma_1min"])
    tab["bias_sigma"], tab["verdict"] = bias_sigma, bias_verdict(bias_sigma, tab["sigma_A_B"], tab.attrs["sigma_prior"])
    tab.loc[tab["instrument"].isin(no_noise), "verdict"] = VERDICT_NO_NOISE
    return tab


def _instrument_depth_row(cfg, planet, r, det, sigma_1min, model, A_B_true, bounds, F_bounds, n_layers, kw):
    F = None if model == "naive" else float(r["F_fit"])
    if all(d in sigma_1min for d in detectors(r["instrument"])):
        c = _detector_cfgs(cfg, planet, r["instrument"], sigma_1min)   # the fit's cfg or detector set (its covariance), ...
        c_det = c[detectors(r["instrument"]).index(det)] if is_detector_set(c) else c   # ... and the detector whose band the depth is in
        depth_fit, sigma_depth = fisher_depth(c, model, float(r["A_B_fit"]), F=F, bounds=bounds, F_bounds=F_bounds, n_layers=n_layers, depth_cfg=c_det, **kw)
    else:   # no noise level (VERDICT_NO_NOISE): the depth only, no Fisher sigma
        c_det = with_planet(instrument_cfg(cfg, det), planet)
        depth_fit, sigma_depth = model_depth(c_det, model, float(r["A_B_fit"]), F=F, n_layers=n_layers, **kw), np.nan
    depth_true = model_depth(c_det, "hyperillum", A_B_true, n_layers=n_layers, **kw)
    return dict(instrument=det, fit=r["instrument"], sigma_1min=float(sigma_1min.get(det, np.nan)), depth_true=depth_true, depth_fit=depth_fit, sigma_depth=sigma_depth)


def instrument_depths(cfg, planet, tab, n_layers=DEPTH_N_LAYERS, joint=False, n_jobs=-1, **kw):
    """The occultation depth of each fit of an instrument_retrievals table next to that of the extended-source truth, from the fitted parameters in
    `tab` (no refit, so cfg, planet and kw must be those that made it). Both depths are computed directly at mid-occultation (model_depth at
    n_layers), sigma_depth is the propagated Fisher sigma (fisher_depth). Adds bias_depth = depth_fit - depth_true [ppm] and
    depth_bias_sigma = |bias_depth| / sigma_depth. A depth is per band, so joint fits are left out by default; joint=True gives one row per
    detector of a joint fit, with that band's depth and the joint covariance."""
    a = tab.attrs
    rows_in = [r for _, r in tab.iterrows() if joint or len(detectors(r["instrument"])) == 1]
    jobs = [(cfg, planet, r, d, a["sigma_1min"], a["model"], a["A_B_true"], a["bounds"], a["F_bounds"], n_layers, kw) for r in rows_in for d in detectors(r["instrument"])]
    rows = pmap(_instrument_depth_row, jobs, n_jobs)
    params = tab.drop(columns=["instrument", "sigma_1min"]).set_index(tab["instrument"])
    out = pd.DataFrame([{**r, **params.loc[r["fit"]].to_dict()} for r in rows])
    out.attrs.update(a, depth_n_layers=n_layers)
    out["bias_depth"] = out["depth_fit"] - out["depth_true"]
    out["depth_bias_sigma"] = np.abs(out["bias_depth"]) / out["sigma_depth"]
    return out


############################################################################################################################################
#####   Table display
############################################################################################################################################

def display_formatted(df, columns, index="name", transpose=False):
    """Display the columns {column: (header, format)} of df under their headers, NaN as "-"; transpose: quantities as rows (for a wide table)."""
    from IPython.display import display
    cell = lambda v, f: "-" if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)) else f.format(v)
    t = pd.DataFrame({h: [cell(v, f) for v in df[c]] for c, (h, f) in columns.items()}, index=pd.Index(list(df[index]), name=None))
    display(t.T if transpose else t)


def band_range_um(instrument):
    '''(blue, red) edge [um] of a band; a joint fit of several detectors spans all of them.'''
    lam = 1e6 * np.array([filter_lambda(d) for d in detectors(instrument)], dtype=float)
    return lam.min(), lam.max()
