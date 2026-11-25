"""Bias and spurious redistribution over a grid of (a/R*, T_eff), and the Roche limit on that grid."""
from dataclasses import replace

import numpy as np
import pandas as pd
from astropy import constants as const
from scipy.interpolate import RegularGridInterpolator

from hyperphase.bias.fisher import false_F_fit, fit_truth
from hyperphase.bias.fits import RAIL_TOL, fit_railed, temperature_bias
from hyperphase.defaults import A_B_FIT_BOUNDS, DEFAULT_MAP_RP_RS, F_BOUNDS
from hyperphase.model import swept_planet
from hyperphase.parallel import pmap
from hyperphase.planet import PlanetParams
from hyperphase.roche import ROCHE_CASES, ROCHE_R_P, kepler_a_rs, kepler_period_days, rho_star, roche_period_h, rocky_mean_density


############################################################################################################################################
#####   Bias maps
############################################################################################################################################
def bias_cell(cfg, A_B_true, free_offset=True, rail_tol=RAIL_TOL, temp_bias=False, bounds=A_B_FIT_BOUNDS, redist=False, F_bounds=F_BOUNDS, return_chi2=False, **kw):
    """One cell of a bias map: the extended-source truth fitted with the point-source model; returns the bias of the fitted A_B (temp_bias=True: the
    temperature bias, temperature_bias). redist=True fits "naive_redist" and returns (bias, F_fit). return_chi2=True appends the fit's chi2 at
    cfg.sigma_1min. Everything is NaN on a rail (RAIL_TOL)."""
    model = "naive_redist" if redist else "naive"
    # same bounds the rail check below uses
    fit = fit_truth(cfg, A_B_true, models=(model,), free_offset=free_offset, bounds=bounds, F_bounds=F_bounds, **kw)[model]
    A_B_fit, F_fit = fit["A_B_fit"], fit.get("F_fit", 0.)
    railed = fit_railed(fit, bounds, F_bounds, rail_tol)
    if railed:
        out = (np.nan, np.nan) if redist else (np.nan,)
    else:
        A_B_bias = A_B_fit - A_B_true
        out = (temperature_bias(A_B_true, A_B_bias, F_fit) if temp_bias else A_B_bias,) + ((F_fit,) if redist else ())
    if return_chi2: out += (np.nan if railed else fit["chi2"],)
    return out if len(out) > 1 else out[0]


def _bias_cell_at(cfg, A_B_true, a_rs, y_param, y, redist, return_chi2, kw):
    return bias_cell(cfg, A_B_true, a_rs=a_rs, redist=redist, return_chi2=return_chi2, **{y_param: y}, **kw)


def bias_map_grid(cfg, A_B_true, a_rs_grid, y_grid, y_param, redist=False, n_jobs=-1, return_chi2=False, **kw):
    """A bias map over (a_rs_grid, y_grid), cells in parallel; y_param is the swept model_pcurve keyword ("T_s" or "rp_rs"). Returns the A_B-bias
    map, shape (len(y_grid), len(a_rs_grid)); redist=True adds the F map, return_chi2=True the chi2 map (for false_F_map_grid(fits=...))."""
    flat = pmap(_bias_cell_at, [(cfg, A_B_true, ar, y_param, y, redist, return_chi2, kw) for y in y_grid for ar in a_rs_grid], n_jobs)
    n_out = 1 + int(redist) + int(return_chi2)
    out = np.array(flat).reshape(len(y_grid), len(a_rs_grid), *((n_out,) if n_out > 1 else ()))
    return tuple(np.moveaxis(out, -1, 0)) if n_out > 1 else out


def _bias_vs_a_rs_point(cfg, A_B_true, a_rs, rho_star, models, bounds, F_bounds, kw):
    c = cfg if rho_star is None else replace(cfg, planet=replace(cfg.planet, period=float(kepler_period_days(a_rs, rho_star))))
    fit = fit_truth(c, A_B_true, models=models, free_offset=True, bounds=bounds, F_bounds=F_bounds, fisher=True, a_rs=a_rs, **kw)
    row = dict(a_rs=a_rs, period_h=24. * c.planet.period, sigma_bin=c.sigma_bin)
    if "naive" in fit:
        n = fit["naive"]
        row.update(A_B_fit=n["A_B_fit"], bias=n["bias_A_B"], sigma=n["sigma_A_B"], chi2=n["chi2"], railed=fit_railed(n, bounds, F_bounds))
    if "naive_redist" in fit:
        j = fit["naive_redist"]
        row.update(A_B_fit_J=j["A_B_fit"], bias_J=j["bias_A_B"], sigma_J=j["sigma_A_B"], F_fit=j["F_fit"], sigma_F=j["sigma_F"], chi2_J=j["chi2"], railed_J=fit_railed(j, bounds, F_bounds))
    return row


def bias_vs_a_rs(cfg, A_B_true, a_rs_grid, rho_star=None, models=("naive", "naive_redist"), bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, n_jobs=-1, **kw):
    """The 1-D counterpart of bias_map_grid along a/Rs: fit_truth (both models, Fisher sigmas) at every a/Rs, points in parallel, one DataFrame row
    per point. A railed fit keeps its fitted values (flagged by fit_railed). rho_star [g/cm^3]: each point gets the period of a circular orbit at
    that a/Rs around a star of that mean density (kepler_period_days). The period enters only sigma_bin, not the bias, so the Fisher sigmas and
    chi2 then belong to a consistent orbit; None keeps cfg.planet.period. Columns: a_rs, period_h, sigma_bin, then those of the A_B-only fit
    (A_B_fit, bias, sigma, chi2, railed) and of the joint fit (the same with _J, plus F_fit, sigma_F), each only if its model is in `models`."""
    return pd.DataFrame(pmap(_bias_vs_a_rs_point, [(cfg, A_B_true, float(a), rho_star, models, bounds, F_bounds, kw) for a in a_rs_grid], n_jobs))


def sweep_geometry(cfg, a_rs_grid):
    """The illumination geometry along an a/Rs grid at cfg's Rp/Rs: the area fractions of PlanetParams.fraction_illuminated (Knuth et al. 2017;
    Carter et al. 2024) and the stellar angular radius."""
    geo = pd.DataFrame([{"a_rs": a, **{k.replace(" ", "_"): v for k, v in swept_planet(cfg, a_rs=a).fraction_illuminated(extra=True).items()}} for a in a_rs_grid])
    geo["night"] = 1. - geo["fully_illuminated"] - geo["penumbra"]
    geo["angular_radius_deg"] = np.degrees(np.arcsin(1. / geo["a_rs"]))
    return geo


def false_F_cell(cfg, A_B_true, free_offset=True, rail_tol=RAIL_TOL, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, **kw):
    """One cell of a false-positive map: (dchi2_F, F_fit) of false_F_fit, NaN on a rail."""
    d = false_F_fit(cfg, A_B_true, free_offset=free_offset, rail_tol=rail_tol, bounds=bounds, F_bounds=F_bounds, **kw)
    return d["dchi2_F"], d["F_fit"]


def _false_F_cell_at(cfg, A_B_true, a_rs, y_param, y, kw):
    return false_F_cell(cfg, A_B_true, a_rs=a_rs, **{y_param: y}, **kw)


def false_F_map_grid(cfg, A_B_true, a_rs_grid, y_grid, y_param, n_jobs=-1, fits=None, **kw):
    """dchi2_F and F maps (like bias_map_grid), shape (len(y_grid), len(a_rs_grid)). fits=(chi2_map, chi2_map_J, F_map_J) builds them without
    refitting, from bias_map_grid(..., return_chi2=True) and bias_map_grid(..., redist=True, return_chi2=True) on the same cfg, A_B_true, grids,
    y_param and **kw: those are the two fits false_F_cell makes per cell. Only the shapes are checked."""
    if fits is not None:
        chi2, chi2_J, F_J = (np.asarray(m, dtype=float) for m in fits)
        shape = (len(y_grid), len(a_rs_grid))
        if not chi2.shape == chi2_J.shape == F_J.shape == shape:
            raise ValueError(f"fits: three maps of shape (len(y_grid), len(a_rs_grid)) = {shape} expected, got {chi2.shape}, {chi2_J.shape}, {F_J.shape}")
        railed = ~np.isfinite(chi2) | ~np.isfinite(chi2_J)   # NaN chi2 = railed fit; false_F_cell is NaN where either fit railed
        return np.where(railed, np.nan, np.maximum(chi2 - chi2_J, 0.)), np.where(railed, np.nan, F_J)
    flat = pmap(_false_F_cell_at, [(cfg, A_B_true, ar, y_param, y, kw) for y in y_grid for ar in a_rs_grid], n_jobs)
    out = np.array(flat).reshape(len(y_grid), len(a_rs_grid), 2)
    return out[..., 0], out[..., 1]


def reference_planet(a_rs=2.5, rp_rs=DEFAULT_MAP_RP_RS, T_eff=4000., incl=90., period=0.3, R_star_Rsun=0.6, name="reference geometry"):
    """A synthetic PlanetParams for planet-agnostic bias maps: the geometry every cell inherits for what the map does not sweep. Units: r2 = 1, so
    r1 = Rp/Rs and R = a/Rs. period and R_star_Rsun are placeholders: the curve is binned per orbit, so the period drops out of the bias, and the
    stellar radius only sets a_AU for the light-travel-time correction. T14 is NaN (only PandExo needs it). The star keeps log g = 4.5 and
    [Fe/H] = 0 at every T_eff (a hypothetical star; see swept_planet)."""
    return PlanetParams(name=name, period=period, incl=incl, ecc=0., w=90., r1=rp_rs, r2=1., R=a_rs, T2_eff=T_eff,
                        a_AU=a_rs * R_star_Rsun * float(const.R_sun / const.au), T14=np.nan)


def moved_planet(pl, period_h, name=None):
    """A real planet moved to another period on a circular orbit around its own host: a/R* and a_AU scale as (P / P_0)^(2/3) (Kepler III at the
    host's density). Rp/R*, the inclination (so b shrinks with a/R*) and the star are kept. T14 is NaN (only PandExo needs it; give the moved
    planet its parent's sigma_1min)."""
    s = (period_h / 24. / pl.period) ** (2. / 3.)
    if pl.R * s <= pl.r1 + pl.r2: raise ValueError(f"{pl.name} at P = {period_h:.2f} h would touch its star (a/R* = {pl.R * s / pl.r2:.3f})")
    return replace(pl, name=name or f"{pl.name} at {period_h:.2f} h", period=period_h / 24., R=pl.R * s, a_AU=pl.a_AU * s, T14=np.nan)


def map_grid(lo, hi, n, include=None, pad=0.1, bounds=None):
    """Linear grid over [lo, hi], widened (by `pad` of the span) so that `include`, typically the target's own a/Rs or Rp/Rs, lies inside it.
    bounds = (min, max): limits the widened grid may not cross, e.g. (TEFF_NODES[0], TEFF_NODES[-1]) for a T_s axis, where a cell off the stellar
    grid's nodes would raise in the worker pool and lose the whole map."""
    if include is not None and np.isfinite(include):
        span = hi - lo
        lo, hi = min(lo, include - pad * span), max(hi, include + pad * span)
    if bounds is not None: lo, hi = max(lo, bounds[0]), min(hi, bounds[1])
    return np.linspace(lo, hi, n)


############################################################################################################################################
#####   The Roche limit on the maps, and printed summaries
############################################################################################################################################

def print_limit_curve(sweep, A_B_true, roche_a_rs, sweep_ubol0=None, u_bol=None):
    """Printed summary of a limit curve (bias_vs_a_rs): the railed points, the fits below A_B = 0, the largest non-railed biases, the u_bol
    sensitivity (with sweep_ubol0) and the sweep at each Roche limit of roche_a_rs."""
    ok, okJ = ~sweep["railed"], ~sweep["railed_J"]
    print(f"A_B-only fit railed at {int((~ok).sum())} of {len(sweep)} points" + (f" (a/Rs <= {sweep.loc[~ok, 'a_rs'].max():.3f})" if (~ok).any() else "")
          + f";  joint fit railed at {int((~okJ).sum())}" + (f" (a/Rs <= {sweep.loc[~okJ, 'a_rs'].max():.3f})" if (~okJ).any() else ""))
    neg, negJ = sweep["A_B_fit"] < 0., sweep["A_B_fit_J"] < 0.   # below the physical floor (allowed by A_B_FIT_BOUNDS)
    print(f"fitted A_B < 0: A_B-only fit at {int(neg.sum())} points" + (f" (a/Rs <= {sweep.loc[neg, 'a_rs'].max():.3f})" if neg.any() else "")
          + f";  joint fit at {int(negJ.sum())}" + (f" (a/Rs <= {sweep.loc[negJ, 'a_rs'].max():.3f})" if negJ.any() else ""))
    if ok.any():
        i = np.abs(sweep.loc[ok, "bias"]).idxmax()
        print(f"largest non-railed |Delta A_B|, A_B-only: {sweep.loc[i, 'bias']:+.3f} ({100 * sweep.loc[i, 'bias'] / A_B_true:+.1f} % of A_B) at a/Rs = {sweep.loc[i, 'a_rs']:.3f}")
    if okJ.any():
        i = np.abs(sweep.loc[okJ, "bias_J"]).idxmax()
        print(f"largest non-railed |Delta A_B|, joint:     {sweep.loc[i, 'bias_J']:+.3f} ({100 * sweep.loc[i, 'bias_J'] / A_B_true:+.1f} % of A_B) at a/Rs = {sweep.loc[i, 'a_rs']:.3f};  largest spurious F = {sweep.loc[okJ, 'F_fit'].max():.4f}")
    if sweep_ubol0 is not None:
        both = ok & ~sweep_ubol0["railed"]
        print(f"u_bol sensitivity: max |bias(u_bol = {u_bol:g}) - bias(u_bol = 0)| = {np.max(np.abs(sweep.loc[both, 'bias'] - sweep_ubol0.loc[both, 'bias'])):.4f} over the {int(both.sum())} points where neither fit railed" if both.any() else "u_bol sensitivity: no point where neither fit railed")
    for lab, a_R in roche_a_rs.items():   # interpolated at each Roche limit (NaN outside the sweep)
        at = lambda col: np.interp(a_R, sweep["a_rs"], sweep[col], left=np.nan, right=np.nan)
        print(f"at the Roche limit, {lab} (a/Rs = {a_R:.2f}): Delta A_B = {at('bias'):+.3f} (A_B-only), {at('bias_J'):+.3f} (joint), F = {at('F_fit'):.4f}, misfit {np.sqrt(at('chi2')):.2f} sigma")


def roche_periods(cases=None, R_p=ROCHE_R_P):
    """Roche periods [h] of the compositions in `cases` ({label: (iron mass fraction, rho_0 / rho_mean)}, default roche.ROCHE_CASES) for a rock/iron
    planet of radius R_p [R_E]: mean density from roche.rocky_mean_density (Fortney et al. 2007), P_R from roche.roche_period_h (Rappaport et al.
    2013 eq. 5). In period the limits hold for any host."""
    cases = ROCHE_CASES if cases is None else cases
    return {f"{lab}, {R_p:g} $R_\\oplus$": float(roche_period_h(rocky_mean_density(R_p, fe), q)) for lab, (fe, q) in cases.items()}


def _roche_cases(roche):
    """The {label: P_R [h]} of a roche= argument: True -> roche_periods(), a dict -> itself, False / None -> None."""
    return roche_periods() if roche is True else (roche or None)


def roche_frontier_a_rs(T_eff, P_R_h):
    """a_R/Rs along a T_eff axis for main-sequence hosts, for a Roche period P_R_h [h]."""
    return kepler_a_rs(P_R_h / 24., rho_star(T_eff))


def roche_allowed_masks(xg, yg, periods):
    """Where a planet lies at or beyond the Roche limit, for main-sequence hosts: {label: boolean mask over the (a/Rs, T_eff) grid (xg, yg), shape
    (len(yg), len(xg))}, one per Roche period of `periods` ({label: P_R [h]})."""
    X, Y = np.meshgrid(xg, yg)
    return {lab: X >= roche_frontier_a_rs(Y, P_R) for lab, P_R in periods.items()}


def print_allowed_extremes(xg, yg, bias_map, A_B_true, allowed, periods):
    """The largest |bias| of a map within each allowed region (roche_allowed_masks), one line per Roche limit: the extreme reachable around
    main-sequence hosts, not the box's corner."""
    fin = np.isfinite(bias_map)
    print(f"non-railed cells: {fin.sum()} of {fin.size};  largest |Delta A_B| on the whole box: {np.nanmax(np.abs(bias_map)):.3f}")
    for comp, m in allowed.items():
        txt = comp.replace("$R_\\oplus$", "R_E")   # plain-text label for print
        label = f"Roche limit, {txt} (P_R = {periods[comp]:.2f} h)"
        sel = m & fin
        if not sel.any():
            print(f"  beyond the {label}: no allowed non-railed cell"); continue
        i = np.unravel_index(np.nanargmax(np.where(sel, np.abs(bias_map), np.nan)), bias_map.shape)
        print(f"  beyond the {label}: {sel.sum()} cells allowed, largest |Delta A_B| = {abs(bias_map[i]):.3f} ({100 * abs(bias_map[i]) / A_B_true:.1f} % of A_B) at a/Rs = {xg[i[1]]:.2f}, T_eff = {yg[i[0]]:.0f} K")


def map_vs_planet_bias(tab, planets, a_rs_grid, y_grid, bias_map, y_param="T_eff"):
    """The A_B bias read off a map at each planet's position (linear interpolation) next to its planet_bias_table value at the full geometry. The
    maps hold the inclination (and the parameter not on the y axis) at reference values, so planets with a large impact parameter b differ.
    map_excess = |map bias| - |own bias| (> 0: the map overstates). y_param: "T_eff" or "rp_rs", the map's y axis. planets: {name: PlanetParams}."""
    interp = RegularGridInterpolator((y_grid, a_rs_grid), bias_map, bounds_error=False)   # NaN outside the map box
    pls = [planets[nm] for nm in tab["name"]]
    y = [pl.T2_eff if y_param == "T_eff" else pl.r1 / pl.r2 for pl in pls]
    out = pd.DataFrame(dict(name=tab["name"], b=[pl.R / pl.r2 * np.cos(np.radians(pl.incl)) for pl in pls],
                            bias_own=tab["bias_A_B"], bias_map=interp(np.column_stack([y, tab["a_rs"]]))))
    out["map_excess"] = np.abs(out["bias_map"]) - np.abs(out["bias_own"])
    return out.sort_values("map_excess", ascending=False).reset_index(drop=True)
