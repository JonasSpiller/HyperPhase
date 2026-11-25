"""Expected evidence: where the Bayes factor between the extended- and the point-source model comes from.

For two models with the same parameters and priors (both A_B + baselines) the Occam factors nearly cancel, so the Bayes factor follows the
likelihood ratio. Linearised around the truth, with the data d = truth + sigma z:
    ln B(ext vs ps) = chi2_min / 2 + zeta sqrt(chi2_min),   zeta = r_perp_hat . z ~ N(0, 1),
where r_perp = (truth - m_ps(A_fit) - baseline) / sigma_bin is the whitened residual of the noiseless A_B-only fit and chi2_min = |r_perp|^2 the
chi2 that fit_naive returns. So <ln B> = chi2_min / 2 with scatter sqrt(chi2_min) over noise realisations, and r_perp^2 says at which orbital
phases it arises. Only for equal parameter counts: against the fit with redistribution the F prior adds an Occam term of ~3 nats and F's posterior
is one-sided, so chi2 / 2 is then only the likelihood part."""
import contextlib
import io

import numpy as np
import pandas as pd

from hyperphase.bias.detectors import is_detector_set, template_cfg
from hyperphase.bias.fits import _chi2_problem, fit_AB_only
from hyperphase.bias.tables import instrument_retrievals, planet_list
from hyperphase.defaults import A_B_FIT_BOUNDS, F_BOUNDS
from hyperphase.phasecurve import occultation_durations


############################################################################################################################################
#####   Expected evidence: where the Bayes factor comes from
############################################################################################################################################

def evidence_residual(cfg, truth, insolation_model="naive", A_B_fit=None, F=None, free_offset=True, bounds=A_B_FIT_BOUNDS, **kw):
    """The whitened residual r_perp of the noiseless A_B-only fit of `insolation_model` to `truth`, bin by bin (detector blocks concatenated, each
    with its own baseline and sigma_bin); sum(r_perp^2) is that fit's chi2. A_B_fit=None fits it (fit_AB_only). F: with the A_B_fit of
    naive_redist (both from fit_redist), the residual of the joint fit. kw: the model_pcurve keywords of the fit. Returns dict(A_B_fit, chi2,
    resid [ppm], sigma [ppm], r_perp, phase [bin centres, 0 = mid-transit], blocks [one slice per detector])."""
    if A_B_fit is None: A_B_fit = fit_AB_only(cfg, truth, insolation_model, free_offset=free_offset, bounds=bounds, **kw)[0]
    resid, _, sigma = _chi2_problem(cfg, truth, insolation_model, free_offset=free_offset, **kw)
    r = resid(A_B_fit, F)
    cfgs = list(cfg) if is_detector_set(cfg) else [cfg]
    edges = np.cumsum([0] + [c.n_bins for c in cfgs])
    return dict(A_B_fit=float(A_B_fit), chi2=float(np.sum((r / sigma) ** 2)), resid=r, sigma=sigma, r_perp=r / sigma,
                # model_pcurve starts at mid-transit; binned() averages consecutive samples
                phase=np.concatenate([(np.arange(c.n_bins) + 0.5) / c.n_bins for c in cfgs]),
                blocks=[slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])])


def eclipse_half_phase(planet):
    """Half-duration of the transit (and of the occultation: circular orbit) in orbital phase, first to fourth contact (occultation_durations)."""
    from types import SimpleNamespace
    T14, _ = occultation_durations(SimpleNamespace(inc=planet.incl, a=planet.R / planet.r2, rp=planet.r1 / planet.r2, per=planet.period))
    return 0.5 * T14 / planet.period


def phase_windows(planet, phase, n_bins, quad_half=0.1):
    """The phase windows of an evidence budget as boolean masks over bin centres `phase` (0 = mid-transit), disjoint in this order: occultation and
    transit (every bin whose average the transit or occultation touches: centre within the half-duration plus half a bin), the quadratures (centre
    within quad_half of 0.25 or 0.75) and the rest."""
    dist = lambda p0: np.abs((np.asarray(phase, float) - p0 + 0.5) % 1. - 0.5)   # phase distance on the circle
    h = eclipse_half_phase(planet) + 0.5 / n_bins
    occ = dist(0.5) < h
    tra = (dist(0.) < h) & ~occ
    quad = ((dist(0.25) < quad_half) | (dist(0.75) < quad_half)) & ~occ & ~tra
    return {"occultation": occ, "transit": tra, "quadratures": quad, "rest": ~(occ | tra | quad)}


def evidence_budget(ev, cfg, labels=None, quad_half=0.1):
    """chi2 of an evidence_residual per phase window (phase_windows) and its share of the total chi2_min, one row per detector block plus "all".
    labels: the detectors' names (default: block numbers). Since 2 <ln B> = chi2_min, the shares are also those of the expected ln B."""
    cfgs = list(cfg) if is_detector_set(cfg) else [cfg]
    labels = list(labels) if labels is not None else [f"detector {k + 1}" for k in range(len(cfgs))]
    rows, total = [], {}
    for c, blk, lab in zip(cfgs, ev["blocks"], labels):
        r2 = ev["r_perp"][blk] ** 2
        win = phase_windows(c.planet, ev["phase"][blk], c.n_bins, quad_half)
        part = {name: float(r2[m].sum()) for name, m in win.items()}
        for name, v in part.items(): total[name] = total.get(name, 0.) + v
        rows.append(dict(detector=lab, chi2=float(r2.sum()), **part))
    if len(rows) > 1: rows.append(dict(detector="all", chi2=ev["chi2"], **total))
    tab = pd.DataFrame(rows)
    for name in total: tab[name + " [%]"] = 100. * tab[name] / ev["chi2"]   # share of the TOTAL chi2_min, so a detector's row sums to its own share
    tab["share [%]"] = 100. * tab["chi2"] / ev["chi2"]
    return tab


def evidence_probability(chi2, threshold, time_factor=1.):
    """P(ln B > threshold) for extended vs point source in the linear prediction ln B ~ N(chi2_min / 2, chi2_min). chi2 scales with the observing
    time: time_factor = n_orbits / (the n_orbits of chi2) gives another number of orbits without a refit."""
    from scipy.stats import norm
    c = time_factor * np.asarray(chi2, dtype=float)
    return norm.sf((threshold - 0.5 * c) / np.sqrt(c))


def evidence_table(cfg, planets, instruments, A_B_true, thresholds=(2.5, 5.), bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, n_jobs=-1, **kw):
    """The expected-evidence table: every planet on every entry of `instruments` (instrument_retrievals per planet: own geometry and noise) with both
    point-source fits, one row per (planet, instrument, fit). The A_B-only rows add lnB_mean = chi2 / 2, lnB_sd = sqrt(chi2) and P_lnB_gt_<x> for
    each threshold, at cfg.n_orbits. The naive_redist rows get none (their chi2 / 2 lacks the Occam term of the F prior, see the module docstring)
    but z_F = sqrt(chi2_naive - chi2), the significance of the spurious F. Rows without a noise level: see VERDICT_NO_NOISE."""
    parts = []
    for pl in planet_list(planets):
        tab = instrument_retrievals(cfg, pl, instruments, A_B_true, bounds=bounds, F_bounds=F_bounds, n_jobs=n_jobs, **kw)
        with contextlib.redirect_stdout(io.StringIO()):   # the same notes as the first call
            tab_J = instrument_retrievals(cfg, pl, instruments, A_B_true, redist=True, sigma_1min=tab.attrs["sigma_1min"], bounds=bounds, F_bounds=F_bounds, n_jobs=n_jobs, **kw)
        # same rows in the same order: both tables come from the same instruments and noise levels
        tab_J["z_F"] = np.sqrt(np.clip(tab["chi2"].to_numpy() - tab_J["chi2"].to_numpy(), 0., None))
        for model, t in (("naive", tab), ("naive_redist", tab_J)):
            t = t.copy(); t.insert(0, "fit", model); t.insert(0, "planet", pl.name)
            parts.append(t)
    out = pd.concat(parts, ignore_index=True)
    out["half_chi2"] = 0.5 * out["chi2"]
    ps = out["fit"] == "naive"
    out["lnB_mean"], out["lnB_sd"] = np.where(ps, 0.5 * out["chi2"], np.nan), np.where(ps, np.sqrt(out["chi2"]), np.nan)
    for x in thresholds: out[f"P_lnB_gt_{x:g}"] = np.where(ps, evidence_probability(out["chi2"], x), np.nan)
    out.attrs.update(A_B_true=A_B_true, n_orbits=template_cfg(cfg).n_orbits, thresholds=tuple(thresholds))
    return out[["planet", "instrument", "fit", "sigma_1min", "A_B_fit", "bias_A_B", "sigma_A_B", "bias_sigma", "F_fit", "z_F", "chi2", "half_chi2", "lnB_mean", "lnB_sd"] + [f"P_lnB_gt_{x:g}" for x in thresholds] + ["verdict"]]
