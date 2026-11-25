"""Expected evidence: where the Bayes factor between the extended- and the point-source model comes from."""
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
#####   Expected evidence: where the Bayes factor comes from (04_evidence_prediction.ipynb)
############################################################################################################################################

# For two models with the same parameters and priors (extended source vs point source, both A_B + baselines) the Occam factors nearly cancel, so the
# Bayes factor follows the likelihood ratio. Linearised around the truth, with the noise d = truth + sigma z: ln B(ext vs ps) = chi2_min / 2 + zeta
# sqrt(chi2_min), zeta = r_perp_hat . z ~ N(0, 1), where r_perp = (truth - m_ps(A_fit) - baseline) / sigma_bin is the whitened residual of the
# NOISELESS A_B-only fit and chi2_min = |r_perp|^2 the chi2 that fit_naive returns. So <ln B> = chi2_min / 2 with scatter sqrt(chi2_min) over noise
# realisations, and r_perp^2 says at which orbital phases the expected evidence arises. Only for equal parameter counts: against the fit with
# redistribution the F prior adds an Occam term of ~3 nats and F's posterior is one-sided, so chi2 / 2 is then only the likelihood part.

# The whitened residual r_perp of the noiseless A_B-only fit of `insolation_model` to `truth`, bin by bin (detector blocks concatenated, each with its
# own baseline removed and its own sigma_bin). A_B_fit=None fits it (fit_AB_only); sum(r_perp^2) is that fit's chi2. Returns dict(A_B_fit, chi2, resid
# [ppm], sigma [ppm], r_perp, phase [bin centres, 0 = transit], blocks [one slice per detector]). kw: the model_pcurve keywords of the fit, so the
# dynesty comparison can evaluate it at the truth's own resolution. F: the fitted redistribution of "naive_redist" (with its A_B_fit, both from
# fit_redist); the residual is then that of the joint fit, and r_perp(A_B-only) - r_perp(with F) the part F absorbs.
def evidence_residual(cfg, truth, insolation_model="naive", A_B_fit=None, F=None, free_offset=True, bounds=A_B_FIT_BOUNDS, **kw):
    if A_B_fit is None: A_B_fit = fit_AB_only(cfg, truth, insolation_model, free_offset=free_offset, bounds=bounds, **kw)[0]
    resid, _, sigma = _chi2_problem(cfg, truth, insolation_model, free_offset=free_offset, **kw)
    r = resid(A_B_fit, F)
    cfgs = list(cfg) if is_detector_set(cfg) else [cfg]
    edges = np.cumsum([0] + [c.n_bins for c in cfgs])
    return dict(A_B_fit=float(A_B_fit), chi2=float(np.sum((r / sigma) ** 2)), resid=r, sigma=sigma, r_perp=r / sigma,
                phase=np.concatenate([(np.arange(c.n_bins) + 0.5) / c.n_bins for c in cfgs]),   # model_pcurve runs from t = 0 (mid-transit) over one period and binned() averages consecutive samples
                blocks=[slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])])


# Half-duration of the transit / occultation in orbital phase (first to fourth contact; a circular orbit, so the two last equally long):
# occultation_durations at the planet's geometry.
def eclipse_half_phase(planet):
    from types import SimpleNamespace
    T14, _ = occultation_durations(SimpleNamespace(inc=planet.incl, a=planet.R / planet.r2, rp=planet.r1 / planet.r2, per=planet.period))
    return 0.5 * T14 / planet.period


# The phase windows of an evidence budget as boolean masks over bin centres `phase` (0 = mid-transit): occultation and transit (every bin that
# overlaps the eclipse, i.e. whose bin average the eclipse touches: centre within the half-duration plus half a bin), the quadratures (centre within
# quad_half of 0.25 or 0.75) and the rest. Disjoint, in that priority.
def phase_windows(planet, phase, n_bins, quad_half=0.1):
    dist = lambda p0: np.abs((np.asarray(phase, float) - p0 + 0.5) % 1. - 0.5)   # phase distance on the circle
    h = eclipse_half_phase(planet) + 0.5 / n_bins
    occ = dist(0.5) < h
    tra = (dist(0.) < h) & ~occ
    quad = ((dist(0.25) < quad_half) | (dist(0.75) < quad_half)) & ~occ & ~tra
    return {"occultation": occ, "transit": tra, "quadratures": quad, "rest": ~(occ | tra | quad)}


# Where in phase the expected evidence of an evidence_residual arises: chi2 and its share of the total chi2_min per phase window (phase_windows), one
# row per detector block plus "all". labels: the detectors' names (default: block numbers). The shares of the "all" row sum to 1; 2 <ln B> = chi2_min,
# so they are also the shares of the expected ln B.
def evidence_budget(ev, cfg, labels=None, quad_half=0.1):
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


# P(ln B > threshold) over noise realisations for extended source vs point source, in the linear prediction ln B ~ N(chi2_min / 2, chi2_min). chi2
# scales with the observing time, so time_factor = n_orbits / (the n_orbits chi2 was computed at) gives another number of orbits without a refit (e.g.
# 2 / 5 for one visit instead of five orbits).
def evidence_probability(chi2, threshold, time_factor=1.):
    from scipy.stats import norm
    c = time_factor * np.asarray(chi2, dtype=float)
    return norm.sf((threshold - 0.5 * c) / np.sqrt(c))


# The expected-evidence table: every planet of `planets` on every entry of `instruments` (instrument_retrievals per planet: own geometry, own noise,
# joint detector fits as tuples) with both point-source fits, one tidy row per (planet, instrument, fit). Columns: planet, instrument, fit ("naive" /
# "naive_redist"), A_B_fit, bias_A_B, sigma_A_B, bias_sigma, F_fit, chi2, half_chi2 = chi2 / 2, and for the A_B-only fit lnB_mean = chi2 / 2, lnB_sd =
# sqrt(chi2) and P_lnB_gt_<x> = P(ln B > x) at cfg.n_orbits for each x of `thresholds`. For "naive_redist" half_chi2 is only the LIKELIHOOD part of ln
# B(ext vs ps+F) (no Occam term of the F prior), so it gets no lnB columns; its z_F = sqrt(chi2_naive - chi2) is the significance of the spurious F.
# Rows without a noise level (MIRI imaging for most planets) keep the fitted values and NaN in everything that needs the noise.
def evidence_table(cfg, planets, instruments, A_B_true, thresholds=(2.5, 5.), bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, n_jobs=-1, **kw):
    parts = []
    for pl in planet_list(planets):
        tab = instrument_retrievals(cfg, pl, instruments, A_B_true, bounds=bounds, F_bounds=F_bounds, n_jobs=n_jobs, **kw)
        with contextlib.redirect_stdout(io.StringIO()):   # the same notes as the first call
            tab_J = instrument_retrievals(cfg, pl, instruments, A_B_true, redist=True, sigma_1min=tab.attrs["sigma_1min"], bounds=bounds, F_bounds=F_bounds, n_jobs=n_jobs, **kw)
        tab_J["z_F"] = np.sqrt(np.clip(tab["chi2"].to_numpy() - tab_J["chi2"].to_numpy(), 0., None))   # same rows in the same order: both tables come from the same instruments and noise levels
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
