"""One-parameter sweeps of the bias: the true Bond albedo, the inclination, the limb darkening, and a truth that redistributes heat."""
from dataclasses import replace

import numpy as np
import pandas as pd

from hyperphase.bias.detectors import is_detector_set
from hyperphase.bias.fisher import fisher_sigma, fit_truth
from hyperphase.bias.fits import fit_railed, fit_redist, truth_curves
from hyperphase.defaults import A_B_FIT_BOUNDS, F_BOUNDS
from hyperphase.parallel import pmap


############################################################################################################################################
#####   Bias against the true Bond albedo
############################################################################################################################################
def _fit_truth_at(cfg, A_B_true, models, bounds, kw):
    return fit_truth(cfg, A_B_true, models=models, free_offset=True, bounds=bounds, fisher=True, **kw)


def bias_vs_true_AB(cfg, A_B_grid, redist=True, bounds=A_B_FIT_BOUNDS, n_jobs=-1, **kw):
    """fit_truth (Fisher sigmas included) at every true A_B of A_B_grid, single geometry, in parallel: {model: {key: array}} with "naive", and
    "naive_redist" if redist."""
    models = ("naive", "naive_redist") if redist else ("naive",)
    rows = pmap(_fit_truth_at, [(cfg, a, models, bounds, kw) for a in A_B_grid], n_jobs)
    return {m: {k: np.array([r[m][k] for r in rows]) for k in rows[0][m]} for m in models}


############################################################################################################################################
#####   One-parameter sweeps of the bias: inclination, u_bol
############################################################################################################################################

def cfg_with_incl(cfg, incl):
    """Sweep transform: the orbital inclination [deg], in truth and fits (the noise level stays the planet's own)."""
    return replace(cfg, planet=replace(cfg.planet, incl=float(incl)))


def cfg_with_u_bol(cfg, u_bol):
    """Sweep transform: the bolometric limb darkening of the irradiance. Only the extended-source truth uses it (the point-source models have no
    u_bol), so this sweeps the truth alone."""
    return replace(cfg, u_bol=float(u_bol))


def _bias_sweep_point(cfg, A_B_true, transform, value, models, bounds, F_bounds, kw):
    c = [transform(d, value) for d in cfg] if is_detector_set(cfg) else transform(cfg, value)
    fit = fit_truth(c, A_B_true, models=models, free_offset=True, bounds=bounds, F_bounds=F_bounds, fisher=True, **kw)
    row = dict(value=float(value))
    if "naive" in fit:
        n = fit["naive"]
        row.update(A_B_fit=n["A_B_fit"], bias=n["bias_A_B"], sigma=n["sigma_A_B"], bias_sigma=abs(n["bias_A_B"]) / n["sigma_A_B"], chi2=n["chi2"], lnB_mean=0.5 * n["chi2"], railed=fit_railed(n, bounds, F_bounds))
    if "naive_redist" in fit:
        j = fit["naive_redist"]
        row.update(A_B_fit_J=j["A_B_fit"], bias_J=j["bias_A_B"], sigma_J=j["sigma_A_B"], bias_sigma_J=abs(j["bias_A_B"]) / j["sigma_A_B"], F_fit=j["F_fit"], chi2_J=j["chi2"], railed_J=fit_railed(j, bounds, F_bounds))
        if "naive" in fit: row["z_F"] = float(np.sqrt(max(fit["naive"]["chi2"] - j["chi2"], 0.)))
    return row


def bias_sweep(cfg, A_B_true, values, transform, models=("naive", "naive_redist"), bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, n_jobs=-1, **kw):
    """One-parameter sweep: transform(cfg, value) -> cfg changes one setting (applied to every detector of a set; cfg_with_incl, cfg_with_u_bol or
    any such function) and the whole fit_truth pipeline, Fisher sigmas included, is repeated at every value, points in parallel. Columns: value;
    A_B_fit, bias, sigma, bias_sigma, chi2, lnB_mean = chi2 / 2, railed (A_B-only fit); A_B_fit_J, bias_J, sigma_J, bias_sigma_J, F_fit, chi2_J,
    railed_J (fit with redistribution); z_F = sqrt(chi2 - chi2_J) when both run."""
    return pd.DataFrame(pmap(_bias_sweep_point, [(cfg, A_B_true, transform, v, models, bounds, F_bounds, kw) for v in values], n_jobs))


def _true_F_point(cfg, A_B_true, F_true, bounds, F_bounds, kw):
    truth = truth_curves(cfg, A_B_true, F_true=F_true, **kw)
    A_fit, F_fit, chi2, _ = fit_redist(cfg, truth, free_offset=True, bounds=bounds, F_bounds=F_bounds, **kw)
    sigma_A, sigma_F = fisher_sigma(cfg, "naive_redist", A_fit, F=F_fit, bounds=bounds, F_bounds=F_bounds, **kw)
    return dict(value=float(F_true), A_B_fit_J=A_fit, bias_J=A_fit - A_B_true, sigma_J=sigma_A, bias_sigma_J=abs(A_fit - A_B_true) / sigma_A, F_fit=F_fit, bias_F=F_fit - F_true,
                sigma_F=sigma_F, bias_F_sigma=abs(F_fit - F_true) / sigma_F, chi2_J=chi2, lnB_mean_J=0.5 * chi2)


def true_F_sweep(cfg, A_B_true, F_values, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, n_jobs=-1, **kw):
    """Bias of the point-source fit with redistribution (fit_redist, Fisher sigmas at the fit) against an extended-source truth that redistributes
    heat: "hyperillum_redist" at (A_B_true, F_true), Kreidberg & Loeb (2016) eq. 1 with the extended-source irradiance, for each F_true of
    F_values. The A_B-only fit is left out: against a warm nightside it fails for a reason unrelated to the irradiation model. Columns: value
    (F_true), A_B_fit_J, bias_J, sigma_J, bias_sigma_J, F_fit, bias_F = F_fit - F_true, sigma_F, bias_F_sigma, chi2_J, and
    lnB_mean_J = chi2_J / 2, the expected ln B of the extended- against the point-source model, both with redistribution: they have the same
    parameters, so their Occam factors cancel, but only for F_true well above the prior edge F = 0."""
    return pd.DataFrame(pmap(_true_F_point, [(cfg, A_B_true, F, bounds, F_bounds, kw) for F in F_values], n_jobs))
