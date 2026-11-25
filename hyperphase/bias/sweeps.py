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


# Refit the models at every TRUE Bond albedo in A_B_grid (single target geometry, free baseline), in parallel.
# Returns {model: {name: array}} with the keys of fit_truth (fisher=True): "naive" always, "naive_redist" with redist=True.
def bias_vs_true_AB(cfg, A_B_grid, redist=True, bounds=A_B_FIT_BOUNDS, n_jobs=-1, **kw):
    models = ("naive", "naive_redist") if redist else ("naive",)
    rows = pmap(_fit_truth_at, [(cfg, a, models, bounds, kw) for a in A_B_grid], n_jobs)
    return {m: {k: np.array([r[m][k] for r in rows]) for k in rows[0][m]} for m in models}


############################################################################################################################################
#####   One-parameter sweeps of the bias (01_albedo_bias.ipynb, 02_redistribution_bias.ipynb: inclination, u_bol)
############################################################################################################################################

# A sweep changes ONE setting of the configuration and repeats the whole bias pipeline (fit_truth: truth, fits, Fisher sigmas) at every value. The
# setting is a transform (cfg, value) -> cfg, applied to every detector of a set, so anything a FittingConfig or its planet holds can be swept without
# a function of its own. cfg_with_incl: the orbital inclination [deg] (truth and fits; the noise level stays the planet's own). cfg_with_u_bol: the
# bolometric limb darkening of the irradiance, which only the hyperillumination truth uses (the point-source models have no u_bol), so it is a sweep
# of the truth alone.
def cfg_with_incl(cfg, incl):
    return replace(cfg, planet=replace(cfg.planet, incl=float(incl)))


def cfg_with_u_bol(cfg, u_bol):
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


# fit_truth (Fisher sigmas included) at every value of a one-parameter sweep, points in parallel, as a DataFrame. cfg: FittingConfig or detector set;
# transform: (cfg, value) -> cfg (cfg_with_incl, cfg_with_u_bol, or any function of that form). Columns: value; A_B_fit, bias, sigma, bias_sigma,
# chi2, lnB_mean = chi2 / 2, railed (A_B-only fit); A_B_fit_J, bias_J, sigma_J, bias_sigma_J, F_fit, chi2_J, railed_J and z_F = sqrt(chi2 - chi2_J)
# (fit with redistribution; z_F needs both models), as in bias_vs_a_rs.
def bias_sweep(cfg, A_B_true, values, transform, models=("naive", "naive_redist"), bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, n_jobs=-1, **kw):
    return pd.DataFrame(pmap(_bias_sweep_point, [(cfg, A_B_true, transform, v, models, bounds, F_bounds, kw) for v in values], n_jobs))


# A hyper-illuminated planet that DOES redistribute heat, fitted with the point-source model with redistribution: the sweep over the true F (paper
# App. G.2, "True redistribution"). Truth: "hyperillum_redist" at (A_B_true, F_true), i.e. Kreidberg & Loeb (2016) eq. 1 with the extended-source
# irradiance; fit: fit_redist ("naive_redist"), with its Fisher sigmas at the fit. The A_B-only point-source fit is left out: against a planet with a
# warm nightside it fails for a reason that has nothing to do with the irradiation model. Columns: value (= F_true), A_B_fit_J, bias_J, sigma_J,
# bias_sigma_J, F_fit, bias_F (= F_fit - F_true), sigma_F, bias_F_sigma, chi2_J and lnB_mean_J = chi2_J / 2. The last is the expected ln B of the
# extended-source model with redistribution against the point-source model with redistribution: both have the parameters (A_B, F), so their Occam
# factors cancel as for the two bare-rock models, as long as F_true lies well above the prior edge F = 0 (at F_true = 0 the edge cuts the posterior of
# the extended-source model only, so there chi2_J / 2 is not the expected ln B). At F_true = 0 the row is the fit with redistribution of every other
# table (bias_F = F_fit, the spurious redistribution).
def _true_F_point(cfg, A_B_true, F_true, bounds, F_bounds, kw):
    truth = truth_curves(cfg, A_B_true, F_true=F_true, **kw)
    A_fit, F_fit, chi2, _ = fit_redist(cfg, truth, free_offset=True, bounds=bounds, F_bounds=F_bounds, **kw)
    sigma_A, sigma_F = fisher_sigma(cfg, "naive_redist", A_fit, F=F_fit, bounds=bounds, F_bounds=F_bounds, **kw)
    return dict(value=float(F_true), A_B_fit_J=A_fit, bias_J=A_fit - A_B_true, sigma_J=sigma_A, bias_sigma_J=abs(A_fit - A_B_true) / sigma_A, F_fit=F_fit, bias_F=F_fit - F_true,
                sigma_F=sigma_F, bias_F_sigma=abs(F_fit - F_true) / sigma_F, chi2_J=chi2, lnB_mean_J=0.5 * chi2)


def true_F_sweep(cfg, A_B_true, F_values, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, n_jobs=-1, **kw):
    return pd.DataFrame(pmap(_true_F_point, [(cfg, A_B_true, F, bounds, F_bounds, kw) for F in F_values], n_jobs))
