"""Fisher uncertainties, measurability verdicts and critical noise levels of the noiseless fits."""

import numpy as np

from hyperphase.bias.detectors import is_detector_set
from hyperphase.bias.fits import RAIL_TOL, _chi2_problem, fit_naive, fit_railed, fit_redist, truth_curves
from hyperphase.defaults import A_B_FIT_BOUNDS, F_BOUNDS
from hyperphase.irradiance import surface_temperature
from hyperphase.model import pcurve_params, swept_planet
from hyperphase.phasecurve import occultation_depth, reflected_lightcurve


############################################################################################################################################
#####   Fisher uncertainties and verdicts
############################################################################################################################################

# The one pipeline behind every bias number: build the noiseless hyperillumination truth at A_B_true and fit it with each of `models`
# ("naive": A_B; "naive_redist": A_B and F). Returns {model: {"A_B_fit", "bias_A_B", "chi2"[, "F_fit"][, "sigma_A_B"[, "sigma_F"]]}}.
# The truth has F = 0, so F_fit is itself the bias in F. The two fits are independent (fit_redist needs no start).
# fisher=True adds the Fisher sigmas, evaluated at the FITS, where the misspecified model's posterior sits.
# cfg may be a detector set (see is_detector_set): one truth per detector, shared (A_B, F), a baseline per detector.
def fit_truth(cfg, A_B_true, models=("naive",), free_offset=True, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, fisher=False, **kw):
    truth = truth_curves(cfg, A_B_true, **kw)   # noiseless data
    out = {}
    if "naive" in models:
        A_fit, chi2, _ = fit_naive(cfg, truth, free_offset=free_offset, bounds=bounds, **kw)
        out["naive"] = dict(A_B_fit=A_fit, bias_A_B=A_fit - A_B_true, chi2=chi2)
        if fisher:
            out["naive"]["sigma_A_B"], = fisher_sigma(cfg, "naive", A_fit, bounds=bounds, **kw)   # 1 free parameter + baseline
    if "naive_redist" in models:
        A_fit, F_fit, chi2, _ = fit_redist(cfg, truth, free_offset=free_offset, bounds=bounds, F_bounds=F_bounds, **kw)
        out["naive_redist"] = dict(A_B_fit=A_fit, bias_A_B=A_fit - A_B_true, F_fit=F_fit, chi2=chi2)
        if fisher:   # 2 free parameters + baseline -> A_B's MARGINAL width grows with the F degeneracy
            out["naive_redist"]["sigma_A_B"], out["naive_redist"]["sigma_F"] = fisher_sigma(cfg, "naive_redist", A_fit, F=F_fit, bounds=bounds, F_bounds=F_bounds, **kw)
    return out


# The systematic in units of the statistical error bar, and the per-minute precision sigma_1min_crit at which |bias| = sigma. sigma_1min_crit =
# sigma_1min * |bias| / sigma is exact rather than an extrapolation: the bias does not depend on the noise, and the Fisher sigma is strictly
# proportional to sigma_1min.
def measurability(bias, sigma, sigma_1min):
    bias_sigma = np.abs(bias) / sigma
    return bias_sigma, sigma_1min * bias_sigma


# Std of the flat prior U(lo, hi) over a parameter's range: (hi - lo) / sqrt(12). A Fisher sigma at or above it means the data add nothing to the
# prior, so the Gaussian posterior the Fisher matrix assumes does not exist. The verdicts always take the PHYSICAL range (A_B_BOUNDS, F_BOUNDS), the
# priors of the dynesty fits, not the wider least-squares range of A_B (A_B_FIT_BOUNDS): "unconstrained" means the data would not narrow A_B within
# what an observer allows.
def flat_prior_sigma(bounds):
    return (bounds[1] - bounds[0]) / np.sqrt(12.)


# Three-way verdict of the bias test. bias_sigma < 1 alone reads as "naive OK", but with sigma >= the flat-prior std it only says the parameter is
# unmeasured — the bias is then divided by a meaningless number. That case is "unconstrained", not "naive OK".
VERDICT_NEEDS_HI, VERDICT_NAIVE_OK, VERDICT_UNCONSTRAINED = "needs hyperillum", "naive OK", "unconstrained"

# A row fitted WITHOUT a noise level (no PandExo mode and no CUSTOM_SIGMA_1MIN entry, e.g. MIRI imaging): the fitted values and the bias are exact,
# since they do not depend on the noise (the fit runs at cfg's own sigma_1min, which only has to be finite: a NaN would make every chi^2 NaN), but
# there is no Fisher sigma and nothing derived from it.
VERDICT_NO_NOISE = "no noise level"


def bias_verdict(bias_sigma, sigma, sigma_prior):
    bias_sigma, sigma = np.asarray(bias_sigma, dtype=float), np.asarray(sigma, dtype=float)
    return np.where(~np.isfinite(sigma) | (sigma >= sigma_prior), VERDICT_UNCONSTRAINED, np.where(bias_sigma >= 1., VERDICT_NEEDS_HI, VERDICT_NAIVE_OK))


# Observing time [h] at which the bias becomes a 1 sigma effect: the Fisher sigma scales as 1/sqrt(t), so t_crit = t_obs / (bias/sigma)^2. Independent
# of the n_orbits choice, unlike sigma_1min_crit. NaN where the quantity is unconstrained - there bias/sigma is a ratio to a meaningless number (the
# figure prints "-").
def critical_time(t_obs, bias_sigma, verdict):
    return np.where(np.asarray(verdict) == VERDICT_UNCONSTRAINED, np.nan, np.asarray(t_obs, dtype=float) / np.asarray(bias_sigma, dtype=float) ** 2)


# Central-difference derivative of fn at x, one-sided at a bound (a railed fit, or F = 0, which is a common fit result) so the step never leaves [lo,
# hi].
def _bounded_diff(fn, x, h, bounds):
    lo, hi = max(x - h, bounds[0]), min(x + h, bounds[1])
    return (fn(hi) - fn(lo)) / (hi - lo)


# Whitened Jacobian of the binned model in (A_B) or (A_B, F): J = d(model) / d(parameter) / sigma_bin per bin, detector blocks concatenated, each
# column with its mean removed per block (free_offset: that is what marginalises the baseline). Central differences, one-sided at a bound. Shared by
# fisher_covariance and the nuisance columns (nuisance_problem).
def whitened_jacobian(cfg, insolation_model, A_B_, F=None, free_offset=True, h_A=1e-3, h_F=1e-3, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, refl_unit=None, **kw):
    # truth=None: the residual is (truth - model) = -model, so any constant truth cancels in a central difference
    resid, _, sigma = _chi2_problem(cfg, None, insolation_model, free_offset=free_offset, refl_unit=refl_unit, **kw)
    cols = [_bounded_diff(lambda a: resid(a, F), A_B_, h_A, bounds)]
    if F is not None:
        cols.append(_bounded_diff(lambda f: resid(A_B_, f), F, h_F, F_bounds))
    return -np.column_stack(cols) / sigma[:, None]


# Fisher covariance of (A_B) or (A_B, F), the baseline(s) marginalised (free_offset removes the mean from each detector's residual): C = (J' W J)^-1,
# W = diag(1 / sigma^2). Its diagonal gives fisher_sigma; the off-diagonal is the A_B-F correlation that a DERIVED quantity (e.g. the occultation
# depth, fisher_depth) needs. cfg may be a detector set.
def fisher_covariance(cfg, insolation_model, A_B_, F=None, free_offset=True, h_A=1e-3, h_F=1e-3, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, refl_unit=None, **kw):
    J = whitened_jacobian(cfg, insolation_model, A_B_, F=F, free_offset=free_offset, h_A=h_A, h_F=h_F, bounds=bounds, F_bounds=F_bounds, refl_unit=refl_unit, **kw)
    try:
        return np.linalg.inv(J.T @ J)   # the inverse is what marginalises; 1/F_ii would instead fix the other parameters
    except np.linalg.LinAlgError:
        return np.full((J.shape[1], J.shape[1]), np.nan)   # singular J'J: a parameter the model does not respond to, or two that are exactly degenerate - NaN, like bias_cell on a rail


# Calculate the Fisher sigma for A_B (and F): the marginal sigmas sqrt(diag(C)) of fisher_covariance.
# sigma_F gets truncated by the prior boundary F=0, so it is not trustworthy.
def fisher_sigma(cfg, insolation_model, A_B_, F=None, **kw):
    return np.sqrt(np.diag(fisher_covariance(cfg, insolation_model, A_B_, F=F, **kw)))


# Occultation depth [ppm] of a fitted model, taken exactly at mid-occultation rather than read off the phase curve just before ingress (thermal +
# reflected; the free baseline drops out of a depth). Thermal: occultation_depth() of the SPIDERMAN model that model_pcurve evaluates (unblocked minus
# blocked flux at the occultation time). Reflected: the planet's reflected flux at full phase, which is exactly what the occultation hides (assumes a
# full, non-grazing occultation, like occultation_depth's single totality point). n_layers: the naive (and naive_redist) depth converges only
# ~1/n_layers (pcurve_params), so it is set well above the fitting default; use the same value for truth and fit. kw: the model_pcurve keywords of the
# fit (n_lo, n_la, insol, a_rs / T_s / rp_rs), the rest is ignored.
DEPTH_N_LAYERS = 200


def model_depth(cfg, insolation_model, A_B_, F=None, n_layers=DEPTH_N_LAYERS, **kw):
    geom = {k: kw[k] for k in ("a_rs", "T_s", "rp_rs") if k in kw}
    p = pcurve_params(cfg, insolation_model, A_B_, F=F, n_layers=n_layers, **geom, **{k: kw[k] for k in ("n_lo", "n_la", "insol") if k in kw})
    depth = occultation_depth(p, stellar_grid=cfg.stellar_grid, emissivity=cfg.emissivity)
    if cfg.reflected:
        planet = swept_planet(cfg, **geom)
        eps_occultation = np.arccos(np.sin(np.radians(planet.incl)))   # epsilon at full phase, as in reflected_PhaseCurve
        depth += 1e6 * float(reflected_lightcurve(eps_occultation, planet, u_bol=cfg.u_bol, A=cfg.A_(A_B_), insolation_model=insolation_model))
    return float(depth)


# Fisher sigma of the occultation depth of a fit, propagated from the parameter covariance: sigma_d^2 = g' C g with g = d(depth)/d(A_B, F) (same steps
# and bounds as fisher_covariance). Exact to first order: the depth is a smooth function of (A_B, F). Returns (depth, sigma_depth) [ppm]. For a
# detector set, C comes from the joint fit and the depth is that of `depth_cfg`, one of its detectors (the depth is a per-band quantity; default: the
# first detector).
def fisher_depth(cfg, insolation_model, A_B_, F=None, h_A=1e-3, h_F=1e-3, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, n_layers=DEPTH_N_LAYERS, depth_cfg=None, **kw):
    C = fisher_covariance(cfg, insolation_model, A_B_, F=F, h_A=h_A, h_F=h_F, bounds=bounds, F_bounds=F_bounds, **kw)
    if depth_cfg is None: depth_cfg = cfg[0] if is_detector_set(cfg) else cfg
    depth = lambda a, f: model_depth(depth_cfg, insolation_model, a, F=f, n_layers=n_layers, **kw)
    g = [_bounded_diff(lambda a: depth(a, F), A_B_, h_A, bounds)]
    if F is not None:
        g.append(_bounded_diff(lambda f: depth(A_B_, f), F, h_F, F_bounds))
    g = np.array(g)
    return depth(A_B_, F), float(np.sqrt(g @ C @ g))


############################################################################################################################################
#####   Spurious redistribution and critical noise
############################################################################################################################################

# Both fits and the statistic as a dict (the row of the false-positive tables): A_B_fit_naive, chi2_naive, A_B_fit, F_fit, chi2_joint, dchi2_F (NaN on
# a rail), railed and T_night_fit (NaN on a rail), the nightside temperature [K] that the fitted (A_B, F) imply (Kreidberg & Loeb eq. 1): what the
# false positive would be reported as. cfg may be a detector set.
def false_F_fit(cfg, A_B_true, free_offset=True, rail_tol=RAIL_TOL, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, **kw):
    fit = fit_truth(cfg, A_B_true, models=("naive", "naive_redist"), free_offset=free_offset, bounds=bounds, F_bounds=F_bounds, **kw)
    c0 = cfg[0] if is_detector_set(cfg) else cfg
    return _false_F_result(fit["naive"], fit["naive_redist"], c0.planet, c0.emissivity_bol, bounds, F_bounds, rail_tol)


# The dict of false_F_fit from its two fits n (A_B-only) and r (joint), given as fit_truth results (A_B_fit, chi2, and F_fit for r). Shared with the
# fits= path of the false-positive tables, which takes the two fits from the bias tables instead of refitting. planet, emissivity_bol: those of the
# fit's cfg (its first detector), for the implied nightside temperature.
def _false_F_result(n, r, planet, emissivity_bol, bounds, F_bounds, rail_tol=RAIL_TOL):
    railed = fit_railed(n, bounds, F_bounds, rail_tol) or fit_railed(r, bounds, F_bounds, rail_tol)
    return dict(A_B_fit_naive=n["A_B_fit"], chi2_naive=n["chi2"], A_B_fit=r["A_B_fit"], F_fit=np.nan if railed else r["F_fit"], chi2_joint=r["chi2"],
                dchi2_F=np.nan if railed else max(n["chi2"] - r["chi2"], 0.), railed=bool(railed),
                T_night_fit=np.nan if railed else float(surface_temperature(np.pi, planet, "naive_redist", A_B=r["A_B_fit"], emissivity_bol=emissivity_bol, F=r["F_fit"])))


# The per-minute noise [ppm/min] at which the spurious F is a z-sigma detection (dchi2_F = z^2), at the n_orbits of the fit: sigma_crit = sigma_1min *
# sqrt(dchi2_F) / z, where sigma_1min is the noise the fit was evaluated with. Exact, like measurability's sigma_1min_crit: dchi2_F ∝ 1 /
# sigma_1min^2. A planet observed with better noise than sigma_crit gets the false positive.
def critical_sigma_1min(dchi2, sigma_1min, z=2.):
    return sigma_1min * np.sqrt(np.asarray(dchi2, float)) / z
