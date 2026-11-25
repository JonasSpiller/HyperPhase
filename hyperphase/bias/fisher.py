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

def fit_truth(cfg, A_B_true, models=("naive",), free_offset=True, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, fisher=False, **kw):
    """The pipeline behind every bias number: the noiseless extended-source truth at A_B_true, fitted with each of `models` ("naive": A_B;
    "naive_redist": A_B and F). Returns {model: {"A_B_fit", "bias_A_B", "chi2"[, "F_fit"][, "sigma_A_B"[, "sigma_F"]]}}. The truth has F = 0, so
    F_fit is itself the bias in F. fisher=True adds the Fisher sigmas, evaluated at the fits, where the misspecified model's posterior sits."""
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
        if fisher:   # 2 parameters + baseline: A_B's marginal width includes the F degeneracy
            out["naive_redist"]["sigma_A_B"], out["naive_redist"]["sigma_F"] = fisher_sigma(cfg, "naive_redist", A_fit, F=F_fit, bounds=bounds, F_bounds=F_bounds, **kw)
    return out


def measurability(bias, sigma, sigma_1min):
    """bias / sigma, and the per-minute noise sigma_1min_crit at which |bias| = sigma. sigma_1min_crit = sigma_1min * |bias| / sigma is exact, not an
    extrapolation: the bias does not depend on the noise, and the Fisher sigma is proportional to sigma_1min."""
    bias_sigma = np.abs(bias) / sigma
    return bias_sigma, sigma_1min * bias_sigma


def flat_prior_sigma(bounds):
    """Std of the flat prior U(lo, hi): (hi - lo) / sqrt(12). A Fisher sigma at or above it means the data add nothing to the prior. The verdicts
    take the physical range (A_B_BOUNDS, F_BOUNDS: the dynesty priors), never the least-squares range A_B_FIT_BOUNDS, so "unconstrained" means the
    data would not narrow the parameter within what an observer allows."""
    return (bounds[1] - bounds[0]) / np.sqrt(12.)


# Verdicts of the bias test (bias_verdict). Where sigma >= the flat-prior std, bias / sigma divides by a meaningless number: "unconstrained", not
# "naive OK".
VERDICT_NEEDS_HI, VERDICT_NAIVE_OK, VERDICT_UNCONSTRAINED = "needs hyperillum", "naive OK", "unconstrained"

# A row without a noise level (no PandExo mode and no CUSTOM_SIGMA_1MIN entry) is fitted at cfg's own sigma_1min, which only has to be finite. Its
# fitted values and bias are exact, since they do not depend on the noise; the Fisher sigma and everything derived from it are NaN.
VERDICT_NO_NOISE = "no noise level"


def bias_verdict(bias_sigma, sigma, sigma_prior):
    """VERDICT_UNCONSTRAINED where sigma is not finite or >= sigma_prior (flat_prior_sigma of the physical range), else VERDICT_NEEDS_HI where
    bias_sigma >= 1, else VERDICT_NAIVE_OK."""
    bias_sigma, sigma = np.asarray(bias_sigma, dtype=float), np.asarray(sigma, dtype=float)
    return np.where(~np.isfinite(sigma) | (sigma >= sigma_prior), VERDICT_UNCONSTRAINED, np.where(bias_sigma >= 1., VERDICT_NEEDS_HI, VERDICT_NAIVE_OK))


def critical_time(t_obs, bias_sigma, verdict):
    """Observing time [h] at which the bias is a 1 sigma effect: t_crit = t_obs / (bias/sigma)^2, since the Fisher sigma scales as 1/sqrt(t). Unlike
    sigma_1min_crit it does not depend on n_orbits. NaN where unconstrained."""
    return np.where(np.asarray(verdict) == VERDICT_UNCONSTRAINED, np.nan, np.asarray(t_obs, dtype=float) / np.asarray(bias_sigma, dtype=float) ** 2)


def _bounded_diff(fn, x, h, bounds):
    """Central-difference derivative of fn at x, one-sided at a bound (a railed fit, or the common F = 0), so the step never leaves [lo, hi]."""
    lo, hi = max(x - h, bounds[0]), min(x + h, bounds[1])
    return (fn(hi) - fn(lo)) / (hi - lo)


def whitened_jacobian(cfg, insolation_model, A_B_, F=None, free_offset=True, h_A=1e-3, h_F=1e-3, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, refl_unit=None, **kw):
    """Whitened Jacobian of the binned model in (A_B) or (A_B, F): J = d(model) / d(parameter) / sigma_bin per bin, detector blocks concatenated,
    each column's mean removed per block (free_offset, which marginalises the baseline). Central differences, one-sided at a bound."""
    # truth=None: the residual is (truth - model) = -model, so any constant truth cancels in a central difference
    resid, _, sigma = _chi2_problem(cfg, None, insolation_model, free_offset=free_offset, refl_unit=refl_unit, **kw)
    cols = [_bounded_diff(lambda a: resid(a, F), A_B_, h_A, bounds)]
    if F is not None:
        cols.append(_bounded_diff(lambda f: resid(A_B_, f), F, h_F, F_bounds))
    return -np.column_stack(cols) / sigma[:, None]


def fisher_covariance(cfg, insolation_model, A_B_, F=None, free_offset=True, h_A=1e-3, h_F=1e-3, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, refl_unit=None, **kw):
    """Fisher covariance C = (J'J)^-1 of (A_B) or (A_B, F), J the whitened Jacobian, baselines marginalised. Its diagonal gives fisher_sigma; the
    off-diagonal is the A_B-F correlation a derived quantity needs (fisher_depth)."""
    J = whitened_jacobian(cfg, insolation_model, A_B_, F=F, free_offset=free_offset, h_A=h_A, h_F=h_F, bounds=bounds, F_bounds=F_bounds, refl_unit=refl_unit, **kw)
    try:
        return np.linalg.inv(J.T @ J)   # the inverse marginalises; 1/F_ii would fix the other parameters
    except np.linalg.LinAlgError:
        return np.full((J.shape[1], J.shape[1]), np.nan)   # singular J'J (an inert or exactly degenerate parameter): NaN


def fisher_sigma(cfg, insolation_model, A_B_, F=None, **kw):
    """The marginal Fisher sigmas sqrt(diag(C)) of A_B (and F). sigma_F is not trustworthy: the prior edge F = 0 truncates the posterior."""
    return np.sqrt(np.diag(fisher_covariance(cfg, insolation_model, A_B_, F=F, **kw)))


# n_layers of model_depth: the naive and naive_redist depths converge only ~1/n_layers (pcurve_params), so it is set well above the fitting default.
DEPTH_N_LAYERS = 200


def model_depth(cfg, insolation_model, A_B_, F=None, n_layers=DEPTH_N_LAYERS, **kw):
    """Occultation depth [ppm] of a fitted model at mid-occultation, rather than read off the phase curve before ingress (thermal + reflected; the
    free baseline drops out of a depth). Thermal: occultation_depth of the SPIDERMAN model that model_pcurve evaluates. Reflected: the planet's
    reflected flux at full phase, which the occultation hides (assumes a full, non-grazing occultation). Use the same n_layers for truth and fit.
    kw: the model_pcurve keywords of the fit (n_lo, n_la, insol, a_rs, T_s, rp_rs); the rest is ignored."""
    geom = {k: kw[k] for k in ("a_rs", "T_s", "rp_rs") if k in kw}
    p = pcurve_params(cfg, insolation_model, A_B_, F=F, n_layers=n_layers, **geom, **{k: kw[k] for k in ("n_lo", "n_la", "insol") if k in kw})
    depth = occultation_depth(p, stellar_grid=cfg.stellar_grid, emissivity=cfg.emissivity)
    if cfg.reflected:
        planet = swept_planet(cfg, **geom)
        eps_occultation = np.arccos(np.sin(np.radians(planet.incl)))   # epsilon at full phase, as in reflected_PhaseCurve
        depth += 1e6 * float(reflected_lightcurve(eps_occultation, planet, u_bol=cfg.u_bol, A=cfg.A_(A_B_), insolation_model=insolation_model))
    return float(depth)


def fisher_depth(cfg, insolation_model, A_B_, F=None, h_A=1e-3, h_F=1e-3, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, n_layers=DEPTH_N_LAYERS, depth_cfg=None, **kw):
    """(depth, sigma_depth) [ppm] of a fit: the Fisher sigma propagated from the parameter covariance, sigma_d^2 = g' C g with g = d(depth) / d(A_B,
    F), exact to first order. A depth is per band: for a detector set C is the joint fit's and the depth that of depth_cfg (default: the first
    detector)."""
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

def false_F_fit(cfg, A_B_true, free_offset=True, rail_tol=RAIL_TOL, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, **kw):
    """Spurious heat redistribution: both fits of the noiseless truth (F = 0) and dchi2_F = chi2(A_B-only fit) - chi2(joint A_B + F fit), both at
    their best fit with a free baseline. sqrt(dchi2_F) is the median expected one-sided significance of "F > 0" (Asimov data; Cowan et al. 2011),
    exact for the nonlinear T ∝ F^(1/4) response and the F = 0 boundary (null distribution 1/2 chi2_1). Not F_fit / sigma_F: the Fisher sigma_F is
    the slope at F_fit, but the model responds ∝ F^~1.5 near 0, so that ratio overstates the significance, and it ignores the boundary.
    dchi2_F ∝ n_orbits / sigma_1min^2 (critical_sigma_1min).
    Returns the false-positive table row: A_B_fit_naive, chi2_naive, A_B_fit, F_fit, chi2_joint, dchi2_F, railed and T_night_fit, the nightside
    temperature [K] the fitted (A_B, F) imply (Kreidberg & Loeb 2016 eq. 1); F_fit, dchi2_F and T_night_fit are NaN on a rail."""
    fit = fit_truth(cfg, A_B_true, models=("naive", "naive_redist"), free_offset=free_offset, bounds=bounds, F_bounds=F_bounds, **kw)
    c0 = cfg[0] if is_detector_set(cfg) else cfg
    return _false_F_result(fit["naive"], fit["naive_redist"], c0.planet, c0.emissivity_bol, bounds, F_bounds, rail_tol)


def _false_F_result(n, r, planet, emissivity_bol, bounds, F_bounds, rail_tol=RAIL_TOL):
    """The dict of false_F_fit from its two fit_truth results n (A_B-only) and r (joint). planet, emissivity_bol: those of the fit's first detector.
    dchi2_F >= 0, since the A_B-only model is the F = 0 member of the joint one; small negative values from the tolerances are clipped."""
    railed = fit_railed(n, bounds, F_bounds, rail_tol) or fit_railed(r, bounds, F_bounds, rail_tol)
    return dict(A_B_fit_naive=n["A_B_fit"], chi2_naive=n["chi2"], A_B_fit=r["A_B_fit"], F_fit=np.nan if railed else r["F_fit"], chi2_joint=r["chi2"],
                dchi2_F=np.nan if railed else max(n["chi2"] - r["chi2"], 0.), railed=bool(railed),
                T_night_fit=np.nan if railed else float(surface_temperature(np.pi, planet, "naive_redist", A_B=r["A_B_fit"], emissivity_bol=emissivity_bol, F=r["F_fit"])))


def critical_sigma_1min(dchi2, sigma_1min, z=2.):
    """The per-minute noise [ppm/min] at which the spurious F is a z-sigma detection (dchi2_F = z^2), at the fit's n_orbits:
    sigma_1min * sqrt(dchi2_F) / z, with sigma_1min the noise of the fit. Exact (see measurability), since dchi2_F ∝ 1 / sigma_1min^2."""
    return sigma_1min * np.sqrt(np.asarray(dchi2, float)) / z
