import contextlib, io
import textwrap
from scipy.optimize import minimize_scalar
from scipy.interpolate import RegularGridInterpolator
from hyperillumination_phasecurve_utils import *
from astro_utils import roche_period_h, roche_proximity, rocky_mean_density, kepler_period_days, kepler_a_rs, rho_star
from HItargets_utils import ROCKY_MAX_RADE, DEFAULT_MAP_RP_RS, A_B_BOUNDS, A_B_FIT_BOUNDS, F_BOUNDS, ROCHE_CASES, ROCHE_R_P, hi_planets_by_name, apply_custom, resolve_instrument, detector_label
from functools import lru_cache
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor


# Process-parallel map over independent SPIDERMAN evaluations (the C extension holds the GIL, so threads give no speedup).
# Workers are recycled afterwards, since SPIDERMAN's allocator growth accumulates in reused workers.
def pmap(fn, items, n_jobs=-1):
    out = Parallel(n_jobs=n_jobs, backend="loky")(delayed(fn)(*item) for item in items)
    get_reusable_executor().shutdown(wait=True)
    return out


# A_B bias converted to a bias in local surface temperature
# With a fitted heat redistribution F_fit (naive_redist) this is the substellar temperature, where T^4 ∝ (1-A_B)(1-3F/2) (Kreidberg & Loeb 2016 eq. 1)
def temperature_bias(A_B_true, A_B_bias, F_fit=0.):
    A_fit = A_B_true + A_B_bias
    return ((1 - A_fit) * (1 - 1.5 * F_fit) / (1 - A_B_true))**0.25 - 1

###########################################################################################################################
#####
#####   Detectability
#####
###########################################################################################################################

# Average y in fixed-width time bins. t in days, minutes_binning in minutes.
# Simple binning, last bin gets remaining points (no need for specific t.size, y.size for equal bins). Only for plotting, use binned for fitting.
def bin_curve(t, y, minutes_binning):
    dt = minutes_binning / 60 / 24
    edges = np.arange(t[0], t[-1] + dt, dt)  # bin edges
    idx = np.digitize(t, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    nb = len(edges) - 1
    cnt = np.bincount(idx[ok], minlength=nb)
    tot = np.bincount(idx[ok], weights=y[ok], minlength=nb)
    m = cnt > 0
    t_bin = 0.5 * (edges[:-1] + edges[1:])
    return t_bin[m], tot[m] / cnt[m]




###########################################################################################################################
#####
#####   Simple A_B-Bias fitting model
#####
###########################################################################################################################


# Exact block-average bins (valid only when n_t is a multiple of n_bins)
def binned(cfg, y):
    k = len(y) // cfg.n_bins
    if k * cfg.n_bins != len(y):
        raise ValueError(f"n_t={len(y)} is not a multiple of n_bins={cfg.n_bins}; "
                         f"use n_t = {cfg.n_bins} * k (e.g. {cfg.n_bins * 40})")
    return y.reshape(cfg.n_bins, k).mean(axis=1)


# Binned truth minus the binned model_pcurve [ppm]; with free_offset the mean is removed (a fitted baseline). Shared by fit_naive, fit_redist and chi^2 landscapes.
# kw goes to model_pcurve (F for naive_redist, refl_unit, n_layers, ...)
def fit_residual(cfg, yb_truth, insolation_model, A_B_, free_offset=True, **kw):
    r = yb_truth - binned(cfg, model_pcurve(cfg, insolation_model, A_B_, **kw)[1])
    return r - r.mean() if free_offset else r


# A detector set: several FittingConfigs fitted JOINTLY with one shared (A_B, F) and a separate baseline per detector (e.g. the two NIRSpec/G395H arrays). Wherever a fit takes `cfg`,
# a list/tuple of configs means such a set; `truth` is then a list of one truth curve per detector. The configs must share the geometry (planet, n_bins, u_bol, A_) and differ only in
# band (stellar_grid, filter_path, l1, l2) and noise. The per-detector baseline is the free_offset of each detector's own residual block, which is what the Fisher matrix marginalises too.
def is_detector_set(cfg):
    return isinstance(cfg, (list, tuple))


# The config that carries the instrument-independent settings of a cfg or detector set (planet, albedo model, n_bins, n_orbits, limb darkening, placeholder noise): the set's first detector.
def template_cfg(cfg):
    return cfg[0] if is_detector_set(cfg) else cfg


# The band (FILTER_LAMBDA_UM key) of each detector of a cfg or detector set, in the set's order.
def cfg_detectors(cfg):
    return [_band_of(c) for c in (cfg if is_detector_set(cfg) else [cfg])]


# The noiseless hyperillumination truth every fit of this module is made against: model_pcurve(cfg, "hyperillum", A_B_true)[1], or for a detector set the list of one curve per detector.
def truth_curves(cfg, A_B_true, **kw):
    return [model_pcurve(c, "hyperillum", A_B_true, **kw)[1] for c in cfg] if is_detector_set(cfg) else model_pcurve(cfg, "hyperillum", A_B_true, **kw)[1]


# The one per-minute noise of a cfg, or of a detector set whose detectors share it (the placeholder noise of the maps). A critical noise read off a map is this value scaled by the
# map's dchi2, which means nothing for detectors fitted at different noise levels, so those raise.
def common_sigma_1min(cfg):
    levels = {float(c.sigma_1min) for c in (cfg if is_detector_set(cfg) else [cfg])}
    if len(levels) != 1: raise ValueError(f"the detectors have different noise levels {sorted(levels)} ppm/min: a critical noise needs one common sigma_1min (give every detector the same placeholder)")
    return levels.pop()


# The chi^2 problem of fitting `insolation_model` to the hyperillumination truth: resid(A_B_, F=None) [ppm], cost(A_B_, F=None) = sum((resid / sigma)^2) and sigma, the per-point
# noise [ppm] (one sigma_bin per detector block). Shared by fit_naive, fit_redist, fit_chi2 and fisher_covariance. The reflected shape does not depend on A_B, F or the band,
# so it is built once (from the first detector) unless refl_unit is passed. truth=None gives the bare model residual (truth = 0), which is all a derivative needs.
def _chi2_problem(cfg, truth, insolation_model, free_offset=True, refl_unit=None, **kw):
    cfgs = list(cfg) if is_detector_set(cfg) else [cfg]
    truths = [None] * len(cfgs) if truth is None else (list(truth) if is_detector_set(cfg) else [truth])
    if refl_unit is None and cfgs[0].reflected: refl_unit = reflected_unit(cfgs[0], insolation_model, **kw)
    yb_truth = [0. if t is None else binned(c, t) for c, t in zip(cfgs, truths)]
    sigma = np.concatenate([np.full(c.n_bins, c.sigma_bin) for c in cfgs])

    def resid(A_B_, F=None):   # each detector block with its own baseline removed (free_offset)
        return np.concatenate([fit_residual(c, y, insolation_model, A_B_, F=F, free_offset=free_offset, refl_unit=refl_unit, **kw) for c, y in zip(cfgs, yb_truth)])

    def cost(A_B_, F=None):
        return np.sum((resid(A_B_, F) / sigma) ** 2)

    return resid, cost, sigma


# Best-fit Bond albedo A_B of the one-parameter model `insolation_model` against a noiseless truth, returning (A_fit, chi2, resid_rms); the band albedo A follows from cfg.A_.
# "naive" against a hyperillumination truth is fit_naive; "hyperillum" against a point-source truth is the null test (the extended-source model fitted to data without hyperillumination).
def fit_AB_only(cfg, truth, insolation_model, free_offset=True, bounds=A_B_FIT_BOUNDS, xatol=1e-4, **kw):
    resid, cost, _ = _chi2_problem(cfg, truth, insolation_model, free_offset=free_offset, **kw)
    res = minimize_scalar(cost, bounds=bounds, method="bounded", options={"xatol": xatol})
    r = resid(res.x)
    resid_rms = np.sqrt(np.mean(r ** 2))
    return res.x, res.fun, resid_rms # chi2 = cost(A_fit) = weighted sum of squared residuals (this χ² is not a goodness-of-fit to an observation — it's the squared distance between two noiseless models, expressed in units of the observational noise)


# Best-fit naive Bond albedo A_B against a hyperillumination-generated truth, returning (A_fit, chi2, resid_rms): fit_AB_only with the point-source model.
def fit_naive(cfg, truth, free_offset=True, bounds=A_B_FIT_BOUNDS, xatol=1e-4, **kw):
    return fit_AB_only(cfg, truth, "naive", free_offset=free_offset, bounds=bounds, xatol=xatol, **kw)


# F nodes of fit_redist's profile scan, as fractions of the F_bounds width: dense near F = 0, where the joint optimum usually sits (F ~ 0.005 for a hyperillum truth).
F_SCAN_FRAC = np.array([0., 0.004, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 0.6, 1.])


# Best-fit naive_redist (A_B, F) against a hyperillumination-generated truth, returning (A_B_fit, F_fit, chi2, resid_rms): the 2-parameter version of fit_naive (same binning, noise weighting and offset handling).
# Profile fit, derivative-free and start-free: A_B is profiled out with the same bounded Brent as fit_naive, chi2_p(F) = min_A_B chi2(A_B, F); chi2_p is scanned on F_SCAN_FRAC to bracket
# its global minimum, then refined with bounded Brent inside that bracket (the scan node wins if it is lower, so F = F_bounds[0] exactly is a possible result).
# Why not a 2-D local optimizer: the K&L nightside is T ∝ F^(1/4), so the model's response to F switches on extremely flat at F = 0 (residual change ∝ F^~1.5; measured, MIRI/LRS, a/Rs = 2.42, T_eff = 4000 K).
# The F = 0 edge is then a near-stationary trap: the former Nelder-Mead fit, started on it, collapsed onto F = 0 in about half the bias-map cells (chi2 0.445 instead of 0.226 there), and gradient methods see ~zero dchi2/dF.
# Profiling also makes the outer problem insensitive to the inner tolerance (the profile error is second order in the A_B error). ~250 model evaluations per fit.
def fit_redist(cfg, truth, free_offset=True, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, xatol=1e-5, F_xatol=1e-5, **kw):
    resid, cost, _ = _chi2_problem(cfg, truth, "naive_redist", free_offset=free_offset, **kw)

    def profile(F):   # (chi2, A_B) at the best A_B for fixed F
        r = minimize_scalar(lambda A_B_: cost(A_B_, F), bounds=bounds, method="bounded", options={"xatol": xatol})
        return r.fun, r.x

    F_scan = F_bounds[0] + (F_bounds[1] - F_bounds[0]) * F_SCAN_FRAC
    scan = [profile(F) for F in F_scan]
    k = int(np.argmin([c for c, _ in scan]))
    bracket = (F_scan[max(k - 1, 0)], F_scan[min(k + 1, len(F_scan) - 1)])
    F_fit = minimize_scalar(lambda F: profile(F)[0], bounds=bracket, method="bounded", options={"xatol": F_xatol}).x
    chi2, A_B_fit = profile(F_fit)
    if scan[k][0] <= chi2:   # bounded Brent never evaluates the bracket ends: keep the scan node (e.g. F exactly at its bound) if it is at least as good
        F_fit, (chi2, A_B_fit) = F_scan[k], scan[k]
    r = resid(A_B_fit, F_fit)
    resid_rms = np.sqrt(np.mean(r ** 2))
    return A_B_fit, F_fit, chi2, resid_rms


# chi^2 of one model against the hyperillumination truth, with the same residual and noise weighting as the fits (F only for naive_redist).
def fit_chi2(cfg, truth, insolation_model, A_B_, F=None, free_offset=True, refl_unit=None, **kw):
    _, cost, _ = _chi2_problem(cfg, truth, insolation_model, free_offset=free_offset, refl_unit=refl_unit, **kw)
    return float(cost(A_B_, F))


def _chi2_redist_at(cfg, truth, A_B_, F, refl_unit, free_offset, kw):
    return fit_chi2(cfg, truth, "naive_redist", A_B_, F=F, free_offset=free_offset, refl_unit=refl_unit, **kw)


# chi^2 of naive_redist over an (A_B, F) grid, shape (len(AB_grid), len(F_grid)), evaluated in parallel. The reflected shape does not depend on F, so it is built once.
# cfg may be a detector set (truth: one curve per detector, truth_curves): chi^2 summed over the detectors, each with its own baseline.
def chi2_landscape(cfg, truth, AB_grid, F_grid, free_offset=True, n_jobs=-1, **kw):
    c0 = template_cfg(cfg)
    refl_unit = reflected_unit(c0, "naive_redist", **kw) if c0.reflected else None   # band-independent, so the first detector's serves the set
    X = pmap(_chi2_redist_at, [(cfg, truth, a, f, refl_unit, free_offset, kw) for a in AB_grid for f in F_grid], n_jobs)
    return np.array(X).reshape(len(AB_grid), len(F_grid))


# Extent of the region d_chi2 < level on a landscape grid: ((A_B min, A_B max), F max).
def chi2_region(d_chi2, AB_grid, F_grid, level=2.30):
    inside = d_chi2 < level
    return (AB_grid[inside.any(axis=1)].min(), AB_grid[inside.any(axis=1)].max()), F_grid[inside.any(axis=0)].max()


###########################################################################################################################
#####
#####   Bias vs true A_B
#####
###########################################################################################################################

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


# The systematic in units of the statistical error bar, and the per-minute precision sigma_1min_crit at which |bias| = sigma.
# sigma_1min_crit = sigma_1min * |bias| / sigma is exact rather than an extrapolation: the bias does not depend on the noise, and the Fisher sigma is strictly proportional to sigma_1min.
def measurability(bias, sigma, sigma_1min):
    bias_sigma = np.abs(bias) / sigma
    return bias_sigma, sigma_1min * bias_sigma


# Std of the flat prior U(lo, hi) over a parameter's range: (hi - lo) / sqrt(12). A Fisher sigma at or above it means the data add nothing to the prior, so the Gaussian posterior the Fisher matrix assumes does not exist.
# The verdicts always take the PHYSICAL range (A_B_BOUNDS, F_BOUNDS), the priors of the dynesty fits, not the wider least-squares range of A_B (A_B_FIT_BOUNDS):
# "unconstrained" means the data would not narrow A_B within what an observer allows.
def flat_prior_sigma(bounds):
    return (bounds[1] - bounds[0]) / np.sqrt(12.)


# Three-way verdict of the bias test. bias_sigma < 1 alone reads as "naive OK", but with sigma >= the flat-prior std it only says the parameter is unmeasured — the bias is then divided by a meaningless number. That case is "unconstrained", not "naive OK".
VERDICT_NEEDS_HI, VERDICT_NAIVE_OK, VERDICT_UNCONSTRAINED = "needs hyperillum", "naive OK", "unconstrained"
# A row fitted WITHOUT a noise level (no PandExo mode and no CUSTOM_SIGMA_1MIN entry, e.g. MIRI imaging): the fitted values and the bias are exact, since they do not depend on the
# noise (the fit runs at cfg's own sigma_1min, which only has to be finite: a NaN would make every chi^2 NaN), but there is no Fisher sigma and nothing derived from it.
VERDICT_NO_NOISE = "no noise level"

def bias_verdict(bias_sigma, sigma, sigma_prior):
    bias_sigma, sigma = np.asarray(bias_sigma, dtype=float), np.asarray(sigma, dtype=float)
    return np.where(~np.isfinite(sigma) | (sigma >= sigma_prior), VERDICT_UNCONSTRAINED, np.where(bias_sigma >= 1., VERDICT_NEEDS_HI, VERDICT_NAIVE_OK))


# Observing time [h] at which the bias becomes a 1 sigma effect: the Fisher sigma scales as 1/sqrt(t), so t_crit = t_obs / (bias/sigma)^2. Independent of the n_orbits choice, unlike sigma_1min_crit.
# NaN where the quantity is unconstrained - there bias/sigma is a ratio to a meaningless number (the figure prints "-").
def critical_time(t_obs, bias_sigma, verdict):
    return np.where(np.asarray(verdict) == VERDICT_UNCONSTRAINED, np.nan, np.asarray(t_obs, dtype=float) / np.asarray(bias_sigma, dtype=float) ** 2)


def _fit_truth_at(cfg, A_B_true, models, bounds, kw):
    return fit_truth(cfg, A_B_true, models=models, free_offset=True, bounds=bounds, fisher=True, **kw)


# Refit the models at every TRUE Bond albedo in A_B_grid (single target geometry, free baseline), in parallel.
# Returns {model: {name: array}} with the keys of fit_truth (fisher=True): "naive" always, "naive_redist" with redist=True.
def bias_vs_true_AB(cfg, A_B_grid, redist=True, bounds=A_B_FIT_BOUNDS, n_jobs=-1, **kw):
    models = ("naive", "naive_redist") if redist else ("naive",)
    rows = pmap(_fit_truth_at, [(cfg, a, models, bounds, kw) for a in A_B_grid], n_jobs)
    return {m: {k: np.array([r[m][k] for r in rows]) for k in rows[0][m]} for m in models}

# Central-difference derivative of fn at x, one-sided at a bound (a railed fit, or F = 0, which is a common fit result) so the step never leaves [lo, hi].
def _bounded_diff(fn, x, h, bounds):
    lo, hi = max(x - h, bounds[0]), min(x + h, bounds[1])
    return (fn(hi) - fn(lo)) / (hi - lo)


# Whitened Jacobian of the binned model in (A_B) or (A_B, F): J = d(model) / d(parameter) / sigma_bin per bin, detector blocks concatenated, each column with its mean removed per
# block (free_offset: that is what marginalises the baseline). Central differences, one-sided at a bound. Shared by fisher_covariance and the nuisance columns (nuisance_problem).
def whitened_jacobian(cfg, insolation_model, A_B_, F=None, free_offset=True, h_A=1e-3, h_F=1e-3, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, refl_unit=None, **kw):
    # truth=None: the residual is (truth - model) = -model, so any constant truth cancels in a central difference
    resid, _, sigma = _chi2_problem(cfg, None, insolation_model, free_offset=free_offset, refl_unit=refl_unit, **kw)
    cols = [_bounded_diff(lambda a: resid(a, F), A_B_, h_A, bounds)]
    if F is not None:
        cols.append(_bounded_diff(lambda f: resid(A_B_, f), F, h_F, F_bounds))
    return -np.column_stack(cols) / sigma[:, None]


# Fisher covariance of (A_B) or (A_B, F), the baseline(s) marginalised (free_offset removes the mean from each detector's residual): C = (J' W J)^-1, W = diag(1 / sigma^2).
# Its diagonal gives fisher_sigma; the off-diagonal is the A_B-F correlation that a DERIVED quantity (e.g. the occultation depth, fisher_depth) needs. cfg may be a detector set.
def fisher_covariance(cfg, insolation_model, A_B_, F=None, free_offset=True, h_A=1e-3, h_F=1e-3, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, refl_unit=None, **kw):
    J = whitened_jacobian(cfg, insolation_model, A_B_, F=F, free_offset=free_offset, h_A=h_A, h_F=h_F, bounds=bounds, F_bounds=F_bounds, refl_unit=refl_unit, **kw)
    try:
        return np.linalg.inv(J.T @ J)   # the inverse is what marginalises; 1/F_ii would instead fix the other parameters
    except np.linalg.LinAlgError:
        return np.full((J.shape[1], J.shape[1]), np.nan)   # singular J'J: a parameter the model does not respond to, or two that are exactly degenerate - NaN, like bias_cell on a rail


# Calculate the Fisher sigma for A_B (and F): the marginal sigmas sqrt(diag(C)) of fisher_covariance.
# sigma_F gets truncated by the prior boundry F=0, so it is not trustworthy.
def fisher_sigma(cfg, insolation_model, A_B_, F=None, **kw):
    return np.sqrt(np.diag(fisher_covariance(cfg, insolation_model, A_B_, F=F, **kw)))


# Occultation depth [ppm] of a fitted model, taken exactly at mid-occultation rather than read off the phase curve just before ingress (thermal + reflected; the free baseline drops out of a depth).
# Thermal: occultation_depth() of the SPIDERMAN model that model_pcurve evaluates (unblocked minus blocked flux at the occultation time). Reflected: the planet's reflected flux at full phase,
# which is exactly what the occultation hides (assumes a full, non-grazing occultation, like occultation_depth's single totality point).
# n_layers: the naive (and naive_redist) depth converges only ~1/n_layers (pcurve_params), so it is set well above the fitting default; use the same value for truth and fit.
# kw: the model_pcurve keywords of the fit (n_lo, n_la, insol, a_rs / T_s / rp_rs), the rest is ignored.
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


# Fisher sigma of the occultation depth of a fit, propagated from the parameter covariance: sigma_d^2 = g' C g with g = d(depth)/d(A_B, F) (same steps and bounds as fisher_covariance).
# Exact to first order: the depth is a smooth function of (A_B, F). Returns (depth, sigma_depth) [ppm].
# For a detector set, C comes from the joint fit and the depth is that of `depth_cfg`, one of its detectors (the depth is a per-band quantity; default: the first detector).
def fisher_depth(cfg, insolation_model, A_B_, F=None, h_A=1e-3, h_F=1e-3, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, n_layers=DEPTH_N_LAYERS, depth_cfg=None, **kw):
    C = fisher_covariance(cfg, insolation_model, A_B_, F=F, h_A=h_A, h_F=h_F, bounds=bounds, F_bounds=F_bounds, **kw)
    if depth_cfg is None: depth_cfg = cfg[0] if is_detector_set(cfg) else cfg
    depth = lambda a, f: model_depth(depth_cfg, insolation_model, a, F=f, n_layers=n_layers, **kw)
    g = [_bounded_diff(lambda a: depth(a, F), A_B_, h_A, bounds)]
    if F is not None:
        g.append(_bounded_diff(lambda f: depth(A_B_, f), F, h_F, F_bounds))
    g = np.array(g)
    return depth(A_B_, F), float(np.sqrt(g @ C @ g))


###########################################################################################################################
#####
#####   Bias Map
#####
###########################################################################################################################


# A railed fit: a fitted parameter within RAIL_TOL of a bound of its fit range, i.e. the point-source model cannot reproduce the truth at any value inside the range, so the bias is undefined
# there (bias_cell / false_F_fit return NaN, the sweeps flag the point). F = 0 is a valid fit, only F's upper bound counts as a rail. With the default A_B_FIT_BOUNDS = (-2, 1) a rail is a
# safety net that does not fire in practice; with bounds=A_B_BOUNDS (the physical range) the A_B = 0 floor is a rail again.
RAIL_TOL = 1e-3

def is_railed(x, bounds, rail_tol=RAIL_TOL):
    """True where x sits within rail_tol of either bound (scalar or array)."""
    x = np.asarray(x, dtype=float)
    return ((x - bounds[0]) < rail_tol) | ((bounds[1] - x) < rail_tol)

def fit_railed(fit, bounds, F_bounds=F_BOUNDS, rail_tol=RAIL_TOL):
    """Rail test of one model's fit_truth result: A_B on either bound, or (joint fit) F on its upper bound."""
    return bool(is_railed(fit["A_B_fit"], bounds, rail_tol)) or ("F_fit" in fit and (F_bounds[1] - fit["F_fit"]) < rail_tol)


# One grid cell of a bias map: generate the hyperillumination truth, fit the naive model, return the recovered-A_B bias.
# NaN if the fit railed against a bound, i.e. the naive model cannot reproduce the truth at ANY albedo.
# redist=True fits naive_redist (A_B and F) instead and returns (A_B bias, F_fit); the truth has no redistribution, so F_fit is itself the bias in F.
# return_chi2=True appends the fit's chi2 (at cfg.sigma_1min), which false_F_map_grid(fits=...) turns into dchi2_F without refitting.
def bias_cell(cfg, A_B_true, free_offset=True, rail_tol=RAIL_TOL, temp_bias=False, bounds=A_B_FIT_BOUNDS, redist=False, F_bounds=F_BOUNDS, return_chi2=False, **kw):
    """Recovered-A_B bias for one map cell. NaN if the fit railed against a bound,
    i.e. the naive model cannot reproduce the truth at ANY albedo.
    With redist=True: (A_B bias, F_fit), both NaN on a rail. F_fit = 0 is a valid fit, only F's upper bound counts as a rail.
    With return_chi2=True the fit's chi2 is appended (bias, chi2) / (bias, F_fit, chi2), also NaN on a rail."""
    model = "naive_redist" if redist else "naive"
    fit = fit_truth(cfg, A_B_true, models=(model,), free_offset=free_offset, bounds=bounds, F_bounds=F_bounds, **kw)[model]   # same bounds the rail check below uses
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


# A whole bias map over (a_rs_grid, y_grid), cells in parallel. y_param is the swept geometry keyword of model_pcurve ("T_s" or "rp_rs").
# Returns the A_B-bias map, shape (len(y_grid), len(a_rs_grid)); with redist=True the pair (A_B-bias map, F map).
# return_chi2=True appends the chi2 map of the fits (NaN on a rail): the two maps' chi2 give false_F_map_grid(fits=...) at no extra cost.
def bias_map_grid(cfg, A_B_true, a_rs_grid, y_grid, y_param, redist=False, n_jobs=-1, return_chi2=False, **kw):
    flat = pmap(_bias_cell_at, [(cfg, A_B_true, ar, y_param, y, redist, return_chi2, kw) for y in y_grid for ar in a_rs_grid], n_jobs)
    n_out = 1 + int(redist) + int(return_chi2)
    out = np.array(flat).reshape(len(y_grid), len(a_rs_grid), *((n_out,) if n_out > 1 else ()))
    return tuple(np.moveaxis(out, -1, 0)) if n_out > 1 else out


# The 1-D counterpart of bias_map_grid along a/Rs alone (the limit curve of HypotheticalExtremePlanet.ipynb): both fits with their Fisher sigmas at every a/Rs, one DataFrame row per point.
# bounds=A_B_FIT_BOUNDS by default, so A_B_fit < 0 is a valid result; `railed` (fit_railed) flags a fit on a bound of `bounds`, and the fitted values are kept, so the caller decides what to do with them.
# rho_star [g/cm^3]: with it each point gets the period a circular orbit at that a/Rs has around a star of that mean density (kepler_period_days). The period never enters the bias,
# only sigma_bin, so the Fisher sigmas and chi2 then belong to a physically consistent orbit; None keeps cfg.planet.period at every point.
# Columns: a_rs, period_h, sigma_bin; A_B_fit, bias, sigma, chi2, railed (A_B-only fit); A_B_fit_J, bias_J, sigma_J, F_fit, sigma_F, chi2_J, railed_J (joint fit); each model's block only if it is in `models`.
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
    """fit_truth (both models, Fisher sigmas) at every a/Rs of a_rs_grid, points in parallel, as a DataFrame; see the comment above for the columns and the rho_star / bounds conventions."""
    return pd.DataFrame(pmap(_bias_vs_a_rs_point, [(cfg, A_B_true, float(a), rho_star, models, bounds, F_bounds, kw) for a in a_rs_grid], n_jobs))


# The illumination geometry along an a/Rs grid at cfg's Rp/Rs: the area fractions of PlanetParams.fraction_illuminated (Knuth et al. 2017 / Carter et al. 2024) and the stellar angular radius.
def sweep_geometry(cfg, a_rs_grid):
    geo = pd.DataFrame([{"a_rs": a, **{k.replace(" ", "_"): v for k, v in swept_planet(cfg, a_rs=a).fraction_illuminated(extra=True).items()}} for a in a_rs_grid])
    geo["night"] = 1. - geo["fully_illuminated"] - geo["penumbra"]
    geo["angular_radius_deg"] = np.degrees(np.arcsin(1. / geo["a_rs"]))
    return geo


# False-positive heat redistribution. The truth has F = 0, so an observer who finds F > 0 significantly would claim heat transport, i.e. an atmosphere. How significant the spurious F is,
# is the likelihood-ratio statistic dchi2_F = chi2(A_B-only fit) - chi2(joint A_B + F fit), both at their best fit with a free baseline, on the noiseless (Asimov) truth: sqrt(dchi2_F)
# is the median expected one-sided significance of "F > 0" (Cowan et al. 2011), exact for the nonlinear T ∝ F^(1/4) response and the F = 0 boundary (null distribution 1/2 chi2_1).
# NOT F_fit / sigma_F: the Fisher sigma_F is the slope AT F_fit, but the model responds ∝ F^~1.5 near 0, so the quadratic extrapolation to F = 0 overstates the significance (by ~1.4x
# on TOI-2431 b / NRS1), and it ignores the boundary. dchi2_F ∝ n_orbits / sigma_1min^2 exactly (the fits do not depend on the noise scale), see critical_sigma_1min.
# One map cell: (dchi2_F, F_fit); NaN on a rail, as in bias_cell. dchi2_F >= 0 since the A_B-only model is the F = 0 member of the joint one; tiny negative values from the tolerances are clipped.
def false_F_cell(cfg, A_B_true, free_offset=True, rail_tol=RAIL_TOL, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, **kw):
    d = false_F_fit(cfg, A_B_true, free_offset=free_offset, rail_tol=rail_tol, bounds=bounds, F_bounds=F_bounds, **kw)
    return d["dchi2_F"], d["F_fit"]


# Both fits and the statistic as a dict (the row of the false-positive tables): A_B_fit_naive, chi2_naive, A_B_fit, F_fit, chi2_joint, dchi2_F (NaN on a rail), railed and T_night_fit (NaN on a rail), the nightside
# temperature [K] that the fitted (A_B, F) imply (Kreidberg & Loeb eq. 1): what the false positive would be reported as. cfg may be a detector set.
def false_F_fit(cfg, A_B_true, free_offset=True, rail_tol=RAIL_TOL, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, **kw):
    fit = fit_truth(cfg, A_B_true, models=("naive", "naive_redist"), free_offset=free_offset, bounds=bounds, F_bounds=F_bounds, **kw)
    c0 = cfg[0] if is_detector_set(cfg) else cfg
    return _false_F_result(fit["naive"], fit["naive_redist"], c0.planet, c0.emissivity_bol, bounds, F_bounds, rail_tol)


# The dict of false_F_fit from its two fits n (A_B-only) and r (joint), given as fit_truth results (A_B_fit, chi2, and F_fit for r). Shared with the fits= path of the false-positive tables,
# which takes the two fits from the bias tables instead of refitting. planet, emissivity_bol: those of the fit's cfg (its first detector), for the implied nightside temperature.
def _false_F_result(n, r, planet, emissivity_bol, bounds, F_bounds, rail_tol=RAIL_TOL):
    from hyperillumination_utils import surface_temperature
    railed = fit_railed(n, bounds, F_bounds, rail_tol) or fit_railed(r, bounds, F_bounds, rail_tol)
    return dict(A_B_fit_naive=n["A_B_fit"], chi2_naive=n["chi2"], A_B_fit=r["A_B_fit"], F_fit=np.nan if railed else r["F_fit"], chi2_joint=r["chi2"],
                dchi2_F=np.nan if railed else max(n["chi2"] - r["chi2"], 0.), railed=bool(railed),
                T_night_fit=np.nan if railed else float(surface_temperature(np.pi, planet, "naive_redist", A_B=r["A_B_fit"], emissivity_bol=emissivity_bol, F=r["F_fit"])))


def _false_F_cell_at(cfg, A_B_true, a_rs, y_param, y, kw):
    return false_F_cell(cfg, A_B_true, a_rs=a_rs, **{y_param: y}, **kw)


# dchi2_F over a map (like bias_map_grid): returns (dchi2 map, F map), shape (len(y_grid), len(a_rs_grid)). The F map should agree with bias_map_grid(redist=True)'s.
# fits=(chi2_map, chi2_map_J, F_map_J) builds both maps without refitting: chi2_map from bias_map_grid(..., return_chi2=True), chi2_map_J and F_map_J from bias_map_grid(..., redist=True,
# return_chi2=True), on the same cfg, A_B_true, grids, y_param and **kw. Those are exactly the two fits false_F_cell makes per cell, so the result is identical. Only the shapes are checked.
def false_F_map_grid(cfg, A_B_true, a_rs_grid, y_grid, y_param, n_jobs=-1, fits=None, **kw):
    if fits is not None:
        chi2, chi2_J, F_J = (np.asarray(m, dtype=float) for m in fits)
        shape = (len(y_grid), len(a_rs_grid))
        if not chi2.shape == chi2_J.shape == F_J.shape == shape:
            raise ValueError(f"fits: three maps of shape (len(y_grid), len(a_rs_grid)) = {shape} expected, got {chi2.shape}, {chi2_J.shape}, {F_J.shape}")
        railed = ~np.isfinite(chi2) | ~np.isfinite(chi2_J)   # the maps' chi2 is NaN where that fit railed; false_F_cell is NaN where either did
        return np.where(railed, np.nan, np.maximum(chi2 - chi2_J, 0.)), np.where(railed, np.nan, F_J)
    flat = pmap(_false_F_cell_at, [(cfg, A_B_true, ar, y_param, y, kw) for y in y_grid for ar in a_rs_grid], n_jobs)
    out = np.array(flat).reshape(len(y_grid), len(a_rs_grid), 2)
    return out[..., 0], out[..., 1]


# The per-minute noise [ppm/min] at which the spurious F is a z-sigma detection (dchi2_F = z^2), at the n_orbits of the fit: sigma_crit = sigma_1min * sqrt(dchi2_F) / z, where sigma_1min
# is the noise the fit was evaluated with. Exact, like measurability's sigma_1min_crit: dchi2_F ∝ 1 / sigma_1min^2. A planet observed with better noise than sigma_crit gets the false positive.
def critical_sigma_1min(dchi2, sigma_1min, z=2.):
    return sigma_1min * np.sqrt(np.asarray(dchi2, float)) / z


# Emission Spectroscopy Metric (Kempton et al. 2018) formatted for a legend entry; the archive column is NaN for a few planets.
def _fmt_esm(esm):
    return f"{esm:.1f}" if np.isfinite(esm) else "n/a"


# Loose name matching ("K2-141b" == "K2-141 b" == "k2_141_b"), the same normalisation HItargets_utils uses to look a planet up,
# so a PlanetParams.name can be matched against the archive's pl_name whichever spelling it carries.
def _norm_name(name):
    return str(name).lower().replace(" ", "").replace("-", "").replace("_", "")


# A planet list as the per-planet plots (their required `planets`) and tables take it: names in any loose spelling (e.g. HItargets_utils.PAPER_PLANETS), PlanetParams,
# or {name: PlanetParams}. planet_list returns PlanetParams (names looked up with hi_planets_by_name); planet_names only the names, without a lookup, so it also takes
# hypothetical planets that are not in the archive (moved_planet, reference_planet).
def _planet_items(planets):
    if isinstance(planets, (str, PlanetParams)): raise TypeError(f"planets must be a list of planets, not a single one ({planets!r}): wrap it in a list")
    return list(planets.values()) if isinstance(planets, dict) else list(planets)

def planet_list(planets):
    return [hi_planets_by_name(p) if isinstance(p, str) else p for p in _planet_items(planets)]

def planet_names(planets):
    return list(dict.fromkeys(p if isinstance(p, str) else p.name for p in _planet_items(planets)))


# The rows of a per-planet table (planet_bias_table, planet_false_F_table) for `planets`, in that order: what every per-planet plot shows. Planets the table does not have
# (the tables leave out planets without a noise level or a stellar spectrum, with a note) are listed in a note; none at all -> ValueError. The attrs are kept.
def select_planets(tab, planets):
    names, key = planet_names(planets), tab["name"].map(_norm_name).to_numpy()
    rows, missing = [], []
    for nm in names:
        hit = np.flatnonzero(key == _norm_name(nm))
        if len(hit): rows.append(hit[0])
        else: missing.append(nm)
    if missing: print(f"not in the table, so not plotted: {', '.join(missing)}")
    if not rows: raise ValueError(f"none of the planets {names} is in the table (it has {list(tab['name'])})")
    out = tab.iloc[rows].reset_index(drop=True)
    out.attrs = dict(tab.attrs)
    return out


# The archive aggregate table (path AGG) as every function here reads it: with HItargets_utils.CUSTOM and OVERRIDES written into it (apply_custom), so that the curated
# geometry and stellar parameters take precedence over the archive in the sample cuts, the background scatter and the per-planet rows, exactly as in hi_planets_by_name.
# Cached per path: planet_archive_value is called once per planet and the CSV read is the slow part. Columns derived in ExoArchive_customcols.py (ESM, pl_Teq, ...) stay the archive's.
@lru_cache(maxsize=4)
def archive_frame(AGG):
    return apply_custom(pd.read_csv(AGG)).drop_duplicates("pl_name")


# ESM of a single archive planet by name, for a target drawn outside overlay_planets (the notebook's own planet).
# One archive value for a planet by name, NaN where the archive has none or the value is missing: "ESM" for the tick labels / rows, "pl_rade" for the Rp row (CUSTOM / OVERRIDES applied).
def planet_archive_value(AGG, pl_name, column):
    d = archive_frame(AGG)
    row = d[d.pl_name.map(_norm_name) == _norm_name(pl_name)]
    return float(row[column].iloc[0]) if len(row) and pd.notna(row[column].iloc[0]) else np.nan

def planet_esm(AGG, pl_name):
    return planet_archive_value(AGG, pl_name, "ESM")


# Where the notebook's own planet sits on a bias map: x is always a/Rs, the key names the archive column on the y axis.
# Everything is read off PlanetParams (a/Rs = R/r2, Rp/Rs = r1/r2, Teff = T2_eff), so swapping `planet` at the top of the
# notebook moves the marker - no target is hard-coded anywhere.
_TARGET_Y = {"a_over_r2": lambda p: p.R / p.r2,
             "pl_ratror": lambda p: p.r1 / p.r2,
             "st_teff":   lambda p: p.T2_eff}


def target_marker(ax, planet, ykey, AGG=None, **kw):
    """The notebook's planet drawn on a bias map as the green star (TARGET style; kw overrides it), labelled with its own name, a/Rs and (with AGG) ESM."""
    if ykey not in _TARGET_Y: raise ValueError(f"no target coordinate for ykey={ykey!r}; known: {tuple(_TARGET_Y)}")
    x, y = planet.R / planet.r2, _TARGET_Y[ykey](planet)
    esm = f", ESM={_fmt_esm(planet_esm(AGG, planet.name))}" if AGG is not None else ""
    return ax.plot(x, y, label=f"{planet.name}  ($a/R_\\star$={x:.2f}{esm})", **{**TARGET, **kw})


# A synthetic PlanetParams for planet-agnostic bias maps: the geometry every map cell inherits for whatever the map does not sweep.
# Units: r2 = 1 (stellar radii), so r1 = Rp/Rs and R = a/Rs. period and R_star_Rsun are placeholders: the phase curve is binned per orbit,
# so the period drops out of the bias, and the stellar radius only sets a_AU for the (tiny) light-travel-time correction.
# T14 is NaN: it is only needed by PandExo, which never runs on the reference geometry.
# The star keeps PlanetParams' default log g = 4.5 and [Fe/H] = 0 for every T_eff (a hypothetical star; see swept_planet for what that does at M-dwarf T_eff).
def reference_planet(a_rs=2.5, rp_rs=DEFAULT_MAP_RP_RS, T_eff=4000., incl=90., period=0.3, R_star_Rsun=0.6, name="reference geometry"):
    return PlanetParams(name=name, period=period, incl=incl, ecc=0., w=90., r1=rp_rs, r2=1., R=a_rs, T2_eff=T_eff,
                        a_AU=a_rs * R_star_Rsun * float(const.R_sun / const.au), T14=np.nan)


# A real planet moved to another period on a circular orbit around its own host: a/R* = (a/R*)_0 (P / P_0)^(2/3) (Kepler III at the host's density), a_AU likewise. Rp/R*, the inclination
# (so the impact parameter shrinks with a/R*), the star and its log g / [Fe/H] are kept. With period_h = the planet's Roche period (roche_periods) it is the most extreme version of a known
# system: the same planet and star at the Roche limit of that composition. T14 is NaN (only PandExo needs it; give the moved planet its parent's sigma_1min). Raises if the planet would
# touch the star (a/R* <= 1 + Rp/R*: SPIDERMAN aborts there).
def moved_planet(pl, period_h, name=None):
    s = (period_h / 24. / pl.period) ** (2. / 3.)
    if pl.R * s <= pl.r1 + pl.r2: raise ValueError(f"{pl.name} at P = {period_h:.2f} h would touch its star (a/R* = {pl.R * s / pl.r2:.3f})")
    return replace(pl, name=name or f"{pl.name} at {period_h:.2f} h", period=period_h / 24., R=pl.R * s, a_AU=pl.a_AU * s, T14=np.nan)


def map_grid(lo, hi, n, include=None, pad=0.1, bounds=None):
    """Linear grid over [lo, hi], widened (by `pad` of the original span) so that `include` - typically the target's own
    a/Rs or Rp/Rs - falls inside it. A planet outside the default box would otherwise simply be missing from its own map.
    bounds = (min, max): limits the widened grid may not cross, e.g. (TEFF_NODES[0], TEFF_NODES[-1]) for a T_s axis, where a cell off the stellar grid's
    nodes would raise inside the worker pool and lose the whole map."""
    if include is not None and np.isfinite(include):
        span = hi - lo
        lo, hi = min(lo, include - pad * span), max(hi, include + pad * span)
    if bounds is not None: lo, hi = max(lo, bounds[0]), min(hi, bounds[1])
    return np.linspace(lo, hi, n)


# The archive planets that fall inside a bias map (x = a/Rs = `a_over_r2`, y = `ykey`), restricted to ROCKY planets, Rp < `max_rade` Earth radii.
# The cut must be on the ABSOLUTE radius `pl_rade`, not on Rp/Rs (see HItargets_utils.ROCKY_MAX_RADE, the shared default, also used by best_hi_planets).
# Factored out of overlay_planets so the maps and the per-planet bar figures select from exactly one sample definition; `exclude` takes loose spellings.
# Reads the archive through archive_frame, i.e. with CUSTOM / OVERRIDES applied (HItargets_utils.apply_custom).
def rocky_sample(AGG, ykey, ylim, a_rs_lim=(1.5, 4.0), max_rade=ROCKY_MAX_RADE, exclude=None):
    _exo = archive_frame(AGG)   # CUSTOM / OVERRIDES applied: a/Rs, Rp/Rs, T_eff and Rp of the curated planets are the curated values, not the archive's
    drop = [exclude] if isinstance(exclude, str) else list(exclude or [])
    keep = ~_exo.pl_name.map(_norm_name).isin({_norm_name(n) for n in drop})
    _phys = _exo[np.isfinite(_exo.a_over_r2) & np.isfinite(_exo.st_teff) & np.isfinite(_exo.pl_ratror)
                 & (_exo.pl_ratror > 0) & (_exo.pl_ratror < 1)
                 & np.isfinite(_exo.pl_rade) & (_exo.pl_rade < max_rade)   # rocky cut (NaN radius -> dropped, cannot be confirmed rocky)
                 & keep].drop_duplicates("pl_name").copy()
    x = "a_over_r2"
    return _phys[(_phys[x] >= a_rs_lim[0]) & (_phys[x] <= a_rs_lim[1])
                 & (_phys[ykey] >= ylim[0]) & (_phys[ykey] <= ylim[1])]


# Distinct colours for the labelled planets of a map: tab10 up to ten, else the dark then the light tab20 shades.
def _label_colors(n):
    base = plt.cm.tab10(np.arange(10)) if n <= 10 else [plt.cm.tab20(i) for i in (*range(0, 20, 2), *range(1, 20, 2))]
    return [base[k % len(base)] for k in range(n)]


# The n rocky archive planets (Rp < max_rade) with the smallest a/Rs, by archive name, from archive_frame (CUSTOM / OVERRIDES applied), optionally without `exclude` (loose spellings).
def closest_in(AGG, n, max_rade=ROCKY_MAX_RADE, exclude=None):
    d = archive_frame(AGG)
    drop = {_norm_name(x) for x in ([exclude] if isinstance(exclude, str) else list(exclude or []))}
    ok = (np.isfinite(d.a_over_r2) & np.isfinite(d.pl_ratror) & (d.pl_ratror > 0) & (d.a_over_r2 > 1. + d.pl_ratror) & np.isfinite(d.pl_rade) & (d.pl_rade < max_rade)
          & ~d.pl_name.map(_norm_name).isin(drop))
    return list(d[ok].nsmallest(n, "a_over_r2").pl_name)


def overlay_planets(AGG, ax, ykey, ylim, a_rs_lim=(1.5, 4.0), max_rade=ROCKY_MAX_RADE, target=None, exclude=None, mark_target=True, n_closest=10, *, planets):
    """Overplot the archive planets on a bias map (x = a/Rs, y = `ykey`).

    planets     : the planets to label (required; the notebooks pass HItargets_utils.PAPER_PLANETS), any form planet_list takes. Each gets its own colour and a
                  legend entry (name, a/Rs, ESM) at its PlanetParams position (hi_planets_by_name: CUSTOM / OVERRIDES apply, as in the per-planet fits), drawn as a circle.
                  Planets outside the map box are not drawn; their names are returned.
    n_closest   : the n rocky archive planets with the smallest a/Rs (closest_in) are labelled too, as diamonds, where they are not already in `planets`; 0 for none.
    The other rocky_sample planets inside the box are faint background dots.
    target      : the notebook's PlanetParams. It is dropped from the scatter and the labels (matched on its name) and, unless
                  `mark_target=False`, drawn as the green star by target_marker - so the highlighted planet follows
                  whatever `planet` the notebook defines instead of being fixed to one target.
    exclude     : extra name(s) to drop from the background scatter; loose spellings are fine.
    """
    if ykey not in _TARGET_Y: raise ValueError(f"no planet coordinate for ykey={ykey!r}; known: {tuple(_TARGET_Y)}")
    not_target = lambda pl: target is None or _norm_name(pl.name) != _norm_name(target.name)   # the target is the green star instead
    labelled = [pl for pl in planet_list(planets) if not_target(pl)]
    given = {_norm_name(pl.name) for pl in labelled}
    extra = [pl for pl in planet_list(closest_in(AGG, n_closest, max_rade=max_rade, exclude=[target.name] if target is not None else None)) if _norm_name(pl.name) not in given and not_target(pl)] if n_closest else []
    marker = {**{pl.name: "o" for pl in labelled}, **{pl.name: "D" for pl in extra}}
    labelled = labelled + extra
    drop = ([exclude] if isinstance(exclude, str) else list(exclude or [])) + [pl.name for pl in labelled] + ([target.name] if target is not None else [])
    bg = rocky_sample(AGG, ykey, ylim, a_rs_lim=a_rs_lim, max_rade=max_rade, exclude=drop)

    x = "a_over_r2"
    ax.scatter(bg[x], bg[ykey], s=16, facecolor="#40c4d6", edgecolor="w", linewidths=.3,
               alpha=.6, zorder=3, label=f"other known rocky planets ($R_p<{max_rade:g}\\,R_\\oplus$, N={len(bg)})")
    arch = archive_frame(AGG)
    esm = dict(zip(arch.pl_name.map(_norm_name), arch["ESM"]))
    pos = {pl.name: (pl.R / pl.r2, _TARGET_Y[ykey](pl)) for pl in labelled}
    inside = [pl for pl in labelled if a_rs_lim[0] <= pos[pl.name][0] <= a_rs_lim[1] and ylim[0] <= pos[pl.name][1] <= ylim[1]]
    for color, pl in zip(_label_colors(len(inside)), inside):
        ax.scatter(*pos[pl.name], s=60 if marker[pl.name] == "o" else 48, marker=marker[pl.name], color=color, edgecolor="w", linewidths=.7, zorder=5,
                   label=f"{pl.name}  ($a/R_\\star$={pos[pl.name][0]:.2f}, ESM={_fmt_esm(esm.get(_norm_name(pl.name), np.nan))})" + ("" if marker[pl.name] == "o" else f"  [{n_closest} closest-in]"))
    if target is not None and mark_target: target_marker(ax, target, ykey, AGG)
    return [pl.name for pl in labelled if pl not in inside]

###########################################################################################################################
#####
#####   Plotting (bias notebooks)
#####
###########################################################################################################################

# Colours as in Hyperillumination/test_thermal_redist.ipynb: F is ordered -> one-hue (blue) ramp; the models keep fixed colours
F_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
C_REDIST, C_HI, C_NAIVE = "#2a78d6", "#eb6834", "#52514e"   # joint A_B + F fit (blue), hyperillumination truth (orange), A_B-only fit (grey)
INK, INK_2, AXIS, MUTED = "#0b0b0b", "#52514e", "#c3c2b7", "#898781"
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
TARGET = dict(marker="*", ls="none", color="#0cff0c", ms=18, mec="k", zorder=6)   # the notebook's planet (green star), on the maps via target_marker and in the line plots
MOCK_KW = dict(fmt="o", ms=3, color="0.25", ecolor="0.6", elinewidth=.8, capsize=2, zorder=5)   # errorbar style of mock data
# The fitted A_B = 0 reference line on every bias figure (contour on the maps, line in the sweeps and retrieval plots): beyond it the point-source fit needs A_B < 0, i.e. more absorbed
# energy than the star delivers to a point, a value a physical prior (A_B_BOUNDS, as in the dynesty fits) would cut off at 0. A thin dash-dot line, set apart from the solid level
# contours and the dashed truth lines; on the maps it switches to white over dark cells, like the level contours.
FIT_ZERO = dict(color=INK, lw=1.2, ls="-.")


def set_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "font.size": 11, "axes.titlesize": 12,
    })


# Linear F axis, restricted to the F range that matters for the plot
def F_axis(ax, F_max):
    ax.set_xlim(0., F_max)
    ax.set_xlabel("Heat redistribution $F$")


# pcolormesh cell edges halfway between grid points, clipped to [lo, hi]
def cell_edges(x, lo, hi):
    return np.concatenate([[lo], 0.5 * (x[1:] + x[:-1]), [hi]])


# Delta chi^2 landscape of the joint fit (d_chi2 from chi2_landscape minus the best fit) with the 68 % / 95 % contours for two parameters.
# Returns fig, ax so the notebook can mark its fits and set the title.
def plot_chi2_landscape(F_grid, AB_grid, d_chi2, F_max, vmax=30., figsize=(10, 7.5)):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(cell_edges(F_grid, 0., F_max), cell_edges(AB_grid, AB_grid[0], AB_grid[-1]), d_chi2,
                       cmap=SEQ_BLUE, vmin=0., vmax=vmax, shading="flat", rasterized=True)
    cs = ax.contour(F_grid, AB_grid, d_chi2, levels=[2.30, 6.18], colors=INK, linewidths=1.)
    ax.clabel(cs, fmt={2.30: "68 %", 6.18: "95 %"}, fontsize=9)
    ax.grid(False)
    F_axis(ax, F_max)
    ax.set(ylabel="Bond albedo $A_B$", ylim=(AB_grid[0], AB_grid[-1]))
    fig.colorbar(im, ax=ax, label="$\\Delta\\chi^2$ to the joint fit (offset free)", extend="max")
    return fig, ax


# Bias summary of bias_vs_true_AB(..., redist=True) in one row: absolute | relative A_B bias | fitted F.
# The joint fit is solid, the A_B-only fit dashed; i0 indexes the target, drawn as the green star with target_label.
def plot_joint_bias_vs_true_AB(A_B_grid, sweep, i0, target_label, title):
    naive, joint = sweep["naive"], sweep["naive_redist"]
    panels = [(joint["bias_A_B"], naive["bias_A_B"], r"absolute bias  $A_{B,\rm fit}-A_B$", r"Absolute bias in recovered $A_B$"),
              (100. * joint["bias_A_B"] / A_B_grid, 100. * naive["bias_A_B"] / A_B_grid, r"relative bias  $(A_{B,\rm fit}-A_B)/A_B$ [%]", r"Relative bias in recovered $A_B$"),
              (joint["F_fit"], None, r"fitted $F$  (= bias, since $F_{\rm true}$ = 0)", r"Fitted heat redistribution $F$")]
    below = (np.asarray(joint["A_B_fit"]) < 0.).any() or (np.asarray(naive["A_B_fit"]) < 0.).any()   # some fit below A_B = 0: draw that line on the two bias panels
    fit_zero = [-np.asarray(A_B_grid, dtype=float), np.full(len(A_B_grid), -100.), None]   # Delta A_B = -A_B, i.e. -100 %
    fig, axs = plt.subplots(1, 3, figsize=(21, 5.5))
    for a_, (y, y_naive, ylab, ttl), y0 in zip(axs, panels, fit_zero):
        a_.axhline(0, color=AXIS, lw=.8)
        if y_naive is not None:
            a_.plot(A_B_grid, y_naive, "o--", color=C_NAIVE, lw=1.5, ms=5, label="$A_B$ only (naive)")
        a_.plot(A_B_grid, y, "o-", color=C_REDIST, lw=1.5, ms=5, label="$A_B$ and $F$ (naive_redist)")
        a_.plot(A_B_grid[i0], y[i0], label=target_label, **TARGET)
        if below and y0 is not None:
            lim = a_.get_ylim()
            a_.plot(A_B_grid, y0, zorder=1.5, label=r"fitted $A_B$ = 0  ($\Delta A_B = -A_B$)", **FIT_ZERO)
            a_.set_ylim(lim)   # keep the data's range: the line is a reference, not data
        a_.set_xlabel(r"true Bond albedo $A_B$"); a_.set_ylabel(ylab); a_.set_title(ttl); a_.legend()
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig, axs


# One measurability panel: |bias| / sigma^Fisher on a log axis for each (y, style, label) curve, with the 1-sigma and 0.3-sigma reference lines.
# The right axis is the same curve scaled by cfg.sigma_1min: the per-minute precision at which the bias is a 1-sigma effect (sigma^Fisher is strictly proportional to sigma_1min).
def measurability_panel(ax, A_B_grid, curves, i0, cfg, target_label, ylabel, title):
    for y, fmt, color, label in curves:
        ax.plot(A_B_grid, y, fmt, color=color, lw=1.5, ms=5, label=label)
    ax.plot(A_B_grid[i0], curves[-1][0][i0], label=target_label, **TARGET)
    ax.axhline(1., color=INK, lw=.9, ls="--", label=r"bias $=1\,\sigma$  ($\sigma_{1\rm min}$ = $\sigma_{1\rm min, crit}$)")
    ax.axhline(.3, color=INK, lw=.8, ls=":", label=r"bias $=0.3\,\sigma$  (negligible: $+4\%$ on the total error)")
    ax.set_yscale("log")
    sec = ax.secondary_yaxis("right", functions=(lambda v: cfg.sigma_1min * v, lambda v: v / cfg.sigma_1min))   # linear rescale -> a shift on the log axis
    sec.set_ylabel(r"required $\sigma_{1\rm min, crit}$ [ppm/min]")
    ax.spines["right"].set_visible(True)
    ax.legend(fontsize=9, frameon=True, facecolor="#fcfcfb", edgecolor="none", framealpha=0.9)   # the frame keeps the reference lines from running through the text
    ax.set_xlabel(r"true Bond albedo $A_B$"); ax.set_ylabel(ylabel); ax.set_title(title)


# Measurability of the joint-fit biases, side by side: |Delta A_B| / sigma_A_B (joint solid, A_B-only dashed) and F_fit / sigma_F (the truth has F = 0, so F_fit is the bias).
# Both sigmas are marginal Fisher widths with the baseline marginalised; sigma_A_B of the joint fit already includes the widening from the F degeneracy.
def plot_joint_measurability(A_B_grid, sweep, i0, cfg, target_label, title):
    naive, joint = sweep["naive"], sweep["naive_redist"]
    bs_naive, _ = measurability(naive["bias_A_B"], naive["sigma_A_B"], cfg.sigma_1min)
    bs_joint, _ = measurability(joint["bias_A_B"], joint["sigma_A_B"], cfg.sigma_1min)
    bs_F, _ = measurability(joint["F_fit"], joint["sigma_F"], cfg.sigma_1min)

    fig, axs = plt.subplots(1, 2, figsize=(16, 5.5))
    measurability_panel(axs[0], A_B_grid, [(bs_naive, "o--", C_NAIVE, "$A_B$ only (naive)"), (bs_joint, "o-", C_REDIST, "$A_B$ and $F$ (naive_redist)")],
                        i0, cfg, target_label, r"$|\Delta A_B|\,/\,\sigma_{A_B}^{\rm Fisher}$", r"Measurability of the $A_B$ bias")
    measurability_panel(axs[1], A_B_grid, [(bs_F, "o-", C_REDIST, "$A_B$ and $F$ (naive_redist)")],
                        i0, cfg, target_label, r"$F_{\rm fit}\,/\,\sigma_F^{\rm Fisher}$", r"Measurability of the spurious $F$")
    fig.suptitle(title + r"  ($n_{\rm orbits}$=%d, $\sigma_{1\rm min}$=%.0f ppm/min)" % (cfg.n_orbits, cfg.sigma_1min), y=1.02)
    fig.tight_layout()
    return fig, axs


# The hyperillumination truth next to the best point-source fits of fit_truth (A_B-only and, if present, joint; baselines refitted) with one mock noise realisation at sigma_bin, and the
# residuals truth - model before (naive at the true A_B) and after absorbing the bias, against the same noise. fit: a fit_truth result for cfg at A_B_true; kw: its model_pcurve keywords.
# Times in hours from transit. Returns fig, ax so the notebook can set the suptitle.
def plot_fit_vs_truth(cfg, A_B_true, fit, seed=43, title=None, **kw):
    t_, truth = model_pcurve(cfg, "hyperillum", A_B_true, **kw)
    naive_0 = model_pcurve(cfg, "naive", A_B_true, **kw)[1]
    fits = [("naive", "best $A_B$-only fit", "--", C_NAIVE), ("naive_redist", "best joint fit", "-.", C_REDIST)]
    curves = []
    for model, label, ls, color in fits:
        if model not in fit: continue
        f = fit[model]
        y = model_pcurve(cfg, model, f["A_B_fit"], F=f.get("F_fit"), **kw)[1]
        tag = f"$A_B$ = {f['A_B_fit']:.3f}" + (f", $F$ = {f['F_fit']:.4f}" if "F_fit" in f else "")
        curves.append((y + (truth - y).mean(), f"{label} ({tag})", ls, color))   # the fitted baseline (free_offset)
    sigma_bin, P = cfg.sigma_bin, cfg.planet.period
    tb = np.linspace(0., P, cfg.n_bins, endpoint=False) + 0.5 * P / cfg.n_bins   # bin centres of binned()
    yb_truth = binned(cfg, truth)
    y_data = yb_truth + np.random.default_rng(seed).normal(0., sigma_bin, size=yb_truth.size)   # one draw ~ N(truth, sigma_bin) per bin

    fig, ax = plt.subplots(1, 2, figsize=(16, 5.2))
    ax[0].errorbar(tb * 24., y_data, yerr=sigma_bin, label=rf"mock data (truth $\pm\,{sigma_bin:.0f}$ ppm)", **MOCK_KW)
    ax[0].plot(t_ * 24., truth, lw=1.2, color=C_HI, label="hyperillumination (truth)")
    for y, label, ls, color in curves: ax[0].plot(t_ * 24., y, lw=1.2, ls=ls, color=color, label=label)
    ax[0].set_xlabel("time [h]"); ax[0].set_ylabel(r"$F_p/F_s$ [ppm]"); ax[0].set_title("truth and best fits")
    ax[1].axhline(0., color=AXIS, lw=.8)
    ax[1].errorbar(tb * 24., y_data - yb_truth, yerr=sigma_bin, label="mock data $-$ truth (the noise realisation)", **MOCK_KW)
    ax[1].plot(t_ * 24., truth - naive_0, lw=1.2, color=C_HI, label=rf"truth $-$ naive at the true $A_B$ = {A_B_true:.2f}")
    for y, label, ls, color in curves: ax[1].plot(t_ * 24., truth - y, lw=1.2, color=color, label=f"truth $-$ {label.split(' (')[0]} (irreducible)")
    ax[1].set_xlabel("time [h]"); ax[1].set_ylabel("residual [ppm]"); ax[1].set_title("residual before and after absorbing the bias")
    for a_ in ax: a_.legend(fontsize=9)
    fig.suptitle(title or rf"{cfg.planet.name}: $a/R_\star$ = {cfg.planet.R / cfg.planet.r2:.2f}, {cfg.n_orbits} orbits, {cfg.n_bins} bins, $\sigma_{{\rm bin}}$ = {sigma_bin:.0f} ppm", y=1.02)
    fig.tight_layout()
    return fig, ax


# Vertical marks of the limit curve, on one axes: the contact limit (hatched), the Roche limits roche_a_rs ({label: a/R*}, e.g. kepler_a_rs(P_R / 24, rho_star) of roche_periods for the
# sweep's host), the anchor planets (grey dashed). Returns the legend handles (one per Roche limit, one per anchor, the contact span).
def _mark_limits(ax, contact, roche_a_rs, anchors):
    handles = [ax.axvspan(ax.get_xlim()[0], contact, facecolor="0.85", edgecolor="0.5", hatch="///", lw=0, zorder=0, label=r"inside the star ($a/R_\star < 1 + R_p/R_\star$)")]
    for col, (lab, a_R) in zip(_ROCHE_COLORS, roche_a_rs.items()):
        handles.append(ax.axvline(a_R, color=col, lw=1.2, label=rf"Roche limit, {lab}  ($a/R_\star$ = {a_R:.2f})"))
    for nm, pl in anchors.items():
        handles.append(ax.axvline(pl.R / pl.r2, color=MUTED, lw=1., ls="--", label=rf"{nm}  ($a/R_\star$ = {pl.R / pl.r2:.2f})"))
    return handles


# The limit curve: a bias_vs_a_rs sweep (both models) on a log a/Rs axis, three stacked panels - the geometry of sweep_geometry (area fractions, stellar angular radius on the right axis),
# the biases (Delta A_B of both fits, the spurious F on the right axis, hollow markers where a fit railed, the fitted A_B = 0 line where a fit goes below it, and sweep_ubol0's
# A_B-only bias as the limb-darkening sensitivity, if given)
# and the measurability (|Delta A_B| / sigma of both fits, sqrt(dchi2_F) of the spurious F, sqrt(chi2) of the best A_B-only fit; all four scale as 1 / sigma_1min and as sqrt(n_orbits), so
# another noise level is a vertical shift of every curve on the log axis). _mark_limits on every panel; roche_a_rs {label: a/R*} are the Roche lines, anchors {name: PlanetParams} the planets marked.
# cfg: the sweep's config (sigma_1min, n_orbits, n_bins for the labels and the right axis); rp_rs: the sweep's Rp/Rs (the contact limit). titles: (geometry, bias, measurability) panel titles.
def plot_limit_curve(sweep, geo, cfg, rp_rs, roche_a_rs, anchors, sweep_ubol0=None, u_bol=None, titles=(None, None, None), suptitle=None):
    contact = 1. + rp_rs
    bad, badJ = sweep["railed"], sweep["railed_J"]
    fig, axs = plt.subplots(3, 1, figsize=(13, 15), sharex=True)
    for a_ in axs: a_.set_xscale("log"); a_.set_xlim(sweep["a_rs"].iloc[0], sweep["a_rs"].iloc[-1])
    limit_handles = [_mark_limits(a_, contact, roche_a_rs, anchors) for a_ in axs][0]

    a0 = axs[0]
    h = [a0.plot(geo["a_rs"], geo["fully_illuminated"], color=C_HI, lw=1.5, label="fully illuminated fraction")[0],
         a0.plot(geo["a_rs"], geo["penumbra"], color=C_REDIST, lw=1.5, label=r"penumbra fraction $R_\star/a$")[0],
         a0.plot(geo["a_rs"], geo["night"], color=C_NAIVE, lw=1.5, label="nightside fraction")[0]]
    a0.set_ylabel("area fraction"); a0.set_ylim(0., 1.)
    a0r = a0.twinx()
    h.append(a0r.plot(geo["a_rs"], geo["angular_radius_deg"], color=INK, lw=1., ls="-.", label="stellar angular radius [deg]")[0])
    a0r.set_ylabel("stellar angular radius [deg]"); a0r.set_ylim(0., 90.); a0r.grid(False)
    a0.set_title(titles[0] or rf"Geometry at $R_p/R_\star$ = {rp_rs:g}")
    a0.legend(handles=h, fontsize=9, loc="center right")

    a1 = axs[1]
    a1.axhline(0., color=AXIS, lw=.8)
    h = [a1.plot(sweep["a_rs"], sweep["bias"], "o-", color=C_NAIVE, lw=1.5, ms=5, label=r"$\Delta A_B$, $A_B$-only fit")[0]]
    if bad.any() or badJ.any():
        h.append(a1.plot(sweep["a_rs"][bad], sweep["bias"][bad], "o", color=C_NAIVE, mfc="#fcfcfb", ms=7, label="railed fit ($A_B$ on a bound of its fit range)")[0])
    A_B_true = float((sweep["A_B_fit"] - sweep["bias"]).iloc[0])
    if (sweep[["bias", "bias_J"]].min().min() < -A_B_true):   # some fit below A_B = 0: the reference line where it crosses
        h.append(a1.axhline(-A_B_true, zorder=1.5, label=rf"fitted $A_B$ = 0  ($\Delta A_B = -A_B$ = {-A_B_true:g})", **FIT_ZERO))
    if sweep_ubol0 is not None:
        h.append(a1.plot(sweep_ubol0["a_rs"], sweep_ubol0["bias"], ":", color=C_NAIVE, lw=1.4, label=r"$\Delta A_B$, $A_B$-only fit, uniform stellar disc ($u_{\rm bol}$ = 0)")[0])
    h.append(a1.plot(sweep["a_rs"], sweep["bias_J"], "s-", color=C_REDIST, lw=1.5, ms=5, label=r"$\Delta A_B$, joint $A_B$ + $F$ fit")[0])
    a1.plot(sweep["a_rs"][badJ], sweep["bias_J"][badJ], "s", color=C_REDIST, mfc="#fcfcfb", ms=7)
    a1.set_ylabel(r"bias $\Delta A_B = A_{B,\rm fit} - A_B$")
    a1r = a1.twinx()
    h.append(a1r.plot(sweep["a_rs"], sweep["F_fit"], "^--", color=F_COLORS[-1], lw=1.2, ms=5, label="spurious $F$ of the joint fit, right axis (truth: $F$ = 0)")[0])
    a1r.set_ylabel("fitted $F$"); a1r.grid(False)
    a1.set_title(titles[1] or "Bias from neglecting hyperillumination" + (rf"  ($u_{{\rm bol}}$ = {u_bol:g})" if u_bol is not None else ""))
    a1.legend(handles=h, fontsize=9, loc="lower right")

    a2 = axs[2]
    bs_n, _ = measurability(sweep["bias"], sweep["sigma"], cfg.sigma_1min)
    bs_j, _ = measurability(sweep["bias_J"], sweep["sigma_J"], cfg.sigma_1min)
    h = [a2.plot(sweep["a_rs"], bs_n, "o-", color=C_NAIVE, lw=1.5, ms=5, label=r"$|\Delta A_B| / \sigma_{A_B}$, $A_B$-only fit")[0],
         a2.plot(sweep["a_rs"], bs_j, "s-", color=C_REDIST, lw=1.5, ms=5, label=r"$|\Delta A_B| / \sigma_{A_B}$, joint fit")[0],
         a2.plot(sweep["a_rs"], np.sqrt(np.clip(sweep["chi2"] - sweep["chi2_J"], 0., None)), "^--", color=F_COLORS[-1], lw=1.2, ms=5, label=r"$\sqrt{\Delta\chi^2_F}$: significance of the spurious $F$")[0],
         a2.plot(sweep["a_rs"], np.sqrt(sweep["chi2"]), "d:", color=C_HI, lw=1.4, ms=5, label=r"$\sqrt{\chi^2}$: misfit of the best $A_B$-only fit")[0],
         a2.axhline(1., color=INK, lw=.9, ls="--", label=r"1 $\sigma$")]
    a2.set_yscale("log"); a2.set_ylabel(r"significance [$\sigma$]")
    a2.set_title(titles[2] or rf"Measurability  ($\sigma_{{1\rm min}}$ = {cfg.sigma_1min:.0f} ppm/min, {cfg.n_orbits} orbits, {cfg.n_bins} bins)")
    a2.legend(handles=h, fontsize=9, loc="upper right", frameon=True, facecolor="#fcfcfb", edgecolor="none")

    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
    x0, x1 = sweep["a_rs"].iloc[0], sweep["a_rs"].iloc[-1]
    ticks = [t for t in (1., 1.2, 1.5, 2., 3., 4., 5., 7., 10., 15., 20.) if x0 <= t <= x1]
    for a_ in axs:
        a_.set_xlim(x0, x1)   # axvline / axvspan widen the autoscaled limits otherwise
        a_.xaxis.set_major_locator(FixedLocator(ticks)); a_.xaxis.set_major_formatter(FormatStrFormatter("%g")); a_.xaxis.set_minor_formatter(NullFormatter())   # plain numbers, not 2 x 10^0
    axs[-1].set_xlabel(r"$a/R_\star$")
    fig.suptitle(suptitle or "The limit curve: bias from neglecting hyperillumination towards the contact limit", y=0.995)
    fig.legend(handles=limit_handles, loc="lower center", bbox_to_anchor=(0.5, 0.), ncol=3, fontsize=9)
    fig.tight_layout(rect=(0., 0.06, 1., 0.985))   # room for the limit legend below the panels and the suptitle above
    return fig, axs


# The printed summary of a limit curve: the rail boundaries, the largest non-railed biases, the u_bol sensitivity (if sweep_ubol0 is given) and the sweep read off at each Roche limit of roche_a_rs.
def print_limit_curve(sweep, A_B_true, roche_a_rs, sweep_ubol0=None, u_bol=None):
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
    for lab, a_R in roche_a_rs.items():   # linear interpolation of the sweep at each Roche limit (NaN outside the sweep range)
        at = lambda col: np.interp(a_R, sweep["a_rs"], sweep[col], left=np.nan, right=np.nan)
        print(f"at the Roche limit, {lab} (a/Rs = {a_R:.2f}): Delta A_B = {at('bias'):+.3f} (A_B-only), {at('bias_J'):+.3f} (joint), F = {at('F_fit'):.4f}, misfit {np.sqrt(at('chi2')):.2f} sigma")


# Stacked bias maps on a shared a/R* axis: |Delta A_B| (top), the relative bias |Delta A_B|/A_B (middle) and, if F_map is given, the fitted F of the joint fit (bottom).
# show_absolute / show_relative / show_F switch the three panels on; by default only the absolute bias is drawn (show_absolute=False, show_relative=True shows
# the relative map alone). show_F needs F_map (passing F_map alone still
# masks the non-occulting cells and draws the grazing lines on every panel, it just does not add the F panel).
# The dash-dot contour, labelled on the line, is the fitted A_B = 0 (Delta A_B = -A_B_true, i.e. |Delta A_B| = A_B_true and 100 % on the relative panel): towards small a/R* the fits go below 0
# (A_B_FIT_BOUNDS), where an observer's physical prior would rail them at 0. Drawn on every panel, also on the F panel (same joint fit), and only where the map crosses it.
# Grey cells = "the model breaks down": railed fits (NaN; with the default fit range they do not occur). With F_map, geometries without an occultation are greyed out as well:
# once the impact parameter b = (a/R*) cos i reaches 1 + Rp/R*, nothing anchors the flux level and A_B, F become degenerate (the best fit is arbitrary along a flat chi^2 valley).
# The dashed lines then bound the grazing zone 1 - Rp/R* < b < 1 + Rp/R* (drawn only where it reaches into the map; at incl = 90 every cell occults and there is none).
# rp_rs_of(y) is Rp/R* at the map's y values; incl [deg] is the inclination the map was computed at (cfg.planet.incl).
# The maps are planet-agnostic: target=None highlights no planet; a PlanetParams there is drawn as the green star (and dropped from the archive scatter). title overrides the suptitle.
# planets (required): the labelled archive planets (overlay_planets), e.g. HItargets_utils.PAPER_PLANETS, drawn as circles; the n_closest rocky archive planets with the smallest a/R*
# are labelled as diamonds where not already among them (n_closest=0: none). Those outside the map box are listed in a note.
# roche: the Roche frontier (draw_roche_frontier) on every panel of an (a/R*, T_eff) map - True: roche_periods() (HItargets_utils.ROCHE_CASES at ROCHE_R_P), a {label: P_R [h]} dict:
# those, False: none. Only a T_eff axis fixes the host density (main sequence), so other maps never show it.
def plot_bias_maps(xg, yg, bias_map, A_B_true, AGG, ykey, ylabel, rp_rs_of, incl=90., F_map=None, F_levels=(0.01, 0.02, 0.05),
                   show_absolute=True, show_relative=False, show_F=False, target=None, title=None, roche=True, legend_fontsize=12, n_closest=10, *, planets):
    if show_F and F_map is None: raise ValueError("show_F=True needs F_map")
    if not (show_absolute or show_relative or show_F): raise ValueError("nothing to draw: set show_absolute, show_relative or show_F")
    cos_i = np.cos(np.radians(incl))
    X, Y = np.meshgrid(xg, yg)
    no_occultation = (X * cos_i >= 1. + rp_rs_of(Y)) if F_map is not None else np.zeros(X.shape, bool)
    y_line = np.linspace(yg[0], yg[-1], 100)
    grazing = F_map is not None and xg[-1] * cos_i > 1. - np.max(rp_rs_of(y_line))   # b = 1 - Rp/R* reached inside the map

    panels = []
    if show_absolute:
        panels.append((np.abs(bias_map), "magma_r", [0.02, 0.05, 0.10], "%g", r"$|\Delta A_B|$", r"Absolute bias in recovered $A_B$"))
    if show_relative:
        panels.append((100. * np.abs(bias_map) / A_B_true, "magma_r", [5, 10, 20, 40], "%g %%", r"$|\Delta A_B|/A_B$ [%]", r"Relative bias in recovered $A_B$"))
    if F_map is not None:
        panels = [p[:5] + (p[5] + " (joint fit)",) for p in panels]
        if show_F:
            panels.append((F_map, SEQ_BLUE, list(F_levels), "%g", r"fitted $F$", r"Fitted heat redistribution $F$ (truth: $F$ = 0)"))

    A_fit = np.ma.masked_where(no_occultation | ~np.isfinite(bias_map), A_B_true + bias_map)   # the fitted A_B itself, for the A_B,fit = 0 contour
    fit_zero = bool((A_fit < 0.).any() and (A_fit > 0.).any())   # a contour needs both sides
    fig, axs = plt.subplots(len(panels), 1, figsize=(12, 6.2 * len(panels)), sharex=True, squeeze=False)
    axs = axs.ravel()
    for ax, (z, cmap, levels, fmt, clabel, panel_title) in zip(axs, panels):
        z = np.ma.masked_where(no_occultation | ~np.isfinite(z), z)
        im = ax.pcolormesh(xg, yg, z, cmap=cmap, shading="auto")
        cs = ax.contour(xg, yg, z, levels=levels, colors=["w" if im.norm(lv) > 0.5 else INK for lv in levels], linewidths=.8)   # contour ink that stays readable on light and dark cells
        ax.clabel(cs, fmt=fmt)
        if fit_zero:   # ink as for the level contours: white where the panel is dark along the line (the cells closest to A_B,fit = 0)
            near = np.abs(A_fit) <= np.percentile(np.abs(A_fit.compressed()), 10)
            ink = "w" if im.norm(np.ma.median(z[near & ~np.ma.getmaskarray(z)])) > 0.5 else INK
            cz = ax.contour(xg, yg, A_fit, levels=[0.], colors=[ink], linewidths=FIT_ZERO["lw"], linestyles=FIT_ZERO["ls"])
            ax.clabel(cz, fmt={0.: r"$A_{B,\rm fit} = 0$"})   # labelled on the line like the level contours, not in the legend
        if grazing:
            for sign in (-1., 1.):
                ax.plot((1. + sign * rp_rs_of(y_line)) / cos_i, y_line, color=INK, ls="--", lw=1.2, label=r"grazing occultation ($b = 1 \mp R_p/R_\star$)" if sign > 0 else None)
        if _roche_cases(roche) and ykey == "st_teff": draw_roche_frontier(ax, xg, yg, _roche_cases(roche), rp_rs_of)
        # `planets` labelled with a/Rs and ESM, the other known ROCKY planets (NASA Exoplanet Archive, Rp < ROCKY_MAX_RADE R_Earth) as dots; a target is drawn as the green star
        outside = overlay_planets(AGG, ax, ykey, (yg[0], yg[-1]), a_rs_lim=(xg[0], xg[-1]), target=target, n_closest=n_closest, planets=planets)
        ax.set_facecolor("0.85")
        ax.grid(False)
        ax.set_xlim(xg[0], xg[-1]); ax.set_ylim(yg[0], yg[-1])
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        fig.colorbar(im, ax=ax, label=clabel)
    axs[-1].set_xlabel(r"$a/R_\star$")
    fig.suptitle(title or f"Biases from neglecting hyperillumination ($A_B$ = {A_B_true:.2f}, $i$ = {incl:g}°)")
    fig.tight_layout()
    fig.legend(*axs[0].get_legend_handles_labels(), loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=legend_fontsize, framealpha=.9)

    if outside: print(f"outside the map box, not drawn: {', '.join(outside)}")
    ok = ~no_occultation & np.isfinite(bias_map)
    line = (f"{'occulting, ' if F_map is not None else ''}non-railed cells: {ok.sum()} of {ok.size};  Delta A_B in [{bias_map[ok].min():+.3f}, {bias_map[ok].max():+.3f}]"
            f" = [{100 * bias_map[ok].min() / A_B_true:+.1f}, {100 * bias_map[ok].max() / A_B_true:+.1f}] % of A_B")
    if F_map is not None:
        line += f",  F in [{F_map[ok].min():.4f}, {F_map[ok].max():.4f}]"
    print(line)
    print(f"fitted A_B < 0 in {int((A_B_true + bias_map[ok] < 0.).sum())} of {ok.sum()} cells;  railed: {(~no_occultation & ~np.isfinite(bias_map)).sum()}" + (f",  without occultation: {no_occultation.sum()}" if F_map is not None else ""))
    return fig, axs


# Roche periods [h] of the compositions in `cases` ({label: (iron mass fraction, rho_0 / rho_mean)}, default HItargets_utils.ROCHE_CASES) for a rock/iron planet of radius R_p [R_E]
# (default ROCHE_R_P): mean density from astro_utils.rocky_mean_density (Fortney et al. 2007), P_R from astro_utils.roche_period_h (Rappaport et al. 2013 eq. 5). The labels carry R_p.
# This is what roche=True draws on the maps; in period the limits hold for any host.
def roche_periods(cases=None, R_p=ROCHE_R_P):
    cases = ROCHE_CASES if cases is None else cases
    return {f"{lab}, {R_p:g} $R_\\oplus$": float(roche_period_h(rocky_mean_density(R_p, fe), q)) for lab, (fe, q) in cases.items()}

# The Roche frontier on an (a/Rs, T_eff) map, for main-sequence hosts: a_R/Rs(T_eff) = kepler_a_rs(P_R, rho_star(T_eff)) per Roche period of `periods` ({label: P_R [h]}, roche_periods),
# the region left of the innermost curve (shortest P_R) hatched - inside every Roche limit shown, under the main-sequence assumption - and the contact limit a/Rs = 1 + Rp/Rs (rp_rs_of,
# as in plot_bias_maps) as a dashed line where it reaches into the map. Legend entries only for what lies inside the map. The line is a main-sequence reference, not a planet's own limit:
# the host densities of real planets scatter around rho_star(T_eff) (by 0.6-1.6x for PAPER_PLANETS), so a planet's own distance from its limit is roche_proximity(P, P_R), not its
# position relative to the line. plot_bias_maps / plot_false_F_map draw it (roche=...); roche_allowed_masks gives the cell masks the statistics need ({label: bool array of the map's
# shape, True where a/Rs is beyond that limit}).
_ROCHE_COLORS = (C_HI, C_REDIST, F_COLORS[-1])
ROCHE_HATCH_LABEL = r"inside every Roche limit shown (main-sequence host)"

def _roche_cases(roche):
    """The {label: P_R [h]} of a roche= argument: True -> roche_periods(), a dict -> itself, False / None -> None."""
    return roche_periods() if roche is True else (roche or None)

def roche_frontier_a_rs(T_eff, P_R_h):
    """a_R/Rs along a T_eff axis for main-sequence hosts, for a Roche period P_R_h [h]."""
    return kepler_a_rs(P_R_h / 24., rho_star(T_eff))

def draw_roche_frontier(ax, xg, yg, periods, rp_rs_of):
    y_line = np.linspace(yg[0], yg[-1], 200)
    visible = lambda x: bool(np.any((x > xg[0]) & (x < xg[-1])))
    for color, (lab, P_R) in zip(_ROCHE_COLORS, periods.items()):
        x = roche_frontier_a_rs(y_line, P_R)
        ax.plot(x, y_line, color=color, lw=1.8, label=rf"Roche limit, {lab} ($P_{{\rm R}}$ = {P_R:.1g} h)" if visible(x) else None)
    x_min = roche_frontier_a_rs(y_line, min(periods.values()))
    ax.fill_betweenx(y_line, xg[0], np.maximum(x_min, xg[0]), facecolor="none", edgecolor=INK, hatch="///", lw=0, label=ROCHE_HATCH_LABEL if np.any(x_min > xg[0]) else None)
    x_contact = 1. + rp_rs_of(y_line)
    if visible(x_contact): ax.plot(x_contact, y_line, color=INK, lw=1.2, ls="--", label=r"contact ($a/R_\star = 1 + R_p/R_\star$)")

def roche_allowed_masks(xg, yg, periods):
    X, Y = np.meshgrid(xg, yg)
    return {lab: X >= roche_frontier_a_rs(Y, P_R) for lab, P_R in periods.items()}

# The largest |bias| a map reaches inside each allowed region of roche_allowed_masks, one line per Roche limit of `periods` (the same labels as the map legend): the physically
# reachable extreme (for main-sequence hosts) instead of the box's corner.
def print_allowed_extremes(xg, yg, bias_map, A_B_true, allowed, periods):
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


# Map of the critical noise sigma_crit(a/Rs, y) for a z-sigma false-positive F (critical_sigma_1min of a false_F_map_grid result), log colour scale, with the archive planets overlaid
# as in plot_bias_maps. cfg: the map's config (its sigma_1min is the noise the dchi2 map was evaluated with, its n_orbits the number of orbits the critical noise refers to).
# The map holds Rp/R* and i at the reference values: the phase-curve signal is ∝ (Rp/R*)^2, so dchi2_F ∝ (Rp/R*)^4 and sigma_crit ∝ (Rp/R*)^2 to a good approximation; planets with a
# different Rp/R* must be rescaled (or fitted at their own geometry) before their noise level is compared with the map.
# planets (required): the labelled archive planets, n_closest: the closest-in planets labelled in addition, roche: the Roche frontier, all as in plot_bias_maps.
def plot_false_F_map(xg, yg, dchi2_map, cfg, AGG, ykey, ylabel, rp_rs_of, incl=90., z=2., sigma_levels=(10., 20., 50., 100., 200., 500., 1000., 2000., 5000.), target=None, title=None,
                     roche=True, n_closest=10, *, planets):
    from matplotlib.colors import LogNorm
    cos_i = np.cos(np.radians(incl))
    X, Y = np.meshgrid(xg, yg)
    no_occultation = X * cos_i >= 1. + rp_rs_of(Y)
    y_line = np.linspace(yg[0], yg[-1], 100)
    grazing = xg[-1] * cos_i > 1. - np.max(rp_rs_of(y_line))

    sigma_1min, n_orbits = common_sigma_1min(cfg), template_cfg(cfg).n_orbits   # a detector set: the common placeholder noise of its detectors
    sig_crit = critical_sigma_1min(dchi2_map, sigma_1min, z=z)
    ok = ~no_occultation & np.isfinite(sig_crit) & (sig_crit > 0.)
    z_ = np.ma.masked_where(~ok, sig_crit)
    levels = [lv for lv in sigma_levels if z_.min() < lv < z_.max()]

    fig, ax = plt.subplots(figsize=(12, 6.2))
    im = ax.pcolormesh(xg, yg, z_, cmap="magma_r", norm=LogNorm(vmin=z_.min(), vmax=z_.max()), shading="auto")
    if levels:
        cs = ax.contour(xg, yg, z_, levels=levels, colors=["w" if im.norm(lv) > 0.5 else INK for lv in levels], linewidths=.8)
        ax.clabel(cs, fmt="%g")
    if grazing:
        for sign in (-1., 1.):
            ax.plot((1. + sign * rp_rs_of(y_line)) / cos_i, y_line, color=INK, ls="--", lw=1.2, label=r"grazing occultation ($b = 1 \mp R_p/R_\star$)" if sign > 0 else None)
    if _roche_cases(roche) and ykey == "st_teff": draw_roche_frontier(ax, xg, yg, _roche_cases(roche), rp_rs_of)
    outside = overlay_planets(AGG, ax, ykey, (yg[0], yg[-1]), a_rs_lim=(xg[0], xg[-1]), target=target, n_closest=n_closest, planets=planets)
    if outside: print(f"outside the map box, not drawn: {', '.join(outside)}")
    ax.set_facecolor("0.85")
    ax.grid(False)
    ax.set_xlim(xg[0], xg[-1]); ax.set_ylim(yg[0], yg[-1])
    ax.set_xlabel(r"$a/R_\star$"); ax.set_ylabel(ylabel)
    ax.set_title(title or rf"Critical noise for a {z:g}$\sigma$ false-positive $F$ ({n_orbits} orbits)")
    fig.colorbar(im, ax=ax, label=rf"$\sigma_{{1\,\rm min,\,crit}}$ [ppm/min] for a {z:g}$\sigma$ detection of $F > 0$")
    fig.tight_layout()
    fig.legend(*ax.get_legend_handles_labels(), loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, framealpha=.9)

    print(f"occulting, non-railed cells: {ok.sum()} of {ok.size};  sigma_1min,crit ({z:g} sigma, {n_orbits} orbits) in [{sig_crit[ok].min():.1f}, {sig_crit[ok].max():.1f}] ppm/min;  "
          f"sqrt(dchi2_F) at sigma_1min = {sigma_1min:g} ppm/min in [{np.sqrt(dchi2_map[ok]).min():.2f}, {np.sqrt(dchi2_map[ok]).max():.2f}] sigma")
    print(f"railed: {(~no_occultation & ~np.isfinite(dchi2_map)).sum()},  without occultation: {no_occultation.sum()}")
    return sig_crit


# One row of planet_bias_table: the planet's own geometry and its own sigma_1min, everything else inherited from cfg.
# The planets of a per-planet table whose host the stellar model can represent. A host outside the NewEra cube (T_eff, log g or [Fe/H], newera_utils.CUBE_*) or off the stellar grid's
# T_eff nodes would raise inside the worker pool and abort the whole table, so such planets are left out with a note instead. Checked here in the parent process, where building a
# planet's stellar grid takes ~0.02 s. A band-less template cfg (l1 = l2 = nan) has no grid to check, so it keeps every planet.
def drop_unsupported_hosts(cfg, pls):
    keep, dropped = [], []
    for pl in pls:
        try:
            for c in (cfg if is_detector_set(cfg) else [cfg]):   # every detector's grid (a band-less template has none)
                c = replace(c, planet=pl)
                if c.stellar_grid is not None: check_teff_nodes(c.stellar_grid, pl.T2_eff)
            keep.append(pl)
        except ValueError as e:
            dropped.append(f"{pl.name} ({e})")
    if dropped: print(f"no stellar spectrum for {len(dropped)} of {len(pls)} planets, left out of the table:\n  " + "\n  ".join(dropped))
    if not keep: raise ValueError("the stellar model covers none of the planets' hosts, so there is nothing to tabulate")
    return keep


# The noise level of one planet in a per-planet table: a number [ppm/min] for a single band, {detector: ppm/min} for a detector set (what rms_1min_pandexo_sample gives for a joint fit).
# NaN stands for "no noise level": the row is fitted at cfg's own placeholder noise (the fitted values do not depend on it) and gets no Fisher sigma (VERDICT_NO_NOISE).
def _has_noise(s1):
    return isinstance(s1, dict) or (s1 is not None and np.isfinite(s1))


# The entry of `sigma_1min` ({name: level}) for one planet, checked against cfg: a detector set needs a dict with every detector of the set (a planet lacking one counts as without
# a noise level, as instrument_retrievals leaves out such joint fits), a single band a number. A non-finite level raises: it would enter sigma_bin and so the chi^2 of the fit itself.
def _table_noise(sigma_1min, cfg, name):
    s = sigma_1min.get(name, np.nan)
    if is_detector_set(cfg):
        if s is None or (not isinstance(s, dict) and np.isnan(s)): return np.nan
        if not isinstance(s, dict): raise ValueError(f"sigma_1min for {name}: a detector set needs {{detector: ppm/min}} (rms_1min_pandexo_sample(names, instrument) gives it), got {s!r}")
        dets = cfg_detectors(cfg)
        if any(d not in s for d in dets): return np.nan
        if not all(np.isfinite(s[d]) for d in dets): raise ValueError(f"non-finite sigma_1min for {name}: {s}")
        return {d: float(s[d]) for d in dets}
    if isinstance(s, dict): raise ValueError(f"sigma_1min for {name}: a single band needs a number, got the per-detector levels {s} (a joint fit needs cfg to be the detector set)")
    if s is not None and not np.isnan(s) and not np.isfinite(s): raise ValueError(f"non-finite sigma_1min for {name}: {s}")
    return np.nan if s is None else float(s)


# The cfg (or detector set) of one planet of a per-planet table: cfg's instrument-level settings (filter, albedo model, n_bins, n_orbits; the stellar grid is per star and rebuilt
# by __post_init__) with the planet's geometry and its noise level s1 (see _table_noise; without one, cfg's own placeholder noise).
def _planet_cfg(cfg, pl, s1):
    if not is_detector_set(cfg): return replace(cfg, planet=pl, sigma_1min=float(s1) if _has_noise(s1) else cfg.sigma_1min)
    return [replace(c, planet=pl, sigma_1min=s1[_band_of(c)] if _has_noise(s1) else c.sigma_1min) for c in cfg]


# The noise columns of a per-planet table row: sigma_1min (NaN for a detector set, which has one level per detector) and, for a detector set, sigma_1min:<label> per detector (detector_label).
def _noise_columns(cfg, s1):
    if not is_detector_set(cfg): return dict(sigma_1min=float(s1) if _has_noise(s1) else np.nan)
    return dict(sigma_1min=np.nan, **{f"sigma_1min:{detector_label(d)}": (s1[d] if _has_noise(s1) else np.nan) for d in cfg_detectors(cfg)})


def _per_detector_crit(tab, factor):
    """sigma_1min_crit:<label> = sigma_1min:<label> * factor for the per-detector noise columns of a joint fit's table (the bias / sigma ratio, or sqrt(dchi2_F) / z, is common to the set)."""
    for col in [c for c in tab.columns if c.startswith("sigma_1min:")]:
        tab["sigma_1min_crit:" + col.split(":", 1)[1]] = tab[col] * factor
    return tab


def _planet_bias_row(cfg, A_B_true, pl, s1, redist, bounds, F_bounds, kw):
    has_noise = _has_noise(s1)   # no noise level -> fitted at cfg's own sigma_1min, no Fisher sigma (VERDICT_NO_NOISE)
    c = _planet_cfg(cfg, pl, s1)
    model = "naive_redist" if redist else "naive"
    fit = fit_truth(c, A_B_true, models=(model,), free_offset=True, bounds=bounds, F_bounds=F_bounds, fisher=has_noise, **kw)[model]
    return dict(name=pl.name, a_rs=pl.R / pl.r2, period=pl.period, T_eff=pl.T2_eff, bias_A_B=fit["bias_A_B"], sigma_A_B=fit.get("sigma_A_B", np.nan),
                F_fit=fit.get("F_fit", np.nan), sigma_F=fit.get("sigma_F", np.nan), **_noise_columns(cfg, s1), has_noise=has_noise,
                # for planet_false_F_table(fits=...): chi2 at the planet's own noise (none at the placeholder noise), and the fitted A_B itself, since the table's A_B_fit = bias_A_B + A_B_true can differ from it in the last bit
                chi2=fit["chi2"] if has_noise else np.nan, A_B_fit_exact=fit["A_B_fit"])


# Bias, Fisher sigma and critical noise for a SAMPLE of planets on one instrument, sorted by a/Rs. cfg may be a detector set (a joint fit of several detectors, e.g. instrument_cfg(cfg,
# "JWST_NIRSpec.G395H_joint")): then sigma_1min holds {name: {detector: ppm/min}}, the table's sigma_1min / sigma_1min_crit are NaN and the per-detector columns sigma_1min:<label> /
# sigma_1min_crit:<label> carry the levels (noise_cols lists the ones to print); bias_sigma and the verdict are those of the joint fit.
# planets: names (e.g. HItargets_utils.PAPER_PLANETS), PlanetParams or {name: PlanetParams} (planet_list). sigma_1min: {name: ppm/min}, e.g. from HItargets_utils.rms_1min_pandexo_sample.
# Planets with no entry in sigma_1min (e.g. an instrument without a PandExo mode, where CUSTOM_SIGMA_1MIN covers only some planets) are fitted without a noise level, with a note:
# their fitted A_B and bias are exact, the Fisher sigma and everything derived from it NaN, verdict VERDICT_NO_NOISE. A planet whose entry is present but not finite raises:
# sigma_1min enters cfg.sigma_bin and so the chi^2 of the fit itself, and a NaN there makes every cost NaN, i.e. the BIAS comes back wrong, not just the sigma (measured: K2-141 b -0.038 / F = 0 instead of -0.055 / F = 0.016). Hence the check rather than NaN propagation.
# quantity picks which bias the decision columns (bias, sigma, bias_sigma, sigma_1min_crit) refer to: "A_B" (default, also for the joint fit) or "F" (requires redist=True; the truth has F = 0, so F_fit IS the bias).
# A_B_true is the SAME assumed albedo for every planet - a notebook assumption, not a per-planet measurement.
# sigma_1min_crit = sigma_1min * |bias| / sigma is exact rather than an extrapolation: the bias does not depend on the noise, and the Fisher sigma is strictly proportional to sigma_1min.
# verdict (bias_verdict): "needs hyperillum" / "naive OK" / "unconstrained" — the last where sigma >= the flat-prior std of the physical range A_B_BOUNDS (A_B) or F_BOUNDS (F), i.e. where the fit
# does not measure the quantity within its physical range. bounds (default A_B_FIT_BOUNDS) is only the range of the fit, so A_B_fit < 0 is a valid result.
# AGG: archive aggregate path; with it the table carries each planet's ESM (`esm`, NaN where the archive has none) for the figure's tick labels.
def planet_bias_table(cfg, A_B_true, planets, sigma_1min, redist=False, quantity=None, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, AGG=None, n_jobs=-1, **kw):
    quantity = quantity or "A_B"
    if quantity not in ("A_B", "F"): raise ValueError(f'quantity must be "A_B" or "F", got {quantity!r}')
    if quantity == "F" and not redist: raise ValueError('quantity="F" needs redist=True (only the joint fit has an F)')
    pls = planet_list(planets)
    noise = {pl.name: _table_noise(sigma_1min, cfg, pl.name) for pl in pls}   # a number per planet, {detector: ppm/min} for a detector set, NaN without a (complete) noise level
    no_noise = [nm for nm, s in noise.items() if not _has_noise(s)]
    if no_noise: print(f"no sigma_1min for {len(no_noise)} of {len(pls)} planets, fitted without one (the fitted A_B does not depend on it), so no Fisher sigma: {', '.join(no_noise)}")
    c0 = template_cfg(cfg)
    if not np.isfinite(c0.sigma_1min): raise ValueError(f"cfg.sigma_1min = {c0.sigma_1min}: the fits without a noise level run at it, so it must be finite")
    pls = drop_unsupported_hosts(cfg, pls)

    rows = pmap(_planet_bias_row, [(cfg, A_B_true, pl, noise[pl.name], redist, bounds, F_bounds, kw) for pl in pls], n_jobs)
    tab = pd.DataFrame(rows).sort_values("a_rs").reset_index(drop=True)
    tab["esm"] = [planet_esm(AGG, nm) for nm in tab["name"]] if AGG is not None else np.nan
    tab["rp_rearth"] = [planet_archive_value(AGG, nm, "pl_rade") for nm in tab["name"]] if AGG is not None else np.nan   # [R_Earth], for the figure's rows only
    tab["t_obs"] = c0.n_orbits * tab["period"] * 24.   # [h] total observation: n_orbits full orbits, so it differs per planet at fixed n_orbits
    tab["A_B_fit"] = tab["bias_A_B"] + A_B_true   # the fitted value itself, for plot_planet_retrievals

    tab.attrs.update(A_B_true=A_B_true, n_orbits=c0.n_orbits, model="naive_redist" if redist else "naive", bounds=bounds, F_bounds=F_bounds, detectors=cfg_detectors(cfg))
    tab.attrs["quantity"], tab.attrs["sigma_prior"] = quantity, {"A_B": flat_prior_sigma(A_B_BOUNDS), "F": flat_prior_sigma(F_BOUNDS)}   # physical ranges, per quantity, so the figure can re-derive the verdict for either
    tab["bias"], tab["sigma"] = tab[f"bias_{quantity}" if quantity == "A_B" else "F_fit"], tab[f"sigma_{quantity}"]
    tab["bias_sigma"], tab["sigma_1min_crit"] = measurability(tab["bias"], tab["sigma"], tab["sigma_1min"])   # at each planet's own sigma_1min (NaN for a detector set: per detector below)
    _per_detector_crit(tab, tab["bias_sigma"])
    tab["verdict"] = bias_verdict(tab["bias_sigma"], tab["sigma"], tab.attrs["sigma_prior"][quantity])
    tab.loc[~tab["has_noise"], "verdict"] = VERDICT_NO_NOISE
    tab["t_crit"] = critical_time(tab["t_obs"], tab["bias_sigma"], tab["verdict"])   # [h] at which the bias is a 1 sigma effect; NaN where unconstrained or without a noise level
    return tab


# The noise columns of a bias or false-positive table to print: sigma_1min / sigma_1min_crit of a single-band table, the per-detector sigma_1min:<label> / sigma_1min_crit:<label>
# of a joint fit's per-planet table (whose scalar columns are NaN). The per-instrument tables keep the scalar columns (NaN in their joint rows, with the levels in tab.attrs["sigma_1min"]).
def noise_cols(tab):
    per = [c for c in tab.columns if c.startswith("sigma_1min:")]
    if not per: return [c for c in ("sigma_1min", "sigma_1min_crit") if c in tab.columns]
    return per + [c for c in tab.columns if c.startswith("sigma_1min_crit:")]


# The planets of a planet_bias_table whose A_B fit railed (A_B_fit = bias_A_B + A_B_true on a bound of the fit's `bounds`, by default the table's own): the table keeps their fitted values, so this is the list to print next to it.
def railed_names(tab, A_B_true, bounds=None):
    bounds = tab.attrs.get("bounds", A_B_FIT_BOUNDS) if bounds is None else bounds
    return list(tab.loc[is_railed(tab["bias_A_B"] + A_B_true, bounds), "name"])


# A planet_bias_table without its unconstrained planets (verdict VERDICT_UNCONSTRAINED), where a hollow marker with a bar as wide as the prior would only clutter a figure. `like`: drop the planets that are unconstrained in THAT table instead (e.g. the joint fit's, whose sigmas are the larger ones), so two fits keep the same planets.
# The attrs are copied, since pandas does not carry them through every selection.
def drop_unconstrained(tab, like=None):
    keep = (like if like is not None else tab)
    names = set(keep.loc[keep["verdict"] != VERDICT_UNCONSTRAINED, "name"])
    out = tab[tab["name"].isin(names)].reset_index(drop=True)
    out.attrs = dict(tab.attrs)
    return out


# False-positive tables. The decision columns, from dchi2_F at each row's own noise (see false_F_cell): z_F = sqrt(dchi2_F), the expected significance of the spurious F; noise_margin =
# z_F / z = sigma_1min,crit / sigma_1min, the factor by which the noise may be worse (> 1) or must be better (< 1) for a z-sigma false positive, which also holds for a joint fit of several
# detectors (all their noise levels scaled alike); sigma_1min_crit (critical_sigma_1min; NaN for a joint fit); t_crit [h] = t_obs * (z / z_F)^2, the observing time at which the spurious F
# reaches z sigma (dchi2_F ∝ t_obs); verdict: FALSE_POSITIVE where z_F >= z. Railed fits (dchi2_F NaN) get no verdict.
FALSE_POSITIVE, NO_FALSE_POSITIVE = "false positive", "below threshold"

def _false_F_decision(tab, z):
    tab["z_F"] = np.sqrt(tab["dchi2_F"])
    tab["noise_margin"] = tab["z_F"] / z
    tab["sigma_1min_crit"] = critical_sigma_1min(tab["dchi2_F"], tab["sigma_1min"], z=z)   # NaN for a joint fit's per-planet rows: per detector below
    _per_detector_crit(tab, tab["noise_margin"])
    tab["t_crit"] = tab["t_obs"] * (z / tab["z_F"]) ** 2
    tab["verdict"] = np.where(~np.isfinite(tab["z_F"]), "-", np.where(tab["z_F"] >= z, FALSE_POSITIVE, NO_FALSE_POSITIVE))
    tab.attrs["z"] = z
    return tab


# fits= of the false-positive tables: (tab, tab_J), the A_B-only and the joint bias table of the same sample (planet_bias_table or instrument_retrievals, with redist=False and redist=True).
# Their two fits per row are exactly those of false_F_fit (same truth, bounds and tolerances), so their chi2 columns give dchi2_F without refitting. The tables must come from this cfg and
# **kw, which is not checked. Checked are the models, A_B_true, bounds, n_orbits and (by the callers) each row's noise level, since chi2 ∝ 1 / sigma_1min^2.
def _bias_table_pair(fits, A_B_true, bounds, F_bounds, n_orbits):
    if len(fits) != 2: raise ValueError("fits: (A_B-only table, joint table) expected")
    want = (A_B_true, tuple(bounds), tuple(F_bounds), n_orbits)
    for tab, model in zip(fits, ("naive", "naive_redist")):
        a = tab.attrs
        if a.get("model") != model: raise ValueError(f"fits: (A_B-only table, joint table) expected, got the models {[t.attrs.get('model') for t in fits]}")
        got = (a["A_B_true"], tuple(a["bounds"]), tuple(a["F_bounds"]), a["n_orbits"])
        if got != want: raise ValueError(f"fits: the {model} table has (A_B_true, bounds, F_bounds, n_orbits) = {got}, this call {want}")
        if "chi2" not in tab: raise ValueError(f"fits: the {model} table has no chi2 column (made before it was stored): re-run it")
    return fits


# The fits (n, r) of false_F_fit for row `key` of a _bias_table_pair, as fit_truth-like dicts, and the two table rows (for the callers' noise check).
# The fitted A_B is A_B_fit_exact where the table has it (planet_bias_table, whose A_B_fit is rebuilt from the bias), else A_B_fit (instrument_retrievals stores the fit itself), so the result is bit-identical to refitting.
def _table_fit_pair(tabs, col, key):
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


# The row of _planet_false_F_row from the fits of a _bias_table_pair (no refit).
def _planet_false_F_row_from(tabs, cfg, pl, s1, bounds, F_bounds):
    n, r, rows = _table_fit_pair(tabs, "name", pl.name)
    noise = _noise_columns(cfg, s1)
    for tab, row in zip(tabs, rows):
        for col, lv in noise.items():   # the scalar level, or the per-detector levels of a detector set (its scalar column is NaN on both sides)
            if np.isfinite(lv) and not _same_noise(row[col], lv): raise ValueError(f"fits: {pl.name} was fitted at {col} = {row[col]} ppm/min in the {tab.attrs['model']} table, this call has {lv}")
    return dict(name=pl.name, a_rs=pl.R / pl.r2, period=pl.period, T_eff=pl.T2_eff, **noise, **_false_F_result(n, r, pl, template_cfg(cfg).emissivity_bol, bounds, F_bounds))


# False-positive F for a SAMPLE of planets on one instrument, each at its own geometry and noise (the counterpart of planet_bias_table; same inputs, same handling of planets without a noise
# level, same per-detector noise columns for a detector set, whose sigma_1min_crit:<label> is each detector's own level times the noise margin). Sorted by a/Rs. z: the significance the decision refers to.
# fits=(tab, tab_J): the planet_bias_table of this sample with redist=False and redist=True (same cfg, A_B_true, sigma_1min, bounds and **kw), whose fits are reused instead of refitting
# (see _bias_table_pair). Same result.
def planet_false_F_table(cfg, A_B_true, planets, sigma_1min, z=2., bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, AGG=None, n_jobs=-1, fits=None, **kw):
    pls = planet_list(planets)
    noise = {pl.name: _table_noise(sigma_1min, cfg, pl.name) for pl in pls}   # as in planet_bias_table
    missing = [nm for nm, s in noise.items() if not _has_noise(s)]
    if len(missing) == len(pls): raise ValueError(f"no planet has a sigma_1min entry (planets: {missing}; sigma_1min has {sorted(sigma_1min)}), so there is nothing to tabulate")
    if missing:
        print(f"no sigma_1min for {len(missing)} of {len(pls)} planets, left out of the table: {', '.join(missing)}")
        pls = [pl for pl in pls if pl.name not in missing]
    c0 = template_cfg(cfg)
    pls = drop_unsupported_hosts(cfg, pls)
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


# The row of _instrument_false_F_row from the fits of a _bias_table_pair (no refit).
def _instrument_false_F_row_from(tabs, cfg, pl, instrument, sigma_1min, bounds, F_bounds):
    dets = detectors(instrument)
    n, r, _ = _table_fit_pair(tabs, "instrument", instrument)
    for tab in tabs:
        if tab.attrs.get("planet") != pl.name: raise ValueError(f"fits: the {tab.attrs['model']} table is of {tab.attrs.get('planet')}, not {pl.name}")
        for d in dets:
            s_tab = tab.attrs["sigma_1min"].get(d, np.nan)
            if not _same_noise(s_tab, sigma_1min[d]): raise ValueError(f"fits: {d} was fitted at sigma_1min = {s_tab} ppm/min in the {tab.attrs['model']} table, this call has {sigma_1min[d]}")
    return dict(instrument=instrument, sigma_1min=float(sigma_1min[dets[0]]) if len(dets) == 1 else np.nan, **_false_F_result(n, r, pl, template_cfg(cfg).emissivity_bol, bounds, F_bounds))


# False-positive F for one planet on several instruments (the counterpart of instrument_retrievals: same `instruments` entries incl. tuples for a joint fit of several detectors, same
# sigma_1min handling). A joint fit's sigma_1min_crit is NaN; its noise_margin applies to all its detectors alike.
# fits=(tab, tab_J): the instrument_retrievals of this planet with redist=False and redist=True (same cfg, A_B_true, noise levels, bounds and **kw), whose fits are reused instead of
# refitting (see _bias_table_pair). Same result.
def instrument_false_F(cfg, planet, instruments, A_B_true, z=2., sigma_1min=None, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, n_jobs=-1, fits=None, **kw):
    from HItargets_utils import rms_1min, NoNoiseLevelError
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
        if missing: print(f"{planet.name}: no noise level on {', '.join(missing)} (no PandExo mode, no CUSTOM_SIGMA_1MIN entry), left out")
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


# Bar chart of the expected significance z_F of the spurious F per row of a false-positive table, with the threshold z as a dashed line; bars at or above it (FALSE_POSITIVE)
# in the hyperillumination colour. Rows under the axis: the fitted F and the nightside temperature it implies, the delivered and critical noise, and the critical observing time.
# plot_false_F_bars: one planet on several instruments (instrument_false_F). plot_planet_false_F_bars: several planets on one instrument (planet_false_F_table), the bars of
# `planets` (required, e.g. HItargets_utils.PAPER_PLANETS) in that order (select_planets), labelled with a/Rs.
def plot_false_F_bars(tab, title=None, figsize=None, ax=None):
    if "name" in tab: raise TypeError("a per-planet table: use plot_planet_false_F_bars(tab, planets=...)")
    return _false_F_bars(tab, title, figsize, ax)

# plot_planet_bias_bars shows one delivered and one critical noise per planet, which a joint fit of several detectors does not have (one level per detector, sigma_1min:<label>);
# the false-positive bars (_false_F_bars) list them per detector instead.
def _one_noise_level(tab):
    if any(c.startswith("sigma_1min:") for c in tab.columns):
        raise ValueError("a joint fit of several detectors has one noise level per detector (sigma_1min:<label> columns), not the single sigma_1min the bars show: print the table (noise_cols) or use plot_planet_retrievals")


def plot_planet_false_F_bars(tab, title=None, figsize=None, ax=None, *, planets):
    return _false_F_bars(select_planets(tab, planets), title, figsize, ax)

def _false_F_bars(tab, title, figsize, ax):
    z = tab.attrs["z"]
    by_planet = "name" in tab
    x = np.arange(len(tab))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (max(6., 1.3 * len(tab) + 2.), 6.5))
    else:
        fig = ax.figure
    colors = [C_HI if v == FALSE_POSITIVE else (MUTED if v == "-" else C_NAIVE) for v in tab["verdict"]]
    bars = ax.bar(x, tab["z_F"].fillna(0.), color=colors, width=.6, zorder=3)
    _bar_values(ax, bars, "{:.2f}", values=tab["z_F"])
    for xi, rl in zip(x, tab["railed"] if "railed" in tab else np.zeros(len(tab), bool)):   # no bar on a rail (dchi2_F undefined): say why
        if rl: ax.annotate("railed:\n$A_B$ fit on\na bound", (xi, 0.), textcoords="offset points", xytext=(0, 4), ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.axhline(z, color=INK, ls="--", lw=1.2)
    ax.set_ylim(0., max(1.2 * np.nanmax(tab["z_F"]), 1.3 * z))
    ax.set_ylabel(r"$\sqrt{\Delta\chi^2_F}$ [$\sigma$]")
    if by_planet:
        ax.set_xticks(x, [f"{_tick_name(nm)}\n$a/R_\\star$ = {ar:.2f}" for nm, ar in zip(tab["name"], tab["a_rs"])], fontsize=8)
    else:
        ax.set_xticks(x, [instrument_label(ins) for ins in tab["instrument"]], fontsize=8)
    ax.set_xlim(-0.6, len(tab) - 0.4)
    ax.grid(axis="x", visible=False)
    def noise_label(col):   # "sigma_1min" / "sigma_1min_crit", or the per-detector column of a joint fit's per-planet table ("sigma_1min:NRS1")
        base, _, det = col.partition(":")
        return (r"$\sigma_{1\,\rm min,\,crit}$" if base.endswith("_crit") else r"$\sigma_{1\,\rm min}$") + (f" {det}" if det else "") + " [ppm/min]"
    rows = [(r"fitted $F$", tab["F_fit"], "{:.4f}"),
            (r"implied $T_{\rm night}$ [K]", tab["T_night_fit"], "{:.0f}"),
            *[(noise_label(c), tab[c], "{:.0f}") for c in noise_cols(tab)],   # the delivered and the critical noise, per detector for a joint fit's per-planet table
            (r"$t_{\rm crit}$ [h]", tab["t_crit"], "{:.0f}")]
    below = _param_rows(ax, x, rows, y0=-0.17, dy=0.055)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_HI, label=f"{FALSE_POSITIVE} ($\\geq$ {z:g}$\\sigma$)"), plt.Rectangle((0, 0), 1, 1, color=C_NAIVE, label=NO_FALSE_POSITIVE),
               plt.Line2D([], [], color=INK, ls="--", lw=1.2, label=f"{z:g}$\\sigma$ threshold")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=3, fontsize=9, title=r"expected significance $\sqrt{\Delta\chi^2_F}$ of the spurious $F$ (Asimov)", title_fontsize=9)
    ax.set_title(title or (f"False-positive heat redistribution ({tab.attrs['n_orbits']} orbits)" if by_planet else f"{tab.attrs['planet']}: false-positive heat redistribution ({tab.attrs['n_orbits']} orbits per instrument)"),
                 fontweight="bold", pad=52)
    fig.tight_layout()
    fig.subplots_adjust(bottom=min(0.6, 0.12 + below * ax.get_position().height))   # room for the tick labels and the parameter rows (axes fraction -> figure fraction)
    return fig, ax


# cfg rebuilt for an instrument entry: a band name gives one FittingConfig with that band (filter, l1, l2; the stellar grid follows, FittingConfig.__post_init__), a joint fit (the
# detector tuple, or a HItargets_utils.JOINT_INSTRUMENTS alias such as "JWST_NIRSpec.G395H_joint") the detector set, one config per detector. sigma_1min: one number for every detector
# or {detector: ppm/min}; None keeps cfg's. Everything else (planet, albedo model, n_bins, n_orbits, limb darkening, ...) is kept, from the first detector if cfg is itself a detector set.
# A band-less FittingConfig (filter_path None, l1 = l2 = nan) is the natural template: FittingConfig(planet=..., ...) and then instrument_cfg(template, instrument).
def instrument_cfg(cfg, instrument, sigma_1min=None):
    from HItargets_utils import filter_loc, filter_lambda
    c0, cfgs = template_cfg(cfg), []
    for d in detectors(instrument):
        l1, l2 = filter_lambda(d)
        s = c0.sigma_1min if sigma_1min is None else (sigma_1min[d] if isinstance(sigma_1min, dict) else sigma_1min)
        cfgs.append(replace(c0, filter_path=filter_loc(d), l1=l1, l2=l2, sigma_1min=float(s)))
    return cfgs[0] if len(cfgs) == 1 else cfgs


# Several detectors fitted jointly (a detector set, see is_detector_set) are one "instrument" of instrument_retrievals: given as a tuple/list of FILTER_LAMBDA_UM keys or as a
# JOINT_INSTRUMENTS alias (resolve_instrument), carried in the tables as one key joined with "+", e.g. "JWST_NIRSpec.G395H_NRS1+JWST_NIRSpec.G395H_NRS2". detectors() turns any of
# the three spellings into the list of bands (a single instrument is its own one-element set) and rejects an unknown band.
JOINT_SEP = "+"

def instrument_key(instrument):
    return JOINT_SEP.join(detectors(instrument))

def detectors(instrument):
    if isinstance(instrument, str) and JOINT_SEP in instrument: instrument = tuple(instrument.split(JOINT_SEP))
    r = resolve_instrument(instrument)
    return [r] if isinstance(r, str) else list(r)


# The cfg (or detector set) of one table row: the planet's geometry with each detector's band and noise level ({detector: ppm/min}).
def _detector_cfgs(cfg, planet, instrument, sigma_1min):
    return instrument_cfg(replace(template_cfg(cfg), planet=planet), instrument, sigma_1min)


# The FittingConfig (one band) or detector set (a tuple of bands, fitted jointly) of one planet in one retrieval mode at the planet's own noise: what instrument_retrievals fits per
# row, for code that needs the cfg itself (per-bin residuals, sweeps, nuisance columns). sigma_1min: {detector: ppm/min}; None looks the levels up (HItargets_utils.noise_levels:
# PandExo cache or CUSTOM_SIGMA_1MIN) and raises NoNoiseLevelError where a detector has none.
def retrieval_cfg(cfg, planet, instrument, sigma_1min=None):
    from HItargets_utils import noise_levels, NoNoiseLevelError
    dets = detectors(instrument)
    if sigma_1min is None: sigma_1min = noise_levels(planet, dets)
    missing = [d for d in dets if d not in sigma_1min]
    if missing: raise NoNoiseLevelError(f"no noise level for {planet.name} on {', '.join(missing)}")
    return _detector_cfgs(cfg, planet, instrument, sigma_1min)


def _instrument_row(cfg, A_B_true, pl, instrument, sigma_1min, redist, bounds, F_bounds, kw):
    dets = detectors(instrument)
    has_noise = all(d in sigma_1min for d in dets)   # False only for a single detector (joint fits without every noise level are left out): fitted at cfg's own sigma_1min, no Fisher sigma
    c = _detector_cfgs(cfg, pl, instrument, sigma_1min) if has_noise else replace(instrument_cfg(cfg, instrument), planet=pl)
    model = "naive_redist" if redist else "naive"
    fit = fit_truth(c, A_B_true, models=(model,), free_offset=True, bounds=bounds, F_bounds=F_bounds, fisher=has_noise, **kw)[model]
    return dict(instrument=instrument, sigma_1min=float(sigma_1min[dets[0]]) if len(dets) == 1 and has_noise else np.nan,   # a joint fit has one noise level per detector: see tab.attrs["sigma_1min"]
                A_B_fit=fit["A_B_fit"], bias_A_B=fit["bias_A_B"], sigma_A_B=fit.get("sigma_A_B", np.nan), F_fit=fit.get("F_fit", np.nan), sigma_F=fit.get("sigma_F", np.nan),
                chi2=fit["chi2"] if has_noise else np.nan)   # chi2 at the detectors' own noise (instrument_false_F(fits=...)); none at the placeholder noise


# One planet, several instruments: the fitted A_B and its Fisher sigma per instrument, for a hyperillumination truth injected at A_B_true (the multi-instrument
# counterpart of planet_bias_table, plotted by plot_instrument_retrievals). cfg supplies everything but the instrument (albedo model, n_bins, n_orbits, limb darkening).
# An entry of `instruments` is a FILTER_LAMBDA_UM key, or a tuple of keys for a JOINT fit of those detectors (one shared A_B / F, a separate baseline per detector; its table
# row carries the "+"-joined key and NaN as sigma_1min).
# sigma_1min: {detector: ppm/min}; by default HItargets_utils.rms_1min per detector (PandExo, or CUSTOM_SIGMA_1MIN where PandExo has no mode). These come from the PandExo cache
# where possible (HItargets_utils.PANDEXO_CACHE); the returned table's attrs["sigma_1min"] can be passed back in to refit with the same levels. A single detector without a noise level is fitted without one (verdict VERDICT_NO_NOISE: exact fitted
# value, no Fisher sigma); a joint fit whose detectors do not all have one is left out (it weights its detectors by their noise). Both with a note.
# Returns a DataFrame in the order of `instruments`, with the verdict of each fit (bias_verdict on A_B, at the flat-prior std of the physical range A_B_BOUNDS).
# bounds (default A_B_FIT_BOUNDS) is the range of the fit only, so A_B_fit < 0 is a valid result.
def instrument_retrievals(cfg, planet, instruments, A_B_true, redist=False, sigma_1min=None, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, n_jobs=-1, **kw):
    from HItargets_utils import rms_1min, NoNoiseLevelError
    c0 = template_cfg(cfg)   # a detector set as cfg: its instrument-level settings (each entry of `instruments` brings its own band(s))
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
    if no_noise: print(f"{planet.name}: no noise level on {', '.join(no_noise)} (no PandExo mode, no CUSTOM_SIGMA_1MIN entry): fitted without one (the fitted A_B does not depend on it), so no Fisher sigma")
    if joint_out: print(f"{planet.name}: joint fit(s) {', '.join(joint_out)} left out: a joint fit weights its detectors by their noise levels, and not all of them have one")
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
    else:   # a row fitted without a noise level (VERDICT_NO_NOISE): the depth of the fit, no Fisher sigma
        c_det = replace(instrument_cfg(cfg, det), planet=planet)
        depth_fit, sigma_depth = model_depth(c_det, model, float(r["A_B_fit"]), F=F, n_layers=n_layers, **kw), np.nan
    depth_true = model_depth(c_det, "hyperillum", A_B_true, n_layers=n_layers, **kw)
    return dict(instrument=det, fit=r["instrument"], sigma_1min=float(sigma_1min.get(det, np.nan)), depth_true=depth_true, depth_fit=depth_fit, sigma_depth=sigma_depth)


# The occultation depth of each fit in an instrument_retrievals table, next to the depth of the hyperillumination truth: does the misspecified fit still reproduce the eclipse?
# Uses the fitted parameters already in `tab` (no refit), so cfg, planet and kw must be those that made it. Both depths are exact (model_depth, at n_layers), sigma_depth is the
# propagated Fisher sigma (fisher_depth). Columns: instrument (the band the depth is in), fit (the table row the parameters come from), the fit's parameter columns, depth_true,
# depth_fit, bias_depth = depth_fit - depth_true [ppm], sigma_depth and depth_bias_sigma = |bias_depth| / sigma_depth.
# Joint fits of several detectors are left out by default (the depth is a per-band quantity, so they have no single depth); joint=True gives one row per detector of such a fit,
# each with that band's depth and the joint covariance, which plot_instrument_retrievals then draws as an extra marker at the detectors' positions.
def instrument_depths(cfg, planet, tab, n_layers=DEPTH_N_LAYERS, joint=False, n_jobs=-1, **kw):
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


# How well a bias map stands in for a planet's own-geometry fit: the A_B bias read off a map at each planet's position (linear interpolation) next to the
# planet_bias_table value, which fits the planet at its full geometry. The maps hold the inclination (and the parameter not on the y axis) at the reference
# values, so planets with a large impact parameter b come out differently. map_excess = |bias read off the map| - |own-geometry bias|: positive = the map overstates.
# y_param: "T_eff" for an (a/Rs, T_eff) map, "rp_rs" for an (a/Rs, Rp/Rs) map. planets: {name: PlanetParams} as passed to planet_bias_table.
def map_vs_planet_bias(tab, planets, a_rs_grid, y_grid, bias_map, y_param="T_eff"):
    interp = RegularGridInterpolator((y_grid, a_rs_grid), bias_map, bounds_error=False)   # NaN outside the map box
    pls = [planets[nm] for nm in tab["name"]]
    y = [pl.T2_eff if y_param == "T_eff" else pl.r1 / pl.r2 for pl in pls]
    out = pd.DataFrame(dict(name=tab["name"], b=[pl.R / pl.r2 * np.cos(np.radians(pl.incl)) for pl in pls],
                            bias_own=tab["bias_A_B"], bias_map=interp(np.column_stack([y, tab["a_rs"]]))))
    out["map_excess"] = np.abs(out["bias_map"]) - np.abs(out["bias_own"])
    return out.sort_values("map_excess", ascending=False).reset_index(drop=True)


# The exact value written at the end of each bar; below the bar for negative values, skipped where the value is NaN.
def _tick_name(name, width=18):
    """A planet name wrapped at spaces onto lines of <= width characters, for the tick labels of the per-planet plots (long hypothetical names collide otherwise)."""
    return textwrap.fill(str(name), width, break_on_hyphens=False, break_long_words=False)


def _bar_values(ax, bars, fmt, values=None, fontsize=8):
    values = [b.get_height() for b in bars] if values is None else list(values)   # `values` prints a different number than the bar height, e.g. the SIGNED bias on an |bias| bar
    for b, v in zip(bars, values):
        h = b.get_height()
        if not (np.isfinite(h) and np.isfinite(v)): continue
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, h), textcoords="offset points",
                    xytext=(0, 3 if h >= 0 else -11), ha="center", va="bottom" if h >= 0 else "top", fontsize=fontsize)


# Axis break between two stacked axes showing the same data: hide the facing spines and draw the diagonal break marks on both.
def _break_marks(top, bottom, d=.5):
    top.spines["bottom"].set_visible(False); bottom.spines["top"].set_visible(False)
    top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    c = bottom.spines["left"].get_edgecolor()
    kw = dict(marker=[(-1, -d), (1, d)], markersize=10, linestyle="none", color=c, mec=c, mew=1, clip_on=False)
    top.plot([0, 1], [0, 0], transform=top.transAxes, **kw)
    bottom.plot([0, 1], [1, 1], transform=bottom.transAxes, **kw)


# Per-planet decision figure for planet_bias_table, one bar group per planet. For each quantity in `quantities` (default: the table's own) the panel
#   bias and its Fisher sigma
# and, only with sigma_crit_plot=True, a second one below it:   delivered sigma_1min vs the required sigma_1min_crit
# The x tick labels carry each planet's a/Rs and (with `esm` in the table) ESM in brackets. In a noise panel a planet needs hyperillumination exactly when the DELIVERED bar is at or below the REQUIRED one - that is bias >= 1 sigma.
# The verdict (bias_verdict) is written above every bar group. "unconstrained" planets - Fisher sigma at or above the flat-prior std (dashed line in the bias panel) - get hatched bars in both panels: their bias / sigma and sigma_1min_crit are ratios to a meaningless sigma and must not be read as "naive OK".
# quantities=("F", "A_B") needs a redist=True table (it carries both fits' columns); the decision columns and the verdict are recomputed per quantity, so the table's own `quantity` does not matter.
# log_noise: sigma_1min spans the sample's brightness range, so the noise panels are logarithmic by default. log_bias does the same for the bias panels, where an unconstrained sigma (~1) would otherwise flatten the measurable ones (~0.05). bias_label / sigma_label override the first quantity's legend labels.
# bias_break=(y_lo, y_hi) is the linear alternative to log_bias: each bias panel is split into a lower axis [0, y_lo] and a compact upper one [y_hi, top] (height ratio BIAS_BREAK_RATIO) with an axis break between them.
# Pick y_lo above the flat-prior line so it stays visible; a bar top that falls into the gap (y_lo, y_hi) shows no value annotation.
# hide_unconstrained=True drops every planet that is unconstrained in any of the plotted quantities, together with the flat-prior line and the axis break (bias_break is ignored),
# since only the unconstrained sigmas (~1) need either. It raises a ValueError if no planet is left.
# planets (required, e.g. HItargets_utils.PAPER_PLANETS): the bar groups, in that order (select_planets); planets the table does not have are listed in a note.
_BAR_Q = {"A_B": ("bias_A_B", "sigma_A_B", r"bias $\Delta A_B$", r"$\sigma_{A_B}^{\rm Fisher}$", r"$|\Delta A_B|$ and $\sigma_{A_B}$", "$A_B$"),
          "F":   ("F_fit",    "sigma_F",   r"spurious $F$",      r"$\sigma_F^{\rm Fisher}$",     r"$F_{\rm fit}$ and $\sigma_F$",     "$F$")}
_VERDICT_STYLE = {VERDICT_NEEDS_HI: dict(color=C_HI, weight="bold"), VERDICT_NAIVE_OK: dict(color=C_NAIVE), VERDICT_UNCONSTRAINED: dict(color=MUTED, style="italic")}
_HATCH = "///"   # bars of unconstrained planets
BIAS_BREAK_RATIO = (1, 3)   # upper : lower axis height of a broken bias panel


# Parameter rows under the bottom panel of plot_planet_bias_bars: one text cell per planet per row, aligned with the bar groups (x in data units, y in axes units below the tick labels).
# rows: [(label, values, fmt)], top row first; a non-finite value prints "-". Returns the axes-fraction the rows extend below the axes, so the caller can reserve the space.
def _param_rows(ax, x, rows, y0=-0.10, dy=0.075, fontsize=8):
    for k, (label, vals, fmt) in enumerate(rows):
        y = y0 - k * dy
        ax.text(x[0] - 0.55, y, label, transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=fontsize, color=MUTED)
        for xi, v in zip(x, vals):
            ax.text(xi, y, fmt.format(v) if np.isfinite(v) else "-", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=fontsize)
    return -y0 + len(rows) * dy


def plot_planet_bias_bars(tab, bias_label=None, sigma_label=None, fmt_bias="{:+.2f}", fmt_sigma="{:.2f}", fmt_noise="{:.0f}",
                          figsize=None, log_noise=True, log_bias=False, colors=(None, None), title=None, quantities=None, grid=True, param_rows=True, sigma_crit_plot=False,
                          bias_break=None, hide_unconstrained=True, *, planets):
    tab = select_planets(tab, planets)
    _one_noise_level(tab)
    no_noise = ~np.isfinite(tab["sigma_1min"])   # VERDICT_NO_NOISE rows: a bias but no Fisher sigma to set it against
    if no_noise.any():
        print(f"no noise level, so not in the bar plot (no Fisher sigma): {', '.join(tab.loc[no_noise, 'name'])}")
        if no_noise.all(): raise ValueError("none of the planets has a noise level on this instrument, so there is no Fisher sigma to plot (plot_planet_retrievals shows the fitted values)")
        attrs = tab.attrs; tab = tab[~no_noise].reset_index(drop=True); tab.attrs = attrs
    quantities = tuple(quantities or (tab.attrs.get("quantity", "A_B"),))
    if hide_unconstrained:
        unconstrained = np.zeros(len(tab), dtype=bool)
        for q in quantities:
            _, scol = _BAR_Q[q][:2]
            sigma_prior = tab.attrs["sigma_prior"][q]
            unconstrained |= bias_verdict(measurability(tab[_BAR_Q[q][0]], tab[scol], tab["sigma_1min"])[0], tab[scol], sigma_prior) == VERDICT_UNCONSTRAINED
        if unconstrained.all():
            detail = "; ".join(f"{_BAR_Q[q][1]} >= {tab.attrs['sigma_prior'][q]:.3f} (smallest: {np.nanmin(tab[_BAR_Q[q][1]]):.3g})" for q in quantities)
            raise ValueError(f"hide_unconstrained=True: all {len(tab)} planets have an unconstrained Fisher sigma (at or above the flat-prior std; {detail}), "
                             f"so nothing is left to plot. Lower the noise (more orbits, another instrument) or pass hide_unconstrained=False.")
        attrs = tab.attrs
        tab = tab[~unconstrained].reset_index(drop=True)
        tab.attrs = attrs
        bias_break = None   # only the unconstrained sigmas need the split
    if bias_break is not None and log_bias: raise ValueError("bias_break is for a linear bias axis; drop log_bias")
    c_bias, c_sig = (colors[0] or C_REDIST), (colors[1] or C_NAIVE)

    x, w = np.arange(len(tab)), 0.38
    per_q = 2 if sigma_crit_plot else 1   # panels per quantity
    n = per_q * len(quantities)
    # param_rows: a/Rs, ESM, Rp, P, t_obs, Teff and one t_crit per quantity under the bottom panel (0.26 in each), and the tick label is then the bare name
    rows = [(r"$a/R_\star$", tab["a_rs"], "{:.2f}"), ("ESM", tab["esm"], "{:.1f}"), (r"$R_p$ [$R_\oplus$]", tab["rp_rearth"], "{:.2f}"),
            ("$P$ [h]", tab["period"] * 24., "{:.2f}"), (r"$t_{\rm obs}$ [h]", tab["t_obs"], "{:.1f}"), (r"$T_{\rm eff}$ [K]", tab["T_eff"], "{:.0f}")] if param_rows else []
    fig = plt.figure(figsize=figsize or (13, 3.7 * n + 0.8 + 0.26 * (len(rows) + len(quantities) * param_rows)))
    outer, panels = fig.add_gridspec(n, 1), []   # panels[i]: the axes of panel i, top to bottom (two for a broken bias panel)
    for i in range(n):
        specs = outer[i].subgridspec(2, 1, height_ratios=BIAS_BREAK_RATIO, hspace=.06) if bias_break is not None and i % per_q == 0 else [outer[i]]
        panels.append([fig.add_subplot(sp, sharex=panels[0][0] if panels else None) for sp in specs])
    ax = np.array([a for p in panels for a in p])   # every axes, top to bottom
    for a_ in ax[:-1]: a_.tick_params(labelbottom=False)

    for k, q in enumerate(quantities):
        bcol, scol, blab, slab, ylab, name = _BAR_Q[q]
        if k == 0: blab, slab = bias_label or blab, sigma_label or slab
        bias, sigma = tab[bcol], tab[scol]
        bias_sigma, crit = measurability(bias, sigma, tab["sigma_1min"])   # at each planet's own sigma_1min
        sigma_prior = tab.attrs["sigma_prior"][q]
        verdict = bias_verdict(bias_sigma, sigma, sigma_prior)
        if param_rows: rows.append((rf"$t_{{\rm crit}}$({name}) [h]", critical_time(tab["t_obs"], bias_sigma, verdict), "{:.1f}"))   # "-" where the quantity is unconstrained
        hatch = [_HATCH if v == VERDICT_UNCONSTRAINED else None for v in verdict]
        bp = panels[per_q * k]   # the bias panel's axes: [a0] or [upper, lower]
        a0 = bp[-1]
        pre = f"{name}: " if len(quantities) > 1 else ""   # the panel pairs need telling apart only when there are several

        for a_ in bp:   # every bar on both halves of a broken panel; each value annotation is clipped to the half its bar top lies in
            b1 = a_.bar(x - w / 2, np.abs(bias), w, color=c_bias, hatch=hatch, label=blab)   # |bias|, so the two bars are directly comparable in height; the annotation keeps the sign
            b2 = a_.bar(x + w / 2, sigma, w, color=c_sig, hatch=hatch, label=slab)
            if not hide_unconstrained:
                a_.axhline(sigma_prior, color=MUTED, ls="--", lw=1., label=f"flat-prior std {sigma_prior:.3f}  (Fisher $\\sigma$ above it: {name} unconstrained)")
            _bar_values(a_, b1, fmt_bias, values=bias); _bar_values(a_, b2, fmt_sigma)
        if bias_break is not None:
            y_lo, y_hi = bias_break
            top = max(np.nanmax(np.abs(bias)), np.nanmax(sigma), sigma_prior)
            bp[0].set_ylim(y_hi, y_hi + max(top - y_hi, .1 * y_hi) * 1.35); a0.set_ylim(0, y_lo)   # headroom relative to the short upper range, else its value annotations hit the top
            _break_marks(*bp)
        for xi, v in zip(x, verdict): bp[0].text(xi, 1.01, v, transform=bp[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=8, **_VERDICT_STYLE[v])   # the verdict row sits between the axes and the title (axes-height units, so independent of the y scale and clear of the legend)
        if log_bias: a0.set_yscale("log")
        a0.set_ylabel(ylab)
        hidden = "unconstrained planets hidden" if hide_unconstrained else f"hatched: {name} not measured at all"
        bp[0].set_title(f"{pre}bias vs. Fisher uncertainty  (bias is measurable where the {blab} bar exceeds the {slab} bar; {hidden})", fontsize=11, pad=16)

        a0.legend(fontsize=9, loc="upper left")
        if not sigma_crit_plot: continue

        a1 = panels[2 * k + 1][0]
        b3 = a1.bar(x - w / 2, tab["sigma_1min"], w, color=C_HI, hatch=hatch, label=r"delivered $\sigma_{1\rm min}$")
        b4 = a1.bar(x + w / 2, crit, w, color=MUTED, hatch=hatch, label=r"required $\sigma_{1\rm min, crit}$  (bias $=1\,\sigma$)")
        _bar_values(a1, b3, fmt_noise); _bar_values(a1, b4, fmt_noise)
        if log_noise: a1.set_yscale("log")
        a1.set_ylabel("[ppm / min]")
        a1.set_title(f"{pre}delivered vs. required photometric precision - hyperillumination matters where delivered $\\leq$ required (hatched: required is meaningless)", fontsize=11)
        a1.legend(fontsize=9, loc="upper left")

    for a_ in ax:
        a_.margins(y=.18)   # headroom for the value annotations
        if not grid: a_.grid(False)
    ax[-1].set_xticks(x)
    if len(tab) < 3:   # room for 3 bar groups, centred: a table restricted to one or two planets (e.g. noise levels for only some of them) would otherwise stretch the bars over the whole width
        ax[-1].set_xlim((len(tab) - 1) / 2 - 1.5, (len(tab) - 1) / 2 + 1.5)
    if param_rows:
        ax[-1].set_xticklabels(tab["name"], fontsize=9)
        _param_rows(ax[-1], x, rows)
    else:
        esm = tab["esm"] if "esm" in tab else np.full(len(tab), np.nan)
        ax[-1].set_xticklabels([f"{nm} \n(ars {a:.2f}, ESM {_fmt_esm(e)})" for nm, a, e in zip(tab["name"], tab["a_rs"], esm)], rotation=30, ha="right")
    if title: fig.suptitle(title)
    fig.tight_layout()   # includes the row texts (they are axes artists), so the bottom margin grows with the row count
    return fig, ax


# Instrument colours of plot_instrument_retrievals (the palette of Ramirez et al. 2026, arXiv:2610.22338, Fig. 4), and short tick labels for the FILTER_LAMBDA_UM keys
INSTRUMENT_COLORS = ["#264653", "#2a9d8f", "#e9c46a", "#b5400f", "#6d597a", "#8ab17d", "#e76f51"]
INSTRUMENT_LABELS = {"JWST_NIRSpec.G395H_NRS1": "NIRSpec G395H\nNRS1", "JWST_NIRSpec.G395H_NRS2": "NIRSpec G395H\nNRS2", "MIRI/LRS": "MIRI LRS",
                     "MIRI/F1280W": "MIRI F1280W", "MIRI/F1500W": "MIRI F1500W", "MIRI/F1800W": "MIRI F1800W", "Spitzer/IRAC2": "Spitzer IRAC2"}


# Plotted quantity of plot_instrument_retrievals: (value column, sigma column, truth column, y label, legend label of the dashed truth line, default ylim, truth value, legend title, name).
# Truth column None: one line across the plot at the truth value - attrs["A_B_true"] for A_B, 0 for F (the truth has no redistribution, so the plotted F_fit is the spurious F itself).
# "depth" (tables from instrument_depths): the true depth differs per band, so each instrument gets its own dashed segment. "F" needs the joint fit's tables (redist=True); its hollow
# marker is F's own verdict (sigma_F >= the flat-prior std of F_BOUNDS), not A_B's.
_RETRIEVAL_Q = {"A_B": ("A_B_fit", "sigma_A_B", None, "Bond albedo $A_B$", "injected $A_B$ = {A_B_true:.2f}", (0., 0.4), "A_B_true", "fitted $A_B$", "$A_B$"),
                "F": ("F_fit", "sigma_F", None, "heat redistribution $F$", "truth: $F$ = 0 (no redistribution)", (0., None), 0., "fitted $F$ (spurious: the truth has none)", "$F$"),
                "depth": ("depth_fit", "sigma_depth", "depth_true", r"Occultation depth $\delta$ [ppm]", "true (hyperillumination) occultation depth", None, None, "occultation depth of the fit", "$A_B$")}


# Tick label of an instrument key; a joint key ("A+B") joins the detector labels, on one head line where they share it ("NIRSpec G395H\nNRS1 + NRS2").
# short=True drops that shared head line ("NRS1 + NRS2", for legends); oneline=True puts the label on one line.
def instrument_label(instrument, oneline=False, short=False):
    labels = [INSTRUMENT_LABELS.get(d, d) for d in detectors(instrument)]
    heads = {l.split("\n")[0] for l in labels}
    if len(labels) > 1 and len(heads) == 1 and all("\n" in l for l in labels):
        label = ("" if short else heads.pop() + "\n") + " + ".join(l.split("\n", 1)[1] for l in labels)
    else:
        label = " + ".join(labels)
    return label.replace("\n", " ") if oneline else label


# Marker groups of the plot: each table is one group, except that the rows of a depth table whose parameters come from a joint fit of several detectors (fit != instrument,
# see instrument_depths) form their own group per joint key - the same parameters evaluated in each detector's band, so they sit at the detector positions with another marker.
def _retrieval_groups(tabs):
    groups = []
    for label, t in tabs.items():
        joint = (t["fit"] != t["instrument"]) if "fit" in t else np.zeros(len(t), bool)
        groups.append((label if not joint.any() or label is not None else "one detector per fit", t[~joint]))
        if joint.any():
            for key, part in t[joint].groupby("fit", sort=False):
                groups.append(((f"{label}, " if label is not None else "") + f"{instrument_label(key, oneline=True, short=True)} jointly", part))
    return groups


# Retrieved A_B per instrument for one planet, after Ramirez et al. 2026 (arXiv:2610.22338), Fig. 4: one colour per instrument, the marker at the fitted A_B,
# dark / light error bars at sigma_levels x the Fisher sigma (default 1 and 2 sigma), the dashed line at the injected A_B_true.
# tabs: one instrument_retrievals table, or {label: table} to set several fits side by side per instrument, one marker shape each (e.g. {"A_B only": ..., "joint A_B + F": ...},
# like the paper's phase curve / transit pairs). An unconstrained fit (A_B's Fisher sigma >= the flat-prior std of the physical range A_B_BOUNDS) gets a hollow marker: its error bar is not a measurement.
# A joint fit of several detectors is its own x position in the A_B plot (its "+"-joined key); in the depth plot (instrument_depths(joint=True)) its rows sit at the detectors'
# positions as an extra marker group (_retrieval_groups), since the depth is a per-band quantity.
# quantity="depth" plots the fitted occultation depth with its propagated Fisher sigma instead (tables from instrument_depths), with a dashed segment at each instrument's true depth.
# ylim defaults to [lower, 0.4] for A_B (the fits sit below the injected A_B, since hyperillumination biases them low; error bars beyond the range are clipped by the axes), with lower = 0 unless a fitted A_B is below 0.1:
# then it reaches 0.1 below the lowest fit, rounded down to a tenth. Below 0 a light band and the A_B = 0 line (FIT_ZERO) mark the values outside the physical range.
# The depth autoscales.
# plot_planet_retrievals is the same plot for planet_bias_table tables: one position per PLANET of `planets` (required, e.g. HItargets_utils.PAPER_PLANETS), in that order, on one
# instrument (the tick label then carries the planet's a/Rs); every planet of the list is shown, unconstrained ones as hollow markers. The fits side by side per position work the same way.
def plot_instrument_retrievals(tabs, quantity="A_B", sigma_levels=(1, 2), markers=("o", "s", "D", "^"), ylim="default", title=None, figsize=None, ax=None):
    tabs = tabs if isinstance(tabs, dict) else {None: tabs}
    if any("instrument" not in t for t in tabs.values()): raise TypeError("a per-planet table: use plot_planet_retrievals(tabs, planets=...)")
    return _plot_retrievals(tabs, quantity, sigma_levels, markers, ylim, title, figsize, ax, "instrument")

def plot_planet_retrievals(tabs, quantity="A_B", sigma_levels=(1, 2), markers=("o", "s", "D", "^"), ylim="default", title=None, figsize=None, ax=None, *, planets):
    tabs = tabs if isinstance(tabs, dict) else {None: tabs}
    return _plot_retrievals({k: select_planets(t, planets) for k, t in tabs.items()}, quantity, sigma_levels, markers, ylim, title, figsize, ax, "name")

# The default y range of an A_B retrieval plot over several tables: (lower, top), lower = 0 unless a fitted A_B is below 0.1, then 0.1 below the lowest fit rounded down to a tenth.
# Pass it as ylim= to every panel of a figure whose retrieval panels share the y axis (sharey=True), so each panel draws the A_B < 0 band of the common range.
def retrieval_ylim(tabs, top=None):
    lo = np.nanmin([np.nanmin(t["A_B_fit"]) for t in tabs])
    return (min(0., np.floor(10. * lo) / 10. - 0.1), _RETRIEVAL_Q["A_B"][5][1] if top is None else top)

def _plot_retrievals(tabs, quantity, sigma_levels, markers, ylim, title, figsize, ax, xcol):
    first = next(iter(tabs.values()))
    A_B_true = first.attrs["A_B_true"]
    col, sig, truth_col, ylabel, ref_label, ylim_q, ref, legend_title, qname = _RETRIEVAL_Q[quantity]
    ref = first.attrs[ref] if isinstance(ref, str) else ref   # the dashed truth line (truth_col None)
    sp = first.attrs.get("sigma_prior")   # F's flat-prior std: planet_bias_table stores {quantity: std}, instrument_retrievals a single A_B value (then F_bounds gives it)
    sigma_prior_F = sp["F"] if isinstance(sp, dict) else flat_prior_sigma(F_BOUNDS)
    groups = _retrieval_groups(tabs)
    if ylim == "default":
        ylim = ylim_q
        if quantity == "A_B":   # down to the lowest fitted A_B: the fits are not railed at the physical floor (A_B_FIT_BOUNDS), so it can lie below 0
            ylim = retrieval_ylim([g for _, g in groups], ylim_q[1])
    instruments = list(dict.fromkeys(ins for _, g in groups for ins in g[xcol]))   # x positions, in order of first appearance
    color = {ins: INSTRUMENT_COLORS[k % len(INSTRUMENT_COLORS)] for k, ins in enumerate(instruments)}
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (max(5.5, 1.5 * len(instruments) + 1.5), 5.5))
    else:
        fig = ax.figure

    for ins in instruments:   # the groups present at this position, centred on it
        present = [k for k, (_, g) in enumerate(groups) if (g[xcol] == ins).any()]
        step = min(0.18, 0.7 / len(present))   # marker spacing within one position
        for j, k in enumerate(present):
            x, c = instruments.index(ins) + (j - (len(present) - 1) / 2) * step, color[ins]
            r = groups[k][1][groups[k][1][xcol] == ins].iloc[0]
            if r.get("verdict") == VERDICT_NO_NOISE:   # no noise level: the fitted value alone, half-filled, no error bar
                ax.plot(x, r[col], markers[k % len(markers)], ms=8, color=c, fillstyle="left", markerfacecoloralt="#fcfcfb", mew=1.6, zorder=4)
                continue
            if quantity == "A_B" and bool(is_railed(r[col], first.attrs.get("bounds", A_B_FIT_BOUNDS))):   # A_B on a bound: no albedo reproduces the truth, a Fisher sigma there means nothing
                ax.plot(x, r[col], markers[k % len(markers)], ms=8, color=c, mfc="#fcfcfb", mew=1.6, zorder=4)
                ax.annotate("railed", (x, r[col]), textcoords="offset points", xytext=(0, 8), ha="center", va="bottom", fontsize=7, color=MUTED, rotation=90)
                continue
            for lev, alpha in zip(sorted(sigma_levels, reverse=True), np.linspace(.3, 1., len(sigma_levels))):   # widest bar first (lightest), the 1 sigma bar on top
                ax.errorbar(x, r[col], yerr=lev * r[sig], fmt="none", ecolor=c, alpha=alpha, elinewidth=1.8, capsize=3.5, capthick=1.5, zorder=3)
            hollow = (r["verdict"] == VERDICT_UNCONSTRAINED) if quantity != "F" else (not np.isfinite(r[sig]) or r[sig] >= sigma_prior_F)
            ax.plot(x, r[col], markers[k % len(markers)], ms=8, color=c, mfc="#fcfcfb" if hollow else c, mew=1.6, zorder=4)

    if truth_col is None:
        ax.axhline(ref, color="#333333", ls="--", lw=1.8, zorder=2)
    else:   # the truth of an instrument does not depend on the fit, so take it from the first group that has the instrument
        for ins in instruments:
            y_true = next(g.loc[g[xcol] == ins, truth_col].iloc[0] for _, g in groups if (g[xcol] == ins).any())
            ax.hlines(y_true, instruments.index(ins) - 0.4, instruments.index(ins) + 0.4, color="#333333", ls="--", lw=1.8, zorder=2)
    ax.set_xticks(range(len(instruments)))
    if xcol == "instrument":
        ax.set_xticklabels([instrument_label(ins) for ins in instruments], fontsize=9)
    else:   # one planet per position: its name and a/Rs (from the first table that has it)
        a_rs = {ins: next(g.loc[g[xcol] == ins, "a_rs"].iloc[0] for _, g in groups if (g[xcol] == ins).any()) for ins in instruments} if "a_rs" in first else {}
        ax.set_xticklabels([f"{_tick_name(ins)}\n$a/R_\\star$ = {a_rs[ins]:.2f}" if ins in a_rs else _tick_name(ins) for ins in instruments], fontsize=8)
    ax.set_xlim(-0.6, len(instruments) - 0.4)
    if ylim is not None: ax.set_ylim(*ylim)
    below_zero = quantity == "A_B" and ax.get_ylim()[0] < 0.
    if below_zero:   # outside the physical range: a light band below the A_B = 0 line
        ax.axhspan(ax.get_ylim()[0], 0., color="#ecebe6", lw=0, zorder=0)
        ax.axhline(0., zorder=2.5, **FIT_ZERO)
    ax.set_ylabel(ylabel)
    ax.grid(False)   # the paper's look: full frame, inward ticks on all sides, minor ticks on y
    for s in ("top", "right"): ax.spines[s].set_visible(True)
    for s in ax.spines.values(): s.set_color(INK); s.set_linewidth(1.)
    ax.minorticks_on(); ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.tick_params(which="both", direction="in", top=True, right=True, colors=INK)
    ax.set_title(title or (f"{first.attrs['planet']}  ({first.attrs['n_orbits']} orbits per instrument)" if "planet" in first.attrs else f"{first.attrs['n_orbits']} orbits"), fontweight="bold")

    levels = " / ".join(f"{lev}$\\sigma$" for lev in sorted(sigma_levels))
    handles = [plt.Line2D([], [], color="#333333", ls="--", lw=1.8, label=ref_label.format(A_B_true=A_B_true))]
    if below_zero: handles.append(plt.Line2D([], [], label=r"$A_B$ = 0 (below: outside the physical range)", **FIT_ZERO))
    for k, (label, _) in enumerate(groups):
        if label is not None:
            handles.append(plt.Line2D([], [], marker=markers[k % len(markers)], ls="none", ms=7, color="#666666", label=label))
    noise = lambda g: g["verdict"] != VERDICT_NO_NOISE if "verdict" in g else np.ones(len(g), bool)
    unconstrained = any((noise(g) & (~np.isfinite(g[sig]) | (g[sig] >= sigma_prior_F))).any() for _, g in groups) if quantity == "F" else any((g["verdict"] == VERDICT_UNCONSTRAINED).any() for _, g in groups)
    if unconstrained:
        handles.append(plt.Line2D([], [], marker="o", ls="none", ms=7, color="#666666", mfc="#fcfcfb", mew=1.4, label=("" if quantity == "A_B" else qname + " ") + "unconstrained (Fisher $\\sigma \\geq$ prior std)"))
    if any((~noise(g)).any() for _, g in groups):
        handles.append(plt.Line2D([], [], marker="o", ls="none", ms=7, color="#666666", fillstyle="left", markerfacecoloralt="#fcfcfb", mew=1.4, label="no noise level: fitted value, no error bar"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=min(len(handles), 3), fontsize=9, handletextpad=0.4, columnspacing=1.2,
              title=legend_title + f", Fisher error bars {levels} (dark to light)", title_fontsize=9)
    fig.tight_layout()
    return fig, ax



###########################################################################################################################
#####
#####   Nightside temperature: tables and plots (NightsideTemperature.ipynb)
#####
###########################################################################################################################

# Presentation helpers of NightsideTemperature.ipynb; the rows come from its band_row (nightside signal, T_b,night, T_b,day) and fit_row (point-source fit with redistribution, T_night^KL).


def display_formatted(df, columns, index="name", transpose=False):
    '''Display the columns of df given as {column: (header, format)} under their headers, NaN as "-"; transpose: quantities as rows (for a wide table that would wrap).'''
    from IPython.display import display
    cell = lambda v, f: "-" if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)) else f.format(v)
    t = pd.DataFrame({h: [cell(v, f) for v in df[c]] for c, (h, f) in columns.items()}, index=pd.Index(list(df[index]), name=None))
    display(t.T if transpose else t)


def band_range_um(instrument):
    '''(blue, red) edge [um] of a band; a joint fit of several detectors spans all of them.'''
    from HItargets_utils import filter_lambda
    lam = 1e6 * np.array([filter_lambda(d) for d in detectors(instrument)], dtype=float)
    return lam.min(), lam.max()


def planet_label(pl, newline=True):
    sep = "\n" if newline else ", "
    return f"{pl.name}{sep}$a/R_\\star$ = {pl.R / pl.r2:.2f}{sep}$i$ = {pl.incl:.0f}°"


def plot_nightside_survey(survey, pls, z_detect=2., noise_label=r"$\sigma_{\rm amp}$"):
    '''All planets in one band (x: planet, ordered as pls). Top: thermal nightside signal at mid-transit, edge-on, against the amplitude noise sigma_amp (FittingConfig.sigma_amp). Bottom: true edge-on T_b,night against the
    uniform nightside T_night^KL of the point-source fit with redistribution, annotated with the offset and z_F (filled: spurious F detected, z_F >= z_detect; railed fits are left out).'''
    d = survey.set_index("name").loc[list(pls)]
    x, det = np.arange(len(d)), (d["z_F"] >= z_detect).to_numpy()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw=dict(height_ratios=(1, 1.4)))
    ax.plot(x, d["night_hi_90"], "o", ms=8, color=C_HI, zorder=4, label="extended source (point source: 0)")
    ax.plot(x, d["sigma_amp"], "_", ms=16, mew=2, color=MUTED, zorder=3, label=noise_label)
    ax.set_ylim(bottom=0.); ax.set_ylabel(r"thermal $F_p/F_s$ at mid-transit [ppm]"); ax.legend(fontsize=9, loc="upper right")
    ax2.vlines(x, d["T_night_90"], d["T_night_fit"], color=MUTED, lw=1.2, zorder=1)
    ax2.plot(x, d["T_night_90"], "o", ms=8, color=C_HI, zorder=4)
    ax2.plot(x[det], d["T_night_fit"][det], "s", ms=8, color=C_REDIST, zorder=4)
    ax2.plot(x[~det], d["T_night_fit"][~det], "s", ms=8, mfc="none", mec=C_REDIST, mew=1.4, zorder=4)
    for xi, t_true, t_fit, z in zip(x, d["T_night_90"], d["T_night_fit"], d["z_F"]):
        if np.isfinite(t_fit): ax2.annotate(f"{t_fit - t_true:+.0f} K" + (f"\n$z_F$ = {z:.1f}" if np.isfinite(z) else ""), (xi, t_fit), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax2.set_ylabel("nightside temperature [K]"); ax2.set_ylim(bottom=0.)
    ax2.legend(handles=[plt.Line2D([], [], marker="o", ls="none", ms=8, color=C_HI, label=r"true $T_{\rm b,night}$ (edge-on)"),
                        plt.Line2D([], [], marker="s", ls="none", ms=8, color=C_REDIST, label=rf"$T_{{\rm night}}^{{\rm KL}}$ of the point-source fit, $F$ detected ($z_F \geq {z_detect:g}$)"),
                        plt.Line2D([], [], marker="s", ls="none", ms=8, mfc="none", mec=C_REDIST, mew=1.4, label=rf"$T_{{\rm night}}^{{\rm KL}}$, $z_F < {z_detect:g}$")], fontsize=9, loc="lower left")
    ax2.set_xticks(x, [planet_label(pls[nm]) for nm in d.index], fontsize=8); ax2.set_xlim(-0.5, len(d) - 0.2)
    for a_ in (ax, ax2): a_.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def plot_nightside_spectra(night, fits, pls, names, z_detect=2., noise_label=r"$\sigma_{\rm amp}$"):
    '''One column per planet, against wavelength (horizontal bars: band ranges). Top: thermal nightside signal at mid-transit, edge-on and at the planet's inclination, the point-source value
    at the planet's inclination (the dayside seen past the pole; 0 for an edge-on planet) and the amplitude noise sigma_amp. Bottom: T_b,night (edge-on, planet's inclination) and T_night^KL of the point-source fit with
    redistribution per band (filled: z_F >= z_detect); a joint fit of several detectors is a line across their bands, a railed fit a grey "railed" mark.'''
    fig, axs = plt.subplots(2, len(names), figsize=(6.2 * len(names), 8.5), sharex=True, sharey="row", squeeze=False)
    for j, nm in enumerate(names):
        ax, ax2 = axs[:, j]
        d = night[night["name"] == nm]
        lo, hi = np.array([band_range_um(ins) for ins in d["instrument"]]).T
        xc, hw = (lo + hi) / 2., (hi - lo) / 2.
        ax.errorbar(xc, d["night_hi_90"], xerr=hw, fmt="o", ms=7, color=C_HI, label="extended source, edge-on")
        ax.errorbar(xc, d["night_hi"], xerr=hw, fmt="o", ms=7, mfc="none", mew=1.3, color=C_HI, label="extended source, planet's $i$")
        ax.errorbar(xc, d["night_ps"], xerr=hw, fmt="o", ms=7, mfc="none", mew=1.3, color=C_NAIVE, label="point source, planet's $i$ (dayside past the pole)")
        ax.plot(xc, d["sigma_amp"], "_", ms=16, mew=2, color=MUTED, label=noise_label)
        ax.set_title(planet_label(pls[nm], newline=False), fontsize=11)
        ax2.errorbar(xc, d["T_night_90"], xerr=hw, fmt="o", ms=7, color=C_HI)
        ax2.errorbar(xc, d["T_night"], xerr=hw, fmt="o", ms=7, mfc="none", mew=1.3, color=C_HI)
        for _, r in fits[fits["name"] == nm].iterrows():
            b0, b1 = band_range_um(r["instrument"])
            if not np.isfinite(r["T_night_fit"]):
                ax2.annotate("railed", ((b0 + b1) / 2., 0.), xycoords=("data", "axes fraction"), xytext=(0, 6), textcoords="offset points", ha="center", rotation=90, fontsize=8, color=MUTED, style="italic")
                continue
            joint, filled = len(detectors(r["instrument"])) > 1, r["z_F"] >= z_detect
            ax2.hlines(r["T_night_fit"], b0, b1, colors=C_REDIST, lw=1.4, ls="--" if joint else "-", zorder=2)
            ax2.plot(np.sqrt(b0 * b1), r["T_night_fit"], "D" if joint else "s", ms=7, color=C_REDIST, mfc=C_REDIST if filled else "none", mew=1.4, zorder=4)
        ax2.set_xscale("log"); ax2.set_xlabel("wavelength [µm]"); ax2.set_ylim(bottom=0.)
        ax2.set_xticks([3, 4, 5, 7, 10, 15, 20], ["3", "4", "5", "7", "10", "15", "20"]); ax2.minorticks_off()
        for a_ in (ax, ax2): a_.grid(axis="x", visible=False)
    axs[0, 0].set_ylim(bottom=0.)
    axs[0, 0].set_ylabel(r"thermal $F_p/F_s$ at mid-transit [ppm]"); axs[1, 0].set_ylabel("nightside temperature [K]")
    axs[0, -1].legend(fontsize=8, loc="upper left")
    axs[1, 0].legend(handles=[plt.Line2D([], [], marker="o", ls="none", ms=7, color=C_HI, label=r"true $T_{\rm b,night}$, edge-on"),
                              plt.Line2D([], [], marker="o", ls="none", ms=7, mfc="none", mew=1.3, color=C_HI, label=r"true $T_{\rm b,night}$, planet's $i$"),
                              plt.Line2D([], [], marker="s", ls="-", ms=7, color=C_REDIST, label=r"$T_{\rm night}^{\rm KL}$, single band"),
                              plt.Line2D([], [], marker="D", ls="--", ms=7, color=C_REDIST, label=r"$T_{\rm night}^{\rm KL}$, joint NRS1 + NRS2"),
                              plt.Line2D([], [], marker="s", ls="none", ms=7, mfc="none", mec=C_REDIST, mew=1.4, label=rf"hollow: $z_F < {z_detect:g}$")], fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def plot_sg_comparison(sg, band_label="MIRI/LRS"):
    '''Sadh & Gavassino (2026) Table 2 comparison, one row per planet: dayside and nightside temperature under each hemisphere definition (columns <key>_day / <key>_night of sg: pub, uniform, area, disk, band and teske, see hemisphere_temperatures), and A_T.'''
    names, y = list(sg["name"]), np.arange(len(sg))[::-1]
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.), sharey=True, gridspec_kw=dict(width_ratios=(1.25, 1.25, 1.)))
    series = [("pub", "*", INK, True, 12, "S&G 2026 Table 2 / eq. 18"),
              ("uniform", "o", INK, False, 9, "S&G recipe (eqs. 15-17, uniform weight), our insolation"),
              ("area", "o", C_REDIST, True, 7, "area mean of the flux"),
              ("disk", "o", C_HI, True, 7, "bolometric, disk-integrated (edge-on)"),
              ("band", "s", INSTRUMENT_COLORS[1], False, 8, f"{band_label} $T_b$ (edge-on)")]
    for ax, side in zip(axs[:2], ("day", "night")):
        for key, m, col, fill, ms, label in series:
            ax.plot(sg[f"{key}_{side}"], y, m, ls="none", ms=ms, color=col, mfc=col if fill else "none", mew=1.3, label=label, zorder=4)
        ax.set_xlabel(f"{side}side temperature [K]")
    axs[0].plot(sg["teske"], y, "D", ls="none", ms=6, color=MUTED, label=r"Teske et al. 2025 $T_{d,\max}$ (point source)", zorder=3)
    for key, m, col, fill, ms, label in series:
        axs[2].plot(sg[f"A_T_{key}"], y, m, ls="none", ms=ms, color=col, mfc=col if fill else "none", mew=1.3, zorder=4)
    axs[2].axvline(1., color=AXIS, lw=.8); axs[2].set_xlabel(r"$A_T = 1 - T_{\rm night}/T_{\rm day}$")
    axs[0].set_yticks(y, [f"{nm}\n$a/R_\\star$ = {r:.2f}" for nm, r in zip(names, sg["a_rs"])])
    for a_ in axs: a_.grid(axis="y", visible=False)
    fig.legend(*axs[0].get_legend_handles_labels(), fontsize=9, ncol=3, loc="lower center")
    fig.suptitle(r"Sadh & Gavassino (2026) Table 2 against consistent hemisphere definitions ($A_B$ = 0)", fontsize=12)
    fig.tight_layout(rect=(0., 0.14, 1., 1.))   # room for the figure legend: lay out with tight=False in paper_figure
    return fig


###########################################################################################################################
#####
#####   Expected evidence: where the Bayes factor comes from (Evidence_Prediction.ipynb)
#####
###########################################################################################################################

# For two models with the same parameters and priors (extended source vs point source, both A_B + baselines) the Occam factors nearly cancel, so the Bayes factor follows the likelihood
# ratio. Linearised around the truth, with the noise d = truth + sigma z: ln B(ext vs ps) = chi2_min / 2 + zeta sqrt(chi2_min), zeta = r_perp_hat . z ~ N(0, 1), where
# r_perp = (truth - m_ps(A_fit) - baseline) / sigma_bin is the whitened residual of the NOISELESS A_B-only fit and chi2_min = |r_perp|^2 the chi2 that fit_naive returns.
# So <ln B> = chi2_min / 2 with scatter sqrt(chi2_min) over noise realisations, and r_perp^2 says at which orbital phases the expected evidence arises.
# Only for equal parameter counts: against the fit with redistribution the F prior adds an Occam term of ~3 nats and F's posterior is one-sided, so chi2 / 2 is then only the likelihood part.


# The whitened residual r_perp of the noiseless A_B-only fit of `insolation_model` to `truth`, bin by bin (detector blocks concatenated, each with its own baseline removed and its own
# sigma_bin). A_B_fit=None fits it (fit_AB_only); sum(r_perp^2) is that fit's chi2. Returns dict(A_B_fit, chi2, resid [ppm], sigma [ppm], r_perp, phase [bin centres, 0 = transit],
# blocks [one slice per detector]). kw: the model_pcurve keywords of the fit, so the dynesty comparison can evaluate it at the truth's own resolution.
def evidence_residual(cfg, truth, insolation_model="naive", A_B_fit=None, free_offset=True, bounds=A_B_FIT_BOUNDS, **kw):
    if A_B_fit is None: A_B_fit = fit_AB_only(cfg, truth, insolation_model, free_offset=free_offset, bounds=bounds, **kw)[0]
    resid, _, sigma = _chi2_problem(cfg, truth, insolation_model, free_offset=free_offset, **kw)
    r = resid(A_B_fit)
    cfgs = list(cfg) if is_detector_set(cfg) else [cfg]
    edges = np.cumsum([0] + [c.n_bins for c in cfgs])
    return dict(A_B_fit=float(A_B_fit), chi2=float(np.sum((r / sigma) ** 2)), resid=r, sigma=sigma, r_perp=r / sigma,
                phase=np.concatenate([(np.arange(c.n_bins) + 0.5) / c.n_bins for c in cfgs]),   # model_pcurve runs from t = 0 (mid-transit) over one period and binned() averages consecutive samples
                blocks=[slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])])


# Half-duration of the transit / occultation in orbital phase (first to fourth contact; a circular orbit, so the two last equally long): occultation_durations at the planet's geometry.
def eclipse_half_phase(planet):
    from types import SimpleNamespace
    T14, _ = occultation_durations(SimpleNamespace(inc=planet.incl, a=planet.R / planet.r2, rp=planet.r1 / planet.r2, per=planet.period))
    return 0.5 * T14 / planet.period


# The phase windows of an evidence budget as boolean masks over bin centres `phase` (0 = mid-transit): occultation and transit (every bin that overlaps the eclipse, i.e. whose
# bin average the eclipse touches: centre within the half-duration plus half a bin), the quadratures (centre within quad_half of 0.25 or 0.75) and the rest. Disjoint, in that priority.
def phase_windows(planet, phase, n_bins, quad_half=0.1):
    dist = lambda p0: np.abs((np.asarray(phase, float) - p0 + 0.5) % 1. - 0.5)   # phase distance on the circle
    h = eclipse_half_phase(planet) + 0.5 / n_bins
    occ = dist(0.5) < h
    tra = (dist(0.) < h) & ~occ
    quad = ((dist(0.25) < quad_half) | (dist(0.75) < quad_half)) & ~occ & ~tra
    return {"occultation": occ, "transit": tra, "quadratures": quad, "rest": ~(occ | tra | quad)}


# Where in phase the expected evidence of an evidence_residual arises: chi2 and its share of the total chi2_min per phase window (phase_windows), one row per detector block plus "all".
# labels: the detectors' names (default: block numbers). The shares of the "all" row sum to 1; 2 <ln B> = chi2_min, so they are also the shares of the expected ln B.
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


# P(ln B > threshold) over noise realisations for extended source vs point source, in the linear prediction ln B ~ N(chi2_min / 2, chi2_min). chi2 scales with the observing time,
# so time_factor = n_orbits / (the n_orbits chi2 was computed at) gives another number of orbits without a refit (e.g. 2 / 5 for one visit instead of five orbits).
def evidence_probability(chi2, threshold, time_factor=1.):
    from scipy.stats import norm
    c = time_factor * np.asarray(chi2, dtype=float)
    return norm.sf((threshold - 0.5 * c) / np.sqrt(c))


# The expected-evidence table: every planet of `planets` on every entry of `instruments` (instrument_retrievals per planet: own geometry, own noise, joint detector fits as tuples) with
# both point-source fits, one tidy row per (planet, instrument, fit). Columns: planet, instrument, fit ("naive" / "naive_redist"), A_B_fit, bias_A_B, sigma_A_B, bias_sigma, F_fit, chi2,
# half_chi2 = chi2 / 2, and for the A_B-only fit lnB_mean = chi2 / 2, lnB_sd = sqrt(chi2) and P_lnB_gt_<x> = P(ln B > x) at cfg.n_orbits for each x of `thresholds`.
# For "naive_redist" half_chi2 is only the LIKELIHOOD part of ln B(ext vs ps+F) (no Occam term of the F prior), so it gets no lnB columns; its z_F = sqrt(chi2_naive - chi2) is the
# significance of the spurious F. Rows without a noise level (MIRI imaging for most planets) keep the fitted values and NaN in everything that needs the noise.
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


# Per-bin evidence figure: one column per detector of every entry of `cases`, in wavelength order ({label: FittingConfig or detector set}, e.g. MIRI/LRS next to the joint NIRSpec fit). Top: the residual
# truth - fitted point-source model (baseline removed) unbinned and per bin, against +- sigma_bin. Bottom: r_perp^2 per bin, which sums to chi2_min = 2 <ln B> over all bins of a fit,
# with the transit and occultation windows shaded and the quadrature windows hatched (phase_windows). Returns fig, axes (2 x n_columns) and {label: evidence_residual} for the budgets.
# kw: the model_pcurve keywords of truth and fit.
def plot_evidence_bins(cases, A_B_true, quad_half=0.1, title=None, figsize=None, **kw):
    evs, cols = {}, []
    for label, cfg in cases.items():
        cfgs = list(cfg) if is_detector_set(cfg) else [cfg]
        truths = [model_pcurve(c, "hyperillum", A_B_true, **kw) for c in cfgs]
        ev = evidence_residual(cfg, [y for _, y in truths] if is_detector_set(cfg) else truths[0][1], **kw)
        evs[label] = ev
        for c, (t_, y), blk in zip(cfgs, truths, ev["blocks"]):
            fit = model_pcurve(c, "naive", ev["A_B_fit"], **kw)[1]
            cols.append((label, c, ev, blk, t_ / c.planet.period, y - fit - (y - fit).mean()))   # the mean of the unbinned residual is the baseline free_offset removes (bins of equal size)
    cols.sort(key=lambda col: 0.5 * (col[1].l1 + col[1].l2))   # columns in wavelength order (NRS1, NRS2, LRS) whatever the order of `cases`; stable, so a joint fit's detectors stay together
    fig, axes = plt.subplots(2, len(cols), figsize=figsize or (5.6 * len(cols), 6.6), sharex=True, sharey="row", squeeze=False, gridspec_kw=dict(height_ratios=[1.25, 1.]))
    for k, (label, c, ev, blk, ph_hi, r_hi) in enumerate(cols):
        ax, axb = axes[:, k]
        ph, r, r2, sb = ev["phase"][blk], ev["resid"][blk], ev["r_perp"][blk] ** 2, c.sigma_bin
        win = phase_windows(c.planet, ph, c.n_bins, quad_half)
        h_ecl = eclipse_half_phase(c.planet)
        for a_ in (ax, axb):
            for p0, col in ((0., "#e3e2dc"), (1., "#e3e2dc"), (0.5, "#f3d9cc")):   # transit (grey, at both ends of the orbit) and occultation (light orange)
                a_.axvspan(p0 - h_ecl, p0 + h_ecl, color=col, lw=0, zorder=0)
            for p0 in (0.25, 0.75): a_.axvspan(p0 - quad_half, p0 + quad_half, facecolor="none", edgecolor="#d9d8d1", hatch="///", lw=0, zorder=0)
            a_.grid(False); a_.set_xlim(0., 1.)
        ax.axhspan(-sb, sb, color=INK_2, alpha=0.10, lw=0, zorder=1, label=rf"$\pm\sigma_{{\rm bin}}$")
        ax.axhline(0., color=AXIS, lw=.8, zorder=1)
        ax.plot(ph_hi, r_hi, color=C_HI, lw=1.2, zorder=3, label="truth $-$ fitted point source")
        ax.plot(ph, r, "o", ms=4, color=INK, zorder=4, label="per bin")
        ax.set_title(f"{instrument_label(_band_of(c), oneline=True)}" + (f"\n{label}" if label else "") + rf"  ($\sigma_{{\rm bin}}$ = {sb:.1f} ppm)", fontsize=10)
        axb.bar(ph, r2, width=0.8 / c.n_bins, color=C_HI, zorder=3)
        share = {name: 100. * r2[m].sum() / ev["chi2"] for name, m in win.items()}
        axb.text(0.5, 0.97, rf"$\Sigma$ = {r2.sum():.2f} of $\chi^2_{{\min}}$ = {ev['chi2']:.2f}" + "\n" + ", ".join(f"{n} {v:.0f} %" for n, v in share.items()), transform=axb.transAxes, ha="center", va="top", fontsize=8,
                 bbox=dict(facecolor="#fcfcfb", edgecolor="none", alpha=0.85, pad=1.5), zorder=5)
        axb.set_xlabel("orbital phase (0 = transit)")
    axes[0, 0].set_ylabel("residual [ppm]"); axes[1, 0].set_ylabel(r"$r_\perp^2$ per bin  (sums to $2\,\langle\ln B\rangle$)")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    axes[1, 0].set_ylim(0., 1.45 * max(float((ev["r_perp"] ** 2).max()) for ev in evs.values()))   # head room for the budget text
    c0 = cols[0][1]
    fig.suptitle(title or rf"{c0.planet.name}: where the expected evidence arises ($A_B$ = {A_B_true:.2f}, {c0.n_orbits} orbits, {c0.n_bins} bins)", y=0.995)
    fig.tight_layout()
    return fig, axes, evs


# The FILTER_LAMBDA_UM key of a FittingConfig's band (matched on its throughput file), for labels; the file name itself where no key matches.
def _band_of(cfg):
    from HItargets_utils import FILTER_FILES
    import os
    name = os.path.basename(str(cfg.filter_path))
    return next((k for k, f in FILTER_FILES.items() if f == name), name)


###########################################################################################################################
#####
#####   Robustness to nuisance parameters: one linear least-squares problem (Nuisance_Robustness.ipynb)
#####
###########################################################################################################################

# sigma_A_B and chi2_min of the A_B-only fit assume a fixed geometry and no other phase-curve terms. A real analysis marginalises over the geometry and fits ellipsoidal terms, and
# every such nuisance is one more column of the same linear least-squares problem. Linearised around the noiseless A_B-only fit, in whitened units and with each column's mean removed
# per detector block (the free baseline): y = r_perp (evidence_residual), X = [j_A, j_nu1, ...] with j = d(model) / d(theta) / sigma_bin, a Gaussian prior N(nu_0, sigma_p) on a
# nuisance as the penalty (delta_nu / sigma_p)^2 and a flat prior as no penalty:
#   delta = argmin |y - X delta|^2 + sum (delta_nu / sigma_p)^2,   Cov = (X'X + Lambda)^-1,   sigma_A_B = sqrt(Cov_AA),   A_B_fit' = A_B_fit + delta_A,   chi2_min' = the minimum.
# If the extended-source model carries the same nuisances with the same priors, <ln B> = chi2_min' / 2 to first order (the determinant terms cancel). The columns:
# - geometry "a_rs", "rp_rs", "incl": central differences of the point-source model (the transit and the occultation are part of it, so their shape constrains the geometry too);
# - geometry "b" (impact parameter) and "a_rs|b" (a/R* at fixed b, i = arccos(b / (a/R*)) following it), for a planet whose source fitted b and derived i (TOI-2431 b, Tas et al. 2026):
#   both follow from the a_rs and incl columns by the chain rule, so they cost no model evaluation. A Gaussian prior on "a_rs" also applies to "a_rs|b";
# - "stellar": the ellipsoidal variation of the STAR, -cos(4 pi phi) [ppm] in every bin including the occultation (the star stays visible), one free amplitude per detector
#   ("stellar:1", "stellar:2", ...: limb and gravity darkening depend on the band). Free per band, it can follow the wavelength dependence of whatever it absorbs: the worst case;
# - "stellar_shared": the same term with ONE amplitude for all detectors [ppm, in a band of ratio 1], each detector's column scaled by `stellar_ratios` (stellar_shared_column). The relative
#   amplitude A_ell = alpha_ell (M_p / M_star) (R_star / a)^3 sin^2 i depends on the band only through alpha_ell(u, g), by ~10 % across the infrared, so this is its physical form, and
#   with the Gaussian prior that the measured mass ratio implies (ellipsoidal_prior) the realistic one. Its prior is centred on the truth (no ellipsoidal term in truth or model);
# - "planetary": the tidal deformation of the PLANET, as in Akinsanmi et al. (2024, A&A 682, A15, eq. 11): its flux F_p(phi) times its projected area relative to mid-transit,
#   A(phi) / A(0), with A(phi) the projected area of a triaxial ellipsoid (their eq. 2, from Leconte et al. 2011) of the Correia (2014) shape r1 = r2 (1 + 3q), r3 = r2 (1 - q)
#   (r1 towards the star, r2 along the orbit, r3 polar). Amplitude delta = (r1 - r2) / r2 = 3q; the column is the derivative at delta = 0, F_p(phi) sin^2 i sin^2(2 pi phi), and
#   planetary_area_factor the exact factor (the nonlinear check). F_p is the point-source planet flux at the fitted A_B. One amplitude shared by all detectors (it is geometric),
#   zero during the occultation, like the hyperillumination residual itself.
# Not for the fit with redistribution: F sits at its F = 0 edge, where the response to F switches on very flat, so nothing linear in F is trustworthy.
NUISANCE_STEPS = dict(a_rs=0.005, rp_rs=1e-4, incl=0.2)   # finite-difference steps of the geometry columns [a/R*, Rp/R*, deg]; nuisance_problem(steps=...) to test them
NUISANCE_LABELS = {"a_rs": r"$a/R_\star$", "rp_rs": r"$R_p/R_\star$", "incl": r"$i$", "b": r"$b$", "a_rs|b": r"$a/R_\star$ (fixed $b$)", "stellar": "stellar ellipsoidal (free per band)", "stellar_shared": "stellar ellipsoidal (shared)", "planetary": "planetary deformation"}


# The whitened, mean-removed binned point-source model at a perturbed geometry: the building block of a geometry column. a_rs / rp_rs go to model_pcurve, incl into the planet.
# The geometry-only caches a caller may pass (insol, refl_unit) belong to the unperturbed geometry and are dropped here.
def _geometry_model(cfg, A_B_, param, value, kw):
    kw = {k: v for k, v in kw.items() if k not in ("insol", "refl_unit")}
    if param == "incl":
        cfg = [replace(c, planet=replace(c.planet, incl=value)) for c in cfg] if is_detector_set(cfg) else replace(cfg, planet=replace(cfg.planet, incl=value))
    else:
        kw = {**kw, param: value}
    resid, _, sigma = _chi2_problem(cfg, None, "naive", **kw)
    return -resid(A_B_) / sigma


# Everything the nuisance fits of one configuration share: y = r_perp of the noiseless A_B-only fit, the A_B column and every nuisance column (see above), whitened and mean-removed
# per detector block. cfg: FittingConfig or detector set; truth: its noiseless truth curve(s); A_B_fit=None fits it. priors: {name: Gaussian sigma} for any column (a missing
# name = flat). stellar_ratios: the band ratios of "stellar_shared" (see stellar_shared_column). Returns dict(A_B_fit, chi2, y, j_A, columns {name: column}, priors,
# values {geometry name: the planet's value}, sigma, blocks, phase, stellar_ratios [one per detector]).
def nuisance_problem(cfg, truth, A_B_fit=None, priors=None, steps=NUISANCE_STEPS, bounds=A_B_FIT_BOUNDS, stellar_ratios=None, **kw):
    ev = evidence_residual(cfg, truth, A_B_fit=A_B_fit, bounds=bounds, **kw)
    A = ev["A_B_fit"]
    cfgs = list(cfg) if is_detector_set(cfg) else [cfg]
    pl = cfgs[0].planet
    values = dict(a_rs=pl.R / pl.r2, rp_rs=pl.r1 / pl.r2, incl=pl.incl)
    cols = {}
    for name, v in values.items():
        h = steps[name]
        cols[name] = (_geometry_model(cfg, A, name, v + h, kw) - _geometry_model(cfg, A, name, v - h, kw)) / (2. * h)
    # b and a/R* at fixed b, from the incl column [per deg] by the chain rule: i = arccos(b / a) gives di/db = -(180/pi) / (a sin i) and di/da|_b = (180/pi) b / (a^2 sin i) [deg]
    a0, i0 = values["a_rs"], np.radians(values["incl"])
    values.update({"b": a0 * np.cos(i0), "a_rs|b": a0})
    cols["b"] = -np.degrees(1.) / (a0 * np.sin(i0)) * cols["incl"]
    cols["a_rs|b"] = cols["a_rs"] + np.degrees(1.) * values["b"] / (a0 ** 2 * np.sin(i0)) * cols["incl"]
    priors = dict(priors or {})
    if "a_rs" in priors: priors.setdefault("a_rs|b", priors["a_rs"])   # one prior on a/R*, whichever of i and b is held
    planetary = []
    for k, (c, blk) in enumerate(zip(cfgs, ev["blocks"])):
        t_ = model_times(c.planet.period, kw.get("n_t") or 50 * c.n_bins)
        ph = 2. * np.pi * t_ / c.planet.period
        ell = -binned(c, np.cos(2. * ph))                                    # -cos(4 pi phi), bin-averaged [per ppm of amplitude]
        col = np.zeros(ev["r_perp"].size)
        col[blk] = (ell - ell.mean()) / c.sigma_bin
        cols[f"stellar:{k + 1}"] = col
        fp = model_pcurve(replace(c, transit=False), "naive", A, **kw)[1]     # the planet's own flux [ppm]: thermal + reflected, zero in occultation
        d = binned(c, fp * np.sin(np.radians(c.planet.incl)) ** 2 * np.sin(ph) ** 2)   # d[F_p A(phi) / A(0)] / d delta at delta = 0 (planetary_area_factor)
        planetary.append((d - d.mean()) / c.sigma_bin)
    cols["planetary"] = np.concatenate(planetary)
    ratios = _stellar_ratios(cfgs, stellar_ratios)
    cols["stellar_shared"] = stellar_shared_column(cols, ratios)
    return dict(A_B_fit=A, chi2=ev["chi2"], y=ev["r_perp"], j_A=whitened_jacobian(cfg, "naive", A, bounds=bounds, **kw)[:, 0], columns=cols, priors=priors, values=values,
                sigma=ev["sigma"], blocks=ev["blocks"], phase=ev["phase"], stellar_ratios=ratios)


# The band ratios of the shared stellar term, one per detector of cfgs: None -> 1 for every detector (achromatic); a dict {band key of HItargets_utils: ratio} -> the ratio of each
# detector's band (1 where the band is missing, which makes the amplitude that of a band of ratio 1); a sequence -> as given, in detector order.
def _stellar_ratios(cfgs, stellar_ratios):
    if stellar_ratios is None: return np.ones(len(cfgs))
    r = np.array([float(stellar_ratios.get(_band_of(c), 1.)) for c in cfgs]) if isinstance(stellar_ratios, dict) else np.asarray(stellar_ratios, float).ravel()
    if r.size != len(cfgs) or not np.all(np.isfinite(r) & (r > 0.)): raise ValueError(f"stellar_ratios: {len(cfgs)} positive values needed, got {r}")
    return r


# The stellar ellipsoidal column with ONE amplitude for all detectors: the per-detector columns "stellar:<k>" of a nuisance problem's `columns`, weighted by `ratios` (one per detector,
# the band's alpha_ell relative to the reference) and added. The per-detector columns are zero outside their own block, so the sum is their concatenation.
def stellar_shared_column(columns, ratios):
    names = sorted((c for c in columns if c.startswith("stellar:")), key=lambda c: int(c.split(":")[1]))
    r = np.asarray(ratios, float).ravel()
    if r.size != len(names): raise ValueError(f"{len(names)} detectors but {r.size} ratios")
    return np.sum([rk * columns[n] for rk, n in zip(r, names)], axis=0)


# Projected area of the tidally deformed planet relative to mid-transit, A(phi) / A(0) (Akinsanmi et al. 2024, A&A 682, A15, eqs. 2 and 11; Leconte et al. 2011): the triaxial
# ellipsoid of Correia (2014), r1 = r2 (1 + 3q), r3 = r2 (1 - q), with the amplitude delta = (r1 - r2) / r2 = 3q, at the orbital angle phi [rad, 0 = mid-transit] and the inclination
# incl [deg]. To first order in delta it is 1 + delta sin^2 i sin^2 phi, the "planetary" column of nuisance_problem.
def planetary_area_factor(delta, incl, phi):
    q = delta / 3.
    r1, r2, r3 = 1. + 3. * q, 1., 1. - q
    ci2, si2 = np.cos(np.radians(incl)) ** 2, np.sin(np.radians(incl)) ** 2
    area = lambda ph: np.sqrt(r1 ** 2 * r2 ** 2 * ci2 + r3 ** 2 * si2 * (r1 ** 2 * np.sin(ph) ** 2 + r2 ** 2 * np.cos(ph) ** 2))
    return area(np.asarray(phi, float)) / area(0.)


# The names of the columns a nuisance selection stands for: "stellar" expands to one amplitude per detector, every other name is its own column.
def nuisance_names(problem, nuisances):
    out = []
    for n in nuisances:
        out += [c for c in problem["columns"] if c.startswith("stellar:")] if n == "stellar" else [n]
    return out


# One nuisance fit: the A_B-only fit of `problem` (nuisance_problem) with the columns of `nuisances` added (names as in nuisance_names; () = the fixed-geometry fit itself).
# Returns dict(sigma_A_B, delta_A_B [shift of the fitted A_B], A_B_fit, chi2, lnB_mean = chi2 / 2, names, delta {name: fitted shift}, sigma_post {name}, shrinkage {name:
# sigma_post / sigma_prior; NaN for a flat prior; 1 = the data add nothing to the prior}, r_perp [the augmented whitened residual y - X delta, without the prior rows], cov).
def nuisance_solve(problem, nuisances=()):
    names = nuisance_names(problem, nuisances)
    X = np.column_stack([problem["j_A"]] + [problem["columns"][n] for n in names])
    prior = np.array([np.nan] + [problem["priors"].get(n, np.nan) for n in names])
    lam = np.where(np.isfinite(prior), 1. / prior ** 2, 0.)
    cov = np.linalg.inv(X.T @ X + np.diag(lam))
    delta = cov @ (X.T @ problem["y"])
    r = problem["y"] - X @ delta
    chi2 = float(r @ r + np.sum(lam * delta ** 2))
    sd = np.sqrt(np.diag(cov))
    return dict(sigma_A_B=float(sd[0]), delta_A_B=float(delta[0]), A_B_fit=float(problem["A_B_fit"] + delta[0]), chi2=chi2, lnB_mean=0.5 * chi2, names=names,
                delta=dict(zip(names, delta[1:])), sigma_post=dict(zip(names, sd[1:])), shrinkage=dict(zip(names, sd[1:] / prior[1:])), r_perp=r, cov=cov)


# The nuisance table of one configuration: one row per entry of `rows` ({row label: tuple of nuisances}) with sigma_A_B, the bias dA_B = A_B_fit' - A_B_true and |dA_B| / sigma_A_B,
# chi2_min' and <ln B> = chi2_min' / 2, and per nuisance of the row its sigma_post / sigma_prior ("shrink <name>"; "-" columns stay NaN) or, for a flat-prior amplitude, its fitted
# value and sigma ("<name>", "sig <name>": stellar amplitudes in ppm, the planetary one as delta = (r1 - r2) / r2).
def nuisance_table(problem, A_B_true, rows):
    out = []
    for label, nuis in rows.items():
        s = nuisance_solve(problem, nuis)
        row = {"row": label, "sigma_A_B": s["sigma_A_B"], "bias_A_B": s["A_B_fit"] - A_B_true, "bias_sigma": abs(s["A_B_fit"] - A_B_true) / s["sigma_A_B"], "chi2": s["chi2"], "lnB_mean": s["lnB_mean"]}
        for n in s["names"]:
            shrink = s["shrinkage"][n]
            if n in problem["values"]:   # geometry: how far the data shrink its prior (the fitted shift and its sigma where it has none)
                if np.isfinite(shrink): row[f"shrink {n}"] = shrink
                else: row[n], row[f"sig {n}"] = s["delta"][n], s["sigma_post"][n]
            else:                        # an amplitude, zero in the truth: what the fit gives it and its sigma, with a prior also the shrinkage
                row[n], row[f"sig {n}"] = s["delta"][n], s["sigma_post"][n]
                if np.isfinite(shrink): row[f"shrink {n}"] = shrink
        out.append(row)
    return pd.DataFrame(out)


# Nonlinear counterpart of nuisance_solve, as a check of the linearisation: scipy least_squares over (A_B, the nuisances), the point-source model re-evaluated at every step
# (geometry through model_pcurve / the planet's inclination, the planetary deformation as amplitude x F_p(theta) x shape, the stellar term per detector), each detector's baseline
# removed by its mean and the Gaussian priors as residual rows (theta - theta_0) / sigma_p. sigma from the Jacobian at the solution. Returns dict(A_B_fit, sigma_A_B, chi2,
# values {name: fitted value}, sigma_post {name}, nfev).
def nuisance_fit_nonlinear(cfg, truth, nuisances, priors=None, A_B_start=None, bounds=A_B_FIT_BOUNDS, stellar_ratios=None, **kw):
    from scipy.optimize import least_squares
    cfgs = list(cfg) if is_detector_set(cfg) else [cfg]
    truths = list(truth) if is_detector_set(cfg) else [truth]
    pl = cfgs[0].planet
    start = dict(a_rs=pl.R / pl.r2, rp_rs=pl.r1 / pl.r2, incl=pl.incl)
    start.update({"b": start["a_rs"] * np.cos(np.radians(pl.incl)), "a_rs|b": start["a_rs"]})   # b: i follows from b and a/R* (see nuisance_problem)
    names = []
    for n in nuisances: names += [f"stellar:{k + 1}" for k in range(len(cfgs))] if n == "stellar" else [n]
    priors = dict(priors or {})
    if "a_rs" in priors: priors.setdefault("a_rs|b", priors["a_rs"])
    ratios = _stellar_ratios(cfgs, stellar_ratios)
    yb = [binned(c, t) for c, t in zip(cfgs, truths)]
    typical = {"a_rs": 0.04, "a_rs|b": 0.04, "rp_rs": 3e-4, "incl": 1., "b": 0.03, "planetary": 0.02}   # typical size of a step per parameter (stellar amplitudes: 1 ppm), for the solver's scaling only
    scale = np.array([0.05] + [priors.get(n, typical.get(n, 1.)) for n in names])

    def residuals(theta):
        p = dict(zip(names, theta[1:]))
        geom = {k: p[k] for k in ("a_rs", "rp_rs") if k in p}
        if "a_rs|b" in p: geom["a_rs"] = p["a_rs|b"]
        out = []
        for k, (c, y) in enumerate(zip(cfgs, yb)):
            if "incl" in p: c = replace(c, planet=replace(c.planet, incl=p["incl"]))
            elif "b" in p: c = replace(c, planet=replace(c.planet, incl=float(np.degrees(np.arccos(p["b"] / geom.get("a_rs", start["a_rs"]))))))
            t_, m = model_pcurve(c, "naive", theta[0], **geom, **kw)
            ph = 2. * np.pi * t_ / c.planet.period
            if "planetary" in p:
                fp = model_pcurve(replace(c, transit=False), "naive", theta[0], **geom, **kw)[1]
                m = m + fp * (planetary_area_factor(p["planetary"], c.planet.incl, ph) - 1.)   # the exact projected-area factor, not its linearisation
            if f"stellar:{k + 1}" in p: m = m - p[f"stellar:{k + 1}"] * np.cos(2. * ph)
            if "stellar_shared" in p: m = m - p["stellar_shared"] * ratios[k] * np.cos(2. * ph)
            r = y - binned(c, m)
            out.append((r - r.mean()) / c.sigma_bin)
        out.append(np.array([(p[n] - start.get(n, 0.)) / priors[n] for n in names if n in priors]))   # amplitudes: prior centred on 0, the truth
        return np.concatenate(out)

    x0 = np.array([A_B_start if A_B_start is not None else fit_naive(cfg, truth, bounds=bounds, **kw)[0]] + [start.get(n, 0.) for n in names])
    res = least_squares(residuals, x0, x_scale=scale, diff_step=1e-3, xtol=1e-10, ftol=1e-12, gtol=1e-12)
    sd = np.sqrt(np.diag(np.linalg.inv(res.jac.T @ res.jac)))
    return dict(A_B_fit=float(res.x[0]), sigma_A_B=float(sd[0]), chi2=float(2. * res.cost), values=dict(zip(names, res.x[1:])), sigma_post=dict(zip(names, sd[1:])), nfev=int(res.nfev))


# Expected semi-amplitude [ppm] of the star's ellipsoidal variation, A_ell = alpha_ell (M_p / M_star) (R_star / a)^3 sin^2 i with alpha_ell = 0.15 (15 + u) (1 + g) / (3 - u)
# (Morris & Naftilan 1993; Shporer 2017, PASP 129, 072001, eqs. 7-8): u the linear limb-darkening and g the gravity-darkening coefficient of the band, mass_ratio = M_p / M_star.
def ellipsoidal_amplitude(planet, mass_ratio, u, g):
    alpha = 0.15 * (15. + u) * (1. + g) / (3. - u)
    return 1e6 * alpha * mass_ratio * (planet.r2 / planet.R) ** 3 * np.sin(np.radians(planet.incl)) ** 2


# The prior of the shared stellar term: the expected semi-amplitude A [ppm] (ellipsoidal_amplitude at the middle of the (u, g) bracket) and its fractional 1-sigma uncertainty, the
# quadrature sum of the fractional errors of M_p / M_star (frac_q), of (R_star / a)^3 (3 sig_a_rs / (a/R*)), of sin^2 i (2 sig_incl [deg] / tan i) and of alpha_ell (half the bracket
# range over its middle, taken as 1 sigma: generous for a bracket). Errors independent (the M_star - a/R* correlation through the stellar density is ignored).
# Returns dict(A, frac, sigma = frac * A [ppm], parts {source: fractional error}).
def ellipsoidal_prior(planet, mass_ratio, frac_q, u_range, g_range, sig_a_rs=0., sig_incl=0.):
    lo, hi = ellipsoidal_amplitude(planet, mass_ratio, u_range[0], g_range[0]), ellipsoidal_amplitude(planet, mass_ratio, u_range[1], g_range[1])
    parts = {"M_p / M_star": float(frac_q), "(R_star / a)^3": 3. * sig_a_rs / (planet.R / planet.r2), "sin^2 i": 2. * np.radians(sig_incl) / np.tan(np.radians(planet.incl)),
             "alpha_ell (u, g)": (hi - lo) / (hi + lo)}
    A, frac = 0.5 * (lo + hi), float(np.sqrt(sum(v ** 2 for v in parts.values())))
    return dict(A=A, frac=frac, sigma=frac * A, parts=parts)


###########################################################################################################################
#####
#####   One-parameter sweeps of the bias (AB_Bias.ipynb, F_AB_Bias.ipynb: inclination, u_bol)
#####
###########################################################################################################################

# A sweep changes ONE setting of the configuration and repeats the whole bias pipeline (fit_truth: truth, fits, Fisher sigmas) at every value. The setting is a transform
# (cfg, value) -> cfg, applied to every detector of a set, so anything a FittingConfig or its planet holds can be swept without a function of its own.
# cfg_with_incl: the orbital inclination [deg] (truth and fits; the noise level stays the planet's own). cfg_with_u_bol: the bolometric limb darkening of the irradiance, which only the
# hyperillumination truth uses (the point-source models have no u_bol), so it is a sweep of the truth alone.
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


# fit_truth (Fisher sigmas included) at every value of a one-parameter sweep, points in parallel, as a DataFrame. cfg: FittingConfig or detector set; transform: (cfg, value) -> cfg
# (cfg_with_incl, cfg_with_u_bol, or any function of that form). Columns: value; A_B_fit, bias, sigma, bias_sigma, chi2, lnB_mean = chi2 / 2, railed (A_B-only fit); A_B_fit_J, bias_J,
# sigma_J, bias_sigma_J, F_fit, chi2_J, railed_J and z_F = sqrt(chi2 - chi2_J) (fit with redistribution; z_F needs both models), as in bias_vs_a_rs.
def bias_sweep(cfg, A_B_true, values, transform, models=("naive", "naive_redist"), bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, n_jobs=-1, **kw):
    return pd.DataFrame(pmap(_bias_sweep_point, [(cfg, A_B_true, transform, v, models, bounds, F_bounds, kw) for v in values], n_jobs))


# A bias_sweep per mode on shared axes: sweeps = {mode label: DataFrame}. joint=False (A_B-only fit): the bias (right axis: in % of the true A_B), |bias| / sigma, and
# <ln B> = chi2_min / 2. joint=True (the sweep was run with both models): the bias of the fit with redistribution (solid) next to the A_B-only fit (dashed), the spurious F, and
# z_F = sqrt(dchi2_F) (solid) next to |bias| / sigma of the fit with redistribution (dashed). mark: the x value of the planet's own setting, drawn as the green star on every curve.
def plot_bias_sweep(sweeps, A_B_true, xlabel, joint=False, mark=None, mark_label=None, title=None, figsize=(18., 5.)):
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    colors = (C_HI, C_REDIST, INK_2, F_COLORS[0])
    star = lambda ax, t, col, **k: ax.plot(mark, np.interp(mark, t["value"].to_numpy()[np.argsort(t["value"].to_numpy())], t[col].to_numpy()[np.argsort(t["value"].to_numpy())]), **{**TARGET, "ms": 14, **k}) if mark is not None else None
    for (label, t), c in zip(sweeps.items(), colors):
        b, s, sig = ("bias_J", "sigma_J", "bias_sigma_J") if joint else ("bias", "sigma", "bias_sigma")
        axs[0].plot(t["value"], t[b], "o-", color=c, lw=1.5, ms=5, label=label + (", $A_B$ + $F$ fit" if joint else ""))
        if joint: axs[0].plot(t["value"], t["bias"], "o--", color=c, lw=1.1, ms=4, mfc="#fcfcfb", label=label + ", $A_B$-only fit")
        star(axs[0], t, b)
        if joint:
            axs[1].plot(t["value"], t["F_fit"], "o-", color=c, lw=1.5, ms=5, label=label); star(axs[1], t, "F_fit")
            axs[2].plot(t["value"], t["z_F"], "o-", color=c, lw=1.5, ms=5, label=label + r": $z_F = \sqrt{\Delta\chi^2_F}$"); star(axs[2], t, "z_F")
            axs[2].plot(t["value"], t[sig], "o--", color=c, lw=1.1, ms=4, mfc="#fcfcfb", label=label + r": $|\Delta A_B|/\sigma_{A_B}$")
        else:
            axs[1].plot(t["value"], t[sig], "o-", color=c, lw=1.5, ms=5, label=label); star(axs[1], t, sig)
            axs[2].plot(t["value"], t["lnB_mean"], "o-", color=c, lw=1.5, ms=5, label=label); star(axs[2], t, "lnB_mean")
    axs[0].axhline(0., color=AXIS, lw=.8)
    axs[0].set_ylabel(r"bias $\Delta A_B = A_{B,\rm fit} - A_B$"); axs[0].set_title(r"Bias in recovered $A_B$")
    sec = axs[0].secondary_yaxis("right", functions=(lambda v: 100. * v / A_B_true, lambda v: v * A_B_true / 100.)); sec.set_ylabel(r"[% of $A_B$]"); axs[0].spines["right"].set_visible(True)
    if joint:
        axs[1].set_ylabel(r"fitted $F$  (truth: $F$ = 0)"); axs[1].set_title("Spurious heat redistribution")
        axs[2].set_ylabel(r"significance [$\sigma$]"); axs[2].set_title("Significance of the spurious $F$ and of the bias")
    else:
        axs[1].axhline(1., color=INK, lw=.9, ls="--")
        axs[1].set_ylabel(r"$|\Delta A_B|\,/\,\sigma_{A_B}^{\rm Fisher}$"); axs[1].set_title("Significance of the bias")
        axs[2].set_ylabel(r"$\langle\ln B\rangle = \chi^2_{\min}/2$"); axs[2].set_title("Expected evidence for the extended source")
    if mark is not None: axs[0].plot([], [], label=mark_label or "the planet's own value", **{**TARGET, "ms": 14})
    for a_ in axs: a_.set_xlabel(xlabel); a_.legend(fontsize=8)
    if title: fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig, axs
