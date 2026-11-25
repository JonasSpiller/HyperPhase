"""Noiseless least-squares fits of the point-source models to the extended-source truth."""

import numpy as np
from scipy.optimize import minimize_scalar

from hyperphase.bias.detectors import is_detector_set, template_cfg
from hyperphase.defaults import A_B_FIT_BOUNDS, F_BOUNDS
from hyperphase.model import model_pcurve, reflected_unit
from hyperphase.parallel import pmap


def temperature_bias(A_B_true, A_B_bias, F_fit=0.):
    """Fractional surface-temperature bias T_fit / T_true - 1 of an A_B bias. With a fitted redistribution F_fit (naive_redist) it is that of the
    substellar temperature, where T^4 ∝ (1 - A_B)(1 - 3F/2) (Kreidberg & Loeb 2016 eq. 1)."""
    A_fit = A_B_true + A_B_bias
    return ((1 - A_fit) * (1 - 1.5 * F_fit) / (1 - A_B_true))**0.25 - 1


############################################################################################################################################
#####   Binning
############################################################################################################################################

def bin_curve(t, y, minutes_binning):
    """Average y in fixed-width time bins (t in days, minutes_binning in minutes); the last bin takes the remaining points. For plotting only: fits
    use binned."""
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


############################################################################################################################################
#####   Noiseless fits
############################################################################################################################################

def binned(cfg, y):
    """Exact block-average bins of y (len(y) a multiple of n_bins)."""
    k = len(y) // cfg.n_bins
    if k * cfg.n_bins != len(y):
        raise ValueError(f"n_t={len(y)} is not a multiple of n_bins={cfg.n_bins}; "
                         f"use n_t = {cfg.n_bins} * k (e.g. {cfg.n_bins * 40})")
    return y.reshape(cfg.n_bins, k).mean(axis=1)


def fit_residual(cfg, yb_truth, insolation_model, A_B_, free_offset=True, **kw):
    """Binned truth minus the binned model_pcurve [ppm]; free_offset removes the mean (a fitted baseline). kw goes to model_pcurve (F for
    naive_redist, refl_unit, n_layers, ...)."""
    r = yb_truth - binned(cfg, model_pcurve(cfg, insolation_model, A_B_, **kw)[1])
    return r - r.mean() if free_offset else r


def truth_curves(cfg, A_B_true, F_true=None, **kw):
    """The noiseless extended-source truth model_pcurve(cfg, "hyperillum", A_B_true)[1], or one curve per detector of a set. F_true: a truth that
    redistributes heat ("hyperillum_redist" at that F); None is the bare rock."""
    model, kw = ("hyperillum", kw) if F_true is None else ("hyperillum_redist", {**kw, "F": float(F_true)})
    return [model_pcurve(c, model, A_B_true, **kw)[1] for c in cfg] if is_detector_set(cfg) else model_pcurve(cfg, model, A_B_true, **kw)[1]


def _chi2_problem(cfg, truth, insolation_model, free_offset=True, refl_unit=None, **kw):
    """The chi^2 problem of fitting `insolation_model` to the truth: resid(A_B_, F=None) [ppm], cost(A_B_, F=None) = sum((resid / sigma)^2) and
    sigma, the per-bin noise [ppm] (one sigma_bin per detector block). The reflected shape depends on neither A_B, F nor the band, so it is built
    once (from the first detector) unless refl_unit is passed. truth=None gives the bare model residual (truth = 0), all a derivative needs."""
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


def fit_AB_only(cfg, truth, insolation_model, free_offset=True, bounds=A_B_FIT_BOUNDS, xatol=1e-4, **kw):
    """Best-fit A_B of the one-parameter model `insolation_model` against a noiseless truth: (A_fit, chi2, resid_rms); the band albedo A follows from
    cfg.A_. "naive" against an extended-source truth is fit_naive; "hyperillum" against a point-source truth is the null test. chi2 is the squared
    distance between two noiseless models in units of the noise, not a goodness of fit."""
    resid, cost, _ = _chi2_problem(cfg, truth, insolation_model, free_offset=free_offset, **kw)
    res = minimize_scalar(cost, bounds=bounds, method="bounded", options={"xatol": xatol})
    r = resid(res.x)
    resid_rms = np.sqrt(np.mean(r ** 2))
    return res.x, res.fun, resid_rms


def fit_naive(cfg, truth, free_offset=True, bounds=A_B_FIT_BOUNDS, xatol=1e-4, **kw):
    """fit_AB_only with the point-source model "naive": (A_fit, chi2, resid_rms)."""
    return fit_AB_only(cfg, truth, "naive", free_offset=free_offset, bounds=bounds, xatol=xatol, **kw)


# F nodes of fit_redist's profile scan, as fractions of the F_bounds width: dense near F = 0, where the joint optimum usually sits (F ~ 0.005 for an
# extended-source truth).
F_SCAN_FRAC = np.array([0., 0.004, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 0.6, 1.])


def fit_redist(cfg, truth, free_offset=True, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, xatol=1e-5, F_xatol=1e-5, **kw):
    """Best-fit naive_redist (A_B, F) against a noiseless truth: (A_B_fit, F_fit, chi2, resid_rms), with the binning, noise weighting and offset of
    fit_naive. Profile fit: chi2_p(F) = min over A_B (bounded Brent), scanned on F_SCAN_FRAC to bracket its global minimum, then refined by
    bounded Brent in the bracket; a lower scan node wins, so F exactly on its bound is a possible result.
    Why not a 2-D local optimizer: the nightside is T ∝ F^(1/4), so the curve responds to F only ∝ F^~1.5 near 0 and dchi2/dF ≈ 0 on the F = 0
    edge. A simplex or gradient fit started there stays there. Profiling also makes the outer problem insensitive to the inner tolerance."""
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
    # bounded Brent never evaluates the bracket ends: keep the scan node (e.g. F on its bound) if it is at least as good
    if scan[k][0] <= chi2:
        F_fit, (chi2, A_B_fit) = F_scan[k], scan[k]
    r = resid(A_B_fit, F_fit)
    resid_rms = np.sqrt(np.mean(r ** 2))
    return A_B_fit, F_fit, chi2, resid_rms


def fit_chi2(cfg, truth, insolation_model, A_B_, F=None, free_offset=True, refl_unit=None, **kw):
    """chi^2 of one model against the truth, with the residual and noise weighting of the fits (F only for naive_redist)."""
    _, cost, _ = _chi2_problem(cfg, truth, insolation_model, free_offset=free_offset, refl_unit=refl_unit, **kw)
    return float(cost(A_B_, F))


def _chi2_redist_at(cfg, truth, A_B_, F, refl_unit, free_offset, kw):
    return fit_chi2(cfg, truth, "naive_redist", A_B_, F=F, free_offset=free_offset, refl_unit=refl_unit, **kw)


def chi2_landscape(cfg, truth, AB_grid, F_grid, free_offset=True, n_jobs=-1, **kw):
    """chi^2 of naive_redist over an (A_B, F) grid, shape (len(AB_grid), len(F_grid)), in parallel; for a detector set summed over the detectors."""
    c0 = template_cfg(cfg)
    refl_unit = reflected_unit(c0, "naive_redist", **kw) if c0.reflected else None   # band-independent, so the first detector's serves the set
    X = pmap(_chi2_redist_at, [(cfg, truth, a, f, refl_unit, free_offset, kw) for a in AB_grid for f in F_grid], n_jobs)
    return np.array(X).reshape(len(AB_grid), len(F_grid))


def chi2_region(d_chi2, AB_grid, F_grid, level=2.30):
    """Extent of the region d_chi2 < level on a landscape grid: ((A_B min, A_B max), F max)."""
    inside = d_chi2 < level
    return (AB_grid[inside.any(axis=1)].min(), AB_grid[inside.any(axis=1)].max()), F_grid[inside.any(axis=0)].max()


############################################################################################################################################
#####   Fits on a bound
############################################################################################################################################

# A railed fit: a fitted parameter within RAIL_TOL of a bound of its fit range, where the point-source model cannot reproduce the truth at any value
# inside the range, so the bias is undefined (bias_cell and false_F_fit return NaN, the sweeps flag the point). F = 0 is a valid fit: only F's upper
# bound counts. With A_B_FIT_BOUNDS the rail is a safety net; with bounds=A_B_BOUNDS the physical floor A_B = 0 is a rail.
RAIL_TOL = 1e-3


def is_railed(x, bounds, rail_tol=RAIL_TOL):
    """True where x sits within rail_tol of either bound (scalar or array)."""
    x = np.asarray(x, dtype=float)
    return ((x - bounds[0]) < rail_tol) | ((bounds[1] - x) < rail_tol)


def fit_railed(fit, bounds, F_bounds=F_BOUNDS, rail_tol=RAIL_TOL):
    """Rail test of one model's fit_truth result: A_B on either bound, or (joint fit) F on its upper bound."""
    return bool(is_railed(fit["A_B_fit"], bounds, rail_tol)) or ("F_fit" in fit and (F_bounds[1] - fit["F_fit"]) < rail_tol)
