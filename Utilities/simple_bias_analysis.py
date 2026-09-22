import contextlib, io
from scipy.optimize import minimize, minimize_scalar
from hyperillumination_phasecurve_utils import *
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


# The chi^2 problem of fitting `insolation_model` to the hyperillumination truth: resid(A_B_, F=None) [ppm] and cost(A_B_, F=None) = sum((resid / sigma_bin)^2).
# Shared by fit_naive, fit_redist, fit_chi2 and fisher_sigma. The reflected shape does not depend on A_B or F, so it is built once unless refl_unit is passed.
# truth=None gives the bare model residual (truth = 0), which is all a derivative needs.
def _chi2_problem(cfg, truth, insolation_model, free_offset=True, refl_unit=None, **kw):
    sigma_bin = cfg.sigma_bin
    if refl_unit is None and cfg.reflected: refl_unit = reflected_unit(cfg, insolation_model, **kw)
    yb_truth = 0. if truth is None else binned(cfg, truth)

    def resid(A_B_, F=None):
        return fit_residual(cfg, yb_truth, insolation_model, A_B_, F=F, free_offset=free_offset, refl_unit=refl_unit, **kw)

    def cost(A_B_, F=None):
        return np.sum((resid(A_B_, F) / sigma_bin) ** 2)

    return resid, cost


# Best-fit naive albedo A (= A_B) against a hyperillumination-generated truth, returning (A_fit, chi2, resid_rms).
def fit_naive(cfg, truth, free_offset=True, bounds=(0.0, 0.6), xatol=1e-4, **kw):
    resid, cost = _chi2_problem(cfg, truth, "naive", free_offset=free_offset, **kw)
    res = minimize_scalar(cost, bounds=bounds, method="bounded", options={"xatol": xatol})
    r = resid(res.x)
    resid_rms = np.sqrt(np.mean(r ** 2))
    return res.x, res.fun, resid_rms # chi2 = cost(A_fit) = weighted sum of squared residuals (this χ² is not a goodness-of-fit to an observation — it's the squared distance between two noiseless models, expressed in units of the observational noise)


# Best-fit naive_redist (A_B, F) against a hyperillumination-generated truth, returning (A_B_fit, F_fit, chi2, resid_rms): the 2-parameter version of fit_naive (same binning, noise weighting and offset handling).
# Nelder-Mead starts on the F = 0 slice (the truth has no redistribution) at A_B_start - by default the best F = 0 albedo; pass fit_naive's A_B to skip that extra 1-D fit.
def fit_redist(cfg, truth, free_offset=True, bounds=(0.0, 0.6), F_bounds=(0.0, 0.5), A_B_start=None, xatol=1e-4, **kw):
    resid, cost = _chi2_problem(cfg, truth, "naive_redist", free_offset=free_offset, **kw)
    if A_B_start is None:
        A_B_start = minimize_scalar(lambda A_B_: cost(A_B_, F_bounds[0]), bounds=bounds, method="bounded", options={"xatol": 1e-3}).x
    dA = 0.05 if A_B_start + 0.05 <= bounds[1] else -0.05   # initial simplex: one step in A_B, one in F
    simplex = [[A_B_start, F_bounds[0]], [A_B_start + dA, F_bounds[0]], [A_B_start, F_bounds[0] + 0.01]]
    res = minimize(lambda p: cost(*p), x0=simplex[0], method="Nelder-Mead", bounds=[bounds, F_bounds], options={"xatol": xatol, "initial_simplex": simplex})
    r = resid(*res.x)
    resid_rms = np.sqrt(np.mean(r ** 2))
    return res.x[0], res.x[1], res.fun, resid_rms


# chi^2 of one model against the hyperillumination truth, with the same residual and noise weighting as the fits (F only for naive_redist).
def fit_chi2(cfg, truth, insolation_model, A_B_, F=None, free_offset=True, refl_unit=None, **kw):
    _, cost = _chi2_problem(cfg, truth, insolation_model, free_offset=free_offset, refl_unit=refl_unit, **kw)
    return float(cost(A_B_, F))


def _chi2_redist_at(cfg, truth, A_B_, F, refl_unit, free_offset, kw):
    return fit_chi2(cfg, truth, "naive_redist", A_B_, F=F, free_offset=free_offset, refl_unit=refl_unit, **kw)


# chi^2 of naive_redist over an (A_B, F) grid, shape (len(AB_grid), len(F_grid)), evaluated in parallel. The reflected shape does not depend on F, so it is built once.
def chi2_landscape(cfg, truth, AB_grid, F_grid, free_offset=True, n_jobs=-1, **kw):
    refl_unit = reflected_unit(cfg, "naive_redist", **kw) if cfg.reflected else None
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
# The truth has F = 0, so F_fit is itself the bias in F. With both models the joint fit starts from the naive albedo; alone it starts from its own F = 0 slice.
# fisher=True adds the Fisher sigmas, evaluated at the FITS, where the misspecified model's posterior sits.
def fit_truth(cfg, A_B_true, models=("naive",), free_offset=True, bounds=(0.0, 0.6), F_bounds=(0.0, 0.5), fisher=False, **kw):
    _, truth = model_pcurve(cfg, "hyperillum", A_B_true, **kw)   # noiseless data
    out, A_B_start = {}, None
    if "naive" in models:
        A_fit, chi2, _ = fit_naive(cfg, truth, free_offset=free_offset, bounds=bounds, **kw)
        out["naive"] = dict(A_B_fit=A_fit, bias_A_B=A_fit - A_B_true, chi2=chi2)
        if fisher:
            out["naive"]["sigma_A_B"], = fisher_sigma(cfg, "naive", A_fit, bounds=bounds, **kw)   # 1 free parameter + baseline
        A_B_start = A_fit
    if "naive_redist" in models:
        A_fit, F_fit, chi2, _ = fit_redist(cfg, truth, free_offset=free_offset, bounds=bounds, F_bounds=F_bounds, A_B_start=A_B_start, **kw)
        out["naive_redist"] = dict(A_B_fit=A_fit, bias_A_B=A_fit - A_B_true, F_fit=F_fit, chi2=chi2)
        if fisher:   # 2 free parameters + baseline -> A_B's MARGINAL width grows with the F degeneracy
            out["naive_redist"]["sigma_A_B"], out["naive_redist"]["sigma_F"] = fisher_sigma(cfg, "naive_redist", A_fit, F=F_fit, bounds=bounds, F_bounds=F_bounds, **kw)
    return out


# The systematic in units of the statistical error bar, and the per-minute precision sigma_1min_crit at which |bias| = sigma.
# sigma_1min_crit = sigma_1min * |bias| / sigma is exact rather than an extrapolation: the bias does not depend on the noise, and the Fisher sigma is strictly proportional to sigma_1min.
def measurability(bias, sigma, sigma_1min):
    bias_sigma = np.abs(bias) / sigma
    return bias_sigma, sigma_1min * bias_sigma


# Std of the flat prior U(lo, hi) behind a fit's `bounds`: (hi - lo) / sqrt(12). A Fisher sigma at or above it means the data add nothing to the prior, so the Gaussian posterior the Fisher matrix assumes does not exist.
def flat_prior_sigma(bounds):
    return (bounds[1] - bounds[0]) / np.sqrt(12.)


# Three-way verdict of the bias test. bias_sigma < 1 alone reads as "naive OK", but with sigma >= the flat-prior std it only says the parameter is unmeasured — the bias is then divided by a meaningless number. That case is "unconstrained", not "naive OK".
VERDICT_NEEDS_HI, VERDICT_NAIVE_OK, VERDICT_UNCONSTRAINED = "needs hyperillum", "naive OK", "unconstrained"

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
def bias_vs_true_AB(cfg, A_B_grid, redist=True, bounds=(0., 1.), n_jobs=-1, **kw):
    models = ("naive", "naive_redist") if redist else ("naive",)
    rows = pmap(_fit_truth_at, [(cfg, a, models, bounds, kw) for a in A_B_grid], n_jobs)
    return {m: {k: np.array([r[m][k] for r in rows]) for k in rows[0][m]} for m in models}

# Calculate the Fisher sigma for A_B (and F)
# sigma_F gets truncated by the prior boundry F=0, so it is not trustworthy.
def fisher_sigma(cfg, insolation_model, A_B_, F=None, free_offset=True, h_A=1e-3, h_F=1e-3, bounds=(0.0, 1.0), F_bounds=(0.0, 0.5), refl_unit=None, **kw):
    # truth=None: the residual is (truth - model), so any constant truth cancels in a central difference
    resid, _ = _chi2_problem(cfg, None, insolation_model, free_offset=free_offset, refl_unit=refl_unit, **kw)

    lo, hi = max(A_B_ - h_A, bounds[0]), min(A_B_ + h_A, bounds[1])   # one-sided at a bound, so a railed fit still yields a derivative
    cols = [(resid(hi, F) - resid(lo, F)) / (hi - lo)]
    if F is not None:
        lo, hi = max(F - h_F, F_bounds[0]), min(F + h_F, F_bounds[1])   # F = 0 is a common fit result, so the step must not leave [0, 0.5]
        cols.append((resid(A_B_, hi) - resid(A_B_, lo)) / (hi - lo))
    J = np.column_stack(cols)
    try:
        return np.sqrt(np.diag(np.linalg.inv(J.T @ J / cfg.sigma_bin ** 2)))   # marginal sigmas = sqrt(diag(F^-1)); the inverse is what marginalises, 1/F_ii would instead fix the other parameters
    except np.linalg.LinAlgError:
        return np.full(J.shape[1], np.nan)   # singular J'J: a parameter the model does not respond to, or two that are exactly degenerate - NaN, like bias_cell on a rail


###########################################################################################################################
#####
#####   Bias Map
#####
###########################################################################################################################


# One grid cell of a bias map: generate the hyperillumination truth, fit the naive model, return the recovered-A_B bias.
# NaN if the fit railed against a bound, i.e. the naive model cannot reproduce the truth at ANY albedo.
# redist=True fits naive_redist (A_B and F) instead and returns (A_B bias, F_fit); the truth has no redistribution, so F_fit is itself the bias in F.
def bias_cell(cfg, A_B_true, free_offset=True, rail_tol=1e-3, temp_bias=False, bounds=(0.0, 0.6), redist=False, F_bounds=(0.0, 0.5), **kw):
    """Recovered-A_B bias for one map cell. NaN if the fit railed against a bound,
    i.e. the naive model cannot reproduce the truth at ANY albedo.
    With redist=True: (A_B bias, F_fit), both NaN on a rail. F_fit = 0 is a valid fit, only F's upper bound counts as a rail."""
    model = "naive_redist" if redist else "naive"
    fit = fit_truth(cfg, A_B_true, models=(model,), free_offset=free_offset, bounds=bounds, F_bounds=F_bounds, **kw)[model]   # same bounds the rail check below uses
    A_B_fit, F_fit = fit["A_B_fit"], fit.get("F_fit", 0.)
    lo, hi = bounds
    if (A_B_fit - lo) < rail_tol or (hi - A_B_fit) < rail_tol or (F_bounds[1] - F_fit) < rail_tol:
        return (np.nan, np.nan) if redist else np.nan
    A_B_bias = A_B_fit - A_B_true
    out = temperature_bias(A_B_true, A_B_bias, F_fit) if temp_bias else A_B_bias
    return (out, F_fit) if redist else out


def _bias_cell_at(cfg, A_B_true, a_rs, y_param, y, redist, kw):
    return bias_cell(cfg, A_B_true, a_rs=a_rs, redist=redist, **{y_param: y}, **kw)


# A whole bias map over (a_rs_grid, y_grid), cells in parallel. y_param is the swept geometry keyword of model_pcurve ("T_s" or "rp_rs").
# Returns the A_B-bias map, shape (len(y_grid), len(a_rs_grid)); with redist=True the pair (A_B-bias map, F map).
def bias_map_grid(cfg, A_B_true, a_rs_grid, y_grid, y_param, redist=False, n_jobs=-1, **kw):
    flat = pmap(_bias_cell_at, [(cfg, A_B_true, ar, y_param, y, redist, kw) for y in y_grid for ar in a_rs_grid], n_jobs)
    out = np.array(flat).reshape(len(y_grid), len(a_rs_grid), *((2,) if redist else ()))
    return tuple(np.moveaxis(out, -1, 0)) if redist else out


# Emission Spectroscopy Metric (Kempton et al. 2018) formatted for a legend entry; the archive column is NaN for a few planets.
def _fmt_esm(esm):
    return f"{esm:.1f}" if np.isfinite(esm) else "n/a"


# Loose name matching ("K2-141b" == "K2-141 b" == "k2_141_b"), the same normalisation HItargets_utils uses to look a planet up,
# so a PlanetParams.name can be matched against the archive's pl_name whichever spelling it carries.
def _norm_name(name):
    return str(name).lower().replace(" ", "").replace("-", "").replace("_", "")


# ESM of a single archive planet by name, for a target drawn outside overlay_planets (the notebook's own planet).
# One archive value for a planet by name, NaN where the archive has none or the value is missing: "ESM" for the tick labels / rows, "pl_rade" for the Rp row.
def planet_archive_value(AGG, pl_name, column):
    d = pd.read_csv(AGG).drop_duplicates("pl_name")
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


def map_grid(lo, hi, n, include=None, pad=0.1):
    """Linear grid over [lo, hi], widened (by `pad` of the original span) so that `include` - typically the target's own
    a/Rs or Rp/Rs - falls inside it. A planet outside the default box would otherwise simply be missing from its own map."""
    if include is not None and np.isfinite(include):
        span = hi - lo
        lo, hi = min(lo, include - pad * span), max(hi, include + pad * span)
    return np.linspace(lo, hi, n)


# The archive planets that fall inside a bias map (x = a/Rs = `a_over_r2`, y = `ykey`), restricted to ROCKY planets, Rp < `max_rade` Earth radii.
# The cut must be on the ABSOLUTE radius `pl_rade`, not on Rp/Rs: a giant around an evolved star has a small ratio (Kepler-91 b, Rp/Rs = 0.022 but 15.3 R_Earth) and would otherwise sneak in.
# Factored out of overlay_planets so the maps and the per-planet bar figures select from exactly one sample definition; `exclude` takes loose spellings.
def rocky_sample(AGG, ykey, ylim, a_rs_lim=(1.5, 4.0), max_rade=1.7, exclude=None):
    _exo = pd.read_csv(AGG)
    drop = [exclude] if isinstance(exclude, str) else list(exclude or [])
    keep = ~_exo.pl_name.map(_norm_name).isin({_norm_name(n) for n in drop})
    _phys = _exo[np.isfinite(_exo.a_over_r2) & np.isfinite(_exo.st_teff) & np.isfinite(_exo.pl_ratror)
                 & (_exo.pl_ratror > 0) & (_exo.pl_ratror < 1)
                 & np.isfinite(_exo.pl_rade) & (_exo.pl_rade < max_rade)   # rocky cut (NaN radius -> dropped, cannot be confirmed rocky)
                 & keep].drop_duplicates("pl_name").copy()
    x = "a_over_r2"
    return _phys[(_phys[x] >= a_rs_lim[0]) & (_phys[x] <= a_rs_lim[1])
                 & (_phys[ykey] >= ylim[0]) & (_phys[ykey] <= ylim[1])]


def overlay_planets(AGG, ax, ykey, ylim, a_rs_lim=(1.5, 4.0), n_highlight=10, max_rade=1.7, target=None, exclude=None, mark_target=True):
    """Overplot the rocky_sample planets that fall inside the current bias map (x = a/Rs = `a_over_r2`, y = `ykey`).
    The `n_highlight` planets with the smallest a/Rs get individual colours + legend labels (name, a/Rs, ESM); the rest are faint background dots.

    target      : the notebook's PlanetParams. It is dropped from the archive scatter (matched on its name) and, unless
                  `mark_target=False`, drawn as the green star by target_marker - so the highlighted planet follows
                  whatever `planet` the notebook defines instead of being fixed to one target.
    exclude     : extra name(s) to drop from the scatter; loose spellings are fine. Defaults to the target's own name.
    """

    drop = [exclude] if isinstance(exclude, str) else list(exclude or [])
    if target is not None: drop.append(target.name)   # the target is drawn separately, as the green star
    m = rocky_sample(AGG, ykey, ylim, a_rs_lim=a_rs_lim, max_rade=max_rade, exclude=drop)

    x = "a_over_r2"
    top = m.nsmallest(n_highlight, x)
    bg = m.drop(top.index)
    ax.scatter(bg[x], bg[ykey], s=16, facecolor="#40c4d6", edgecolor="w", linewidths=.3,
               alpha=.6, zorder=3, label=f"other known rocky planets ($R_p<{max_rade:g}\\,R_\\oplus$, N={len(bg)})")
    for color, (_, r) in zip(plt.cm.tab10(np.arange(n_highlight)), top.iterrows()):
        ax.scatter(r[x], r[ykey], s=60, color=color, edgecolor="w", linewidths=.7, zorder=5,
                   label=f"{r.pl_name}  ($a/R_\\star$={r[x]:.2f}, ESM={_fmt_esm(r.get('ESM', np.nan))})")
    if target is not None and mark_target: target_marker(ax, target, ykey, AGG)
    return m

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
    fig, axs = plt.subplots(1, 3, figsize=(21, 5.5))
    for a_, (y, y_naive, ylab, ttl) in zip(axs, panels):
        a_.axhline(0, color=AXIS, lw=.8)
        if y_naive is not None:
            a_.plot(A_B_grid, y_naive, "o--", color=C_NAIVE, lw=1.5, ms=5, label="$A_B$ only (naive)")
        a_.plot(A_B_grid, y, "o-", color=C_REDIST, lw=1.5, ms=5, label="$A_B$ and $F$ (naive_redist)")
        a_.plot(A_B_grid[i0], y[i0], label=target_label, **TARGET)
        a_.set_xlabel(r"true Bond albedo $A_B$"); a_.set_ylabel(ylab); a_.set_title(ttl); a_.legend()
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    plt.show()


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
    plt.show()


# Stacked bias maps on a shared a/R* axis: |Delta A_B| (top), the relative bias |Delta A_B|/A_B (middle) and, if F_map is given, the fitted F of the joint fit (bottom).
# Grey cells = "the model breaks down": railed fits (NaN). With F_map, geometries without an occultation are greyed out as well:
# once the impact parameter b = (a/R*) cos i reaches 1 + Rp/R*, nothing anchors the flux level and A_B, F become degenerate (the best fit is arbitrary along a flat chi^2 valley).
# The dashed lines then bound the grazing zone 1 - Rp/R* < b < 1 + Rp/R*. rp_rs_of(y) is Rp/R* at the map's y values; the inclination is the planet's.
def plot_bias_maps(xg, yg, bias_map, A_B_true, planet, AGG, ykey, ylabel, rp_rs_of, F_map=None, F_levels=(0.01, 0.02, 0.05)):
    cos_i = np.cos(np.radians(planet.incl))
    X, Y = np.meshgrid(xg, yg)
    no_occultation = (X * cos_i >= 1. + rp_rs_of(Y)) if F_map is not None else np.zeros(X.shape, bool)
    y_line = np.linspace(yg[0], yg[-1], 100)

    panels = [(np.abs(bias_map), "magma_r", [0.02, 0.05, 0.10], "%g", r"$|\Delta A_B|$", r"Absolute bias in recovered $A_B$"),
              (100. * np.abs(bias_map) / A_B_true, "magma_r", [5, 10, 20, 40], "%g %%", r"$|\Delta A_B|/A_B$ [%]", r"Relative bias in recovered $A_B$")]
    if F_map is not None:
        panels = [p[:5] + (p[5] + " (joint fit)",) for p in panels]
        panels.append((F_map, SEQ_BLUE, list(F_levels), "%g", r"fitted $F$", r"Fitted heat redistribution $F$ (truth: $F$ = 0)"))

    fig, axs = plt.subplots(len(panels), 1, figsize=(12, 6.2 * len(panels)), sharex=True)
    for ax, (z, cmap, levels, fmt, clabel, title) in zip(axs, panels):
        z = np.ma.masked_where(no_occultation | ~np.isfinite(z), z)
        im = ax.pcolormesh(xg, yg, z, cmap=cmap, shading="auto")
        cs = ax.contour(xg, yg, z, levels=levels, colors=["w" if im.norm(lv) > 0.5 else INK for lv in levels], linewidths=.8)   # contour ink that stays readable on light and dark cells
        ax.clabel(cs, fmt=fmt)
        if F_map is not None:
            for sign in (-1., 1.):
                ax.plot((1. + sign * rp_rs_of(y_line)) / cos_i, y_line, color=INK, ls="--", lw=1.2, label=r"grazing occultation ($b = 1 \mp R_p/R_\star$)" if sign > 0 else None)
        # known ROCKY planets (NASA Exoplanet Archive, Rp < 1.7 R_Earth), labelled with a/Rs and ESM; target=planet draws the notebook's planet as the green star
        overlay_planets(AGG, ax, ykey, (yg[0], yg[-1]), a_rs_lim=(xg[0], xg[-1]), target=planet)
        ax.set_facecolor("0.85")
        ax.grid(False)
        ax.set_xlim(xg[0], xg[-1]); ax.set_ylim(yg[0], yg[-1])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label=clabel)
    axs[-1].set_xlabel(r"$a/R_\star$")
    fig.suptitle(f"Biases from neglecting hyperillumination ({planet.name}, $A_B$ = {A_B_true:.2f})")
    fig.tight_layout()
    fig.legend(*axs[0].get_legend_handles_labels(), loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, framealpha=.9)
    plt.show()

    ok = ~no_occultation & np.isfinite(bias_map)
    line = (f"{'occulting, ' if F_map is not None else ''}non-railed cells: {ok.sum()} of {ok.size};  Delta A_B in [{bias_map[ok].min():+.3f}, {bias_map[ok].max():+.3f}]"
            f" = [{100 * bias_map[ok].min() / A_B_true:+.1f}, {100 * bias_map[ok].max() / A_B_true:+.1f}] % of A_B")
    if F_map is not None:
        line += f",  F in [{F_map[ok].min():.4f}, {F_map[ok].max():.4f}]"
    print(line)
    print(f"railed: {(~no_occultation & ~np.isfinite(bias_map)).sum()}" + (f",  without occultation: {no_occultation.sum()}" if F_map is not None else ""))


# One row of planet_bias_table: the planet's own geometry and its own sigma_1min, everything else inherited from cfg.
def _planet_bias_row(cfg, A_B_true, pl, s1, redist, bounds, F_bounds, kw):
    c = replace(cfg, planet=pl, sigma_1min=float(s1))   # stellar_grid / filter / A_ / n_bins / n_orbits are instrument-level, not per planet
    model = "naive_redist" if redist else "naive"
    fit = fit_truth(c, A_B_true, models=(model,), free_offset=True, bounds=bounds, F_bounds=F_bounds, fisher=True, **kw)[model]
    return dict(name=pl.name, a_rs=pl.R / pl.r2, period=pl.period, T_eff=pl.T2_eff, bias_A_B=fit["bias_A_B"], sigma_A_B=fit["sigma_A_B"],
                F_fit=fit.get("F_fit", np.nan), sigma_F=fit.get("sigma_F", np.nan), sigma_1min=float(s1))


# Bias, Fisher sigma and critical noise for a SAMPLE of planets on one instrument, sorted by a/Rs.
# planets: {name: PlanetParams} or a list of PlanetParams. sigma_1min: {name: ppm/min}, e.g. from HItargets_utils.rms_1min_sample - NaN entries propagate instead of raising.
# quantity picks which bias the decision columns (bias, sigma, bias_sigma, sigma_1min_crit) refer to: "A_B" (default, also for the joint fit) or "F" (requires redist=True; the truth has F = 0, so F_fit IS the bias).
# A_B_true is the SAME assumed albedo for every planet - a notebook assumption, not a per-planet measurement.
# sigma_1min_crit = sigma_1min * |bias| / sigma is exact rather than an extrapolation: the bias does not depend on the noise, and the Fisher sigma is strictly proportional to sigma_1min.
# verdict (bias_verdict): "needs hyperillum" / "naive OK" / "unconstrained" — the last where sigma >= the flat-prior std of `bounds` (A_B) or `F_bounds` (F), i.e. where the fit does not measure the quantity at all.
# AGG: archive aggregate path; with it the table carries each planet's ESM (`esm`, NaN where the archive has none) for the figure's tick labels.
def planet_bias_table(cfg, A_B_true, planets, sigma_1min, redist=False, quantity=None, bounds=(0.0, 1.0), F_bounds=(0.0, 0.5), AGG=None, n_jobs=-1, **kw):
    quantity = quantity or "A_B"
    if quantity not in ("A_B", "F"): raise ValueError(f'quantity must be "A_B" or "F", got {quantity!r}')
    if quantity == "F" and not redist: raise ValueError('quantity="F" needs redist=True (only the joint fit has an F)')
    pls = list(planets.values()) if isinstance(planets, dict) else list(planets)

    rows = pmap(_planet_bias_row, [(cfg, A_B_true, pl, sigma_1min.get(pl.name, np.nan), redist, bounds, F_bounds, kw) for pl in pls], n_jobs)   # a planet PandExo could not do keeps its bias but gets no sigma_1min_crit
    tab = pd.DataFrame(rows).sort_values("a_rs").reset_index(drop=True)
    tab["esm"] = [planet_esm(AGG, nm) for nm in tab["name"]] if AGG is not None else np.nan
    tab["rp_rearth"] = [planet_archive_value(AGG, nm, "pl_rade") for nm in tab["name"]] if AGG is not None else np.nan   # [R_Earth], for the figure's rows only
    tab["t_obs"] = cfg.n_orbits * tab["period"] * 24.   # [h] total observation: n_orbits full orbits, so it differs per planet at fixed n_orbits

    tab.attrs["quantity"], tab.attrs["sigma_prior"] = quantity, {"A_B": flat_prior_sigma(bounds), "F": flat_prior_sigma(F_bounds)}   # per quantity, so the figure can re-derive the verdict for either
    tab["bias"], tab["sigma"] = tab[f"bias_{quantity}" if quantity == "A_B" else "F_fit"], tab[f"sigma_{quantity}"]
    tab["bias_sigma"], tab["sigma_1min_crit"] = measurability(tab["bias"], tab["sigma"], tab["sigma_1min"])   # at each planet's own sigma_1min
    tab["verdict"] = bias_verdict(tab["bias_sigma"], tab["sigma"], tab.attrs["sigma_prior"][quantity])
    tab["t_crit"] = critical_time(tab["t_obs"], tab["bias_sigma"], tab["verdict"])   # [h] at which the bias is a 1 sigma effect; NaN where unconstrained
    return tab


# The exact value written at the end of each bar; below the bar for negative values, skipped where the value is NaN.
def _bar_values(ax, bars, fmt, values=None, fontsize=8):
    values = [b.get_height() for b in bars] if values is None else list(values)   # `values` prints a different number than the bar height, e.g. the SIGNED bias on an |bias| bar
    for b, v in zip(bars, values):
        h = b.get_height()
        if not (np.isfinite(h) and np.isfinite(v)): continue
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, h), textcoords="offset points",
                    xytext=(0, 3 if h >= 0 else -11), ha="center", va="bottom" if h >= 0 else "top", fontsize=fontsize)


# Per-planet decision figure for planet_bias_table, one bar group per planet. For each quantity in `quantities` (default: the table's own) two panels:
#   bias and its Fisher sigma      delivered sigma_1min vs the required sigma_1min_crit
# The x tick labels carry each planet's a/Rs and (with `esm` in the table) ESM in brackets. In a noise panel a planet needs hyperillumination exactly when the DELIVERED bar is at or below the REQUIRED one - that is bias >= 1 sigma.
# The verdict (bias_verdict) is written above every bar group. "unconstrained" planets - Fisher sigma at or above the flat-prior std (dashed line in the bias panel) - get hatched bars in both panels: their bias / sigma and sigma_1min_crit are ratios to a meaningless sigma and must not be read as "naive OK".
# quantities=("F", "A_B") needs a redist=True table (it carries both fits' columns); the decision columns and the verdict are recomputed per quantity, so the table's own `quantity` does not matter.
# log_noise: sigma_1min spans the sample's brightness range, so the noise panels are logarithmic by default. log_bias does the same for the bias panels, where an unconstrained sigma (~1) would otherwise flatten the measurable ones (~0.05). bias_label / sigma_label override the first quantity's legend labels.
_BAR_Q = {"A_B": ("bias_A_B", "sigma_A_B", r"bias $\Delta A_B$", r"$\sigma_{A_B}^{\rm Fisher}$", r"$|\Delta A_B|$ and $\sigma_{A_B}$", "$A_B$"),
          "F":   ("F_fit",    "sigma_F",   r"spurious $F$",      r"$\sigma_F^{\rm Fisher}$",     r"$F_{\rm fit}$ and $\sigma_F$",     "$F$")}
_VERDICT_STYLE = {VERDICT_NEEDS_HI: dict(color=C_HI, weight="bold"), VERDICT_NAIVE_OK: dict(color=C_NAIVE), VERDICT_UNCONSTRAINED: dict(color=MUTED, style="italic")}
_HATCH = "///"   # bars of unconstrained planets


# Parameter rows under the bottom panel of plot_planet_bias_bars: one text cell per planet per row, aligned with the bar groups (x in data units, y in axes units below the tick labels).
# rows: [(label, values, fmt)], top row first; a non-finite value prints "-". Returns the axes-fraction the rows extend below the axes, so the caller can reserve the space.
def _param_rows(ax, x, rows, y0=-0.10, dy=0.075, fontsize=8):
    for k, (label, vals, fmt) in enumerate(rows):
        y = y0 - k * dy
        ax.text(x[0] - 0.55, y, label, transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=fontsize, color=MUTED)
        for xi, v in zip(x, vals):
            ax.text(xi, y, fmt.format(v) if np.isfinite(v) else "-", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=fontsize)
    return -y0 + len(rows) * dy


def plot_planet_bias_bars(tab, bias_label=None, sigma_label=None, fmt_bias="{:+.4f}", fmt_sigma="{:.4f}", fmt_noise="{:.0f}",
                          figsize=None, log_noise=True, log_bias=False, colors=(None, None), title=None, quantities=None, grid=True, param_rows=True):
    quantities = tuple(quantities or (tab.attrs.get("quantity", "A_B"),))
    c_bias, c_sig = (colors[0] or C_REDIST), (colors[1] or C_NAIVE)

    x, w = np.arange(len(tab)), 0.38
    n = 2 * len(quantities)
    # param_rows: a/Rs, ESM, Rp, P, t_obs, Teff and one t_crit per quantity under the bottom panel (0.26 in each), and the tick label is then the bare name
    rows = [(r"$a/R_\star$", tab["a_rs"], "{:.2f}"), ("ESM", tab["esm"], "{:.1f}"), (r"$R_p$ [$R_\oplus$]", tab["rp_rearth"], "{:.2f}"),
            ("$P$ [h]", tab["period"] * 24., "{:.2f}"), (r"$t_{\rm obs}$ [h]", tab["t_obs"], "{:.1f}"), (r"$T_{\rm eff}$ [K]", tab["T_eff"], "{:.0f}")] if param_rows else []
    fig, ax = plt.subplots(n, 1, figsize=figsize or (13, 3.7 * n + 0.8 + 0.26 * (len(rows) + len(quantities) * param_rows)), sharex=True)

    for k, q in enumerate(quantities):
        bcol, scol, blab, slab, ylab, name = _BAR_Q[q]
        if k == 0: blab, slab = bias_label or blab, sigma_label or slab
        bias, sigma = tab[bcol], tab[scol]
        bias_sigma, crit = measurability(bias, sigma, tab["sigma_1min"])   # at each planet's own sigma_1min
        sigma_prior = tab.attrs["sigma_prior"][q]
        verdict = bias_verdict(bias_sigma, sigma, sigma_prior)
        if param_rows: rows.append((rf"$t_{{\rm crit}}$({name}) [h]", critical_time(tab["t_obs"], bias_sigma, verdict), "{:.1f}"))   # "-" where the quantity is unconstrained
        hatch = [_HATCH if v == VERDICT_UNCONSTRAINED else None for v in verdict]
        a0, a1 = ax[2 * k], ax[2 * k + 1]
        pre = f"{name}: " if len(quantities) > 1 else ""   # the panel pairs need telling apart only when there are several

        b1 = a0.bar(x - w / 2, np.abs(bias), w, color=c_bias, hatch=hatch, label=blab)   # |bias|, so the two bars are directly comparable in height; the annotation keeps the sign
        b2 = a0.bar(x + w / 2, sigma, w, color=c_sig, hatch=hatch, label=slab)
        a0.axhline(sigma_prior, color=MUTED, ls="--", lw=1., label=f"flat-prior std {sigma_prior:.3f}  (Fisher $\\sigma$ above it: {name} unconstrained)")
        _bar_values(a0, b1, fmt_bias, values=bias); _bar_values(a0, b2, fmt_sigma)
        for xi, v in zip(x, verdict): a0.text(xi, 1.01, v, transform=a0.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, **_VERDICT_STYLE[v])   # the verdict row sits between the axes and the title (axes-height units, so independent of the y scale and clear of the legend)
        if log_bias: a0.set_yscale("log")
        a0.set_ylabel(ylab)
        a0.set_title(f"{pre}bias vs. Fisher uncertainty  (bias is measurable where the {blab} bar exceeds the {slab} bar; hatched: {name} not measured at all)", fontsize=11, pad=16)

        b3 = a1.bar(x - w / 2, tab["sigma_1min"], w, color=C_HI, hatch=hatch, label=r"delivered $\sigma_{1\rm min}$")
        b4 = a1.bar(x + w / 2, crit, w, color=MUTED, hatch=hatch, label=r"required $\sigma_{1\rm min, crit}$  (bias $=1\,\sigma$)")
        _bar_values(a1, b3, fmt_noise); _bar_values(a1, b4, fmt_noise)
        if log_noise: a1.set_yscale("log")
        a1.set_ylabel("[ppm / min]")
        a1.set_title(f"{pre}delivered vs. required photometric precision - hyperillumination matters where delivered $\\leq$ required (hatched: required is meaningless)", fontsize=11)
        for a_ in (a0, a1): a_.legend(fontsize=9, loc="upper left")

    for a_ in ax:
        a_.margins(y=.18)   # headroom for the value annotations
        if not grid: a_.grid(False)
    ax[-1].set_xticks(x)
    if param_rows:
        ax[-1].set_xticklabels(tab["name"], fontsize=9)
        _param_rows(ax[-1], x, rows)
    else:
        esm = tab["esm"] if "esm" in tab else np.full(len(tab), np.nan)
        ax[-1].set_xticklabels([f"{nm} \n(ars {a:.2f}, ESM {_fmt_esm(e)})" for nm, a, e in zip(tab["name"], tab["a_rs"], esm)], rotation=30, ha="right")
    if title: fig.suptitle(title)
    fig.tight_layout()   # includes the row texts (they are axes artists), so the bottom margin grows with the row count
    return fig, ax
