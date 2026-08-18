import contextlib, io
from scipy.optimize import minimize_scalar
from hyperillumination_phasecurve_utils import *
import pandas as pd
import matplotlib.pyplot as plt


# A_B bias converted to a bias in local surface temperature
def temperature_bias(A_B_true, A_B_bias):
    A_fit = A_B_true + A_B_bias
    return ((1 - A_fit) / (1 - A_B_true))**0.25 - 1

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


# Best-fit naive albedo A (= A_B) against a hyperillumination-generated truth, returning (A_fit, chi2).
def fit_naive(cfg, truth, free_offset=True, bounds=(0.0, 0.6), xatol=1e-4, **kw):
    sigma_bin = cfg.sigma_bin

    refl_unit = reflected_unit(cfg, "naive", **kw) if cfg.reflected else None

    yb_truth = binned(cfg, truth)

    def resid(A_B_):
        r = yb_truth - binned(cfg, model_lc(cfg, "naive", A_B_, refl_unit=refl_unit, **kw)[1])
        return r - r.mean() if free_offset else r

    def cost(A_B_):
        return np.sum((resid(A_B_) / sigma_bin) ** 2)

    res = minimize_scalar(cost, bounds=bounds, method="bounded", options={"xatol": xatol})
    r = resid(res.x)
    resid_rms = np.sqrt(np.mean(r ** 2))
    return res.x, res.fun, resid_rms # chi2 = cost(A_fit) = weighted sum of squared residuals (this χ² is not a goodness-of-fit to an observation — it's the squared distance between two noiseless models, expressed in units of the observational noise)


###########################################################################################################################
#####
#####   Bias Map
#####
###########################################################################################################################


# One grid cell of a bias map: generate the hyperillumination truth, fit the naive model, return the recovered-A_B bias.
# NaN if the fit railed against a bound, i.e. the naive model cannot reproduce the truth at ANY albedo.
def bias_cell(cfg, A_B_true, free_offset=True, rail_tol=1e-3, temp_bias=False, bounds=(0.0, 0.6), **kw):
    """Recovered-A_B bias for one map cell. NaN if the fit railed against a bound,
    i.e. the naive model cannot reproduce the truth at ANY albedo."""
    _, truth = model_lc(cfg, "hyperillum", A_B_true, **kw) # noiseless data
    A_B_fit, _, _ = fit_naive(cfg, truth, free_offset=free_offset, bounds=bounds, **kw)  # same bounds the rail check below uses
    lo, hi = bounds
    if (A_B_fit - lo) < rail_tol or (hi - A_B_fit) < rail_tol:
        return np.nan
    A_B_bias = A_B_fit - A_B_true
    if temp_bias: return temperature_bias(A_B_true, A_B_bias)
    else: return A_B_bias


# Emission Spectroscopy Metric (Kempton et al. 2018) formatted for a legend entry; the archive column is NaN for a few planets.
def _fmt_esm(esm):
    return f"{esm:.1f}" if np.isfinite(esm) else "n/a"

# ESM of a single archive planet by name, for targets drawn outside overlay_planets (e.g. the K2-141b reference marker).
def planet_esm(AGG, pl_name):
    row = pd.read_csv(AGG).drop_duplicates("pl_name").pipe(lambda d: d[d.pl_name == pl_name])
    return float(row.ESM.iloc[0]) if len(row) else np.nan


def overlay_planets(AGG, ax, ykey, ylim, a_rs_lim=(1.5, 4.0), n_highlight=10, max_rade=1.6, exclude="K2-141 b"):
    """Overplot archive planets that fall inside the current bias map (x = a/Rs = `a_over_r2`, y = `ykey`).
    Restricted to rocky planets, Rp < `max_rade` Earth radii (default 1.6, Rogers 2015: above this most planets are not rocky).
    The cut must be on the ABSOLUTE radius `pl_rade`, not on Rp/Rs: a giant around an evolved star has a small ratio
    (e.g. Kepler-91 b, Rp/Rs = 0.022 but 15.3 R_Earth) and would otherwise sneak into the map.
    The `n_highlight` planets with the smallest a/Rs get individual colours + legend labels (name, a/Rs, ESM); the rest are faint background dots.
    """

    _exo = pd.read_csv(AGG)
    # physical ROCKY planets only: finite a/Rs, T_eff, Rp/Rs and Rp/Rs < 1, plus a known radius below the rocky threshold.
    # K2-141b is drawn separately.
    _phys = _exo[np.isfinite(_exo.a_over_r2) & np.isfinite(_exo.st_teff) & np.isfinite(_exo.pl_ratror)
                 & (_exo.pl_ratror > 0) & (_exo.pl_ratror < 1)
                 & np.isfinite(_exo.pl_rade) & (_exo.pl_rade < max_rade)   # rocky cut (NaN radius -> dropped, cannot be confirmed rocky)
                 & (_exo.pl_name != exclude)].drop_duplicates("pl_name").copy()

    x = "a_over_r2"
    m = _phys[(_phys[x] >= a_rs_lim[0]) & (_phys[x] <= a_rs_lim[1])
              & (_phys[ykey] >= ylim[0]) & (_phys[ykey] <= ylim[1])]
    top = m.nsmallest(n_highlight, x)
    bg = m.drop(top.index)
    ax.scatter(bg[x], bg[ykey], s=16, facecolor="#40c4d6", edgecolor="w", linewidths=.3,
               alpha=.6, zorder=3, label=f"other known rocky planets ($R_p<{max_rade:g}\\,R_\\oplus$, N={len(bg)})")
    for color, (_, r) in zip(plt.cm.tab10(np.arange(n_highlight)), top.iterrows()):
        ax.scatter(r[x], r[ykey], s=60, color=color, edgecolor="w", linewidths=.7, zorder=5,
                   label=f"{r.pl_name}  ($a/R_\\star$={r[x]:.2f}, ESM={_fmt_esm(r.get('ESM', np.nan))})")
    return m