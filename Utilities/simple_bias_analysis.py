from dataclasses import dataclass, replace
from collections.abc import Callable
import numpy as np
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):   # mute "looking for spidermanrc..." in every process
    import fspiderman as sp
from scipy.optimize import minimize_scalar
from hyperillumination_phasecurve_utils import (custom_temperature_grid, reflected_PhaseCurve, fast_lightcurve)
import pandas as pd
import matplotlib.pyplot as plt


###########################################################################################################################
#####
#####   Detectability
#####
###########################################################################################################################

# Average y in fixed-width time bins. t in days, minutes_binning in minutes.
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

@dataclass
class BiasConfig:
    """Fixed physical + instrument context for the bias analysis."""
    planet: object          # PlanetParams for the reference system (e.g. K2-141b); single source of
                            # truth for geometry (r1, r2, R, a_AU), orbit (period, incl, ecc, w) and T2_eff
    stellar_grid: object    # sp.stellar_grid.gen_grid(...) output for the chosen filter
    filter_path: str        # SPIDERMAN throughput file
    l1: float               # filter blue edge [m]
    l2: float               # filter red edge [m]
    A_B_true: float         # true Bond albedo used to generate the hyperillumination truth
    A_: Callable = None     # band integrated albedo as function of bond albedo (only needed when reflected=True)
    sigma_1min: float = 323.  # detection error [ppm per minute of integration]
    n_orbits: int = 3  # number of orbits observed and phase-folded
    n_bins: int = 27  # bins per period; the width is DERIVED so it tiles exactly
    u_B: float = 0.         # hyperillumination-model limb-darkening coefficient (0 = uniform disk)
    reflected: bool = True  # True -> full (reflected + thermal); False -> thermal only
    T_floor: float = 200.   # min temperature for the direct_T grid [K]; blocks negative-T spline overshoot
    bounds: tuple = (0.0, 0.6)  # albedo search bounds for the naive fit

# Exact block-average bins (valid only when n_t is a multiple of n_bins)
def binned(cfg, y):
    k = len(y) // cfg.n_bins
    if k * cfg.n_bins != len(y):
        raise ValueError(f"n_t={len(y)} is not a multiple of n_bins={cfg.n_bins}; "
                         f"use n_t = {cfg.n_bins} * k (e.g. {cfg.n_bins * 40})")
    return y.reshape(cfg.n_bins, k).mean(axis=1)

# Reflected phase-curve shape at unit albedo (A = 1)
def reflected_unit(cfg, insolation_model, a_rs=None, T_s=None, rp_=None, n_t=None, **_):
    pl = cfg.planet
    a_rs = pl.R / pl.r2  if a_rs is None else a_rs   # default a/Rs from PlanetParams
    T_s = pl.T2_eff      if T_s is None else T_s
    rp_ = pl.r1 / pl.r2  if rp_ is None else rp_      # default Rp/Rs from PlanetParams
    n_t = 50 * cfg.n_bins if n_t is None else n_t

    t_ = np.linspace(0., pl.period, int(n_t), endpoint=False)
    # set the ratios in the planet's own units (reflected is geometric -> a_AU unused, left as-is)
    planet = replace(pl, r1=rp_ * pl.r2, R=a_rs * pl.r2, T2_eff=T_s)
    _, refl, _ = reflected_PhaseCurve(planet, t=t_, include_transit=False, A=1.,
                                      u_B=cfg.u_B, insolation_model=insolation_model)
    return refl * 1e6


# Fast, stripped-down phase curve [ppm] for the fitting loop
def model_lc(cfg, insolation_model, A_B_, a_rs=None, T_s=None, rp_=None, n_t=None, n_layers=40, n_lo=300, n_la=150, refl_unit=None):
    pl = cfg.planet
    a_rs0 = pl.R / pl.r2 # default a/Rs (for a_AU scaling)
    a_rs = a_rs0 if a_rs is None else a_rs
    T_s = pl.T2_eff if T_s is None else T_s
    rp_ = pl.r1 / pl.r2 if rp_ is None else rp_
    n_t = 50 * cfg.n_bins if n_t is None else n_t

    t_ = np.linspace(0., pl.period, int(n_t), endpoint=False)
    # rebuild the planet at the swept geometry in pl's own units; a_AU scales with a/Rs (stellar size fixed)
    planet = replace(pl, r1=rp_ * pl.r2, R=a_rs * pl.r2, T2_eff=T_s, a_AU=pl.a_AU * a_rs / a_rs0)

    # Thermal Part
    lo, la, Tg = custom_temperature_grid(planet, u_B=cfg.u_B, n_lo=n_lo, n_la=n_la, insolation_model=insolation_model, A_B=A_B_)
    LO, LA = np.meshgrid(lo, la, indexing="ij")
    p = sp.ModelParams(brightness_model="direct_T", stellar_model="blackbody")
    p.n_layers = n_layers
    p.t0, p.per = 0., planet.period
    p.a_abs = planet.a_AU
    p.inc, p.ecc, p.w = planet.incl, planet.ecc, planet.w
    p.rp, p.a = planet.r1 / planet.r2, planet.R / planet.r2
    p.p_u1 = p.p_u2 = 0.
    p.T_s, p.l1, p.l2 = T_s, cfg.l1, cfg.l2
    p.grid = sp.format_grid(LO, LA, np.maximum(Tg, cfg.T_floor).T)
    p.filter, p.eclipse = cfg.filter_path, True
    thermal = (fast_lightcurve(p, cfg.stellar_grid, t_, n_phase=24) - 1) * 1e6
    if not cfg.reflected:
        return t_, thermal

    # Reflected Part (linear in A)
    A = cfg.A_(A_B_)
    if refl_unit is None:
        refl_unit = reflected_unit(cfg, insolation_model, a_rs=a_rs, T_s=T_s, rp_=rp_, n_t=n_t)

    return t_, thermal + A * refl_unit


# Best-fit naive albedo A (= A_B) against a hyperillumination-generated truth, returning (A_fit, chi2).
def fit_naive(cfg, truth, free_offset=True, bounds=None, xatol=1e-4, **kw):
    sigma_bin = cfg.sigma_1min / np.sqrt(cfg.n_orbits * cfg.planet.period * 24 * 60 / cfg.n_bins)

    bounds = cfg.bounds if bounds is None else bounds
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
    return res.x, res.fun, resid_rms # chi2 = cost(A_fit) = weighted sum of squared residuals


# One grid cell of a bias map: generate the hyperillumination truth, fit the naive model, return the recovered-A_B bias.
# NaN if the fit railed against a bound, i.e. the naive model cannot reproduce the truth at ANY albedo.
def bias_cell(cfg, free_offset=True, rail_tol=1e-3, temp_bias=False, **kw):
    """Recovered-A_B bias for one map cell. NaN if the fit railed against a bound,
    i.e. the naive model cannot reproduce the truth at ANY albedo."""
    _, truth = model_lc(cfg, "hyperillum", cfg.A_B_true, **kw)
    A_B_fit, _, _ = fit_naive(cfg, truth, free_offset=free_offset, **kw)
    lo, hi = cfg.bounds
    if (A_B_fit - lo) < rail_tol or (hi - A_B_fit) < rail_tol:
        return np.nan
    A_B_bias = A_B_fit - cfg.A_B_true
    if temp_bias: return temperature_bias(cfg, A_B_bias)
    else: return A_B_bias

# A_B bias converted to a bias in local surface temperature
def temperature_bias(cfg, A_B_bias):
    A_fit = cfg.A_B_true + A_B_bias
    return ((1 - A_fit) / (1 - cfg.A_B_true))**0.25 - 1


# Emission Spectroscopy Metric (Kempton et al. 2018) formatted for a legend entry; the archive column is NaN for a few planets.
def _fmt_esm(esm):
    return f"{esm:.1f}" if np.isfinite(esm) else "n/a"

# ESM of a single archive planet by name, for targets drawn outside overlay_planets (e.g. the K2-141b reference marker).
def planet_esm(AGG, pl_name):
    row = pd.read_csv(AGG).drop_duplicates("pl_name").pipe(lambda d: d[d.pl_name == pl_name])
    return float(row.ESM.iloc[0]) if len(row) else np.nan


def overlay_planets(AGG, ax, ykey, ylim, a_rs_lim=(1.5, 4.0), n_highlight=10, max_rade=1.6):
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
                 & (_exo.pl_name != "K2-141 b")].drop_duplicates("pl_name").copy()

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