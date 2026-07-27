from dataclasses import dataclass, replace
import numpy as np
import spiderman as sp
from scipy.optimize import minimize_scalar
from hyperillumination_phasecurve_utils import (custom_temperature_grid,
                                                reflected_PhaseCurve, fast_lightcurve)


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
#####   A_B-Bias fitting model
#####
###########################################################################################################################

@dataclass
class BiasConfig:
    """Fixed physical + instrument context for the bias analysis."""
    planet: object          # PlanetParams for the reference system (e.g. K2-141b)
    stellar_grid: object    # sp.stellar_grid.gen_grid(...) output for the chosen filter
    filter_path: str        # SPIDERMAN throughput file
    l1: float               # filter blue edge [m]
    l2: float               # filter red edge [m]
    rs: float               # stellar radius [R_sun]
    incl: float             # orbital inclination [deg]
    ecc: float              # eccentricity
    w: float                # argument of periastron [deg]
    period: float           # orbital period [days]
    a: float                # default semi-major axis [stellar radii]
    T_eff: float            # default stellar effective temperature [K]
    rp: float               # default Rp/Rs
    A_B_true: float         # true Bond albedo used to generate the hyperillumination truth
    sigma_bin: float        # per-bin detection error [ppm]
    reflected: bool = True  # True -> full (reflected + thermal); False -> thermal only
    T_floor: float = 200.   # min temperature for the direct_T grid [K]; blocks negative-T spline overshoot
    bounds: tuple = (0.0, 0.6)  # albedo search bounds for the naive fit


# Reflected phase-curve shape at unit albedo (A = 1)
def reflected_unit(cfg, insolation_model, a_rs=None, T_s=None, rp_=None, n_t=1000, **_):
    a_rs = cfg.a if a_rs is None else a_rs
    T_s = cfg.T_eff if T_s is None else T_s
    rp_ = cfg.rp if rp_ is None else rp_
    t_ = np.linspace(0., cfg.period, int(n_t))
    planet = replace(cfg.planet, R=a_rs, T2_eff=T_s, r1=rp_)
    _, refl, _ = reflected_PhaseCurve(planet, t=t_, include_transit=False, A=1.,
                                      insolation_model=insolation_model)
    return refl * 1e6


# Fast, stripped-down phase curve [ppm] for the fitting loop
def model_lc(cfg, insolation_model, A_B_, a_rs=None, T_s=None, rp_=None, n_t=1000, n_layers=40, n_lo=300, n_la=150, refl_unit=None):
    a_rs = cfg.a if a_rs is None else a_rs
    T_s = cfg.T_eff if T_s is None else T_s
    rp_ = cfg.rp if rp_ is None else rp_

    t_ = np.linspace(0., cfg.period, int(n_t))
    planet = replace(cfg.planet, R=a_rs, T2_eff=T_s, r1=rp_)

    # Thermal Part
    lo, la, Tg = custom_temperature_grid(planet, n_lo=n_lo, n_la=n_la,
                                         insolation_model=insolation_model, A_B=A_B_)
    LO, LA = np.meshgrid(lo, la, indexing="ij")
    p = sp.ModelParams(brightness_model="direct_T", stellar_model="blackbody")
    p.n_layers = n_layers
    p.t0, p.per = 0., cfg.period
    p.a_abs = a_rs * cfg.rs * 6.957e8 / 1.495978707e11
    p.inc, p.ecc, p.w = cfg.incl, cfg.ecc, cfg.w
    p.rp, p.a = rp_, a_rs
    p.p_u1 = p.p_u2 = 0.
    p.T_s, p.l1, p.l2 = T_s, cfg.l1, cfg.l2
    p.grid = sp.format_grid(LO, LA, np.maximum(Tg, cfg.T_floor).T)
    p.filter, p.eclipse = cfg.filter_path, True
    thermal = (fast_lightcurve(p, cfg.stellar_grid, t_, n_phase=24) - 1) * 1e6
    if not cfg.reflected:
        return t_, thermal

    # Reflected Part (linear in A=A_B)
    if refl_unit is None:
        refl_unit = reflected_unit(cfg, insolation_model, a_rs=a_rs, T_s=T_s, rp_=rp_, n_t=n_t)
    return t_, thermal + A_B_ * refl_unit


# Best-fit naive albedo A (= A_B) against a hyperillumination-generated truth, returning (A_fit, chi2).
def fit_naive(cfg, truth, free_offset=True, bounds=None, xatol=1e-3, **kw):
    bounds = cfg.bounds if bounds is None else bounds
    refl_unit = reflected_unit(cfg, "naive", **kw) if cfg.reflected else None

    def cost(A_):
        r = truth - model_lc(cfg, "naive", A_, refl_unit=refl_unit, **kw)[1]
        if free_offset:
            r = r - r.mean()  # real TSO fits have a free baseline
        return np.sum((r / cfg.sigma_bin) ** 2)

    res = minimize_scalar(cost, bounds=bounds, method="bounded", options={"xatol": xatol})
    return res.x, res.fun  # chi2 = cost(A_fit) = weighted sum of squared residuals


# One grid cell of a bias map: generate the kopal truth, fit the naive model, return the recovered-A_B bias.
def bias_cell(cfg, free_offset=True, **kw):
    _, truth = model_lc(cfg, "kopal", cfg.A_B_true, **kw)
    A_fit, _ = fit_naive(cfg, truth, free_offset=free_offset, **kw)
    return A_fit - cfg.A_B_true
