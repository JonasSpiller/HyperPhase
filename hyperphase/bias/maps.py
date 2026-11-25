"""Bias and spurious redistribution over a grid of (a/R*, T_eff), and the Roche limit on that grid."""
from dataclasses import replace

import numpy as np
import pandas as pd
from astropy import constants as const
from scipy.interpolate import RegularGridInterpolator

from hyperphase.bias.fisher import false_F_fit, fit_truth
from hyperphase.bias.fits import RAIL_TOL, fit_railed, temperature_bias
from hyperphase.defaults import A_B_FIT_BOUNDS, DEFAULT_MAP_RP_RS, F_BOUNDS
from hyperphase.model import swept_planet
from hyperphase.parallel import pmap
from hyperphase.planet import PlanetParams
from hyperphase.roche import ROCHE_CASES, ROCHE_R_P, kepler_a_rs, kepler_period_days, rho_star, roche_period_h, rocky_mean_density


############################################################################################################################################
#####   Bias maps
############################################################################################################################################
def bias_cell(cfg, A_B_true, free_offset=True, rail_tol=RAIL_TOL, temp_bias=False, bounds=A_B_FIT_BOUNDS, redist=False, F_bounds=F_BOUNDS, return_chi2=False, **kw):
    """One cell of a bias map: generate the extended-source truth, fit the point-source model and return the bias of the recovered A_B (with
    temp_bias=True the temperature bias instead). NaN if the fit railed against a bound, i.e. the point-source model cannot reproduce the truth at ANY
    albedo. redist=True fits "naive_redist" (A_B and F) instead and returns (bias, F_fit), both NaN on a rail; the truth has no redistribution, so
    F_fit is itself the bias in F (F_fit = 0 is a valid fit, only F's upper bound counts as a rail). return_chi2=True appends the fit's chi2 at
    cfg.sigma_1min, also NaN on a rail, which false_F_map_grid(fits=...) turns into dchi2_F without refitting."""
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


# The 1-D counterpart of bias_map_grid along a/Rs alone (the limit curve of 06_roche_limit.ipynb): both fits with their Fisher sigmas at
# every a/Rs, one DataFrame row per point. bounds=A_B_FIT_BOUNDS by default, so A_B_fit < 0 is a valid result; `railed` (fit_railed) flags a fit on a
# bound of `bounds`, and the fitted values are kept, so the caller decides what to do with them. rho_star [g/cm^3]: with it each point gets the period
# a circular orbit at that a/Rs has around a star of that mean density (kepler_period_days). The period never enters the bias, only sigma_bin, so the
# Fisher sigmas and chi2 then belong to a physically consistent orbit; None keeps cfg.planet.period at every point. Columns: a_rs, period_h,
# sigma_bin; A_B_fit, bias, sigma, chi2, railed (A_B-only fit); A_B_fit_J, bias_J, sigma_J, F_fit, sigma_F, chi2_J, railed_J (joint fit); each model's
# block only if it is in `models`.
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


# The illumination geometry along an a/Rs grid at cfg's Rp/Rs: the area fractions of PlanetParams.fraction_illuminated (Knuth et al. 2017 / Carter et
# al. 2024) and the stellar angular radius.
def sweep_geometry(cfg, a_rs_grid):
    geo = pd.DataFrame([{"a_rs": a, **{k.replace(" ", "_"): v for k, v in swept_planet(cfg, a_rs=a).fraction_illuminated(extra=True).items()}} for a in a_rs_grid])
    geo["night"] = 1. - geo["fully_illuminated"] - geo["penumbra"]
    geo["angular_radius_deg"] = np.degrees(np.arcsin(1. / geo["a_rs"]))
    return geo


# False-positive heat redistribution. The truth has F = 0, so an observer who finds F > 0 significantly would claim heat transport, i.e. an
# atmosphere. How significant the spurious F is, is the likelihood-ratio statistic dchi2_F = chi2(A_B-only fit) - chi2(joint A_B + F fit), both at
# their best fit with a free baseline, on the noiseless (Asimov) truth: sqrt(dchi2_F) is the median expected one-sided significance of "F > 0" (Cowan
# et al. 2011), exact for the nonlinear T ∝ F^(1/4) response and the F = 0 boundary (null distribution 1/2 chi2_1). NOT F_fit / sigma_F: the Fisher
# sigma_F is the slope AT F_fit, but the model responds ∝ F^~1.5 near 0, so the quadratic extrapolation to F = 0 overstates the significance (by ~1.4x
# on TOI-2431 b / NRS1), and it ignores the boundary. dchi2_F ∝ n_orbits / sigma_1min^2 exactly (the fits do not depend on the noise scale), see
# critical_sigma_1min. One map cell: (dchi2_F, F_fit); NaN on a rail, as in bias_cell. dchi2_F >= 0 since the A_B-only model is the F = 0 member of
# the joint one; tiny negative values from the tolerances are clipped.
def false_F_cell(cfg, A_B_true, free_offset=True, rail_tol=RAIL_TOL, bounds=A_B_FIT_BOUNDS, F_bounds=F_BOUNDS, **kw):
    d = false_F_fit(cfg, A_B_true, free_offset=free_offset, rail_tol=rail_tol, bounds=bounds, F_bounds=F_bounds, **kw)
    return d["dchi2_F"], d["F_fit"]


def _false_F_cell_at(cfg, A_B_true, a_rs, y_param, y, kw):
    return false_F_cell(cfg, A_B_true, a_rs=a_rs, **{y_param: y}, **kw)


# dchi2_F over a map (like bias_map_grid): returns (dchi2 map, F map), shape (len(y_grid), len(a_rs_grid)). The F map should agree with
# bias_map_grid(redist=True)'s. fits=(chi2_map, chi2_map_J, F_map_J) builds both maps without refitting: chi2_map from bias_map_grid(...,
# return_chi2=True), chi2_map_J and F_map_J from bias_map_grid(..., redist=True, return_chi2=True), on the same cfg, A_B_true, grids, y_param and
# **kw. Those are exactly the two fits false_F_cell makes per cell, so the result is identical. Only the shapes are checked.
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


# A synthetic PlanetParams for planet-agnostic bias maps: the geometry every map cell inherits for whatever the map does not sweep. Units: r2 = 1
# (stellar radii), so r1 = Rp/Rs and R = a/Rs. period and R_star_Rsun are placeholders: the phase curve is binned per orbit, so the period drops out
# of the bias, and the stellar radius only sets a_AU for the (tiny) light-travel-time correction. T14 is NaN: it is only needed by PandExo, which
# never runs on the reference geometry. The star keeps PlanetParams' default log g = 4.5 and [Fe/H] = 0 for every T_eff (a hypothetical star; see
# swept_planet for what that does at M-dwarf T_eff).
def reference_planet(a_rs=2.5, rp_rs=DEFAULT_MAP_RP_RS, T_eff=4000., incl=90., period=0.3, R_star_Rsun=0.6, name="reference geometry"):
    return PlanetParams(name=name, period=period, incl=incl, ecc=0., w=90., r1=rp_rs, r2=1., R=a_rs, T2_eff=T_eff,
                        a_AU=a_rs * R_star_Rsun * float(const.R_sun / const.au), T14=np.nan)


# A real planet moved to another period on a circular orbit around its own host: a/R* = (a/R*)_0 (P / P_0)^(2/3) (Kepler III at the host's density),
# a_AU likewise. Rp/R*, the inclination (so the impact parameter shrinks with a/R*), the star and its log g / [Fe/H] are kept. With period_h = the
# planet's Roche period (roche_periods) it is the most extreme version of a known system: the same planet and star at the Roche limit of that
# composition. T14 is NaN (only PandExo needs it; give the moved planet its parent's sigma_1min). Raises if the planet would touch the star (a/R* <= 1
# + Rp/R*: SPIDERMAN raises a ValueError there).
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


############################################################################################################################################
#####   The Roche limit on the maps, and printed summaries
############################################################################################################################################

# The printed summary of a limit curve: the rail boundaries, the largest non-railed biases, the u_bol sensitivity (if sweep_ubol0 is given) and the
# sweep read off at each Roche limit of roche_a_rs.
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


# Roche periods [h] of the compositions in `cases` ({label: (iron mass fraction, rho_0 / rho_mean)}, default roche.ROCHE_CASES) for a
# rock/iron planet of radius R_p [R_E] (default ROCHE_R_P): mean density from roche.rocky_mean_density (Fortney et al. 2007), P_R from
# roche.roche_period_h (Rappaport et al. 2013 eq. 5). The labels carry R_p. This is what roche=True draws on the maps; in period the limits hold
# for any host.
def roche_periods(cases=None, R_p=ROCHE_R_P):
    cases = ROCHE_CASES if cases is None else cases
    return {f"{lab}, {R_p:g} $R_\\oplus$": float(roche_period_h(rocky_mean_density(R_p, fe), q)) for lab, (fe, q) in cases.items()}


def _roche_cases(roche):
    """The {label: P_R [h]} of a roche= argument: True -> roche_periods(), a dict -> itself, False / None -> None."""
    return roche_periods() if roche is True else (roche or None)


def roche_frontier_a_rs(T_eff, P_R_h):
    """a_R/Rs along a T_eff axis for main-sequence hosts, for a Roche period P_R_h [h]."""
    return kepler_a_rs(P_R_h / 24., rho_star(T_eff))


def roche_allowed_masks(xg, yg, periods):
    X, Y = np.meshgrid(xg, yg)
    return {lab: X >= roche_frontier_a_rs(Y, P_R) for lab, P_R in periods.items()}


# The largest |bias| a map reaches inside each allowed region of roche_allowed_masks, one line per Roche limit of `periods` (the same labels as the
# map legend): the physically reachable extreme (for main-sequence hosts) instead of the box's corner.
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


# How well a bias map stands in for a planet's own-geometry fit: the A_B bias read off a map at each planet's position (linear interpolation) next to
# the planet_bias_table value, which fits the planet at its full geometry. The maps hold the inclination (and the parameter not on the y axis) at the
# reference values, so planets with a large impact parameter b come out differently. map_excess = |bias read off the map| - |own-geometry bias|:
# positive = the map overstates. y_param: "T_eff" for an (a/Rs, T_eff) map, "rp_rs" for an (a/Rs, Rp/Rs) map. planets: {name: PlanetParams} as passed
# to planet_bias_table.
def map_vs_planet_bias(tab, planets, a_rs_grid, y_grid, bias_map, y_param="T_eff"):
    interp = RegularGridInterpolator((y_grid, a_rs_grid), bias_map, bounds_error=False)   # NaN outside the map box
    pls = [planets[nm] for nm in tab["name"]]
    y = [pl.T2_eff if y_param == "T_eff" else pl.r1 / pl.r2 for pl in pls]
    out = pd.DataFrame(dict(name=tab["name"], b=[pl.R / pl.r2 * np.cos(np.radians(pl.incl)) for pl in pls],
                            bias_own=tab["bias_A_B"], bias_map=interp(np.column_stack([y, tab["a_rs"]]))))
    out["map_excess"] = np.abs(out["bias_map"]) - np.abs(out["bias_own"])
    return out.sort_values("map_excess", ascending=False).reset_index(drop=True)
