"""Bias maps over (a/R*, T_eff) with the planet sample and the Roche limit."""

import matplotlib.pyplot as plt
import numpy as np

from hyperphase._util import _normalise
from hyperphase.bias.detectors import common_sigma_1min, template_cfg
from hyperphase.bias.fisher import critical_sigma_1min, measurability
from hyperphase.bias.maps import _roche_cases, roche_frontier_a_rs
from hyperphase.bias.tables import planet_list
from hyperphase.plotting.style import AXIS, C_HI, C_NAIVE, C_REDIST, FIT_ZERO, F_COLORS, INK, MUTED, SEQ_BLUE, TARGET, _fmt_esm
from hyperphase.targets import ROCKY_MAX_RADE, archive_frame, closest_in, planet_esm, rocky_sample


############################################################################################################################################
#####   Planets on the maps
############################################################################################################################################

# A planet's y coordinate on a map, keyed by the archive column of the y axis (x is always a/Rs = R/r2)
_TARGET_Y = {"a_over_r2": lambda p: p.R / p.r2,
             "pl_ratror": lambda p: p.r1 / p.r2,
             "st_teff":   lambda p: p.T2_eff}


def target_marker(ax, planet, ykey, AGG=None, **kw):
    """The planet as the green star on a bias map (TARGET style; kw overrides it), labelled with its name, a/Rs and (with AGG) ESM."""
    if ykey not in _TARGET_Y: raise ValueError(f"no target coordinate for ykey={ykey!r}; known: {tuple(_TARGET_Y)}")
    x, y = planet.R / planet.r2, _TARGET_Y[ykey](planet)
    esm = f", ESM={_fmt_esm(planet_esm(AGG, planet.name))}" if AGG is not None else ""
    return ax.plot(x, y, label=f"{planet.name}  ($a/R_\\star$={x:.2f}{esm})", **{**TARGET, **kw})


def _label_colors(n):
    """n distinct colours for the labelled planets: tab10 up to ten, else the dark then the light tab20 shades."""
    base = plt.cm.tab10(np.arange(10)) if n <= 10 else [plt.cm.tab20(i) for i in (*range(0, 20, 2), *range(1, 20, 2))]
    return [base[k % len(base)] for k in range(n)]


def overlay_planets(AGG, ax, ykey, ylim, a_rs_lim=(1.5, 4.0), max_rade=ROCKY_MAX_RADE, target=None, exclude=None, mark_target=True, n_closest=10, *, planets):
    """Overplot the archive planets on a bias map (x = a/Rs, y = `ykey`). Returns the names of labelled planets outside the box (not drawn).
    planets: the planets to label, in any form planet_list takes (curated parameters apply): a circle, colour and legend entry (name, a/Rs, ESM) each.
    n_closest: also label the n rocky archive planets with the smallest a/Rs (closest_in) as diamonds, unless in `planets`; 0 for none.
    target: a PlanetParams, left out of the scatter and the labels and drawn as the green star (target_marker) unless mark_target=False.
    exclude: further name(s) to drop from the background scatter; loose spellings are fine.
    The other rocky_sample planets inside the box are faint background dots."""
    if ykey not in _TARGET_Y: raise ValueError(f"no planet coordinate for ykey={ykey!r}; known: {tuple(_TARGET_Y)}")
    not_target = lambda pl: target is None or _normalise(pl.name) != _normalise(target.name)   # the target is the green star instead
    labelled = [pl for pl in planet_list(planets) if not_target(pl)]
    given = {_normalise(pl.name) for pl in labelled}
    extra = [pl for pl in planet_list(closest_in(AGG, n_closest, max_rade=max_rade, exclude=[target.name] if target is not None else None)) if _normalise(pl.name) not in given and not_target(pl)] if n_closest else []
    marker = {**{pl.name: "o" for pl in labelled}, **{pl.name: "D" for pl in extra}}
    labelled = labelled + extra
    drop = ([exclude] if isinstance(exclude, str) else list(exclude or [])) + [pl.name for pl in labelled] + ([target.name] if target is not None else [])
    bg = rocky_sample(AGG, ykey, ylim, a_rs_lim=a_rs_lim, max_rade=max_rade, exclude=drop)

    x = "a_over_r2"
    ax.scatter(bg[x], bg[ykey], s=16, facecolor="#40c4d6", edgecolor="w", linewidths=.3,
               alpha=.6, zorder=3, label=f"other known rocky planets ($R_p<{max_rade:g}\\,R_\\oplus$, N={len(bg)})")
    arch = archive_frame(AGG)
    esm = dict(zip(arch.pl_name.map(_normalise), arch["ESM"]))
    pos = {pl.name: (pl.R / pl.r2, _TARGET_Y[ykey](pl)) for pl in labelled}
    inside = [pl for pl in labelled if a_rs_lim[0] <= pos[pl.name][0] <= a_rs_lim[1] and ylim[0] <= pos[pl.name][1] <= ylim[1]]
    for color, pl in zip(_label_colors(len(inside)), inside):
        ax.scatter(*pos[pl.name], s=60 if marker[pl.name] == "o" else 48, marker=marker[pl.name], color=color, edgecolor="w", linewidths=.7, zorder=5,
                   label=f"{pl.name}  ($a/R_\\star$={pos[pl.name][0]:.2f}, ESM={_fmt_esm(esm.get(_normalise(pl.name), np.nan))})" + ("" if marker[pl.name] == "o" else f"  [{n_closest} closest-in]"))
    if target is not None and mark_target: target_marker(ax, target, ykey, AGG)
    return [pl.name for pl in labelled if pl not in inside]


############################################################################################################################################
#####   Bias maps and the Roche limit
############################################################################################################################################

def _mark_limits(ax, contact, roche_a_rs, anchors):
    """Vertical marks of the limit curve: the contact limit (hatched), the Roche limits roche_a_rs ({label: a/R*}) and the anchor planets (grey
    dashed). Returns the legend handles."""
    handles = [ax.axvspan(ax.get_xlim()[0], contact, facecolor="0.85", edgecolor="0.5", hatch="///", lw=0, zorder=0, label=r"inside the star ($a/R_\star < 1 + R_p/R_\star$)")]
    for col, (lab, a_R) in zip(_ROCHE_COLORS, roche_a_rs.items()):
        handles.append(ax.axvline(a_R, color=col, lw=1.2, label=rf"Roche limit, {lab}  ($a/R_\star$ = {a_R:.2f})"))
    for nm, pl in anchors.items():
        handles.append(ax.axvline(pl.R / pl.r2, color=MUTED, lw=1., ls="--", label=rf"{nm}  ($a/R_\star$ = {pl.R / pl.r2:.2f})"))
    return handles


def plot_limit_curve(sweep, geo, cfg, rp_rs, roche_a_rs, anchors, sweep_ubol0=None, u_bol=None, titles=(None, None, None), suptitle=None):
    """The limit curve: a bias_vs_a_rs sweep (both models) on a log a/Rs axis, in three stacked panels.
    Geometry (sweep_geometry): area fractions, and the stellar angular radius on the right axis.
    Bias: Delta A_B of both fits, the spurious F on the right axis, hollow markers where a fit railed, the fitted A_B = 0 line where a fit goes
    below it, and the A_B-only bias of sweep_ubol0 (uniform stellar disc) if given.
    Measurability: |Delta A_B| / sigma of both fits, sqrt(dchi2_F) of the spurious F and sqrt(chi2) of the best A_B-only fit.
    Every panel marks the contact limit, the Roche limits roche_a_rs ({label: a/R*}) and the planets of anchors ({name: PlanetParams}).
    cfg: the sweep's config (sigma_1min, n_orbits, n_bins for the labels); rp_rs: the sweep's Rp/Rs (contact limit). Returns fig, axs."""
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
    a2.set_title(titles[2] or rf"Measurability  ($\sigma_{{1\rm min}}$ = {cfg.sigma_1min:.0f} ppm, {cfg.n_orbits} orbits, {cfg.n_bins} bins)")
    a2.legend(handles=h, fontsize=9, loc="upper right", frameon=True, facecolor="#fcfcfb", edgecolor="none")

    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
    x0, x1 = sweep["a_rs"].iloc[0], sweep["a_rs"].iloc[-1]
    ticks = [t for t in (1., 1.2, 1.5, 2., 3., 4., 5., 7., 10., 15., 20.) if x0 <= t <= x1]
    for a_ in axs:
        a_.set_xlim(x0, x1)   # axvline / axvspan widen the autoscaled limits otherwise
        # plain numbers, not 2 x 10^0
        a_.xaxis.set_major_locator(FixedLocator(ticks)); a_.xaxis.set_major_formatter(FormatStrFormatter("%g")); a_.xaxis.set_minor_formatter(NullFormatter())
    axs[-1].set_xlabel(r"$a/R_\star$")
    fig.suptitle(suptitle or "The limit curve: bias from neglecting hyperillumination towards the contact limit", y=0.995)
    fig.legend(handles=limit_handles, loc="lower center", bbox_to_anchor=(0.5, 0.), ncol=3, fontsize=9)
    fig.tight_layout(rect=(0., 0.06, 1., 0.985))   # room for the limit legend below the panels and the suptitle above
    return fig, axs


def plot_bias_maps(xg, yg, bias_map, A_B_true, AGG, ykey, ylabel, rp_rs_of, incl=90., F_map=None, F_levels=(0.01, 0.02, 0.05),
                   show_absolute=True, show_relative=False, show_F=False, target=None, title=None, roche=True, legend_fontsize=12, n_closest=10, *, planets, verbose=True):
    """Stacked maps over (a/R*, ykey): |Delta A_B|, |Delta A_B|/A_B and the fitted F of the joint fit (show_absolute / show_relative / show_F;
    show_F needs F_map). Dash-dot contour: the fitted A_B = 0, on every panel where the map crosses it.
    With F_map, bias_map must be the joint fit's bias (the titles say "(joint fit)").
    Grey: railed cells and, with F_map, cells without an occultation (b = (a/R*) cos i >= 1 + Rp/R*), where nothing anchors the flux level and
    A_B and F are degenerate; dashed lines bound the grazing zone 1 - Rp/R* < b < 1 + Rp/R*.
    rp_rs_of(y): Rp/R* at the map's y values; incl [deg]: the inclination the map was computed at.
    planets, target, n_closest: see overlay_planets. roche: the Roche frontier (draw_roche_frontier): True for roche_periods(), a {label: P_R [h]}
    dict, or False; drawn on a T_eff axis only, the only one that fixes the host density (main sequence). Returns fig, axs."""
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
        # contour ink that stays readable on light and dark cells
        cs = ax.contour(xg, yg, z, levels=levels, colors=["w" if im.norm(lv) > 0.5 else INK for lv in levels], linewidths=.8)
        ax.clabel(cs, fmt=fmt)
        if fit_zero:   # white where the panel is dark along the line, as for the level contours
            near = np.abs(A_fit) <= np.percentile(np.abs(A_fit.compressed()), 10)
            ink = "w" if im.norm(np.ma.median(z[near & ~np.ma.getmaskarray(z)])) > 0.5 else INK
            cz = ax.contour(xg, yg, A_fit, levels=[0.], colors=[ink], linewidths=FIT_ZERO["lw"], linestyles=FIT_ZERO["ls"])
            ax.clabel(cz, fmt={0.: r"$A_{B,\rm fit} = 0$"})   # on the line, not in the legend
        if grazing:
            for sign in (-1., 1.):
                ax.plot((1. + sign * rp_rs_of(y_line)) / cos_i, y_line, color=INK, ls="--", lw=1.2, label=r"grazing occultation ($b = 1 \mp R_p/R_\star$)" if sign > 0 else None)
        if _roche_cases(roche) and ykey == "st_teff": draw_roche_frontier(ax, xg, yg, _roche_cases(roche), rp_rs_of)
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

    if outside and verbose: print(f"outside the map box, not drawn: {', '.join(outside)}")
    ok = ~no_occultation & np.isfinite(bias_map)
    line = (f"{'occulting, ' if F_map is not None else ''}non-railed cells: {ok.sum()} of {ok.size};  Delta A_B in [{bias_map[ok].min():+.3f}, {bias_map[ok].max():+.3f}]"
            f" = [{100 * bias_map[ok].min() / A_B_true:+.1f}, {100 * bias_map[ok].max() / A_B_true:+.1f}] % of A_B")
    if F_map is not None:
        line += f",  F in [{F_map[ok].min():.4f}, {F_map[ok].max():.4f}]"
    if verbose: print(line)
    if verbose: print(f"fitted A_B < 0 in {int((A_B_true + bias_map[ok] < 0.).sum())} of {ok.sum()} cells;  railed: {(~no_occultation & ~np.isfinite(bias_map)).sum()}" + (f",  without occultation: {no_occultation.sum()}" if F_map is not None else ""))
    return fig, axs


_ROCHE_COLORS = (C_HI, C_REDIST, F_COLORS[-1])
ROCHE_HATCH_LABEL = r"inside every Roche limit shown (main-sequence host)"


def draw_roche_frontier(ax, xg, yg, periods, rp_rs_of):
    """The Roche frontier on an (a/Rs, T_eff) map for main-sequence hosts: a_R/Rs(T_eff) per Roche period of `periods` ({label: P_R [h]},
    roche_periods), the region inside the innermost limit (shortest P_R) hatched, and the contact limit a/Rs = 1 + Rp/Rs (rp_rs_of) dashed.
    Legend entries only for what lies inside the map. The curve is a main-sequence reference, not a planet's own limit: real host densities
    scatter around rho_star(T_eff), so a planet's distance from its limit is roche_proximity(P, P_R), not its position relative to the curve."""
    y_line = np.linspace(yg[0], yg[-1], 200)
    visible = lambda x: bool(np.any((x > xg[0]) & (x < xg[-1])))
    for color, (lab, P_R) in zip(_ROCHE_COLORS, periods.items()):
        x = roche_frontier_a_rs(y_line, P_R)
        ax.plot(x, y_line, color=color, lw=1.8, label=rf"Roche limit, {lab} ($P_{{\rm R}}$ = {P_R:.1g} h)" if visible(x) else None)
    x_min = roche_frontier_a_rs(y_line, min(periods.values()))
    ax.fill_betweenx(y_line, xg[0], np.maximum(x_min, xg[0]), facecolor="none", edgecolor=INK, hatch="///", lw=0, label=ROCHE_HATCH_LABEL if np.any(x_min > xg[0]) else None)
    x_contact = 1. + rp_rs_of(y_line)
    if visible(x_contact): ax.plot(x_contact, y_line, color=INK, lw=1.2, ls="--", label=r"contact ($a/R_\star = 1 + R_p/R_\star$)")


def plot_false_F_map(xg, yg, dchi2_map, cfg, AGG, ykey, ylabel, rp_rs_of, incl=90., z=2., sigma_levels=(10., 20., 50., 100., 200., 500., 1000., 2000., 5000.), target=None, title=None,
                     roche=True, n_closest=10, *, planets, verbose=True):
    """Map of the critical noise sigma_crit(a/Rs, y) for a z-sigma false-positive F (critical_sigma_1min of a false_F_map_grid result), on a log
    colour scale, with the archive planets as in plot_bias_maps. Returns the sigma_crit array, not the figure.
    cfg: the map's config; its sigma_1min is the noise dchi2_map was evaluated with, its n_orbits those sigma_crit refers to.
    The map holds Rp/R* and i at their reference values. The phase-curve signal is ∝ (Rp/R*)^2, so dchi2_F ∝ (Rp/R*)^4 and sigma_crit ∝ (Rp/R*)^2
    to a good approximation: rescale a planet with another Rp/R* (or fit it at its own geometry) before comparing its noise with the map.
    planets, target, n_closest, roche: as in plot_bias_maps."""
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
    if outside and verbose: print(f"outside the map box, not drawn: {', '.join(outside)}")
    ax.set_facecolor("0.85")
    ax.grid(False)
    ax.set_xlim(xg[0], xg[-1]); ax.set_ylim(yg[0], yg[-1])
    ax.set_xlabel(r"$a/R_\star$"); ax.set_ylabel(ylabel)
    ax.set_title(title or rf"Critical noise for a {z:g}$\sigma$ false-positive $F$ ({n_orbits} orbits)")
    fig.colorbar(im, ax=ax, label=rf"$\sigma_{{1\,\rm min,\,crit}}$ [ppm] for a {z:g}$\sigma$ detection of $F > 0$")
    fig.tight_layout()
    fig.legend(*ax.get_legend_handles_labels(), loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, framealpha=.9)

    if verbose: print(f"occulting, non-railed cells: {ok.sum()} of {ok.size};  sigma_1min,crit ({z:g} sigma, {n_orbits} orbits) in [{sig_crit[ok].min():.1f}, {sig_crit[ok].max():.1f}] ppm;  "
          f"sqrt(dchi2_F) at sigma_1min = {sigma_1min:g} ppm in [{np.sqrt(dchi2_map[ok]).min():.2f}, {np.sqrt(dchi2_map[ok]).max():.2f}] sigma")
    if verbose: print(f"railed: {(~no_occultation & ~np.isfinite(dchi2_map)).sum()},  without occultation: {no_occultation.sum()}")
    return sig_crit
