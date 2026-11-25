"""Figures of the noiseless analysis: bias against the true albedo, per-planet and per-instrument bars, retrieval scatter plots,
nightside signal, evidence per phase bin and parameter sweeps."""

import matplotlib.pyplot as plt
import numpy as np

from hyperphase.bands import band_of
from hyperphase.bias.detectors import detectors, is_detector_set
from hyperphase.bias.evidence import eclipse_half_phase, evidence_residual, phase_windows
from hyperphase.bias.fisher import (VERDICT_NAIVE_OK, VERDICT_NEEDS_HI, VERDICT_NO_NOISE, VERDICT_UNCONSTRAINED, bias_verdict, critical_time,
                                    flat_prior_sigma, measurability)
from hyperphase.bias.fits import binned, is_railed
from hyperphase.bias.tables import FALSE_POSITIVE, NO_FALSE_POSITIVE, band_range_um, noise_cols, select_planets
from hyperphase.defaults import A_B_FIT_BOUNDS, F_BOUNDS
from hyperphase.model import model_pcurve
from hyperphase.plotting.style import (AXIS, C_HI, C_NAIVE, C_REDIST, FIT_ZERO, F_COLORS, F_axis, INK, INK_2, INSTRUMENT_COLORS, MOCK_KW, MUTED,
                                       SEQ_BLUE, TARGET, _fmt_esm, _tick_name, cell_edges, instrument_label)


############################################################################################################################################
#####   Bias against the true albedo, per-planet bars and retrieval scatter plots
############################################################################################################################################

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


# One measurability panel: |bias| / sigma^Fisher on a log axis for each (y, style, label) curve, with the 1-sigma and 0.3-sigma reference lines. The
# right axis is the same curve scaled by cfg.sigma_1min: the per-minute precision at which the bias is a 1-sigma effect (sigma^Fisher is strictly
# proportional to sigma_1min).
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


# Measurability of the joint-fit biases, side by side: |Delta A_B| / sigma_A_B (joint solid, A_B-only dashed) and F_fit / sigma_F (the truth has F =
# 0, so F_fit is the bias). Both sigmas are marginal Fisher widths with the baseline marginalised; sigma_A_B of the joint fit already includes the
# widening from the F degeneracy.
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


# The hyperillumination truth next to the best point-source fits of fit_truth (A_B-only and, if present, joint; baselines refitted) with one mock
# noise realisation at sigma_bin, and the residuals truth - model before (naive at the true A_B) and after absorbing the bias, against the same noise.
# fit: a fit_truth result for cfg at A_B_true; kw: its model_pcurve keywords. Times in hours from transit. Returns fig, ax so the notebook can set the
# suptitle.
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


# Bar chart of the expected significance z_F of the spurious F per row of a false-positive table, with the threshold z as a dashed line; bars at or
# above it (FALSE_POSITIVE) in the hyperillumination colour. Rows under the axis: the fitted F and the nightside temperature it implies, the delivered
# and critical noise, and the critical observing time. plot_false_F_bars: one planet on several instruments (instrument_false_F).
# plot_planet_false_F_bars: several planets on one instrument (planet_false_F_table), the bars of `planets` (required, e.g.
# paper.PAPER_PLANETS) in that order (select_planets), labelled with a/Rs.
def plot_false_F_bars(tab, title=None, figsize=None, ax=None):
    if "name" in tab: raise TypeError("a per-planet table: use plot_planet_false_F_bars(tab, planets=...)")
    return _false_F_bars(tab, title, figsize, ax)


# plot_planet_bias_bars shows one delivered and one critical noise per planet, which a joint fit of several detectors does not have (one level per
# detector, sigma_1min:<label>); the false-positive bars (_false_F_bars) list them per detector instead.
def _one_noise_level(tab):
    if any(c.startswith("sigma_1min:") for c in tab.columns):
        raise ValueError("a joint fit of several detectors has one noise level per detector (sigma_1min:<label> columns), not the single sigma_1min the bars show: print the table (noise_cols) or use plot_planet_retrievals")


def plot_planet_false_F_bars(tab, title=None, figsize=None, ax=None, *, planets, verbose=True):
    return _false_F_bars(select_planets(tab, planets, verbose), title, figsize, ax)


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
# The x tick labels carry each planet's a/Rs and (with `esm` in the table) ESM in brackets. In a noise panel a planet needs hyperillumination exactly
# when the DELIVERED bar is at or below the REQUIRED one - that is bias >= 1 sigma. The verdict (bias_verdict) is written above every bar group.
# "unconstrained" planets - Fisher sigma at or above the flat-prior std (dashed line in the bias panel) - get hatched bars in both panels: their bias
# / sigma and sigma_1min_crit are ratios to a meaningless sigma and must not be read as "naive OK". quantities=("F", "A_B") needs a redist=True table
# (it carries both fits' columns); the decision columns and the verdict are recomputed per quantity, so the table's own `quantity` does not matter.
# log_noise: sigma_1min spans the sample's brightness range, so the noise panels are logarithmic by default. log_bias does the same for the bias
# panels, where an unconstrained sigma (~1) would otherwise flatten the measurable ones (~0.05). bias_label / sigma_label override the first
# quantity's legend labels. bias_break=(y_lo, y_hi) is the linear alternative to log_bias: each bias panel is split into a lower axis [0, y_lo] and a
# compact upper one [y_hi, top] (height ratio BIAS_BREAK_RATIO) with an axis break between them. Pick y_lo above the flat-prior line so it stays
# visible; a bar top that falls into the gap (y_lo, y_hi) shows no value annotation. hide_unconstrained=True drops every planet that is unconstrained
# in any of the plotted quantities, together with the flat-prior line and the axis break (bias_break is ignored), since only the unconstrained sigmas
# (~1) need either. It raises a ValueError if no planet is left. planets (required, e.g. paper.PAPER_PLANETS): the bar groups, in that order
# (select_planets); planets the table does not have are listed in a note.
_BAR_Q = {"A_B": ("bias_A_B", "sigma_A_B", r"bias $\Delta A_B$", r"$\sigma_{A_B}^{\rm Fisher}$", r"$|\Delta A_B|$ and $\sigma_{A_B}$", "$A_B$"),
          "F":   ("F_fit",    "sigma_F",   r"spurious $F$",      r"$\sigma_F^{\rm Fisher}$",     r"$F_{\rm fit}$ and $\sigma_F$",     "$F$")}
_VERDICT_STYLE = {VERDICT_NEEDS_HI: dict(color=C_HI, weight="bold"), VERDICT_NAIVE_OK: dict(color=C_NAIVE), VERDICT_UNCONSTRAINED: dict(color=MUTED, style="italic")}
_HATCH = "///"   # bars of unconstrained planets
BIAS_BREAK_RATIO = (1, 3)   # upper : lower axis height of a broken bias panel


# Parameter rows under the bottom panel of plot_planet_bias_bars: one text cell per planet per row, aligned with the bar groups (x in data units, y in
# axes units below the tick labels). rows: [(label, values, fmt)], top row first; a non-finite value prints "-". Returns the axes-fraction the rows
# extend below the axes, so the caller can reserve the space.
def _param_rows(ax, x, rows, y0=-0.10, dy=0.075, fontsize=8):
    for k, (label, vals, fmt) in enumerate(rows):
        y = y0 - k * dy
        ax.text(x[0] - 0.55, y, label, transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=fontsize, color=MUTED)
        for xi, v in zip(x, vals):
            ax.text(xi, y, fmt.format(v) if np.isfinite(v) else "-", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=fontsize)
    return -y0 + len(rows) * dy


def plot_planet_bias_bars(tab, bias_label=None, sigma_label=None, fmt_bias="{:+.2f}", fmt_sigma="{:.2f}", fmt_noise="{:.0f}",
                          figsize=None, log_noise=True, log_bias=False, colors=(None, None), title=None, quantities=None, grid=True, param_rows=True, sigma_crit_plot=False,
                          bias_break=None, hide_unconstrained=True, *, planets, verbose=True):
    tab = select_planets(tab, planets, verbose)
    _one_noise_level(tab)
    no_noise = ~np.isfinite(tab["sigma_1min"])   # VERDICT_NO_NOISE rows: a bias but no Fisher sigma to set it against
    if no_noise.any():
        if verbose: print(f"no noise level, so not in the bar plot (no Fisher sigma): {', '.join(tab.loc[no_noise, 'name'])}")
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
    # param_rows: a/Rs, ESM, Rp, P, t_obs, Teff and one t_crit per quantity under the bottom panel (0.26 in each), and the tick label is then the bare
    # name
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


# Plotted quantity of plot_instrument_retrievals: (value column, sigma column, truth column, y label, legend label of the dashed truth line, default
# ylim, truth value, legend title, name). Truth column None: one line across the plot at the truth value - attrs["A_B_true"] for A_B, 0 for F (the
# truth has no redistribution, so the plotted F_fit is the spurious F itself). "depth" (tables from instrument_depths): the true depth differs per
# band, so each instrument gets its own dashed segment. "F" needs the joint fit's tables (redist=True); its hollow marker is F's own verdict (sigma_F
# >= the flat-prior std of F_BOUNDS), not A_B's.
_RETRIEVAL_Q = {"A_B": ("A_B_fit", "sigma_A_B", None, "Bond albedo $A_B$", "injected $A_B$ = {A_B_true:.2f}", (0., 0.4), "A_B_true", "fitted $A_B$", "$A_B$"),
                "F": ("F_fit", "sigma_F", None, "heat redistribution $F$", "truth: $F$ = 0 (no redistribution)", (0., None), 0., "fitted $F$ (spurious: the truth has none)", "$F$"),
                "depth": ("depth_fit", "sigma_depth", "depth_true", r"Occultation depth $\delta$ [ppm]", "true (hyperillumination) occultation depth", None, None, "occultation depth of the fit", "$A_B$")}


# Marker groups of the plot: each table is one group, except that the rows of a depth table whose parameters come from a joint fit of several
# detectors (fit != instrument, see instrument_depths) form their own group per joint key - the same parameters evaluated in each detector's band, so
# they sit at the detector positions with another marker.
def _retrieval_groups(tabs):
    groups = []
    for label, t in tabs.items():
        joint = (t["fit"] != t["instrument"]) if "fit" in t else np.zeros(len(t), bool)
        groups.append((label if not joint.any() or label is not None else "one detector per fit", t[~joint]))
        if joint.any():
            for key, part in t[joint].groupby("fit", sort=False):
                groups.append(((f"{label}, " if label is not None else "") + f"{instrument_label(key, oneline=True, short=True)} jointly", part))
    return groups


# Retrieved A_B per instrument for one planet, after Ramirez et al. 2026 (arXiv:2610.22338), Fig. 4: one colour per instrument, the marker at the
# fitted A_B, dark / light error bars at sigma_levels x the Fisher sigma (default 1 and 2 sigma), the dashed line at the injected A_B_true. tabs: one
# instrument_retrievals table, or {label: table} to set several fits side by side per instrument, one marker shape each (e.g. {"A_B only": ..., "joint
# A_B + F": ...}, like the paper's phase curve / transit pairs). An unconstrained fit (A_B's Fisher sigma >= the flat-prior std of the physical range
# A_B_BOUNDS) gets a hollow marker: its error bar is not a measurement. A joint fit of several detectors is its own x position in the A_B plot (its
# "+"-joined key); in the depth plot (instrument_depths(joint=True)) its rows sit at the detectors' positions as an extra marker group
# (_retrieval_groups), since the depth is a per-band quantity. quantity="depth" plots the fitted occultation depth with its propagated Fisher sigma
# instead (tables from instrument_depths), with a dashed segment at each instrument's true depth. ylim defaults to [lower, 0.4] for A_B (the fits sit
# below the injected A_B, since hyperillumination biases them low; error bars beyond the range are clipped by the axes), with lower = 0 unless a
# fitted A_B is below 0.1: then it reaches 0.1 below the lowest fit, rounded down to a tenth. Below 0 a light band and the A_B = 0 line (FIT_ZERO)
# mark the values outside the physical range. The depth autoscales. plot_planet_retrievals is the same plot for planet_bias_table tables: one position
# per PLANET of `planets` (required, e.g. paper.PAPER_PLANETS), in that order, on one instrument (the tick label then carries the planet's
# a/Rs); every planet of the list is shown, unconstrained ones as hollow markers. The fits side by side per position work the same way.
def plot_instrument_retrievals(tabs, quantity="A_B", sigma_levels=(1, 2), markers=("o", "s", "D", "^"), ylim="default", title=None, figsize=None, ax=None):
    tabs = tabs if isinstance(tabs, dict) else {None: tabs}
    if any("instrument" not in t for t in tabs.values()): raise TypeError("a per-planet table: use plot_planet_retrievals(tabs, planets=...)")
    return _plot_retrievals(tabs, quantity, sigma_levels, markers, ylim, title, figsize, ax, "instrument")


def plot_planet_retrievals(tabs, quantity="A_B", sigma_levels=(1, 2), markers=("o", "s", "D", "^"), ylim="default", title=None, figsize=None, ax=None, *, planets, verbose=True):
    tabs = tabs if isinstance(tabs, dict) else {None: tabs}
    return _plot_retrievals({k: select_planets(t, planets, verbose) for k, t in tabs.items()}, quantity, sigma_levels, markers, ylim, title, figsize, ax, "name")


# The default y range of an A_B retrieval plot over several tables: (lower, top), lower = 0 unless a fitted A_B is below 0.1, then 0.1 below the
# lowest fit rounded down to a tenth. Pass it as ylim= to every panel of a figure whose retrieval panels share the y axis (sharey=True), so each panel
# draws the A_B < 0 band of the common range.
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


############################################################################################################################################
#####   Nightside temperature (06_nightside_temperature.ipynb)
############################################################################################################################################
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
    fig.tight_layout(rect=(0., 0.14, 1., 1.))   # room for the figure legend
    return fig


############################################################################################################################################
#####   Expected evidence: where the Bayes factor comes from (04_evidence_prediction.ipynb)
############################################################################################################################################

# Per-bin evidence figure: one column per detector of every entry of `cases`, in wavelength order ({label: FittingConfig or detector set}, e.g.
# MIRI/LRS next to the joint NIRSpec fit). Top: the residual truth - fitted point-source model (baseline removed) unbinned and per bin, against +-
# sigma_bin. Bottom: r_perp^2 per bin, which sums to chi2_min = 2 <ln B> over all bins of a fit, with the transit and occultation windows shaded and
# the quadrature windows hatched (phase_windows). Returns fig, axes (2 x n_columns) and {label: evidence_residual} for the budgets. kw: the
# model_pcurve keywords of truth and fit.
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
        ax.set_title(f"{instrument_label(band_of(c), oneline=True)}" + (f"\n{label}" if label else "") + rf"  ($\sigma_{{\rm bin}}$ = {sb:.1f} ppm)", fontsize=10)
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


############################################################################################################################################
#####   One-parameter sweeps of the bias (01_albedo_bias.ipynb, 02_redistribution_bias.ipynb: inclination, u_bol)
############################################################################################################################################

# A bias_sweep per mode on shared axes: sweeps = {mode label: DataFrame}. joint=False (A_B-only fit): the bias (right axis: in % of the true A_B),
# |bias| / sigma, and <ln B> = chi2_min / 2. joint=True (the sweep was run with both models): the bias of the fit with redistribution (solid) next to
# the A_B-only fit (dashed), the spurious F, and z_F = sqrt(dchi2_F) (solid) next to |bias| / sigma of the fit with redistribution (dashed). mark: the
# x value of the planet's own setting, drawn as the green star on every curve.
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
