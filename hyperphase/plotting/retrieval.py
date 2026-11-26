"""Figures of the retrievals: simulated data, posteriors, posterior predictive checks, and series of noise realizations."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

from hyperphase.bias.detectors import _blocks, detector_cfgs
from hyperphase.bias.fisher import fisher_sigma, measurability
from hyperphase.bias.fits import binned, temperature_bias
from hyperphase.defaults import A_B_FIT_BOUNDS, DEFAULT_N_LA, DEFAULT_N_LO, F_BOUNDS
from hyperphase.irradiance import surface_temperature
from hyperphase.model import model_pcurve
from hyperphase.plotting.style import C_HI, C_NAIVE, C_REDIST, MUTED
from hyperphase.retrieval.likelihood import model_samples, theta_of
from hyperphase.retrieval.prediction import noiseless_reference
from hyperphase.retrieval.sampler import equal_samples, posterior_summary
from hyperphase.retrieval.simulate import detector_data, truth_model_kw


############################################################################################################################################
#####   Simulated data
############################################################################################################################################
def plot_simulated_data(cfg, A_B_true, data, insolation_model="hyperillum", labels=None, verbose=True, **model_kw):
    """The simulated observation against the truth it was drawn from (re-evaluated per detector at the truth's resolution, truth_model_kw;
    model_kw overrides it), with the residuals in units of sigma, and a printed chi2 check of the noise (verbose). One column per detector for a
    detector set; `labels` names the columns. Returns fig, (ax, axr), or fig, axes (2 x n_detectors) for a set."""
    cfgs, datas = detector_cfgs(cfg), detector_data(data)
    labels = list(labels) if labels is not None else [None] * len(cfgs)
    fig, axes = plt.subplots(2, len(cfgs), figsize=(8 * len(cfgs), 5.5), sharex=True, squeeze=False, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.06))
    for k, (c, d, lab) in enumerate(zip(cfgs, datas, labels)):
        ax, axr = axes[0, k], axes[1, k]
        ph = d["t"] / c.planet.period  # orbital phase

        # Full phase curve
        t_hi, y_hi = model_pcurve(c, insolation_model, A_B_true, **{**truth_model_kw(d, c), **model_kw})
        y_true = binned(c, y_hi) + float((d.get("truth") or {}).get("offset", 0.))   # with the injected baseline, if any
        ph_hi = t_hi / c.planet.period

        ax.plot(ph_hi, y_hi, lw=1., color="0.5", label=f'truth ({insolation_model}, $A_B$={A_B_true:.2f})')
        ax.step(ph, y_true, where="mid", color="C0", lw=1.4, label="binned truth")
        ax.errorbar(ph, d["y_obs"], yerr=d["sigma"], fmt="o", ms=4, color="k", capsize=2,
                    elinewidth=1., label=f'simulated data ($\\sigma$={c.sigma_bin:.1f} ppm)')
        ax.set_ylabel("$F_p/F_\\star$ [ppm]");
        ax.legend(frameon=False, fontsize=8);
        ax.margins(x=0.01)
        if lab: ax.set_title(lab, fontsize=9)

        r = (d["y_obs"] - y_true) / d["sigma"]
        axr.axhline(0, color="0.7", lw=0.8);
        axr.axhline(1, color="0.85", lw=0.6, ls=":")
        axr.axhline(-1, color="0.85", lw=0.6, ls=":")
        axr.plot(ph, r, "ko", ms=4)
        axr.set_xlabel("orbital phase");
        axr.set_ylabel("resid / $\\sigma$");
        axr.set_ylim(-3.5, 3.5)

        # Numerical Checks
        chi2 = np.sum(r ** 2)
        pre = f"{lab}: " if lab else ""
        if verbose: print(f"{pre}sigma_bin: {c.sigma_bin:.2f} ppm  (bin width {c.planet.period * 24 * 60 / c.n_bins:.2f} min)")
        if verbose: print(f"{pre}chi2 / n_bins (normalized average noise per bin ~1): {chi2 / c.n_bins:.2f}")
        if verbose: print(f"{pre}residual mean / sigma (~0): {r.mean():.2f}")
    return (fig, (axes[0, 0], axes[1, 0])) if len(cfgs) == 1 else (fig, axes)


############################################################################################################################################
#####   Posterior visualization
############################################################################################################################################
_COLOR = {"naive": "C0", "hyperillum": "C3", "naive_redist": "C2", "hyperillum_redist": "C1"}


def _density_levels(x, y, rng=None, bins=60, smooth=1.0, levels=(0.68, 0.95)):
    """2-D density of equal-weight draws and the contour values that enclose the requested posterior masses: highest-density regions (cells
    sorted by height, summed until the mass is reached), not Gaussian sigmas."""
    H, xe, ye = np.histogram2d(x, y, bins=bins, range=rng)
    if smooth: H = gaussian_filter(H, smooth)
    H /= H.sum()
    flat = np.sort(H.ravel())[::-1]
    cum = np.cumsum(flat)
    v = sorted(float(flat[min(int(np.searchsorted(cum, l)), len(flat) - 1)]) for l in levels)
    # coarse bins can put two mass levels in the same cell; contour() needs strictly increasing levels
    v = sorted(set(v))
    return 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), H.T, v


def plot_posterior_overlay(fits, x, y, truth=None, profile=None, n=None, bins=60, smooth=1.0, colors=None, figsize=(6.4, 6.4), seed=0):
    """The models' 2-D posteriors of (x, y) on one pair of axes, with the 1-D marginals on the sides. Returns fig, ax.
    fits: {name: NestedFit}, same parameters and same data; x, y: parameter names sampled in every fit.
    truth: {name: injected value}.
    profile: {name: (x_vals, y_vals, lnL)} from logl_grid, drawn as dotted contours at the 2-parameter 1/2/3 sigma levels (no prior weighting).
    n: equal-weight draws per model (None: all)."""
    colors = colors or _COLOR
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[1, 0]); axx = fig.add_subplot(gs[0, 0], sharex=ax); axy = fig.add_subplot(gs[1, 1], sharey=ax)
    axx.tick_params(labelbottom=False); axy.tick_params(labelleft=False)

    S = {name: equal_samples(f, n, seed) for name, f in fits.items()}
    ix = {name: f.pset.names.index(x) for name, f in fits.items()}
    iy = {name: f.pset.names.index(y) for name, f in fits.items()}
    xs = np.concatenate([S[k][:, ix[k]] for k in S]); ys = np.concatenate([S[k][:, iy[k]] for k in S])
    rng = [[xs.min(), xs.max()], [ys.min(), ys.max()]]   # shared range, so the two densities are directly comparable

    for name, s in S.items():
        c = colors.get(name, "k")
        xc, yc, H, lv = _density_levels(s[:, ix[name]], s[:, iy[name]], rng=rng, bins=bins, smooth=smooth)
        ax.contourf(xc, yc, H, levels=lv + [max(H.max(), lv[-1] * 1.001)], colors=c, alpha=0.25)
        ax.contour(xc, yc, H, levels=lv, colors=c, linewidths=1.2)
        ax.plot([], [], color=c, lw=1.2, label=name)
        axx.hist(s[:, ix[name]], bins=bins, range=rng[0], density=True, histtype="step", color=c, lw=1.3)
        axy.hist(s[:, iy[name]], bins=bins, range=rng[1], density=True, histtype="step", color=c, lw=1.3, orientation="horizontal")

    if profile is not None:
        for name, (xv, yv, lnL) in profile.items():
            ax.contour(xv, yv, lnL - lnL.max(), levels=[-5.91, -3.09, -1.15], colors=colors.get(name, "k"), linewidths=0.8, linestyles=":")

    if truth is not None:
        if x in truth: ax.axvline(truth[x], color="0.4", lw=1.0, ls="--"); axx.axvline(truth[x], color="0.4", lw=1.0, ls="--")
        if y in truth: ax.axhline(truth[y], color="0.4", lw=1.0, ls="--"); axy.axhline(truth[y], color="0.4", lw=1.0, ls="--")
        if x in truth and y in truth: ax.plot(truth[x], truth[y], "*", ms=13, mfc="#0cff0c", mec="k", mew=0.8, label="injected")

    lab = next(iter(fits.values())).pset
    ax.set_xlabel(dict(zip(lab.names, lab.labels))[x]); ax.set_ylabel(dict(zip(lab.names, lab.labels))[y])
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    axx.set_yticks([]); axy.set_xticks([])
    return fig, ax


def plot_bias(fits, truth, colors=None, figsize=None):
    """Median +/- 68 % per model against the injected value, one panel per parameter, annotated with the bias in sigma (posterior_summary).
    The panels are the union of the fits' parameters (first-seen order); a model that does not sample one has no marker there. Returns fig, axes."""
    colors = colors or _COLOR
    names, labels = [], {}
    for f in fits.values():   # e.g. only "naive_redist" samples F
        for nm, lb in zip(f.pset.names, f.pset.labels):
            if nm not in labels: names.append(nm); labels[nm] = lb
    fig, axes = plt.subplots(1, len(names), figsize=figsize or (4.2 * len(names), 2.4))
    axes = np.atleast_1d(axes)

    for ax, nm in zip(axes, names):
        for k, (name, f) in enumerate(fits.items()):
            if nm not in f.pset.names: continue   # this model does not sample it
            d = posterior_summary(f, truth, verbose=False)["params"][nm]
            ax.errorbar(d["med"], k, xerr=[[d["minus"]], [d["plus"]]], fmt="o", ms=6, color=colors.get(name, "k"), capsize=3, elinewidth=1.4)
            if "bias_sigma" in d: ax.annotate(f"{d['bias_sigma']:+.2f}$\\sigma$", (d["med"], k), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
        if nm in truth: ax.axvline(truth[nm], color="0.4", lw=1.0, ls="--")
        ax.set_yticks(range(len(fits))); ax.set_yticklabels(list(fits)); ax.set_ylim(-0.6, len(fits) - 0.2)
        ax.set_xlabel(labels[nm])
    fig.tight_layout()
    return fig, axes


def plot_evidence_jitter(bf, figsize=(5.2, 3.0), bins=40):
    """Distribution of ln B over dynesty's jittered realisations (output of bayes_factor): the sampling uncertainty of the Bayes factor,
    without assuming either lnZ is Gaussian. Returns fig, ax."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(bf["samples"], bins=bins, density=True, histtype="stepfilled", color="0.75", edgecolor="0.35")
    ax.axvline(bf["lnB"], color="k", lw=1.4, label=f"$\\ln B$ = {bf['lnB']:+.2f}")
    ax.axvline(0., color="C3", lw=1.0, ls="--")
    for d in (1., 2.5, 5.): ax.axvline(d, color="0.85", lw=0.7, ls=":")   # Jeffreys thresholds
    ax.set_xlabel(r"$\ln B$ (jittered)"); ax.set_ylabel("density"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_posterior_predictive(logls, fits, num="hyperillum", den="naive", n=200, colors=None, seed=0, n_jobs=-1, figsize=None, levels=(68., 95.)):
    """Posterior predictive check, and where in orbital phase the two models differ. One column per detector for a joint fit.
    Panel 1: the data with each model's median curve and its posterior bands at `levels` %.
    Panel 2: (data - median model) / sigma per model.
    Panel 3: the difference of the two median curves against +/- sigma_bin, and as bars the per-bin delta chi2 between the two best-fit
    (max-lnL) models. The bars sum to 2 delta lnL_max over all detectors: where in phase the likelihood ratio arises.
    num / den: the models of panel 3 (num - den, as in bayes_factor()). n: posterior draws per model.
    Returns fig, (ax, axr, axd), or fig, axes (3 x n_detectors) for a joint fit."""
    colors = colors or _COLOR
    names = list(fits)
    f0, L0 = fits[names[0]], logls[names[0]]
    cfgs, datas = detector_cfgs(L0.cfg), detector_data(f0.meta["data"])
    labels = list(getattr(L0, "labels", None) or [None] * len(cfgs))
    blocks = _blocks([d["y_obs"].size for d in datas])
    n_det = len(cfgs)

    # the draws once per model: a joint theta gives every detector's curve at once
    curves, best = {}, {}
    for name in names:
        L, f = logls[name], fits[name]
        curves[name] = model_samples(L, equal_samples(f, n, seed), n_jobs=n_jobs)
        best[name] = L.model(f.results.samples[int(np.argmax(f.results.logl))])

    fig, axes = plt.subplots(3, n_det, figsize=figsize or (8.5 * n_det, 8.0), sharex=True, squeeze=False, gridspec_kw=dict(height_ratios=[3, 1.3, 1.6], hspace=0.07))
    mk = ("o", "s", "^", "v")   # the residuals coincide when the models fit almost equally well
    for kd in range(n_det):
        ax, axr, axd = axes[:, kd]
        blk, d = blocks[kd], datas[kd]
        t, y, s = d["t"], d["y_obs"], d["sigma"]
        ph = t / cfgs[kd].planet.period
        ax.errorbar(ph, y, yerr=s, fmt="o", ms=4, color="k", capsize=2, elinewidth=1., zorder=5, label="data")

        med = {}
        for k, name in enumerate(names):
            cv = curves[name][:, blk]
            lo1, m, hi1 = np.percentile(cv, [50 - levels[0] / 2, 50, 50 + levels[0] / 2], axis=0)
            lo2, hi2 = np.percentile(cv, [50 - levels[1] / 2, 50 + levels[1] / 2], axis=0)
            med[name] = m # pointwise median model curve
            c = colors.get(name, "k")

            # Panel 1
            ax.fill_between(ph, lo2, hi2, color=c, alpha=0.15, lw=0)
            ax.fill_between(ph, lo1, hi1, color=c, alpha=0.30, lw=0)
            ax.plot(ph, m, color=c, lw=1.5, label=f"{name} (pointwise median of {n} draws)")

            # Panel 2
            axr.plot(ph, (y - m) / s, mk[k % len(mk)], ms=4, mfc="none" if k else c, color=c)

        ax.set_ylabel("$F_p/F_\\star$ [ppm]"); ax.margins(x=0.01)
        if kd == 0: ax.legend(frameon=False, fontsize=8)
        if labels[kd]: ax.set_title(labels[kd], fontsize=9)
        for dd in (-1., 1.): axr.axhline(dd, color="0.85", lw=0.6, ls=":")
        axr.axhline(0., color="0.7", lw=0.8); axr.set_ylabel("resid / $\\sigma$"); axr.set_ylim(-3.5, 3.5)

        # Panel 3: the num - den pair only (any further model is in panels 1-2)
        if num in fits and den in fits:
            a, b = den, num
            axd.axhspan(-s.mean(), s.mean(), fc="0.9", ec="0.6", lw=0.8, label=r"$\pm\sigma_{bin}$")
            axd.axhline(0., color="0.7", lw=0.8)
            dmed = med[b] - med[a]
            axd.plot(ph, dmed, color="0.2", lw=1.5, label=f"median {b} - {a} [ppm]")
            dchi2 = ((y - best[a][blk]) ** 2 - (y - best[b][blk]) ** 2) / s ** 2   # > 0 where b fits better
            ax2 = axd.twinx()
            ax2.bar(ph, dchi2, width=0.7 / len(ph), color="C2", alpha=0.45, zorder=0)
            # Both axes symmetric about zero: twinx() autoscales on its own, so one-signed bars would grow from the panel floor, not from axd's
            # zero line.
            w = 1.05 * max(float(np.abs(dmed).max()), float(s.mean())); axd.set_ylim(-w, w)
            v = 1.05 * float(np.abs(dchi2).max()) or 1.; ax2.set_ylim(-v, v)
            ax2.set_ylabel(f"$\\Delta\\chi^2$ per bin (sum {dchi2.sum():+.2f})", fontsize=8, color="C2")
            ax2.tick_params(axis="y", labelcolor="C2", labelsize=8)
            if kd == 0: axd.legend(frameon=False, fontsize=8, loc="upper left")
        axd.set_xlabel("orbital phase"); axd.set_ylabel("[ppm]")
    return (fig, tuple(axes[:, 0])) if n_det == 1 else (fig, axes)


def plot_derived(derived, keys=("T_sub", "T_term", "depth"), truth=None, colors=None, bins=50, figsize=None):
    """Overlaid posteriors of the derived physical quantities from derived_samples().
    derived : {model_name: dict from derived_samples}
    truth   : optional {model_name: dict} (e.g. the same call evaluated at the injected theta) - drawn as vertical lines."""
    colors = colors or _COLOR
    unit = dict(T_sub="[K]", T_term="[K]", T_anti="[K]", depth="[ppm]", depth_th="[ppm]", depth_refl="[ppm]", A_B="")
    fig, axes = plt.subplots(1, len(keys), figsize=figsize or (3.6 * len(keys), 2.7))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        for j, (name, d) in enumerate(derived.items()):
            v = np.asarray(d[k], float)
            # a constant (e.g. the 0 K nightside T_anti of "naive" and "hyperillum"): a line, whose dash length tells coincident models apart
            if np.ptp(v) == 0.:
                ax.axvline(v[0], color=colors.get(name, "k"), lw=1.4, ls=(0, (4 + 2 * j, 2)), label=f"{name} (= {v[0]:g})")
            else:
                ax.hist(v, bins=bins, density=True, histtype="step", lw=1.4, color=colors.get(name, "k"), label=name)
            if truth and name in truth: ax.axvline(np.median(truth[name][k]), color=colors.get(name, "k"), lw=1.0, ls="--")
        ax.set_xlabel(f"{k} {unit.get(k.split(':')[0], '')}"); ax.set_yticks([])   # "depth:<label>" of a joint fit shares the unit of "depth"
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, axes


def plot_noisy_vs_truth(cfg, data, logls, fits, A_B_true, base, insolation_truth="hyperillum",
                        A_B_bounds=A_B_FIT_BOUNDS, xatol=None, truth_kw=None, fisher=True, colors=None, figsize=None, verbose=False):
    """The one-seed fits against the noiseless truth. One column per detector for a joint fit.
    Top: the truth (unbinned, and binned = the noiseless data), the simulated data, each model's posterior median, and, dashed, the noiseless
    fits of "naive" and "naive_redist" (where loaded) to the truth (noiseless_reference: fit_AB_only, fit_redist), which are the bias.
    Bottom: everything minus the binned truth, against +/- sigma_bin; dotted: the numerical floor, the truth model at the injected theta on the
    likelihood's grid (as in evidence_reference).
    logls / fits: {model: GaussianLogL or JointGaussianLogL} and {model: NestedFit}, same keys.
    base: {param: injected value}, every baseline included (as for theta_of).
    insolation_truth: the model the data were drawn from; the floor needs it in logls.
    A_B_bounds, xatol, truth_kw: passed to noiseless_reference.
    fisher: also quote the bias in units of the Fisher sigma (fisher_sigma), next to this seed's posterior width.
    verbose: also print the per-model median and max-lnL rows, the absorbed offsets and the floor detail.
    Returns fig, (ax, axr), out, or fig, axes, out for a joint fit. out["noiseless"][model]: the noiseless fit (A_B, F, offset, chi2), its bias,
    the bias in posterior and Fisher sigmas with the matching sigma_1min_crit (None for a joint fit), and bias_over_floor; the rest of out:
    the plotted curves, the posterior summaries and each median's chi2 against the truth."""
    colors = colors or _COLOR
    cfgs, datas = detector_cfgs(cfg), detector_data(data)
    n_det, cfg0, joint = len(cfgs), detector_cfgs(cfg)[0], len(detector_cfgs(cfg)) > 1
    L0 = next(iter(logls.values()))
    labels = list(getattr(L0, "labels", None) or [None] * n_det)
    blocks = _blocks([c.n_bins for c in cfgs])

    # --- the noiseless reference and the misspecified models fitted to it ---
    ref = noiseless_reference(cfg, data, [m for m in ("naive", "naive_redist") if m in fits], A_B_true=A_B_true, insolation_truth=insolation_truth, A_B_bounds=A_B_bounds, xatol=xatol, truth_kw=truth_kw)
    t_hi, y_hi, y_true, s_bin, fit_cfg, fit_truth = (ref[k] for k in ("t_hi", "y_hi", "y_true", "s_bin", "fit_cfg", "fit_truth"))
    truth_kw, F_truth, off_inj, insols, noiseless = (ref[k] for k in ("truth_kw", "F_truth", "off_inj", "insols", "noiseless"))
    y_true_all, sig_all = np.concatenate(y_true), np.concatenate([np.full(c.n_bins, sb) for c, sb in zip(cfgs, s_bin)])
    for m, d in noiseless.items():
        ys, offs = [], []
        for c, yt in zip(cfgs, y_true):   # each detector's own baseline, as free_offset absorbed it
            y0 = binned(c, model_pcurve(c, m, d["A_B"], F=d["F"], insol=insols[m], **truth_kw)[1])
            offs.append(float(np.mean(yt - y0))); ys.append(y0 + offs[-1])
        d["y"], d["offset"] = (ys if joint else ys[0]), (offs if joint else offs[0])
        d["bias_A_B"] = d["A_B"] - A_B_true
        d["dT_over_T"] = float(temperature_bias(A_B_true, d["bias_A_B"], d["F"] or 0.))   # substellar temperature
        # the nightside the false-positive F implies (0 K in the truth)
        if d["F"] is not None: d["T_night"] = float(surface_temperature(np.pi, cfg0.planet, "naive_redist", A_B=d["A_B"], emissivity_bol=cfg0.emissivity_bol, F=d["F"]))

    # --- the noisy fits: posterior median (what you would report) and max-lnL (= MAP, uniform priors) -------------
    summ   = {m: posterior_summary(f, base, verbose=False)["params"] for m, f in fits.items()}
    th_med = {m: np.array([summ[m][n]["med"] for n in f.pset.names], float) for m, f in fits.items()}
    th_map = {m: f.results.samples[int(np.argmax(f.results.logl))] for m, f in fits.items()}
    y_med  = {m: logls[m].model(th_med[m]) for m in fits}                  # concatenated over the detectors (blocks)
    y_map  = {m: logls[m].model(th_map[m]) for m in fits}
    # the numerical floor: same model, same parameters, the likelihood's coarser grid
    y_floor = logls[insolation_truth].model(theta_of(logls[insolation_truth], base)) if insolation_truth in logls else None

    fig, axes = plt.subplots(2, n_det, figsize=figsize or (8.5 * n_det, 6.2), sharex=True, squeeze=False, gridspec_kw=dict(height_ratios=[3, 1.7], hspace=0.06))
    for kd in range(n_det):
        ax, axr = axes[:, kd]
        c, dk, blk, sb = cfgs[kd], datas[kd], blocks[kd], s_bin[kd]
        ph, ph_hi = dk["t"] / c.planet.period, t_hi / c.planet.period
        ax.plot(ph_hi, y_hi[kd] + off_inj, lw=1.0, color="0.55", zorder=1, label=f"truth ({insolation_truth}, $A_B$={A_B_true:.3f}" + (f", $F$={F_truth:g})" if F_truth is not None else ")"))
        ax.step(ph, y_true[kd], where="mid", color="0.25", lw=1.2, zorder=2, label="binned truth = noiseless data")
        ax.errorbar(ph, dk["y_obs"], yerr=dk["sigma"], fmt="o", ms=4, color="k", capsize=2, elinewidth=1., zorder=6, label=f"simulated data ($\\sigma$={sb:.1f} ppm)")
        for m in fits:
            ax.step(ph, y_med[m][blk], where="mid", color=colors.get(m, "k"), lw=1.6, zorder=4, label=f"{m} posterior median (one seed)")
        for m, d in noiseless.items():
            ax.step(ph, d["y"][kd] if joint else d["y"], where="mid", color=colors.get(m, "k"), lw=1.6, ls="--", zorder=5,
                    label=f"{m} fitted to the noiseless truth ($A_B$={d['A_B']:.4f}" + (f", $F$={d['F']:.4f})" if d["F"] is not None else ")"))
        ax.set_ylabel("$F_p/F_\\star$ [ppm]"); ax.margins(x=0.01)
        if kd == 0: ax.legend(frameon=False, fontsize=8)
        if labels[kd]: ax.set_title(labels[kd], fontsize=9)

        axr.axhspan(-sb, sb, fc="0.9", ec="0.6", lw=0.8, zorder=0, label=r"$\pm\sigma_{bin}$")
        axr.axhline(0., color="0.7", lw=0.8)
        axr.plot(ph, dk["y_obs"] - y_true[kd], "o", ms=4, color="k", zorder=6, label="data - truth (the noise realisation)")
        for m in fits:
            axr.step(ph, y_med[m][blk] - y_true[kd], where="mid", color=colors.get(m, "k"), lw=1.6, zorder=4)
        for m, d in noiseless.items():
            axr.step(ph, (d["y"][kd] if joint else d["y"]) - y_true[kd], where="mid", color=colors.get(m, "k"), lw=1.6, ls="--", zorder=5)
        if y_floor is not None:
            axr.step(ph, y_floor[blk] - y_true[kd], where="mid", color="0.45", lw=1.0, ls=":", zorder=3, label="numerical floor (truth model at the injected $\\theta$)")
        axr.set_xlabel("orbital phase"); axr.set_ylabel("model - truth [ppm]"); axr.margins(x=0.01)
        axr.secondary_yaxis("right", functions=(lambda v, sb=sb: v / sb, lambda v, sb=sb: v * sb)).set_ylabel(r"$\sigma_{bin}$", fontsize=8)
        if kd == 0: axr.legend(frameon=False, fontsize=7, ncol=3, loc="upper left")

    # --- numbers -----------------------------------------------------------------------------------
    chi2_tr = lambda y: float(np.sum(((y - y_true_all) / sig_all) ** 2))
    chi2_floor = chi2_tr(y_floor) if y_floor is not None else None
    for m, d in noiseless.items():
        sig = 0.5 * (summ[m]["A_B"]["hi"] - summ[m]["A_B"]["lo"])        # posterior width of A_B under this model, this seed
        d["sigma_A_B"] = float(sig)
        d["bias_sigma0"] = abs(d["bias_A_B"]) / sig                       # the noiseless bias in units of this seed's posterior width
        # |bias| = sigma_A_B at this sigma_1min (measurability), from one seed's posterior width; None for a joint fit (one sigma_1min per detector)
        d["sigma_1min_crit"] = None if joint else cfg0.sigma_1min * d["bias_sigma0"]
        if d["F"] is not None:
            d["sigma_F"] = float(0.5 * (summ[m]["F"]["hi"] - summ[m]["F"]["lo"]))
            d["F_sigma0"] = d["F"] / d["sigma_F"]                            # F in units of its (one-sided) posterior width
        if fisher:   # Fisher sigma at the noiseless fit, marginalised over the baselines (and F)
            sf = fisher_sigma(fit_cfg, m, d["A_B"], F=d["F"], bounds=A_B_bounds, F_bounds=F_BOUNDS, insol=insols[m], **truth_kw)
            d["sigma_A_B_fisher"] = float(sf[0])
            d["bias_sigma_fisher"], crit = (float(x) for x in measurability(d["bias_A_B"], sf[0], cfg0.sigma_1min))
            d["sigma_1min_crit_fisher"] = None if joint else crit
            if d["F"] is not None: d["sigma_F_fisher"] = float(sf[1])
        d["bias_over_floor"] = (d["chi2"] / chi2_floor) if chi2_floor else None   # model-to-truth distance over the numerical floor

    out = dict(t_hi=t_hi, y_hi=y_hi if joint else y_hi[0], y_true=y_true if joint else y_true[0], y_med=y_med, y_map=y_map, y_floor=y_floor, summary=summ, noiseless=noiseless,
               chi2_truth={m: chi2_tr(y_med[m]) for m in fits}, chi2_floor=chi2_floor, blocks=blocks)

    off_names = {m: [n for n in f.pset.names if n.startswith("offset")] for m, f in fits.items()}
    if verbose:
        for m in fits:
            a, pm = summ[m]["A_B"], logls[m].unpack(th_map[m])
            o_med = "   ".join(f"{o} {summ[m][o]['med']:+7.2f} ppm ({summ[m][o]['bias']:+.2f} = {summ[m][o]['bias_sigma']:+.2f} sigma)" for o in off_names[m])
            o_map = "   ".join(f"{o} {pm[o]:+7.2f} ppm ({pm[o] - off_inj:+.2f})" for o in off_names[m])
            fF = (f"   F {summ[m]['F']['med']:.4f} ({summ[m]['F']['bias']:+.4f})", f"   F {pm['F']:.4f} ({pm['F'] - base.get('F', 0.):+.4f})") if "F" in summ[m] else ("", "")
            print(f"{m:12s} median   A_B {a['med']:.4f} ({a['bias']:+.4f} = {a['bias_sigma']:+.2f} sigma)   {o_med}{fF[0]}"
                  f"   chi2 vs truth {chi2_tr(y_med[m]):6.2f}   vs data {logls[m].chi2(th_med[m]):6.2f}")
            print(f"{'':12s} max-lnL  A_B {pm['A_B']:.4f} ({pm['A_B'] - A_B_true:+.4f})              {o_map}{fF[1]}"
                  f"           chi2 vs truth {chi2_tr(y_map[m]):6.2f}   vs data {logls[m].chi2(th_map[m]):6.2f}")
        print()

    if noiseless: print("NOISELESS fit(s) to the NOISELESS truth  (offset free, no noise, no seed - this is THE bias; the least-squares fits of the bias analysis" + (", jointly over the detectors):" if joint else "):"))
    for m, d in noiseless.items():
        fac = (1. / d["bias_sigma0"]) if d["bias_sigma0"] > 0 else np.inf      # how much finer sigma_1min must get for a 1-sigma bias
        fisher_txt = f" = {np.sign(d['bias_A_B']) * d['bias_sigma_fisher']:+.2f} sigma_Fisher" if fisher else ""
        print(f"   {m:12s} A_B {d['A_B']:.4f}   bias {d['bias_A_B']:+.4f} = {d['bias_A_B'] / d['sigma_A_B']:+.2f} sigma_A_B (posterior, this seed){fisher_txt}   ->   dT_sub/T = {100 * d['dT_over_T']:+.2f} %")
        if d["F"] is not None:
            print(f"   {'':12s} F   {d['F']:.4f}   false positive (truth F = 0) = {d['F_sigma0']:.2f} sigma_F (posterior)" + (f", sigma_F,Fisher = {d['sigma_F_fisher']:.4f} (one-sided at F = 0: indicative only)" if fisher else "")
                  + f"   ->   implied nightside T = {d['T_night']:.0f} K")
        if joint:
            print(f"   {'':12s} |dA_B|/sigma_A_B = {d['bias_sigma0']:.2f}   (all detectors scaled alike: " + (f"{fac:.1f}x finer photometry - or {fac ** 2:.0f}x the orbits - before the bias reaches 1 sigma)" if d["bias_sigma0"] < 1. else f"the bias is a {d['bias_sigma0']:.1f} sigma effect at the delivered noise)"))
        else:
            print(f"   {'':12s} |dA_B|/sigma_A_B = {d['bias_sigma0']:.2f}   ->   sigma_1min,crit = {d['sigma_1min_crit']:.1f} ppm/min"
                  + (f" (posterior) / {d['sigma_1min_crit_fisher']:.1f} ppm/min (Fisher)" if fisher else "")
                  + (f"   (now {cfg0.sigma_1min:.1f}, so {fac:.1f}x finer photometry - or {fac ** 2:.0f}x the orbits - before the bias reaches 1 sigma)" if d["bias_sigma0"] < 1.
                     else f"   (now {cfg0.sigma_1min:.1f}, already finer: the bias is a {d['bias_sigma0']:.1f} sigma effect at the delivered noise)"))
        if verbose:
            offs = d["offset"] if joint else [d["offset"]]
            print(f"   {'':12s} offset absorbed " + ", ".join(f"{o:+.2f}" for o in offs) + f" ppm    chi2 {d['chi2']:.3f}  (squared truth-to-manifold distance in units of sigma_bin)    resid rms {d['rms']:.2f} ppm")
            print(f"   {'':12s} the noisy {m} median instead says {summ[m]['A_B']['bias']:+.4f}; the gap to {d['bias_A_B']:+.4f} is this one seed's noise (sigma_A_B = {d['sigma_A_B']:.4f})")
        if chi2_floor:
            r = d["bias_over_floor"]
            print(f"   {'':12s} bias / floor = {d['chi2']:.3f} / {chi2_floor:.3f} = {r:.1f}x   ->   " + ("bias resolved" if r >= 10. else "NOT RESOLVED - refine model_kw (n_lo, n_la, n_layers, n_t) before quoting any bias from this run"))

    if y_floor is not None and verbose:
        print(f"\nnumerical floor: {insolation_truth} at the injected theta vs its own truth curve, chi2 = {chi2_floor:.3f}   max |diff| = {np.abs(y_floor - y_true_all).max():.2f} ppm = {np.abs((y_floor - y_true_all) / sig_all).max():.2f} sigma_bin")
        print(f"   the likelihood runs model_pcurve at its DEFAULTS (n_t=50*n_bins, n_lo={DEFAULT_N_LO}, n_la={DEFAULT_N_LA}), the truth at {truth_kw} - anything here is grid resolution, not model bias")
    return (fig, (axes[0, 0], axes[1, 0]), out) if n_det == 1 else (fig, axes, out)


############################################################################################################################################
#####   Retrievals against the linear prediction of the Bayes factor
############################################################################################################################################
def plot_ns_check(tab, groups=None, figsize=None):
    """The retrievals of a series_table against their noiseless predictions (run_prediction), one-to-one lines dashed:
    (a) the paired posterior-median difference (misspecified minus the truth's model) against the noiseless bias;
    (b) the posterior width of A_B under the misspecified model against the Fisher sigma;
    (c) the measured ln B (extended vs point source, with the sampler's error) against the per-run prediction (bias.evidence);
    (d) with series_table(..., redist=True): the significance of the spurious F in each realisation against its prediction max(z_F + zeta_F, 0).
    groups: {legend label: boolean mask over tab}, the marker groups (default: one per series). Returns fig, axes."""
    groups = groups or {name: (tab["series"] == name).to_numpy() for name in dict.fromkeys(tab["series"])}
    mk = ("o", "s", "D", "^", "v", "P", "X", "<", ">", "*", "h", "p")
    panels = [("dA_noiseless", "dA_median", r"noiseless bias $\Delta A_B$", r"paired median difference $\tilde A_B^{\rm alt} - \tilde A_B^{\rm true}$", "(a) bias"),
              ("sigma_fisher", "sigma_post", r"Fisher $\sigma_{A_B}$", r"posterior $\sigma_{A_B}$ (misspecified model)", "(b) uncertainty"),
              ("lnB_pred_ext_vs_ps", "lnB_ext_vs_ps", r"predicted $\ln B = \chi^2_{\min}/2 + \zeta\sqrt{\chi^2_{\min}}$", r"measured $\ln B$ (extended vs point source)", "(c) evidence")]
    if "zF_pred" in tab: panels.append(("zF_pred", "zF_obs", r"predicted $\max(z_F + \zeta_F,\,0)$", r"measured $z_F = \sqrt{2\,\Delta\ln\mathcal{L}_{\max}}$ (with vs without $F$)", "(d) spurious redistribution"))
    fig, axes = plt.subplots(1, len(panels), figsize=figsize or (5.7 * len(panels), 5.2))
    for ax, (xc, yc, xl, yl, ttl) in zip(axes, panels):
        for k, (label, m) in enumerate(groups.items()):
            t = tab[m & tab[xc].notna().to_numpy() & tab[yc].notna().to_numpy()]
            if not len(t): continue
            yerr = np.where(np.isfinite(t["lnB_err_jitter"]), t["lnB_err_jitter"], t["lnB_err"]) if yc == "lnB_ext_vs_ps" else None
            ax.errorbar(t[xc], t[yc], yerr=yerr, fmt=mk[k % len(mk)], ms=6, color=f"C{k % 10}", mec="k", mew=.5, elinewidth=1., capsize=2, label=f"{label} ({int(m.sum())})", zorder=3)
        # the axis range follows the plotted rows only (e.g. not null tests, whose bias has the other sign)
        shown = tab[np.any(list(groups.values()), axis=0)]
        lo, hi = np.nanmin([shown[xc].min(), shown[yc].min()]), np.nanmax([shown[xc].max(), shown[yc].max()])
        pad = 0.06 * (hi - lo) or 1e-3
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.4", lw=1., ls="--", zorder=1)
        if xc == "sigma_fisher": ax.set_xscale("log"); ax.set_yscale("log")
        else: ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl, fontsize=10)
    axes[2].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig, axes


def plot_ns_realizations(tab, panels, A_B_ylim=None, figsize=None, order="seed"):
    """The noise realisations of one configuration side by side, one column per entry of `panels` ({title: boolean mask over tab}), sorted by
    `order` (default: the seed, so the same number is the same standardised noise in every column). Above 15 realisations the markers shrink,
    the z_F numbers of (b) are left out and every fifth realisation is ticked.
    (a) the posterior of A_B under the extended-source, point-source and point-source-with-redistribution models: median, 68 % (thick) and 95 %
        (thin) intervals; dashed: the injected value; thin lines: the noiseless fits of the two point-source models.
    (b) the posterior of F, the noiseless F (line) and above each realisation its zF_obs.
    (c) ln B, extended vs point source: measured (filled) and predicted from that realisation's noise (open), on the expectation chi2_min / 2
        with its +/- sqrt(chi2_min) band; squares: ln B with vs without redistribution. Dotted: ln B = 2.5 and 5 (Trotta 2008).
    Needs a series_table(..., redist=True) of runs with all three models. Returns fig, axes (3 x number of panels)."""
    n = len(panels)
    fig = plt.figure(figsize=figsize or (3.4 * n + 0.8, 9.))
    gs = fig.add_gridspec(4, n, height_ratios=[0.75, 3., 1.7, 2.6], hspace=0.09, wspace=0.05)   # row 0 holds the legend
    axes = np.empty((3, n), dtype=object)
    for r in range(3):
        for c in range(n):
            axes[r, c] = fig.add_subplot(gs[r + 1, c], sharex=axes[0, c] if r else None, sharey=axes[r, 0] if c else None)
            axes[r, c].tick_params(labelbottom=r == 2, labelleft=c == 0)
    models = (("_truth", C_HI, "o", "extended source", -0.26), ("_alt", C_NAIVE, "o", "point source", 0.), ("_redist", C_REDIST, "s", "point source with redistribution", 0.26))
    L_INJ, L_NOISELESS, L_EXP, L_LNB, L_PRED, L_LNB_F = ("injected value", "noiseless fits", r"expected, $\chi^2_{\min}/2 \pm \sqrt{\chi^2_{\min}}$", r"$\ln B$, extended vs point source",
                                                        "predicted from the noise", r"$\ln B$, with vs without redistribution")
    for (title, mask), (ax_A, ax_F, ax_B) in zip(panels.items(), axes.T):
        t = tab[mask].sort_values(order)
        x = np.arange(1, len(t) + 1)
        dense = len(t) > 15   # many realisations: smaller markers, thinner bars, no z_F numbers
        ms, lw = (3., 1.4) if dense else (5., 2.4)
        for tag, color, marker, label, dx in models:   # (a) A_B under the three models
            ax_A.vlines(x + dx, t["A_B_lo95" + tag], t["A_B_hi95" + tag], color=color, lw=0.8 if not dense else 0.6, alpha=0.6, zorder=2)
            ax_A.vlines(x + dx, t["A_B_lo" + tag], t["A_B_hi" + tag], color=color, lw=lw, zorder=3)
            ax_A.plot(x + dx, t["A_B_med" + tag], marker, ms=ms, color=color, mec="w", mew=0.4, ls="none", zorder=4, label=label)
        ax_A.axhline(t["A_B_true"].iloc[0], color="k", lw=1., ls="--", zorder=1, label=L_INJ)
        ax_A.axhline(t["A_B_true"].iloc[0] + t["dA_noiseless"].iloc[0], color=C_NAIVE, lw=0.9, zorder=1, label=L_NOISELESS)
        ax_A.axhline(t["A_B_true"].iloc[0] + t["dA_noiseless_redist"].iloc[0], color=C_REDIST, lw=0.9, zorder=1)
        ax_A.set_title(title, fontsize=10)

        ax_F.vlines(x, t["F_lo95"], t["F_hi95"], color=C_REDIST, lw=0.8, alpha=0.6, zorder=2)   # (b) F and its significance in each realisation
        ax_F.vlines(x, t["F_lo"], t["F_hi"], color=C_REDIST, lw=lw, zorder=3)
        ax_F.plot(x, t["F_med"], "s", ms=ms, color=C_REDIST, mec="w", mew=0.4, zorder=4)
        ax_F.axhline(t["F_noiseless"].iloc[0], color=C_REDIST, lw=0.9, zorder=1)
        ax_F.axhline(0., color="k", lw=1., ls="--", zorder=1)
        if not dense:
            for xi, top, zF in zip(x, t["F_hi95"], t["zF_obs"]): ax_F.annotate(f"{zF:.1f}", (xi, top), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7, color=MUTED)

        c2 = t["chi2_min"].iloc[0]   # (c) the Bayes factors
        ax_B.axhspan(0.5 * c2 - np.sqrt(c2), 0.5 * c2 + np.sqrt(c2), color="0.9", lw=0, zorder=0)
        ax_B.axhline(0.5 * c2, color="0.45", lw=1., zorder=1, label=L_EXP)
        for level in (2.5, 5.): ax_B.axhline(level, color="0.5", lw=0.7, ls=":", zorder=1)
        ax_B.axhline(0., color="k", lw=0.7, zorder=1)
        ax_B.errorbar(x, t["lnB"], yerr=np.where(np.isfinite(t["lnB_err_jitter"]), t["lnB_err_jitter"], t["lnB_err"]), fmt="o", ms=ms - 0.5, color="k", elinewidth=1., zorder=4, label=L_LNB)
        ax_B.plot(x, t["lnB_pred"], "o", ms=1.8 * ms - 1., mfc="none", mec="k", mew=0.9 if not dense else 0.6, zorder=3, label=L_PRED)
        ax_B.plot(x, t["lnB_redist_vs_ps"], "s", ms=ms - 0.5, color=C_REDIST, mec="w", mew=0.4, zorder=3, label=L_LNB_F)
        ax_B.set_xticks(x[4::5] if dense else x); ax_B.set_xlim(0.4, len(t) + 0.6)
        ax_B.set_xlabel("noise realisation" + ("" if order == "seed" else f" (ordered by {order})"))
    axes[0, 0].set_ylabel("$A_B$"); axes[1, 0].set_ylabel("$F$"); axes[2, 0].set_ylabel(r"$\ln B$")
    if A_B_ylim is not None: axes[0, 0].set_ylim(*A_B_ylim)
    axes[1, 0].set_ylim(-0.0015, 1.5 * np.nanmax(tab[np.any(list(panels.values()), axis=0)]["F_hi95"]))   # room for the z_F numbers and the panel tag
    many = max(int(m.sum()) for m in panels.values()) > 15
    for ax, tag in zip(axes[:, 0], ("(a)", "(b)" if many else "(b) numbers: $z_F$", "(c)")): ax.text(0.03, 0.96, tag, transform=ax.transAxes, fontsize=9, va="top")
    entries = {l: h for ax in (axes[0, 0], axes[2, 0]) for h, l in zip(*ax.get_legend_handles_labels())}
    # filled column by column: the models and references of (a) and (b), then the Bayes factors of (c)
    order = [models[0][3], models[1][3], models[2][3], L_INJ, L_NOISELESS, L_EXP, L_LNB, L_PRED, L_LNB_F]
    ax_leg = fig.add_subplot(gs[0, :])
    ax_leg.axis("off")
    ax_leg.legend([entries[l] for l in order], order, loc="center", ncol=5, frameon=False, fontsize=8)
    return fig, axes


def plot_ns_distribution(tab, rows, others=None, figsize=None):
    """The noise realisations of one configuration as distributions, one row of three panels per entry of `rows` ({title: boolean mask over
    tab}). Needs a series_table(..., redist=True) of runs with all three models.
    (a) the posterior median of A_B under the point-source model (circles) and with redistribution (squares) against the extended-source median
        of the same realisation. Dashed: equal medians; lines: shifted by each fit's noiseless bias; cross: the injected value; dash-dotted:
        A_B = 0; corner bar: the median 68 % half-widths of the extended- and point-source posteriors.
    (b) the fraction of realisations with ln B (extended vs point source) above x (steps), against the Gaussian prediction N(chi2_min / 2,
        chi2_min) (dashed); others: one {label: mask} per row, further configurations in grey. Dotted: ln B = 2.5 and 5.
    (c) z_F of the spurious F against ln B per realisation, filled by ln B_F (point source with vs without redistribution). Horizontal line:
        the noiseless z_F; vertical line: chi2_min / 2.
    Returns fig, axes (number of rows x 3)."""
    fig, axes = plt.subplots(len(rows), 3, figsize=figsize or (13., 4.1 * len(rows)), squeeze=False)
    x_B = np.linspace(-4., 20., 500)
    for k, ((title, mask), (ax_A, ax_B, ax_F)) in enumerate(zip(rows.items(), axes)):
        t = tab[mask].sort_values("seed")
        A0, dA, dAF, c2 = t["A_B_true"].iloc[0], t["dA_noiseless"].iloc[0], t["dA_noiseless_redist"].iloc[0], t["chi2_min"].iloc[0]

        # (a) the point-source medians against the extended-source median
        lo, hi = min(t["A_B_med_truth"].min(), A0) - 0.03, max(t["A_B_med_truth"].max(), A0) + 0.03
        y_lo = min(-0.03, t["A_B_med_redist"].min() - 0.03)
        x = np.array([lo, hi])
        ax_A.plot(x, x, color="k", lw=1., ls="--", zorder=1, label="equal medians")
        ax_A.plot(x, x + dA, color=C_NAIVE, lw=1., zorder=1, label="shifted by the noiseless bias")
        ax_A.plot(x, x + dAF, color=C_REDIST, lw=1., zorder=1)
        ax_A.axhline(0., color="0.6", lw=0.8, ls="-.", zorder=1)
        ax_A.plot(t["A_B_med_truth"], t["A_B_med_alt"], "o", ms=5, color=C_NAIVE, mec="w", mew=0.4, ls="none", zorder=3, label="point source")
        ax_A.plot(t["A_B_med_truth"], t["A_B_med_redist"], "s", ms=5, color=C_REDIST, mec="w", mew=0.4, ls="none", zorder=3, label="point source with redistribution")
        ax_A.plot([A0], [A0], "+", ms=13, mew=1.6, color=C_HI, zorder=4, label="injected value")
        sx, sy = t["sigma_post_truth"].median(), t["sigma_post"].median()
        ax_A.errorbar([hi - sx - 0.04 * (hi - lo)], [y_lo + sy + 0.05 * (hi - y_lo)], xerr=sx, yerr=sy, color="0.4", lw=1., capsize=2, zorder=2)
        ax_A.set_xlim(lo, hi); ax_A.set_ylim(y_lo, hi)
        ax_A.set_xlabel(r"$A_B$, extended-source model"); ax_A.set_ylabel(r"$A_B$, point-source models")

        # (b) fraction of realisations above ln B
        groups = [(title.split(",")[0], t, "k")] + [(lab, tab[m], "0.6") for lab, m in (others[k].items() if others else ())]
        for lab, g, color in groups:
            v, c = np.sort(g["lnB"].to_numpy()), g["chi2_min"].iloc[0]
            ax_B.step(np.concatenate([[x_B[0]], v, [x_B[-1]]]), np.concatenate([[1.], 1. - np.arange(1, len(v) + 1) / len(v), [0.]]), where="post", color=color, lw=1.6, label=f"{lab} ({len(v)})")
            ax_B.plot(x_B, norm.sf((x_B - 0.5 * c) / np.sqrt(c)), color=color, lw=1., ls="--", label="Gaussian prediction" if color == "k" else None)
        ax_B.axvline(0., color="k", lw=0.7, zorder=1)
        ax_B.set_ylim(0., 1.02); ax_B.set_ylabel(r"fraction of realizations above $\ln B$")

        b = t["lnB_redist_vs_ps"].to_numpy()   # (c) z_F against ln B, filled by ln B_F
        for sel, fc, lab in ((b <= 0., "w", r"$\ln B_F \leq 0$"), ((b > 0.) & (b <= 2.5), "#a9cdf0", r"$0 < \ln B_F \leq 2.5$"), (b > 2.5, C_REDIST, r"$\ln B_F > 2.5$")):
            ax_F.plot(t["lnB"][sel], t["zF_obs"][sel], "s", ms=6, mfc=fc, mec=C_REDIST, mew=1., ls="none", zorder=3, label=f"{lab} ({int(sel.sum())})")
        ax_F.axhline(t["zF_noiseless"].iloc[0], color=C_REDIST, lw=1., zorder=1)
        ax_F.axvline(0.5 * c2, color="0.45", lw=1., zorder=1)
        ax_F.set_ylim(0., max(5.6, 1.08 * t["zF_obs"].max())); ax_F.set_ylabel(r"$z_F$")
        x_hi = max(17., 1.08 * max(g["lnB"].max() for _, g, _ in groups))
        for ax in (ax_B, ax_F):
            for level in (2.5, 5.): ax.axvline(level, color="0.5", lw=0.7, ls=":", zorder=1)
            ax.set_xlim(-3., x_hi); ax.set_xlabel(r"$\ln B$, extended vs point source")
        for ax, tag in zip((ax_A, ax_B, ax_F), "abc"): ax.set_title(f"({tag})" + (f" {title}" if tag == "a" else ""), fontsize=10, loc="left")
        if k == 0: ax_A.legend(frameon=False, fontsize=7.5, loc="upper left")
        ax_B.legend(frameon=False, fontsize=8, loc="upper right"); ax_F.legend(frameon=False, fontsize=8, loc="lower right", title=r"$\ln B_F$: with vs without $F$", title_fontsize=8)
    fig.tight_layout()
    return fig, axes
