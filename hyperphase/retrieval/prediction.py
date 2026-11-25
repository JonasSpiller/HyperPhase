"""Stored retrievals against the linear prediction of the Bayes factor and the noiseless least-squares fits."""
import contextlib
import glob
import io
import os
import re

import numpy as np
import pandas as pd
from dynesty import utils as dyfunc
from scipy.stats import chi2 as chi2_dist

from hyperphase.bands import band_of, joint_name
from hyperphase.bias.detectors import detector_cfgs, is_detector_set
from hyperphase.bias.evidence import evidence_residual
from hyperphase.bias.fisher import fisher_sigma
from hyperphase.bias.fits import binned, fit_AB_only, fit_redist
from hyperphase.bias.nuisance import nuisance_problem, nuisance_solve
from hyperphase.defaults import A_B_FIT_BOUNDS, DEFAULT_N_LA, DEFAULT_N_LO, F_BOUNDS
from hyperphase.irradiance import surface_temperature
from hyperphase.model import insolation_cache, model_pcurve
from hyperphase.parallel import pmap
from hyperphase.retrieval.likelihood import theta_of
from hyperphase.retrieval.priors import Normal, TruncNormal, _GEOMETRY, _ORBIT
from hyperphase.retrieval.sampler import bayes_factor, posterior_summary
from hyperphase.retrieval.simulate import truth_model_kw
from hyperphase.retrieval.storage import check_fit_current, find_dataset, load_fits, logl_from_fit, parse_result_file


############################################################################################################################################
#####   Noiseless reference fits
############################################################################################################################################
def noiseless_reference(cfg, data, models, A_B_true=None, insolation_truth=None, A_B_bounds=A_B_FIT_BOUNDS, xatol=None, truth_kw=None):
    """The noiseless counterpart of a simulated dataset: the truth curve simulate_data drew the data from, re-evaluated at the truth's own resolution (truth_model_kw), and the
    misspecified `models` fitted to it with the least-squares fits of the bias analysis, offset free (fit_AB_only for a one-parameter model: "naive" against a hyperillumination
    truth, "hyperillum" against a point-source truth; fit_redist, jointly in (A_B, F), for "naive_redist"; for a detector set the joint fit with one baseline per detector).
    Deterministic: no seed, no noise. Shared by plot_noisy_vs_truth and the series table (evidence_reference).

    A_B_true / insolation_truth default to what simulate_data recorded in data["truth"]; xatol=None keeps the fitters' own tolerances, so the numbers are those of the bias analysis.
    Returns dict(t_hi, y_hi [one unbinned curve per detector], y_true [binned, with the injected baseline, per detector], s_bin [per detector], fit_cfg, fit_truth [what the
    bias-analysis fitters take], truth_kw [model_pcurve keywords of the truth, without F], F_truth, off_inj, insols {model: insolation cache at the truth's resolution or None},
    noiseless {model: dict(A_B, F, chi2, rms)})."""
    cfgs = detector_cfgs(cfg)
    cfg0, joint = cfgs[0], len(cfgs) > 1
    tr = dict(data.get("truth") or {})
    A_B_true = tr["A_B"] if A_B_true is None else A_B_true
    insolation_truth = tr.get("insolation_model", "hyperillum") if insolation_truth is None else insolation_truth
    truth_kw = truth_model_kw(data, cfg0) if truth_kw is None else dict(truth_kw)
    F_truth = truth_kw.pop("F", None)                                 # only a "naive_redist" truth has one
    off_inj = float(tr.get("offset", 0.))                             # the injected baseline, if any
    fit_kw = {} if xatol is None else dict(xatol=xatol)

    # --- the noiseless reference per detector: exactly the curves simulate_data drew the data from -------------
    hi = [model_pcurve(c, insolation_truth, A_B_true, F=F_truth, **truth_kw) for c in cfgs]
    t_hi, y_hi = hi[0][0], [y for _, y in hi]
    y_true = [binned(c, y) + off_inj for c, y in zip(cfgs, y_hi)]                 # the noiseless data
    fit_cfg, fit_truth = (tuple(cfgs), y_hi) if joint else (cfg0, y_hi[0])        # what the bias-analysis fitters take: a detector set with one truth per detector

    # --- the misspecified models fitted to the NOISELESS truth: the deterministic bias, no seed involved (the fits of the bias analysis, at the
    # truth's resolution) ---
    noiseless, insols = {}, {}
    for m in models:
        if m == "naive_redist":
            insols[m] = None                                               # the Louden model has no map to cache
            A0, F0, chi2_0, rms0 = fit_redist(fit_cfg, fit_truth, free_offset=True, bounds=A_B_bounds, F_bounds=F_BOUNDS, **({} if xatol is None else dict(xatol=xatol, F_xatol=xatol)), **truth_kw)
            noiseless[m] = dict(A_B=float(A0), F=float(F0), chi2=float(chi2_0), rms=float(rms0))
        else:
            insols[m] = insolation_cache(cfg0, m, n_lo=truth_kw.get("n_lo", DEFAULT_N_LO), n_la=truth_kw.get("n_la", DEFAULT_N_LA))   # A_B only rescales the map (geometry-only, so one for all detectors)
            A0, chi2_0, rms0 = fit_AB_only(fit_cfg, fit_truth, m, free_offset=True, bounds=A_B_bounds, insol=insols[m], **fit_kw, **truth_kw)
            noiseless[m] = dict(A_B=float(A0), F=None, chi2=float(chi2_0), rms=float(rms0))
    return dict(t_hi=t_hi, y_hi=y_hi, y_true=y_true, s_bin=[c.sigma_bin for c in cfgs], fit_cfg=fit_cfg, fit_truth=fit_truth, truth_kw=truth_kw, F_truth=F_truth, off_inj=off_inj,
                insols=insols, noiseless=noiseless)


############################################################################################################################################
#####   Retrievals against the linear prediction of the Bayes factor (03_retrieval_series.ipynb)
############################################################################################################################################

# The prediction (bias.evidence): for two models with the same parameters and priors, ln B(true vs alt) = chi2_min / 2 +
# zeta sqrt(chi2_min), with chi2_min and r_perp from the NOISELESS A_B-only fit of the misspecified model `alt` to the truth and zeta = r_perp_hat .
# z. In a simulation the standardised noise z = (y_obs - y_true) / sigma is known exactly, so every stored run has a prediction of its own, without
# sampling. Only for extended source vs point source; Bayes factors against the fit with redistribution carry the Occam term of the F prior and are
# reported as measured.
def run_kind(fits):
    """What a stored run is beyond its tag, read from the pickles themselves: "null_test" (the data were drawn from the point-source model), "free_geometry" (a/R*, Rp/R* or T_s
    sampled) or "" (the usual run: hyperillumination truth, fixed geometry). The subfolder a run sits in must say the same (series_table checks it)."""
    f = next(iter(fits.values()))
    if f.meta["data"]["truth"]["insolation_model"] != "hyperillum": return "null_test"
    if any(n in f.pset.names for n in _GEOMETRY + _ORBIT): return "free_geometry"
    return ""


def alternative_model(fits):
    """(truth model, misspecified one-parameter model) of a run: the point-source model for a hyperillumination truth, the extended-source model for a point-source truth (null test)."""
    true_m = next(iter(fits.values())).meta["data"]["truth"]["insolation_model"]
    return true_m, ("naive" if true_m == "hyperillum" else "hyperillum")


def evidence_reference(fits, fisher=True, floor=True, A_B_bounds=A_B_FIT_BOUNDS, redist=False):
    """The seed-independent half of run_prediction, for the configuration of a stored run {model: NestedFit}: the noiseless truth regenerated at its own resolution
    (noiseless_reference), the misspecified model (alternative_model) fitted to it, its whitened residual r_perp (bias.evidence.evidence_residual) and its Fisher
    sigma_A_B, all at the truth's resolution. floor: also the numerical floor, chi2 of the truth's own model at the injected parameters on the likelihood's coarser grid
    against the truth curve (model resolution, not model bias). Runs that differ only in seed and pfrac share it.
    A_B_bounds: the range of the noiseless fit (A_B_FIT_BOUNDS: the bias analysis; A_B_BOUNDS: held at the physical floor, where r_perp is no longer orthogonal to the A_B response).
    A run that samples the geometry (a_rs, rp_rs and incl or b with Gaussian priors, read from its ParameterSet) gets the prediction WITH those nuisances: the linear nuisance fit of
    bias.nuisance (nuisance_problem / nuisance_solve) around the fixed-geometry fit, so that A_B_fit, bias_A_B, chi2, sigma_fisher and r_perp (the data rows of the augmented
    residual) are those of the marginalised fit; the fixed-geometry numbers are kept as chi2_fixed and sigma_fisher_fixed, and `nuisance` holds {name: (prior sigma, predicted
    posterior sigma)}.
    redist: also the noiseless fit of the point-source model with redistribution (fit_redist) at the same resolution, for a run that sampled it (hyperillumination truth, fixed
    geometry): out["redist"] = dict(A_B_fit, F_fit, chi2, sigma_fisher [of A_B, marginalised over F], r_F), where r_F = r_perp - r_perp(with F) is the part of the A_B-only residual
    that F absorbs (|r_F|^2 = chi2 - chi2_redist = z_F^2 to first order); run_prediction projects the noise on it. About 250 more model evaluations per configuration.
    Returns dict(alt, A_B_true, A_B_fit, bias_A_B, chi2, r_perp, y_true [concatenated], sigma_fisher, chi2_floor[, chi2_fixed, sigma_fisher_fixed, nuisance][, redist])."""
    true_m, alt = alternative_model(fits)
    f = fits[alt]
    cfg, data = f.meta["cfg"], f.meta["data"]
    geom = [n for n in f.pset.names if n in _GEOMETRY + _ORBIT]
    with_F = redist and true_m == "hyperillum" and "naive_redist" in fits and not geom
    ref = noiseless_reference(cfg, data, [alt, "naive_redist"] if with_F else [alt], A_B_bounds=A_B_bounds)
    nl = ref["noiseless"][alt]
    ev = evidence_residual(ref["fit_cfg"], ref["fit_truth"], alt, A_B_fit=nl["A_B"], insol=ref["insols"][alt], **ref["truth_kw"])
    if not np.isclose(ev["chi2"], nl["chi2"], rtol=1e-9): raise ValueError(f"the residual's chi2 {ev['chi2']} is not the fit's {nl['chi2']}")
    y_true = np.concatenate(ref["y_true"])
    out = dict(alt=alt, A_B_true=float(data["truth"]["A_B"]), A_B_fit=nl["A_B"], bias_A_B=nl["A_B"] - float(data["truth"]["A_B"]), chi2=nl["chi2"], r_perp=ev["r_perp"], y_true=y_true,
               sigma_fisher=float(fisher_sigma(ref["fit_cfg"], alt, nl["A_B"], bounds=A_B_bounds, insol=ref["insols"][alt], **ref["truth_kw"])[0]) if fisher else np.nan, chi2_floor=np.nan)
    if with_F:
        nr = ref["noiseless"]["naive_redist"]
        r_redist = evidence_residual(ref["fit_cfg"], ref["fit_truth"], "naive_redist", A_B_fit=nr["A_B"], F=nr["F"], **ref["truth_kw"])["r_perp"]
        out["redist"] = dict(A_B_fit=nr["A_B"], F_fit=nr["F"], chi2=nr["chi2"], r_F=out["r_perp"] - r_redist,
                             sigma_fisher=float(fisher_sigma(ref["fit_cfg"], "naive_redist", nr["A_B"], F=nr["F"], bounds=A_B_bounds, F_bounds=F_BOUNDS, insol=None, **ref["truth_kw"])[0]) if fisher else np.nan)
    if geom:
        prior = {p.name: p.prior for p in f.pset.params if p.name in geom}
        bad = [n for n in geom if n not in ("a_rs", "rp_rs", "incl", "b") or not isinstance(prior[n], (Normal, TruncNormal))]
        if bad: raise ValueError(f"no linear prediction for the sampled geometry {bad}: only a_rs, rp_rs, incl and b with a Normal / TruncNormal prior are supported")
        col = {n: ("a_rs|b" if n == "a_rs" and "b" in geom else n) for n in geom}   # the nuisance column of each sampled name: with b sampled, a/R* varies at fixed b
        prob = nuisance_problem(ref["fit_cfg"], ref["fit_truth"], A_B_fit=nl["A_B"], priors={col[n]: prior[n].sd for n in geom}, bounds=A_B_bounds, insol=ref["insols"][alt], **ref["truth_kw"])
        sol = nuisance_solve(prob, [col[n] for n in geom])   # a TruncNormal enters as its Gaussian: the truncation is ignored (5 sigma, or the one-sided i <= 90 deg of K2-141 b)
        out.update(chi2_fixed=out["chi2"], sigma_fisher_fixed=out["sigma_fisher"], A_B_fit=sol["A_B_fit"], bias_A_B=sol["A_B_fit"] - out["A_B_true"], chi2=sol["chi2"], r_perp=sol["r_perp"],
                   sigma_fisher=sol["sigma_A_B"], nuisance={n: (float(prior[n].sd), float(sol["sigma_post"][col[n]])) for n in geom})
    if floor and true_m in fits:
        L = logl_from_fit(fits[true_m])   # warns (stderr) if the forward model changed since the run; run_prediction also records it per run (grid_change)
        pl = detector_cfgs(cfg)[0].planet
        base = dict(A_B=data["truth"]["A_B"], F=data["truth"].get("F") or 0., a_rs=pl.R / pl.r2, rp_rs=pl.r1 / pl.r2, T_s=pl.T2_eff, incl=pl.incl, b=pl.R / pl.r2 * np.cos(np.radians(pl.incl)),
                    **{n: data["truth"]["offset"] for n in L.names if n.startswith("offset")})
        out["chi2_floor"] = float(np.sum(((L.model(theta_of(L, base)) - y_true) / np.asarray(data["sigma"], float)) ** 2))
    return out


def _quantiles(fit, name, q):
    """Weighted posterior quantiles `q` of one sampled parameter of a NestedFit, as floats."""
    return [float(v) for v in dyfunc.quantile(fit.results.samples[:, fit.pset.names.index(name)], list(q), weights=fit.results.importance_weights())]


def run_prediction(fits, ref=None, K=200, seed=0, z_tol=1e-6):
    """One stored run {model: NestedFit} against the linear prediction of its Bayes factor, as a row (dict).

    ref   : evidence_reference(fits) (computed here if None; pass it to share it between the seeds of one configuration, or to replace chi2 / r_perp by a prediction with
            nuisance parameters, e.g. for the free-geometry runs).
    K     : jittered realisations for the sampler error of ln B (bayes_factor); 0 skips it.
    z_tol : the standardised noise z = (y_obs - regenerated truth) / sigma must reproduce the draws of default_rng(seed) to this absolute tolerance - which checks the regenerated
            truth against the one the data were drawn from (a forward model changed since the run fails here).
    Columns: grid_change (check_fit_current: 0 unless the stellar grid changed since the run, i.e. the pickle is stale), chi2_min, lnB_expected = chi2_min / 2, lnB_sd = sqrt(chi2_min), zeta = r_perp_hat . z (direct), lnB_pred = chi2_min / 2 + zeta sqrt(chi2_min); lnB (measured, truth's
    model vs the misspecified one), lnB_err (analytic), lnB_err_jitter, dlnL_max and zeta_inferred = (2 dlnL_max - chi2_min) / (2 sqrt(chi2_min)), dH (information, truth - alt);
    A_B medians of both models, dA_median = alt - truth's model (the paired difference: the noise largely cancels) against dA_noiseless (the noiseless bias); sigma_post (half the
    16-84 % width of A_B under alt) against sigma_fisher. lnB_ext_vs_ps is ln B in the fixed orientation extended source vs point source (= -lnB for a null test). With a
    "naive_redist" fit: lnB_vs_redist (truth's model vs ps+F), lnB_redist_vs_ps, F_med, F_sigma (half width), F_95 (upper limit) - measured only, no prediction - and
    A_B_med_redist, dA_median_redist (ps+F minus the truth's model), sigma_post_redist, T_night_lo / _med / _hi (16, 50, 84 %: the uniform nightside of Kreidberg & Loeb eq. 1),
    zF_obs = sqrt(2 dlnL_max) between the best fits with and without redistribution (the significance of F in this realisation, one-sided) and occam_F = lnB_redist_vs_ps - dlnL_max.
    With ref["redist"] (evidence_reference(..., redist=True)): dA_noiseless_redist, F_noiseless, chi2_min_redist, sigma_fisher_redist, zF_noiseless = sqrt(chi2_min - chi2_min_redist),
    zeta_F = r_F_hat . z and zF_pred = max(zF_noiseless + zeta_F, 0), the linear prediction of zF_obs (F >= 0, hence the floor).
    A_B_lo95_<m>, A_B_lo_<m>, A_B_hi_<m>, A_B_hi95_<m> (m = truth, alt, redist) and F_lo95, F_lo, F_hi, F_hi95 are the 95 % and 68 % posterior intervals (plot_ns_realizations).
    chi2_best_<m> and p_best_<m> (m = truth, alt, redist): chi2 of the model's best sample against the noisy data and its p-value for n_data - k degrees of freedom (k: every sampled
    parameter), the goodness-of-fit test an observer could make with that model alone."""
    true_m, alt = alternative_model(fits)
    if alt not in fits or true_m not in fits: raise KeyError(f"the run has {tuple(fits)}; the prediction needs {true_m!r} and {alt!r}")
    if ref is None: ref = evidence_reference(fits)
    f_t, f_a = fits[true_m], fits[alt]
    data = f_a.meta["data"]
    tr = data["truth"]
    c0 = detector_cfgs(f_a.meta["cfg"])[0]

    z = (np.asarray(data["y_obs"], float) - ref["y_true"]) / np.asarray(data["sigma"], float)
    z_dev = float(np.max(np.abs(z - np.random.default_rng(tr["seed"]).normal(0., 1., z.size))))   # one stream across the detectors of a joint dataset
    if z_dev > z_tol: raise ValueError(f"the regenerated truth does not reproduce the data of seed {tr['seed']}: max |z - z_rng| = {z_dev:.3g} (forward model changed since the run?)")
    chi2, r = ref["chi2"], ref["r_perp"]
    with contextlib.redirect_stderr(io.StringIO()): grid_change = max(check_fit_current(f) for f in (f_t, f_a))   # recorded in the row instead of printed from a worker
    zeta = float(r @ z / np.sqrt(chi2))   # chi2 = |r_perp|^2, except with prior rows (free geometry): there r_perp holds the data rows only and chi2 the whole augmented residual, so that ln B = chi2 / 2 + r_perp . z
    dlnL = float(np.max(f_t.results.logl) - np.max(f_a.results.logl))
    lnB = f_t.meta["lnZ"] - f_a.meta["lnZ"]
    sign = 1. if true_m == "hyperillum" else -1.
    s_t, s_a = (posterior_summary(f, verbose=False)["params"]["A_B"] for f in (f_t, f_a))
    row = dict(planet=c0.planet.name, instrument=joint_name([band_of(c) for c in detector_cfgs(f_a.meta["cfg"])]) if is_detector_set(f_a.meta["cfg"]) else band_of(c0),
               kind=run_kind(fits), truth=true_m, alt=alt, A_B_true=float(tr["A_B"]), n_bins=c0.n_bins, n_orbits=c0.n_orbits, seed=int(tr["seed"]), pfrac=float(f_a.scfg.pfrac),
               converged=bool(f_t.meta.get("converged", True) and f_a.meta.get("converged", True)), z_dev=z_dev, grid_change=float(grid_change),
               chi2_min=float(chi2), chi2_floor=float(ref.get("chi2_floor", np.nan)), lnB_expected=float(0.5 * chi2), lnB_sd=float(np.sqrt(chi2)), zeta=zeta, lnB_pred=float(0.5 * chi2 + zeta * np.sqrt(chi2)),
               lnB=float(lnB), lnB_err=float(np.hypot(f_t.meta["lnZ_err"], f_a.meta["lnZ_err"])), lnB_err_jitter=np.nan,
               dlnL_max=dlnL, zeta_inferred=float((2. * dlnL - chi2) / (2. * np.sqrt(chi2))), dH=float(f_t.meta["information"] - f_a.meta["information"]),
               A_B_med_truth=s_t["med"], A_B_med_alt=s_a["med"], dA_median=s_a["med"] - s_t["med"], dA_noiseless=ref["bias_A_B"],
               sigma_post=0.5 * (s_a["hi"] - s_a["lo"]), sigma_post_truth=0.5 * (s_t["hi"] - s_t["lo"]), sigma_fisher=ref["sigma_fisher"])
    interval = lambda fit, name, tag: dict(zip((f"{name}_lo95{tag}", f"{name}_lo{tag}", f"{name}_hi{tag}", f"{name}_hi95{tag}"), _quantiles(fit, name, (0.025, 0.16, 0.84, 0.975))))
    row.update(**interval(f_t, "A_B", "_truth"), **interval(f_a, "A_B", "_alt"))
    def best(fit, tag):   # chi2 of a fit's best sample and its p-value (ln L = -(chi2 + sum ln 2 pi sigma^2) / 2: GaussianLogL keeps the normalisation)
        c = float(-2. * np.max(fit.results.logl) - np.sum(np.log(2. * np.pi * np.asarray(data["sigma"], float) ** 2)))
        return {f"chi2_best{tag}": c, f"p_best{tag}": float(chi2_dist.sf(c, z.size - len(fit.pset.names)))}
    row.update(n_data=int(z.size), **best(f_t, "_truth"), **best(f_a, "_alt"))
    row["lnB_ext_vs_ps"], row["lnB_pred_ext_vs_ps"] = sign * row["lnB"], sign * row["lnB_pred"]
    if K: row["lnB_err_jitter"] = bayes_factor(fits, true_m, alt, K=K, seed=seed, verbose=False)["err_jitter"]
    if "chi2_fixed" in ref: row.update(chi2_min_fixed=float(ref["chi2_fixed"]), sigma_fisher_fixed=float(ref["sigma_fisher_fixed"]))
    for n, (s_prior, s_pred) in ref.get("nuisance", {}).items():   # sampled geometry: the posterior width under the misspecified model against its prior and against the linear prediction
        w, k = f_a.results.importance_weights(), f_a.pset.names.index(n)
        lo, med, hi = dyfunc.quantile(f_a.results.samples[:, k], [0.16, 0.5, 0.84], weights=w)
        row.update({f"{n}_med": float(med), f"{n}_prior": s_prior, f"{n}_shrink": float(0.5 * (hi - lo) / s_prior), f"{n}_shrink_pred": s_pred / s_prior})
    row.update(lnB_vs_redist=np.nan, lnB_redist_vs_ps=np.nan, F_med=np.nan, F_sigma=np.nan, F_95=np.nan)   # always present, so a table of runs without a fit with redistribution keeps the columns
    if "naive_redist" in fits:
        f_r = fits["naive_redist"]
        q16, q50, q84, q95 = _quantiles(f_r, "F", (0.16, 0.5, 0.84, 0.95))
        row.update(lnB_vs_redist=float(f_t.meta["lnZ"] - f_r.meta["lnZ"]), lnB_redist_vs_ps=float(f_r.meta["lnZ"] - fits["naive"].meta["lnZ"]) if "naive" in fits else np.nan,
                   F_med=float(q50), F_sigma=float(0.5 * (q84 - q16)), F_95=float(q95))
        s_r = posterior_summary(f_r, verbose=False)["params"]["A_B"]
        row.update(A_B_med_redist=s_r["med"], dA_median_redist=s_r["med"] - s_t["med"], sigma_post_redist=0.5 * (s_r["hi"] - s_r["lo"]), **interval(f_r, "A_B", "_redist"), **interval(f_r, "F", ""), **best(f_r, "_redist"))
        x = (1. - f_r.results.samples[:, f_r.pset.names.index("A_B")]) * f_r.results.samples[:, f_r.pset.names.index("F")]   # the nightside temperature depends on the two only through (1 - A_B) F, and monotonically: transform its quantiles
        row.update(zip(("T_night_lo", "T_night_med", "T_night_hi"), (float(surface_temperature(np.pi, c0.planet, "naive_redist", A_B=0., emissivity_bol=c0.emissivity_bol, F=float(v)))
                                                                      for v in dyfunc.quantile(x, [0.16, 0.5, 0.84], weights=f_r.results.importance_weights()))))
        if "naive" in fits:
            dlnL_F = float(np.max(f_r.results.logl) - np.max(fits["naive"].results.logl))
            row.update(zF_obs=float(np.sqrt(max(2. * dlnL_F, 0.))), occam_F=row["lnB_redist_vs_ps"] - dlnL_F)   # ln B(ps+F vs ps) = dlnL_max + the Occam term of the F prior (< 0)
        if "redist" in ref:   # the noiseless fit with redistribution: the bias of its A_B, its F, and the significance of F predicted for this realisation's noise
            rd = ref["redist"]
            zF, zeta_F = float(np.sqrt(max(chi2 - rd["chi2"], 0.))), float(rd["r_F"] @ z / np.linalg.norm(rd["r_F"]))
            row.update(dA_noiseless_redist=rd["A_B_fit"] - float(tr["A_B"]), F_noiseless=rd["F_fit"], chi2_min_redist=rd["chi2"], sigma_fisher_redist=rd["sigma_fisher"],
                       zF_noiseless=zF, zeta_F=zeta_F, zF_pred=max(zF + zeta_F, 0.))
    return row


def _reference_of(pattern, redist=False):
    return evidence_reference(load_fits(pattern), redist=redist)


def _prediction_of(pattern, ref, K):
    return run_prediction(load_fits(pattern), ref=ref, K=K)


def series_table(results_dir, series, K=200, n_jobs=-1, redist=False):
    """Every stored run of `series` against its prediction (run_prediction), one row per run with its series name and dataset stem in front.

    series : {series name: [find_dataset keyword dicts]} (planet, instrument, the settings that deviate from the baseline, and subfolder= for the runs in a subfolder).
    Runs that are not there (still queued) or lack one of the two models are listed and skipped, so the table can be rebuilt as the queue finishes. A run whose pickles contradict
    its subfolder (run_kind) raises. The noiseless reference is computed once per configuration (the runs of one planet, instrument and setting that differ only in seed and pfrac
    share it), configurations and runs in parallel. redist: also the noiseless fit with redistribution per configuration (evidence_reference), which adds the columns that compare
    the sampled point-source model with redistribution with it. Returns (table, missing) with missing = [(series name, keywords, reason)]."""
    runs, missing = [], []
    for name, entries in series.items():
        for kw in entries:
            try: stem = find_dataset(results_dir, **kw)
            except FileNotFoundError: missing.append((name, kw, "no run")); continue
            pattern = os.path.join(results_dir, stem + "_*.pkl")
            models = sorted(d["insolation_model"] for d in map(parse_result_file, glob.glob(pattern)) if d)
            if not {"hyperillum", "naive"} <= set(models): missing.append((name, kw, f"incomplete: only {', '.join(models)}")); continue
            runs.append((name, kw, stem, pattern))
    if not runs: return pd.DataFrame(), missing
    config = lambda stem: re.sub(r"_seed\d+_pfrac[\d.]+$", "", stem)   # the tag without seed and pfrac (with its subfolder): what the noiseless reference depends on
    first = {}
    for _, _, stem, pattern in runs: first.setdefault(config(stem), pattern)
    refs = dict(zip(first, pmap(_reference_of, [(p, redist) for p in first.values()], n_jobs)))
    rows = pmap(_prediction_of, [(pattern, refs[config(stem)], K) for _, _, stem, pattern in runs], n_jobs)
    for (name, kw, stem, _), row in zip(runs, rows):
        if row["kind"] != (kw.get("subfolder") or ""): raise ValueError(f"{stem}: the pickles say {row['kind'] or 'a baseline run'!r}, the entry says subfolder={kw.get('subfolder')!r}")
    tab = pd.DataFrame([dict(series=name, stem=stem, **row) for (name, _, stem, _), row in zip(runs, rows)])
    return tab, missing


def series_summary(tab, by="series"):
    """Mean and standard deviation over the runs of each series of a series_table (sd NaN for a single run; by: the grouping column(s), e.g. ["planet", "instrument"]): n, the measured
    and predicted ln B (extended source vs point source) and their difference, zeta, the paired median difference against the noiseless bias, and sigma_post / sigma_fisher. For a
    table with the noiseless fit with redistribution (series_table(..., redist=True)) also that fit's paired median difference, bias and width ratio, the posterior median of F and
    its noiseless value, the significance of F (zF_obs, its noiseless value and zF_obs - zF_pred), the Bayes factors with the point-source model with redistribution and occam_F."""
    t = tab.assign(dlnB=tab["lnB"] - tab["lnB_pred"], dzeta=tab["zeta"] - tab["zeta_inferred"], sigma_ratio=tab["sigma_post"] / tab["sigma_fisher"])
    cols = dict(lnB_ext_vs_ps="lnB", lnB_pred_ext_vs_ps="lnB_pred", lnB_expected="lnB_expected", dlnB="lnB - pred", zeta="zeta", dzeta="zeta - zeta_inferred", dA_median="dA_median", dA_noiseless="dA_noiseless", sigma_ratio="sigma_post/sigma_F")
    if "zF_pred" in t:
        t = t.assign(dzF=t["zF_obs"] - t["zF_pred"], sigma_ratio_redist=t["sigma_post_redist"] / t["sigma_fisher_redist"])
        cols.update(dA_median_redist="dA_median_redist", dA_noiseless_redist="dA_noiseless_redist", sigma_ratio_redist="sigma_post/sigma_F redist", F_med="F_med", F_noiseless="F_noiseless",
                    zF_obs="zF_obs", zF_noiseless="zF_noiseless", dzF="zF_obs - pred", lnB_redist_vs_ps="lnB_redist_vs_ps", lnB_vs_redist="lnB_vs_redist", occam_F="occam_F")
    g = t.groupby(by, sort=False)
    out = pd.DataFrame({"n": g.size(), "seeds": g["seed"].agg(lambda s: ",".join(str(v) for v in sorted(set(s)))), "pfrac": g["pfrac"].agg(lambda s: "/".join(f"{v:g}" for v in sorted(set(s))))})
    for col, name in cols.items():
        out[name], out[name + " sd"] = g[col].mean(), g[col].std(ddof=1)
    return out
