"""The nested-sampling driver (dynesty) and the post-processing of its results."""
import contextlib
import copy
import os
import pickle
import time
import warnings
from dataclasses import dataclass
from datetime import datetime

import dynesty
import numpy as np
from dynesty import utils as dyfunc
from dynesty.dynamicsampler import stopping_function
from dynesty.pool import Pool
from dynesty.utils import get_neff_from_logwt

from hyperphase.bias.detectors import detector_cfgs
from hyperphase.retrieval.priors import ParameterSet
from hyperphase.stellar import T_bright


############################################################################################################################################
#####   Nested sampling driver
############################################################################################################################################
@dataclass
class SamplerConfig:
    """Everything tunable about the dynesty run."""
    dynamic: bool = True
    pfrac: float = 0.8       # Importance of evidence (pfrac=0) vs posterior (pfrac=1)
    nlive: int = 500              # nlive (static) / nlive_init (dynamic)
    nlive_batch: int = 250        # dynamic only
    maxbatch: int = 15            # dynamic only
    dlogz: float = 0.05      # dlogz (static) / dlogz_init (dynamic)
    n_effective: int = None       # dynamic only; None -> target_n_effective = max(ndim², 10000) = 10000
    bound: str = "multi"
    sample: str = "auto"          # 'auto' -> 'unif' for ndim<10, 'rslice' for 10-20
    nprocs: int = 1               # >1 -> dynesty.pool.Pool (spawned workers)
    seed: int = 42
    maxiter: int = None
    maxcall: int = None
    checkpoint_file: str = None   # resume with run_nested_fit(..., resume=True)
    print_progress: bool = True
    outfile: str = None           # pickle the finished NestedFit here; None -> not saved. {insolation_model}, {date} and {seed} are filled in, so ONE scfg can serve several models without them overwriting each other


@dataclass
class NestedFit: # output of fitting
    results: object  # the dynesty Results object
    pset: ParameterSet # the priors
    scfg: SamplerConfig # dynesty params
    meta: dict  # insolation_model, caches, seed, runtime, ncall, lnZ - everything needed to reproduce


def run_status(res, scfg, stop_kwargs=None):
    """Did the run stop because dynesty's criterion was met, or because it ran out of budget? Re-evaluates the same stopping_function on the FINAL results - the check run_nested would have made before the next batch - free at the default n_mc=0 (no likelihood calls). Returns {'converged': bool, 'status': str}."""
    caps = [n for n, cap, v in (("maxiter", scfg.maxiter, int(res.niter)), ("maxcall", scfg.maxcall, int(np.sum(res.ncall)))) if cap is not None and v >= cap]
    if not scfg.dynamic: return dict(converged=not caps, status=f"static, {res.niter} it" + (f" | RAN OUT OF {'+'.join(caps)}" if caps else f" | dlogz {scfg.dlogz} reached"))
    n_batches = len(res.batch_logl_bounds) - 1   # row 0 is the (-inf, inf) base run
    if n_batches >= scfg.maxbatch: caps.append("maxbatch")
    stop = stopping_function(res, dict(stop_kwargs or {"pfrac": scfg.pfrac}), return_vals=True)[1][2]   # pfrac*target_ESS/ESS + (1-pfrac)*lnZerr/evid_thresh; dynesty stops at <= 1
    ok = bool(stop <= 1.)
    return dict(converged=ok, status=("CONVERGED" if ok else f"NOT CONVERGED, ran out of {'+'.join(caps)}") + f" | stop {stop:.3f}, {n_batches}/{scfg.maxbatch} batches, ESS {get_neff_from_logwt(res.logwt):.0f}") # ESS: Kish effective sample size of the importance-weighted samples


def run_nested_fit(logl, pset, scfg=None, resume=False):
    """One dataset + one insolation model + one ParameterSet -> one dynesty NestedFit.

    logl : GaussianLogL (build it with GaussianLogL.from_pset so its names cannot drift from pset)
    pset : ParameterSet supplying the prior transform; its names must match logl.names
    scfg : SamplerConfig; defaults to SamplerConfig()
    """
    scfg = scfg or SamplerConfig()
    if tuple(pset.names) != tuple(logl.names): raise ValueError(f"pset names {pset.names} != logl names {logl.names}; theta columns would be scrambled")
    if not 0. <= scfg.pfrac <= 1.: raise ValueError(f"pfrac must be in [0, 1], got {scfg.pfrac}")
    if resume and not scfg.checkpoint_file: raise ValueError("resume=True needs scfg.checkpoint_file")

    # Resolve the save path BEFORE sampling, so a bad directory fails in a second rather than after an hour of CPU
    path = None
    if scfg.outfile:
        path = os.path.abspath(os.path.expanduser(scfg.outfile.format(insolation_model=logl.insolation_model, date=datetime.now().strftime("%Y-%m-%d"), seed=scfg.seed)))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    stop_kwargs = {"pfrac": scfg.pfrac}   # run_nested resolves target_n_effective INTO this dict in place, so afterwards it is exactly the criterion that was applied
    rstate = np.random.default_rng(scfg.seed)
    use_pool = (scfg.nprocs or 1) > 1
    t0 = time.perf_counter()

    with (Pool(scfg.nprocs, logl, pset.prior_transform) if use_pool else contextlib.nullcontext()) as pool:
        loglike, ptform = (pool.loglike, pool.prior_transform) if use_pool else (logl, pset.prior_transform)
        kw = dict(bound=scfg.bound, sample=scfg.sample, rstate=rstate)
        if use_pool: kw.update(pool=pool, queue_size=pool.size)

        if scfg.dynamic:
            sampler = (dynesty.DynamicNestedSampler.restore(scfg.checkpoint_file, pool=pool) if resume else dynesty.DynamicNestedSampler(loglike, ptform, pset.ndim, **kw))
            # wt_kwargs={"pfrac": scfg.pfrac}: where the next batch is placed; stop_kwargs={"pfrac": scfg.pfrac}: when to stop
            sampler.run_nested(nlive_init=scfg.nlive, dlogz_init=scfg.dlogz, nlive_batch=scfg.nlive_batch, maxbatch=scfg.maxbatch, n_effective=scfg.n_effective, maxiter=scfg.maxiter, maxcall=scfg.maxcall, wt_kwargs={"pfrac": scfg.pfrac}, stop_kwargs=stop_kwargs, print_progress=scfg.print_progress, checkpoint_file=scfg.checkpoint_file, resume=resume)
        else:
            sampler = (dynesty.NestedSampler.restore(scfg.checkpoint_file, pool=pool) if resume else dynesty.NestedSampler(loglike, ptform, pset.ndim, nlive=scfg.nlive, **kw))
            # add_live=True must stay on, or the posterior peak is truncated
            sampler.run_nested(dlogz=scfg.dlogz, maxiter=scfg.maxiter, maxcall=scfg.maxcall, add_live=True, print_progress=scfg.print_progress, checkpoint_file=scfg.checkpoint_file, resume=resume)
        res = sampler.results
    runtime = time.perf_counter() - t0

    meta = dict(insolation_model=logl.insolation_model, caches=logl.cache_used, fixed=dict(logl.fixed), seed=scfg.seed, runtime_s=runtime,
                ncall=int(np.sum(res.ncall)), niter=int(res.niter), lnZ=float(res.logz[-1]), lnZ_err=float(res.logzerr[-1]), information=float(res.information[-1]), n_bins=int(logl.y.size), sigma_bin=float(np.mean(logl.sigma)),
                data=copy.deepcopy(logl.data),   # the fit travels with its data (and the injected truth; per detector for a joint fit), so a reloaded NestedFit can be re-plotted without re-running the notebook
                cfg=logl.cfg, model_kw=dict(logl.model_kw), labels=getattr(logl, "labels", None))   # everything logl_from_fit needs to rebuild the likelihood from the pickle alone (cfg is the detector set and labels the detector names for a joint fit)

    # The geometry actually used: sampled, held fixed, or (the usual case) inherited from cfg.planet - an empty `fixed` on its own would not record
    # which.
    pl = detector_cfgs(logl.cfg)[0].planet
    meta["geometry"] = {k: ("sampled" if k in logl.names else float(logl.fixed.get(k, d))) for k, d in (("a_rs", pl.R / pl.r2), ("T_s", pl.T2_eff), ("rp_rs", pl.r1 / pl.r2), ("incl", pl.incl), ("b", pl.R / pl.r2 * np.cos(np.radians(pl.incl))))}
    for k, other in (("incl", "b"), ("b", "incl")):   # a sampled inclination makes the other spelling a derived quantity, not a fixed one
        if other in logl.names: meta["geometry"][k] = f"derived from {other}"

    meta.update(run_status(res, scfg, stop_kwargs))
    if scfg.print_progress:
        print(f"\n{logl.insolation_model}: lnZ = {meta['lnZ']:.4f} +/- {meta['lnZ_err']:.4f},  {meta['ncall']} calls, {runtime / 60:.1f} min ({runtime / max(meta['ncall'], 1) * 1e3:.0f} ms/call)")
        print(f"   {meta['status']}")
    if not meta["converged"]: warnings.warn(f"{logl.insolation_model}: dynesty stopped WITHOUT meeting its stopping criterion -- {meta['status']}. Raise scfg.maxbatch, or relax the target with scfg.n_effective. lnZ and the posterior are still valid, just noisier than requested.", RuntimeWarning, stacklevel=2)

    fit = NestedFit(results=res, pset=pset, scfg=scfg, meta=meta)
    if path:
        try:
            with open(path, "wb") as f: pickle.dump(fit, f)
            meta["file"] = path
            if scfg.print_progress: print(f"   saved -> {os.path.relpath(path)}")   # relative to the working directory, so notebook outputs carry no local path
        except Exception as e:   # a failed save must never destroy an hour-long run
            warnings.warn(f"could not save to {path}: {type(e).__name__}: {e}; the NestedFit is still returned", RuntimeWarning, stacklevel=2)
    return fit


############################################################################################################################################
#####   Posterior post-processing
############################################################################################################################################
def equal_samples(fit, n=None, seed=0):
    """Equal-weight posterior draws, shape (n, ndim), columns in fit.pset.names order. n=None -> all of them.
    res.samples is NOT posterior-distributed (it is ~uniform in ln X); this is the weighted cloud turned into plain draws, so every downstream mean/quantile/histogram can ignore weights."""
    rng = np.random.default_rng(seed)
    s = fit.results.samples_equal(rstate=rng)
    if n is None or n >= len(s): return s
    return s[rng.choice(len(s), int(n), replace=False)]


def posterior_summary(fit, truth=None, quantiles=(0.16, 0.5, 0.84), verbose=True):
    """Weighted posterior summary of one fit: per-parameter median +/- 68%, mean +/- sd, and - with `truth` - the BIAS in units of sigma.
    Returns dict(params={name: {...}}, corr=(ndim, ndim) correlation matrix, ess=effective sample size).
    """
    res, names = fit.results, fit.pset.names
    w = res.importance_weights() # res.samples is not a set of posterior samples, they need to be weighted
    mean, cov = dyfunc.mean_and_cov(res.samples, w)
    sd = np.sqrt(np.diag(cov))

    params = {}
    for i, nm in enumerate(names):
        lo, med, hi = dyfunc.quantile(res.samples[:, i], list(quantiles), weights=w) # get 1-sigma bounds for two one-sided error bars
        d = dict(med=float(med), lo=float(lo), hi=float(hi), minus=float(med - lo), plus=float(hi - med), mean=float(mean[i]), sd=float(sd[i]))
        if truth is not None and nm in truth:
            d["truth"] = float(truth[nm])
            d["bias"] = d["med"] - d["truth"] # retrieved (posterior median) minus true value
            d["bias_sigma"] = d["bias"] / (0.5 * (d["hi"] - d["lo"])) # bias in units of bias error/ sigmas
        params[nm] = d

    # corr: correlation matrix of parameter posteriors (corr_ij = cov_ij / sigma_i*sigma_j)
    # ess: Kish effective sample size (how many independent draws the weighted cloud is worth) -> sets the MC noise
    out = dict(params=params, corr=cov / np.outer(sd, sd), ess=float(get_neff_from_logwt(res.logwt)))
    if verbose:
        print(f"{fit.meta['insolation_model']}   ESS {out['ess']:.0f}, information {fit.meta['information']:.2f} nats")
        for nm, d in params.items():
            b = f"   truth {d['truth']:g}, bias {d['bias']:+.4f} = {d['bias_sigma']:+.2f} sigma" if "truth" in d else ""
            print(f"   {nm:8s} = {d['med']:9.4f} +{d['plus']:.4f} -{d['minus']:.4f}{b}")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                print(f"   corr({names[i]}, {names[j]}) = {out['corr'][i, j]:+.3f}")
    return out


def evidence_jitter(fit, K=200, seed=0):
    """K realisations of lnZ from dynesty's own statistical uncertainty: jitter_run resamples the prior-volume shrinkage t_i ~ Beta(N, 1) that nested sampling only knows statistically.
    std of the output should reproduce res.logzerr[-1]; the point of having the full sample is that a DIFFERENCE of two lnZ can then be quoted without assuming either is Gaussian."""
    rng = np.random.default_rng(seed)
    return np.array([float(dyfunc.jitter_run(fit.results, rstate=rng).logz[-1]) for _ in range(int(K))])


def bayes_factor(fits, num="hyperillum", den="naive", K=200, seed=0, verbose=True):
    """ln B = lnZ[num] - lnZ[den] with both error estimates: dynesty's analytic logzerr added in quadrature, and the spread of K jittered realisations.
    The two runs are independent, so the jittered differences are simply paired draw by draw. Returns dict(lnB, err_analytic, err_jitter, samples)."""
    if num not in fits or den not in fits: raise ValueError(f"fits has {tuple(fits)}, need {num!r} and {den!r}")
    lnB = fits[num].meta["lnZ"] - fits[den].meta["lnZ"]
    err_a = float(np.hypot(fits[num].meta["lnZ_err"], fits[den].meta["lnZ_err"]))
    s = evidence_jitter(fits[num], K, seed) - evidence_jitter(fits[den], K, seed + 1) # get draws of actual random realisation of dX_i (dynesty error in lnZ)
    out = dict(lnB=float(lnB), err_analytic=err_a, err_jitter=float(np.std(s, ddof=1)), samples=s)
    if verbose:
        print(f"ln B ({num} vs {den}) = {lnB:+.3f} +/- {err_a:.3f} (analytic) +/- {out['err_jitter']:.3f} (jitter, K={K})  ->  B = {np.exp(lnB):.3g}")
        print(f"P(sign of ln B secure) = {np.mean(s > 0.):.3f}  [sampler noise only, NOT a model probability]")
    return out


def brightness_temperature(depth_th, cfg, quantiles=(0.16, 0.5, 0.84)):
    """Dayside brightness temperature [K] at the given quantiles of the THERMAL occultation depth (a blackbody brightness temperature: below the surface temperature for cfg.emissivity < 1).
    T_bright is monotone in the depth, so transforming the quantiles is exact - and it keeps the brentq/band_avg_planck cost at three calls instead of one per draw."""
    q = np.quantile(np.asarray(depth_th, float), list(quantiles))
    return np.array([T_bright(d, cfg.planet, (cfg.l1, cfg.l2), cfg.filter_path, stellar_model=cfg.stellar_model) for d in q])
