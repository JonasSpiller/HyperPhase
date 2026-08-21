import numpy as np
from dataclasses import dataclass
import time
import os
import pickle
from datetime import datetime
import dynesty
from dynesty.pool import Pool
from joblib import Parallel, delayed
import contextlib
from hyperillumination_phasecurve_utils import model_pcurve, reflected_unit, insolation_cache  # forward model + its geometry-only caches
from simple_bias_analysis import binned
from scipy.stats import norm, truncnorm
import warnings
from dynesty.dynamicsampler import stopping_function   # re-evaluated post-run to tell convergence from budget truncation
from dynesty.utils import get_neff_from_logwt
import matplotlib.pyplot as plt


###########################################################################################################################
#####
#####   Simulate Observations
#####
###########################################################################################################################

def plot_simulated_data(cfg, A_B_true, data, insolation_model="hyperillum"):
    ph = data["t"] / cfg.planet.period  # orbital phase

    # Full Phase Curve (for custom reproduction of simulate_data)
    t_hi, y_hi = model_pcurve(cfg, insolation_model, A_B_true, n_t=200 * cfg.n_bins, n_layers=80, n_lo=600, n_la=300)
    y_true = binned(cfg, y_hi)
    ph_hi = t_hi / cfg.planet.period

    # Plotting
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1], hspace=0.06))

    ax.plot(ph_hi, y_hi, lw=1., color="0.5", label=f'truth ({insolation_model}, $A_B$={A_B_true:.2f})')
    ax.step(ph, y_true, where="mid", color="C0", lw=1.4, label="binned truth")
    ax.errorbar(ph, data["y_obs"], yerr=data["sigma"], fmt="o", ms=4, color="k", capsize=2,
                elinewidth=1., label=f'simulated data ($\\sigma$={cfg.sigma_bin:.1f} ppm)')
    ax.set_ylabel("$F_p/F_\\star$ [ppm]");
    ax.legend(frameon=False, fontsize=8);
    ax.margins(x=0.01)

    r = (data["y_obs"] - y_true) / data["sigma"]
    axr.axhline(0, color="0.7", lw=0.8);
    axr.axhline(1, color="0.85", lw=0.6, ls=":")
    axr.axhline(-1, color="0.85", lw=0.6, ls=":")
    axr.plot(ph, r, "ko", ms=4)
    axr.set_xlabel("orbital phase");
    axr.set_ylabel("resid / $\\sigma$");
    axr.set_ylim(-3.5, 3.5)
    plt.show()

    # Numerical Checks
    chi2 = np.sum(r ** 2)
    print(f"sigma_bin: {cfg.sigma_bin:.2f} ppm  (bin width {cfg.planet.period * 24 * 60 / cfg.n_bins:.2f} min)")
    print(f"chi2 / n_bins (normalized average noise per bin ~1): {chi2 / cfg.n_bins:.2f}")
    print(f"residual mean / sigma (~0): {r.mean():.2f}")


def simulate_data(cfg, A_B_true, insolation_model="hyperillum", seed=42, noise=True, offset=0.,
                  k_truth=200, n_layers=80, n_lo=600, n_la=300, plotting=True, **model_kw):
    """Simulated phase-folded observation: binned(truth) + N(0, sigma_bin).

    cfg          : FittingConfig — the only mandatory input; supplies planet, filter, stellar_grid, A_(A_B), sigma_1min, n_orbits, n_bins, u_B, reflected.
    A_B_true     : true Bond albedo.
    noise        : False -> the noiseless ("Asimov") dataset. Use it for bias maps; do NOT use it for lnZ / Bayes factors, where chi2 = 0 at the truth inflates Z.
    offset       : constant baseline injected into the data [ppm]; set non-zero to check that the fitted nuisance offset is actually recovered.
    k_truth      : sub-samples per bin for the truth (n_t = k_truth * n_bins, must tile).
    """

    # high-resolution truth (thermal + reflected, since cfg.reflected=True)
    t_hi, y_hi = model_pcurve(cfg, insolation_model, A_B_true, n_t=k_truth * cfg.n_bins, n_layers=n_layers, n_lo=n_lo, n_la=n_la, **model_kw)

    # reduce to the observed bins
    y_true = binned(cfg, y_hi) + offset

    # adding noise
    s = cfg.sigma_bin
    rng = np.random.default_rng(seed)
    y_obs = y_true + rng.normal(0., s, cfg.n_bins) if noise else y_true.copy()

    t=(np.arange(cfg.n_bins) + 0.5) * cfg.planet.period / cfg.n_bins
    sigma = np.full(cfg.n_bins, s)

    data = {"t": t, "y_obs": y_obs, "sigma":sigma}

    if plotting: plot_simulated_data(cfg, A_B_true, data, insolation_model=insolation_model)

    return data


###########################################################################################################################
#####
#####   Priors
#####
###########################################################################################################################

# -------------------------------------------------------------------------------------------
#  Prior Transforms
# -------------------------------------------------------------------------------------------

# Prior transforms map the unit hypercube u ~ U(0,1) to a physical value.
# They are classes because dynesty pickles the ptform to its workers, where a lambda/closure dies.

@dataclass(frozen=True)
class Uniform:
    lo: float
    hi: float
    def __call__(self, u): return self.lo + u * (self.hi - self.lo)
    @property
    def bounds(self): return (self.lo, self.hi)


@dataclass(frozen=True)
class LogUniform:
    lo: float                                    # must be > 0
    hi: float
    def __call__(self, u): return self.lo * (self.hi / self.lo) ** u
    @property
    def bounds(self): return (self.lo, self.hi)


@dataclass(frozen=True)
class Normal:
    mu: float
    sd: float
    def __call__(self, u): return float(norm.ppf(u, self.mu, self.sd))
    @property
    def bounds(self): return (self.mu - 5. * self.sd, self.mu + 5. * self.sd)   # grid range, NOT the support


@dataclass(frozen=True)
class TruncNormal:
    """Normal(mu, sd) restricted to [lo, hi]. Use eg for r1/r2: the literature value with its error bar, hard-clipped to a physically allowed range."""
    mu: float
    sd: float
    lo: float
    hi: float
    def __call__(self, u):
        a, b = (self.lo - self.mu) / self.sd, (self.hi - self.mu) / self.sd   # scipy wants the bounds in sigma units
        return float(truncnorm.ppf(u, a, b, loc=self.mu, scale=self.sd))
    @property
    def bounds(self): return (self.lo, self.hi)


# -------------------------------------------------------------------------------------------
#  Prior Parameters
# -------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Parameter:
    name: str                # must be one of _KNOWN below
    prior: object            # callable u -> value, e.g. Uniform(0., 0.8)
    label: str = None        # LaTeX label for plots; defaults to the name


@dataclass(frozen=True)
class ParameterSet:
    """This class defines the priors: the ordered free parameters of a fit.
    Call by e.g.: pset = ParameterSet((Parameter("A_B", Uniform(0., 0.8), r"$A_B$"), Parameter("offset", Uniform(-50., 50.), "offset [ppm]")))
    """
    params: tuple

    def __post_init__(self):
        if not self.params: raise ValueError("a ParameterSet needs at least one free Parameter")
        if len(set(self.names)) != len(self.names): raise ValueError(f"duplicate parameter names in {self.names}")
        bad = [n for n in self.names if n not in _KNOWN]
        if bad: raise ValueError(f"unsupported parameter(s) {bad}; supported: {_KNOWN}")

    @property
    def ndim(self): return len(self.params)
    @property
    def names(self): return tuple(p.name for p in self.params)
    @property
    def labels(self): return tuple(p.label or p.name for p in self.params)
    @property
    def bounds(self): return [p.prior.bounds for p in self.params]     # feeds evidence_bruteforce(bounds=...)

    def prior_transform(self, u):
        """The dynesty `ptform`: unit hypercube -> physical parameters, component-wise."""
        return np.array([p.prior(ui) for p, ui in zip(self.params, u)], float)

    def __call__(self, u): # so the ParameterSet itself can be passed as ptform
        return self.prior_transform(u)

    def sample(self, n, seed=0):
        """n draws from the prior, shape (n, ndim). Feed straight to check_likelihood()."""
        u = np.random.default_rng(seed).random((int(n), self.ndim))
        return np.array([self.prior_transform(ui) for ui in u])

    def grid(self, name, n=61):
        """n evenly spaced values spanning one parameter's prior bounds — the x-axis for logl_slice / logl_grid."""
        lo, hi = self.bounds[self.names.index(name)]
        return np.linspace(lo, hi, int(n))


###########################################################################################################################
#####
#####   Log-Likelihood
#####
###########################################################################################################################

# Geometry parameters model_pcurve accepts by name (its own kwarg spelling: a_rs = R/r2, rp_rs = r1/r2).
_GEOMETRY = ("a_rs", "T_s", "rp_rs")
_KNOWN = ("A_B", "offset") + _GEOMETRY

# Which sampled parameter invalidates which geometry-only cache.
_CACHE_DEPS = {
    "refl":  {"naive": ("a_rs", "rp_rs"),        "hyperillum": ("a_rs", "rp_rs")},
    "insol": {"naive": ("a_rs", "T_s"),        "hyperillum": ("a_rs", "T_s", "rp_rs")},
}


class GaussianLogL:
    """Gaussian log-likelihood in dynesty's `loglike` format: theta (1-D array) -> float.

    Pass the instance straight to dynesty as `loglike`; `ndim` is `logl.ndim`:

        logl = GaussianLogL(cfg, data, "naive", names=("A_B", "offset"))
        sampler = dynesty.NestedSampler(logl, ptform, logl.ndim, ...)

    theta is ordered by `names`. Everything model_pcurve needs that is NOT sampled goes into `fixed`.

    cfg     : FittingConfig — planet, filter, stellar_grid, A_(A_B), sigma_1min, n_orbits, n_bins, ...
    data    : dict from simulate_data (or any dict with "t", "y_obs", "sigma")
    names   : ordered tuple of SAMPLED parameter names.
    fixed   : {name: value} for supported parameters held fixed (e.g. dict(A_B=0.1)).
    bad_logl: returned for a non-finite model.
    model_kw: forwarded to model_pcurve (n_t, n_layers, n_lo, n_la).
    """

    def __init__(self, cfg, data, insolation_model, names=("A_B", "offset"), fixed=None, bad_logl=-1e100, cache=True, **model_kw):
        self.cfg = cfg
        self.insolation_model = insolation_model
        self.names = tuple(names)
        self.fixed = dict(fixed or {})
        self.bad_logl = float(bad_logl)
        self.model_kw = dict(model_kw)
        self.n_calls = 0  # accounting; compare against sum(res.ncall). Won't be used if Pool is on
        self._warned = False

        # Validate the parameter bookkeeping
        unknown = [n for n in self.names + tuple(self.fixed) if n not in _KNOWN]
        if unknown: raise ValueError(f"unsupported parameter(s) {unknown}; supported: {_KNOWN}")
        if len(set(self.names)) != len(self.names): raise ValueError(f"duplicate names in {self.names}")
        clash = [n for n in self.names if n in self.fixed]
        if clash: raise ValueError(f"parameter(s) {clash} are both sampled and fixed")
        if "A_B" not in self.names and "A_B" not in self.fixed: raise ValueError("A_B must be either sampled (in `names`) or supplied in `fixed`")
        geom_in_kw = [k for k in self.model_kw if k in _GEOMETRY]
        if geom_in_kw: raise ValueError(f"pass geometry {geom_in_kw} via `fixed`, not as model_kw")
        if insolation_model not in _CACHE_DEPS["insol"]: raise ValueError(f"insolation_model must be one of {tuple(_CACHE_DEPS['insol'])}")

        # Data
        self.t = np.asarray(data["t"], float)
        self.y = np.asarray(data["y_obs"], float)
        self.sigma = np.broadcast_to(np.asarray(data["sigma"], float), self.y.shape).astype(float)
        if self.y.size != cfg.n_bins: raise ValueError(f"data has {self.y.size} points but cfg.n_bins={cfg.n_bins}")
        if not np.all(self.sigma > 0): raise ValueError("all sigma must be > 0")
        self._inv_s = 1. / self.sigma
        self._const = float(np.sum(np.log(2. * np.pi * self.sigma ** 2)))  # theta-independent

        # Geometry-only caches for speedup
        # Each cache is dropped as soon as ANY parameter it depends on is sampled, so it can never be used at a geometry it was not built at. Both are pure speed-ups: bit-identical either way.
        geom = {k: v for k, v in self.fixed.items() if k in _GEOMETRY}
        self._live = {k: cache and not any(n in self.names for n in deps[insolation_model]) for k, deps in _CACHE_DEPS.items()}
        self._live["refl"] &= cfg.reflected  # nothing to cache when the reflected part is switched off
        self.refl_unit = reflected_unit(cfg, insolation_model, n_t=self.model_kw.get("n_t"), **geom) if self._live["refl"] else None
        self.insol = insolation_cache(cfg, insolation_model, n_lo=self.model_kw.get("n_lo", 300), n_la=self.model_kw.get("n_la", 150), **geom) if self._live["insol"] else None

    # Build the likelihood from a ParameterSet, so the sampled names have a single source of truth. Fixed values are not the pset's business: pass them here, e.g. from_pset(..., pset, fixed=dict(a_rs=2.5))
    @classmethod
    def from_pset(cls, cfg, data, insolation_model, pset, **kw):
        return cls(cfg, data, insolation_model, names=pset.names, **kw)

    # -------------------------------------------------------------------------------------------
    #  Utilities
    # -------------------------------------------------------------------------------------------
    @property
    def ndim(self):
        return len(self.names)

    @property
    def cache_used(self):
        # Which geometry-only caches are live, e.g. ("refl", "insol"). A cache drops out when one of its dependencies is sampled — see _CACHE_DEPS for the measured dependency of each.
        return tuple(k for k in _CACHE_DEPS if self._live[k])

    def __repr__(self):
        return (f"GaussianLogL({self.insolation_model}, names={self.names}, fixed={self.fixed}, ndim={self.ndim}, caches={self.cache_used})")

    def unpack(self, theta):
        """theta -> {name: value}, merged with `fixed`. Inverse of theta_of()."""
        theta = np.atleast_1d(np.asarray(theta, float))
        if theta.size != self.ndim: raise ValueError(f"theta has {theta.size} entries, expected {self.ndim} for {self.names}")
        p = dict(self.fixed)
        p.update(zip(self.names, theta))
        return p

    # -------------------------------------------------------------------------------------------
    #  forward model
    # -------------------------------------------------------------------------------------------
    def curve(self, p):
        """Binned model [ppm] for a parameter dict, WITHOUT the baseline offset. Split out from
        model() because the offset is purely additive: one curve serves every offset."""
        geom = {k: p[k] for k in _GEOMETRY if k in p}
        _, y = model_pcurve(self.cfg, self.insolation_model, p["A_B"], refl_unit=self.refl_unit, insol=self.insol, **geom, **self.model_kw)
        return binned(self.cfg, y)

    def model(self, theta):
        """Binned model [ppm] on the data bins, offset applied. Use this for overlay plots."""
        p = self.unpack(theta)
        return self.curve(p) + p.get("offset", 0.)

    def residuals(self, theta):
        return self.y - self.model(theta)

    def chi2(self, theta):
        return float(np.sum((self.residuals(theta) * self._inv_s) ** 2))

    # -------------------------------------------------------------------------------------------
    #  likelihood
    # -------------------------------------------------------------------------------------------
    def _logl_from_model(self, m):
        if not np.all(np.isfinite(m)): return self.bad_logl
        # lnL(theta) = -0.5 * sum_i [ ((y_i - m_i(theta)) / s_i)^2 + ln(2*pi*s_i^2) ]
        return -0.5 * (float(np.sum(((self.y - m) * self._inv_s) ** 2)) + self._const)

    def __call__(self, theta):
        """The dynesty `loglike`."""
        self.n_calls += 1
        try:
            m = self.model(theta)
        except ArithmeticError as e: # catch numerical failures
            if not self._warned:
                self._warned = True
                warnings.warn(f"{type(e).__name__} at {dict(zip(self.names, np.atleast_1d(theta)))}: {e} -> bad_logl. Further occurrences silenced; if this covers a region of the prior, lnZ is biased.", RuntimeWarning)
            return self.bad_logl
        return self._logl_from_model(m)

    def logl_offsets(self, theta, offsets):
        # only for visualization - making offset sweeps cheap
        p = self.unpack(theta)
        c = self.curve(p)
        offsets = np.atleast_1d(np.asarray(offsets, float))
        if not np.all(np.isfinite(c)):
            return np.full(offsets.shape, self.bad_logl)
        r = (self.y - c)[None, :] - offsets[:, None]
        return -0.5 * (np.sum((r * self._inv_s[None, :]) ** 2, axis=1) + self._const)


def theta_of(logl, values):
    """{name: value} dict -> theta array in logl.names order. Inverse of logl.unpack()."""
    missing = [n for n in logl.names if n not in values]
    if missing: raise ValueError(f"missing value(s) for sampled parameter(s) {missing}")
    return np.array([values[n] for n in logl.names], float)


###########################################################################################################################
#####
#####   Likelihood visualization
#####
###########################################################################################################################

def _map(func, items, n_jobs=-1, min_parallel=4):
    """Evaluate func over items, process-parallel via joblib/loky when it is worth it.

    Processes, never threads: the SPIDERMAN lightcurve() C extension holds the GIL, so threading
    measures ~1x. Each SPIDERMAN call is single-threaded, so one core per process is clean."""
    items = list(items)
    if n_jobs in (1, None) or len(items) < min_parallel: return [func(x) for x in items]
    return list(Parallel(n_jobs=n_jobs, backend="loky")(delayed(func)(x) for x in items))


def logl_slice(logl, name, values, base, n_jobs=-1):
    """lnL along one parameter, every other sampled parameter held at `base`.

    name   : parameter to scan (must be in logl.names)
    values : 1-D array of values for it
    base   : {name: value} covering ALL of logl.names (the entry for `name` is overwritten)
    Returns lnL, same shape as values.

    Scanning "offset" is free (one forward model, vectorised) because the offset is additive.
    """
    values = np.asarray(values, float)
    if name not in logl.names: raise ValueError(f"{name!r} is not sampled; logl.names = {logl.names}")
    if name == "offset": return logl.logl_offsets(theta_of(logl, base), values)

    def _one(v):
        b = dict(base); b[name] = v
        return logl(theta_of(logl, b))
    return np.asarray(_map(_one, values, n_jobs), float)


def logl_grid(logl, x_name, x_vals, y_name, y_vals, base, n_jobs=-1):
    """lnL on a 2-D grid over two sampled parameters.

    Returns lnL with shape (len(y_vals), len(x_vals)) — i.e. lnL[j, i] = lnL(x_vals[i], y_vals[j]), the
    layout expected by plt.pcolormesh(x_vals, y_vals, lnL) and by plt.contour(X, Y, lnL) with
    X, Y = np.meshgrid(x_vals, y_vals).

    With a uniform prior the exponential of this grid IS the un-normalised posterior, so the map is a
    direct preview of what dynesty should return — and evidence_bruteforce() turns the same grid into
    a reference lnZ.

    If one axis is "offset" the grid costs only len(other axis) forward-model evaluations, since the
    offset enters additively.
    """
    for n in (x_name, y_name):
        if n not in logl.names: raise ValueError(f"{n!r} is not sampled; logl.names = {logl.names}")
    if x_name == y_name: raise ValueError("x_name and y_name must differ")
    x_vals = np.asarray(x_vals, float)
    y_vals = np.asarray(y_vals, float)

    # fast path: the offset is purely additive -> one model call per non-offset grid value
    if "offset" in (x_name, y_name):
        off_is_y = (y_name == "offset")
        other_name, other_vals = (x_name, x_vals) if off_is_y else (y_name, y_vals)
        off_vals = y_vals if off_is_y else x_vals

        def _col(v):
            b = dict(base); b[other_name] = v
            return logl.logl_offsets(theta_of(logl, b), off_vals)

        lnL = np.asarray(_map(_col, other_vals, n_jobs), float).T   # (n_off, n_other)
        return lnL if off_is_y else lnL.T

    def _one(ij):
        i, j = ij
        b = dict(base); b[x_name] = x_vals[i]; b[y_name] = y_vals[j]
        return logl(theta_of(logl, b))
    idx = [(i, j) for j in range(len(y_vals)) for i in range(len(x_vals))]
    return np.asarray(_map(_one, idx, n_jobs), float).reshape(len(y_vals), len(x_vals))


def grid_max(lnL_max, *axes_vals):
    """Grid MAP: (lnL_max, (coord_along_axis0, coord_along_axis1, ...)).
    axes_vals are the 1-D coordinate arrays in the AXIS ORDER of Z — so for the output of
    logl_grid(..., x_vals, ..., y_vals) that is grid_max(Z, y_vals, x_vals)."""
    k = np.unravel_index(np.argmax(lnL_max), lnL_max.shape)
    return float(lnL_max[k]), tuple(np.asarray(v, float)[i] for v, i in zip(axes_vals, k))


def evidence_bruteforce(lnL, axes_vals, bounds=None):
    """Brute-force lnZ for UNIFORM priors, by direct integration of a lnL grid.

    With 1-2 free parameters the evidence is just a 1-D/2-D integral, so this validates the entire
    prior-box / likelihood / binning chain against dynesty's res.logz[-1] (they should agree to
    <~0.1 in ln). Cheap enough to keep as a permanent regression test.

    lnL       : n-d array of log-likelihoods
    axes_vals : one 1-D coordinate array per axis of lnL, IN AXIS ORDER (for logl_grid output that
                is [y_vals, x_vals])
    bounds    : [(lo, hi), ...] per axis — the support of the uniform prior. Defaults to the grid
                span. The prior normalisation is 1/prod(hi-lo), so the grid must span the FULL prior
                range or Z is underestimated.

    Returns (lnZ, edge_ratio). edge_ratio = exp(max lnL on the grid boundary - max lnL overall): if
    it is not << 1 the likelihood is still rising at the edge and the prior truncates the posterior.
    """
    lnL = np.asarray(lnL, float)
    axes_vals = [np.asarray(v, float) for v in axes_vals]
    if lnL.ndim != len(axes_vals) or any(lnL.shape[k] != len(v) for k, v in enumerate(axes_vals)):
        raise ValueError(f"lnL shape {lnL.shape} does not match axes {[len(v) for v in axes_vals]}")
    if bounds is None:
        bounds = [(v[0], v[-1]) for v in axes_vals]

    m = float(np.max(lnL))
    I = np.exp(lnL - m)                     # max-subtracted for stability
    for v in reversed(axes_vals):           # integrate the last axis first
        I = np.trapezoid(I, x=v, axis=-1)
    ln_prior = -sum(np.log(hi - lo) for lo, hi in bounds)
    lnZ = m + float(np.log(I)) + ln_prior

    edge = -np.inf
    for k in range(lnL.ndim):
        for s in (0, -1):
            edge = max(edge, float(np.max(np.take(lnL, s, axis=k))))

    edge_ratio = float(np.exp(edge - m)) # L_best_on_boundry / L_best (if not <<1: L leaks out of priors, data to noisy)
    return lnZ, edge_ratio


###########################################################################################################################
#####
#####   Nested sampling driver
#####
###########################################################################################################################

@dataclass
class SamplerConfig:
    """Everything tunable about the dynesty run."""
    dynamic: bool = True
    pfrac: float = 0.8       # Importance of evidence (pfrac=0) vs posterior (pfrac=1)
    nlive: int = 500              # nlive (static) / nlive_init (dynamic)
    nlive_batch: int = 250        # dynamic only
    maxbatch: int = 15            # dynamic only
    dlogz: float = 0.01      # dlogz (static) / dlogz_init (dynamic)
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
    pset: ParameterSet
    scfg: SamplerConfig
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
                data=dict(t=logl.t.copy(), y_obs=logl.y.copy(), sigma=logl.sigma.copy()))   # the fit travels with its data, so a reloaded NestedFit can be re-plotted without re-running the notebook

    # The geometry actually used: sampled, held fixed, or (the usual case) inherited from cfg.planet - an empty `fixed` on its own would not record which.
    pl = logl.cfg.planet
    meta["geometry"] = {k: ("sampled" if k in logl.names else float(logl.fixed.get(k, d))) for k, d in (("a_rs", pl.R / pl.r2), ("T_s", pl.T2_eff), ("rp_rs", pl.r1 / pl.r2))}

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
            if scfg.print_progress: print(f"   saved -> {path}")
        except Exception as e:   # a failed save must never destroy an hour-long run
            warnings.warn(f"could not save to {path}: {type(e).__name__}: {e}; the NestedFit is still returned", RuntimeWarning, stacklevel=2)
    return fit


def load_fit(path):
    """Read back a NestedFit written by run_nested_fit (scfg.outfile). Unpickling needs this module importable, since pset/scfg/NestedFit are defined here."""
    with open(os.path.expanduser(path), "rb") as f: return pickle.load(f)
