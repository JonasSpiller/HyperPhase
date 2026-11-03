import numpy as np
from dataclasses import dataclass
import time
import os
import glob
import pickle
from datetime import datetime
import dynesty
from dynesty.pool import Pool
from joblib import Parallel, delayed
import contextlib
from hyperillumination_phasecurve_utils import model_pcurve, reflected_unit, insolation_cache, pcurve_params, occultation_depth, custom_temperature_grid, swept_planet  # forward model, its geometry-only caches, and the pieces the derived quantities need
import astro_utils   # T_bright
from dynesty import utils as dyfunc
from scipy.ndimage import gaussian_filter
from simple_bias_analysis import binned, fit_naive, temperature_bias
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

def plot_simulated_data(cfg, A_B_true, data, insolation_model="hyperillum", **model_kw):
    ph = data["t"] / cfg.planet.period  # orbital phase

    # Full Phase Curve (for custom reproduction of simulate_data)
    t_hi, y_hi = model_pcurve(cfg, insolation_model, A_B_true, n_t=200 * cfg.n_bins, n_layers=80, n_lo=600, n_la=300, **model_kw)
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
    return fig, (ax, axr)

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

    # the injected truth travels with the data (and from there into NestedFit.meta), so a saved fit knows what it was fitted to
    truth = dict(A_B=float(A_B_true), insolation_model=insolation_model, offset=float(offset), seed=int(seed), noise=bool(noise))
    data = {"t": t, "y_obs": y_obs, "sigma":sigma, "truth": truth}

    if plotting: plot_simulated_data(cfg, A_B_true, data, insolation_model=insolation_model, **{k: v for k, v in model_kw.items() if k == "F"}); plt.show()   # plot_simulated_data returns the figure and no longer shows it itself

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
# F (heat redistribution) belongs to insolation_model="naive_redist" alone: required there, rejected for the other models.
_KNOWN = ("A_B", "offset", "F") + _GEOMETRY

# Which sampled parameter invalidates which geometry-only cache. "naive_redist" reflects exactly like "naive" (F-independent),
# and its thermal part is the analytic Louden model, so it has no insolation map to cache at all (the entry only lists the models).
_CACHE_DEPS = {
    "refl":  {"naive": ("a_rs", "rp_rs"),        "hyperillum": ("a_rs", "rp_rs"),        "naive_redist": ("a_rs", "rp_rs")},
    "insol": {"naive": ("a_rs", "T_s"),        "hyperillum": ("a_rs", "T_s", "rp_rs"),   "naive_redist": ()},
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
    model_kw: forwarded to model_pcurve (n_t, n_layers, n_lo, n_la, spman_kw).
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
        has_F = "F" in self.names or "F" in self.fixed
        if insolation_model == "naive_redist" and not has_F: raise ValueError('insolation_model="naive_redist" needs F either sampled (in `names`) or supplied in `fixed`')
        if has_F and insolation_model != "naive_redist": raise ValueError(f'F is only defined for insolation_model="naive_redist", not {insolation_model!r}')
        geom_in_kw = [k for k in self.model_kw if k in _GEOMETRY]
        if geom_in_kw: raise ValueError(f"pass geometry {geom_in_kw} via `fixed`, not as model_kw")
        if insolation_model not in _CACHE_DEPS["insol"]: raise ValueError(f"insolation_model must be one of {tuple(_CACHE_DEPS['insol'])}")

        # Data
        self.truth = dict(data.get("truth") or {})   # what simulate_data injected, if the data came from there; empty for real observations
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
        self._live["insol"] &= insolation_model != "naive_redist"   # the Louden model never touches an insolation map
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
    #  forward illumination model
    # -------------------------------------------------------------------------------------------
    def curve(self, p):
        """Binned model [ppm] for a parameter dict, WITHOUT the baseline offset. Split out from
        model() because the offset is purely additive: one curve serves every offset."""
        geom = {k: p[k] for k in _GEOMETRY if k in p}
        _, y = model_pcurve(self.cfg, self.insolation_model, p["A_B"], F=p.get("F"), refl_unit=self.refl_unit, insol=self.insol, **geom, **self.model_kw)
        return binned(self.cfg, y)

    def model(self, theta):
        """Binned model [ppm] on the data bins, offset applied. Used for overlay plots."""
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


def logl_grid(logl, x_name, x_vals, y_name, y_vals, base, n_jobs=-1, profile_offset=None):
    """lnL on a 2-D grid over two sampled parameters.

    Returns lnL with shape (len(y_vals), len(x_vals)) — i.e. lnL[j, i] = lnL(x_vals[i], y_vals[j]), the
    layout expected by plt.pcolormesh(x_vals, y_vals, lnL) and by plt.contour(X, Y, lnL) with
    X, Y = np.meshgrid(x_vals, y_vals).

    With a uniform prior the exponential of this grid IS the un-normalised posterior, so the map is a
    direct preview of what dynesty should return — and evidence_bruteforce() turns the same grid into
    a reference lnZ.

    If one axis is "offset" the grid costs only len(other axis) forward-model evaluations, since the
    offset enters additively.

    profile_offset : array of offsets. Instead of holding the offset at base["offset"], each cell is
    MAXIMISED over them - the profile likelihood in the two plotted parameters, with the additive
    baseline treated as the nuisance it is. Free: the offset reuses the cell's one forward model.
    Use it when the plotted parameters are correlated with the offset (e.g. F), where a slice at a
    fixed offset is narrower than the posterior actually is. Only for axes that are not "offset".
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

    if profile_offset is not None:
        if "offset" not in logl.names: raise ValueError("profile_offset needs the offset to be sampled")
        profile_offset = np.asarray(profile_offset, float)

    def _one(ij):
        i, j = ij
        b = dict(base); b[x_name] = x_vals[i]; b[y_name] = y_vals[j]
        th = theta_of(logl, b)
        return float(np.max(logl.logl_offsets(th, profile_offset))) if profile_offset is not None else logl(th)
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
                data=dict(t=logl.t.copy(), y_obs=logl.y.copy(), sigma=logl.sigma.copy(), truth=dict(logl.truth)),   # the fit travels with its data (and the injected truth), so a reloaded NestedFit can be re-plotted without re-running the notebook
                cfg=logl.cfg, model_kw=dict(logl.model_kw))   # everything logl_from_fit needs to rebuild the likelihood from the pickle alone

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
    path = os.path.abspath(os.path.expanduser(path))
    with open(path, "rb") as f: fit = pickle.load(f)
    fit.meta["file"] = path   # run_nested_fit records the path only AFTER pickling, so the file on disk never carries its own name
    return fit


def load_fits(paths):
    """Several saved fits as {insolation_model: NestedFit} - the `fits` dict every plotting helper takes.
    `paths` is a glob pattern or an iterable of paths; a repeated insolation model is an error rather than a silent overwrite."""
    paths = sorted(glob.glob(os.path.expanduser(paths))) if isinstance(paths, str) else list(paths)
    if not paths: raise FileNotFoundError("no fit files matched")
    out = {}
    for p in paths:
        f = load_fit(p)
        m = f.meta["insolation_model"]
        if m in out: raise ValueError(f"two files for insolation_model={m!r}: {out[m].meta.get('file')} and {p}")
        out[m] = f
    return out


def logl_from_fit(fit, cfg=None, **model_kw):
    """The GaussianLogL a saved fit was run with, rebuilt from the pickle alone - what the plotting helpers that still
    evaluate the forward model (logl_slice/logl_grid, plot_posterior_predictive, derived_samples, plot_noisy_vs_truth) need.

    Uses the FittingConfig stored in meta by run_nested_fit; pass `cfg` explicitly for fits saved before that was added.
    The geometry-only caches are rebuilt here, so the first call costs a second or two."""
    cfg = cfg if cfg is not None else fit.meta.get("cfg")
    if cfg is None: raise ValueError("this fit predates cfg being stored in meta - pass cfg=... (or re-run the fit)")
    kw = {**fit.meta.get("model_kw", {}), **model_kw}
    return GaussianLogL.from_pset(cfg, fit.meta["data"], fit.meta["insolation_model"], fit.pset, fixed=fit.meta.get("fixed"), **kw)


###########################################################################################################################
#####
#####   Posterior post-processing
#####
###########################################################################################################################

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
            d["bias"] = d["med"] - d["truth"] # deviation of true parameter value to retrieved parameter value (mean of posterior)
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


def derived_samples(logl, thetas, n_jobs=-1):
    """Posterior draws pushed through to physical quantities -> {key: array of length n_draws}.

    T_sub / T_term / T_anti : equatorial temperature [K] at the substellar point, the terminator (90 deg) and the antistellar point. Under "naive" T_anti is the 0 K nightside and T_term is only np.interp of the grid between the last dayside and the first nightside node; under "hyperillum" the extended star pushes real flux past 90 deg, which is exactly the physical difference the Bayes factor is measuring.
    depth_th / depth_refl / depth : occultation depth [ppm], thermal / reflected / total. Feed depth_th to brightness_temperature().
    """
    cfg, m = logl.cfg, logl.insolation_model
    n_lo, n_la = logl.model_kw.get("n_lo", 300), logl.model_kw.get("n_la", 150)

    def _one(theta):
        p_ = logl.unpack(theta)
        geom = {k: p_[k] for k in _GEOMETRY if k in p_}
        A_B_, F_ = p_["A_B"], p_.get("F")

        lo, la, Tg = custom_temperature_grid(swept_planet(cfg, **geom), u_B=cfg.u_B, n_lo=n_lo, n_la=n_la, insolation_model=m, F=F_, A_B=A_B_, insol=logl.insol)
        eq = int(np.argmin(np.abs(la)))   # equator row; Tg has shape (n_la, n_lo)
        T = list(np.interp([0., 90., 180.], lo, Tg[eq]))   # interpolated, not nearest-cell: T falls steeply through the terminator, so the nearest grid longitude is off by hundreds of K there

        pp = pcurve_params(cfg, m, A_B_, F=F_, **geom, n_layers=logl.model_kw.get("n_layers"), n_lo=n_lo, n_la=n_la, insol=logl.insol)
        d_th = float(occultation_depth(pp, stellar_grid=cfg.stellar_grid))

        # Reflected part: linear in A, and its maximum over the orbit is the last point before ingress - within ~1% of the full-phase value, since reflected_unit itself carries the occultation.
        ru = logl.refl_unit if logl.refl_unit is not None else (reflected_unit(cfg, m, n_t=logl.model_kw.get("n_t"), **geom) if cfg.reflected else None)
        d_re = float(cfg.A_(A_B_) * np.max(ru)) if ru is not None else 0.
        return T + [d_th, d_re]

    thetas = np.atleast_2d(np.asarray(thetas, float))
    a = np.asarray(_map(_one, list(thetas), n_jobs), float)
    out = {k: a[:, i] for i, k in enumerate(("T_sub", "T_term", "T_anti", "depth_th", "depth_refl"))}
    out["depth"] = out["depth_th"] + out["depth_refl"]
    out["A_B"] = thetas[:, logl.names.index("A_B")] if "A_B" in logl.names else np.full(len(thetas), logl.fixed["A_B"])
    return out


def brightness_temperature(depth_th, cfg, quantiles=(0.16, 0.5, 0.84)):
    """Dayside brightness temperature [K] at the given quantiles of the THERMAL occultation depth.
    T_bright is monotone in the depth, so transforming the quantiles is exact - and it keeps the brentq/band_avg_planck cost at three calls instead of one per draw."""
    q = np.quantile(np.asarray(depth_th, float), list(quantiles))
    return np.array([astro_utils.T_bright(d, cfg.planet, (cfg.l1, cfg.l2), cfg.filter_path) for d in q])


###########################################################################################################################
#####
#####   Posterior visualization
#####
###########################################################################################################################

_COLOR = {"naive": "C0", "hyperillum": "C3", "naive_redist": "C2"}


def _density_levels(x, y, rng=None, bins=60, smooth=1.0, levels=(0.68, 0.95)):
    """2-D density of equal-weight draws plus the contour VALUES that enclose the requested posterior mass.
    Levels are found by sorting the histogram cells by height and walking down until the cumulative mass is reached - the usual credible-region construction, not a Gaussian sigma."""
    H, xe, ye = np.histogram2d(x, y, bins=bins, range=rng)
    if smooth: H = gaussian_filter(H, smooth)
    H /= H.sum()
    flat = np.sort(H.ravel())[::-1]
    cum = np.cumsum(flat)
    v = sorted(float(flat[min(int(np.searchsorted(cum, l)), len(flat) - 1)]) for l in levels)
    v = sorted(set(v))   # coarse bins can make two mass levels land in the same cell; contour() needs strictly increasing levels
    return 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), H.T, v


def plot_posterior_overlay(fits, x, y, truth=None, profile=None, n=None, bins=60, smooth=1.0, colors=None, figsize=(6.4, 6.4), seed=0):
    """Both insolation models' 2-D posterior on ONE pair of axes, with the 1-D marginals on the sides. This is the figure that shows the bias instead of tabulating it.

    fits    : {name: NestedFit}, same parameters and same data
    x, y    : parameter names (must be sampled in every fit)
    truth   : {name: value} - the injected values
    profile : optional {name: (x_vals, y_vals, lnL)} from logl_grid, drawn as DASHED contours at the 2-parameter 1/2/3 sigma levels. Where these differ from the filled posterior contours you are seeing the prior volume (Occam) at work, since the likelihood map carries no prior weighting.
    n       : equal-weight draws per model (None -> all)"""
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
    """Median +/- 68% per model against the injected value, one panel per parameter, annotated with the bias in sigma. The compact version of posterior_summary().
    The fits need not share a ParameterSet: the panels are the UNION of their parameters (first-seen order), and a model that does not sample one simply has no marker in that panel."""
    colors = colors or _COLOR
    names, labels = [], {}
    for f in fits.values():   # union, not the first fit's names: e.g. only "naive_redist" samples F
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
    """Distribution of ln B over dynesty's jittered realisations (output of bayes_factor). The width is the sampling uncertainty on the Bayes factor without assuming either lnZ is Gaussian."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(bf["samples"], bins=bins, density=True, histtype="stepfilled", color="0.75", edgecolor="0.35")
    ax.axvline(bf["lnB"], color="k", lw=1.4, label=f"$\\ln B$ = {bf['lnB']:+.2f}")
    ax.axvline(0., color="C3", lw=1.0, ls="--")
    for d in (1., 2.5, 5.): ax.axvline(d, color="0.85", lw=0.7, ls=":")   # Jeffreys thresholds
    ax.set_xlabel(r"$\ln B$ (jittered)"); ax.set_ylabel("density"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, ax


def model_samples(logl, thetas, n_jobs=-1):
    """Binned model curves [ppm] for a stack of thetas -> (n_draws, n_bins). One forward model per row, process-parallel via _map."""
    thetas = np.atleast_2d(np.asarray(thetas, float))
    return np.asarray(_map(logl.model, list(thetas), n_jobs), float)

def plot_posterior_predictive(logls, fits, num="hyperillum", den="naive", n=200, colors=None, seed=0, n_jobs=-1, figsize=(8.5, 8.0), levels=(68., 95.)):
    """Posterior predictive check + where in orbital phase the two models actually differ.

    Panel 1: data with each model's median curve and its 68/95% posterior band.
    Panel 2: (data - median model) / sigma per model.
    Panel 3: the difference of the two median curves [ppm] against +/- sigma_bin, and the per-bin delta chi2 between the two BEST-FIT (max-lnL) models as bars. Those bars sum to 2 * delta lnL_max, so they attribute the evidence to specific orbital phases.

    num / den : which model is which in panel 3 - both the curve and the bars are num - den, same convention as bayes_factor(). Panel 3 previously took them from the order of `fits`, which is silent and easy to get backwards.
    n         : posterior draws per model (n forward models per model)."""
    colors = colors or _COLOR
    names = list(fits)
    f0 = fits[names[0]]
    t = f0.meta["data"]["t"]; y = f0.meta["data"]["y_obs"]
    s = f0.meta["data"]["sigma"]
    ph = t / logls[names[0]].cfg.planet.period

    fig, (ax, axr, axd) = plt.subplots(3, 1, figsize=figsize, sharex=True, gridspec_kw=dict(height_ratios=[3, 1.3, 1.6], hspace=0.07))
    ax.errorbar(ph, y, yerr=s, fmt="o", ms=4, color="k", capsize=2, elinewidth=1., zorder=5, label="data")

    med, best = {}, {}
    mk = ("o", "s", "^", "v")   # the two residual sets sit on top of each other when the models fit almost equally well
    for k, name in enumerate(names):
        L, f = logls[name], fits[name]
        curves = model_samples(L, equal_samples(f, n, seed), n_jobs=n_jobs)
        lo1, m, hi1 = np.percentile(curves, [50 - levels[0] / 2, 50, 50 + levels[0] / 2], axis=0)
        lo2, hi2 = np.percentile(curves, [50 - levels[1] / 2, 50 + levels[1] / 2], axis=0)
        med[name] = m # pointwise median model curve
        best[name] = L.model(f.results.samples[int(np.argmax(f.results.logl))])
        c = colors.get(name, "k")

        # Panel 1
        ax.fill_between(ph, lo2, hi2, color=c, alpha=0.15, lw=0)
        ax.fill_between(ph, lo1, hi1, color=c, alpha=0.30, lw=0)
        ax.plot(ph, m, color=c, lw=1.5, label=f"{name} (pointwise median of {n} draws)")

        # Panel 2
        axr.plot(ph, (y - m) / s, mk[k % len(mk)], ms=4, mfc="none" if k else c, color=c)

    ax.set_ylabel("$F_p/F_\\star$ [ppm]"); ax.legend(frameon=False, fontsize=8); ax.margins(x=0.01)
    for d in (-1., 1.): axr.axhline(d, color="0.85", lw=0.6, ls=":")
    axr.axhline(0., color="0.7", lw=0.8); axr.set_ylabel("resid / $\\sigma$"); axr.set_ylim(-3.5, 3.5)

    # Panel 3: the num/den pair only - with more than two models every other one is still in panels 1-2, but a single difference curve can only be a pair
    if num in fits and den in fits:
        a, b = den, num
        axd.axhspan(-s.mean(), s.mean(), fc="0.9", ec="0.6", lw=0.8, label=r"$\pm\sigma_{bin}$")
        axd.axhline(0., color="0.7", lw=0.8)
        dmed = med[b] - med[a]
        axd.plot(ph, dmed, color="0.2", lw=1.5, label=f"median {b} - {a} [ppm]")
        dchi2 = ((y - best[a]) ** 2 - (y - best[b]) ** 2) / s ** 2   # > 0 where b fits better; sums to 2*dlnL_max
        ax2 = axd.twinx()
        ax2.bar(ph, dchi2, width=0.7 / len(ph), color="C2", alpha=0.45, zorder=0)
        # Both axes forced symmetric about zero. twinx() autoscales independently, so a one-signed dchi2 makes matplotlib clamp ax2 to [0, max] and the bars then grow from the panel floor while axd's zero line sits mid-panel - the reader reads the bars against a baseline that is not theirs.
        w = 1.05 * max(float(np.abs(dmed).max()), float(s.mean())); axd.set_ylim(-w, w)
        v = 1.05 * float(np.abs(dchi2).max()) or 1.; ax2.set_ylim(-v, v)
        ax2.set_ylabel(f"$\\Delta\\chi^2$ per bin (sum {dchi2.sum():+.2f})", fontsize=8, color="C2")
        ax2.tick_params(axis="y", labelcolor="C2", labelsize=8)
        axd.legend(frameon=False, fontsize=8, loc="upper left")
    axd.set_xlabel("orbital phase"); axd.set_ylabel("[ppm]")
    return fig, (ax, axr, axd)


def plot_derived(derived, keys=("T_sub", "T_term", "depth"), truth=None, colors=None, bins=50, figsize=None):
    """Overlaid posteriors of the derived physical quantities from derived_samples().
    derived : {model_name: dict from derived_samples}
    truth   : optional {model_name: dict} (e.g. the same call evaluated at the injected theta) - drawn as vertical lines."""
    colors = colors or _COLOR
    unit = dict(T_sub="[K]", T_term="[K]", T_anti="[K]", depth="[ppm]", depth_th="[ppm]", depth_refl="[ppm]", A_B="")
    fig, axes = plt.subplots(1, len(keys), figsize=figsize or (3.6 * len(keys), 2.7))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        for name, d in derived.items():
            ax.hist(d[k], bins=bins, density=True, histtype="step", lw=1.4, color=colors.get(name, "k"), label=name)
            if truth and name in truth: ax.axvline(np.median(truth[name][k]), color=colors.get(name, "k"), lw=1.0, ls="--")
        ax.set_xlabel(f"{k} {unit.get(k, '')}"); ax.set_yticks([])
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, axes


def plot_noisy_vs_truth(cfg, data, logls, fits, A_B_true, base, insolation_truth="hyperillum",
                        A_B_bounds=(0.0, 0.8), xatol=1e-3, truth_kw=None, colors=None, figsize=(8.5, 6.2), verbose=False):
    """The one-seed fits against the noiseless truth: what you would report vs. what the bias actually is.

    Top panel   : the high-resolution truth and its binned version (= the noiseless data), the simulated data, each model's posterior MEDIAN curve, and the naive model fitted to the NOISELESS truth (simple_bias_analysis.fit_naive, offset free) - that last one is deterministic (no seed, no noise) and IS the bias.
    Lower panel : everything minus the binned truth against +/- sigma_bin. The black points are the noise realisation the medians are chasing; the dotted grey line is the numerical floor, i.e. the truth model re-evaluated at the injected theta on the likelihood's own coarser grid, so anything at that level is grid resolution and not model bias.

    cfg / data       : the FittingConfig and the simulate_data() output the fits were run on.
    logls / fits     : {model_name: GaussianLogL} and {model_name: NestedFit}, same keys.
    base             : {param: injected value} (the dict handed to theta_of / posterior_summary).
    insolation_truth : which insolation model the data were drawn from; must be a key of logls for the numerical floor to be drawn.
    truth_kw         : forward-model resolution of the truth; defaults to simulate_data's own settings.
    verbose          : False prints only the noiseless bias, |dA_B|/sigma_A_B with the critical sigma_1min, and the bias/floor gate; True adds the per-model median and max-lnL rows, the absorbed offset and the floor detail.

    Returns fig, (ax, axr), out - where out carries every number printed at either verbosity (naive-to-truth fit, bias_sigma0, sigma_1min_crit, per-model median/max-lnL chi2, the floor).
    """
    colors = colors or _COLOR
    truth_kw = dict(n_t=200 * cfg.n_bins, n_layers=80, n_lo=600, n_la=300) if truth_kw is None else dict(truth_kw)

    # --- the noiseless reference: exactly the curve simulate_data drew the data from ---------------
    t_hi, y_hi = model_pcurve(cfg, insolation_truth, A_B_true, **truth_kw)
    y_true = binned(cfg, y_hi)                                    # the noiseless data
    ph, ph_hi, s_bin = data["t"] / cfg.planet.period, t_hi / cfg.planet.period, cfg.sigma_bin

    # --- naive fitted to the NOISELESS truth: the deterministic bias, no seed involved -------------
    insol0 = insolation_cache(cfg, "naive", n_lo=truth_kw["n_lo"], n_la=truth_kw["n_la"])   # A_B only rescales the map, so one cache serves the whole scan
    A_naive0, chi2_naive0, rms_naive0 = fit_naive(cfg, y_hi, free_offset=True, bounds=A_B_bounds, xatol=xatol, insol=insol0, **truth_kw)
    y_naive0 = binned(cfg, model_pcurve(cfg, "naive", A_naive0, insol=insol0, **truth_kw)[1])
    off_naive0 = float(np.mean(y_true - y_naive0))                # what free_offset absorbed
    y_naive0 = y_naive0 + off_naive0

    # --- the two noisy fits: posterior median (what you would report) and max-lnL (= MAP, uniform priors)
    summ   = {m: posterior_summary(f, base, verbose=False)["params"] for m, f in fits.items()}
    th_med = {m: np.array([summ[m][n]["med"] for n in f.pset.names], float) for m, f in fits.items()}
    th_map = {m: f.results.samples[int(np.argmax(f.results.logl))] for m, f in fits.items()}
    y_med  = {m: logls[m].model(th_med[m]) for m in fits}
    y_map  = {m: logls[m].model(th_map[m]) for m in fits}
    y_floor = logls[insolation_truth].model(theta_of(logls[insolation_truth], base)) if insolation_truth in logls else None   # same model, same parameters, coarser grid: the numerical floor of this figure

    fig, (ax, axr) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw=dict(height_ratios=[3, 1.7], hspace=0.06))

    ax.plot(ph_hi, y_hi, lw=1.0, color="0.55", zorder=1, label=f"truth ({insolation_truth}, $A_B$={A_B_true:.3f})")
    ax.step(ph, y_true, where="mid", color="0.25", lw=1.2, zorder=2, label="binned truth = noiseless data")
    ax.errorbar(ph, data["y_obs"], yerr=data["sigma"], fmt="o", ms=4, color="k", capsize=2, elinewidth=1., zorder=6, label=f"simulated data ($\\sigma$={s_bin:.1f} ppm)")
    for m in fits:
        ax.step(ph, y_med[m], where="mid", color=colors.get(m, "k"), lw=1.6, zorder=4, label=f"{m} posterior median (one seed)")
    ax.step(ph, y_naive0, where="mid", color=colors.get("naive", "k"), lw=1.6, ls="--", zorder=5, label=f"naive fitted to the noiseless truth ($A_B$={A_naive0:.4f})")
    ax.set_ylabel("$F_p/F_\\star$ [ppm]"); ax.legend(frameon=False, fontsize=8); ax.margins(x=0.01)

    axr.axhspan(-s_bin, s_bin, fc="0.9", ec="0.6", lw=0.8, zorder=0, label=r"$\pm\sigma_{bin}$")
    axr.axhline(0., color="0.7", lw=0.8)
    axr.plot(ph, data["y_obs"] - y_true, "o", ms=4, color="k", zorder=6, label="data - truth (the noise realisation)")
    for m in fits:
        axr.step(ph, y_med[m] - y_true, where="mid", color=colors.get(m, "k"), lw=1.6, zorder=4)
    axr.step(ph, y_naive0 - y_true, where="mid", color=colors.get("naive", "k"), lw=1.6, ls="--", zorder=5)
    if y_floor is not None:
        axr.step(ph, y_floor - y_true, where="mid", color="0.45", lw=1.0, ls=":", zorder=3, label="numerical floor (truth model at the injected $\\theta$)")
    axr.set_xlabel("orbital phase"); axr.set_ylabel("model - truth [ppm]"); axr.margins(x=0.01)
    axr.secondary_yaxis("right", functions=(lambda v: v / s_bin, lambda v: v * s_bin)).set_ylabel(r"$\sigma_{bin}$", fontsize=8)
    axr.legend(frameon=False, fontsize=7, ncol=3, loc="upper left")

    # --- numbers -----------------------------------------------------------------------------------
    chi2_tr = lambda y: float(np.sum(((y - y_true) / s_bin) ** 2))
    sig_AB  = 0.5 * (summ["naive"]["A_B"]["hi"] - summ["naive"]["A_B"]["lo"])
    out = dict(t_hi=t_hi, y_hi=y_hi, y_true=y_true, y_med=y_med, y_map=y_map, y_naive0=y_naive0, y_floor=y_floor,
               summary=summ, A_naive0=float(A_naive0), chi2_naive0=float(chi2_naive0), rms_naive0=float(rms_naive0),
               offset_naive0=off_naive0, bias_naive0=float(A_naive0 - A_B_true), sigma_A_B=float(sig_AB),
               dT_over_T=float(temperature_bias(A_B_true, A_naive0 - A_B_true)),
               chi2_truth={m: chi2_tr(y_med[m]) for m in fits},
               chi2_floor=chi2_tr(y_floor) if y_floor is not None else None)
    out["bias_over_floor"] = (out["chi2_naive0"] / out["chi2_floor"]) if out["chi2_floor"] else None   # the gate on this whole figure: how far the naive-vs-truth distance stands above the forward model's own discretisation error
    out["bias_sigma0"] = abs(out["bias_naive0"]) / sig_AB                    # the NOISELESS bias in units of the posterior width - the seed-free counterpart of summ["naive"]["A_B"]["bias_sigma"], whose numerator is one realisation's median
    out["sigma_1min_crit"] = cfg.sigma_1min * out["bias_sigma0"]             # sigma_A_B scales linearly with sigma_1min (at fixed n_orbits/n_bins) while the bias does not, so |dA_B| = sigma_A_B exactly at this per-minute precision; sigma_A_B is read off THIS seed's posterior, so it carries one realisation's scatter, and the linear scaling assumes the posterior is not truncated by the prior box
    if verbose:
        for m in fits:
            a, o, pm = summ[m]["A_B"], summ[m]["offset"], logls[m].unpack(th_map[m])
            print(f"{m:11s} median   A_B {a['med']:.4f} ({a['bias']:+.4f} = {a['bias_sigma']:+.2f} sigma)   offset {o['med']:+7.2f} ppm ({o['bias']:+.2f} = {o['bias_sigma']:+.2f} sigma)"
                  f"   chi2 vs truth {chi2_tr(y_med[m]):6.2f}   vs data {logls[m].chi2(th_med[m]):6.2f}")
            print(f"{'':11s} max-lnL  A_B {pm['A_B']:.4f} ({pm['A_B'] - A_B_true:+.4f})              offset {pm['offset']:+7.2f} ppm ({pm['offset']:+.2f})"
                  f"           chi2 vs truth {chi2_tr(y_map[m]):6.2f}   vs data {logls[m].chi2(th_map[m]):6.2f}")
        print()

    fac = (1. / out["bias_sigma0"]) if out["bias_sigma0"] > 0 else np.inf      # how much finer sigma_1min must get before the bias is a 1-sigma effect
    print(f"NOISELESS naive vs the NOISELESS truth  (offset free, no noise, no seed - this is THE bias):")
    print(f"   A_B      {A_naive0:.4f}   bias {out['bias_naive0']:+.4f} = {out['bias_naive0'] / sig_AB:+.2f} sigma_A_B   ->   dT/T = {100 * out['dT_over_T']:+.2f} %")
    print(f"   |dA_B|/sigma_A_B = {out['bias_sigma0']:.2f}   ->   sigma_1min,crit = {out['sigma_1min_crit']:.1f} ppm/min   (now {cfg.sigma_1min:.1f}, so {fac:.1f}x finer photometry - or {fac ** 2:.0f}x the orbits - before the bias reaches 1 sigma)")
    if verbose:
        print(f"   offset absorbed {off_naive0:+.2f} ppm    chi2 {chi2_naive0:.3f}  (squared truth-to-manifold distance in units of sigma_bin)    resid rms {rms_naive0:.2f} ppm = {rms_naive0 / s_bin:.2f} sigma_bin")
        print(f"   the noisy naive median instead says {summ['naive']['A_B']['bias']:+.4f}; the gap to {out['bias_naive0']:+.4f} is this one seed's noise (sigma_A_B = {sig_AB:.4f})")

    if y_floor is not None:
        if verbose:
            print(f"\nnumerical floor: {insolation_truth} at the injected theta vs its own truth curve, chi2 = {chi2_tr(y_floor):.3f}   max |diff| = {np.abs(y_floor - y_true).max():.2f} ppm = {np.abs(y_floor - y_true).max() / s_bin:.2f} sigma_bin")
            print(f"   the likelihood runs model_pcurve at its DEFAULTS (n_t=50*n_bins, n_lo=300, n_la=150), the truth at {truth_kw} - anything here is grid resolution, not model bias")
        r = out["bias_over_floor"]
        print(f"   bias / floor = {chi2_naive0:.3f} / {out['chi2_floor']:.3f} = {r:.1f}x   ->   " + ("bias resolved" if r >= 10. else "NOT RESOLVED - refine model_kw (n_lo, n_la, n_layers, n_t) before quoting any bias from this run"))
    return fig, (ax, axr), out
