import numpy as np
from dataclasses import dataclass, field
import time
from hyperillumination_phasecurve_utils import model_pcurve, reflected_unit, insolation_cache  # forward model + its geometry-only caches
from simple_bias_analysis import binned # exact binning


###########################################################################################################################
#####
#####   Simulate Observations
#####
###########################################################################################################################

def simulate_data(cfg, A_B_true, insolation_model="hyperillum", seed=42, noise=True, offset=0.,
                  k_truth=200, n_layers=80, n_lo=600, n_la=300, **model_kw):
    """Simulated phase-folded observation: binned(truth) + N(0, sigma_bin).

    cfg          : FittingConfig — the only mandatory input; supplies planet, filter, stellar_grid, A_(A_B), sigma_1min, n_orbits, n_bins, u_B, reflected.
    A_B_true     : true Bond albedo; defaults to cfg.A_B_true.
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

    return {"t": t, "y_obs": y_obs, "sigma":sigma}


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

        # Validate the parameter bookkeeping
        unknown = [n for n in self.names + tuple(self.fixed) if n not in _KNOWN]
        if unknown:
            raise ValueError(f"unsupported parameter(s) {unknown}; supported: {_KNOWN}")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"duplicate names in {self.names}")
        clash = [n for n in self.names if n in self.fixed]
        if clash:
            raise ValueError(f"parameter(s) {clash} are both sampled and fixed")
        if "A_B" not in self.names and "A_B" not in self.fixed:
            raise ValueError("A_B must be either sampled (in `names`) or supplied in `fixed`")
        geom_in_kw = [k for k in self.model_kw if k in _GEOMETRY]
        if geom_in_kw:
            raise ValueError(f"pass geometry {geom_in_kw} via `fixed`, not as model_kw")
        if insolation_model not in _CACHE_DEPS["insol"]:
            raise ValueError(f"insolation_model must be one of {tuple(_CACHE_DEPS['insol'])}")

        # Data
        self.t = np.asarray(data["t"], float)
        self.y = np.asarray(data["y_obs"], float)
        self.sigma = np.broadcast_to(np.asarray(data["sigma"], float), self.y.shape).astype(float)
        if self.y.size != cfg.n_bins:
            raise ValueError(f"data has {self.y.size} points but cfg.n_bins={cfg.n_bins}")
        if not np.all(self.sigma > 0):
            raise ValueError("all sigma must be > 0")
        self._inv_s = 1. / self.sigma
        self._const = float(np.sum(np.log(2. * np.pi * self.sigma ** 2)))  # theta-independent

        # Geometry-only caches for speedup
        # Each cache is dropped as soon as ANY parameter it depends on is sampled, so it can never be used at a geometry it was not built at. Both are pure speed-ups: bit-identical either way.
        geom = {k: v for k, v in self.fixed.items() if k in _GEOMETRY}
        self._live = {k: bool(cache) and not any(n in self.names for n in deps[insolation_model])
                      for k, deps in _CACHE_DEPS.items()}

        self.refl_unit = None
        if self._live["refl"] and cfg.reflected:
            self.refl_unit = reflected_unit(cfg, insolation_model, n_t=self.model_kw.get("n_t"), **geom)
        else:
            self._live["refl"] = False          # nothing to cache when cfg.reflected is off

        self.insol = None
        if self._live["insol"]:
            self.insol = insolation_cache(cfg, insolation_model,
                                          n_lo=self.model_kw.get("n_lo", 300),
                                          n_la=self.model_kw.get("n_la", 150), **geom)

        self.n_calls = 0   # cheap accounting; compare against sum(res.ncall)

    # -------------------------------------------------------------------------------------------
    #  Utilities
    # -------------------------------------------------------------------------------------------
    @property
    def ndim(self):
        return len(self.names)

    @property
    def cache_used(self):
        # Which geometry-only caches are live, e.g. ("refl", "insol"). A cache drops out when one of its dependencies is sampled — see _CACHE_DEPS for the measured dependency of each.
        return tuple(k for k in ("refl", "insol") if self._live[k])

    def __repr__(self):
        return (f"GaussianLogL({self.insolation_model}, names={self.names}, "
                f"fixed={self.fixed}, ndim={self.ndim}, caches={self.cache_used})")

    def unpack(self, theta):
        """theta -> {name: value}, merged with `fixed`. Inverse of theta_of()."""
        theta = np.atleast_1d(np.asarray(theta, float))
        if theta.size != self.ndim:
            raise ValueError(f"theta has {theta.size} entries, expected {self.ndim} for {self.names}")
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
        _, y = model_pcurve(self.cfg, self.insolation_model, p["A_B"],
                            refl_unit=self.refl_unit, insol=self.insol, **geom, **self.model_kw)
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
        try: m = self.model(theta)
        except Exception: return self.bad_logl # to catch exit(0) spiderman error
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
    if missing:
        raise ValueError(f"missing value(s) for sampled parameter(s) {missing}")
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
    if n_jobs in (1, None) or len(items) < min_parallel:
        return [func(x) for x in items]
    from joblib import Parallel, delayed
    return list(Parallel(n_jobs=n_jobs, backend="loky")(delayed(func)(x) for x in items))


def check_likelihood(logl, thetas, verbose=True):
    """SERIAL smoke test of the likelihood over a set of parameter vectors.

    Run this before ever handing the likelihood to a pool. SPIDERMAN kills the process with exit(0)
    rather than raising, so a bad point inside a worker silently takes the whole run with it — you
    want to meet that failure here, in one process, where the death is attributable to a specific
    theta and you can move the prior bound accordingly.

    thetas : (n, ndim) array-like of parameter vectors.
    Returns dict(lnl, n_bad, bad_thetas, t_call, lnl_best, theta_best).
    """
    thetas = np.atleast_2d(np.asarray(thetas, float))
    lnl = np.empty(len(thetas))
    t0 = time.perf_counter()
    for i, th in enumerate(thetas):
        if verbose and i and i % 25 == 0:
            print(f"  {i}/{len(thetas)} ... {(time.perf_counter()-t0)/i*1e3:.0f} ms/call", flush=True)
        lnl[i] = logl(th)
    dt = (time.perf_counter() - t0) / len(thetas)

    bad = ~np.isfinite(lnl) | (lnl <= logl.bad_logl)
    i_best = int(np.argmax(np.where(bad, -np.inf, lnl)))
    out = dict(lnl=lnl, n_bad=int(bad.sum()), bad_thetas=thetas[bad], t_call=dt,
               lnl_best=float(lnl[i_best]), theta_best=thetas[i_best])
    if verbose:
        print(f"{len(thetas)} draws, {dt*1e3:.0f} ms per lnL call "
              f"(-> {dt*1e4/60:.1f} min per 1e4 calls, serial)")
        print(f"bad (non-finite model): {out['n_bad']}/{len(thetas)}")
        print(f"best: lnL = {out['lnl_best']:.2f} at "
              + ", ".join(f"{n}={v:.4g}" for n, v in zip(logl.names, out['theta_best'])))
        if out["n_bad"]:
            print("  -> tighten the prior bounds so these are never proposed:")
            for th in out["bad_thetas"][:5]:
                print("     " + ", ".join(f"{n}={v:.4g}" for n, v in zip(logl.names, th)))
    return out


def logl_slice(logl, name, values, base, n_jobs=-1):
    """lnL along one parameter, every other sampled parameter held at `base`.

    name   : parameter to scan (must be in logl.names)
    values : 1-D array of values for it
    base   : {name: value} covering ALL of logl.names (the entry for `name` is overwritten)
    Returns lnL, same shape as values.

    Scanning "offset" is free (one forward model, vectorised) because the offset is additive.
    """
    values = np.asarray(values, float)
    if name not in logl.names:
        raise ValueError(f"{name!r} is not sampled; logl.names = {logl.names}")
    if name == "offset":
        return logl.logl_offsets(theta_of(logl, base), values)

    def _one(v):
        b = dict(base); b[name] = v
        return logl(theta_of(logl, b))
    return np.asarray(_map(_one, values, n_jobs), float)


def logl_grid(logl, x_name, x_vals, y_name, y_vals, base, n_jobs=-1):
    """lnL on a 2-D grid over two sampled parameters.

    Returns Z with shape (len(y_vals), len(x_vals)) — i.e. Z[j, i] = lnL(x_vals[i], y_vals[j]), the
    layout expected by plt.pcolormesh(x_vals, y_vals, Z) and by plt.contour(X, Y, Z) with
    X, Y = np.meshgrid(x_vals, y_vals).

    With a uniform prior the exponential of this grid IS the un-normalised posterior, so the map is a
    direct preview of what dynesty should return — and evidence_bruteforce() turns the same grid into
    a reference lnZ.

    If one axis is "offset" the grid costs only len(other axis) forward-model evaluations, since the
    offset enters additively.
    """
    for n in (x_name, y_name):
        if n not in logl.names:
            raise ValueError(f"{n!r} is not sampled; logl.names = {logl.names}")
    if x_name == y_name:
        raise ValueError("x_name and y_name must differ")
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
        Z = np.asarray(_map(_col, other_vals, n_jobs), float).T   # (n_off, n_other)
        return Z if off_is_y else Z.T

    def _one(ij):
        i, j = ij
        b = dict(base); b[x_name] = x_vals[i]; b[y_name] = y_vals[j]
        return logl(theta_of(logl, b))
    idx = [(i, j) for j in range(len(y_vals)) for i in range(len(x_vals))]
    return np.asarray(_map(_one, idx, n_jobs), float).reshape(len(y_vals), len(x_vals))


def grid_max(Z, *axes_vals):
    """Grid MAP: (lnL_max, (coord_along_axis0, coord_along_axis1, ...)).
    axes_vals are the 1-D coordinate arrays in the AXIS ORDER of Z — so for the output of
    logl_grid(..., x_vals, ..., y_vals) that is grid_max(Z, y_vals, x_vals)."""
    k = np.unravel_index(np.argmax(Z), Z.shape)
    return float(Z[k]), tuple(np.asarray(v, float)[i] for v, i in zip(axes_vals, k))


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
    return lnZ, float(np.exp(edge - m))
