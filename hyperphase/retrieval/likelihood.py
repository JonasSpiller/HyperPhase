"""Gaussian likelihoods of one band or a detector set, and scans of the likelihood."""
import warnings
from dataclasses import replace

import numpy as np
from joblib import Parallel, delayed

from hyperphase.bias.detectors import _blocks, is_detector_set
from hyperphase.bias.fits import binned
from hyperphase.defaults import DEFAULT_N_LA, DEFAULT_N_LO
from hyperphase.model import insolation_cache, model_pcurve, pcurve_params, reflected_unit, swept_planet
from hyperphase.phasecurve import custom_temperature_grid, occultation_depth
from hyperphase.retrieval.priors import _GEOMETRY, _KNOWN, _ORBIT
from hyperphase.retrieval.simulate import detector_data


############################################################################################################################################
#####   Log-Likelihood
############################################################################################################################################

# Which sampled parameter invalidates which geometry-only cache. "naive_redist" reflects exactly like "naive" (F-independent),
# and its thermal part is the analytic Louden model, so it has no insolation map to cache at all (the entry only lists the models).
_CACHE_DEPS = {
    "refl":  {"naive": ("a_rs", "rp_rs", "incl", "b"), "hyperillum": ("a_rs", "rp_rs", "incl", "b"), "naive_redist": ("a_rs", "rp_rs", "incl", "b")},   # the viewing geometry; the insolation map does not depend on i
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
        if all(any(n in s for s in (self.names, self.fixed)) for n in _ORBIT): raise ValueError('give the inclination either as "incl" or as "b", not both')
        geom_in_kw = [k for k in self.model_kw if k in _GEOMETRY + _ORBIT]
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
        self.data = {"t": self.t, "y_obs": self.y, "sigma": self.sigma, "truth": dict(self.truth)}   # what run_nested_fit stores with the fit

        # Geometry-only caches for speedup
        # Each cache is dropped as soon as ANY parameter it depends on is sampled, so it can never be used at a geometry it was not built at. Both are
        # pure speed-ups: bit-identical either way.
        geom = {k: v for k, v in self.fixed.items() if k in _GEOMETRY}
        self._live = {k: cache and not any(n in self.names for n in deps[insolation_model]) for k, deps in _CACHE_DEPS.items()}
        self._live["refl"] &= cfg.reflected  # nothing to cache when the reflected part is switched off
        self._live["insol"] &= insolation_model != "naive_redist"   # the Louden model never touches an insolation map
        self.refl_unit = reflected_unit(cfg, insolation_model, n_t=self.model_kw.get("n_t"), **geom) if self._live["refl"] else None
        self.insol = insolation_cache(cfg, insolation_model, n_lo=self.model_kw.get("n_lo", DEFAULT_N_LO), n_la=self.model_kw.get("n_la", DEFAULT_N_LA), **geom) if self._live["insol"] else None

    # Build the likelihood from a ParameterSet, so the sampled names have a single source of truth. Fixed values are not the pset's business: pass
    # them here, e.g. from_pset(..., pset, fixed=dict(a_rs=2.5))
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
        # Which geometry-only caches are live, e.g. ("refl", "insol"). A cache drops out when one of its dependencies is sampled — see _CACHE_DEPS for
        # the measured dependency of each.
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
        _, y = model_pcurve(_orbit_cfg(self.cfg, p), self.insolation_model, p["A_B"], F=p.get("F"), refl_unit=self.refl_unit, insol=self.insol, **geom, **self.model_kw)
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

    def logl_offsets(self, theta, offsets, name="offset"):
        # only for visualization - making offset sweeps cheap (`name` is the baseline's name, "offset" here; JointGaussianLogL has one per detector)
        p = self.unpack(theta)
        c = self.curve(p)
        offsets = np.atleast_1d(np.asarray(offsets, float))
        if not np.all(np.isfinite(c)):
            return np.full(offsets.shape, self.bad_logl)
        r = (self.y - c)[None, :] - offsets[:, None]
        return -0.5 * (np.sum((r * self._inv_s[None, :]) ** 2, axis=1) + self._const)

    def logl_profile_offsets(self, theta, offsets):
        """lnL at theta with the baseline MAXIMISED over `offsets`: the profile likelihood in the other parameters, the baseline treated as the nuisance it is. Free (one forward model)."""
        return float(np.max(self.logl_offsets(theta, offsets)))


def _orbit_cfg(cfg, p):
    """cfg at the inclination of the parameter dict p: "incl" [deg] as given, or "b" with i = arccos(|b| / (a/R*)), a/R* from p or the planet. cfg itself when p has neither.
    A b at or beyond a/R* raises FloatingPointError, which GaussianLogL.__call__ turns into bad_logl."""
    if "incl" in p: incl = float(p["incl"])
    elif "b" in p:
        x = abs(float(p["b"])) / float(p.get("a_rs", cfg.planet.R / cfg.planet.r2))
        if not x < 1.: raise FloatingPointError(f"impact parameter b = {p['b']} >= a/R*")
        incl = float(np.degrees(np.arccos(x)))
    else: return cfg
    return replace(cfg, planet=replace(cfg.planet, incl=incl))   # FittingConfig rebuilds its stellar grid from the lru cache: cheap


class JointGaussianLogL:
    """Several detectors fitted JOINTLY - the dynesty counterpart of a detector set in bias.detectors (e.g. the two NIRSpec/G395H arrays): one GaussianLogL per detector, the
    physical parameters (A_B, F, geometry) shared, one baseline per detector, and lnL = the sum of the detectors' lnL (independent data).

    theta is ordered by `names`: the shared names plus one "offset_<label>" per detector; `labels` are the detectors' short names in the order of `cfgs` (e.g. ("NRS1", "NRS2"),
    bands.detector_label). Each component sees names = shared + ("offset",) and its own data dict, so every cache, check and model keyword works per detector exactly as
    in a single fit. A baseline is additive within its own block only, so the offset scans stay free (logl_offsets for one baseline, logl_profile_offsets for all of them at once).

    cfgs   : the detector set (list/tuple of FittingConfig; same planet and n_bins, different band and noise)
    data   : a joint simulate_data output (with "detectors") or a list of per-detector data dicts
    Pass the instance straight to dynesty as `loglike`, like a GaussianLogL; build it with make_logl / from_pset so the names cannot drift from the ParameterSet.
    """

    def __init__(self, cfgs, data, insolation_model, names, labels, fixed=None, bad_logl=-1e100, cache=True, **model_kw):
        self.cfgs = list(cfgs)
        self.cfg = tuple(self.cfgs)   # the detector set, as run_nested_fit stores it and the bias-analysis fitters take it
        self.labels = tuple(labels)
        if len(self.labels) != len(self.cfgs): raise ValueError(f"{len(self.labels)} detector labels for {len(self.cfgs)} FittingConfigs")
        datas = detector_data(data) if isinstance(data, dict) else list(data)
        if len(datas) != len(self.cfgs): raise ValueError(f"{len(datas)} datasets for {len(self.cfgs)} detectors")
        self.names = tuple(names)
        self.offset_names = tuple(f"offset_{l}" for l in self.labels)
        missing = [o for o in self.offset_names if o not in self.names]
        if missing: raise ValueError(f"a joint fit samples one baseline per detector: {missing} missing from names {self.names}")
        if "offset" in self.names: raise ValueError('a joint fit names its baselines "offset_<label>", not "offset"')
        self.shared = tuple(n for n in self.names if n not in self.offset_names)
        self.insolation_model = insolation_model
        self.fixed = dict(fixed or {})
        self.model_kw = dict(model_kw)
        self.components = [GaussianLogL(c, d, insolation_model, names=self.shared + ("offset",), fixed=fixed, bad_logl=bad_logl, cache=cache, **model_kw) for c, d in zip(self.cfgs, datas)]
        self.t = np.concatenate([L.t for L in self.components])
        self.y = np.concatenate([L.y for L in self.components])
        self.sigma = np.concatenate([L.sigma for L in self.components])
        self.truth = dict(self.components[0].truth)
        self.data = {"t": self.t, "y_obs": self.y, "sigma": self.sigma, "truth": self.truth, "detectors": [L.data for L in self.components]}
        self.blocks = _blocks([L.y.size for L in self.components])   # each detector's slice of the concatenated model / residual vector
        self.n_calls = 0

    @classmethod
    def from_pset(cls, cfgs, data, insolation_model, pset, labels, **kw):
        return cls(cfgs, data, insolation_model, names=pset.names, labels=labels, **kw)

    @property
    def ndim(self): return len(self.names)
    @property
    def cache_used(self): return self.components[0].cache_used
    @property
    def insol(self): return self.components[0].insol        # geometry-only, hence band-independent: the first detector's serves the derived quantities
    @property
    def refl_unit(self): return self.components[0].refl_unit

    def __repr__(self):
        return f"JointGaussianLogL({self.insolation_model}, detectors={self.labels}, names={self.names}, fixed={self.fixed}, ndim={self.ndim}, caches={self.cache_used})"

    def unpack(self, theta):
        """theta -> {name: value}, merged with `fixed`. Inverse of theta_of()."""
        theta = np.atleast_1d(np.asarray(theta, float))
        if theta.size != self.ndim: raise ValueError(f"theta has {theta.size} entries, expected {self.ndim} for {self.names}")
        p = dict(self.fixed)
        p.update(zip(self.names, theta))
        return p

    def thetas(self, theta):
        """The joint theta -> one component theta per detector, its own baseline under the component's name "offset"."""
        p = self.unpack(theta)
        return [np.array([p[o] if n == "offset" else p[n] for n in L.names], float) for L, o in zip(self.components, self.offset_names)]

    def models(self, theta):
        """Binned model [ppm] per detector, offsets applied."""
        return [L.model(th) for L, th in zip(self.components, self.thetas(theta))]

    def model(self, theta):
        """The detectors' binned models concatenated (self.blocks slices them)."""
        return np.concatenate(self.models(theta))

    def residuals(self, theta): return self.y - self.model(theta)

    def chi2(self, theta): return float(np.sum((self.residuals(theta) / self.sigma) ** 2))

    def __call__(self, theta):
        """The dynesty `loglike`: the sum over the detectors."""
        self.n_calls += 1
        return float(sum(L(th) for L, th in zip(self.components, self.thetas(theta))))

    def logl_offsets(self, theta, offsets, name):
        """lnL along ONE detector's baseline `name` ("offset_<label>"), everything else at theta: that detector's block is scanned analytically, the other blocks are constants."""
        k = self.offset_names.index(name)
        ths = self.thetas(theta)
        rest = sum(L(th) for j, (L, th) in enumerate(zip(self.components, ths)) if j != k)
        return rest + self.components[k].logl_offsets(ths[k], offsets)

    def logl_profile_offsets(self, theta, offsets):
        """lnL at theta with EVERY baseline maximised over `offsets`: the blocks are independent, so the maxima separate."""
        return float(sum(np.max(L.logl_offsets(th, offsets)) for L, th in zip(self.components, self.thetas(theta))))


def make_logl(cfg, data, insolation_model, pset, labels=None, **kw):
    """The likelihood of a ParameterSet for a FittingConfig (GaussianLogL) or a detector set (JointGaussianLogL; needs the detectors' `labels`, one per FittingConfig)."""
    if is_detector_set(cfg):
        if labels is None: raise ValueError("a joint fit needs the detector labels (one per FittingConfig, e.g. ('NRS1', 'NRS2'))")
        return JointGaussianLogL.from_pset(cfg, data, insolation_model, pset, labels, **kw)
    return GaussianLogL.from_pset(cfg, data, insolation_model, pset, **kw)


def theta_of(logl, values):
    """{name: value} dict -> theta array in logl.names order. Inverse of logl.unpack()."""
    missing = [n for n in logl.names if n not in values]
    if missing: raise ValueError(f"missing value(s) for sampled parameter(s) {missing}")
    return np.array([values[n] for n in logl.names], float)


############################################################################################################################################
#####   Likelihood scans
############################################################################################################################################
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

    Scanning a baseline ("offset", or "offset_<label>" of a joint fit) is free (one forward model, vectorised) because it is additive.
    """
    values = np.asarray(values, float)
    if name not in logl.names: raise ValueError(f"{name!r} is not sampled; logl.names = {logl.names}")
    if name.startswith("offset"): return logl.logl_offsets(theta_of(logl, base), values, name)

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

    profile_offset : array of offsets. Instead of holding the baseline(s) at `base`, each cell is
    MAXIMISED over them - the profile likelihood in the two plotted parameters, with the additive
    baseline treated as the nuisance it is (every detector's own baseline for a joint fit). Free:
    the offsets reuse the cell's one forward model. Use it when the plotted parameters are correlated
    with the offset (e.g. F), where a slice at a fixed offset is narrower than the posterior actually
    is. Only for axes that are not baselines.
    """
    for n in (x_name, y_name):
        if n not in logl.names: raise ValueError(f"{n!r} is not sampled; logl.names = {logl.names}")
    if x_name == y_name: raise ValueError("x_name and y_name must differ")
    x_vals = np.asarray(x_vals, float)
    y_vals = np.asarray(y_vals, float)

    # fast path: a baseline is purely additive -> one model call per non-baseline grid value
    off_axes = [n for n in (x_name, y_name) if n.startswith("offset")]
    if len(off_axes) == 2: raise ValueError("a map of one baseline against another is not supported (and not informative: the blocks are independent); scan a baseline against a physical parameter")
    if off_axes:
        off_name = off_axes[0]
        off_is_y = (y_name == off_name)
        other_name, other_vals = (x_name, x_vals) if off_is_y else (y_name, y_vals)
        off_vals = y_vals if off_is_y else x_vals

        def _col(v):
            b = dict(base); b[other_name] = v
            return logl.logl_offsets(theta_of(logl, b), off_vals, off_name)

        lnL = np.asarray(_map(_col, other_vals, n_jobs), float).T   # (n_off, n_other)
        return lnL if off_is_y else lnL.T

    if profile_offset is not None:
        if not any(n.startswith("offset") for n in logl.names): raise ValueError("profile_offset needs the baseline(s) to be sampled")
        profile_offset = np.asarray(profile_offset, float)

    def _one(ij):
        i, j = ij
        b = dict(base); b[x_name] = x_vals[i]; b[y_name] = y_vals[j]
        th = theta_of(logl, b)
        return logl.logl_profile_offsets(th, profile_offset) if profile_offset is not None else logl(th)   # every baseline maximised (per detector for a joint fit)
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

    edge_ratio = float(np.exp(edge - m)) # L_best_on_boundary / L_best (if not <<1: L leaks out of priors, data too noisy)
    return lnZ, edge_ratio


############################################################################################################################################
#####   Derived quantities
############################################################################################################################################
def derived_samples(logl, thetas, n_jobs=-1):
    """Posterior draws pushed through to physical quantities -> {key: array of length n_draws}.

    T_sub / T_term / T_anti : equatorial temperature [K] at the substellar point, the terminator (90 deg) and the antistellar point. Under "naive" T_anti is the 0 K nightside and T_term is only np.interp of the grid between the last dayside and the first nightside node; under "hyperillum" the extended star pushes real flux past 90 deg, which is exactly the physical difference the Bayes factor is measuring. Under "naive_redist" T_anti is the redistributed nightside, T_anti^4 = (1-A_B) F S / (2 emissivity_bol sigma) - the nightside temperature an observer would infer from that model, and 0 K in the truth.
    depth_th / depth_refl / depth : occultation depth [ppm], thermal / reflected / total, in the fit's band. For a joint fit (JointGaussianLogL) the temperatures are band-independent and computed once; the depths come per detector as "depth_th:<label>", "depth_refl:<label>", "depth:<label>", with the un-suffixed keys those of the first detector. Feed depth_th to brightness_temperature() with that detector's cfg.
    """
    comps = logl.components if isinstance(logl, JointGaussianLogL) else [logl]
    labels = list(logl.labels) if isinstance(logl, JointGaussianLogL) else None
    cfg, m = comps[0].cfg, logl.insolation_model
    n_lo, n_la = logl.model_kw.get("n_lo", DEFAULT_N_LO), logl.model_kw.get("n_la", DEFAULT_N_LA)
    # The temperature map of "naive_redist" is the naive insolation map, redistributed (K&L eq. 1). model_pcurve never needs it (Louden is analytic),
    # so GaussianLogL does not cache it - but it is geometry-only, so unless a geometry parameter is sampled it is built ONCE here rather than once
    # per draw (~0.4 s each).
    insol = comps[0].insol
    if insol is None and not any(n in logl.names for n in _GEOMETRY):
        insol = insolation_cache(cfg, m, n_lo=n_lo, n_la=n_la, **{k: v for k, v in logl.fixed.items() if k in _GEOMETRY})

    def _one(theta):
        p_ = logl.unpack(theta)
        geom = {k: p_[k] for k in _GEOMETRY if k in p_}
        A_B_, F_ = p_["A_B"], p_.get("F")

        lo, la, Tg = custom_temperature_grid(swept_planet(cfg, **geom), u_bol=cfg.u_bol, n_lo=n_lo, n_la=n_la, insolation_model=m, F=F_, A_B=A_B_, emissivity_bol=cfg.emissivity_bol, insol=insol)
        eq = int(np.argmin(np.abs(la)))   # equator row; Tg has shape (n_la, n_lo)
        T = list(np.interp([0., 90., 180.], lo, Tg[eq]))   # interpolated, not nearest-cell: T falls steeply through the terminator, so the nearest grid longitude is off by hundreds of K there

        depths = []
        for L in comps:   # one band per detector
            pp = pcurve_params(_orbit_cfg(L.cfg, p_), m, A_B_, F=F_, **geom, n_layers=logl.model_kw.get("n_layers"), n_lo=n_lo, n_la=n_la, insol=insol)
            d_th = float(occultation_depth(pp, stellar_grid=L.cfg.stellar_grid, emissivity=L.cfg.emissivity))
            # Reflected part: linear in A, and its maximum over the orbit is the last point before ingress - within ~1% of the full-phase value, since
            # reflected_unit itself carries the occultation.
            ru = L.refl_unit if L.refl_unit is not None else (reflected_unit(_orbit_cfg(L.cfg, p_), m, n_t=logl.model_kw.get("n_t"), **geom) if L.cfg.reflected else None)
            depths += [d_th, float(L.cfg.A_(A_B_) * np.max(ru)) if ru is not None else 0.]
        return T + depths

    thetas = np.atleast_2d(np.asarray(thetas, float))
    a = np.asarray(_map(_one, list(thetas), n_jobs), float)
    out = {k: a[:, i] for i, k in enumerate(("T_sub", "T_term", "T_anti"))}
    for k in range(len(comps)):
        suf = f":{labels[k]}" if labels else ""
        out["depth_th" + suf], out["depth_refl" + suf] = a[:, 3 + 2 * k], a[:, 4 + 2 * k]
        out["depth" + suf] = out["depth_th" + suf] + out["depth_refl" + suf]
    if labels:   # un-suffixed = the first detector, so single-detector code keeps working
        for key in ("depth_th", "depth_refl", "depth"): out[key] = out[f"{key}:{labels[0]}"]
    out["A_B"] = thetas[:, logl.names.index("A_B")] if "A_B" in logl.names else np.full(len(thetas), logl.fixed["A_B"])
    return out


############################################################################################################################################
#####   Model curves of posterior samples
############################################################################################################################################
def model_samples(logl, thetas, n_jobs=-1):
    """Binned model curves [ppm] for a stack of thetas -> (n_draws, n_bins). One forward model per row, process-parallel via _map."""
    thetas = np.atleast_2d(np.asarray(thetas, float))
    return np.asarray(_map(logl.model, list(thetas), n_jobs), float)
