import numpy as np
import sys
from dataclasses import dataclass, fields, replace
import time
import os
import glob
import re
import pickle
import pandas as pd
import copy
from datetime import datetime
import dynesty
from dynesty.pool import Pool
from joblib import Parallel, delayed
import contextlib, io
from hyperillumination_phasecurve_utils import DEFAULT_N_LO, DEFAULT_N_LA, model_pcurve, reflected_unit, insolation_cache, pcurve_params, occultation_depth, custom_temperature_grid, swept_planet, stellar_spidergrid, FittingConfig, ConstantBandAlbedo  # forward model, its geometry-only caches, and the pieces the derived quantities need
import astro_utils   # T_bright
from dynesty import utils as dyfunc
from scipy.ndimage import gaussian_filter
from simple_bias_analysis import binned, fit_naive, fit_AB_only, fit_redist, fisher_sigma, measurability, temperature_bias, is_detector_set, evidence_residual, nuisance_problem, nuisance_solve, pmap, C_HI, C_NAIVE, C_REDIST, MUTED   # the noiseless least-squares fits of the bias analysis, which plot_noisy_vs_truth and the series table compare the sampler with
from hyperillumination_utils import surface_temperature   # the nightside temperature a fitted F implies
from HItargets_utils import A_B_BOUNDS, A_B_FIT_BOUNDS, F_BOUNDS, OFFSET_BOUNDS, DEFAULT_A, DEFAULT_A_B, DEFAULT_N_BINS, DEFAULT_N_ORBITS, resolve_instrument, joint_name, observed_band, FILTER_DIR   # the parameter ranges: priors (physical) and the least-squares fit range of A_B; the baseline settings and instrument spellings of the dataset tags
from scipy.stats import norm, truncnorm
from scipy.stats import chi2 as chi2_dist   # p-value of a best-fit chi2 (run_prediction)
import warnings
from dynesty.dynamicsampler import stopping_function   # re-evaluated post-run to tell convergence from budget truncation
from dynesty.utils import get_neff_from_logwt
import matplotlib.pyplot as plt


###########################################################################################################################
#####
#####   Simulate Observations
#####
###########################################################################################################################

# Forward-model resolution of the TRUTH simulate_data draws the data from. Well above the likelihood's defaults (model_pcurve: n_t = 50 n_bins,
# n_layers 40 / 60, n_lo / n_la = DEFAULT_N_LO / DEFAULT_N_LA), so that a truth-vs-model residual is model bias and not grid resolution;
# plot_noisy_vs_truth checks that against its "numerical floor". k_truth: sub-samples per bin (n_t = k_truth * n_bins).
TRUTH_KW = dict(k_truth=200, n_layers=80, n_lo=600, n_la=300)


def truth_model_kw(data, cfg):
    """The model_pcurve settings the truth of `data` was computed with, as model_pcurve keywords (n_t, n_layers, n_lo, n_la, and F for a "naive_redist" truth).
    simulate_data records them in data["truth"]["model_kw"]; data without this record are assumed to have the defaults TRUTH_KW."""
    tr = dict(data.get("truth") or {})
    kw = dict(tr.get("model_kw") or dict(n_t=TRUTH_KW["k_truth"] * cfg.n_bins, n_layers=TRUTH_KW["n_layers"], n_lo=TRUTH_KW["n_lo"], n_la=TRUTH_KW["n_la"]))
    if tr.get("F") is not None: kw.setdefault("F", tr["F"])
    return kw


# A detector set (simple_bias_analysis.is_detector_set: a list/tuple of FittingConfigs with the same planet and binning, different band and noise) is
# fitted JOINTLY - one shared (A_B, F), one baseline per detector - by JointGaussianLogL; its dataset is the per-detector simulate_data outputs under
# data["detectors"]. These helpers let every plot treat a single FittingConfig and a set alike.
def detector_cfgs(cfg):
    """A FittingConfig or a detector set -> the list of FittingConfigs."""
    return list(cfg) if is_detector_set(cfg) else [cfg]

def detector_data(data):
    """A simulate_data output -> the list of per-detector data dicts (data["detectors"] of a joint dataset, else the dict itself)."""
    return list(data["detectors"]) if "detectors" in data else [data]

def _blocks(sizes):
    """Index slices of consecutive blocks of the given sizes in a concatenated array (the detectors' blocks of a joint model or residual vector)."""
    edges = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    return [slice(int(edges[k]), int(edges[k + 1])) for k in range(len(sizes))]


def plot_simulated_data(cfg, A_B_true, data, insolation_model="hyperillum", labels=None, verbose=True, **model_kw):
    """The simulated observation against the truth it was drawn from (re-evaluated here, one forward model per detector at the truth's own resolution, truth_model_kw), with the residuals
    in units of sigma. Prints the chi2 check of the noise realisation (verbose=False: silent). One column per detector for a detector set (joint dataset); `labels` names the columns.
    Returns fig, (ax, axr) for a single detector and fig, axes (2 x n_detectors) for a set; does not show (the caller does)."""
    cfgs, datas = detector_cfgs(cfg), detector_data(data)
    labels = list(labels) if labels is not None else [None] * len(cfgs)
    fig, axes = plt.subplots(2, len(cfgs), figsize=(8 * len(cfgs), 5.5), sharex=True, squeeze=False, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.06))
    for k, (c, d, lab) in enumerate(zip(cfgs, datas, labels)):
        ax, axr = axes[0, k], axes[1, k]
        ph = d["t"] / c.planet.period  # orbital phase

        # Full Phase Curve (the exact curve simulate_data drew the data from; model_kw overrides the recorded settings)
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


def simulate_data(cfg, A_B_true, insolation_model="hyperillum", seed=42, noise=True, offset=0.,
                  k_truth=TRUTH_KW["k_truth"], n_layers=TRUTH_KW["n_layers"], n_lo=TRUTH_KW["n_lo"], n_la=TRUTH_KW["n_la"], plotting=True, labels=None, _rng=None, **model_kw):
    """Simulated phase-folded observation: binned(truth) + N(0, sigma_bin).

    cfg          : FittingConfig — supplies planet, filter, stellar_grid, A_(A_B), sigma_1min, n_orbits, n_bins, u_bol, reflected.
                   Or a detector set (list/tuple of FittingConfigs sharing the planet and binning, differing in band and noise, as in the bias analysis): one dataset per detector from the
                   same truth, with ONE noise stream continued across the detectors - the first detector's draws are exactly those of its single-detector dataset with the same seed, the
                   next detector's the following, independent ones. The output then carries the per-detector dicts in data["detectors"] next to the concatenated "t", "y_obs", "sigma".
    A_B_true     : true Bond albedo.
    noise        : False -> the noiseless ("Asimov") dataset. Use it for bias maps; do NOT use it for lnZ / Bayes factors, where chi2 = 0 at the truth inflates Z.
    offset       : constant baseline injected into the data [ppm]; set non-zero to check that the fitted nuisance offset is actually recovered.
    k_truth      : sub-samples per bin for the truth (n_t = k_truth * n_bins, must tile); with n_layers, n_lo, n_la the truth's resolution (TRUTH_KW).
    labels       : detector names for the plot of a detector set.
    model_kw     : further model_pcurve keywords of the truth, e.g. F for insolation_model="naive_redist".
    The injected truth (A_B, insolation model, F, offset, seed, and the forward-model settings) travels with the data in data["truth"], and from there into NestedFit.meta.
    """
    if is_detector_set(cfg):
        rng = np.random.default_rng(seed)   # one stream for all detectors
        parts = [simulate_data(c, A_B_true, insolation_model, seed=seed, noise=noise, offset=offset, k_truth=k_truth, n_layers=n_layers, n_lo=n_lo, n_la=n_la, plotting=False, _rng=rng, **model_kw) for c in cfg]
        data = {k: np.concatenate([d[k] for d in parts]) for k in ("t", "y_obs", "sigma")}
        data["truth"] = dict(parts[0]["truth"])
        data["detectors"] = parts
        if plotting: plot_simulated_data(cfg, A_B_true, data, insolation_model=insolation_model, labels=labels); plt.show()
        return data

    # high-resolution truth (thermal + reflected, since cfg.reflected=True)
    model_kw_truth = dict(n_t=k_truth * cfg.n_bins, n_layers=n_layers, n_lo=n_lo, n_la=n_la, **model_kw)
    t_hi, y_hi = model_pcurve(cfg, insolation_model, A_B_true, **model_kw_truth)

    # reduce to the observed bins
    y_true = binned(cfg, y_hi) + offset

    # adding noise (a stream handed in by the detector-set branch continues across detectors)
    s = cfg.sigma_bin
    rng = np.random.default_rng(seed) if _rng is None else _rng
    y_obs = y_true + rng.normal(0., s, cfg.n_bins) if noise else y_true.copy()

    t=(np.arange(cfg.n_bins) + 0.5) * cfg.planet.period / cfg.n_bins
    sigma = np.full(cfg.n_bins, s)

    # the injected truth travels with the data (and from there into NestedFit.meta), so a saved fit knows what it was fitted to; model_kw lets
    # plot_simulated_data / plot_noisy_vs_truth re-evaluate exactly this truth
    F_true = model_kw.get("F")
    truth = dict(A_B=float(A_B_true), insolation_model=insolation_model, F=None if F_true is None else float(F_true), offset=float(offset), seed=int(seed), noise=bool(noise), model_kw=model_kw_truth)
    data = {"t": t, "y_obs": y_obs, "sigma":sigma, "truth": truth}

    if plotting: plot_simulated_data(cfg, A_B_true, data, insolation_model=insolation_model); plt.show()   # plot_simulated_data returns the figure and does not show it itself

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
    """Normal(mu, sd) restricted to [lo, hi]. Use e.g. for r1/r2: the literature value with its error bar, hard-clipped to a physically allowed range."""
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
    prior: object            # callable u -> value, e.g. Uniform(*A_B_BOUNDS)
    label: str = None        # LaTeX label for plots; defaults to the name


@dataclass(frozen=True)
class ParameterSet:
    """This class defines the priors: the ordered free parameters of a fit.
    Call by e.g.: pset = ParameterSet((Parameter("A_B", Uniform(*A_B_BOUNDS), r"$A_B$"), Parameter("offset", Uniform(*OFFSET_BOUNDS), "offset [ppm]")))
    The ranges are the shared ones of HItargets_utils: A_B_BOUNDS, F_BOUNDS (physical) and OFFSET_BOUNDS.
    """
    params: tuple

    def __post_init__(self):
        if not self.params: raise ValueError("a ParameterSet needs at least one free Parameter")
        if len(set(self.names)) != len(self.names): raise ValueError(f"duplicate parameter names in {self.names}")
        bad = [n for n in self.names if n not in _KNOWN and not n.startswith("offset_")]   # "offset_<label>": one baseline per detector of a joint fit (JointGaussianLogL)
        if bad: raise ValueError(f"unsupported parameter(s) {bad}; supported: {_KNOWN} and offset_<label> for the detectors of a joint fit")

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
# F (heat redistribution) belongs to insolation_model="naive_redist" alone: required there, rejected for the other models. The orbital inclination,
# sampled either directly ("incl" [deg]) or through the impact parameter ("b", with i = arccos(b / (a/R*)) at the sampled or the planet's own a/R*):
# whichever the planet's source fitted (Tas et al. 2026 fit b for TOI-2431 b, Malavolta et al. 2018 fit i for K2-141 b). Not a model_pcurve keyword:
# _orbit_cfg puts it into the planet.
_ORBIT = ("incl", "b")
_KNOWN = ("A_B", "offset", "F") + _GEOMETRY + _ORBIT

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
    """Several detectors fitted JOINTLY - the dynesty counterpart of a detector set in simple_bias_analysis (e.g. the two NIRSpec/G395H arrays): one GaussianLogL per detector, the
    physical parameters (A_B, F, geometry) shared, one baseline per detector, and lnL = the sum of the detectors' lnL (independent data).

    theta is ordered by `names`: the shared names plus one "offset_<label>" per detector; `labels` are the detectors' short names in the order of `cfgs` (e.g. ("NRS1", "NRS2"),
    HItargets_utils.detector_label). Each component sees names = shared + ("offset",) and its own data dict, so every cache, check and model keyword works per detector exactly as
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


def load_fit(path):
    """Read back a NestedFit written by run_nested_fit (scfg.outfile). Unpickling needs this module importable, since pset/scfg/NestedFit are defined here."""
    path = os.path.abspath(os.path.expanduser(path))
    with open(path, "rb") as f: fit = pickle.load(f)
    fit.meta["file"] = path   # run_nested_fit records the path only AFTER pickling, so the file on disk never carries its own name
    _relocate_filter_paths(fit)
    return fit


def _relocate_filter_paths(fit):
    """Point the filter files of a saved fit's cfg at this repository's inputs/filters/Spiderman/. The pickle stores the path on the machine that ran the fit
    (absolute, and possibly in another folder layout; the example runs store it relative to the repository), and check_fit_current,
    logl_from_fit and every forward-model call read the file. A stored path that exists is kept, unless it is the same file as the one in
    inputs/filters/Spiderman/ under another spelling (a symlink, or a different case on a case-insensitive disk): it then takes the canonical spelling,
    which observed_band looks the instrument up by."""
    if fit.meta.get("cfg") is None: return
    for c in detector_cfgs(fit.meta["cfg"]):
        path = getattr(c, "filter_path", None)
        if not isinstance(path, str): continue
        local = FILTER_DIR / os.path.basename(path)
        if local.exists() and (not os.path.exists(path) or os.path.samefile(path, local)):
            c.filter_path = str(local)


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


###########################################################################################################################
#####
#####   Dataset tags: the file names run_dynesty.py writes, and finding a stored dataset again
#####
###########################################################################################################################

# A dataset is one simulated observation (planet, instrument, albedos, binning, noise, seed) sampled with every model at one pfrac; its tag is
# everything the models share, and a result file is tag + "_{insolation_model}_priors-<sampled names>.pkl". The free parameters come LAST, so one glob
# on the tag picks up every model of the dataset whatever each was sampled with, while runs with different free parameters cannot overwrite each
# other. A joint fit lists the noise levels of its detectors joined by "+". pfrac (the sampler's posterior/evidence weighting, shared by every model
# of a run) ends the tag: the same data sampled with another pfrac is another run.
DATASET_TAG = "{planet}_{instrument}_{A_label}_AB{A_B:g}_nbins{n_bins}_norbits{n_orbits}_rms1min{rms}_sigmascale{sigma_scaling:g}_seed{seed}_pfrac{pfrac:g}"
_RESULT_FILE_RE = re.compile(r"^(?P<planet>[^_]+)_(?P<instrument>.+?)_(?P<A_label>(?:fixed|scaled)A[\d.]+)_AB(?P<A_B>[\d.]+)_nbins(?P<n_bins>\d+)_norbits(?P<n_orbits>\d+)"
                             r"_rms1min(?P<rms>[\d.+]+)_sigmascale(?P<sigma_scaling>[\d.]+)_seed(?P<seed>\d+)_pfrac(?P<pfrac>[\d.]+)"
                             r"_(?P<insolation_model>hyperillum|naive_redist|naive)_priors-(?P<priors>.+)\.pkl$")
DATASET_COLUMNS = ["planet", "instrument", "A_B", "n_bins", "n_orbits", "seed", "pfrac", "sigma_scaling", "A_label", "rms", "models", "subfolder", "tag"]
# Runs whose difference from the baseline is NOT in the tag sit in a subfolder of the results folder: the truth model ("null_test": data drawn from
# the point-source model) and the sampled geometry ("free_geometry": a/R* and Rp/R* free in every model). A dataset entry names it with the keyword
# subfolder= (find_dataset, check_datasets); list_datasets lists the main folder and every subfolder. run_kind reads the same two properties from the
# pickles themselves.


def _planet_name(planet):
    """File-name spelling of a planet: its PlanetParams name or the string given, without spaces ("TOI-2431 b" -> "TOI-2431b")."""
    return (planet.name if hasattr(planet, "name") else str(planet)).replace(" ", "")


def _instrument_name(instrument):
    """File-name spelling of an `instruments` entry of run_dynesty.py: the band ("MIRI/LRS" -> "MIRI-LRS"), or the joint alias / 'joint-<labels>' of a detector tuple."""
    dets = resolve_instrument(instrument)
    return (joint_name(dets) if isinstance(dets, tuple) else dets).replace("/", "-")


def dataset_tag(planet, instrument, sigmas=None, *, A_label=None, A_B=DEFAULT_A_B, n_bins=DEFAULT_N_BINS, n_orbits=DEFAULT_N_ORBITS, sigma_scaling=1., seed=42, pfrac=SamplerConfig.pfrac):
    """The tag of a dataset (DATASET_TAG). `planet` is a PlanetParams or its name, `instrument` an entry of run_dynesty.py's `instruments` (a band, a joint alias or a
    detector tuple), `sigmas` the noise levels sigma_1min [ppm] per detector - None puts the glob wildcard in their place, to find a stored run without knowing them
    (find_dataset). The keywords are the data settings and the sampler's pfrac; unset ones are the baseline (the DEFAULT_* of HItargets_utils, seed 42, SamplerConfig's
    pfrac). `A_label` is the band albedo's label (A_model.label, e.g. 'fixedA0.1'); None -> ConstantBandAlbedo(DEFAULT_A)."""
    rms = "*" if sigmas is None else "+".join(f"{s:.1f}" for s in np.atleast_1d(sigmas))
    return DATASET_TAG.format(planet=_planet_name(planet), instrument=_instrument_name(instrument), A_label=A_label or ConstantBandAlbedo(DEFAULT_A).label, A_B=A_B,
                              n_bins=n_bins, n_orbits=n_orbits, rms=rms, sigma_scaling=sigma_scaling, seed=seed, pfrac=pfrac)


def parse_result_file(path):
    """The fields of a result file's name as a dict: dataset_tag's keywords (as numbers), 'rms' (the noise levels, '+'-joined), 'insolation_model', 'priors' (the sampled
    names) and the dataset 'tag'. None for a file that is not a result file."""
    name = os.path.basename(path)
    m = _RESULT_FILE_RE.match(name)
    if not m: return None
    d = m.groupdict()
    d.update(A_B=float(d["A_B"]), n_bins=int(d["n_bins"]), n_orbits=int(d["n_orbits"]), sigma_scaling=float(d["sigma_scaling"]), seed=int(d["seed"]), pfrac=float(d["pfrac"]))
    d["tag"] = name[:m.start("insolation_model") - 1]
    return d


def list_datasets(results_dir):
    """Every dataset in `results_dir` and its subfolders as a DataFrame, one row per folder and tag (DATASET_COLUMNS: the settings, the noise levels 'rms' per detector, the
    insolation models sampled, the subfolder - '' for the main folder - and the tag) - the menu of find_dataset. Files that are not result files are ignored."""
    rows = {}
    for f in sorted(glob.glob(os.path.join(results_dir, "*.pkl")) + glob.glob(os.path.join(results_dir, "*", "*.pkl"))):
        d = parse_result_file(f)
        if d is None: continue
        sub = os.path.relpath(os.path.dirname(f), results_dir)
        d["subfolder"] = "" if sub == "." else sub
        rows.setdefault((d["subfolder"], d["tag"]), {**{k: d[k] for k in DATASET_COLUMNS if k != "models"}, "models": []})["models"].append(d["insolation_model"])
    df = pd.DataFrame(list(rows.values()), columns=DATASET_COLUMNS)
    df["models"] = [", ".join(m) for m in df["models"]]
    return df.sort_values(["subfolder", "planet", "instrument", "A_label", "A_B", "n_bins", "n_orbits", "sigma_scaling", "seed", "pfrac"], ignore_index=True)


def find_dataset(results_dir, planet, instrument, subfolder=None, **settings):
    """The ONE stored dataset of `planet` and `instrument` with these settings (dataset_tag's keywords; unset ones are the baseline), whatever noise levels the run had, as the
    stem 01_retrieval_example.ipynb globs on: its tag, or 'subfolder/tag' for a run in a subfolder of `results_dir` (subfolder='null_test', 'free_geometry'), so that
    f"{results_dir}/{stem}_*.pkl" finds its files either way. Raises when none or several match, listing what `results_dir` holds for the planet."""
    pattern = dataset_tag(planet, instrument, None, **settings)
    folder = os.path.join(results_dir, subfolder) if subfolder else results_dir
    parsed = [parse_result_file(f) for f in glob.glob(os.path.join(folder, pattern + "_*.pkl"))]
    tags = sorted({d["tag"] for d in parsed if d})
    if len(tags) == 1: return os.path.join(subfolder, tags[0]) if subfolder else tags[0]
    have = list_datasets(results_dir)
    have = have[have.planet == _planet_name(planet)].drop(columns="tag")
    msg = (f"{len(tags) or 'no'} dataset(s) in {folder} match {pattern}" + ("".join("\n   " + t for t in tags)) +
           f"\n{results_dir} holds for {_planet_name(planet)}:\n" + (have.to_string(index=False) if len(have) else "   (nothing)"))
    raise (ValueError if tags else FileNotFoundError)(msg)


def check_datasets(results_dir, datasets):
    """Keep a notebook's list of datasets, {key: the keywords of find_dataset}, and `results_dir` (with its subfolders) in step: prints the entries that have no run and the
    runs that are not on the list, and returns {key: stem} of the entries found (find_dataset's return). An entry matching several runs raises (find_dataset)."""
    found = {}
    for key, kw in datasets.items():
        try: found[key] = find_dataset(results_dir, **kw)
        except FileNotFoundError: print(f"no run for {key!r}")
    have = list_datasets(results_dir)
    for stem in sorted({os.path.join(sub, tag) if sub else tag for sub, tag in zip(have.subfolder, have.tag)} - set(found.values())): print(f"run not listed: {stem}")
    return found


def check_fit_current(fit, cfg=None, rtol=1e-6):
    """Is a saved fit still a fit of the current forward model? Returns the relative difference between the stellar grid the fit was run with (pickled inside its cfg) and the one the
    current code builds from the same inputs (star, band, filter file, stellar model), the largest over the detectors of a joint fit. 0 means identical; anything else means the
    thermal normalisation changed since the run (stellar SED, filter convention, band integration, ...), so the saved samples are not the posterior of the current likelihood and every
    plot that re-evaluates the forward model would mix the two. inf (with a printed WARNING) where the planet's band itself changed since the run (a HItargets_utils.PARTIAL_BANDS
    entry: the current code would integrate another band). Also raises for a cfg saved with an older FittingConfig (missing fields), which the forward model could not evaluate anyway."""
    cfg = cfg if cfg is not None else fit.meta["cfg"]
    rel_max, band_changed = 0., False
    for c in detector_cfgs(cfg):
        missing = [f.name for f in fields(FittingConfig) if not hasattr(c, f.name)]
        if missing: raise ValueError(f"the saved FittingConfig predates the field(s) {missing}: this fit ({fit.meta.get('file', '?')}) was run with an older forward model and has to be re-run")
        band = observed_band(c.filter_path, c.l1, c.l2, c.planet)   # the band the current code observes this planet in (a PARTIAL_BANDS entry added since the run moves it)
        if not np.allclose(band, (c.l1, c.l2), rtol=0., atol=1e-12):
            print(f"WARNING [{fit.meta['insolation_model']}]: {fit.meta.get('file', 'this fit')} was run over {1e6 * c.l1:.2f}-{1e6 * c.l2:.2f} um, but the current code observes "
                  f"{c.planet.name} in {1e6 * band[0]:.2f}-{1e6 * band[1]:.2f} um of this band (HItargets_utils.PARTIAL_BANDS). Re-run the fit before quoting it.", file=sys.stderr, flush=True)
            band_changed = True
        now = stellar_spidergrid((c.l1, c.l2), c.filter_path, c.planet.logg, c.planet.feh, stellar_model=c.stellar_model)
        then = c.stellar_grid
        rel = np.inf if (np.shape(now[1]) != np.shape(then[1]) or not np.allclose(now[0], then[0])) else float(np.max(np.abs(np.asarray(now[1], float) / np.asarray(then[1], float) - 1.)))   # inf: different T_eff nodes
        rel_max = max(rel_max, rel)
    if rel_max > rtol:
        print(f"WARNING [{fit.meta['insolation_model']}]: the stellar grid of the current code differs from the one this fit was run with by up to {100 * rel_max:.3g} % "
              f"(stellar model, filter or band changed since {fit.meta.get('file', 'the run')}). Its samples are not the posterior of the current likelihood: re-run the fit before quoting it.", file=sys.stderr, flush=True)
    return np.inf if band_changed else rel_max


def logl_from_fit(fit, cfg=None, **model_kw):
    """The GaussianLogL a saved fit was run with, rebuilt from the pickle alone - what the plotting helpers that still
    evaluate the forward model (logl_slice/logl_grid, plot_posterior_predictive, derived_samples, plot_noisy_vs_truth) need.

    Uses the FittingConfig (or detector set) stored in meta by run_nested_fit (with the stellar grid as it was at run time); pass `cfg` explicitly for a fit whose meta has none.
    check_fit_current warns when the current code would build a different stellar grid, i.e. when the forward model has changed since the run.
    The geometry-only caches are rebuilt here, so the first call costs a second or two."""
    cfg = cfg if cfg is not None else fit.meta.get("cfg")
    if cfg is None: raise ValueError("this fit has no cfg in its meta - pass cfg=... (or re-run the fit)")
    check_fit_current(fit, cfg)
    kw = {**fit.meta.get("model_kw", {}), **model_kw}
    return make_logl(cfg, fit.meta["data"], fit.meta["insolation_model"], fit.pset, labels=fit.meta.get("labels"), fixed=fit.meta.get("fixed"), **kw)   # a JointGaussianLogL for a detector set


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


def brightness_temperature(depth_th, cfg, quantiles=(0.16, 0.5, 0.84)):
    """Dayside brightness temperature [K] at the given quantiles of the THERMAL occultation depth (a blackbody brightness temperature: below the surface temperature for cfg.emissivity < 1).
    T_bright is monotone in the depth, so transforming the quantiles is exact - and it keeps the brentq/band_avg_planck cost at three calls instead of one per draw."""
    q = np.quantile(np.asarray(depth_th, float), list(quantiles))
    return np.array([astro_utils.T_bright(d, cfg.planet, (cfg.l1, cfg.l2), cfg.filter_path, stellar_model=cfg.stellar_model) for d in q])


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

def plot_posterior_predictive(logls, fits, num="hyperillum", den="naive", n=200, colors=None, seed=0, n_jobs=-1, figsize=None, levels=(68., 95.)):
    """Posterior predictive check + where in orbital phase the two models actually differ. One column per detector for a joint fit.

    Panel 1: data with each model's median curve and its 68/95% posterior band.
    Panel 2: (data - median model) / sigma per model.
    Panel 3: the difference of the two median curves [ppm] against +/- sigma_bin, and the per-bin delta chi2 between the two BEST-FIT (max-lnL) models as bars. Those bars sum to 2 * delta lnL_max (over all detectors for a joint fit), so they attribute the evidence to specific orbital phases.

    num / den : which model is which in panel 3 - both the curve and the bars are num - den, same convention as bayes_factor().
    n         : posterior draws per model (n forward models per model and detector).
    Returns fig, (ax, axr, axd) for a single detector and fig, axes (3 x n_detectors) for a joint fit."""
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
    mk = ("o", "s", "^", "v")   # the residual sets sit on top of each other when the models fit almost equally well
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

        # Panel 3: the num/den pair only - with more than two models every other one is still in panels 1-2, but a single difference curve can only be
        # a pair
        if num in fits and den in fits:
            a, b = den, num
            axd.axhspan(-s.mean(), s.mean(), fc="0.9", ec="0.6", lw=0.8, label=r"$\pm\sigma_{bin}$")
            axd.axhline(0., color="0.7", lw=0.8)
            dmed = med[b] - med[a]
            axd.plot(ph, dmed, color="0.2", lw=1.5, label=f"median {b} - {a} [ppm]")
            dchi2 = ((y - best[a][blk]) ** 2 - (y - best[b][blk]) ** 2) / s ** 2   # > 0 where b fits better; sums to 2*dlnL_max over all detectors
            ax2 = axd.twinx()
            ax2.bar(ph, dchi2, width=0.7 / len(ph), color="C2", alpha=0.45, zorder=0)
            # Both axes forced symmetric about zero. twinx() autoscales independently, so a one-signed dchi2 makes matplotlib clamp ax2 to [0, max]
            # and the bars then grow from the panel floor while axd's zero line sits mid-panel - the reader reads the bars against a baseline that is
            # not theirs.
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
            if np.ptp(v) == 0.:   # a constant, e.g. the 0 K nightside (T_anti) of "naive" and "hyperillum": a histogram of one value has an arbitrary width, so draw a line (dash lengths distinguish coincident models)
                ax.axvline(v[0], color=colors.get(name, "k"), lw=1.4, ls=(0, (4 + 2 * j, 2)), label=f"{name} (= {v[0]:g})")
            else:
                ax.hist(v, bins=bins, density=True, histtype="step", lw=1.4, color=colors.get(name, "k"), label=name)
            if truth and name in truth: ax.axvline(np.median(truth[name][k]), color=colors.get(name, "k"), lw=1.0, ls="--")
        ax.set_xlabel(f"{k} {unit.get(k.split(':')[0], '')}"); ax.set_yticks([])   # "depth:<label>" of a joint fit shares the unit of "depth"
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, axes


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


def plot_noisy_vs_truth(cfg, data, logls, fits, A_B_true, base, insolation_truth="hyperillum",
                        A_B_bounds=A_B_FIT_BOUNDS, xatol=None, truth_kw=None, fisher=True, colors=None, figsize=None, verbose=False):
    """The one-seed fits against the noiseless truth: what you would report vs. what the bias actually is. One column per detector for a joint fit.

    Top panel   : the high-resolution truth and its binned version (= the noiseless data), the simulated data, each model's posterior MEDIAN curve, and the misspecified model(s) fitted
                  to the NOISELESS truth with the least-squares fits of the bias analysis (simple_bias_analysis.fit_naive for "naive"; fit_redist, jointly in (A_B, F), when a
                  "naive_redist" fit is loaded; for a detector set the joint fit with one baseline per detector), offset free - those are deterministic (no seed, no noise) and ARE the
                  bias. For "naive_redist" the fitted F is the false-positive redistribution the point-source model needs to mimic hyperillumination, and the nightside temperature it implies is printed.
    Lower panel : everything minus the binned truth against +/- sigma_bin. The black points are the noise realisation the medians are chasing; the dotted grey line is the numerical
                  floor, i.e. the truth model re-evaluated at the injected theta on the likelihood's own coarser grid, so anything at that level is grid resolution and not model bias.

    cfg / data       : the FittingConfig (or detector set) and the simulate_data() output the fits were run on.
    logls / fits     : {model_name: GaussianLogL or JointGaussianLogL} and {model_name: NestedFit}, same keys. "naive" and "naive_redist" get a noiseless fit each when present; the truth's own model only its median.
    base             : {param: injected value} (the dict handed to theta_of / posterior_summary; every baseline included).
    insolation_truth : which insolation model the data were drawn from; must be a key of logls for the numerical floor to be drawn.
    A_B_bounds       : range of the noiseless fits, i.e. of the bias; A_B_FIT_BOUNDS by default, as in the bias analysis (A_B_fit < 0 allowed), not the sampler's prior.
    xatol            : tolerance of the noiseless fits; None -> the fitters' own defaults, so the numbers are exactly those of the bias analysis.
    truth_kw         : forward-model resolution of the truth; defaults to what simulate_data recorded in data["truth"] (truth_model_kw).
    fisher           : also evaluate the Fisher sigma of each noiseless fit (simple_bias_analysis.fisher_sigma, the error bar of the bias analysis) and quote the bias in its units next to
                       the posterior width of this one seed: the two agree where the posterior is Gaussian and not truncated by the prior.
    verbose          : False prints the noiseless fits, the bias in sigma with the critical sigma_1min and the bias/floor gate; True adds the per-model median and max-lnL rows, the absorbed offsets and the floor detail.

    Returns fig, (ax, axr), out for a single detector (fig, axes, out with axes 2 x n_detectors for a joint fit). out["noiseless"][model] carries every number of a noiseless fit
    (A_B, F, offset [per detector], chi2, rms, bias_A_B, dT_over_T [substellar], T_night [naive_redist], sigma_A_B / sigma_F [posterior, this seed], bias_sigma0, sigma_1min_crit
    [None for a joint fit: its detectors would all be scaled alike], bias_over_floor, and with fisher=True sigma_A_B_fisher / sigma_F_fisher, bias_sigma_fisher, sigma_1min_crit_fisher);
    the rest are the curves (t_hi, y_hi, y_true, y_med, y_map, y_floor; per detector where applicable), the posterior summaries and the per-model chi2 against the truth.
    """
    colors = colors or _COLOR
    cfgs, datas = detector_cfgs(cfg), detector_data(data)
    n_det, cfg0, joint = len(cfgs), detector_cfgs(cfg)[0], len(detector_cfgs(cfg)) > 1
    L0 = next(iter(logls.values()))
    labels = list(getattr(L0, "labels", None) or [None] * n_det)
    blocks = _blocks([c.n_bins for c in cfgs])

    # --- the noiseless reference (the truth curves at their own resolution) and the misspecified models fitted to it: the deterministic bias, no seed
    # involved ---
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
        if d["F"] is not None: d["T_night"] = float(surface_temperature(np.pi, cfg0.planet, "naive_redist", A_B=d["A_B"], emissivity_bol=cfg0.emissivity_bol, F=d["F"]))   # the nightside the false-positive F implies (0 K in the truth)

    # --- the noisy fits: posterior median (what you would report) and max-lnL (= MAP, uniform priors) -------------
    summ   = {m: posterior_summary(f, base, verbose=False)["params"] for m, f in fits.items()}
    th_med = {m: np.array([summ[m][n]["med"] for n in f.pset.names], float) for m, f in fits.items()}
    th_map = {m: f.results.samples[int(np.argmax(f.results.logl))] for m, f in fits.items()}
    y_med  = {m: logls[m].model(th_med[m]) for m in fits}                  # concatenated over the detectors (blocks)
    y_map  = {m: logls[m].model(th_map[m]) for m in fits}
    y_floor = logls[insolation_truth].model(theta_of(logls[insolation_truth], base)) if insolation_truth in logls else None   # same model, same parameters, coarser grid: the numerical floor of this figure

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
        d["bias_sigma0"] = abs(d["bias_A_B"]) / sig                       # the NOISELESS bias in units of the posterior width - the seed-free counterpart of summ[m]["A_B"]["bias_sigma"], whose numerator is one realisation's median
        d["sigma_1min_crit"] = None if joint else cfg0.sigma_1min * d["bias_sigma0"]   # sigma_A_B scales linearly with sigma_1min (at fixed n_orbits/n_bins) while the bias does not, so |dA_B| = sigma_A_B exactly at this per-minute precision; sigma_A_B is read off THIS seed's posterior, so it carries one realisation's scatter, and the linear scaling assumes the posterior is not truncated by the prior box. A joint fit has one sigma_1min per detector: only the common factor is quoted.
        if d["F"] is not None:
            d["sigma_F"] = float(0.5 * (summ[m]["F"]["hi"] - summ[m]["F"]["lo"]))
            d["F_sigma0"] = d["F"] / d["sigma_F"]                            # the false-positive F in units of its (one-sided, prior-truncated) posterior width
        if fisher:   # the error bar of the bias analysis: Fisher sigma at the noiseless fit, marginalised over the baseline(s) (and over F for naive_redist)
            sf = fisher_sigma(fit_cfg, m, d["A_B"], F=d["F"], bounds=A_B_bounds, F_bounds=F_BOUNDS, insol=insols[m], **truth_kw)
            d["sigma_A_B_fisher"] = float(sf[0])
            d["bias_sigma_fisher"], crit = (float(x) for x in measurability(d["bias_A_B"], sf[0], cfg0.sigma_1min))
            d["sigma_1min_crit_fisher"] = None if joint else crit
            if d["F"] is not None: d["sigma_F_fisher"] = float(sf[1])
        d["bias_over_floor"] = (d["chi2"] / chi2_floor) if chi2_floor else None   # the gate on this whole figure: how far the model-vs-truth distance stands above the forward model's own discretisation error

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
        fac = (1. / d["bias_sigma0"]) if d["bias_sigma0"] > 0 else np.inf      # how much finer sigma_1min must get before the bias is a 1-sigma effect
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


###########################################################################################################################
#####
#####   Retrievals against the linear prediction of the Bayes factor (03_retrieval_series.ipynb)
#####
###########################################################################################################################

# The prediction (simple_bias_analysis, "Expected evidence"): for two models with the same parameters and priors, ln B(true vs alt) = chi2_min / 2 +
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
    (noiseless_reference), the misspecified model (alternative_model) fitted to it, its whitened residual r_perp (simple_bias_analysis.evidence_residual) and its Fisher
    sigma_A_B, all at the truth's resolution. floor: also the numerical floor, chi2 of the truth's own model at the injected parameters on the likelihood's coarser grid
    against the truth curve (model resolution, not model bias). Runs that differ only in seed and pfrac share it.
    A_B_bounds: the range of the noiseless fit (A_B_FIT_BOUNDS: the bias analysis; A_B_BOUNDS: held at the physical floor, where r_perp is no longer orthogonal to the A_B response).
    A run that samples the geometry (a_rs, rp_rs and incl or b with Gaussian priors, read from its ParameterSet) gets the prediction WITH those nuisances: the linear nuisance fit of
    simple_bias_analysis (nuisance_problem / nuisance_solve) around the fixed-geometry fit, so that A_B_fit, bias_A_B, chi2, sigma_fisher and r_perp (the data rows of the augmented
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
    row = dict(planet=c0.planet.name, instrument=joint_name([_band_key(c) for c in detector_cfgs(f_a.meta["cfg"])]) if is_detector_set(f_a.meta["cfg"]) else _band_key(c0),
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


def _band_key(cfg):
    """The FILTER_LAMBDA_UM key of a FittingConfig's band, matched on its throughput file (the file name itself where none matches)."""
    from HItargets_utils import FILTER_FILES
    name = os.path.basename(str(cfg.filter_path))
    return next((k for k, v in FILTER_FILES.items() if v == name), name)


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


def plot_ns_check(tab, groups=None, figsize=None):
    """The retrievals against the noiseless predictions, three panels over the rows of a series_table: (a) the paired posterior-median difference (misspecified model minus the
    truth's model, same data) against the noiseless bias; (b) the posterior width of A_B under the misspecified model against the Fisher sigma; (c) the measured ln B (extended
    source vs point source) against the per-run prediction chi2_min / 2 + zeta sqrt(chi2_min), with the sampler's error (jitter where available, else analytic) as error bars.
    A table with the noiseless fit with redistribution (series_table(..., redist=True)) gets a fourth panel: (d) the significance of the spurious F in each realisation, from the
    likelihood ratio of the two best fits, against its prediction max(z_F + zeta_F, 0).
    Dashed: the one-to-one lines. groups: {legend label: boolean mask over tab} for the marker groups (default: one per series). Returns fig, axes."""
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
        shown = tab[np.any(list(groups.values()), axis=0)]   # the axis range follows the plotted rows, not rows that no group shows (e.g. null tests, whose bias has the other sign)
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
    """The noise realisations of one configuration side by side: what each simulated observation returns. One column per entry of `panels` ({title: boolean mask over tab}), the
    runs of a column ordered by `order` (default: the seed, so the same number is the same standardised noise in every column; e.g. "lnB_pred" sorts each column by its predicted
    ln B). With more than 15 realisations the markers shrink, the z_F numbers of (b) are left out and every fifth realisation is ticked.
    (a) the posterior of A_B under the extended-source model, the point-source model and the point-source model with redistribution: median, 68 % (thick) and 95 % (thin) intervals;
        dashed: the injected value; thin lines in the model colours: the noiseless fits of the two point-source models.
    (b) the posterior of F with the same intervals, the noiseless fitted F (line) and, above each realisation, the significance of F in it (zF_obs).
    (c) ln B of the extended-source over the point-source model: measured (filled, with the sampler's error) and predicted for that realisation's noise (open), on the expectation
        chi2_min / 2 (line) with its +/- sqrt(chi2_min) band; squares: ln B of the point-source model with over without redistribution. Dotted: ln B = 2.5 and 5 (Trotta 2008).
    Needs a series_table(..., redist=True) and runs with all three models. The legend has a row of its own above the panels. Returns fig, axes (3 x number of panels)."""
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
    order = [models[0][3], models[1][3], models[2][3], L_INJ, L_NOISELESS, L_EXP, L_LNB, L_PRED, L_LNB_F]   # filled column by column: the models and references of (a) and (b), then the Bayes factors of (c)
    ax_leg = fig.add_subplot(gs[0, :])
    ax_leg.axis("off")
    ax_leg.legend([entries[l] for l in order], order, loc="center", ncol=5, frameon=False, fontsize=8)
    return fig, axes


def plot_ns_distribution(tab, rows, others=None, figsize=None):
    """The noise realisations of one configuration as distributions: what a single observation can return, and how often. One row of three panels per entry of `rows`
    ({title: boolean mask over tab}); needs a series_table(..., redist=True) and runs with all three models.
    (a) the posterior median of A_B under the point-source model (circles) and the point-source model with redistribution (squares) against the median under the extended-source
        model in the same realisation. Dashed: equal medians; lines in the model colours: the extended-source median shifted by the noiseless bias of each fit; cross: the injected
        value; dash-dotted: the prior edge A_B = 0; the bar in the corner: the median 68 % half-widths of the extended-source and the point-source posterior.
    (b) the fraction of realisations whose ln B (extended vs point source) exceeds the value on the x axis (steps), against the Gaussian prediction N(chi2_min / 2, chi2_min) of
        the noiseless fit (dashed). others: one {label: mask} per row for further configurations in grey (e.g. another planet in the same mode). Dotted: ln B = 2.5 and 5.
    (c) the significance z_F of the spurious F in each realisation against ln B of that realisation; the fill tells what the Bayes factor of the point-source model with over
        without redistribution (ln B_F) concludes. Horizontal line: the noiseless z_F (its predicted median); vertical line: chi2_min / 2.
    Returns fig, axes (number of rows x 3)."""
    fig, axes = plt.subplots(len(rows), 3, figsize=figsize or (13., 4.1 * len(rows)), squeeze=False)
    x_B = np.linspace(-4., 20., 500)
    for k, ((title, mask), (ax_A, ax_B, ax_F)) in enumerate(zip(rows.items(), axes)):
        t = tab[mask].sort_values("seed")
        A0, dA, dAF, c2 = t["A_B_true"].iloc[0], t["dA_noiseless"].iloc[0], t["dA_noiseless_redist"].iloc[0], t["chi2_min"].iloc[0]

        lo, hi = min(t["A_B_med_truth"].min(), A0) - 0.03, max(t["A_B_med_truth"].max(), A0) + 0.03   # (a) the point-source medians against the extended-source median
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

        groups = [(title.split(",")[0], t, "k")] + [(lab, tab[m], "0.6") for lab, m in (others[k].items() if others else ())]   # (b) fraction of realisations above ln B
        for lab, g, color in groups:
            v, c = np.sort(g["lnB"].to_numpy()), g["chi2_min"].iloc[0]
            ax_B.step(np.concatenate([[x_B[0]], v, [x_B[-1]]]), np.concatenate([[1.], 1. - np.arange(1, len(v) + 1) / len(v), [0.]]), where="post", color=color, lw=1.6, label=f"{lab} ({len(v)})")
            ax_B.plot(x_B, norm.sf((x_B - 0.5 * c) / np.sqrt(c)), color=color, lw=1., ls="--", label="Gaussian prediction" if color == "k" else None)
        ax_B.axvline(0., color="k", lw=0.7, zorder=1)
        ax_B.set_ylim(0., 1.02); ax_B.set_ylabel(r"fraction of realizations above $\ln B$")

        b = t["lnB_redist_vs_ps"].to_numpy()   # (c) the significance of F against ln B; the fill: the verdict of the Bayes factor with vs without redistribution
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
