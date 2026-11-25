"""Priors of the nested-sampling fits: prior transforms and the ordered parameter set."""
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, truncnorm


############################################################################################################################################
#####   Priors
############################################################################################################################################

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
    """Normal(mu, sd) truncated to [lo, hi], e.g. a literature value with its error bar, clipped to the physical range."""
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
    name: str                # one of _KNOWN below, or "offset_<label>"
    prior: object            # callable u -> value, e.g. Uniform(*A_B_BOUNDS)
    label: str = None        # LaTeX label for plots; defaults to the name


@dataclass(frozen=True)
class ParameterSet:
    """The priors: the ordered free parameters of a fit, with the shared ranges of defaults (A_B_BOUNDS, F_BOUNDS, OFFSET_BOUNDS), e.g.

        pset = ParameterSet((Parameter("A_B", Uniform(*A_B_BOUNDS), r"$A_B$"), Parameter("offset", Uniform(*OFFSET_BOUNDS), "offset [ppm]")))
    """
    params: tuple

    def __post_init__(self):
        if not self.params: raise ValueError("a ParameterSet needs at least one free Parameter")
        if len(set(self.names)) != len(self.names): raise ValueError(f"duplicate parameter names in {self.names}")
        bad = [n for n in self.names if n not in _KNOWN and not n.startswith("offset_")]
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
        """n draws from the prior, shape (n, ndim)."""
        u = np.random.default_rng(seed).random((int(n), self.ndim))
        return np.array([self.prior_transform(ui) for ui in u])

    def grid(self, name, n=61):
        """n evenly spaced values spanning one parameter's prior bounds — the x-axis for logl_slice / logl_grid."""
        lo, hi = self.bounds[self.names.index(name)]
        return np.linspace(lo, hi, int(n))


############################################################################################################################################
#####   Geometry parameters
############################################################################################################################################

# Geometry parameters model_pcurve accepts by name (its own kwarg spelling: a_rs = R/r2, rp_rs = r1/r2).
_GEOMETRY = ("a_rs", "T_s", "rp_rs")

# The orbital inclination, as "incl" [deg] or the impact parameter "b" (whichever the planet's source fitted: paper.GEOMETRY_PARAMS).
# Not a model_pcurve keyword: _orbit_cfg puts it into the planet. F is for insolation_model="naive_redist" only.
_ORBIT = ("incl", "b")
_KNOWN = ("A_B", "offset", "F") + _GEOMETRY + _ORBIT
