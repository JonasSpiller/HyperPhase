"""The model of the fits: FittingConfig (planet, band, albedo, emissivities, noise and observing plan) and model_pcurve, the phase curve of a
configuration on the time grid that the fits bin."""
import contextlib
import io
from dataclasses import dataclass, field, replace

import numpy as np
with contextlib.redirect_stdout(io.StringIO()):   # mute "looking for spidermanrc..." in every process
    import fspiderman as sp

from hyperphase.bands import observed_band
from hyperphase.defaults import (DEFAULT_A, DEFAULT_EMISSIVITY, DEFAULT_EMISSIVITY_BOL, DEFAULT_N_BINS, DEFAULT_N_LA, DEFAULT_N_LO, DEFAULT_N_ORBITS,
                                 DEFAULT_U, DEFAULT_U_BOL, STELLAR_MODEL)
from hyperphase.irradiance import EXTENDED_SOURCE_MODELS, check_redistribution
from hyperphase.phasecurve import custom_temperature_grid, fast_lightcurve, insolation_grid, louden_params, reflected_PhaseCurve
from hyperphase.stellar import check_teff_nodes, stellar_spidergrid
from hyperphase.transit import flux_transit


# The band albedo A as a function of the Bond albedo A_B (FittingConfig.A_). Classes, not functions, because dynesty's Pool and checkpoints use
# plain pickle, which cannot handle functions defined in a notebook.
@dataclass(frozen=True)
class ConstantBandAlbedo:
    """Fixed band-integrated albedo, independent of A_B."""
    A: float = DEFAULT_A
    def __call__(self, A_B): return self.A
    @property
    def label(self): return f"fixedA{self.A:g}"      # short tag for filenames and plot legends, e.g. 'fixedA0.1'


@dataclass(frozen=True)
class ScaledBandAlbedo:
    """A = scale * A_B; scale=1 sets the band-integrated albedo equal to the Bond albedo."""
    scale: float = 1.0
    def __call__(self, A_B): return self.scale * A_B
    @property
    def label(self): return f"scaledA{self.scale:g}"   # short tag for filenames and plot legends, e.g. 'scaledA1'


@dataclass
class FittingConfig:
    """The setup of one modelled light curve: the planet, the band, the albedo and emissivities, the noise and the observing plan. model_pcurve
    evaluates it, and the fits and retrievals take it as their configuration.

    The band (filter_path, l1, l2) is used as given. bias.detectors.instrument_cfg builds a configuration from a planet and a band name, and
    with_planet moves one to another planet; both give the planet its own part of the band where its host saturates the rest
    (bands.filter_lambda)."""
    planet: object          # PlanetParams: geometry, orbit and host star
    filter_path: str = None   # throughput file; None with l1 = l2 = nan: a band-less template for instrument_cfg
    l1: float = np.nan      # band blue edge [m]
    l2: float = np.nan      # band red edge [m]
    A_: object = ConstantBandAlbedo()   # band albedo A as a function of the Bond albedo A_B (used when reflected=True)
    emissivity_bol: float = DEFAULT_EMISSIVITY_BOL   # bolometric emissivity: sets the temperature map with A_B
    emissivity: float = DEFAULT_EMISSIVITY       # band emissivity: scales the thermal planet flux
    sigma_1min: float = 323.  # noise [ppm/min]; a placeholder, the analyses pass noise.rms_1min
    n_orbits: int = DEFAULT_N_ORBITS  # orbits observed and phase-folded
    n_bins: int = DEFAULT_N_BINS  # bins per period; the width is derived, so they tile it exactly
    u_bol: float = DEFAULT_U_BOL   # bolometric linear limb darkening of the irradiance (0 = uniform disk)
    u: tuple = DEFAULT_U       # quadratic limb darkening (u1, u2) of the transit
    reflected: bool = True  # False: thermal only
    transit: bool = True    # include the transit (batman)
    stellar_model: str = STELLAR_MODEL   # stellar SED: "newera", an stsynphot grid name or "blackbody"
    stellar_grid: object = field(init=False, repr=False, default=None)   # derived from planet and band in __post_init__, so replace() rebuilds it

    def __post_init__(self):
        if np.isfinite(self.l1) and np.isfinite(self.l2): self.stellar_grid = stellar_spidergrid((self.l1, self.l2), self.filter_path, self.planet.logg, self.planet.feh, stellar_model=self.stellar_model)

    @property
    def sigma_bin(self):
        """Noise per phase bin [ppm]."""
        minutes_binning = self.planet.period * 24 * 60 / self.n_bins
        return self.sigma_1min / np.sqrt(minutes_binning * self.n_orbits)

    @property
    def sigma_amp(self):
        """Noise of a fitted phase-curve semi-amplitude (first harmonic over n_orbits): sigma_1min sqrt(2 / t_obs[min]), i.e. sqrt(2/N) times the
        noise per point (Montgomery & O'Donoghue 1999, DSSN 13, 28; Kallinger et al. 2008, A&A 481, 571, eq. 1). Independent of the binning,
        unlike sigma_bin, so it is the reference for a flux level such as the nightside signal at mid-transit. A lower bound: it ignores the
        correlation with A_B."""
        return self.sigma_1min * np.sqrt(2. / (self.planet.period * 24 * 60 * self.n_orbits))


def with_planet(cfg, planet, **changes):
    """cfg (or each detector of a detector set) moved to another planet: dataclasses.replace(cfg, planet=planet, **changes), with the band edges
    taken for the new planet. On an instrument's standard band (bands.observed_band) that is the planet's own part of the band where its host
    saturates the rest (inputs/noise/partial_bands.csv), else the band itself; any other (l1, l2), such as a top-hat, is kept."""
    if isinstance(cfg, (list, tuple)): return [with_planet(c, planet, **changes) for c in cfg]
    c = replace(cfg, planet=planet, **changes)
    l1, l2 = observed_band(c.filter_path, c.l1, c.l2, planet)
    return c if (l1, l2) == (c.l1, c.l2) else replace(c, l1=l1, l2=l2)


def swept_planet(cfg, a_rs=None, T_s=None, rp_rs=None):
    """The PlanetParams that model_pcurve evaluates: cfg.planet at the swept a/R*, Rp/R* and T_eff, in its own units (a_AU scales with a/R*, the
    star's size is fixed). Shared with insolation_cache, so a cached map always has the model's geometry. A T_s sweep keeps the log g and [Fe/H]
    of cfg.planet, whose star cfg.stellar_grid belongs to, so the swept stars are hypothetical."""
    pl = cfg.planet
    a_rs0 = pl.R / pl.r2  # default a/Rs (for the a_AU scaling)
    a_rs = a_rs0 if a_rs is None else a_rs
    T_s = pl.T2_eff if T_s is None else T_s
    rp_rs = pl.r1 / pl.r2 if rp_rs is None else rp_rs
    check_teff_nodes(cfg.stellar_grid, T_s)
    return replace(pl, r1=rp_rs * pl.r2, R=a_rs * pl.r2, T2_eff=T_s, a_AU=pl.a_AU * a_rs / a_rs0)


def insolation_cache(cfg, insolation_model, n_lo=DEFAULT_N_LO, n_la=DEFAULT_N_LA, **geom):
    """Ready-made insol argument of model_pcurve, built at exactly the geometry model_pcurve uses. Valid only while a/r2, r1/r2, T2_eff, cfg.u_bol,
    n_lo and n_la stay fixed (retrieval.likelihood.GaussianLogL drops it when one of them is sampled)."""
    planet = swept_planet(cfg, **geom)   # **geom: whatever swept_planet takes
    return insolation_grid(planet, u_bol=cfg.u_bol, n_lo=n_lo, n_la=n_la, insolation_model=insolation_model)


def model_times(period, n_t):
    """Midpoints of n_t equal sub-intervals of one period. bias.fits.binned averages the samples per bin; left edges would make that a first-order
    Riemann sum, biased by ~(step/2) x slope where the transit ingress and egress fall (several ppm at 50 samples per bin)."""
    return np.linspace(0., period, int(n_t), endpoint=False) + 0.5 * period / int(n_t)


def reflected_unit(cfg, insolation_model, a_rs=None, T_s=None, rp_rs=None, n_t=None, **_):
    """Reflected planet flux [ppm] at unit band albedo (A = 1) on model_times, without the transit: the reflected light is linear in A, so
    model_pcurve scales it. a_rs, T_s, rp_rs as in swept_planet. **_ takes the other model_pcurve keywords, which do not affect the reflected
    light."""
    pl = cfg.planet
    a_rs = pl.R / pl.r2  if a_rs is None else a_rs   # default a/Rs from PlanetParams
    T_s = pl.T2_eff      if T_s is None else T_s
    rp_rs = pl.r1 / pl.r2  if rp_rs is None else rp_rs      # default Rp/Rs from PlanetParams
    n_t = 50 * cfg.n_bins if n_t is None else n_t

    t_ = model_times(pl.period, n_t)
    # a_AU is left as is: it enters only the returned times, which are not used
    planet = replace(pl, r1=rp_rs * pl.r2, R=a_rs * pl.r2, T2_eff=T_s)
    _, refl, _ = reflected_PhaseCurve(planet, t=t_, include_transit=False, A=1., u_bol=cfg.u_bol, insolation_model=insolation_model)
    return refl * 1e6


def pcurve_params(cfg, insolation_model, A_B_, F=None, a_rs=None, T_s=None, rp_rs=None, n_layers=None, n_lo=DEFAULT_N_LO, n_la=DEFAULT_N_LA, insol=None):
    """The sp.ModelParams that model_pcurve feeds to SPIDERMAN, factored out so that occultation_depth can be taken at the fitted geometry.
    n_layers: default as in thermal_PhaseCurve; insol: see custom_temperature_grid; the other parameters as in model_pcurve."""
    check_redistribution(insolation_model, F)
    if n_layers is None:
        if insolation_model in EXTENDED_SOURCE_MODELS:
            n_layers = 40
        else:
            n_layers = 60

    planet = swept_planet(cfg, a_rs=a_rs, T_s=T_s, rp_rs=rp_rs)

    # "naive_redist": the Louden model (see phasecurve.Louden_PhaseCurve)
    if insolation_model == "naive_redist":
        p = louden_params(planet, (cfg.l1, cfg.l2), F, t0=0., filter_path=cfg.filter_path, A_B=A_B_, emissivity_bol=cfg.emissivity_bol, N_LAYERS=n_layers)
        p.eclipse = True
        return p

    lo, la, Tg = custom_temperature_grid(planet, u_bol=cfg.u_bol, n_lo=n_lo, n_la=n_la, insolation_model=insolation_model, F=F, A_B=A_B_, emissivity_bol=cfg.emissivity_bol, insol=insol)
    LO, LA = np.meshgrid(lo, la, indexing="ij")
    p = sp.ModelParams(brightness_model="direct_T")   # the stellar SED enters only through cfg.stellar_grid
    p.n_layers = n_layers
    p.t0, p.per = 0., planet.period
    p.a_abs = planet.a_AU
    p.inc, p.ecc, p.w = planet.incl, planet.ecc, planet.w
    p.rp, p.a = planet.r1 / planet.r2, planet.R / planet.r2
    p.p_u1 = p.p_u2 = 0.
    p.T_s, p.l1, p.l2 = planet.T2_eff, cfg.l1, cfg.l2
    p.grid = sp.format_grid(LO, LA, Tg.T)   # no temperature floor, see thermal_PhaseCurve
    p.filter, p.eclipse = cfg.filter_path, True
    return p


def model_pcurve(cfg, insolation_model, A_B_, F=None, a_rs=None, T_s=None, rp_rs=None, n_t=None, n_layers=None, n_lo=DEFAULT_N_LO, n_la=DEFAULT_N_LA, refl_unit=None, insol=None, spman_kw=None):
    """The light curve of cfg that the fits and retrievals evaluate: thermal (SPIDERMAN through fast_lightcurve), reflected (A = cfg.A_(A_B_) times
    reflected_unit) and the transit, on model_times(period, n_t) (default 50 samples per bin).

    A_B_: Bond albedo. F: heat redistribution of the redistribution models. a_rs, T_s, rp_rs: geometry and stellar T_eff instead of the planet's.
    refl_unit, insol: precomputed geometry-only pieces (reflected_unit, insolation_cache), a pure speed-up. spman_kw: options of fast_lightcurve.

    Returns (t, flux): times [days] over one orbit from mid-transit, and the flux [ppm] relative to the star alone (0 when the planet is hidden)."""
    check_redistribution(insolation_model, F)
    pl = cfg.planet
    n_t = 50 * cfg.n_bins if n_t is None else n_t

    t_ = model_times(pl.period, n_t)
    planet = swept_planet(cfg, a_rs=a_rs, T_s=T_s, rp_rs=rp_rs)

    # Thermal Part
    p = pcurve_params(cfg, insolation_model, A_B_, F=F, a_rs=a_rs, T_s=T_s, rp_rs=rp_rs, n_layers=n_layers, n_lo=n_lo, n_la=n_la, insol=insol)
    thermal = cfg.emissivity * (fast_lightcurve(p, cfg.stellar_grid, t_, **{"n_phase": 24, **(spman_kw or {})}) - 1) * 1e6

    # Transit. Pure geometry: independent of insolation_model and A_B, so it cancels exactly in a truth - model residual.
    f_transit = (flux_transit(planet, t_, t0=0., u=cfg.u) - 1.) * 1e6 if cfg.transit else 0.

    if not cfg.reflected:
        return t_, thermal + f_transit

    # Reflected Part (linear in A)
    A = cfg.A_(A_B_)
    if refl_unit is None:
        refl_unit = reflected_unit(cfg, insolation_model, a_rs=planet.R / planet.r2, T_s=planet.T2_eff, rp_rs=planet.r1 / planet.r2, n_t=n_t)

    return t_, thermal + A * refl_unit + f_transit
