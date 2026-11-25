"""The model of the fits: FittingConfig (planet, band, albedo, emissivities, noise and observing plan) and model_pcurve, the binned
phase curve of a configuration."""
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


# Band-integrated albedo A in the observing band as a function of the Bond albedo A_B — i.e. the FittingConfig.A_ callable. Classes rather than a
# `def` in a notebook cell because dynesty.pool.Pool spawns its workers and dynesty checkpoints with plain pickle which cannot handle notebook
# functions.
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
    evaluates it, and the fits and retrievals take it as their configuration. stellar_grid is derived from the planet and the band.

    The band (filter_path, l1, l2) is used as given. To build a configuration from a planet and a band name use bias.detectors.instrument_cfg, and
    to move one to another planet with_planet: both give the planet its own part of the band where its host saturates the rest
    (bands.filter_lambda; inputs/noise/partial_bands.csv)."""
    planet: object          # PlanetParams for the reference system (e.g. K2-141b); single source of
                            # truth for geometry (r1, r2, R, a_AU), orbit (period, incl, ecc, w) and the star (T2_eff, logg, feh)
    filter_path: str = None   # SPIDERMAN throughput file. Left out (with l1, l2) it is a band-less template, which bias.detectors.instrument_cfg completes with the band(s) of an instrument
    l1: float = np.nan      # filter blue edge [m]
    l2: float = np.nan      # filter red edge [m]
    A_: object = ConstantBandAlbedo()   # band-integrated albedo A as a function of the Bond albedo A_B (only needed when reflected=True)
    emissivity_bol: float = DEFAULT_EMISSIVITY_BOL   # bolometric emissivity: sets the temperature map with A_B (exactly degenerate with it)
    emissivity: float = DEFAULT_EMISSIVITY       # band-integrated emissivity: scales the thermal planet flux
    sigma_1min: float = 323.  # noise [ppm/min]; a placeholder for configs built without one (the analyses pass noise.rms_1min)
    n_orbits: int = DEFAULT_N_ORBITS  # number of orbits observed and phase-folded
    n_bins: int = DEFAULT_N_BINS  # bins per period; the width is DERIVED so it tiles exactly
    u_bol: float = DEFAULT_U_BOL   # bolometric linear limb darkening of the irradiance (0 = uniform disk)
    u: tuple = DEFAULT_U       # quadratic stellar limb darkening [u1, u2] of the transit
    reflected: bool = True  # True -> full (reflected + thermal); False -> thermal only
    transit: bool = True    # include the primary transit (batman)
    stellar_model: str = STELLAR_MODEL   # stellar SED of stellar_grid (and of retrieval.sampler.brightness_temperature): "newera", an stsynphot grid name or "blackbody"
    stellar_grid: object = field(init=False, repr=False, default=None)   # DERIVED from planet (logg, feh) and the band in __post_init__, so every replace(cfg, planet=..., l1=..., ...) rebuilds it (cached in stellar_spidergrid) and it can never belong to another star or band

    def __post_init__(self):
        if np.isfinite(self.l1) and np.isfinite(self.l2): self.stellar_grid = stellar_spidergrid((self.l1, self.l2), self.filter_path, self.planet.logg, self.planet.feh, stellar_model=self.stellar_model)   # l1 = l2 = nan: a band-less template cfg, instrument_cfg fills the band

    @property
    def sigma_bin(self):
        minutes_binning = self.planet.period * 24 * 60 / self.n_bins
        return self.sigma_1min / np.sqrt(minutes_binning * self.n_orbits)

    # Noise of a fitted phase-curve semi-amplitude (first harmonic over n_orbits): sigma_1min sqrt(2 / t_obs[min]), i.e. sigma(a) = sqrt(2/N) sigma
    # per point (Montgomery & O'Donoghue 1999, DSSN 13, 28; Kallinger et al. 2008, A&A 481, 571, eq. 1). Binning-independent, so it is the reference
    # for a flux level such as the nightside signal at mid-transit (sigma_bin is not: it scales with sqrt(n_bins)). A lower bound: it ignores the
    # compensation by A_B that dchi2_F / z_F includes.
    @property
    def sigma_amp(self):
        return self.sigma_1min * np.sqrt(2. / (self.planet.period * 24 * 60 * self.n_orbits))


def with_planet(cfg, planet, **changes):
    """cfg (or each detector of a detector set) moved to another planet: dataclasses.replace(cfg, planet=planet, **changes), with the band edges
    taken for the new planet. On an instrument's standard band (bands.observed_band) that is the planet's own part of the band where its host
    saturates the rest (inputs/noise/partial_bands.csv; 55 Cnc e on MIRI/LRS), else the band itself; any other (l1, l2), such as a top-hat, is kept.
    The stellar grid follows the new star and band."""
    if isinstance(cfg, (list, tuple)): return [with_planet(c, planet, **changes) for c in cfg]
    c = replace(cfg, planet=planet, **changes)
    l1, l2 = observed_band(c.filter_path, c.l1, c.l2, planet)
    return c if (l1, l2) == (c.l1, c.l2) else replace(c, l1=l1, l2=l2)


# The PlanetParams that model_pcurve actually evaluates, at the swept geometry, in pl's own units. Factored out so that a cached insolation map
# (insolation_cache) is guaranteed to be built at exactly the geometry the corresponding model_pcurve call will use — the caches cannot silently go
# stale. A T_s sweep changes T_eff only: log g and [Fe/H] stay those of cfg.planet (cfg.stellar_grid belongs to that star), so the swept stars are
# hypothetical. A real M dwarf has log g ~ 5 rather than the reference star's 4.5, which changes its mid-IR flux by ~2 % at 3000 K and ~6 % at 2600 K
# (LRS); bias maps are insensitive to it (the bias does not depend on k), their significances are not.
def swept_planet(cfg, a_rs=None, T_s=None, rp_rs=None):
    pl = cfg.planet
    a_rs0 = pl.R / pl.r2  # default a/Rs (for the a_AU scaling)
    a_rs = a_rs0 if a_rs is None else a_rs
    T_s = pl.T2_eff if T_s is None else T_s
    rp_rs = pl.r1 / pl.r2 if rp_rs is None else rp_rs
    check_teff_nodes(cfg.stellar_grid, T_s)
    # rebuild the planet at the swept geometry in pl's own units; a_AU scales with a/Rs (stellar size fixed)
    return replace(pl, r1=rp_rs * pl.r2, R=a_rs * pl.r2, T2_eff=T_s, a_AU=pl.a_AU * a_rs / a_rs0)


# Ready-made `insol` argument for model_pcurve, built at exactly the geometry model_pcurve would use. ONLY valid while a/r2, r1/r2, T2_eff, cfg.u_bol,
# n_lo and n_la are held fixed — see retrieval.likelihood.GaussianLogL, which builds this and invalidates it automatically when the corresponding parameter
# is sampled.
def insolation_cache(cfg, insolation_model, n_lo=DEFAULT_N_LO, n_la=DEFAULT_N_LA, **geom):
    planet = swept_planet(cfg, **geom)   # **geom rather than named args, so adding a parameter to swept_planet is enough
    return insolation_grid(planet, u_bol=cfg.u_bol, n_lo=n_lo, n_la=n_la, insolation_model=insolation_model)


# Uniform time grid for a model that is going to be BIN-AVERAGED: the midpoints of the n_t sub-intervals, not their left edges.
# binned() averages these samples per bin, so left edges would make that a left Riemann sum - first order, with error
# ~ (step/2) x slope, i.e. ~ transit_depth / (2 * n_t/n_bins) in the bins the transit ingress/egress falls into. At the usual
# n_t = 50 per bin that is several ppm (measured 4.62 ppm for a 454 ppm transit), orders of magnitude above every other
# discretisation in the forward model (n_layers 40 vs 80: 0.05 ppm; n_lo/n_la 300/150 vs 600/300: 0.001 ppm). The midpoint
# rule removes it for free: same number of evaluations, ~70x smaller error, and it keeps converging with n_t.
def model_times(period, n_t):
    return np.linspace(0., period, int(n_t), endpoint=False) + 0.5 * period / int(n_t)


def reflected_unit(cfg, insolation_model, a_rs=None, T_s=None, rp_rs=None, n_t=None, **_):
    """Reflected planet flux [ppm] at unit albedo (A = 1) on model_times, without the transit; the reflected light is linear in A, so model_pcurve
    scales it. a_rs, T_s, rp_rs replace the planet's geometry and T_eff. **_ takes the other model_pcurve keywords (n_lo, n_layers, spman_kw, ...),
    which the fits pass to every part of the model and which do not affect the reflected light."""
    pl = cfg.planet
    a_rs = pl.R / pl.r2  if a_rs is None else a_rs   # default a/Rs from PlanetParams
    T_s = pl.T2_eff      if T_s is None else T_s
    rp_rs = pl.r1 / pl.r2  if rp_rs is None else rp_rs      # default Rp/Rs from PlanetParams
    n_t = 50 * cfg.n_bins if n_t is None else n_t

    t_ = model_times(pl.period, n_t)
    # set the ratios in the planet's own units (reflected is geometric -> a_AU unused, left as-is)
    planet = replace(pl, r1=rp_rs * pl.r2, R=a_rs * pl.r2, T2_eff=T_s)
    _, refl, _ = reflected_PhaseCurve(planet, t=t_, include_transit=False, A=1., u_bol=cfg.u_bol, insolation_model=insolation_model)
    return refl * 1e6


# The sp.ModelParams that model_pcurve feeds to SPIDERMAN, factored out exactly like swept_planet so the OCCULTATION DEPTH can be taken at the fitted
# geometry with the existing occultation_depth(p, stellar_grid=cfg.stellar_grid) instead of duplicating this setup. F: heat redistribution, required
# by insolation_model="naive_redist"
def pcurve_params(cfg, insolation_model, A_B_, F=None, a_rs=None, T_s=None, rp_rs=None, n_layers=None, n_lo=DEFAULT_N_LO, n_la=DEFAULT_N_LA, insol=None):
    check_redistribution(insolation_model, F)
    # n_layers = 30-40 usually sufficient for insolation_model="hyperillum", insolation_model="naive" (and "naive_redist") needs n_layers = 60-80
    # Measured (TOI-2431 b/NIRSpec and K2-141 b/MIRI-LRS, vs n_layers=400): on the light CURVE the two models converge alike
    # (max |d| at n_layers=40: naive 0.09-0.17 ppm, hyperillum 0.08-0.14 ppm; <0.02 sigma_bin once binned), and the naive-vs-truth
    # A_B bias is stable to ~1e-4 from n_layers=20 up. The 60-80 matters for the OCCULTATION DEPTH, where naive converges only
    # ~N^-1 against hyperillum's ~N^-2 (K2-141 b: 0.33 / 0.20 / 0.08 ppm at n_layers 40 / 60 / 120, vs 0.13 / 0.07 / 0.02).
    # Since the hyperillum-naive depth difference is itself only ~1 ppm, use n_layers >= 120 when COMPARING depths between models.
    if n_layers is None:
        if insolation_model in EXTENDED_SOURCE_MODELS:
            n_layers = 40
        else:
            n_layers = 60

    planet = swept_planet(cfg, a_rs=a_rs, T_s=T_s, rp_rs=rp_rs)

    # "naive_redist": SPIDERMAN's analytic Louden map, as in thermal_PhaseCurve (insol, n_lo and n_la are unused)
    if insolation_model == "naive_redist":
        p = louden_params(planet, (cfg.l1, cfg.l2), F, t0=0., filter_path=cfg.filter_path, A_B=A_B_, emissivity_bol=cfg.emissivity_bol, N_LAYERS=n_layers)
        p.eclipse = True
        return p

    # insol=... reuses the A_B-independent insolation map, leaving only the exact T = T(A_B=0, emissivity_bol=1)*((1-A_B)/emissivity_bol)^(1/4)
    # rescale (valid because insol depends on geometry alone).
    lo, la, Tg = custom_temperature_grid(planet, u_bol=cfg.u_bol, n_lo=n_lo, n_la=n_la, insolation_model=insolation_model, F=F, A_B=A_B_, emissivity_bol=cfg.emissivity_bol, insol=insol)   # F: used by "hyperillum_redist" only
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

    A_B_: Bond albedo. F: heat redistribution, required by the redistribution models (the thermal part of "naive_redist" is the Louden model, its
    reflected part that of "naive"). a_rs, T_s, rp_rs: geometry and stellar T_eff instead of the planet's. refl_unit, insol: precomputed
    geometry-only pieces (reflected_unit, insolation_cache), a pure speed-up: passing them is bit-identical to rebuilding them here.

    Returns (t, flux): times [days] over one orbit from mid-transit, and the flux [ppm] relative to the star alone (0 when the planet is hidden)."""
    check_redistribution(insolation_model, F)
    pl = cfg.planet
    n_t = 50 * cfg.n_bins if n_t is None else n_t

    t_ = model_times(pl.period, n_t)
    planet = swept_planet(cfg, a_rs=a_rs, T_s=T_s, rp_rs=rp_rs)

    # Thermal Part
    p = pcurve_params(cfg, insolation_model, A_B_, F=F, a_rs=a_rs, T_s=T_s, rp_rs=rp_rs, n_layers=n_layers, n_lo=n_lo, n_la=n_la, insol=insol)
    thermal = cfg.emissivity * (fast_lightcurve(p, cfg.stellar_grid, t_, **{"n_phase": 24, **(spman_kw or {})}) - 1) * 1e6  # SPIDERMAN emits as a blackbody, the band emissivity scales it. spman_kw overrides n_phase / pad / split_totality; it reaches here through **model_kw / **kw in the fitting loop

    # Transit. Pure geometry: independent of insolation_model and A_B, so it cancels exactly in a truth - model residual.
    f_transit = (flux_transit(planet, t_, t0=0., u=cfg.u) - 1.) * 1e6 if cfg.transit else 0.

    if not cfg.reflected:
        return t_, thermal + f_transit

    # Reflected Part (linear in A)
    A = cfg.A_(A_B_)
    if refl_unit is None:
        refl_unit = reflected_unit(cfg, insolation_model, a_rs=planet.R / planet.r2, T_s=planet.T2_eff, rp_rs=planet.r1 / planet.r2, n_t=n_t)

    return t_, thermal + A * refl_unit + f_transit
