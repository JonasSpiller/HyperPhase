"""Configurations of one band or of several detectors fitted jointly (a detector set), built from instrument names."""
from dataclasses import replace

import numpy as np

from hyperphase.bands import band_of, filter_lambda, filter_loc, resolve_instrument
from hyperphase.noise import NoNoiseLevelError, noise_levels


############################################################################################################################################
#####   Detector sets
############################################################################################################################################

# A detector set: several FittingConfigs fitted JOINTLY with one shared (A_B, F) and a separate baseline per detector (e.g. the two NIRSpec/G395H
# arrays). Wherever a fit takes `cfg`, a list/tuple of configs means such a set; `truth` is then a list of one truth curve per detector. The configs
# must share the geometry (planet, n_bins, u_bol, A_) and differ only in band (stellar_grid, filter_path, l1, l2) and noise. The per-detector baseline
# is the free_offset of each detector's own residual block, which is what the Fisher matrix marginalises too.
def is_detector_set(cfg):
    return isinstance(cfg, (list, tuple))


# The config that carries the instrument-independent settings of a cfg or detector set (planet, albedo model, n_bins, n_orbits, limb darkening,
# placeholder noise): the set's first detector.
def template_cfg(cfg):
    return cfg[0] if is_detector_set(cfg) else cfg


# The band (FILTER_LAMBDA_UM key) of each detector of a cfg or detector set, in the set's order.
def cfg_detectors(cfg):
    return [band_of(c) for c in (cfg if is_detector_set(cfg) else [cfg])]


# The one per-minute noise of a cfg, or of a detector set whose detectors share it (the placeholder noise of the maps). A critical noise read off a
# map is this value scaled by the map's dchi2, which means nothing for detectors fitted at different noise levels, so those raise.
def common_sigma_1min(cfg):
    levels = {float(c.sigma_1min) for c in (cfg if is_detector_set(cfg) else [cfg])}
    if len(levels) != 1: raise ValueError(f"the detectors have different noise levels {sorted(levels)} ppm/min: a critical noise needs one common sigma_1min (give every detector the same placeholder)")
    return levels.pop()


############################################################################################################################################
#####   Configurations from instrument names
############################################################################################################################################

# cfg rebuilt for an instrument entry: a band name gives one FittingConfig with that band (filter, l1, l2; the stellar grid follows,
# FittingConfig.__post_init__), a joint fit (the detector tuple, or a bands.JOINT_INSTRUMENTS alias such as "JWST_NIRSpec.G395H_joint") the
# detector set, one config per detector. sigma_1min: one number for every detector or {detector: ppm/min}; None keeps cfg's. Everything else (planet,
# albedo model, n_bins, n_orbits, limb darkening, ...) is kept, from the first detector if cfg is itself a detector set. A band-less FittingConfig
# (filter_path None, l1 = l2 = nan) is the natural template: FittingConfig(planet=..., ...) and then instrument_cfg(template, instrument).
def instrument_cfg(cfg, instrument, sigma_1min=None):
    c0, cfgs = template_cfg(cfg), []
    for d in detectors(instrument):
        l1, l2 = filter_lambda(d, c0.planet)   # the planet's own part of the band where its host saturates the rest (PARTIAL_BANDS)
        s = c0.sigma_1min if sigma_1min is None else (sigma_1min[d] if isinstance(sigma_1min, dict) else sigma_1min)
        cfgs.append(replace(c0, filter_path=filter_loc(d), l1=l1, l2=l2, sigma_1min=float(s)))
    return cfgs[0] if len(cfgs) == 1 else cfgs


# Several detectors fitted jointly (a detector set, see is_detector_set) are one "instrument" of instrument_retrievals: given as a tuple/list of
# FILTER_LAMBDA_UM keys or as a JOINT_INSTRUMENTS alias (resolve_instrument), carried in the tables as one key joined with "+", e.g.
# "JWST_NIRSpec.G395H_NRS1+JWST_NIRSpec.G395H_NRS2". detectors() turns any of the three spellings into the list of bands (a single instrument is its
# own one-element set) and rejects an unknown band.
JOINT_SEP = "+"


def instrument_key(instrument):
    return JOINT_SEP.join(detectors(instrument))


def detectors(instrument):
    if isinstance(instrument, str) and JOINT_SEP in instrument: instrument = tuple(instrument.split(JOINT_SEP))
    r = resolve_instrument(instrument)
    return [r] if isinstance(r, str) else list(r)


# The cfg (or detector set) of one table row: the planet's geometry with each detector's band and noise level ({detector: ppm/min}).
def _detector_cfgs(cfg, planet, instrument, sigma_1min):
    return instrument_cfg(replace(template_cfg(cfg), planet=planet), instrument, sigma_1min)


# The FittingConfig (one band) or detector set (a tuple of bands, fitted jointly) of one planet in one retrieval mode at the planet's own noise: what
# instrument_retrievals fits per row, for code that needs the cfg itself (per-bin residuals, sweeps, nuisance columns). sigma_1min: {detector:
# ppm/min}; None looks the levels up (noise.noise_levels: PandExo cache or CUSTOM_SIGMA_1MIN) and raises NoNoiseLevelError where a detector
# has none.
def retrieval_cfg(cfg, planet, instrument, sigma_1min=None):
    dets = detectors(instrument)
    if sigma_1min is None: sigma_1min = noise_levels(planet, dets)
    missing = [d for d in dets if d not in sigma_1min]
    if missing: raise NoNoiseLevelError(f"no noise level for {planet.name} on {', '.join(missing)}")
    return _detector_cfgs(cfg, planet, instrument, sigma_1min)


############################################################################################################################################
#####   Detector sets of the retrievals
############################################################################################################################################

# A detector set (is_detector_set: a list/tuple of FittingConfigs with the same planet and binning, different band and noise) is
# fitted JOINTLY - one shared (A_B, F), one baseline per detector - by JointGaussianLogL; its dataset is the per-detector simulate_data outputs under
# data["detectors"]. These helpers let every plot treat a single FittingConfig and a set alike.
def detector_cfgs(cfg):
    """A FittingConfig or a detector set -> the list of FittingConfigs."""
    return list(cfg) if is_detector_set(cfg) else [cfg]


def _blocks(sizes):
    """Index slices of consecutive blocks of the given sizes in a concatenated array (the detectors' blocks of a joint model or residual vector)."""
    edges = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    return [slice(int(edges[k]), int(edges[k + 1])) for k in range(len(sizes))]
