"""Configurations of one band or of a detector set, built from instrument names.

A detector set is a list or tuple of FittingConfigs fitted jointly: one shared (A_B, F) and one baseline per detector (e.g. the two NIRSpec/G395H
arrays). Wherever a fit takes `cfg`, a set may stand in, and `truth` is then one curve per detector. The configs share the geometry (planet,
n_bins, u_bol, A_) and differ only in band and noise. Each detector's baseline is the free offset of its own residual block, which the Fisher
matrix marginalises too."""
from dataclasses import replace

import numpy as np

from hyperphase.bands import band_of, filter_lambda, filter_loc, resolve_instrument
from hyperphase.noise import NoNoiseLevelError, noise_levels


############################################################################################################################################
#####   Detector sets
############################################################################################################################################

def is_detector_set(cfg):
    """True for a detector set (a list or tuple of FittingConfigs)."""
    return isinstance(cfg, (list, tuple))


def template_cfg(cfg):
    """The config that holds the instrument-independent settings (planet, albedo model, n_bins, n_orbits, limb darkening, placeholder noise): cfg
    itself, or a set's first detector."""
    return cfg[0] if is_detector_set(cfg) else cfg


def cfg_detectors(cfg):
    """The band (FILTER_LAMBDA_UM key) of each detector, in the set's order."""
    return [band_of(c) for c in (cfg if is_detector_set(cfg) else [cfg])]


def common_sigma_1min(cfg):
    """The one per-minute noise of a cfg, or of a detector set whose detectors share it (the placeholder noise of the maps)."""
    levels = {float(c.sigma_1min) for c in (cfg if is_detector_set(cfg) else [cfg])}
    if len(levels) != 1: raise ValueError(f"the detectors have different noise levels {sorted(levels)} ppm/min: a critical noise needs one common sigma_1min (give every detector the same placeholder)")
    return levels.pop()


############################################################################################################################################
#####   Configurations from instrument names
############################################################################################################################################

def instrument_cfg(cfg, instrument, sigma_1min=None):
    """cfg rebuilt for an instrument entry: a band name gives one FittingConfig with that band (the stellar grid follows), a joint fit (detector
    tuple or bands.JOINT_INSTRUMENTS alias) a detector set. sigma_1min: one number for every detector, or {detector: ppm/min}; None keeps cfg's.
    All other settings come from template_cfg(cfg), so a band-less FittingConfig(planet=..., ...) is the natural template."""
    c0, cfgs = template_cfg(cfg), []
    for d in detectors(instrument):
        l1, l2 = filter_lambda(d, c0.planet)   # the planet's partial band, if any (bands.PARTIAL_BANDS)
        s = c0.sigma_1min if sigma_1min is None else (sigma_1min[d] if isinstance(sigma_1min, dict) else sigma_1min)
        cfgs.append(replace(c0, filter_path=filter_loc(d), l1=l1, l2=l2, sigma_1min=float(s)))
    return cfgs[0] if len(cfgs) == 1 else cfgs


# Separator of a joint fit's detectors in table keys, e.g. "JWST_NIRSpec.G395H_NRS1+JWST_NIRSpec.G395H_NRS2".
JOINT_SEP = "+"


def instrument_key(instrument):
    """The table key of an instrument: its detectors joined by JOINT_SEP."""
    return JOINT_SEP.join(detectors(instrument))


def detectors(instrument):
    """The bands of an instrument given as a band name, a detector tuple, a JOINT_INSTRUMENTS alias or a JOINT_SEP-joined key (a single band is its
    own one-element set). Unknown bands raise."""
    if isinstance(instrument, str) and JOINT_SEP in instrument: instrument = tuple(instrument.split(JOINT_SEP))
    r = resolve_instrument(instrument)
    return [r] if isinstance(r, str) else list(r)


def _detector_cfgs(cfg, planet, instrument, sigma_1min):
    """The cfg or detector set of one table row: the planet's geometry with each detector's band and noise level ({detector: ppm/min})."""
    return instrument_cfg(replace(template_cfg(cfg), planet=planet), instrument, sigma_1min)


def retrieval_cfg(cfg, planet, instrument, sigma_1min=None):
    """The cfg or detector set of one planet in one retrieval mode at the planet's own noise, as instrument_retrievals fits it. sigma_1min:
    {detector: ppm/min}; None looks the levels up (noise.noise_levels)."""
    dets = detectors(instrument)
    if sigma_1min is None: sigma_1min = noise_levels(planet, dets)
    missing = [d for d in dets if d not in sigma_1min]
    if missing: raise NoNoiseLevelError(f"no noise level for {planet.name} on {', '.join(missing)}")
    return _detector_cfgs(cfg, planet, instrument, sigma_1min)


############################################################################################################################################
#####   Detector sets of the retrievals
############################################################################################################################################

def detector_cfgs(cfg):
    """A FittingConfig or a detector set -> the list of FittingConfigs."""
    return list(cfg) if is_detector_set(cfg) else [cfg]


def _blocks(sizes):
    """Index slices of consecutive blocks of the given sizes in a concatenated array (the detectors' blocks of a joint model or residual vector)."""
    edges = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    return [slice(int(edges[k]), int(edges[k + 1])) for k in range(len(sizes))]
