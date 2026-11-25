"""Observing bands: wavelength ranges, throughput files and PandExo modes (inputs/filters/bands.csv), the joint detector fits, and the
bands that a planet can be observed in only in part (inputs/noise/partial_bands.csv).

register_band adds a band for the session. Every forward-model function also takes the band directly (a wavelength range and a throughput file)."""
import csv
import os
import re
from functools import lru_cache

import numpy as np

from hyperphase._util import _normalise
from hyperphase.paths import BANDS_TABLE, FILTER_DIR, PARTIAL_BANDS_TABLE


############################################################################################################################################
#####   Throughput files
############################################################################################################################################
@lru_cache(maxsize=None)
def _read_filter_file(filter_path):
    d = np.loadtxt(filter_path)
    o = np.argsort(d[:, 0])
    return d[o, 0], d[o, 1]


def load_filter(lambda_range, filter_path=None, n=300):
    """The throughput T(lambda) of a band within lambda_range [m]: (wavelengths [m], throughput); a top-hat on n points without filter_path."""
    l1, l2 = float(lambda_range[0]), float(lambda_range[1])
    if filter_path is None:
        lam = np.linspace(l1, l2, n)
        return lam, np.ones_like(lam)
    lam, w = _read_filter_file(str(filter_path))
    mask = (lam >= l1) & (lam <= l2)
    # the files are plain throughputs T(lambda) [electrons per incident photon], the SVO / Pandeia convention; the photon weighting lambda/(hc) is
    # applied by the consumer (stellar.band_avg_planck, get_filter in fspiderman)
    return lam[mask], w[mask]


############################################################################################################################################
#####   The band table
############################################################################################################################################

# One row per band of inputs/filters/bands.csv, keyed by its name ("MIRI/LRS", "JWST_NIRSpec.G395H_NRS1", ...):
# FILTER_LAMBDA_UM: the wavelength range [um] used for observations, i.e. integrated over; FULL_LAMBDA_UM: the extent of the throughput file;
# FILTER_FILES: the throughput file (a name in paths.FILTER_DIR, or an absolute path); PANDEXO_MODES: the PandExo mode (jdi.load_mode_dict), None where
# PandExo has none (noise.rms_1min then takes the noise level from inputs/noise/custom_sigma_1min.csv). register_band adds rows for the session.
FILTER_LAMBDA_UM, FULL_LAMBDA_UM, FILTER_FILES, PANDEXO_MODES = {}, {}, {}, {}
FILTER_LAMBDA_M, FULL_LAMBDA_M = {}, {}   # the same ranges in m, as arrays
_INSTRUMENT_OF_FILE = {}                  # real path of a throughput file -> band name


def register_band(name, l1, l2, filter_path, full_range=None, pandexo_mode=None, overwrite=False):
    """Add a band for this session. l1, l2: the range [m] integrated over; filter_path: the throughput file, T(lambda) in electrons per incident photon
    with wavelengths in m (an absolute path, or a file name in paths.FILTER_DIR); full_range [m]: the extent of the file (default (l1, l2));
    pandexo_mode: the PandExo mode name, None if PandExo has none. The band is then known to filter_lambda, filter_loc, resolve_instrument and
    noise.rms_1min like the bands of inputs/filters/bands.csv."""
    if name in FILTER_LAMBDA_UM and not overwrite: raise KeyError(f"band {name!r} exists; pass overwrite=True to replace it")
    if not l1 < l2: raise ValueError(f"band {name!r}: l1 = {l1} must be below l2 = {l2} [m]")
    full = (l1, l2) if full_range is None else full_range
    _add_band(name, (1e6 * l1, 1e6 * l2), (1e6 * full[0], 1e6 * full[1]), str(filter_path), pandexo_mode or None)


def _add_band(name, range_um, full_um, file, pandexo_mode):
    FILTER_LAMBDA_UM[name] = [float(range_um[0]), float(range_um[1])]
    FULL_LAMBDA_UM[name] = [float(full_um[0]), float(full_um[1])]
    FILTER_FILES[name] = file
    PANDEXO_MODES[name] = pandexo_mode
    FILTER_LAMBDA_M[name] = 1e-6 * np.array(FILTER_LAMBDA_UM[name])
    FULL_LAMBDA_M[name] = 1e-6 * np.array(FULL_LAMBDA_UM[name])
    _INSTRUMENT_OF_FILE.setdefault(os.path.realpath(str(FILTER_DIR / file)), name)   # a file shared by several bands belongs to the first


with open(BANDS_TABLE, newline="") as _f:
    for _r in csv.DictReader(_f):
        _add_band(_r["band"], (float(_r["l1_um"]), float(_r["l2_um"])), (float(_r["full_l1_um"]), float(_r["full_l2_um"])), _r["file"], _r["pandexo_mode"] or None)


def filter_lambda(instrument, planet=None):
    """The wavelength range [m] used for observations as a 2-element array: the band's FILTER_LAMBDA_UM range or, for a planet (PlanetParams or
    name) with a PARTIAL_BANDS entry on this instrument, the part of it that planet can be observed in (its host saturates the rest)."""
    if planet is not None:
        part = PARTIAL_BANDS_N.get(_normalise(getattr(planet, "name", planet)), {}).get(instrument)
        if part is not None: return part.copy()
    return FILTER_LAMBDA_M[instrument]


def full_lambda(instrument):
    """Full wavelength range [m] of the throughput file as a 2-element array."""
    return FULL_LAMBDA_M[instrument]


def filter_loc(instrument):
    """Path [str] to the throughput file of a band (a FILTER_LAMBDA_UM key)."""
    return str(FILTER_DIR / FILTER_FILES[instrument])


def pandexo_mode(instrument):
    """The PandExo mode name (as jdi.load_mode_dict expects it) for a band key of FILTER_LAMBDA_UM."""
    mode = PANDEXO_MODES[instrument]
    if mode is None:
        raise ValueError(f"PandExo has no mode for {instrument!r}; noise.rms_1min takes such noise levels from inputs/noise/custom_sigma_1min.csv.")
    return mode


def instruments():
    """The band keys, for iterating over all of them."""
    return list(FILTER_FILES)


def band_of(cfg):
    """The band key (FILTER_LAMBDA_UM) of a FittingConfig, matched on the name of its throughput file; the file name itself where no band matches."""
    name = os.path.basename(str(cfg.filter_path))
    return next((k for k, f in FILTER_FILES.items() if os.path.basename(f) == name), name)


############################################################################################################################################
#####   Joint detector fits
############################################################################################################################################

# The JWST bands, by band centre. A list, not a tuple: in a list of fits a tuple always means one joint fit of several detectors.
JWST_BANDS = ["JWST_NIRSpec.G395H_NRS1", "JWST_NIRSpec.G395H_NRS2", "MIRI/LRS", "MIRI/F1280W", "MIRI/F1500W", "MIRI/F1800W"]
_bad = set(JWST_BANDS) - set(FILTER_LAMBDA_UM)
if _bad: raise KeyError(f"JWST_BANDS uses unknown instrument(s) {sorted(_bad)}; valid keys are {list(FILTER_LAMBDA_UM)}")

# Joint detector fits by name, for the `instruments` of retrievals/run_dynesty.py: the alias stands for the detector tuple; both spellings are accepted
# by resolve_instrument, and joint_name turns the tuple back into the alias for file names.
JOINT_INSTRUMENTS = {"JWST_NIRSpec.G395H_joint": tuple(JWST_BANDS[:2])}


def resolve_instrument(instrument):
    """An instrument entry -> a band name (str: a single-detector fit) or the tuple of band names of a joint fit (given as a tuple, or as one of the JOINT_INSTRUMENTS aliases)."""
    if isinstance(instrument, str) and instrument in JOINT_INSTRUMENTS: return JOINT_INSTRUMENTS[instrument]
    dets = tuple(instrument) if isinstance(instrument, (list, tuple)) else (instrument,)
    bad = [d for d in dets if d not in FILTER_LAMBDA_UM]
    if bad: raise KeyError(f"unknown instrument(s) {bad}; valid: {list(FILTER_LAMBDA_UM)} or a joint alias {list(JOINT_INSTRUMENTS)}")
    if len(dets) < 1 or len(set(dets)) != len(dets): raise ValueError(f"a joint fit needs distinct detectors, got {dets}")
    return dets if isinstance(instrument, (list, tuple)) else instrument


def detector_label(instrument):
    """Short label of a band for parameter names and panel titles: the part after the last '_' or '/' ("JWST_NIRSpec.G395H_NRS1" -> "NRS1", "MIRI/LRS" -> "LRS", "MIRI/F1280W" -> "F1280W")."""
    return re.split(r"[_/]", instrument)[-1]


def joint_name(detectors):
    """File-name spelling of a joint fit: its JOINT_INSTRUMENTS alias if there is one, else 'joint-' + the detector labels joined by '+'."""
    for alias, dets in JOINT_INSTRUMENTS.items():
        if tuple(dets) == tuple(detectors): return alias
    return "joint-" + "+".join(detector_label(d) for d in detectors)


############################################################################################################################################
#####   Partial bands
############################################################################################################################################

# Bands a planet can be observed in only in part (inputs/noise/partial_bands.csv): {planet name: {band: [lo, hi] um}}. The host saturates the rest of
# the band in the first group (PandExo's full_saturation_mask: unrecoverable), so the white-light curve is the sum of the clean pixels, and the
# forward model must integrate the same range as the noise is computed over. filter_lambda(instrument, planet) returns this range, and the
# name-based constructors use it: bias.detectors.instrument_cfg and model.with_planet (so the per-planet tables and retrievals/run_dynesty.py);
# noise.rms_1min bins the PandExo spectrum over it. A FittingConfig built by hand keeps the band it is given. Planet names are matched loosely; a
# planet renamed around the same host (bias.maps.moved_planet, the Roche cases) does not inherit the entry, although the saturation belongs to the
# host. A partial band changes the forward model: a stored retrieval made over another band is flagged by retrieval.storage.check_fit_current.
PARTIAL_BANDS = {}
with open(PARTIAL_BANDS_TABLE, newline="") as _f:
    for _r in csv.DictReader(_f):
        PARTIAL_BANDS.setdefault(_r["planet"], {})[_r["band"]] = [float(_r["l1_um"]), float(_r["l2_um"])]

for _pl, _d in PARTIAL_BANDS.items():
    for _ins, (_lo, _hi) in _d.items():
        if _ins not in FILTER_LAMBDA_UM: raise KeyError(f"{PARTIAL_BANDS_TABLE.name}: unknown band {_ins!r}; valid keys are {list(FILTER_LAMBDA_UM)}")
        if not FILTER_LAMBDA_UM[_ins][0] <= _lo < _hi <= FILTER_LAMBDA_UM[_ins][1]: raise ValueError(f"{PARTIAL_BANDS_TABLE.name}: {_pl!r} on {_ins!r}: {[_lo, _hi]} is not inside the band {FILTER_LAMBDA_UM[_ins]}")
PARTIAL_BANDS_N = {_normalise(k): {ins: 1e-6 * np.array(r, float) for ins, r in v.items()} for k, v in PARTIAL_BANDS.items()}


def observed_band(filter_path, l1, l2, planet):
    """The band [m] that `planet` (PlanetParams or name) is observed in, for a configuration with this throughput file and band edges: where (l1, l2)
    is a STANDARD band of the instrument of filter_path (its FILTER_LAMBDA_UM range, or any planet's PARTIAL_BANDS range of it), the planet's own
    filter_lambda(instrument, planet); any other (l1, l2) - a top-hat, a band-edge sweep - as given. Returns (l1, l2)."""
    if filter_path is None or not (np.isfinite(l1) and np.isfinite(l2)): return l1, l2
    ins = _INSTRUMENT_OF_FILE.get(os.path.realpath(str(filter_path)))
    if ins is None: return l1, l2
    standard = [FILTER_LAMBDA_M[ins]] + [d[ins] for d in PARTIAL_BANDS_N.values() if ins in d]
    if not any(np.allclose((l1, l2), b, rtol=0., atol=1e-12) for b in standard): return l1, l2
    lo, hi = filter_lambda(ins, planet)
    return float(lo), float(hi)
