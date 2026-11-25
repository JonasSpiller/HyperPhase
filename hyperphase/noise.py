"""Noise levels: PandExo runs, their cache, and the per-planet exceptions (inputs/noise/custom_sigma_1min.csv, inputs/noise/unobservable.csv)."""
import csv
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from hyperphase._util import _normalise
from hyperphase.bands import FILTER_LAMBDA_UM, PANDEXO_MODES, PARTIAL_BANDS, filter_lambda, joint_name, pandexo_mode, resolve_instrument
from hyperphase.paths import CUSTOM_SIGMA_TABLE, PANDEXO_CACHE, PANDEXO_USER_CACHE, PARTIAL_BANDS_TABLE, UNOBSERVABLE_TABLE
from hyperphase.targets import INPUT_GLOB, _archive, hi_planets_by_name


############################################################################################################################################
#####   PandExo noise levels (after a PandExo notebook of Sebastian Zieba)
############################################################################################################################################

# PandExo inputs and the aggregate columns they come from. Every one of these is required:
# a missing value raises, and the escape hatch is to pass it as a keyword override.
PANDEXO_COLUMNS = {
    "period":    "pl_orbper",   # days (not used by PandExo itself, reported for sanity)
    "rp_rearth": "pl_rade",     # Earth radii
    "rs_rsun":   "st_rad",      # solar radii
    "teff":      "st_teff",     # K
    "logg":      "st_logg",     # log surface gravity, cgs
    "metal":     "st_met",      # log Fe/H
    "t14_h":     "pl_trandur",  # transit duration T14, hours
}

# Reference wavelength of the normalisation band [micron], see exo_dict['star']['ref_wave'].
REF_WAVE_UM = {"sy_kmag": 2.22, "sy_jmag": 1.25}

# Keys pandexo_params returns, and therefore the keys accepted as overrides.
PANDEXO_PARAM_KEYS = (*PANDEXO_COLUMNS, "mag", "ref_wave")


# sigma_1min [ppm/min] from the S/N of one integration and its duration: rms_int = 1e6 / snr_int ppm, scaled to one minute by sqrt(t_int / 60 s).
def sigma_1min_from_integration(snr_int, t_int_s):
    return 1e6 / snr_int * np.sqrt(t_int_s / 60.)


def _read_planet_table(path, fields):
    """{planet name: {band: value}} of a per-planet table; value = fields(row). Unknown bands raise."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["band"] not in FILTER_LAMBDA_UM: raise KeyError(f"{path.name}: unknown band {r['band']!r}; valid keys are {list(FILTER_LAMBDA_UM)}")
            out.setdefault(r["planet"], {})[r["band"]] = fields(r)
    return out


# Noise levels sigma_1min [ppm/min] for bands PandExo has no mode for (MIRI imaging, ...), from the S/N and duration of one integration
# (inputs/noise/custom_sigma_1min.csv): {planet name: {band: sigma_1min}}. rms_1min falls back to these only where PandExo cannot supply a value.
# Planet names are matched loosely ("TOI-2431b" == "TOI-2431 b").
CUSTOM_SIGMA_1MIN = _read_planet_table(CUSTOM_SIGMA_TABLE, lambda r: sigma_1min_from_integration(float(r["snr_int"]), float(r["t_int_s"])))
CUSTOM_SIGMA_1MIN_N = {_normalise(k): v for k, v in CUSTOM_SIGMA_1MIN.items()}

# Modes in which a planet cannot be observed (inputs/noise/unobservable.csv): {planet name: {band: reason}}. rms_1min raises NoNoiseLevelError for
# them before PandExo runs, exactly as for a band without any noise level: bias.tables.planet_bias_table fits them without a Fisher sigma
# (VERDICT_NO_NOISE), joint fits that need the detector and the false-positive tables leave the planet out, retrievals/run_dynesty.py skips the
# entry, and the retrieval series of retrievals/03_retrieval_series.ipynb leave it out (observable). Planet names are matched loosely.
UNOBSERVABLE = _read_planet_table(UNOBSERVABLE_TABLE, lambda r: r["reason"])
UNOBSERVABLE_N = {_normalise(k): v for k, v in UNOBSERVABLE.items()}
for _pl, _d in PARTIAL_BANDS.items():
    for _ins in _d:
        if _ins in UNOBSERVABLE_N.get(_normalise(_pl), {}): raise ValueError(f"{_pl!r} on {_ins!r} is both in {UNOBSERVABLE_TABLE.name} and in {PARTIAL_BANDS_TABLE.name}")


def observable(planet_name, instrument):
    """False if any detector of `instrument` (a band, a detector tuple or a JOINT_INSTRUMENTS alias) is in UNOBSERVABLE for the planet."""
    dets = resolve_instrument(instrument)
    return not any(d in UNOBSERVABLE_N.get(_normalise(planet_name), {}) for d in ([dets] if isinstance(dets, str) else dets))


# Calculate the rms noise per minute of observation
def rms_1min_pandexo(result, T14, lambda_range, visualize=False, extra=False):
    """The uncertainty per transit time interval is sqrt(2) times smaller than the transit precision itself.
    In other words: if the precision (or rms) of 1 hour of observations is 10 ppm and the transit lasts 1 hour, the transit depth is measured to 10 * sqrt(2) ppm, because the transit is always measured relative to the baseline.
    This example assumes in-transit time = out-of-transit time. More precisely,
            rms_transit = rms_integration * sqrt(1 / Nin + 1 / Nout),
    with Nin = Nout the number of integrations in transit and rms_integration the precision per integration.
    """
    l1, l2 = lambda_range*1e6 # wavelength range in um
    rms_transit_bin = binning(l1, l2, result, visualize=visualize)
    integration_duration = result['timing']['Time/Integration incl reset (sec)']  # [sec]
    transit_duration = T14 * 60 * 60  # [sec]; T14 is in hours
    integrations_per_transit = transit_duration / integration_duration  # Nin
    rms_per_integration = rms_transit_bin / np.sqrt(2/integrations_per_transit)  # frac = 1 and therefore Nin = Nout = integrations_per_transit (for this to work it is important to keep exo_dict['observation']['noccultations']=1, exo_dict['observation']['baseline_unit']='frac', exo_dict['observation']['baseline']=1)
    rms_per_minute = rms_per_integration * np.sqrt(integration_duration / 60)

    if extra: return {"Precision Transit": rms_transit_bin, "Integration Duration": integration_duration, "RMS_1min_pandexo": rms_per_minute}
    else: return rms_per_minute


def pandexo_params(planet_name, **overrides):
    """Everything PandExo needs about one planet, taken from the INPUT_GLOB aggregate table by name. A required value that is neither in the table nor overridden raises ValueError.
    """
    bad = set(overrides) - set(PANDEXO_PARAM_KEYS)
    if bad:
        raise TypeError(f"unknown override(s) {sorted(bad)}; valid keys are {list(PANDEXO_PARAM_KEYS)}")

    row = _archive().pipe(lambda d: d[d["_key"] == _normalise(planet_name)])
    if row.empty:
        raise KeyError(f"{planet_name!r} not found in {INPUT_GLOB}")
    row = row.iloc[0]

    params = {key: row[col] for key, col in PANDEXO_COLUMNS.items()}
    params["name"] = row["pl_name"]

    # normalisation magnitude: K if the archive has it, else J
    mag_col = next((c for c in ("sy_kmag", "sy_jmag") if pd.notna(row[c])), None)
    params["mag"] = row[mag_col] if mag_col else np.nan
    params["ref_wave"] = REF_WAVE_UM[mag_col] if mag_col else np.nan
    params["mag_band"] = mag_col if mag_col else "override"

    params.update(overrides)

    missing = [k for k in PANDEXO_PARAM_KEYS if pd.isna(params[k])]
    if missing:
        where = ", ".join(f"{k} (column {PANDEXO_COLUMNS.get(k, 'sy_kmag/sy_jmag')})" for k in missing)
        raise ValueError(f"{params['name']}: {INPUT_GLOB} has no value for {where}. "
                         f"Pass it explicitly, e.g. pandexo_params({planet_name!r}, {missing[0]}=<value>).")
    return params


# The settings of run_pandexo that do not come from the planet. Also part of the PandExo cache key (see rms_1min), so change them here only.
PANDEXO_FIXED = dict(sat_level=60, sat_unit="%", noccultations=1, R=None, baseline_unit="frac", baseline=1, noise_floor=0,
                     star_type="phoenix", planet_type="constant", ngroup="optimize", background="ecliptic", background_level="high")


def run_pandexo(planet_name, instrument, verbose=True, **overrides):
    """PandExo noise estimate for one archive planet, identified by name alone. instrument is a FILTER_LAMBDA_UM key; the PandExo mode is looked up via pandexo_mode.
    Any PANDEXO_PARAM_KEYS entry can be overridden by keyword.
    """
    import pandexo.engine.justdoit as jdi   # imported lazily: PandExo is optional, and a module-level import would break every other notebook that imports this file

    mode = pandexo_mode(instrument)
    p = pandexo_params(planet_name, **overrides)

    if verbose:   # the full set of values actually used, once per call, each tagged with where it came from
        src = lambda k: "override" if k in overrides else PANDEXO_COLUMNS.get(k, p["mag_band"])
        unit = {"period": "d", "rp_rearth": "R_earth", "rs_rsun": "R_sun", "teff": "K", "logg": "log cgs",
                "metal": "log Fe/H", "t14_h": "h", "mag": f"mag ({p['mag_band']})", "ref_wave": "um"}
        print(f"PandExo inputs for {p['name']}  [instrument={instrument} (PandExo: {mode})]")
        for k in PANDEXO_PARAM_KEYS:
            print(f"    {k:10s} = {float(p[k]):<12.5g} {unit[k]:<16s} <- {src(k)}")
        print("    fixed: " + ", ".join(f"{k}={v}" for k, v in PANDEXO_FIXED.items()) + "\n")

    fx = PANDEXO_FIXED
    exo_dict = jdi.load_exo_dict()
    exo_dict['observation']['sat_level'] = fx["sat_level"]       # saturation level in percent of full well
    exo_dict['observation']['sat_unit'] = fx["sat_unit"]
    exo_dict['observation']['noccultations'] = fx["noccultations"]   # number of transits
    exo_dict['observation']['R'] = fx["R"]                       # None: no fixed binning -- bin afterwards with binning() instead of redoing the calculation
    exo_dict['observation']['baseline_unit'] = fx["baseline_unit"]   # 'frac' -> baseline is the in/out time ratio
    exo_dict['observation']['baseline'] = fx["baseline"]
    exo_dict['observation']['noise_floor'] = fx["noise_floor"]   # fixed level [ppm], or a path to a wavelength-dependent solution

    exo_dict['star']['type'] = fx["star_type"]          # phoenix or user (if you have your own)
    exo_dict['star']['mag'] = p['mag']
    exo_dict['star']['ref_wave'] = p['ref_wave']  # J = 1.25, H = 1.6, K = 2.22 micron
    exo_dict['star']['temp'] = p['teff']
    exo_dict['star']['metal'] = p['metal']        # log Fe/H
    exo_dict['star']['logg'] = p['logg']          # log surface gravity, cgs
    exo_dict['star']['radius'] = p['rs_rsun']
    exo_dict['star']['r_unit'] = 'R_sun'

    exo_dict['planet']['type'] = fx["planet_type"]      # constant: fixed transit depth
    exo_dict['planet']['transit_duration'] = p['t14_h'] * 3600.
    exo_dict['planet']['td_unit'] = 's'
    exo_dict['planet']['radius'] = p['rp_rearth']
    exo_dict['planet']['r_unit'] = 'R_earth'
    exo_dict['planet']['f_unit'] = 'rp^2/r*^2'    # primary transit

    inst_dict = jdi.load_mode_dict(mode)
    inst_dict["configuration"]["detector"]["ngroup"] = fx["ngroup"]   # optimize: picks the maximum that stays under sat_level
    inst_dict['background'] = fx["background"]
    inst_dict['background_level'] = fx["background_level"]

    result = jdi.run_pandexo(exo_dict, inst_dict, save_file=False)

    if verbose: print(result['warning'])

    return result


# PandExo's transit-depth error (jwst.py, var_tot) from the in/out-of-transit electron counts N and their variances V; t_ratio = on-source time out /
# in of transit.
def pandexo_depth_error(N_in, N_out, V_in, V_out, t_ratio):
    return t_ratio * np.sqrt(V_in / N_out**2 + N_in**2 * V_out / N_out**4)


# White-light transit-depth precision [ppm] of the band (lam_start, lam_end) [micron] from a PandExo result run with R = None (run_pandexo): sum the
# electrons and variances of the band's pixels, then apply PandExo's depth error to the sums. This is how PandExo's own engine bins when R is set, and
# the noise of the photon-weighted band depth SPIDERMAN computes. Saturated pixels are left out. Not sqrt(sum sigma_i^2) / N: that is the error of an
# equal-weight mean of the pixel depths, 1.4-1.6 times larger on MIRI/LRS.
def binning(lam_start, lam_end, result, visualize=True):
    raw, spec, timing = result["RawData"], result["FinalSpectrum"], result["timing"]
    wave, error = np.asarray(raw["wave"]), np.asarray(spec["error_w_floor"])
    N_in, N_out, V_in, V_out = (np.asarray(raw[k], float) for k in ("electrons_in", "electrons_out", "var_in", "var_out"))
    t_ratio = timing["On Source Time Out of Transit"] / timing["On Source Time In Transit"]

    # PandExo sets the error of a fully saturated pixel to NaN (mask_fully_saturated_final_spectrum), so the self-check and the sums below use the
    # other pixels only
    saturated = np.asarray(spec.get("full_saturation_mask", np.zeros(wave.size)), bool)
    if saturated.shape != wave.shape: raise ValueError(f"binning(): full_saturation_mask has shape {saturated.shape}, the spectrum {wave.shape}")

    # self-check: the formula must reproduce PandExo's own per-pixel errors (fails e.g. with a noise floor, which PandExo applies per binned point and
    # this sum would drop)
    if not np.allclose(pandexo_depth_error(N_in, N_out, V_in, V_out, t_ratio)[~saturated], error[~saturated], rtol=1e-6):
        raise ValueError("binning(): the summed-electron error does not reproduce PandExo's per-pixel errors (noise floor set?)")

    in_band = (lam_start < wave) & (wave < lam_end)
    use = in_band & ~saturated
    if in_band.any() and not use.any(): raise NoNoiseLevelError(f"every PandExo pixel in {lam_start:g}-{lam_end:g} um is fully saturated: no noise level in this band")   # not observable
    if not use.any(): raise ValueError(f"no usable PandExo pixels in {lam_start:g}-{lam_end:g} um")
    if (in_band & saturated).any():   # the noise then belongs to the unsaturated part of the band, while the forward model integrates all of it unless the planet has a PARTIAL_BANDS entry
        lo, hi = wave[in_band & saturated].min(), wave[in_band & saturated].max()
        print(f"binning(): {int((in_band & saturated).sum())} of {int(in_band.sum())} PandExo pixels in {lam_start:g}-{lam_end:g} um are fully saturated "
              f"({lo:.2f}-{hi:.2f} um) and left out of the noise; give the planet the clean range in inputs/noise/partial_bands.csv, so that the forward model integrates the same band", file=sys.stderr)
    rms_transit = 1e6 * pandexo_depth_error(N_in[use].sum(), N_out[use].sum(), V_in[use].sum(), V_out[use].sum(), t_ratio)

    if visualize:
        import matplotlib.pyplot as plt
        plt.errorbar(wave, spec["spectrum"], yerr=error, fmt=".", color="0.7", label="PandExo pixels")
        plt.errorbar(wave[use], np.asarray(spec["spectrum"])[use], yerr=error[use], fmt=".", label=f"band, white light: {rms_transit:.2f} ppm")
        plt.xlabel("wavelength [micron]"); plt.ylabel("transit depth"); plt.legend()
        plt.show()
    return rms_transit


# Raised by rms_1min when neither PandExo nor CUSTOM_SIGMA_1MIN can supply a noise level, so callers can skip exactly that case (and not PandExo /
# archive errors).
class NoNoiseLevelError(ValueError):
    pass


# PandExo cache: sigma_1min of every PandExo run made through rms_1min. A run takes seconds, a lookup milliseconds. Two files: the shipped
# paths.PANDEXO_CACHE (inputs/noise/pandexo_cache.csv: the planets and bands of the paper), which is only read, and paths.PANDEXO_USER_CACHE, which is
# read first and receives every new run (default ~/.cache/hyperphase/pandexo_cache.csv; HYPERPHASE_PANDEXO_CACHE moves it, also onto the shipped
# file). The key is everything the value depends on: the PandExo inputs (pandexo_params: planet, radii, T_eff, log g, [Fe/H], T14, normalisation
# magnitude and band), the PandExo mode and the band binned over, PANDEXO_FIXED, PANDEXO_NOISE_VERSION and the installed PandExo / Pandeia
# versions. Any change (an archive update, an override, a new PandExo) is a miss: rms_1min runs PandExo and adds the row, so a value computed from
# other inputs is never served. The one exception: without PandExo installed (version "n/a"), a row of any PandExo version is served, with a note
# naming the version, since nothing could rerun it; a miss then raises ImportError. A missing file is skipped (a deleted shipped cache means PandExo
# reruns the paper's planets into the user cache). A file is locked while it is read or written (fcntl), so parallel kernels and joblib workers can
# share it. To force a rerun: rms_1min(..., use_cache=False), which replaces the row in the user cache, or delete rows / the file.
PANDEXO_NOISE_VERSION = 2   # version of the noise computation from a PandExo result (binning, rms_1min_pandexo), part of the cache key; bump it when that changes
_PANDEXO_KEY_PARAMS = ("rp_rearth", "rs_rsun", "teff", "logg", "metal", "t14_h", "mag", "ref_wave")   # pandexo_params without the period, which PandExo does not use
_PANDEXO_TEXT = ("planet", "instrument", "mode", "mag_band", "settings", "pandexo_version", "pandeia_version", "created")   # read as text, so "2026.10" stays a version


def _package_version(dist):
    from importlib.metadata import version, PackageNotFoundError
    try: return version(dist)
    except PackageNotFoundError: return "n/a"


def _pandexo_key(p, instrument, band):
    """Cache key of one sigma_1min: p from pandexo_params, band [m]."""
    return dict(planet=p["name"], instrument=instrument, mode=pandexo_mode(instrument), band_lo_um=1e6 * float(band[0]), band_hi_um=1e6 * float(band[1]),
                **{k: float(p[k]) for k in _PANDEXO_KEY_PARAMS}, mag_band=p["mag_band"],
                settings=f"noise v{PANDEXO_NOISE_VERSION}; " + ", ".join(f"{k}={v}" for k, v in PANDEXO_FIXED.items()),
                pandexo_version=_package_version("pandexo.engine"), pandeia_version=_package_version("pandeia.engine"))


def _pandexo_match(df, key):
    """Rows of the cache table that match key: numbers to 1e-9 relative, text exactly. A column the table lacks matches nothing."""
    m = np.ones(len(df), bool)
    for k, v in key.items():
        if k not in df: return np.zeros(len(df), bool)
        m &= np.isclose(pd.to_numeric(df[k], errors="coerce"), v, rtol=1e-9, atol=0.) if isinstance(v, float) else (df[k].astype(str) == str(v)).to_numpy()
    return m


def _read_pandexo_cache(f):
    f.seek(0)
    return pd.read_csv(f, dtype={c: str for c in _PANDEXO_TEXT}) if os.fstat(f.fileno()).st_size else pd.DataFrame()


def _lock(f, exclusive):
    """Lock the open cache file for parallel kernels and workers (fcntl, POSIX only; no lock on Windows)."""
    try: import fcntl
    except ImportError: return
    fcntl.flock(f, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)


_PANDEXO_VERSIONS = ("pandexo_version", "pandeia_version")
_VERSION_NOTED = set()   # PandExo versions whose cached values have been served without PandExo installed (noted once per session)


def _pandexo_installed(key):
    return "n/a" not in (key["pandexo_version"], key["pandeia_version"])


def _pandexo_cache_row(key):
    """The newest row matching key (a pandas Series), from the user cache or else the shipped one; None if neither has it. Without PandExo installed,
    a row of any PandExo version matches when no row matches exactly."""
    tables = []
    for path in dict.fromkeys(p.resolve() for p in (PANDEXO_USER_CACHE, PANDEXO_CACHE)):   # one file if HYPERPHASE_PANDEXO_CACHE names the shipped one
        if not path.exists(): continue
        with open(path) as f:
            _lock(f, exclusive=False)
            tables.append(_read_pandexo_cache(f))
    keys = [key] if _pandexo_installed(key) else [key, {k: v for k, v in key.items() if k not in _PANDEXO_VERSIONS}]
    for k in keys:
        for df in tables:
            m = _pandexo_match(df, k)
            if m.any(): return df.loc[m].iloc[-1]
    return None


def pandexo_cache_lookup(key):
    """sigma_1min [ppm/min] of the newest row matching key, from the user cache or else the shipped one; None if neither has it."""
    row = _pandexo_cache_row(key)
    return None if row is None else float(row["sigma_1min"])


def _pandexo_cache_store(entries):
    """Add (key, values) entries to the user cache, replacing rows with the same key; read-modify-write under an exclusive lock."""
    PANDEXO_USER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PANDEXO_USER_CACHE.touch(exist_ok=True)
    with open(PANDEXO_USER_CACHE, "r+") as f:
        _lock(f, exclusive=True)
        df = _read_pandexo_cache(f)
        for key, _ in entries:
            if len(df): df = df[~_pandexo_match(df, key)]
        new = pd.DataFrame([{**key, **values} for key, values in entries])
        df = pd.concat([df, new], ignore_index=True) if len(df) else new
        f.seek(0); f.truncate()
        df.to_csv(f, index=False)


def rms_1min(planet, instrument, lrs_range=None, use_cache=True, **overrides):
    """Noise level sigma_1min [ppm/min] of one planet on one instrument: from the PandExo cache (user cache, then the shipped one) if it has this exact configuration,
    else run_pandexo + rms_1min_pandexo, stored in the cache. instrument is a FILTER_LAMBDA_UM key (the PandExo mode is looked up via pandexo_mode),
    lrs_range [m] the band the PandExo spectrum is binned over (default: filter_lambda(instrument, planet), i.e. the planet's PARTIAL_BANDS range
    where it has one). overrides go to run_pandexo (verbose, mag, ...); the transit duration is planet.T14 unless t14_h overrides it. A new run also
    stores every other detector of the same PandExo mode at its own band, from the same result (one NIRSpec G395H run gives NRS1 and NRS2).
    use_cache=False reruns PandExo and replaces the row in the user cache. Where PandExo has no mode for the instrument, the value comes from CUSTOM_SIGMA_1MIN
    (lrs_range and overrides then do not apply); if that has none either, a NoNoiseLevelError says where to add one. A mode listed in UNOBSERVABLE for
    the planet, and a band whose PandExo pixels are all fully saturated, raise NoNoiseLevelError as well (the latter is not cached: PandExo reruns on
    the next call). Without PandExo installed, a cached value of any PandExo version is served (with a note naming it), and a miss raises ImportError."""
    reason = UNOBSERVABLE_N.get(_normalise(planet.name), {}).get(instrument)
    if reason: raise NoNoiseLevelError(f"no noise level for {planet.name!r} on {instrument!r}: {reason} ({UNOBSERVABLE_TABLE.name})")
    if PANDEXO_MODES[instrument] is None:
        custom = CUSTOM_SIGMA_1MIN_N.get(_normalise(planet.name), {})
        if instrument not in custom:
            raise NoNoiseLevelError(f"no noise level for {planet.name!r} on {instrument!r}: PandExo has no mode for this instrument and inputs/noise/custom_sigma_1min.csv has no entry "
                             f"for this planet/instrument (it has {sorted(custom) or 'none'} for this planet). Add one there, or pass sigma_1min by hand.")
        if overrides.pop("verbose", False): print(f"{planet.name} [{instrument}]: sigma_1min = {custom[instrument]:.1f} ppm/min from CUSTOM_SIGMA_1MIN (no PandExo mode)")
        return custom[instrument]
    if lrs_range is None: lrs_range = filter_lambda(instrument, planet)   # the planet's own part of the band where its host saturates the rest (PARTIAL_BANDS)
    overrides.setdefault("t14_h", planet.T14)
    verbose = overrides.pop("verbose", True)
    p = pandexo_params(planet.name, **overrides)   # exactly the inputs run_pandexo would use: the cache key
    key = _pandexo_key(p, instrument, lrs_range)
    if use_cache:
        row = _pandexo_cache_row(key)
        if row is not None:
            cached, version = float(row["sigma_1min"]), f"PandExo {row['pandexo_version']} / Pandeia {row['pandeia_version']}"
            if not _pandexo_installed(key) and version not in _VERSION_NOTED:
                _VERSION_NOTED.add(version)
                print(f"note: PandExo is not installed, so the noise levels come from the cached runs of {version}")
            if verbose: print(f"{p['name']} [{instrument}]: sigma_1min = {cached:.2f} ppm/min from the PandExo cache" + ("" if _pandexo_installed(key) else f" ({version})"))
            return cached
    if not _pandexo_installed(key):
        raise ImportError(f"no cached noise level for {planet.name!r} on {instrument!r} at these inputs, and PandExo is not installed to compute one: install the "
                          f"noise extra (pip install -e \".[noise]\") with the Pandeia reference data (pandeia_refdata, PYSYN_CDBS), or pass sigma_1min by hand")
    result = run_pandexo(planet.name, instrument, verbose=verbose, **overrides)
    created, entries, value = datetime.now().isoformat(timespec="seconds"), [], None
    siblings = [(d, filter_lambda(d, planet)) for d, m in PANDEXO_MODES.items() if m == PANDEXO_MODES[instrument] and d != instrument]
    for ins, band in [(instrument, lrs_range)] + siblings:
        try:
            out = rms_1min_pandexo(result, overrides["t14_h"], band, extra=True)
        except ValueError:
            if ins == instrument: raise
            continue   # a sibling band the run does not cover
        entries.append((_pandexo_key(p, ins, band), dict(sigma_1min=float(out["RMS_1min_pandexo"]), transit_precision_ppm=float(out["Precision Transit"]),
                                                         integration_s=float(out["Integration Duration"]), created=created)))
        if ins == instrument: value = float(out["RMS_1min_pandexo"])
    _pandexo_cache_store(entries)
    return value


# sigma_1min [ppm] for a whole sample of archive planets on ONE instrument: rms_1min per target, keyed by PlanetParams.name. For a joint fit (the
# detector tuple or a JOINT_INSTRUMENTS alias) the value per planet is {detector: ppm/min}, the form planet_bias_table takes for a detector set.
# Served from the PandExo cache where possible; only a planet's first run on a PandExo mode takes seconds. A target the archive cannot supply (no T14,
# no magnitude) or that PandExo fails on raises (KeyError / ValueError from pandexo_params, or PandExo's own error) rather than yielding NaN: a NaN
# sigma_1min would silently corrupt the fits downstream (planet_bias_table). Only planets with no noise level at all (instrument without a PandExo
# mode and no CUSTOM_SIGMA_1MIN entry; for a joint fit: on any of its detectors, since the fit weights them by their noise) are left out of the dict,
# with a note (possibly all of them: then {}). planet_bias_table fits them without a noise level (no Fisher sigma); the false-positive tables, which
# need one, leave them out.
def rms_1min_pandexo_sample(names, instrument, verbose=False, **overrides):
    dets = resolve_instrument(instrument)
    joint = not isinstance(dets, str)
    out, missing = {}, []
    for nm in names:
        pl = hi_planets_by_name(nm)
        try:
            out[pl.name] = {d: rms_1min(pl, d, verbose=verbose, **overrides) for d in dets} if joint else rms_1min(pl, dets, verbose=verbose, **overrides)
        except NoNoiseLevelError:
            missing.append(pl.name)
    if missing:
        print(f"no noise level on {joint_name(dets) if joint else dets} (no PandExo mode and no CUSTOM_SIGMA_1MIN entry, or not observable: UNOBSERVABLE) for {len(missing)} of {len(missing) + len(out)} planets: {', '.join(missing)}")
    return out


# sigma_1min [ppm/min] of ONE planet on several instruments: {instrument: rms_1min(...)}, PandExo (cache) or CUSTOM_SIGMA_1MIN; an instrument with
# neither is left out silently (NoNoiseLevelError), so callers test `ins in levels`. A planet the archive cannot supply PandExo inputs for raises, as
# in rms_1min_pandexo_sample.
def noise_levels(planet, instruments, **overrides):
    out = {}
    for ins in instruments:
        try: out[ins] = rms_1min(planet, ins, verbose=False, **overrides)
        except NoNoiseLevelError: pass
    return out
