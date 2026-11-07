from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from astropy import constants as const
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

###########################################################################################################################
#####
#####   Baseline Settings
#####
###########################################################################################################################

# The one set of model settings of the repo: the default of every function argument and FittingConfig field, and what every notebook imports.
# Only sweeps and comparisons with other codes deviate from them, on purpose.

# Bond albedo A_B, which sets the temperature map: a bare-rock value at the upper end of what airless rocky planets show.
# Those literature values neglect hyperillumination, so for the lava planets they are likely somewhat low.
DEFAULT_A_B = 0.2

# Band-integrated albedo A of the reflected light (not a geometric albedo), independent of A_B: molten and glassy silicates reflect weakly (Essack et al. 2020).
DEFAULT_A = 0.1

# Bolometric linear limb darkening u_B of the irradiance: a model-atmosphere value instead of the grey Eddington value of 0.6.
# For TOI-2431 b, the primary target, the bolometric profile gives 0.49 (PHOENIX) and 0.50 (MPS-ATLAS) (Stellar Model/LimbDarkening.ipynb).
DEFAULT_U_B = 0.5

# Quadratic limb darkening (u1, u2) of the batman transit. The transit is identical in both irradiation models and cancels exactly in every comparison,
# so the value only needs to be plausible: the median of ExoTiC-LD fits over the hosts of PAPER_PLANETS, in the band throughputs of the forward model (Stellar Model/LimbDarkening.ipynb).
DEFAULT_U = (0.09, 0.14)

# Orbits observed and phase-folded. The periods are only 4-7 h, so five orbits are roughly 1-1.5 days of JWST time (about 27 h for TOI-2431 b, 34 h for K2-141 b):
# ambitious, but within the range of a large program.
DEFAULT_N_ORBITS = 5

# Bins per orbit of the phase-folded light curve: 12 min for TOI-2431 b, 15 min for K2-141 b. The width is derived from the period, so the bins tile the orbit exactly.
DEFAULT_N_BINS = 27

# Fit ranges of the point-source fits (simple_bias_analysis): the physical ranges of A_B and of the Kreidberg & Loeb (2016) redistribution F,
# so a fit at a bound ("railed") means the model cannot reproduce the truth with any allowed value.
A_B_BOUNDS = (0., 1.)
F_BOUNDS = (0., 0.5)

# Imported only AFTER the baseline settings: hyperillumination_utils imports them from here.
from hyperillumination_utils import PlanetParams

###########################################################################################################################
#####
#####   Planetary Parameters
#####
###########################################################################################################################


# ==========================================================================
# SETTINGS
# ==========================================================================

# Aggregate table from ExoArchive_customcols.py. The newest match is used.
INPUT_GLOB = "NASAExoArchive_*_aggregate_short.csv"

# Reject companions comparable in size to their star (white dwarf systems, unconverged grazing fits).
MAX_R1_OVER_R2 = 0.3

# Rocky planets: Rp < ROCKY_MAX_RADE Earth radii. The cut is on the ABSOLUTE radius pl_rade, not on Rp/Rs: a giant around an evolved star has a small ratio
# (Kepler-91 b, Rp/Rs = 0.022 but 15.3 R_Earth) and would otherwise pass. A planet without pl_rade is dropped, as it cannot be confirmed rocky.
# Single definition, shared by best_hi_planets and simple_bias_analysis.rocky_sample / overlay_planets.
ROCKY_MAX_RADE = 1.7

# The planets of the paper (tab:targets): the twelve rocky archive planets with the smallest a/R* (aggregate of 2026-08-21), in that order. Hard-coded so that the
# sample does not move with archive updates. best_hi_planets(12) is NOT this set: it ranks by (1 - Rp/R*)/(a/R*), which puts K2-223 b (tiny Rp/R*) ahead of TOI-561 b (no. 12 in a/R*).
PAPER_PLANETS = ("TOI-2431 b", "TOI-2260 b", "Kepler-1340 b", "K2-141 b", "KOI-1843.03", "K2-157 b", "Kepler-990 c", "K2-183 b", "Kepler-78 b", "EPIC 206042996 c", "K2-131 b", "TOI-561 b")

# Argument of periastron where the archive has none. 90 deg is the transit convention and is irrelevant at e = 0.
DEFAULT_W_DEG = 90.0

AU_IN_R_SUN = float(const.au / const.R_sun)

# ==========================================================================
# Custom parameters
# ==========================================================================

# entire planet with custom parameters
CUSTOM = {
    "K2-141 b": PlanetParams(name="K2-141 b", period=0.2803244, incl=86.3, ecc=0.0, w=90.0, r1=0.02037, r2=1.0, R=2.292, T2_eff=4599.0, a_AU=0.00747, T14=0.94, logg=4.62, feh=-0.06), # Malavolta et al. 2018 (logg, [Fe/H]: their Table 2), Barragán et al. 2018
}


OVERRIDES = {
    # T14 in HOURS, matching PlanetParams. The archive's pl_trandur for TOI-2431 b is an ExoFOP TESS duration
    # (0.292 h) inconsistent with the Tas et al. 2026 geometry in the same row, which implies 0.7475 h.
    "TOI 2431 b": {"T14": 0.7475}
}

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _archive():
    """The aggregate table, loaded once, with the ranking metric and validity mask attached."""
    path = sorted((Path(__file__).resolve().parent.parent / "NASAExoplanetArchive").glob(INPUT_GLOB))[-1]
    df = pd.read_csv(path, low_memory=False)
    df["hyperillum_metric"] = (df["pl_ratror"] - 1.0) / df["a_over_r2"]   # (r1 - r2) / R
    df["hyperillum_valid"] = (df["pl_ratror"] < MAX_R1_OVER_R2) & (df["a_over_r2"] > 1.0 + df["pl_ratror"]) # & df["st_teff"].notna()
    df["_key"] = df["pl_name"].str.lower().str.replace(r"[\s\-_]", "", regex=True)
    return df

def _normalise(name):
    """To accept naming inconsistencies"""
    return name.lower().replace(" ", "").replace("-", "").replace("_", "")

# Keyed on the normalised name so loose spellings hit, exactly as hi_planets_by_name does for the archive:
# without this an entry written 'TOI 2431 b' silently misses the table's 'TOI-2431 b'.
# Built at import, so edit CUSTOM / OVERRIDES in this file rather than at runtime.
CUSTOM_N = {_normalise(k): v for k, v in CUSTOM.items()}
OVERRIDES_N = {_normalise(k): v for k, v in OVERRIDES.items()}


_NOTED = set()   # (planet, note) pairs already printed, so each stellar-parameter fallback is reported once per session

def _note_once(name, msg):
    if (name, msg) not in _NOTED:
        _NOTED.add((name, msg)); print(f"{name}: {msg}")

# log g and [Fe/H] of the host from an archive row (only the stellar SED uses them, astro_utils.stellar_spectrum), with fallbacks that are reported instead of silent:
# - no st_logg: log g = log10(G M / R^2) with M the main-sequence mass at st_teff (astro_utils.main_sequence_M_star) and R = st_rad (the main-sequence radius where st_rad is missing too);
#   the archive has no stellar-mass column. Where st_logg IS known this reproduces it to +0.003 dex (median; 16-84 %: +-0.04, M dwarfs +-0.06-0.09), while a fixed 4.5 is off by -0.28 dex for
#   M dwarfs (log g 4.5 instead of 5.0 shifts the LRS stellar flux by ~6 % at 2600 K). Without st_teff, 4.5 (such a host has no stellar spectrum anyway).
# - no st_met: solar [Fe/H] = 0, the standard assumption (k changes by <= 0.8 % per 0.1 dex; Stellar Model/build_newera.ipynb).
# Keys in `skip` (values given in OVERRIDES) are neither derived nor reported.
def _stellar_logg_feh(row, skip=()):
    name, out = row["pl_name"], {}
    if "logg" not in skip:
        logg = row["st_logg"]
        if pd.isna(logg):
            from astro_utils import main_sequence_M_star, main_sequence_R_star   # imported here: astro_utils imports this module
            T, R = row["st_teff"], row["st_rad"]
            if pd.notna(T):
                src = "st_rad" if pd.notna(R) else "the main-sequence radius"
                R = R if pd.notna(R) else float(main_sequence_R_star(T))
                logg = float(np.log10(const.G.cgs.value * float(main_sequence_M_star(T)) * const.M_sun.cgs.value / (R * const.R_sun.cgs.value) ** 2))
                _note_once(name, f"no st_logg in the archive; log g = {logg:.2f} from the main-sequence mass at T_eff = {T:.0f} K and {src}")
            else:
                logg = 4.5
                _note_once(name, "no st_logg and no st_teff in the archive; log g = 4.5 assumed")
        out["logg"] = float(logg)
    if "feh" not in skip:
        feh = row["st_met"]
        if pd.isna(feh):
            feh = 0.0
            _note_once(name, "no st_met in the archive; solar [Fe/H] = 0 assumed for the stellar spectrum")
        out["feh"] = float(feh)
    return out

def _params_from_row(row):
    """One PlanetParams from an archive row, with r1, r2, R in stellar radii and OVERRIDES applied."""
    overrides = OVERRIDES_N.get(row["_key"], {})   # row["_key"] is the normalised pl_name
    fields = dict(
        name=row["pl_name"],
        period=row["pl_orbper"],                        # days
        incl=row["pl_orbincl"],                         # deg
        ecc=0.0 if pd.isna(row["pl_orbeccen"]) else row["pl_orbeccen"],
        w=DEFAULT_W_DEG if pd.isna(row["pl_orblper"]) else row["pl_orblper"],
        r1=row["pl_ratror"],                            # stellar radii
        r2=1.0,                                         # stellar radii, by construction
        R=row["a_over_r2"],                             # stellar radii
        T2_eff=row["st_teff"],  # K
        **_stellar_logg_feh(row, skip=overrides),  # logg [log cgs], feh [dex]: archive values, or reported fallbacks
        # a_AU only feeds the light-travel-time correction, so reconstruct it from R and the stellar radius where pl_orbsmax is missing.
        a_AU=row["pl_orbsmax"] if pd.notna(row["pl_orbsmax"]) else row["a_over_r2"] * row["st_rad"] / AU_IN_R_SUN,
        T14=row["pl_trandur"],                   # hours
    )
    fields.update(overrides)
    return PlanetParams(**fields)


# --------------------------------------------------------------------------
# Planet Parameter Functions
# --------------------------------------------------------------------------

def best_hi_planets(n_planets, max_rade=ROCKY_MAX_RADE):
    """The n_planets ROCKY archive planets (pl_rade < max_rade Earth radii) most suitable for hyper-illumination, as {name: PlanetParams}, ordered strongest first.
    max_rade=None drops the radius cut (giant planets included)."""
    df = _archive()
    ok = df["hyperillum_valid"]
    if max_rade is not None: ok = ok & (df["pl_rade"] < max_rade)   # NaN radius -> False: cannot be confirmed rocky
    selected = df[ok].nsmallest(n_planets, "hyperillum_metric")
    return {row["pl_name"]: CUSTOM.get(row["pl_name"], _params_from_row(row)) for _, row in selected.iterrows()}

def _archive_row(name):
    """The archive row of one planet by loose name; KeyError (not pandas' IndexError) when the archive does not have it."""
    rows = _archive().pipe(lambda d: d[d["_key"] == _normalise(name)])
    if rows.empty: raise KeyError(f"{name!r} not found in {INPUT_GLOB} (nor in CUSTOM)")
    return rows.iloc[0]

def hi_planets_by_name(names):
    """One planet or a list of them as {name: PlanetParams}. Accepts loose spellings ('TOI-2431b'). CUSTOM entries are returned whether or not they exist in the archive. An unknown name raises KeyError."""
    if isinstance(names, str):
        custom = CUSTOM_N.get(_normalise(names))
        if custom is not None:
            out = custom
        else:
            out = _params_from_row(_archive_row(names))
    else:
        out = {}
        for name in names:
            custom = CUSTOM_N.get(_normalise(name))
            if custom is not None:
                out[custom.name] = custom
                continue
            row = _archive_row(name)
            out[row["pl_name"]] = _params_from_row(row)
    return out

def best_hi_planets_df(n_planets, max_rade=ROCKY_MAX_RADE):
    """Returns df of n_planets most suitable (rocky, see best_hi_planets) planets for hyper-illumination, including simple metrics."""
    planets = best_hi_planets(n_planets, max_rade=max_rade)
    df = pd.DataFrame([[name, *params.args, params.T14, params.terminator_penumbra(unit="deg"), params.terminator_nightside(unit="deg"), params.fraction_illuminated()*100, params.substellar_boost()*100] for name, params in planets.items()],
                        columns=["Name", "Period [d]", "Incl [°]", "Ecc", "w [°]", "Rp/Rs", "Rs/Rs", "a/Rs", "Ts_eff [K]", "a [AU]", "T14 [h]", "gamma_f [°]", "gamma_t [°]", "f_illum [%]", "sub_boost [%]"])
    return df[["Name", "Period [d]", "Incl [°]", "Ecc", "w [°]", "Rp/Rs", "Rs/Rs", "a/Rs", "a [AU]", "Ts_eff [K]", "T14 [h]", "gamma_f [°]", "gamma_t [°]", "f_illum [%]", "sub_boost [%]"]]


def archive_densities(name):
    """(rho_star, rho_p) [g/cm^3] of an archive planet by loose name: the host from st_logg and st_rad (rho = 3 g / (4 pi G R), the archive has no mass column), the planet from
    pl_bmasse and pl_rade. NaN where the archive lacks a value; KeyError for a planet the archive does not have (CUSTOM entries are not consulted)."""
    row = _archive_row(name)
    g = 10. ** row["st_logg"]                                                   # [cm/s^2]
    R = row["st_rad"] * const.R_sun.cgs.value
    m_p, r_p = row["pl_bmasse"] * const.M_earth.cgs.value, row["pl_rade"] * const.R_earth.cgs.value
    return 3. * g / (4. * np.pi * const.G.cgs.value * R), m_p / (4. / 3. * np.pi * r_p ** 3)




###########################################################################################################################
#####
#####   Wavelength Filters
#####
###########################################################################################################################

# return throughput
@lru_cache(maxsize=None)
def _read_filter_file(filter_path):
    # Cached disk read
    d = np.loadtxt(filter_path)
    o = np.argsort(d[:, 0])
    return d[o, 0], d[o, 1]

def load_filter(lambda_range, filter_path=None, n=300):
    l1, l2 = float(lambda_range[0]), float(lambda_range[1])
    if filter_path is None:
        lam = np.linspace(l1, l2, n)
        return lam, np.ones_like(lam)
    lam, w = _read_filter_file(str(filter_path))
    mask = (lam >= l1) & (lam <= l2)

    lam_out = lam[mask]
    w_out = w[mask]   # the files are plain throughputs T(lambda) [electrons per incident photon], the SVO / Pandeia convention; the photon weighting lambda/(hc) is applied by the consumer (band_avg_planck here, get_filter in fspiderman)
    return lam_out, w_out


FILTER_DIR = Path(__file__).resolve().parent.parent / "Filters" / "Spiderman"


# Used wavelength ranges [um] for observations -- what to integrate over
FILTER_LAMBDA_UM = {
    "MIRI/F1280W":   [11.6, 14.0],   # MIRI imaging TSO. Below the CO2 band, so combined with F1500W it tests for a CO2 atmosphere
    "MIRI/F1500W":   [13.5, 16.6],   # MIRI imaging TSO. Used on TRAPPIST-1 (15 um CO2 band), long wavelengths suit a cold nightside
    "MIRI/F1800W":   [16.5, 19.5],   # MIRI imaging TSO. MIRI's thermal background climbs steeply past ~15 um
    "MIRI/LRS":      [5.0, 10.5],    # suggested by Sebastian. LRS runs to ~12 um but the red end is trimmed where throughput drops
    "Spitzer/IRAC2": [4.0, 5.0],     # Spitzer IRAC channel 2 (4.5 um), to compare with the measured occultation depth fp = 142 ppm (Zieba 2022)
    "JWST_NIRSpec.G395H_NRS1": [2.9, 3.7], # two arrays of NIRSpec/G395H
    "JWST_NIRSpec.G395H_NRS2": [3.8, 5.2] # two arrays of NIRSpec/G395H
}

# Full filter ranges [um] -- the extent of the throughput files
FULL_LAMBDA_UM = {
    "MIRI/F1280W":   [11.0654, 14.5816],
    "MIRI/F1500W":   [12.7876, 17.4925],
    "MIRI/F1800W":   [15.5725, 20.8505],
    "MIRI/LRS":      [4.8900, 10.6100],   # 5.0-12.0 um and 5.0-14.0 um versions also exist in FILTER_DIR
    "Spitzer/IRAC2": [3.6955, 5.1008],
    "JWST_NIRSpec.G395H_NRS1": [2.8530, 3.7363], # two arrays of NIRSpec/G395H
    "JWST_NIRSpec.G395H_NRS2": [3.7960, 5.2039] # two arrays of NIRSpec/G395H
}

# SPIDERMAN throughput files
FILTER_FILES = {
    "MIRI/F1280W":   "JWST_MIRI.F1280W.spiderman.dat",
    "MIRI/F1500W":   "JWST_MIRI.F1500W.spiderman.dat",
    "MIRI/F1800W":   "JWST_MIRI.F1800W.spiderman.dat",
    "MIRI/LRS":      "MIRI_LRS.5.0-10.5um.spiderman.dat",
    "Spitzer/IRAC2": "Spitzer_IRAC.I2.spiderman.dat",
    "JWST_NIRSpec.G395H_NRS1": "JWST_NIRSpec.G395H_NRS1.spiderman.dat",
    "JWST_NIRSpec.G395H_NRS2": "JWST_NIRSpec.G395H_NRS2.spiderman.dat"
}

# PandExo mode names (jdi.load_mode_dict) -- None where PandExo has no mode
PANDEXO_MODES = {
    "MIRI/F1280W":   None,           # MIRI imaging TSO is not in PandExo: rms_1min uses CUSTOM_SIGMA_1MIN instead
    "MIRI/F1500W":   None,
    "MIRI/F1800W":   None,
    "MIRI/LRS":      "MIRI LRS",
    "Spitzer/IRAC2": None,
    "JWST_NIRSpec.G395H_NRS1": "NIRSpec G395H", # one PandExo mode covers both arrays,
    "JWST_NIRSpec.G395H_NRS2": "NIRSpec G395H"  # rms_1min_pandexo then bins to the array's filter_lambda range
}


# The JWST bands of the paper, by band centre, and the fit list of the per-instrument analyses: every band singly plus one joint fit of the two NIRSpec/G395H arrays
# (a tuple inside the list = one joint fit of those detectors: shared A_B / F, a baseline per detector). Lists, not tuples, so a tuple always means a joint fit.
JWST_BANDS = ["JWST_NIRSpec.G395H_NRS1", "JWST_NIRSpec.G395H_NRS2", "MIRI/LRS", "MIRI/F1280W", "MIRI/F1500W", "MIRI/F1800W"]
JWST_RETRIEVALS = JWST_BANDS[:2] + [tuple(JWST_BANDS[:2])] + JWST_BANDS[2:]
_bad = set(JWST_BANDS) - set(FILTER_LAMBDA_UM)
if _bad: raise KeyError(f"JWST_BANDS uses unknown instrument(s) {sorted(_bad)}; valid keys are {list(FILTER_LAMBDA_UM)}")

# Converted once at import
FILTER_LAMBDA_M = {name: 1e-6 * np.array(r) for name, r in FILTER_LAMBDA_UM.items()}
FULL_LAMBDA_M = {name: 1e-6 * np.array(r) for name, r in FULL_LAMBDA_UM.items()}

def filter_lambda(instrument):
    """The wavelength range [m] used for observations as a 2-element array."""
    return FILTER_LAMBDA_M[instrument]

def full_lambda(instrument):
    """Full wavelength range [m] of the throughput file as a 2-element array."""
    return FULL_LAMBDA_M[instrument]

def filter_loc(instrument):
    """Path to the SPIDERMAN throughput file."""
    return str(FILTER_DIR / FILTER_FILES[instrument])

def pandexo_mode(instrument):
    """The PandExo mode name (as jdi.load_mode_dict expects it) for an instrument key of FILTER_LAMBDA_UM."""
    mode = PANDEXO_MODES[instrument]
    if mode is None:
        raise ValueError(f"PandExo has no mode for {instrument!r}; rms_1min takes such noise levels from CUSTOM_SIGMA_1MIN.")
    return mode


def instruments():
    """The instrument keys, for iterating over all of them."""
    return list(FILTER_FILES)



###########################################################################################################################
#####
#####   Pandexo (Noise Calculation) - inspired by Sebastian Zieba
#####
###########################################################################################################################


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


# Custom noise levels sigma_1min [ppm/min] for instruments PandExo has no mode for (MIRI imaging, ...): {planet name: {instrument (FILTER_LAMBDA_UM key): sigma_1min}}.
# rms_1min falls back to these only where PandExo cannot supply a value. Planet names are matched loosely ("TOI-2431b" == "TOI-2431 b").
CUSTOM_SIGMA_1MIN = {
    "TOI-2431 b": {
        "MIRI/F1280W": sigma_1min_from_integration(989.38, 3.59),    # 3.59 s/int  -> 247.2 ppm/min
        "MIRI/F1500W": sigma_1min_from_integration(1135.44, 5.39),   # 5.39 s/int  -> 264.0 ppm/min
        "MIRI/F1800W": sigma_1min_from_integration(1196.43, 13.48),  # 13.48 s/int -> 396.2 ppm/min
    },
}
_bad = {ins for d in CUSTOM_SIGMA_1MIN.values() for ins in d} - set(FILTER_LAMBDA_UM)
if _bad: raise KeyError(f"CUSTOM_SIGMA_1MIN uses unknown instrument(s) {sorted(_bad)}; valid keys are {list(FILTER_LAMBDA_UM)}")
CUSTOM_SIGMA_1MIN_N = {_normalise(k): v for k, v in CUSTOM_SIGMA_1MIN.items()}

# Calculate the rms noise per minute of observation
def rms_1min_pandexo(result, T14, lambda_range, visualize=False, extra=False):
    """The uncertainty per transit time interval is sqrt(2) times smaller than the transit precision itself.
    Or differently said, lets say you know that the precision (or rms) of 1 hour of observations is 10 ppm, and your transit has a duration of 1 hour, the precision youll measure the transit depth is 10 * sqrt(2) ppm because you always measure the transit relative to the baseline.
    This example assumes in transit time = out of transit time. More precisely its
            rms_transit = rms_integration * sqrt(1 / Nin + 1 / Nout)
    , with Nin=Nout being the number of integrations in transit and rms_integration the precision per integration.
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
        print("    fixed: noccultations=1, sat_level=60%, baseline=1 (frac), R=None (native), noise_floor=0 ppm, "
              "ngroup=optimize, background=ecliptic/high\n")

    exo_dict = jdi.load_exo_dict()
    exo_dict['observation']['sat_level'] = 60     # saturation level in percent of full well
    exo_dict['observation']['sat_unit'] = '%'
    exo_dict['observation']['noccultations'] = 1  # number of transits
    exo_dict['observation']['R'] = None           # no fixed binning -- bin afterwards with binning() instead of redoing the calculation
    exo_dict['observation']['baseline_unit'] = 'frac'  # 'frac' -> baseline is the in/out time ratio
    exo_dict['observation']['baseline'] = 1
    exo_dict['observation']['noise_floor'] = 0    # fixed level [ppm], or a path to a wavelength-dependent solution

    exo_dict['star']['type'] = 'phoenix'          # phoenix or user (if you have your own)
    exo_dict['star']['mag'] = p['mag']
    exo_dict['star']['ref_wave'] = p['ref_wave']  # J = 1.25, H = 1.6, K = 2.22 micron
    exo_dict['star']['temp'] = p['teff']
    exo_dict['star']['metal'] = p['metal']        # log Fe/H
    exo_dict['star']['logg'] = p['logg']          # log surface gravity, cgs
    exo_dict['star']['radius'] = p['rs_rsun']
    exo_dict['star']['r_unit'] = 'R_sun'

    exo_dict['planet']['type'] = 'constant'       # fixed transit depth
    exo_dict['planet']['transit_duration'] = p['t14_h'] * 3600.
    exo_dict['planet']['td_unit'] = 's'
    exo_dict['planet']['radius'] = p['rp_rearth']
    exo_dict['planet']['r_unit'] = 'R_earth'
    exo_dict['planet']['f_unit'] = 'rp^2/r*^2'    # primary transit

    inst_dict = jdi.load_mode_dict(mode)
    inst_dict["configuration"]["detector"]["ngroup"] = 'optimize'   # picks the maximum that stays under sat_level
    inst_dict['background'] = 'ecliptic'
    inst_dict['background_level'] = 'high'

    result = jdi.run_pandexo(exo_dict, inst_dict, save_file=False)

    if verbose: print(result['warning'])

    return result



# PandExo's transit-depth error (jwst.py, var_tot) from the in/out-of-transit electron counts N and their variances V; t_ratio = on-source time out / in of transit.
def pandexo_depth_error(N_in, N_out, V_in, V_out, t_ratio):
    return t_ratio * np.sqrt(V_in / N_out**2 + N_in**2 * V_out / N_out**4)


# White-light transit-depth precision [ppm] of the band (lam_start, lam_end) [micron] from a PandExo result run with R = None (run_pandexo): sum the electrons and variances of the
# band's pixels, then apply PandExo's depth error to the sums. This is how PandExo's own engine bins when R is set, and the noise of the photon-weighted band depth SPIDERMAN computes.
# Saturated pixels are left out. Not sqrt(sum sigma_i^2) / N: that is the error of an equal-weight mean of the pixel depths, 1.4-1.6 times larger on MIRI/LRS.
def binning(lam_start, lam_end, result, visualize=True):
    raw, spec, timing = result["RawData"], result["FinalSpectrum"], result["timing"]
    wave, error = np.asarray(raw["wave"]), np.asarray(spec["error_w_floor"])
    N_in, N_out, V_in, V_out = (np.asarray(raw[k], float) for k in ("electrons_in", "electrons_out", "var_in", "var_out"))
    t_ratio = timing["On Source Time Out of Transit"] / timing["On Source Time In Transit"]

    # self-check: the formula must reproduce PandExo's own per-pixel errors (fails e.g. with a noise floor, which PandExo applies per binned point and this sum would drop)
    if not np.allclose(pandexo_depth_error(N_in, N_out, V_in, V_out, t_ratio), error, rtol=1e-6):
        raise ValueError("binning(): the summed-electron error does not reproduce PandExo's per-pixel errors (noise floor set?)")

    use = (lam_start < wave) & (wave < lam_end) & ~np.asarray(spec.get("full_saturation_mask", np.zeros(wave.size)), bool)
    if not use.any(): raise ValueError(f"no usable PandExo pixels in {lam_start:g}-{lam_end:g} um")
    rms_transit = 1e6 * pandexo_depth_error(N_in[use].sum(), N_out[use].sum(), V_in[use].sum(), V_out[use].sum(), t_ratio)

    if visualize:
        plt.errorbar(wave, spec["spectrum"], yerr=error, fmt=".", color="0.7", label="PandExo pixels")
        plt.errorbar(wave[use], np.asarray(spec["spectrum"])[use], yerr=error[use], fmt=".", label=f"band, white light: {rms_transit:.2f} ppm")
        plt.xlabel("wavelength [micron]"); plt.ylabel("transit depth"); plt.legend()
        plt.show()
    return rms_transit


# Raised by rms_1min when neither PandExo nor CUSTOM_SIGMA_1MIN can supply a noise level, so callers can skip exactly that case (and not PandExo / archive errors).
class NoNoiseLevelError(ValueError):
    pass


# sigma_1min [ppm/min] of one planet on one instrument: run_pandexo + rms_1min_pandexo. instrument is a FILTER_LAMBDA_UM key (the PandExo mode is looked up via pandexo_mode),
# lrs_range [m] the band the PandExo spectrum is binned over (default: filter_lambda(instrument)). overrides go to run_pandexo (verbose, mag, ...); the transit duration is planet.T14 unless t14_h overrides it.
# Where PandExo has no mode for the instrument, the value comes from CUSTOM_SIGMA_1MIN (lrs_range and overrides then do not apply); if that has none either, a NoNoiseLevelError says where to add one.
def rms_1min(planet, instrument, lrs_range=None, **overrides):
    if PANDEXO_MODES[instrument] is None:
        custom = CUSTOM_SIGMA_1MIN_N.get(_normalise(planet.name), {})
        if instrument not in custom:
            raise NoNoiseLevelError(f"no noise level for {planet.name!r} on {instrument!r}: PandExo has no mode for this instrument and CUSTOM_SIGMA_1MIN (HItargets_utils.py) has no entry "
                             f"for this planet/instrument (it has {sorted(custom) or 'none'} for this planet). Add one there, or pass sigma_1min by hand.")
        if overrides.pop("verbose", False): print(f"{planet.name} [{instrument}]: sigma_1min = {custom[instrument]:.1f} ppm/min from CUSTOM_SIGMA_1MIN (no PandExo mode)")
        return custom[instrument]
    if lrs_range is None: lrs_range = filter_lambda(instrument)
    overrides.setdefault("t14_h", planet.T14)
    return rms_1min_pandexo(run_pandexo(planet.name, instrument, **overrides), overrides["t14_h"], lrs_range)


# sigma_1min [ppm] for a whole sample of archive planets on ONE instrument: rms_1min per target, keyed by PlanetParams.name.
# PandExo is NOT cached and takes seconds per planet, so call this once and keep the dict. A target the archive cannot supply (no T14, no magnitude) or that PandExo
# fails on raises (KeyError / ValueError from pandexo_params, or PandExo's own error) rather than yielding NaN: a NaN sigma_1min would silently corrupt the fits downstream (planet_bias_table).
# Only planets with no noise level at all (instrument without a PandExo mode and no CUSTOM_SIGMA_1MIN entry) are left out of the dict, with a note (possibly all of them: then {}).
# planet_bias_table fits them without a noise level (no Fisher sigma); the false-positive tables, which need one, leave them out.
def rms_1min_pandexo_sample(names, instrument, verbose=False, **overrides):
    out, missing = {}, []
    for nm in names:
        pl = hi_planets_by_name(nm)
        try:
            out[pl.name] = rms_1min(pl, instrument, verbose=verbose, **overrides)
        except NoNoiseLevelError:
            missing.append(pl.name)
    if missing:
        print(f"no noise level on {instrument} (no PandExo mode, no CUSTOM_SIGMA_1MIN entry) for {len(missing)} of {len(missing) + len(out)} planets: {', '.join(missing)}")
    return out
