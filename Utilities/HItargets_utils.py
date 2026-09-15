from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from astropy import constants as const
import matplotlib.pyplot as plt
from hyperillumination_utils import PlanetParams
import warnings
warnings.filterwarnings('ignore')

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

# Argument of periastron where the archive has none. 90 deg is the transit convention and is irrelevant at e = 0.
DEFAULT_W_DEG = 90.0

AU_IN_R_SUN = float(const.au / const.R_sun)

# ==========================================================================
# Custom parameters
# ==========================================================================

# entire planet with custom parameters
CUSTOM = {
    "K2-141 b": PlanetParams(name="K2-141 b", period=0.2803244, incl=86.3, ecc=0.0, w=90.0, r1=0.02037, r2=1.0, R=2.292, T2_eff=4599.0, a_AU=0.00747, T14=0.94), # Malavolta et al. 2018, Barragán et al. 2018
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


def _params_from_row(row):
    """One PlanetParams from an archive row, with r1, r2, R in stellar radii and OVERRIDES applied."""
    fields = dict(
        name=row["pl_name"],
        period=row["pl_orbper"],                        # days
        incl=row["pl_orbincl"],                         # deg
        ecc=0.0 if pd.isna(row["pl_orbeccen"]) else row["pl_orbeccen"],
        w=DEFAULT_W_DEG if pd.isna(row["pl_orblper"]) else row["pl_orblper"],
        r1=row["pl_ratror"],                            # stellar radii
        r2=1.0,                                         # stellar radii, by construction
        R=row["a_over_r2"],                             # stellar radii
        T2_eff=row["st_teff"],                          # K
        # a_AU only feeds the light-travel-time correction, so reconstruct it from R and the stellar radius where pl_orbsmax is missing.
        a_AU=row["pl_orbsmax"] if pd.notna(row["pl_orbsmax"]) else row["a_over_r2"] * row["st_rad"] / AU_IN_R_SUN,
        T14=row["pl_trandur"],                   # hours
    )
    fields.update(OVERRIDES_N.get(row["_key"], {}))   # row["_key"] is the normalised pl_name
    return PlanetParams(**fields)


# --------------------------------------------------------------------------
# Planet Parameter Functions
# --------------------------------------------------------------------------

def best_hi_planets(n_planets):
    """The n_planets archive planets most suitable for hyper-illumination, as {name: PlanetParams}, ordered strongest first."""
    df = _archive()
    selected = df[df["hyperillum_valid"]].nsmallest(n_planets, "hyperillum_metric")
    return {row["pl_name"]: CUSTOM.get(row["pl_name"], _params_from_row(row)) for _, row in selected.iterrows()}

def hi_planets_by_name(names):
    """One planet or a list of them as {name: PlanetParams}. Accepts loose spellings ('TOI-2431b'). CUSTOM entries are returned whether or not they exist in the archive."""
    df = _archive()
    if isinstance(names, str):
        custom = CUSTOM_N.get(_normalise(names))
        if custom is not None:
            out = custom
        else:
            row = df[df["_key"] == _normalise(names)].iloc[0]
            out = _params_from_row(row)
    else:
        out = {}
        for name in names:
            custom = CUSTOM_N.get(_normalise(name))
            if custom is not None:
                out[custom.name] = custom
                continue
            row = df[df["_key"] == _normalise(name)].iloc[0]
            out[row["pl_name"]] = _params_from_row(row)
    return out

def best_hi_planets_df(n_planets):
    """Returns df of n_planets most suitable planets for hyper-illumination, including simple metrics."""
    planets = best_hi_planets(n_planets)
    df = pd.DataFrame([[name, *params.args, params.T14, params.terminator_penumbra(unit="deg"), params.terminator_nightside(unit="deg"), params.fraction_illuminated()*100, params.substellar_boost()*100] for name, params in planets.items()],
                        columns=["Name", "Period [d]", "Incl [°]", "Ecc", "w [°]", "Rp/Rs", "Rs/Rs", "a/Rs", "Ts_eff [K]", "a [AU]", "T14 [h]", "theta_hi [°]", "theta_pen [°]", "f_illum [%]", "sub_boost [%]"])
    return df[["Name", "Period [d]", "Incl [°]", "Ecc", "w [°]", "Rp/Rs", "Rs/Rs", "a/Rs", "a [AU]", "Ts_eff [K]", "T14 [h]", "theta_hi [°]", "theta_pen [°]", "f_illum [%]", "sub_boost [%]"]]




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
    w_out = w[mask] / (lam_out*1e6)**2 # correct for SPIDERMAN correction factor that was applied when creating the throughput dat-files. w_out is a dimensionless throughput.
    return lam_out, w_out


FILTER_DIR = Path(__file__).resolve().parent.parent / "Filters" / "Spiderman"


# Used wavelength ranges [um] for observations -- what to integrate over
FILTER_LAMBDA_UM = {
    "MIRI/F1280W":   [11.6, 14.0],   # MIRI imaging TSO. Below the CO2 band, so combined with F1500W it tests for a CO2 atmosphere
    "MIRI/F1500W":   [13.5, 16.6],   # MIRI imaging TSO. Used on TRAPPIST-1 (15 um CO2 band), long wavelengths suit a cold nightside
    "MIRI/F1800W":   [16.5, 19.5],   # MIRI imaging TSO. MIRI's thermal background climbs steeply past ~15 um
    "MIRI/LRS":      [5.0, 10.5],    # suggested by Sebastian. LRS runs to ~12 um but the red end is trimmed where throughput drops
    "Spitzer/IRAC2": [4.0, 5.0],     # Spitzer IRAC channel 2 (4.5 um), to compare with the measured occultation depth fp = 142 ppm (Zieba 2022)
    "JWST_NIRSpec.G395H_NRS1": [2.8530, 3.7363], # two arrays of NIRSpec/G395H
    "JWST_NIRSpec.G395H_NRS2": [3.7959, 5.2040] # two arrays of NIRSpec/G395H
}

# Full filter ranges [um] -- the extent of the throughput files
FULL_LAMBDA_UM = {
    "MIRI/F1280W":   [11.0654, 14.5816],
    "MIRI/F1500W":   [12.7875, 17.4925],
    "MIRI/F1800W":   [15.5725, 20.8505],
    "MIRI/LRS":      [4.8900, 10.6100],   # 5.0-12.0 um and 5.0-14.0 um versions also exist in FILTER_DIR
    "Spitzer/IRAC2": [3.6955, 5.1008],
    "JWST_NIRSpec.G395H_NRS1": [2.8530, 3.7363], # two arrays of NIRSpec/G395H
    "JWST_NIRSpec.G395H_NRS2": [3.7959, 5.2040] # two arrays of NIRSpec/G395H
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

# Noise Level for MIRI/Imaging of TOI-2431b (not included in Pandexo)
# F12800W: 3.59 sec/int: rms_int = 1/989.38*1e6 ppm -> rms_1min = 247.2 ppm
# F15000W: 5.39 sec/int: rms_int = 1/1135.44*1e6 ppm -> rms_1min = 264.0 ppm
# F18000W: 13.48 sec/int: rms_int = 1/1196.43*1e6 ppm -> rms_1min = 396.2 ppm

# Calculate the rms noise per minute of observation
def rms_1min(result, T14, lambda_range, visualize=False, extra=False):
    """The uncertainty per transit time interval is 1.4 times smaller than the transit precision itself.
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

    if extra: return {"Precision Transit": rms_transit_bin, "Integration Duration": integration_duration, "RMS_1min": rms_per_minute}
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
    """PandExo noise estimate for one archive planet, identified by name alone. Any PANDEXO_PARAM_KEYS entry can be overridden by keyword.
    """
    import pandexo.engine.justdoit as jdi   # imported lazily: PandExo is optional, and a module-level import would break every other notebook that imports this file

    p = pandexo_params(planet_name, **overrides)

    if verbose:   # the full set of values actually used, once per call, each tagged with where it came from
        src = lambda k: "override" if k in overrides else PANDEXO_COLUMNS.get(k, p["mag_band"])
        unit = {"period": "d", "rp_rearth": "R_earth", "rs_rsun": "R_sun", "teff": "K", "logg": "log cgs",
                "metal": "log Fe/H", "t14_h": "h", "mag": f"mag ({p['mag_band']})", "ref_wave": "um"}
        print(f"PandExo inputs for {p['name']}  [instrument={instrument}]")
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

    inst_dict = jdi.load_mode_dict(instrument)
    inst_dict["configuration"]["detector"]["ngroup"] = 'optimize'   # picks the maximum that stays under sat_level
    inst_dict['background'] = 'ecliptic'
    inst_dict['background_level'] = 'high'

    result = jdi.run_pandexo(exo_dict, inst_dict, save_file=False)

    if verbose: print(result['warning'])

    return result



def binning(lam_start, lam_end, result, visualize=True):
    # out_NIRSpec = pk.load(open(f'{output_file}', 'rb'))

    # Calculate the average value of the array
    average_value = np.median(result['FinalSpectrum']['error_w_floor'])
    # print(result['FinalSpectrum']['wave'])
    mask1 = result['FinalSpectrum']['error_w_floor'] > 10 * average_value
    # print(mask1)
    mask2 = (lam_start < result['FinalSpectrum']['wave']) & (result['FinalSpectrum']['wave'] < lam_end)
    # print(mask2)
    # mask10 = (5.0974<result['FinalSpectrum']['wave'])&( result['FinalSpectrum']['wave'] <5.10)
    # print(mask10)

    mask = ~mask1 & mask2  # & ~mask10

    if visualize:
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.errorbar(result['FinalSpectrum']['wave'], result['FinalSpectrum']['spectrum'],
                     yerr=result['FinalSpectrum']['error_w_floor'], fmt='.')
        plt.errorbar(result['FinalSpectrum']['wave'][mask], result['FinalSpectrum']['spectrum'][mask],
                     yerr=result['FinalSpectrum']['error_w_floor'][mask], fmt='.')
        # plt.ylim(rprs**2-0.002, rprs**2+0.002)
        # .xlim(5,12)
        # plt.axvline(5.25)
        plt.xlabel('wavelengths (micron)')
        plt.ylabel('tansit depth')
        plt.show()

    errs = result['FinalSpectrum']['error_w_floor']
    errs_masked = errs[mask]
    rms_transit = np.sqrt(np.sum(errs_masked ** 2)) / len(errs_masked) * 1e6
    # print(rms_transit)
    return rms_transit
