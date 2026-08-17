from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import constants as const

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

# Argument of periastron where the archive has none. 90 deg is the transit convention and is irrelevant at e = 0.
DEFAULT_W_DEG = 90.0

AU_IN_R_SUN = float(const.au / const.R_sun)

# ==========================================================================
# Custom parameters
# ==========================================================================

# entire planet with custom parameters
CUSTOM = {
    "K2-141 b": PlanetParams(period=0.2803244, incl=86.3, ecc=0.0, w=90.0, r1=0.02037, r2=1.0, R=2.292, T2_eff=4599.0, a_AU=0.00747), # Malavolta et al. 2018, Barragán et al. 2018
}


OVERRIDES = {
    # "K2-141 b": {"R": 2.292, "r1": 0.02037, "T2_eff": 4599.0},
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

def _params_from_row(row):
    """One PlanetParams from an archive row, with r1, r2, R in stellar radii and OVERRIDES applied."""
    fields = dict(
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
    )
    fields.update(OVERRIDES.get(row["pl_name"], {}))
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
        if names in CUSTOM:
            out = CUSTOM[names]
        else:
            row = df[df["_key"] == _normalise(names)].iloc[0]
            out = _params_from_row(row)
    else:
        out = {}
        for name in names:
            if name in CUSTOM:
                out[name] = CUSTOM[name]
                continue
            row = df[df["_key"] == _normalise(name)].iloc[0]
            out[row["pl_name"]] = _params_from_row(row)
    return out

def best_hi_planets_df(n_planets):
    """Returns df of n_planets most suitable planets for hyper-illumination, including simple metrics."""
    planets = best_hi_planets(n_planets)
    df = pd.DataFrame([[name, *params.args, params.terminator_penumbra(unit="deg"), params.terminator_nightside(unit="deg"), params.fraction_illuminated()*100, params.substellar_boost()*100] for name, params in planets.items()],
                        columns=["Name", "Period [d]", "Incl [°]", "Ecc", "w [°]", "Rp/Rs", "Rs/Rs", "a/Rs", "Ts_eff [K]", "a [AU]", "theta_hi [°]", "theta_pen [°]", "f_illum [%]", "sub_boost [%]"])
    return df[["Name", "Period [d]", "Incl [°]", "Ecc", "w [°]", "Rp/Rs", "Rs/Rs", "a/Rs", "a [AU]", "Ts_eff [K]", "theta_hi [°]", "theta_pen [°]", "f_illum [%]", "sub_boost [%]"]]




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
    mask = (lam > l1) & (lam < l2)
    return lam[mask], w[mask]


FILTER_DIR = Path(__file__).resolve().parent.parent / "Filters" / "Spiderman"


# Effective wavelength ranges [um] -- what to integrate over
EFFECTIVE_LAMBDA_UM = {
    "MIRI/F1280W":   [11.6, 14.0],   # MIRI imaging TSO. Below the CO2 band, so combined with F1500W it tests for a CO2 atmosphere
    "MIRI/F1500W":   [13.5, 16.6],   # MIRI imaging TSO. Used on TRAPPIST-1 (15 um CO2 band), long wavelengths suit a cold nightside
    "MIRI/F1800W":   [16.5, 19.5],   # MIRI imaging TSO. MIRI's thermal background climbs steeply past ~15 um
    "MIRI/LRS":      [5.0, 10.5],    # suggested by Sebastian. LRS runs to ~12 um but the red end is trimmed where throughput drops
    "Spitzer/IRAC2": [4.0, 5.0],     # Spitzer IRAC channel 2 (4.5 um), to compare with the measured occultation depth fp = 142 ppm (Zieba 2022)
}

# Full filter ranges [um] -- the extent of the throughput files
FILTER_LAMBDA_UM = {
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
EFFECTIVE_LAMBDA_M = {name: 1e-6 * np.array(r) for name, r in EFFECTIVE_LAMBDA_UM.items()}
FILTER_LAMBDA_M = {name: 1e-6 * np.array(r) for name, r in FILTER_LAMBDA_UM.items()}

def effective_lambda(instrument):
    """Effective wavelength range [m] as a 2-element array (tophat approximation of throughput)."""
    return EFFECTIVE_LAMBDA_M[instrument]

def filter_lambda(instrument):
    """Full wavelength range [m] of the throughput file as a 2-element array."""
    return FILTER_LAMBDA_M[instrument]

def filter_loc(instrument):
    """Path to the SPIDERMAN throughput file."""
    return str(FILTER_DIR / FILTER_FILES[instrument])


def instruments():
    """The instrument keys, for iterating over all of them."""
    return list(FILTER_FILES)