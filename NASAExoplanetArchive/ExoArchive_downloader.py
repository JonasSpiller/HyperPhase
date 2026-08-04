# Inspired by Sebastian Zieba

"""
NASA Exoplanet Archive downloader (PSCompPars).

Pulls **PSCompPars** (Planetary Systems Composite Parameters) instead of PS.
PSCompPars is already one row per planet, so no aggregation step is needed downstream and only a single file is written (no default_flag=0/1 split).

Then run ExoArchive_customcols.py to add custom columns, which picks up the newest data file automatically.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"



# ==========================================================================
# SETTINGS
# ==========================================================================

# Which archive table. "pscomppars" is one row per planet already; "ps" is one row per planet per paper and would need an aggregation step this pipeline no longer has.
TABLE = "pscomppars"

# Restrict to planets observed to transit. (not necessarily detected by transit)
TRANSITING_ONLY = True

# Also fetch the per-column "<name>_reflink" columns, which record which paper each individual value came from. The aggregator needs these to spot archive-calculated masses and to check that T0 and P share a reference.
WITH_REFLINKS = True

# Output
OUTPUT_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# Column wishlist.
#MEASURED_*: Physical measurements. The archive stores each of these as FOUR columns -- the value, plus <name>err1 (upper error), <name>err2 (lower error, negative), and in PSCompPars also <name>_reflink (the source paper).
# --------------------------------------------------------------------------


PLAIN_COLUMNS = [
    # identity
    "pl_name", "hostname", "pl_letter", "tic_id",
    # system make-up
    "sy_snum", "sy_pnum", "sy_mnum", "cb_flag",
    # discovery
    "discoverymethod", "disc_year", "disc_facility", "disc_telescope",
    "disc_instrument", "disc_locale", "disc_pubdate", "disc_refname",
    # how the planet has been detected
    "tran_flag",       # transits
    "rv_flag",         # radial velocity
    "ttv_flag",        # transit timing variations
    "ima_flag",        # direct imaging
    "obm_flag",        # orbital brightness modulation, i.e. a phase curve
    # misc
    "pl_controv_flag", "pl_bmassprov", "st_spectype", "st_metratio",
    "rastr", "decstr",
    "pl_ntranspec", "pl_nespec", "st_nphot", "st_nrvc",
    # PS-only. Harmless here: they are dropped automatically when the schema
    # check runs against pscomppars.
    "default_flag", "soltype", "pl_refname", "st_refname", "sy_refname",
    "pl_pubdate", "rowupdate", "releasedate",
]

MEASURED_PLANET = [
    "pl_orbper",      # orbital period [days]
    "pl_orbsmax",     # semi-major axis [AU]
    "pl_rade", "pl_radj",           # planet radius [R_earth], [R_jup]
    "pl_bmasse", "pl_bmassj",       # best mass estimate [M_earth], [M_jup]
    "pl_dens",        # bulk density [g/cm3]
    "pl_orbeccen",    # eccentricity
    "pl_insol",       # insolation [S_earth]
    "pl_eqt",         # equilibrium temperature [K] (assumptions vary by paper)
    "pl_orbincl",     # inclination [deg]
    "pl_tranmid",     # transit midpoint [BJD]
    "pl_imppar",      # impact parameter
    "pl_trandep",     # transit depth [percent]
    "pl_trandur",     # transit duration [hours]
    "pl_ratdor",      # a/R* at mid-transit
    "pl_ratror",      # Rp/R*
    "pl_occdep",      # occultation depth [percent]
    "pl_orbtper", "pl_orblper",     # periastron epoch, argument of periastron
    "pl_rvamp",       # RV semi-amplitude [m/s]
    "pl_projobliq", "pl_trueobliq", # obliquity [deg]
]

MEASURED_STAR = [
    "st_teff",        # effective temperature [K]
    "st_rad",         # radius [R_sun]
    "st_mass",        # mass [M_sun]
    "st_met",         # metallicity [dex]
    "st_lum",         # luminosity [log10 L_sun]
    "st_logg",        # surface gravity
    "st_age",         # age [Gyr]
    "st_dens",        # density [g/cm3]
    "st_vsin", "st_rotp", "st_radv",
]

MEASURED_SYSTEM = [
    "ra", "dec", "sy_dist", "sy_plx", "sy_pm", "sy_pmra", "sy_pmdec",
    # host-star apparent magnitudes
    "sy_bmag", "sy_vmag", "sy_jmag", "sy_hmag", "sy_kmag",
    "sy_gaiamag", "sy_tmag", "sy_icmag", "sy_kepmag",
]

MEASURED_COLUMNS = MEASURED_PLANET + MEASURED_STAR + MEASURED_SYSTEM


# Build list of all columns
def build_wishlist(with_reflinks: bool = True) -> list[str]:
    wishlist = list(PLAIN_COLUMNS)
    for name in MEASURED_COLUMNS:
        wishlist += [name, f"{name}err1", f"{name}err2"]
        if with_reflinks:
            wishlist.append(f"{name}_reflink")
    return wishlist


# --------------------------------------------------------------------------
# Querying
# --------------------------------------------------------------------------

def tap_query(adql, fmt="csv"):
    """Download data (Run a synchronous TAP query and return the result as a DataFrame)."""
    url = f"{TAP_URL}?{urlencode({'query': adql, 'format': fmt})}"
    return pd.read_csv(url)


def fetch_schema(table):
    """Return the set of column names present in `table`."""
    adql = (
        "SELECT column_name "
        "FROM TAP_SCHEMA.columns "
        f"WHERE table_name = '{table}'"
    )

    df = tap_query(adql)

    col = "column_name" if "column_name" in df.columns else df.columns[0]
    return set(df[col].dropna().astype(str).str.strip())


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

stamp = datetime.now(timezone.utc)
out_path = OUTPUT_DIR / f"{TABLE}_{stamp:%Y-%m-%d}.csv"

# Build Wishlist

columns = build_wishlist(with_reflinks=WITH_REFLINKS)
print(f"Checking which of {len(columns)} wanted columns exist in '{TABLE}'...")

# Fetch available columns
available = fetch_schema(TABLE) or set(columns)
dropped = [c for c in columns if c not in available]
columns = [c for c in columns if c in available]
print(f"  keeping {len(columns)}, skipping {len(dropped)}: {', '.join(dropped[:8])} {' ...' if len(dropped) > 8 else ''}" if dropped else "")

# Query
where = " where tran_flag = 1" if TRANSITING_ONLY else ""
adql = f"select {', '.join(columns)} from {TABLE}{where}"

# Download
print(f"Downloading {TABLE}...")
df = tap_query(adql)
print(f"  got {len(df)} rows x {len(df.columns)} columns")

# Sanity Check
if df.empty:
    raise RuntimeError("query returned no rows -- check the settings")
n_planets = df["pl_name"].nunique()
if TABLE == "pscomppars" and n_planets != len(df):
    print(f"  ! WARNING: {len(df)} rows but only {n_planets} unique planet names -- pscomppars should be one row per planet")

# Save table
df.to_csv(out_path, index=False)

print(f"Wrote {out_path}")
