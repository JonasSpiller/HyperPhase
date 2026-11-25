# Inspired by the NASA Exoplanet Archive scripts of Sebastian Zieba

"""
NASA Exoplanet Archive aggregator (PSCompPars).

Reads the newest pscomppars_<date>.csv written by ExoArchive_downloader.py and derives phase-curve and observability quantities from it.
"""

import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import constants as const
from hyperphase.orbit import roche_radius_over_r1, transit_duration_t14
from hyperphase.stellar import planck_ratio, planck_ratio_logderiv


# ==========================================================================
# SETTINGS
# ==========================================================================

# Input: the newest file matching this pattern.
INPUT_GLOB = "pscomppars_*.csv"

# Extra planets to splice in. Lines starting with "#" are disabled. Columns absent from the file are left as NaN, and a custom planet replaces the
# archive row of the same name.
CUSTOM_PLANETS_FILE = "custom_planets.txt"

# ADS bibcodes whose values get blanked out, e.g. ["2017AJ....153..136S"] for Stassun et al. 2017. Matched against the *_reflink columns. There is no
# second row to fall back on in PSCompPars, so this creates NaNs rather than substituting another reference.
MASK_BIBCODES = []

# Stellar-parameter overrides, by exact planet name.
STELLAR_OVERRIDES = {"HD 80653 b": {"st_teff": 5838.0}} # st_teff missing


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

R_JUP_IN_R_EARTH = float(const.R_jup / const.R_earth)
M_JUP_IN_M_EARTH = float(const.M_jup / const.M_earth)
R_EARTH_IN_R_SUN = float(const.R_earth / const.R_sun)
M_EARTH_IN_M_SUN = float(const.M_earth / const.M_sun)
AU_IN_R_SUN = float(const.au / const.R_sun)


# Photon-noise reference: WASP-103, 122 ppm per 103 s exposure.
NOISE_REF_PPM = 122.0
JMAG_REF, KMAG_REF, VMAG_REF = 11.100, 10.767, 12.1

# Reference epoch for the ephemeris-drift calculation: 2025-01-01.
T_REF_JD = 2460676.5

# ESM is defined at 7.5 um (JWST MIRI LRS).
LAM_ESM = 7.5e-6

# Wavelengths for the predicted occultation depths, in nanometres.
OCCDEP_WAVELENGTHS_NM = {
    600: "optical",
    640: "optical",
    800: "optical",
    1400: "HST/WFC3",
    4000: "JWST NIRSpec",
    7500: "JWST MIRI",
    8000: "JWST MIRI",
    15000: "JWST MIRI F1500W",
}

# Columns kept in the trimmed output table.
SHORT_COLUMNS = [
    "pl_name",                  # planet name (key)
    "hostname",                 # host star name
    "pl_orbper",                # orbital period [days]
    "pl_rade",                  # planet radius [R_earth]
    "pl_bmasse",                # best mass estimate [M_earth], see pl_bmassprov
    "pl_ratror",                # r1/r2, planet-to-star radius ratio
    "pl_rprs2",                 # (r1/r2)^2, geometric occultation depth [fraction]
    "pl_ratdor",                # a/r2 at MID-TRANSIT, as the archive defines it
    "a_over_r2",                # true semi-major axis / r2; differs from pl_ratdor when e > 0
    "pl_orbsmax",               # semi-major axis [AU]
    "pl_orbeccen",              # eccentricity, often fixed at 0 rather than measured
    "pl_orbincl",               # inclination [deg], 90 where missing (pl_orbincl_derived)
    "pl_orblper",               # argument of periastron
    "e_fitted",                 # True only if the eccentricity was actually fit
    "st_teff",                  # stellar effective temperature [K]
    "st_rad",                   # stellar radius [R_sun]
    "st_met",                   # metallicity [dex]
    "st_logg",                  # surface gravity
    "sy_jmag",                  # J magnitude, drives TSM and the WFC3 noise model
    "sy_kmag",                  # K magnitude, drives ESM
    "sy_vmag",                  # V magnitude, optical noise estimate
    "pl_Teq",                   # equilibrium temperature [K], A_B = 0, full redistribution
    "pl_Teq_err",               # 1-sigma on pl_Teq from st_teff and pl_ratdor
    "pl_Teq_ecc",               # orbit-averaged Teq [K] with the (1-e^2)^(-1/8) correction
    "pl_Tsubst",                # substellar temperature [K], no redistribution
    "pl_Trock",                 # dayside-average bare rock [K] -- the hot limit
    "occdep_1400nm_Trock",      # predicted occultation depth [ppm] at 1.4 um, hot limit
    "occdep_1400nm_Teq",        # same at 1.4 um, cold limit (full redistribution)
    "occdep_7500nm_Trock",      # predicted occultation depth [ppm] at 7.5 um, hot limit
    "occdep_15000nm_Trock",     # predicted occultation depth [ppm] at 15 um, hot limit
    "occdep_refl_A0p1",         # reflected-light contrast [ppm] at albedo A = 0.1
    "ESM",                      # emission metric, ~S/N of a 7.5 um occultation
    "occdep_ESM_err",           # 1-sigma [ppm] on the depth behind ESM
    "TSM",                      # transmission metric, ~S/N of a NIRISS transmission spectrum
    "SNR_HST",                  # 1.4 um bare-rock occultation depth over the WFC3 photon noise
    "t0_n_err",                 # accumulated ephemeris uncertainty [minutes] at T_REF_JD
    "pl_trandur",               # transit duration T14 [hours]
    "delta_Na",                 # Roche-lobe-filling cloud depth [ppm], bound on a Na tail
    "disc_facility",            # discovery facility
    "is_custom",                # True for planets spliced in from CUSTOM_PLANETS_FILE
]

# Decimal places for the trimmed table.
SHORT_ROUNDING = {
    "pl_Teq": 1,
    "pl_Teq_err": 1,
    "pl_Teq_ecc": 1,
    "pl_Tsubst": 1,
    "pl_Trock": 1,
    "occdep_1400nm_Trock": 1,
    "occdep_1400nm_Teq": 1,
    "occdep_7500nm_Trock": 1,
    "occdep_15000nm_Trock": 1,
}


# --------------------------------------------------------------------------
# Helpers  (the physics helpers planck_ratio, transit_duration_t14 and roche_radius_over_r1 come from hyperphase/stellar.py and orbit.py)
# --------------------------------------------------------------------------

def symmetric_sigma(df, base):
    return (df[f"{base}err1"] - df[f"{base}err2"]) / 2.0 # (err1 - err2)/2


def fill_derived(df, col, values):
    """Fill NaNs in `col` from `values`, and record what was filled in <col>_derived so a consumer can tell a published value from one this script computed."""
    was_na = df[col].isna()
    df[col] = df[col].fillna(values)
    df[f"{col}_derived"] = was_na & df[col].notna()


def read_custom_planets(path, columns):
    """Parse custom_planets.txt: a comma-separated header line, then one planet per line, whitespace padded. Lines starting with "#" are disabled."""
    lines = [l.strip() for l in Path(path).read_text().splitlines() if l.strip() and not l.strip().startswith("#")]
    rows = [[field.strip() for field in line.split(",")] for line in lines]
    df = pd.DataFrame(rows[1:], columns=rows[0])
    for col in [c for c in df.columns if c != "pl_name"]:
        df[col] = pd.to_numeric(df[col])
    for col in [c for c in df.columns if c.endswith(("err1", "err2"))]:
        df[col] = df[col].replace(0, np.nan)
    return df.reindex(columns=columns)


# ==========================================================================
# MAIN
# ==========================================================================

HERE = Path(__file__).resolve().parent

# Load Table

input_path = sorted(HERE.glob(INPUT_GLOB))[-1]      # dated names sort chronologically
df = pd.read_csv(input_path, header=0)              # the downloader writes no index column (index=False)
df["is_custom"] = False
print(f"Loaded {input_path.name}: {len(df)} rows x {len(df.columns)} columns")


# Blank the values of the papers in MASK_BIBCODES

if MASK_BIBCODES:
    pattern = "|".join(re.escape(b) for b in MASK_BIBCODES)
    for ref_col in [c for c in df.columns if c.endswith("_reflink")]:
        hit = df[ref_col].astype("string").str.contains(pattern, case=False, na=False)
        df.loc[hit, ref_col[: -len("_reflink")]] = np.nan


# Add custom planets

custom = read_custom_planets(HERE / CUSTOM_PLANETS_FILE, [c for c in df.columns if c != "is_custom"])
custom["is_custom"] = True
df = df[~df["pl_name"].isin(custom["pl_name"])]
df = pd.concat([df, custom], ignore_index=True)
df["is_custom"] = df["is_custom"].fillna(False).astype(bool)


# Stellar-parameter overrides
for name, fields in STELLAR_OVERRIDES.items():
    for col, value in fields.items():
        df.loc[df["pl_name"] == name, col] = value


# Fill gaps
fill_derived(df, "pl_rade", df["pl_radj"] * R_JUP_IN_R_EARTH)
fill_derived(df, "pl_bmasse", df["pl_bmassj"] * M_JUP_IN_M_EARTH)
fill_derived(df, "pl_ratror", R_EARTH_IN_R_SUN * df["pl_rade"] / df["st_rad"])
fill_derived(df, "pl_ratdor", df["pl_orbsmax"] * AU_IN_R_SUN / df["st_rad"])
fill_derived(df, "pl_insol", df["st_teff"] ** 4 / df["pl_ratdor"] ** 2 * AU_IN_R_SUN ** 2 / 5772.0 ** 4)
fill_derived(df, "pl_orbincl", pd.Series(90.0, index=df.index))

df = df.copy() # defragment before adding ~60 columns


# Ratios
df["a_over_r2"] = (df["pl_orbsmax"] * AU_IN_R_SUN / df["st_rad"]).fillna(df["pl_ratdor"]) # true semi-major axis ratio (what the eccentric Teq uses). pl_ratdor is the separation at MID-TRANSIT over r2, which equals the semi-major axis ratio only at e = 0.
df["pl_rprs2"] = df["pl_ratror"] ** 2  # naive transit depth (ignoring limb darkening and bandpass)


# Temperatures (assuming Bond albedo A_B = 0)
df["pl_Teq"] = df["st_teff"] / np.sqrt(2.0 * df["pl_ratdor"]) # full heat redistribution, f = 1/4:  Teff * sqrt(r2 / 2R)
df["pl_Teq_ecc"] = df["st_teff"] / np.sqrt(2.0 * df["a_over_r2"]) * (1.0 - df["pl_orbeccen"].fillna(0.0).clip(0.0, 0.99)**2)**(-0.125)  # orbit-averaged. Insolation goes as a^-2 (1-e^2)^-1/2, so the temperature picks up (1-e^2)^(-1/8). Uses the semi-major axis, not the mid-transit separation.
df["pl_Tsubst"] = df["pl_Teq"] * np.sqrt(2.0)  # substellar point, no redistribution
df["pl_Trock"] = df["pl_Tsubst"] * (2.0 / 3.0) ** 0.25 # dayside average of a bare rock, no redistribution -- the hot limit

# 1-sigma on Teq
rel_teff = symmetric_sigma(df, "st_teff") / df["st_teff"]
rel_ratdor = symmetric_sigma(df, "pl_ratdor") / df["pl_ratdor"]
df["pl_Teq_err"] = df["pl_Teq"] * np.sqrt(rel_teff ** 2 + 0.25 * rel_ratdor ** 2) # d ln Teq = d ln Teff - d ln(a/r2) / 2, hence the 0.25.


# Predicted occultation depths (ppm)
for nm in OCCDEP_WAVELENGTHS_NM:
    for t_name in ("Trock", "Teq"):
        df[f"occdep_{nm}nm_{t_name}"] = planck_ratio(df["pl_rprs2"], df["st_teff"], df[f"pl_{t_name}"], nm * 1e-9)

# Reflected light occ depth at A=0.1 (placeholder)
df["occdep_refl_A0p1"] = 0.1 * df["pl_rprs2"] / df["pl_ratdor"] ** 2 * 1e6

df = df.copy()


# JWST Observability metrics: TSM and ESM (Kempton et al. 2018)

# ESM: Emission Spectroscopy Metric. S/N of an OCCULTATION at 7.5 um with MIRI LRS. The Planck ratio times (r1/r2)^2 is just the occultation depth in
# ppm; the 4.29 puts it on a S/N scale. ESM > 7.5 is roughly "worth observing". ESM evaluates the Planck function at 1.1 * Teq, a blanket stand-in for
# a dayside slightly hotter than the full-redistribution value.
df["occdep_ESM"] = planck_ratio(df["pl_rprs2"], df["st_teff"], 1.1 * df["pl_Teq"], LAM_ESM)
df["ESM"] = 4.29 * df["occdep_ESM"] * 10 ** (-0.2 * df["sy_kmag"])

rel_rprs2 = 2.0 * symmetric_sigma(df, "pl_ratror") / df["pl_ratror"]
rel_tp = df["pl_Teq_err"] / df["pl_Teq"]
df["occdep_ESM_err"] = df["occdep_ESM"].abs() * np.sqrt(rel_rprs2 ** 2 + (planck_ratio_logderiv(1.1 * df["pl_Teq"], LAM_ESM) * rel_tp) ** 2)

# TSM: Transmission Spectroscopy Metric. S/N of a transmission spectrum, where starlight filters through the terminator during transit. The signal is
# the extra area blocked by the atmospheric annulus, ~ 2 r1 H / r2^2, with scale height H = kT/(mu g) and surface gravity g ~ M/r1^2.
df["scale"] = pd.cut(df["pl_rade"], [0, 1.5, 2.75, 4.0, np.inf], labels=[0.19, 1.26, 1.28, 1.15]).astype(float)
df["TSM"] = df["scale"] * df["pl_rade"] * df["pl_rprs2"] / R_EARTH_IN_R_SUN ** 2 * df["pl_Teq"] / df["pl_bmasse"] * 10 ** (-0.2 * df["sy_jmag"])

# Photon noise and HST SNR
# efficiency_* is a WFC3 duty-cycle penalty for bright stars: 1 for faint ones, floored at 0.3 below magnitude 5, an empirical quadratic in between.
for band, mag_ref in (("jmag", JMAG_REF), ("kmag", KMAG_REF), ("vmag", VMAG_REF)):
    mag = df[f"sy_{band}"]
    eff = pd.Series(1.0, index=df.index)
    eff[mag <= 8.8] = ((1.375 * mag ** 2 - 1.214 * mag - 26.68) / 64.58)[mag <= 8.8]
    eff[mag <= 5.0] = 0.3
    df[f"efficiency_{band}"] = eff
    df[f"noise_{band}"] = NOISE_REF_PPM * np.sqrt(10 ** (-0.4 * (mag_ref - mag))) * np.sqrt(1.0 / eff)

df["SNR_HST"] = df["occdep_1400nm_Trock"] / df["noise_jmag"]


# Transit duration
fill_derived(df, "pl_trandur", transit_duration_t14(df["pl_orbper"], df["pl_ratdor"], df["pl_ratror"], df["pl_orbincl"]))


# Roche lobe and sodium tail
df["q"] = df["pl_bmasse"] / df["st_mass"] * M_EARTH_IN_M_SUN
df["RR_RP"] = roche_radius_over_r1(df["q"], df["pl_ratdor"], df["pl_ratror"])
df["RR_RE"] = df["RR_RP"] * df["pl_rade"]
df["delta_Na"] = (df["RR_RE"] * R_EARTH_IN_R_SUN / df["st_rad"]) ** 2 * 1e6 # transit depth of a cloud filling the Roche lobe, i.e. an upper bound on an escaping sodium tail.


# Ephemeris drift
# How far the predicted transit time has drifted by T_REF_JD, in minutes. Large values mean the ephemeris is stale and an observation scheduled
# from it could miss the transit. The propagation assumes T0 and P came from one fit and so share a covariance; in PSCompPars they need not, hence
# ephem_same_ref.
n_orbits = (T_REF_JD - df["pl_tranmid"]) / df["pl_orbper"]
df["t0_n_err"] = np.sqrt((n_orbits * symmetric_sigma(df, "pl_orbper")) ** 2 + symmetric_sigma(df, "pl_tranmid") ** 2) * 24 * 60
df["ephem_same_ref"] = df["pl_tranmid_reflink"] == df["pl_orbper_reflink"]

df = df.copy()



# Calculated
# Marks values PSCompPars derived from a mass-radius relation rather than measured; the archive writes "Calculated Value" into the reflink for those.
for base in ("pl_bmasse", "pl_rade"):
    df[f"{base}_calculated"] = df[f"{base}_reflink"].astype("string").str.contains("calculated", case=False, na=False)

# e_fitted separates a measured eccentricity from one fixed at 0 by assumption or quoted only as an upper limit - important for tidal heating
# modelling
e_err = symmetric_sigma(df, "pl_orbeccen")
df["e_fitted"] = df["pl_orbeccen"].notna() & e_err.notna() & (e_err > 0)


# Report
short_period = df[df["pl_orbper"] < 5]
ratror_from_radii = R_EARTH_IN_R_SUN * df["pl_rade"] / df["st_rad"]
ratror_mismatch = ((df["pl_ratror"] - ratror_from_radii) / df["pl_ratror"]).abs()

print("\n--- QC " + "-" * 55)
print(f"planets                          : {len(df)}  ({int(df.is_custom.sum())} custom)")
print(f"phase-curve detections (obm_flag): {int((df.obm_flag == 1).sum())}")
print(f"pl_ratror vs radii, median |diff|: {ratror_mismatch.median():.2%}, >20% for {int((ratror_mismatch > 0.20).sum())}")
print(f"P < 5 d with fitted eccentricity : {int(short_period.e_fitted.sum())} / {len(short_period)}")
print(f"pl_bmasse from M-R relation      : {int(df.pl_bmasse_calculated.sum())}")
for col in ("pl_ratdor", "pl_ratror", "pl_orbeccen", "pl_bmasse", "pl_tranmid", "pl_occdep", "st_teff"):
    print(f"missing {col:<25}: {int(df[col].isna().sum()):5d}  ({df[col].isna().mean():.1%})")
for col in [c for c in df.columns if c.endswith("_derived")]:
    print(f"derived {col[:-8]:<25}: {int(df[col].sum()):5d}")
print("-" * 62)


# Writing file
prefix = f"NASAExoArchive_{datetime.now(timezone.utc):%Y-%m-%d}"
df.to_csv(HERE / f"{prefix}_aggregate.csv", index=False)
df.reindex(columns=SHORT_COLUMNS).round(SHORT_ROUNDING).to_csv(HERE / f"{prefix}_aggregate_short.csv", index=False)
print(f"\nWrote {prefix}_aggregate.csv  ({len(df)} x {len(df.columns)})")
print(f"Wrote {prefix}_aggregate_short.csv")
