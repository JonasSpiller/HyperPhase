"""Planet parameters: the NASA Exoplanet Archive snapshot in inputs/exoplanet_archive/ and the curated parameters of
inputs/planets/curated_planets.csv, which take precedence over the archive (see CUSTOM).

hi_planets_by_name looks a planet up by name. register_planet and load_planets add planets (or replace curated ones) for the session; a
PlanetParams built by hand needs neither, since every forward-model function takes the object itself."""
import csv
from functools import lru_cache

import numpy as np
import pandas as pd
from astropy import constants as const

from hyperphase._util import _normalise
from hyperphase.paths import ARCHIVE_DIR, PLANETS_TABLE
from hyperphase.planet import PlanetParams
from hyperphase.roche import main_sequence_M_star, main_sequence_R_star
from hyperphase.stellar import planck_ratio


############################################################################################################################################
#####   Archive
############################################################################################################################################

# Aggregate table written by ExoArchive_customcols.py; the newest match is used.
INPUT_GLOB = "NASAExoArchive_*_aggregate_short.csv"

# Reject companions comparable in size to their star (white dwarf systems, unconverged grazing fits).
MAX_R1_OVER_R2 = 0.3

# Emission spectroscopy metric (Kempton et al. 2018, Eq. 4): occultation depth at 7.5 um with T_day = 1.1 T_eq, as in ExoArchive_customcols.py.
ESM_WAVELENGTH = 7.5e-6

# Rocky planets: Rp < ROCKY_MAX_RADE Earth radii. The cut is on the absolute radius pl_rade, not on Rp/Rs: a giant around an evolved star has a
# small ratio (Kepler-91 b, Rp/Rs = 0.022 but 15.3 R_Earth) and would otherwise pass. A planet without pl_rade cannot be confirmed rocky and
# is dropped.
ROCKY_MAX_RADE = 1.7

# Argument of periastron where the archive has none. 90 deg is the transit convention and is irrelevant at e = 0.
DEFAULT_W_DEG = 90.0
AU_IN_R_SUN = float(const.au / const.R_sun)


############################################################################################################################################
#####   Curated planets
############################################################################################################################################

# Curated planets (inputs/planets/curated_planets.csv; register_planet and load_planets add more for the session): CUSTOM = {name: PlanetParams}.
# They take precedence over the archive everywhere: hi_planets_by_name returns these objects, and apply_custom writes them into every archive frame.
# Every value comes from a published source (the table's `source` column): a/R*, Rp/R* and i as fitted; where no paper fits a/R* or Rp/R*, the
# ratio of the paper's tabulated a, Rp and R* (TOI-2431 b, TOI-2260 b, Kepler-1323 b) or its stellar density (TOI-1798.02); i = acos(b / (a/R*))
# from the published b, or 90 deg for two Adams et al. (2021) fits whose i is consistent with it (stated per entry); T14 as published, else
# from the geometry (it only sets PandExo's in-transit integration count). Stellar parameters are published spectroscopic values or a named
# catalogue (TICv8, Fulton & Petigura 2018, Berger et al. 2020). r2 = 1, so R is a/R*; T14 in hours. a_AU = (a/R*) x R* of the adopted star: it only feeds the light-travel-time
# correction, and apply_custom recovers R* and Rp from it. Left out for lack of a citable, consistent geometry: Kepler-780 b, -1340 b, -1368 b,
# -1547 b, -1566 b, -879 c.
_PLANET_FIELDS = ("period", "incl", "ecc", "w", "r1", "r2", "R", "T2_eff", "a_AU", "T14", "logg", "feh")
CUSTOM = {}
SOURCES = {}   # name -> the source of a curated planet's values, where the table gives one


def register_planet(planet, source=None, overwrite=False):
    """Add a PlanetParams to the curated planets (CUSTOM) for this session; set its a_AU as described there. A planet that is already curated
    raises unless overwrite=True."""
    key = _normalise(planet.name)
    old = next((k for k in CUSTOM if _normalise(k) == key), None)
    if old is not None and not overwrite: raise KeyError(f"{planet.name!r} is already a curated planet; pass overwrite=True to replace it")
    if old is not None: del CUSTOM[old]; SOURCES.pop(old, None)
    CUSTOM[planet.name] = planet
    if source: SOURCES[planet.name] = source
    _archive.cache_clear(); archive_frame.cache_clear()


def load_planets(path, overwrite=False):
    """Register every planet of a table in the format of inputs/planets/curated_planets.csv (columns name, period [d], incl [deg], ecc, w [deg],
    r1, r2, R [consistent units], T2_eff [K], a_AU, T14 [h], logg, feh; optional: source). Returns the names."""
    names = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            register_planet(PlanetParams(name=r["name"], **{k: float(r[k]) for k in _PLANET_FIELDS}), source=r.get("source"), overwrite=overwrite)
            names.append(r["name"])
    return names


def _curated(name):
    """The curated PlanetParams of a loosely spelled name, or None."""
    key = _normalise(name)
    return next((pl for k, pl in CUSTOM.items() if _normalise(k) == key), None)


@lru_cache(maxsize=1)
def _archive():
    """The aggregate table, loaded once, with the ranking metric and validity mask attached."""
    path = sorted(ARCHIVE_DIR.glob(INPUT_GLOB))[-1]
    df = apply_custom(pd.read_csv(path, low_memory=False))   # curated planets applied
    df["hyperillum_metric"] = (df["pl_ratror"] - 1.0) / df["a_over_r2"]   # (r1 - r2) / R
    df["hyperillum_valid"] = (df["pl_ratror"] < MAX_R1_OVER_R2) & (df["a_over_r2"] > 1.0 + df["pl_ratror"])
    df["_key"] = df["pl_name"].str.lower().str.replace(r"[\s\-_]", "", regex=True)
    return df


# Archive columns that a PlanetParams field of a curated planet replaces.
_FIELD_COLUMNS = {"period": "pl_orbper", "incl": "pl_orbincl", "ecc": "pl_orbeccen", "w": "pl_orblper", "T2_eff": "st_teff", "logg": "st_logg", "feh": "st_met", "a_AU": "pl_orbsmax", "T14": "pl_trandur"}
R_EARTH_IN_R_SUN = float(const.R_earth / const.R_sun)


def apply_custom(df):
    """A copy of an archive table with the curated planets (CUSTOM) written in, matched on the normalised name; a curated planet the table lacks
    is skipped. It replaces the _FIELD_COLUMNS, pl_ratror and pl_rprs2 (r1/r2), a_over_r2 and pl_ratdor (R/r2), st_rad = a_AU / (a/R*) and
    pl_rade = Rp/R* x st_rad. Of the columns derived from these in ExoArchive_customcols.py, pl_Teq, occdep_ESM and ESM are recomputed;
    occdep_ESM_err is set to NaN (it needs the errors of the curated values), and the rest (the other occultation depths, pl_insol, RR_RP) are left
    as they are: the package does not read them."""
    df = df.copy()
    key = df["pl_name"].map(_normalise)
    for name, pl in CUSTOM.items():
        rows = key == _normalise(name)
        if not rows.any(): continue
        for field, col in _FIELD_COLUMNS.items():
            if col in df: df.loc[rows, col] = getattr(pl, field)
        rp_rs, a_rs = pl.r1 / pl.r2, pl.R / pl.r2
        for col, val in (("pl_ratror", rp_rs), ("pl_rprs2", rp_rs ** 2), ("a_over_r2", a_rs), ("pl_ratdor", a_rs)):
            if col in df: df.loc[rows, col] = val
        r_star = pl.a_AU * AU_IN_R_SUN / a_rs   # [R_sun]
        if "st_rad" in df: df.loc[rows, "st_rad"] = r_star
        if "pl_rade" in df: df.loc[rows, "pl_rade"] = rp_rs * r_star / R_EARTH_IN_R_SUN
        teq = pl.T2_eff / np.sqrt(2.0 * a_rs)                                  # full redistribution, A_B = 0
        depth = planck_ratio(rp_rs ** 2, pl.T2_eff, 1.1 * teq, ESM_WAVELENGTH)  # [ppm]
        for col, val in (("pl_Teq", teq), ("occdep_ESM", depth), ("occdep_ESM_err", np.nan)):
            if col in df: df.loc[rows, col] = val
        if "ESM" in df and "sy_kmag" in df: df.loc[rows, "ESM"] = 4.29 * depth * 10 ** (-0.2 * df.loc[rows, "sy_kmag"])
    return df


_NOTED = set()   # (planet, note) pairs already printed


def _note_once(name, msg):
    if (name, msg) not in _NOTED:
        _NOTED.add((name, msg)); print(f"{name}: {msg}")


def _stellar_logg_feh(row):
    """{"logg", "feh"} of the host from an archive row (only the stellar SED uses them), with reported fallbacks:
    no st_logg: log10(G M / R^2), M the main-sequence mass at st_teff (the short aggregate has no st_mass) and R = st_rad, or the main-sequence
      radius. Where st_logg is known, this reproduces it within +-0.04 dex (16-84 %), while a fixed 4.5 is off by -0.28 dex for M dwarfs.
      Without st_teff: 4.5.
    no st_met: solar [Fe/H] = 0."""
    name, out = row["pl_name"], {}
    logg = row["st_logg"]
    if pd.isna(logg):
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
    feh = row["st_met"]
    if pd.isna(feh):
        feh = 0.0
        _note_once(name, "no st_met in the archive; solar [Fe/H] = 0 assumed for the stellar spectrum")
    out["feh"] = float(feh)
    return out


def _params_from_row(row):
    """One PlanetParams from an archive row, with r1, r2, R in stellar radii. To change single values: dataclasses.replace(planet, incl=...)."""
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
        **_stellar_logg_feh(row),  # logg [log cgs], feh [dex]: archive values, or reported fallbacks
        # a_AU only feeds the light-travel-time correction, so reconstruct it from R and the stellar radius where pl_orbsmax is missing.
        a_AU=row["pl_orbsmax"] if pd.notna(row["pl_orbsmax"]) else row["a_over_r2"] * row["st_rad"] / AU_IN_R_SUN,
        T14=row["pl_trandur"],                   # hours
    )
    return PlanetParams(**fields)


############################################################################################################################################
#####   Planet parameter functions
############################################################################################################################################
def best_hi_planets(n_planets, max_rade=ROCKY_MAX_RADE):
    """The n_planets rocky archive planets (pl_rade < max_rade Earth radii; None: no cut) most suitable for hyper-illumination, ranked by
    hyperillum_metric = (r1 - r2) / R, strongest first: {name: PlanetParams}."""
    df = _archive()
    ok = df["hyperillum_valid"]
    if max_rade is not None: ok = ok & (df["pl_rade"] < max_rade)   # NaN radius -> False
    selected = df[ok].nsmallest(n_planets, "hyperillum_metric")
    return {row["pl_name"]: _curated(row["pl_name"]) or _params_from_row(row) for _, row in selected.iterrows()}


def _archive_row(name):
    """The archive row of one planet by loose name; KeyError (not pandas' IndexError) when the archive does not have it."""
    rows = _archive().pipe(lambda d: d[d["_key"] == _normalise(name)])
    if rows.empty: raise KeyError(f"{name!r} not found in {INPUT_GLOB} (nor among the curated planets)")
    return rows.iloc[0]


def hi_planets_by_name(names):
    """The PlanetParams of one planet, or {name: PlanetParams} for a list of names. Accepts loose spellings ('TOI-2431b'). Curated planets
    are returned whether or not they exist in the archive. An unknown name raises KeyError."""
    if isinstance(names, str):
        custom = _curated(names)
        if custom is not None:
            out = custom
        else:
            out = _params_from_row(_archive_row(names))
    else:
        out = {}
        for name in names:
            custom = _curated(name)
            if custom is not None:
                out[custom.name] = custom
                continue
            row = _archive_row(name)
            out[row["pl_name"]] = _params_from_row(row)
    return out


def best_hi_planets_df(n_planets, max_rade=ROCKY_MAX_RADE):
    """best_hi_planets as a DataFrame, with simple hyper-illumination metrics."""
    planets = best_hi_planets(n_planets, max_rade=max_rade)
    df = pd.DataFrame([[name, *params.args, params.T14, params.terminator_penumbra(unit="deg"), params.terminator_nightside(unit="deg"), params.fraction_illuminated()*100, params.substellar_boost()*100] for name, params in planets.items()],
                        columns=["Name", "Period [d]", "Incl [°]", "Ecc", "w [°]", "Rp/Rs", "Rs/Rs", "a/Rs", "Ts_eff [K]", "a [AU]", "T14 [h]", "gamma_f [°]", "gamma_t [°]", "f_illum [%]", "sub_boost [%]"])
    return df[["Name", "Period [d]", "Incl [°]", "Ecc", "w [°]", "Rp/Rs", "Rs/Rs", "a/Rs", "a [AU]", "Ts_eff [K]", "T14 [h]", "gamma_f [°]", "gamma_t [°]", "f_illum [%]", "sub_boost [%]"]]


def archive_densities(name):
    """(rho_star, rho_p) [g/cm^3] of an archive planet by loose name: the host from st_logg and st_rad (rho = 3 g / (4 pi G R); the short aggregate
    has no st_mass), the planet from pl_bmasse and pl_rade. NaN where the archive lacks a value; KeyError for a planet the archive does not
    have (the curated planets are not consulted)."""
    row = _archive_row(name)
    g = 10. ** row["st_logg"]                                                   # [cm/s^2]
    R = row["st_rad"] * const.R_sun.cgs.value
    m_p, r_p = row["pl_bmasse"] * const.M_earth.cgs.value, row["pl_rade"] * const.R_earth.cgs.value
    return 3. * g / (4. * np.pi * const.G.cgs.value * R), m_p / (4. / 3. * np.pi * r_p ** 3)


############################################################################################################################################
#####   The archive sample of the bias maps and per-planet tables
############################################################################################################################################

@lru_cache(maxsize=4)
def archive_frame(AGG):
    """The archive aggregate table at path AGG with the curated planets applied (apply_custom), one row per planet; cached per path."""
    return apply_custom(pd.read_csv(AGG)).drop_duplicates("pl_name")


def planet_archive_value(AGG, pl_name, column):
    """One archive value (e.g. "ESM", "pl_rade") of a planet by loose name, curated planets applied; NaN where the planet or the value is missing."""
    d = archive_frame(AGG)
    row = d[d.pl_name.map(_normalise) == _normalise(pl_name)]
    return float(row[column].iloc[0]) if len(row) and pd.notna(row[column].iloc[0]) else np.nan


def planet_esm(AGG, pl_name):
    return planet_archive_value(AGG, pl_name, "ESM")


def rocky_sample(AGG, ykey, ylim, a_rs_lim=(1.5, 4.0), max_rade=ROCKY_MAX_RADE, exclude=None):
    """The rocky archive planets (pl_rade < max_rade, see ROCKY_MAX_RADE) inside a bias map: a/Rs (a_over_r2) within a_rs_lim and ykey within
    ylim, curated planets applied (archive_frame). exclude: names to leave out (loose spellings)."""
    _exo = archive_frame(AGG)
    drop = [exclude] if isinstance(exclude, str) else list(exclude or [])
    keep = ~_exo.pl_name.map(_normalise).isin({_normalise(n) for n in drop})
    _phys = _exo[np.isfinite(_exo.a_over_r2) & np.isfinite(_exo.st_teff) & np.isfinite(_exo.pl_ratror)
                 & (_exo.pl_ratror > 0) & (_exo.pl_ratror < 1)
                 & np.isfinite(_exo.pl_rade) & (_exo.pl_rade < max_rade)   # rocky cut (NaN radius -> dropped)
                 & keep].drop_duplicates("pl_name").copy()
    x = "a_over_r2"
    return _phys[(_phys[x] >= a_rs_lim[0]) & (_phys[x] <= a_rs_lim[1])
                 & (_phys[ykey] >= ylim[0]) & (_phys[ykey] <= ylim[1])]


def closest_in(AGG, n, max_rade=ROCKY_MAX_RADE, exclude=None):
    """Archive names of the n rocky planets (pl_rade < max_rade) with the smallest a/Rs, curated planets applied. exclude: loose spellings."""
    d = archive_frame(AGG)
    drop = {_normalise(x) for x in ([exclude] if isinstance(exclude, str) else list(exclude or []))}
    ok = (np.isfinite(d.a_over_r2) & np.isfinite(d.pl_ratror) & (d.pl_ratror > 0) & (d.a_over_r2 > 1. + d.pl_ratror) & np.isfinite(d.pl_rade) & (d.pl_rade < max_rade)
          & ~d.pl_name.map(_normalise).isin(drop))
    return list(d[ok].nsmallest(n, "a_over_r2").pl_name)


load_planets(PLANETS_TABLE)
