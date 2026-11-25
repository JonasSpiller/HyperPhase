"""Where the package finds its data and writes its caches.

DATA_DIR holds everything the model reads but does not compute: filter curves, band and planet tables, the archive snapshot, the NewEra spectral
cube and the shipped PandExo noise cache. It defaults to inputs/ of the repository and can be moved with the environment variable HYPERPHASE_DATA.

New PandExo results are written to PANDEXO_USER_CACHE (default ~/.cache/hyperphase/pandexo_cache.csv, or HYPERPHASE_PANDEXO_CACHE), so running
a new planet never changes the shipped PANDEXO_CACHE. Point HYPERPHASE_PANDEXO_CACHE at inputs/noise/pandexo_cache.csv to extend the shipped cache.
"""
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("HYPERPHASE_DATA", REPO_ROOT / "inputs")).expanduser()
FILTER_DIR = DATA_DIR / "filters" / "Spiderman"            # throughput files (format: bands.load_filter)
BANDS_TABLE = DATA_DIR / "filters" / "bands.csv"            # band edges, throughput file and PandExo mode of every band
ARCHIVE_DIR = DATA_DIR / "exoplanet_archive"                # NASA Exoplanet Archive snapshot (ExoArchive_downloader.py, ExoArchive_customcols.py)
PLANETS_TABLE = DATA_DIR / "planets" / "curated_planets.csv"  # curated planet parameters, one published source per planet
NEWERA_DIR = DATA_DIR / "stellar_model" / "NewEra"          # NewEra spectral cube (built by newera.build_cube)
LD_DIR = DATA_DIR / "stellar_model" / "ExoTiC-LD"           # ExoTiC-LD model cache, filled on first use
NOISE_DIR = DATA_DIR / "noise"
PARTIAL_BANDS_TABLE = NOISE_DIR / "partial_bands.csv"       # bands a planet can be observed in only in part
UNOBSERVABLE_TABLE = NOISE_DIR / "unobservable.csv"         # modes a planet cannot be observed in
CUSTOM_SIGMA_TABLE = NOISE_DIR / "custom_sigma_1min.csv"    # noise levels of bands PandExo has no mode for
PANDEXO_CACHE = NOISE_DIR / "pandexo_cache.csv"             # shipped PandExo noise levels (read)
# new PandExo noise levels (read first, and written)
PANDEXO_USER_CACHE = Path(os.environ.get("HYPERPHASE_PANDEXO_CACHE", Path.home() / ".cache" / "hyperphase" / "pandexo_cache.csv")).expanduser()
