"""Planet parameters, instrument bands, noise levels and the definition of the paper's retrievals."""
import importlib.util
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from hyperphase.bands import FILTER_FILES, filter_lambda, filter_loc, full_lambda
from hyperphase.noise import _package_version, _pandexo_cache_store, _pandexo_key, NoNoiseLevelError, observable, pandexo_params, rms_1min
from hyperphase.paper import PAPER_PLANETS
from hyperphase.paths import ARCHIVE_DIR
from hyperphase.targets import CUSTOM, INPUT_GLOB, hi_planets_by_name, planet_archive_value


def test_filter_files_and_bands():
    for instrument in FILTER_FILES:
        assert Path(filter_loc(instrument)).exists(), instrument
        (l1, l2), (f1, f2) = filter_lambda(instrument), full_lambda(instrument)
        assert f1 <= l1 < l2 <= f2, instrument


@pytest.mark.parametrize("name", PAPER_PLANETS)
def test_paper_planets_have_a_curated_geometry(name):
    planet = hi_planets_by_name(name)
    assert planet is CUSTOM[name]
    assert planet.R / planet.r2 > 1. + planet.r1 / planet.r2   # the planet orbits outside its star


def test_curated_parameters_take_precedence_over_the_archive():
    """TOI-1798.02: the T_eff of the curated table, not the different value of the archive aggregate, by name and in the archive frame."""
    agg = sorted(ARCHIVE_DIR.glob(INPUT_GLOB))[-1]
    raw = pd.read_csv(agg, low_memory=False).drop_duplicates("pl_name").set_index("pl_name").loc["TOI-1798.02", "st_teff"]
    curated = CUSTOM["TOI-1798.02"].T2_eff
    assert curated != raw
    assert hi_planets_by_name("TOI-1798.02").T2_eff == curated
    assert planet_archive_value(agg, "TOI-1798.02", "st_teff") == curated


def test_noise_level_from_the_cache(toi2431b, monkeypatch):
    """The PandExo cache serves sigma_1min without running PandExo (its key includes the PandExo version of the paper)."""
    if _package_version("pandexo.engine") != "2026.7" or _package_version("pandeia.engine") != "2026.7":
        pytest.skip("the cache holds PandExo/Pandeia 2026.7 values")
    monkeypatch.setattr("hyperphase.noise.run_pandexo", lambda *a, **k: pytest.fail("PandExo ran instead of the cache"))
    assert rms_1min(toi2431b, "MIRI/LRS", verbose=False) == pytest.approx(62.93, abs=0.01)


@pytest.fixture
def empty_user_cache(tmp_path, monkeypatch):
    """A user cache of its own for the test, so the developer's ~/.cache does not enter."""
    path = tmp_path / "pandexo_cache.csv"
    monkeypatch.setattr("hyperphase.noise.PANDEXO_USER_CACHE", path)
    return path


def _cache_key(planet, instrument):
    """The key rms_1min looks up for the planet at its own inputs and band."""
    return _pandexo_key(pandexo_params(planet.name, t14_h=planet.T14), instrument, filter_lambda(instrument, planet))


def test_user_cache_comes_first(toi2431b, empty_user_cache, monkeypatch):
    """A row of the user cache shadows the same row of the shipped cache."""
    monkeypatch.setattr("hyperphase.noise.run_pandexo", lambda *a, **k: pytest.fail("PandExo ran instead of the cache"))
    _pandexo_cache_store([(_cache_key(toi2431b, "MIRI/LRS"), dict(sigma_1min=1.))])
    assert rms_1min(toi2431b, "MIRI/LRS", verbose=False) == 1.


def test_shipped_cache_without_pandexo(toi2431b, empty_user_cache, monkeypatch):
    """Without PandExo installed the shipped values are served whatever PandExo version made them, and a miss says what to install."""
    monkeypatch.setattr("hyperphase.noise._package_version", lambda dist: "n/a")
    monkeypatch.setattr("hyperphase.noise.run_pandexo", lambda *a, **k: pytest.fail("PandExo ran instead of the cache"))
    assert rms_1min(toi2431b, "MIRI/LRS", verbose=False) == pytest.approx(62.93, abs=0.01)
    with pytest.raises(ImportError, match="PandExo is not installed"):
        rms_1min(toi2431b, "MIRI/LRS", lrs_range=(6e-6, 9e-6), verbose=False)   # a band the cache does not hold


def test_missing_shipped_cache_runs_pandexo(toi2431b, empty_user_cache, tmp_path, monkeypatch):
    """Without the shipped cache (and an empty user cache) every noise level is a new PandExo run."""
    class Ran(Exception): pass
    def run_pandexo(*a, **k): raise Ran
    monkeypatch.setattr("hyperphase.noise.PANDEXO_CACHE", tmp_path / "deleted.csv")
    monkeypatch.setattr("hyperphase.noise._package_version", lambda dist: "2026.7")
    monkeypatch.setattr("hyperphase.noise.run_pandexo", run_pandexo)
    with pytest.raises(Ran):
        rms_1min(toi2431b, "MIRI/LRS", verbose=False)


@pytest.mark.parametrize("name", sorted(CUSTOM))
def test_shipped_cache_covers_the_curated_planets(name, empty_user_cache, monkeypatch):
    """Every curated planet has its PandExo noise in the shipped cache, in NRS1, NRS2 and MIRI/LRS where it is observable, so the tables
    need no PandExo run."""
    if _package_version("pandexo.engine") != "2026.7" or _package_version("pandeia.engine") != "2026.7":
        pytest.skip("the cache holds PandExo/Pandeia 2026.7 values")
    monkeypatch.setattr("hyperphase.noise.run_pandexo", lambda *a, **k: pytest.fail(f"PandExo ran for {name}"))
    planet = hi_planets_by_name(name)
    for band in ("JWST_NIRSpec.G395H_NRS1", "JWST_NIRSpec.G395H_NRS2", "MIRI/LRS"):
        if observable(name, band): assert rms_1min(planet, band, verbose=False) > 0., band


def test_noise_levels_without_pandexo(toi2431b):
    """MIRI imaging takes its noise from CUSTOM_SIGMA_1MIN (TOI-2431 b only); 55 Cnc e saturates NIRSpec/G395H."""
    assert rms_1min(toi2431b, "MIRI/F1280W") > 0.
    with pytest.raises(NoNoiseLevelError):
        rms_1min(hi_planets_by_name("K2-141 b"), "MIRI/F1280W")
    with pytest.raises(NoNoiseLevelError):
        rms_1min(hi_planets_by_name("55 Cnc e"), "JWST_NIRSpec.G395H_NRS1")


def test_a_planet_the_archive_lacks_has_no_noise_level(toi2431b):
    """The PandExo inputs are read by name, so a planet of one's own raises NoNoiseLevelError, which the tables and run_dynesty.py handle."""
    with pytest.raises(NoNoiseLevelError, match="sigma_1min"):
        rms_1min(replace(toi2431b, name="My planet"), "MIRI/LRS")


def test_paper_runs_define_every_retrieval(root):
    """retrievals/paper_runs/make_runs.py: 132 run files with 249 datasets, each written from run_dynesty.py without a missing input line and
    valid Python (the files are generated by regular expressions, so a changed line of run_dynesty.py could break them silently); the null tests
    draw their data from the point-source model."""
    spec = importlib.util.spec_from_file_location("make_runs", root / "retrievals" / "paper_runs" / "make_runs.py")
    make_runs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_runs)
    assert len(make_runs.RUNS) == 132
    assert sum(len(settings["instruments"]) for _, settings in make_runs.RUNS.values()) == 249
    for stem, (description, settings) in make_runs.RUNS.items():
        src = make_runs.make_run(description, **settings)
        compile(src, stem + ".py", "exec")
        if settings.get("truth"):   # run_dynesty.py saves it to null_test/ (run_subfolder)
            assert 'INSOL_TRUTH = "naive"' in src, stem
