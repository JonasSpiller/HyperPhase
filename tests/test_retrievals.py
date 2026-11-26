"""Simulated observations and the stored nested-sampling retrievals (Section 3.4 and Appendix E)."""
import pickle
from pathlib import Path

import numpy as np
import pytest

from hyperphase.bias.detectors import detector_cfgs
from hyperphase.retrieval.simulate import simulate_data
from hyperphase.retrieval.storage import check_fit_current, load_fit, load_fits, logl_from_fit

FIXED_GEOMETRY = ["TOI-2431b_MIRI-LRS_*.pkl", "TOI-2431b_JWST_NIRSpec*.pkl"]
FREE_GEOMETRY = ["free_geometry/TOI-2431b_MIRI-LRS_*.pkl"]


def example_fits(root, pattern):
    fits = load_fits(str(root / "retrievals" / "example_runs" / pattern))
    assert set(fits) == {"hyperillum", "naive", "naive_redist"}
    return fits


def best_sample(fit):
    i = np.argmax(fit.results.logl)
    return fit.results.samples[i], fit.results.logl[i]


def test_simulated_noise_is_reproducible(cfg_lrs):
    """The noise of a simulated observation is sigma_bin times default_rng(seed).normal, so every run can be regenerated from its seed."""
    data = simulate_data(cfg_lrs, 0.2, seed=7, plotting=False)
    noiseless = simulate_data(cfg_lrs, 0.2, seed=7, noise=False, plotting=False)
    z = (data["y_obs"] - noiseless["y_obs"]) / data["sigma"]
    assert np.allclose(z, np.random.default_rng(7).normal(0., 1., z.size), rtol=0., atol=1e-10)


@pytest.mark.parametrize("pattern", FIXED_GEOMETRY)
def test_example_runs_reproduce_their_likelihood(root, pattern, tmp_path, monkeypatch):
    """The shipped retrievals load on any machine (their filter paths move into this repository), and for every model the likelihood the current
    code rebuilds from the pickle reproduces the stored best fit. The pickle carries the stellar grid of the run, so this needs no NewEra cube:
    without it check_fit_current skips the comparison with today's grid (NaN), which test_example_runs_are_current makes with the cube."""
    monkeypatch.setattr("hyperphase.retrieval.storage.CUBE_FLUX", tmp_path / "no_cube.npy")   # as on a machine without the NewEra cube
    for model, fit in example_fits(root, pattern).items():
        assert np.isnan(check_fit_current(fit)), model
        theta, logl = best_sample(fit)
        assert logl_from_fit(fit)(theta) == pytest.approx(logl, abs=1e-6), model


@pytest.mark.parametrize("pattern", FIXED_GEOMETRY + FREE_GEOMETRY)
def test_example_runs_are_current(root, newera_cube, pattern):
    """With the NewEra cube: every shipped retrieval was run with the stellar grid the current code builds (check_fit_current = 0). The
    free-geometry runs rebuild the stellar grid at every sampled geometry, so their likelihood is checked here too."""
    for model, fit in example_fits(root, pattern).items():
        assert check_fit_current(fit) == 0., model
        if pattern in FREE_GEOMETRY:
            theta, logl = best_sample(fit)
            assert logl_from_fit(fit)(theta) == pytest.approx(logl, abs=1e-6), model


def test_runs_from_another_machine_find_their_filters(root, tmp_path):
    """A pickle stores the absolute filter path of the machine that ran it; loading it elsewhere points the path into this repository."""
    source = sorted((root / "retrievals" / "example_runs").glob("TOI-2431b_JWST_NIRSpec*naive_priors*.pkl"))[0]
    fit = load_fit(source)   # also maps the classes of a run saved before the package existed
    for cfg in detector_cfgs(fit.meta["cfg"]):
        cfg.filter_path = "/another/machine/HyperPhase/inputs/filters/Spiderman/" + Path(cfg.filter_path).name
    moved = tmp_path / source.name
    with open(moved, "wb") as f:
        pickle.dump(fit, f)
    for cfg in detector_cfgs(load_fit(moved).meta["cfg"]):
        assert Path(cfg.filter_path).parent == root / "inputs" / "filters" / "Spiderman"
        assert Path(cfg.filter_path).exists()


def test_runs_with_another_spelling_of_the_filter_path_take_the_canonical_one(root, tmp_path):
    """A stored filter path that reaches the repository's file under another spelling (a symlink here, or another case on a case-insensitive disk)
    is replaced by the canonical path, which observed_band looks the instrument up by."""
    source = sorted((root / "retrievals" / "example_runs").glob("TOI-2431b_JWST_NIRSpec*naive_priors*.pkl"))[0]
    fit = load_fit(source)   # also maps the classes of a run saved before the package existed
    link = tmp_path / "Spiderman"
    link.symlink_to(root / "inputs" / "filters" / "Spiderman", target_is_directory=True)
    for cfg in detector_cfgs(fit.meta["cfg"]):
        cfg.filter_path = str(link / Path(cfg.filter_path).name)
    moved = tmp_path / source.name
    with open(moved, "wb") as f:
        pickle.dump(fit, f)
    for cfg in detector_cfgs(load_fit(moved).meta["cfg"]):
        assert Path(cfg.filter_path).parent == root / "inputs" / "filters" / "Spiderman"
