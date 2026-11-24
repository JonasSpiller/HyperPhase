"""Simulated observations and the stored nested-sampling retrievals (Section 3.4 and Appendix E)."""
import pickle
from pathlib import Path

import numpy as np
import pytest

import dynesty_utils as du


def test_simulated_noise_is_reproducible(cfg_lrs):
    """The noise of a simulated observation is sigma_bin times default_rng(seed).normal, so every run can be regenerated from its seed."""
    data = du.simulate_data(cfg_lrs, 0.2, seed=7, plotting=False)
    noiseless = du.simulate_data(cfg_lrs, 0.2, seed=7, noise=False, plotting=False)
    z = (data["y_obs"] - noiseless["y_obs"]) / data["sigma"]
    assert np.allclose(z, np.random.default_rng(7).normal(0., 1., z.size), rtol=0., atol=1e-10)


@pytest.mark.parametrize("pattern", ["TOI-2431b_MIRI-LRS_*.pkl", "TOI-2431b_JWST_NIRSpec*.pkl", "free_geometry/TOI-2431b_MIRI-LRS_*.pkl"])
def test_example_runs_reproduce_their_likelihood(root, newera_cube, pattern):
    """The shipped retrievals load on any machine (their filter paths move into this repository), were run with the current forward model
    (check_fit_current = 0), and their rebuilt likelihood reproduces the stored best fit."""
    fits = du.load_fits(str(root / "Dynesty Fits" / "example_runs" / pattern))
    assert set(fits) == {"hyperillum", "naive", "naive_redist"}
    for fit in fits.values():
        assert du.check_fit_current(fit) == 0.
    fit = fits["naive"]
    best = fit.results.samples[np.argmax(fit.results.logl)]
    assert du.logl_from_fit(fit)(best) == pytest.approx(fit.results.logl.max(), abs=1e-6)


def test_runs_from_another_machine_find_their_filters(root, tmp_path):
    """A pickle stores the absolute filter path of the machine that ran it; loading it elsewhere points the path into this repository."""
    source = sorted((root / "Dynesty Fits" / "example_runs").glob("TOI-2431b_JWST_NIRSpec*naive_priors*.pkl"))[0]
    fit = pickle.load(open(source, "rb"))
    for cfg in du.detector_cfgs(fit.meta["cfg"]):
        cfg.filter_path = "/another/machine/HyperPhase/Filters/Spiderman/" + Path(cfg.filter_path).name
    moved = tmp_path / source.name
    pickle.dump(fit, open(moved, "wb"))
    for cfg in du.detector_cfgs(du.load_fit(moved).meta["cfg"]):
        assert Path(cfg.filter_path).parent == root / "Filters" / "Spiderman"
        assert Path(cfg.filter_path).exists()
