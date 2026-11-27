"""The noiseless fits of the bias analysis (Section 3.3)."""
import numpy as np
import pytest

from hyperphase.model import insolation_cache, model_pcurve, reflected_unit
from hyperphase.bias.fisher import fit_truth
from hyperphase.bias.fits import chi2_landscape, fit_AB_only, fit_naive, fit_redist, truth_curves


def test_point_source_fit_recovers_its_own_truth(cfg_lrs):
    truth = model_pcurve(cfg_lrs, "naive", 0.2)[1]
    A_B, chi2, _ = fit_AB_only(cfg_lrs, truth, "naive")
    assert A_B == pytest.approx(0.2, abs=1e-3)
    assert chi2 < 1e-4


def test_point_source_fit_underestimates_the_albedo(cfg_lrs):
    """The paper's result for TOI-2431 b: fitted to the extended-source truth, the point-source model returns a lower Bond albedo."""
    A_B, chi2, _ = fit_naive(cfg_lrs, truth_curves(cfg_lrs, 0.2))
    assert A_B < 0.2 - 0.02
    assert chi2 > 1.


@pytest.mark.slow
def test_redistribution_fit_finds_the_grid_minimum(cfg_lrs):
    """With redistribution the point-source fit returns a spurious F > 0. fit_redist profiles A_B and brackets F, because a local optimizer
    started at F = 0 stays there; its minimum must be the minimum of a brute-force chi2 grid around it."""
    truth = truth_curves(cfg_lrs, 0.2)
    A_B, F, chi2, _ = fit_redist(cfg_lrs, truth)
    assert F > 0.
    grid = chi2_landscape(cfg_lrs, truth, A_B + np.linspace(-0.05, 0.05, 11), np.linspace(0., 2. * F, 11))
    assert chi2 <= grid.min() + 1e-6


def test_geometry_caches_are_exact_and_refuse_another_setup(cfg_lrs):
    """insol and refl_unit reproduce the uncached curve bit for bit, serve the redistribution model of the same source, and are refused for
    another source or geometry; the truth of fit_truth builds its own instead of taking the fitted model's."""
    y = model_pcurve(cfg_lrs, "hyperillum", 0.2)[1]
    caches = dict(insol=insolation_cache(cfg_lrs, "hyperillum"), refl_unit=reflected_unit(cfg_lrs, "hyperillum"))
    assert np.array_equal(model_pcurve(cfg_lrs, "hyperillum", 0.2, **caches)[1], y)
    assert np.array_equal(model_pcurve(cfg_lrs, "hyperillum_redist", 0.2, F=0.1, **caches)[1], model_pcurve(cfg_lrs, "hyperillum_redist", 0.2, F=0.1)[1])
    for wrong in (dict(insol=insolation_cache(cfg_lrs, "naive")), dict(insol=insolation_cache(cfg_lrs, "hyperillum", a_rs=3.)),
                  dict(refl_unit=reflected_unit(cfg_lrs, "naive")), dict(refl_unit=reflected_unit(cfg_lrs, "hyperillum", rp_rs=0.03))):
        with pytest.raises(ValueError, match="was built for"):
            model_pcurve(cfg_lrs, "hyperillum", 0.2, **wrong)
    assert fit_truth(cfg_lrs, 0.2, refl_unit=reflected_unit(cfg_lrs, "naive")) == fit_truth(cfg_lrs, 0.2)
