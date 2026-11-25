"""Surface temperatures and heat redistribution (Eqs. 6, 8, 9)."""
import numpy as np
import pytest

from hyperphase.irradiance import (BBtemp_from_insolation, effective_insolation, insolation, naive_insolation, sigma,
                                   surface_temperature)


def test_energy_balance_and_emissivity_degeneracy():
    """emissivity_bol sigma T^4 = (1 - A_B) I (Eq. 6), so only (1 - A_B) / emissivity_bol enters the temperature."""
    I = 1e6   # W m^-2
    T = BBtemp_from_insolation(I, A_B=0.2, emissivity_bol=0.8)
    assert 0.8 * sigma * T**4 == pytest.approx((1. - 0.2) * I)
    assert T == pytest.approx(BBtemp_from_insolation(I, A_B=0., emissivity_bol=1.))


def test_extended_source_redistribution_without_redistribution(toi2431b):
    """The extended star with heat redistribution F (Eq. 9) is the plain extended star at F = 0."""
    gamma = np.linspace(0., np.pi, 50)
    assert np.array_equal(surface_temperature(gamma, toi2431b, "hyperillum"),
                          surface_temperature(gamma, toi2431b, "hyperillum_redist", F=0.))


@pytest.mark.parametrize("model", ["naive_redist", "hyperillum_redist"])
def test_redistribution_conserves_energy(toi2431b, model):
    """Redistribution moves absorbed power to the nightside but keeps its total: the sphere means of the redistributed and the incident
    irradiance agree."""
    gamma = np.linspace(0., np.pi, 20001)
    incident = insolation(gamma, toi2431b, model)
    redistributed = effective_insolation(incident, toi2431b, model, F=0.2)
    weight = np.sin(gamma)
    assert np.trapezoid(redistributed * weight, gamma) == pytest.approx(np.trapezoid(incident * weight, gamma), rel=1e-6)


def test_full_redistribution_is_uniform(toi2431b):
    """F = 0.5 spreads the absorbed power evenly (Eq. 8): S/4 everywhere for the point source."""
    gamma = np.linspace(0., np.pi, 200)
    redistributed = effective_insolation(insolation(gamma, toi2431b, "naive_redist"), toi2431b, "naive_redist", F=0.5)
    assert np.allclose(redistributed, float(naive_insolation(0., toi2431b)) / 4., rtol=1e-12)
