"""The irradiance of a planet by its extended host star (paper, Section 2 and Appendix A)."""
from dataclasses import replace

import numpy as np
import pytest

from hyperphase.irradiance import insolation, insolation_hyperillum, insolation_penumbra_hyperillum, mean_insolation, naive_insolation
from hyperphase.validation import insolation_hyperillum_bruteforce, insolation_penumbra_hyperillum_simpleint


def substellar(planet):
    """The point-source irradiance at the substellar point, S = (R*/a)^2 sigma T_eff^4."""
    return float(naive_insolation(0., planet))


def test_absorbed_power_matches_eq5(toi2431b):
    """The power the planet intercepts from the extended star relative to a point source, P_hi/P_ps = 2 / (1 + sqrt(1 - R*^2/a^2)) (Eq. 5)."""
    a_rs = toi2431b.R / toi2431b.r2
    ratio = mean_insolation(toi2431b, "hyperillum", u_bol=0.) / mean_insolation(toi2431b, "naive")
    assert ratio == pytest.approx(2. / (1. + np.sqrt(1. - 1. / a_rs**2)), rel=1e-4)


def test_point_source_limit(toi2431b):
    """Far from its star the planet sees a point source: on the dayside the two models agree up to the finite planet radius (R_p/a ~ 1e-4)."""
    far = replace(toi2431b, R=200. * toi2431b.r2)
    gamma = np.radians(np.linspace(0., 80., 9))
    assert np.allclose(insolation(gamma, far, "hyperillum"), insolation(gamma, far, "naive"), rtol=1e-3)


def test_gauss_legendre_matches_adaptive_quadrature(toi2431b):
    """The vectorized Gauss-Legendre penumbra (Eqs. A7, A9, A10) against scipy.integrate.quad per point."""
    gamma = np.linspace(toi2431b.terminator_penumbra() + 1e-4, toi2431b.terminator_nightside() - 1e-4, 12)
    fast = insolation_penumbra_hyperillum(gamma, toi2431b, u_bol=0.5)
    slow = insolation_penumbra_hyperillum_simpleint(gamma, toi2431b, u_bol=0.5)
    assert np.max(np.abs(fast - slow)) < 1e-5 * substellar(toi2431b)


def test_matches_brute_force_integration(toi2431b):
    """An independent check: integration over a grid of the stellar surface, without the geometry of Appendix A."""
    gamma = np.radians([60., 95., 105.])
    brute = insolation_hyperillum_bruteforce(gamma, toi2431b, u_bol=0.5, n_theta=600, n_phi=1200)
    assert np.max(np.abs(brute - insolation_hyperillum(gamma, toi2431b, u_bol=0.5))) < 1e-5 * substellar(toi2431b)


def test_zone_boundaries(toi2431b):
    """Continuous at the edge of the fully illuminated zone (Eq. 1), zero at the terminator and on the nightside."""
    edge, terminator = toi2431b.terminator_penumbra(), toi2431b.terminator_nightside()
    inside, outside = insolation_hyperillum(np.array([edge - 1e-7, edge + 1e-7]), toi2431b, u_bol=0.5)
    assert outside == pytest.approx(inside, rel=1e-5)
    night = insolation_hyperillum(np.array([terminator - 1e-7, terminator, np.pi]), toi2431b, u_bol=0.5)
    assert np.all(np.abs(night) < 1e-12 * substellar(toi2431b))


def test_limb_darkening_acts_only_in_the_penumbra(toi2431b):
    """In the fully illuminated zone the irradiance does not depend on the limb darkening (Appendix A); in the penumbra limb darkening lowers it."""
    full = np.linspace(0., toi2431b.terminator_penumbra() - 1e-6, 20)
    assert np.array_equal(insolation(full, toi2431b, "hyperillum", u_bol=0.), insolation(full, toi2431b, "hyperillum", u_bol=0.6))
    penumbra = np.linspace(toi2431b.terminator_penumbra() + 1e-2, toi2431b.terminator_nightside() - 1e-2, 10)
    assert np.all(insolation(penumbra, toi2431b, "hyperillum", u_bol=0.6) < insolation(penumbra, toi2431b, "hyperillum", u_bol=0.))
