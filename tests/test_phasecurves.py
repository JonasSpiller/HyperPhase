"""Thermal and reflected phase curves with fSPIDERMAN and batman (Section 3.1 and Appendix B)."""
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
from scipy.constants import c, h, k

from hyperphase.stellar import T_bright
from hyperphase.phasecurve import full_PhaseCurve, Louden_PhaseCurve, nightside_flux, reflected_PhaseCurve, thermal_PhaseCurve
from hyperphase.irradiance import surface_temperature


def thermal(planet, band, model, **kw):
    (l1, l2), path = band
    return thermal_PhaseCurve(planet, (l1, l2), filter_path=path, insolation_model=model, stellar_model="blackbody", **kw)


def band_radiance(T, band):
    """Photon-counting band radiance of a blackbody, the integral of B_lambda(T) R(lambda) lambda over the band, with the throughput R of the filter
    file (electrons per photon): written out here, independently of SPIDERMAN and hyperphase.stellar."""
    (l1, l2), path = band
    lam_file, throughput = np.loadtxt(path).T
    lam = np.linspace(l1, l2, 2001)
    T = np.atleast_1d(T)[:, None]
    with np.errstate(over="ignore", divide="ignore"):
        B = 2. * h * c**2 / lam**5 / np.expm1(h * c / (lam * k * T))   # 0 at T = 0
    return np.trapezoid(B * np.interp(lam, lam_file, throughput) * lam, lam, axis=1)


def disk_flux(planet, band, model, hemisphere, **kw):
    """F_p/F_star [ppm] of the hemisphere centred on the substellar ("day") or antistellar ("night") point, seen edge-on along the star-planet axis:
    (R_p/R*)^2 2 int B(T(gamma)) |cos gamma| sin gamma dgamma / B(T_eff), with the surface temperature T(gamma) of hyperphase.irradiance."""
    gamma = np.linspace(0., np.pi / 2., 1001) if hemisphere == "day" else np.linspace(np.pi / 2., np.pi, 1001)
    B = band_radiance(surface_temperature(gamma, planet, model, **kw), band)
    integral = 2. * np.trapezoid(B * np.abs(np.cos(gamma)) * np.sin(gamma), gamma)
    return 1e6 * (planet.r1 / planet.r2)**2 * integral / band_radiance(planet.T2_eff, band)[0]


@pytest.mark.parametrize("model", ["naive", "hyperillum"])
def test_disk_integrated_flux_matches_direct_integration(toi2431b, lrs, model):
    """SPIDERMAN's occultation depth and mid-transit nightside flux against the disk integral of the temperature map, edge-on. Catches a rotated or
    transposed direct_T grid (which fails silently), a wrong stellar normalisation and a wrong band weighting. The penumbra is a thin ring along
    the limb at mid-transit, so the nightside needs many more radial layers than the default of thermal_PhaseCurve (40 layers: about 3 % too faint)."""
    planet = replace(toi2431b, incl=90.)
    kw = dict(A_B=0.2, emissivity_bol=1., u_bol=0.5)
    _, _, depth, _ = thermal(planet, lrs, model, emissivity=1., include_transit=False, **kw)
    assert depth == pytest.approx(disk_flux(planet, lrs, model, "day", **kw), rel=2e-3)
    night, _ = nightside_flux(planet, lrs[0], model, filter_path=lrs[1], stellar_model="blackbody", emissivity=1., N_LAYERS=400, **kw)
    if model == "naive":
        assert night == 0.   # the point-source nightside is at 0 K
    else:
        assert night == pytest.approx(disk_flux(planet, lrs, model, "night", **kw), rel=5e-3)


def test_point_source_without_redistribution(toi2431b, lrs):
    """naive_redist at F = 0 (SPIDERMAN's analytic Louden map) reproduces naive (a tabulated temperature map)."""
    _, naive, depth_naive, _ = thermal(toi2431b, lrs, "naive", include_transit=False)
    _, louden, depth_louden, _ = thermal(toi2431b, lrs, "naive_redist", F=0., include_transit=False)
    assert 1e6 * np.max(np.abs(naive - louden)) < 0.05   # ppm
    assert depth_louden == pytest.approx(depth_naive, abs=0.05)


@pytest.mark.parametrize("model", ["hyperillum", "naive"])
def test_fast_lightcurve_matches_direct_evaluation(toi2431b, lrs, model):
    """fast_lightcurve interpolates the phase modulation and skips the totality; the direct evaluation calls SPIDERMAN at every time."""
    _, fast, _, _ = thermal(toi2431b, lrs, model)
    _, direct, _, _ = thermal(toi2431b, lrs, model, spman="normal")
    assert 1e6 * np.max(np.abs(fast - direct)) < 0.1   # ppm


def test_reflected_light_is_linear_in_albedo(toi2431b):
    t = np.linspace(0., toi2431b.period, 400, endpoint=False)
    _, curve_1, depth_1 = reflected_PhaseCurve(toi2431b, t=t, A=0.1, include_transit=False)
    _, curve_2, depth_2 = reflected_PhaseCurve(toi2431b, t=t, A=0.2, include_transit=False)
    assert np.allclose(2. * curve_1, curve_2, rtol=0., atol=1e-9 * np.max(curve_2))
    assert depth_2 == pytest.approx(2. * depth_1, rel=1e-12)


def test_totality_is_the_star_alone(toi2431b, lrs):
    """While the planet is fully hidden behind the star, the flux is exactly that of the star."""
    t, flux, _ = full_PhaseCurve(toi2431b, lrs[0], filter_path=lrs[1], stellar_model="blackbody")
    totality = np.abs(t / toi2431b.period - 0.5) < 0.02
    assert totality.any() and np.all(flux[totality] == 1.)


def test_brightness_temperature_inverts_the_occultation_depth(toi2431b, lrs):
    """With full redistribution (F = 0.5) the planet has one temperature, which T_bright must recover from the occultation depth."""
    A_B = 0.2
    _, _, depth, _ = Louden_PhaseCurve(toi2431b, lrs[0], F=0.5, A_B=A_B, filter_path=lrs[1], stellar_model="blackbody", include_transit=False)
    T_uniform = toi2431b.T2_eff * np.sqrt(toi2431b.r2 / toi2431b.R) * ((1. - A_B) / 4.) ** 0.25
    assert T_bright(depth, toi2431b, lrs[0], lrs[1], stellar_model="blackbody") == pytest.approx(T_uniform, abs=0.5)


def test_extended_star_brightens_the_planet(toi2431b, lrs):
    """The extended star heats the planet beyond the point-source terminator: the occultation is deeper and the nightside at transit brighter."""
    kw = dict(filter_path=lrs[1], stellar_model="blackbody")
    depth = {m: full_PhaseCurve(toi2431b, lrs[0], insolation_model=m, **kw)[2] for m in ("hyperillum", "naive")}
    night = {m: nightside_flux(toi2431b, lrs[0], insolation_model=m, **kw)[0] for m in ("hyperillum", "naive")}
    assert depth["hyperillum"] > depth["naive"]
    assert night["hyperillum"] > night["naive"]


INSIDE_THE_STAR = """
import sys
sys.path.insert(0, sys.argv[1])
from dataclasses import replace
from hyperphase.bands import filter_lambda, filter_loc
from hyperphase.phasecurve import thermal_PhaseCurve
from hyperphase.targets import hi_planets_by_name
planet = hi_planets_by_name("TOI-2431 b")
planet = replace(planet, R=1.01 * planet.r2)   # a/R* = 1.01 < 1 + R_p/R* = 1.02
thermal_PhaseCurve(planet, filter_lambda("MIRI/LRS"), filter_path=filter_loc("MIRI/LRS"), stellar_model="blackbody")
"""


def test_planet_inside_its_star_raises(root):
    """A planet inside its star raises ValueError (fSPIDERMAN patch 7). Upstream SPIDERMAN and fSPIDERMAN builds before 81d300d end the process with
    exit status 0 instead, which would stop this test run without a summary and report it as passed: so the call runs in a process of its own."""
    run = subprocess.run([sys.executable, "-c", INSIDE_THE_STAR, str(root)], cwd=root, capture_output=True, text=True, timeout=300)
    error = run.stderr.strip().splitlines()[-1] if run.stderr.strip() else ""
    assert error.startswith("ValueError") and "inside the star" in error, \
        "fspiderman ends the process instead of raising: install fSPIDERMAN 81d300d or later (environment.yml)\n" + run.stdout[-2000:] + run.stderr[-2000:]
