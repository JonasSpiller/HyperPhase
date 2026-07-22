import batman
import numpy as np
from kepler_utils import *

def light_travel_time_correction(t, planet, t0=0.):
    """
    Returns timestamps corrected for the light travel time delay.
    """
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    c = 2.998e8             # m/s
    seconds_per_day = 86400.

    a_phys = a_AU * 1.495978707e11  #physical semi-major axis in m
    f = true_anomaly(t, period, ecc, w, t0=t0)
    r = a_phys * (1. - ecc**2) / (1. + ecc * np.cos(f))  #orbital radius in m
    z = - r * np.sin(f + np.radians(w)) * np.sin(np.radians(incl))  #line-of-sight displacement in m
    dt = z / c / seconds_per_day  #delay in days

    return t + dt


def flux_occultation(planet, fp, t, t0=0., u=[0.1, 0.3]):
    """
    Returns the secondary eclipse / occultation light curve (normalized flux array) computed via batman.
    """
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    params_occultation = batman.TransitParams() # called params_eclipse in Batman tutorials

    params_occultation.t0 = t0  # time of transit center (units consistent with period)
    params_occultation.per = period  # orbital period
    params_occultation.rp = r1/r2  # planet radius (in units of stellar radii)
    params_occultation.a = R/r2  # semi-major axis (in units of stellar radii)
    params_occultation.inc = incl  # orbital inclination (in degrees)
    params_occultation.ecc = ecc  # eccentricity
    params_occultation.w = w  # longitude of periastron (in degrees)

    params_occultation.limb_dark = "quadratic"  # limb darkening model (assumed "quadratic", consistent with Hyperillumination model)
    params_occultation.u = u  # limb darkening coefficients [u1, u2] (placeholders, not from K2-141 data)

    params_occultation.fp = fp  # planet-to-star flux ratio (occultation depth)
    params_occultation.t_secondary = occultation_time(period, ecc, w, t0=t0)  # time of occultation center

    m_occultation = batman.TransitModel(params_occultation, t, transittype="secondary", max_err=0.001)  # initializes model

    return m_occultation.light_curve(params_occultation)


def flux_transit(planet, t, t0=0., u=[0.1, 0.3]):
    """
    Returns the primary transit light curve (normalized flux array) computed via batman.
    Limb darkening is nonlinear with coefficients [0.5, 0.1, 0.1, -0.1].
    """
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    params_transit = batman.TransitParams()

    params_transit.t0 = t0  # time of transit center (units consistent with period)
    params_transit.per = period  # orbital period
    params_transit.rp = r1/r2  # planet radius (in units of stellar radii)
    params_transit.a = R/r2  # semi-major axis (in units of stellar radii)
    params_transit.inc = incl  # orbital inclination (in degrees)
    params_transit.ecc = ecc  # eccentricity
    params_transit.w = w  # longitude of periastron (in degrees)

    params_transit.limb_dark = "quadratic"  # limb darkening model (assumed "quadratic", consistent with Hyperillumination model)
    params_transit.u = u  # limb darkening coefficients [u1, u2] (placeholders, not from K2-141 data)

    m_transit = batman.TransitModel(params_transit, t, max_err=0.001)  # initializes model

    return m_transit.light_curve(params_transit)


def Cosine_PhaseCurve(planet, fp, offset=0, t=None, t0=0., Amp=1., u=[0.1, 0.3]):
    """
    Returns (t, flux): timestamps and the full phase curve combining transit, secondary eclipse / occultation, and sinusoidal phase modulation.
    Phase modulation is computed from the true anomaly (correct for eccentric orbits); offset shifts the atmospheric hotspot in degrees.
    """
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    if t is None: t = np.linspace(0., period, 10000)

    f = true_anomaly(t, period, ecc, w, t0=t0)
    f_occultation = 3. * np.pi / 2. - np.radians(w)
    phase_mod = 1 + Amp/2 * (np.cos(f - f_occultation + np.radians(offset)) - 1)

    t_corr = light_travel_time_correction(t, planet, t0=t0)
    phase_curve = flux_transit(planet, t, t0=t0, u=u)+(flux_occultation(planet, fp, t, t0=t0, u=u)-1)*phase_mod

    return t_corr, phase_curve
