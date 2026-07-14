import batman
import numpy as np
from kepler_utils import *

def light_travel_time_correction(t, period, ecc, w, incl, a, R_star, t0=0.):
    """
    Returns timestamps corrected for the light travel time delay.
    a is in stellar radii, R_star in solar radii, w in degrees.
    """
    R_sun = 6.957e8         # m
    c = 2.998e8             # m/s
    seconds_per_day = 86400.

    a_phys = a * R_star * R_sun  #physical semi-major axis in m
    f = true_anomaly(t, period, ecc, w, t0=t0)
    r = a_phys * (1. - ecc**2) / (1. + ecc * np.cos(f))  #orbital radius in m
    z = - r * np.sin(f + np.radians(w)) * np.sin(np.radians(incl))  #line-of-sight displacement in m
    dt = z / c / seconds_per_day  #delay in days

    return t + dt


def flux_eclipse(period, rp, a, incl, ecc, w, fp, t, t0=0.):
    """
    Returns the secondary eclipse light curve (normalized flux array) computed via batman.
    """
    params_eclipse = batman.TransitParams()

    params_eclipse.t0 = t0  # time of transit center
    params_eclipse.per = period  # orbital period
    params_eclipse.rp = rp  # planet radius (in units of stellar radii)
    params_eclipse.a = a  # semi-major axis (in units of stellar radii)
    params_eclipse.inc = incl  # orbital inclination (in degrees)
    params_eclipse.ecc = ecc  # eccentricity
    params_eclipse.w = w  # longitude of periastron (in degrees)

    params_eclipse.limb_dark = "nonlinear"  # limb darkening model
    params_eclipse.u = [0.5, 0.1, 0.1, -0.1]  # limb darkening coefficients [u1, u2, u3, u4]

    params_eclipse.fp = fp  # planet-to-star flux ratio (eclipse depth)
    params_eclipse.t_secondary = eclipse_time(period, ecc, w, t0=t0)  # time of secondary eclipse center

    m_eclipse = batman.TransitModel(params_eclipse, t, transittype="secondary", max_err=0.001)  # initializes model

    return m_eclipse.light_curve(params_eclipse)


def flux_transit(period, rp, a, incl, ecc, w, t, t0=0.):
    """
    Returns the primary transit light curve (normalized flux array) computed via batman.
    Limb darkening is nonlinear with coefficients [0.5, 0.1, 0.1, -0.1].
    """
    params_transit = batman.TransitParams()

    params_transit.t0 = t0  # time of transit center
    params_transit.per = period  # orbital period
    params_transit.rp = rp  # planet radius (in units of stellar radii)
    params_transit.a = a  # semi-major axis (in units of stellar radii)
    params_transit.inc = incl  # orbital inclination (in degrees)
    params_transit.ecc = ecc  # eccentricity
    params_transit.w = w  # longitude of periastron (in degrees)

    params_transit.limb_dark = "nonlinear"  # limb darkening model
    params_transit.u = [0.5, 0.1, 0.1, -0.1]  # limb darkening coefficients [u1, u2, u3, u4]

    m_transit = batman.TransitModel(params_transit, t, max_err=0.001)  # initializes model

    return m_transit.light_curve(params_transit)


def PhaseCurve(period, rp, a, incl, ecc, w, fp, R_star, offset=0, t=[], t0=0., A=1.):
    """
    Returns (t, flux): timestamps and the full phase curve combining transit, secondary eclipse, and sinusoidal phase modulation.
    Phase modulation is computed from the true anomaly (correct for eccentric orbits); offset shifts the atmospheric hotspot in degrees.
    """
    if len(t) == 0: t = np.linspace(t0 - period / 4, t0 + period * 3 / 4, 10000)

    f = true_anomaly(t, period, ecc, w, t0=t0)
    f_eclipse = 3. * np.pi / 2. - np.radians(w)
    phase_mod = 1 + A/2 * (np.cos(f - f_eclipse + np.radians(offset)) - 1)

    t_corr = light_travel_time_correction(t, period, ecc, w, incl, a, R_star, t0=t0)
    phase_curve = flux_transit(period, rp, a, incl, ecc, w, t, t0=t0)+(flux_eclipse(period, rp, a, incl, ecc, w, fp, t, t0=t0)-1)*phase_mod

    return t_corr, phase_curve
