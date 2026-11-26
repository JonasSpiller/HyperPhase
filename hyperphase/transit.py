"""Transits and occultations with batman, and the light-travel-time correction."""

import batman
import numpy as np
from astropy import constants as const

from hyperphase.defaults import DEFAULT_U
from hyperphase.orbit import occultation_time, true_anomaly


c = const.c.value # m/s
seconds_per_day = 86400.
au = const.au.value


def light_travel_time_correction(t, planet, t0=0.):
    """The times t [days] shifted by the light travel time across the orbit (t0: mid-transit)."""
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    a_phys = a_AU * au  #physical semi-major axis in m
    f = true_anomaly(t, period, ecc, w, t0=t0)
    r = a_phys * (1. - ecc**2) / (1. + ecc * np.cos(f))  #orbital radius in m
    z = - r * np.sin(f + np.radians(w)) * np.sin(np.radians(incl))  #line-of-sight displacement in m
    dt = z / c / seconds_per_day  #delay in days

    return t + dt


def flux_occultation(planet, fp, t, t0=0.):
    """Occultation light curve (normalised flux) from batman, for the planet-to-star flux ratio fp (the occultation depth)."""
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    params_occultation = batman.TransitParams()

    params_occultation.t0 = t0  # time of mid-transit [days, same units as period]
    params_occultation.per = period
    params_occultation.rp = r1/r2
    params_occultation.a = R/r2
    params_occultation.inc = incl  # [deg]
    params_occultation.ecc = ecc
    params_occultation.w = w  # argument of periastron [deg]

    params_occultation.limb_dark = "quadratic"  # required by batman, irrelevant for the occultation
    params_occultation.u = DEFAULT_U

    params_occultation.fp = fp
    params_occultation.t_secondary = occultation_time(period, ecc, w, t0=t0)  # time of mid-occultation

    m_occultation = batman.TransitModel(params_occultation, t, transittype="secondary", max_err=0.001)

    return m_occultation.light_curve(params_occultation)


def flux_transit(planet, t, t0=0., u=DEFAULT_U):
    """Transit light curve (normalised flux) from batman, with quadratic stellar limb darkening u = (u1, u2)."""
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    params_transit = batman.TransitParams()

    params_transit.t0 = t0  # time of mid-transit [days, same units as period]
    params_transit.per = period
    params_transit.rp = r1/r2
    params_transit.a = R/r2
    params_transit.inc = incl  # [deg]
    params_transit.ecc = ecc
    params_transit.w = w  # argument of periastron [deg]

    params_transit.limb_dark = "quadratic"
    params_transit.u = u

    m_transit = batman.TransitModel(params_transit, t, max_err=0.001)

    return m_transit.light_curve(params_transit)


def Cosine_PhaseCurve(planet, fp, offset=0, t=None, t0=0., Amp=1., u=DEFAULT_U):
    """Returns (t, flux): the times corrected for the light travel time, and a phase curve of the transit, the occultation and a sinusoidal phase
    modulation. The modulation follows the true anomaly (valid for eccentric orbits); offset shifts the hotspot [deg]."""
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    if t is None: t = np.linspace(0., period, 10000)

    f = true_anomaly(t, period, ecc, w, t0=t0)
    f_occultation = 3. * np.pi / 2. - np.radians(w)
    phase_mod = 1 + Amp/2 * (np.cos(f - f_occultation + np.radians(offset)) - 1)

    t_corr = light_travel_time_correction(t, planet, t0=t0)
    phase_curve = flux_transit(planet, t, t0=t0, u=u)+(flux_occultation(planet, fp, t, t0=t0)-1)*phase_mod

    return t_corr, phase_curve
