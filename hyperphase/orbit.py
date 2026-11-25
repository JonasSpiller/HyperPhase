"""Kepler orbits: anomalies, occultation times and transit durations."""

import numpy as np
from astropy import constants as const
from scipy.optimize import newton


au = const.au.value
R_sun = const.R_sun.value


# --------------------------------------------------------------------------
# Kepler
# --------------------------------------------------------------------------
def true_to_eccentric(f, ecc):
    """Returns the eccentric anomaly (radians) for a given true anomaly f (radians) and eccentricity."""
    return 2. * np.arctan(np.sqrt((1. - ecc) / (1. + ecc)) * np.tan(f / 2.))


def eccentric_to_mean(E, ecc):
    """Returns the mean anomaly (radians) for a given eccentric anomaly E (radians) and eccentricity."""
    return E - ecc * np.sin(E)


def mean_to_eccentric(M, ecc):
    """Returns the eccentric anomaly (radians) by solving Kepler's equation M = E - e*sin(E) via scipy.optimize.newton."""
    return newton(lambda E: E - ecc * np.sin(E) - M, M)


def eccentric_to_true(E, ecc):
    """Returns the true anomaly (radians) for a given eccentric anomaly E (radians) and eccentricity."""
    return 2. * np.arctan2(np.sqrt(1. + ecc) * np.sin(E / 2.), np.sqrt(1. - ecc) * np.cos(E / 2.))


def occultation_time(period, ecc, w, t0=0.):
    """
    Returns the time (same units as period) of secondary eclipse / occultation.
    Uses true-anomaly → eccentric anomaly → mean anomaly conversion (Kepler's equation).
    """
    w_rad = np.radians(w)
    f_transit = np.pi / 2. - w_rad
    f_occultation = 3. * np.pi / 2. - w_rad

    M_transit = eccentric_to_mean(true_to_eccentric(f_transit, ecc), ecc)
    M_occultation = eccentric_to_mean(true_to_eccentric(f_occultation, ecc), ecc)

    delta_M = (M_occultation - M_transit) % (2. * np.pi)
    t_ecl = delta_M / (2. * np.pi) * period + t0
    return t_ecl


def AU_to_RS(a, rs):
    """Returns the distance in solar radii for a given distance a in AU.
    a in AU, rs (stellar radius) in solar radii."""
    return a * (au / R_sun) / rs


def true_anomaly(t, period, ecc, w, t0=0.):
    """
    Returns the true anomaly (radians, in [0, 2π)) at each timestamp in t.
    t0 is the time of transit center.
    """
    w_rad = np.radians(w)
    f_transit = np.pi / 2. - w_rad
    M_transit = eccentric_to_mean(true_to_eccentric(f_transit, ecc), ecc)

    M = (M_transit + 2. * np.pi * (t - t0) / period) % (2. * np.pi)
    E = mean_to_eccentric(M, ecc)
    return eccentric_to_true(E, ecc) % (2. * np.pi)


def transit_duration_t14(period_days, a_over_r2, r1_over_r2, inclination_deg):
    """
    First to fourth transit contact T14 in hours (Winn 2010, eq. 14).
    NaN where the impact parameter exceeds 1 + r1/r2, i.e. the geometry does not actually transit.
    Assumes circular orbit
    """
    inc = np.radians(inclination_deg)
    b = a_over_r2 * np.cos(inc)
    arg = (1.0 + r1_over_r2) ** 2 - b ** 2
    root = np.sqrt(arg.where(arg > 0))
    sin_arg = (root / (a_over_r2 * np.sin(inc))).where(lambda s: s.abs() <= 1.0)
    return period_days / np.pi * np.arcsin(sin_arg) * 24.0


def roche_radius_over_r1(q, a_over_r2, r1_over_r2):
    """Eggleton (1983) Roche-lobe radius around planet, in units of the planet radius."""
    q13, q23 = q ** (1.0 / 3.0), q ** (2.0 / 3.0)
    return 0.49 * q23 / (0.6 * q23 + np.log1p(q13)) * a_over_r2 / r1_over_r2
