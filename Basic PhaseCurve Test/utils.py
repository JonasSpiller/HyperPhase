import batman
import numpy as np

def eclipse_time(period, ecc, w):
    w_rad = np.radians(w)

    f_transit = np.pi/2. - w_rad
    f_eclipse = 3.*np.pi/2. - w_rad

    def true_to_time(f, ecc, period):
        E = 2. * np.arctan(np.sqrt((1. - ecc) / (1. + ecc)) * np.tan(f / 2.))
        M = E - ecc * np.sin(E)
        return (M / (2. * np.pi)) * period

    t_transit = true_to_time(f_transit, ecc, period)
    t_eclipse = true_to_time(f_eclipse, ecc, period)

    dt = t_eclipse - t_transit
    if dt < 0:
        dt += period

    return dt

def flux_eclipse(period, rp, a, incl, ecc, w, fp, t):
    params_eclipse = batman.TransitParams()

    params_eclipse.t0 = 0.  # time of inferior conjunction
    params_eclipse.per = period  # orbital period
    params_eclipse.rp = rp  # planet radius (in units of stellar radii)
    params_eclipse.a = a  # semi-major axis (in units of stellar radii)
    params_eclipse.inc = incl  # orbital inclination (in degrees)
    params_eclipse.ecc = ecc  # eccentricity
    params_eclipse.w = w  # longitude of periastron (in degrees)

    params_eclipse.limb_dark = "nonlinear"  # limb darkening model
    params_eclipse.u = [0.5, 0.1, 0.1, -0.1]  # limb darkening coefficients [u1, u2, u3, u4]

    params_eclipse.fp = fp  # planet-to-star flux ratio (eclipse depth)
    params_eclipse.t_secondary = period / 2  # time of secondary eclipse center (gets adjusted automatically for eccentric orbits)

    m_eclipse = batman.TransitModel(params_eclipse, t, transittype="secondary", max_err=0.001)  # initializes model

    #params_eclipse.t_secondary = m_eclipse.get_t_secondary(params_eclipse)

    return m_eclipse.light_curve(params_eclipse)


def flux_transit(period, rp, a, incl, ecc, w, t):
    params_transit = batman.TransitParams()

    params_transit.t0 = 0.  # time of inferior conjunction
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


def PhaseCurve(period, rp, a, incl, ecc, w, fp, offset, t=[]):
    if len(t) == 0: t = np.linspace(-period / 4, period * 3 / 4, 10000)

    phase = (np.cos(2 * np.pi * (t / period - 0.5) - offset) + 1) / 2

    return t, flux_transit(period, rp, a, incl, ecc, w, t)+(flux_eclipse(period, rp, a, incl, ecc, w, fp, t)-1)*phase
