import numpy as np
from scipy.optimize import newton
from scipy.integrate import quad
from astropy import constants as const
from HItargets_utils import load_filter
from scipy.optimize import brentq

au = const.au.value
R_sun = const.R_sun.value
H_PLANCK = const.h.value
C_LIGHT = const.c.value
K_BOLTZ = const.k_B.value
b = 2.897771955e-3 # Wien
STEFAN_BOLTZMANN = (2.0 * np.pi**5 * K_BOLTZ**4) / (15.0 * H_PLANCK**3 * C_LIGHT**2) # Stefan-Boltzmann constant


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


# --------------------------------------------------------------------------
# Roche limit and main-sequence hosts
# --------------------------------------------------------------------------

# Roche LIMIT - the tidal-disruption distance of the planet, not the Roche LOBE of roche_radius_over_r1: a_R / R_star = c (rho_star / rho_p)^(1/3) for a satellite of mean density rho_p
# around a star of mean density rho_star. c = 2.456 for a fluid body, 2^(1/3) for a rigid strengthless sphere (tidal stretching = self-gravity at the surface); the two bracket a rocky planet.
# In period the limit is host-independent (roche_limit_period_h); in a/R_star it moves with the host through rho_star (kepler_period_days), so on an a/R_star axis it is a family labelled by rho_star / rho_p.
C_ROCHE_FLUID = 2.456
C_ROCHE_RIGID = 2. ** (1. / 3.)
G_CGS = const.G.cgs.value                                                        # [cm^3 g^-1 s^-2]
RHO_SUN_CGS = (const.M_sun / (4. / 3. * np.pi * const.R_sun ** 3)).cgs.value   # [g/cm^3]


def roche_limit_a_rs(rho_star_over_rho_p, c=C_ROCHE_FLUID):
    """a/R_star of the Roche limit for the density ratio rho_star / rho_p."""
    return c * np.asarray(rho_star_over_rho_p, dtype=float) ** (1. / 3.)


def roche_limit_period_h(rho_p, c=C_ROCHE_FLUID):
    """Orbital period [h] at the Roche limit, P = sqrt(3 pi c^3 / (G rho_p)): independent of the star. rho_p in g/cm^3."""
    return np.sqrt(3. * np.pi * c ** 3 / (G_CGS * rho_p)) / 3600.


def kepler_period_days(a_over_r2, rho_star):
    """Period [d] of a circular orbit at a/R_star around a star of mean density rho_star [g/cm^3] (Kepler III: P^2 = 3 pi (a/R_star)^3 / (G rho_star))."""
    return np.sqrt(3. * np.pi * np.asarray(a_over_r2, dtype=float) ** 3 / (G_CGS * rho_star)) / 86400.


# Main-sequence hosts: (T_eff [K], M [M_sun], R [R_sun]) per spectral type, Pecaut & Mamajek 2013 (online table, rounded). Hand-maintained; the main_sequence_* functions interpolate it in
# T_eff (log rho linearly) and clip outside its range, so a T_eff beyond F0V / M6V silently gets the end value.
MAIN_SEQUENCE = {"F0V": (7220, 1.59, 1.73), "F5V": (6510, 1.33, 1.47), "G0V": (5930, 1.06, 1.10), "G2V": (5770, 1.02, 1.01), "G5V": (5660, 0.98, 0.95),
                 "K0V": (5280, 0.88, 0.81), "K2V": (5040, 0.82, 0.76), "K5V": (4440, 0.70, 0.71), "K7V": (4050, 0.63, 0.63), "M0V": (3850, 0.57, 0.59),
                 "M1V": (3660, 0.50, 0.50), "M2V": (3560, 0.44, 0.45), "M3V": (3430, 0.37, 0.36), "M4V": (3210, 0.23, 0.27), "M5V": (3060, 0.16, 0.20), "M6V": (2810, 0.10, 0.14)}
_MS_T = np.array([v[0] for v in MAIN_SEQUENCE.values()], dtype=float)[::-1]                   # ascending T_eff, as np.interp needs
_MS_RHO = np.array([RHO_SUN_CGS * v[1] / v[2] ** 3 for v in MAIN_SEQUENCE.values()])[::-1]   # [g/cm^3], falls monotonically with T_eff
_MS_R = np.array([v[2] for v in MAIN_SEQUENCE.values()], dtype=float)[::-1]


def main_sequence_rho_star(T_eff):
    """Mean density [g/cm^3] of a main-sequence star of this T_eff [K]."""
    return 10. ** np.interp(T_eff, _MS_T, np.log10(_MS_RHO))


def main_sequence_T_eff(rho_star):
    """The inverse: the main-sequence T_eff [K] whose mean density is rho_star [g/cm^3]."""
    return np.interp(np.log10(rho_star), np.log10(_MS_RHO)[::-1], _MS_T[::-1])


def main_sequence_R_star(T_eff):
    """Radius [R_sun] of a main-sequence star of this T_eff [K]."""
    return np.interp(T_eff, _MS_T, _MS_R)


# calculate Brightness Temperature (use occultation depth for dayside brightness temperature)
def T_bright(thermal_depth, planet, lambda_range, filter_path=None, **kw):
    # thermal_depth in ppm
    rp = planet.r1 / planet.r2
    tgt = 1e-6*thermal_depth * band_avg_planck(planet.T2_eff, lambda_range, filter_path, **kw) / rp**2
    return brentq(lambda T: band_avg_planck(T, lambda_range, filter_path, **kw) - tgt, 10., 9000.)


# --------------------------------------------------------------------------
# Planck
# --------------------------------------------------------------------------


def planck_ratio(rprs2, Ts, Tp, wavelength_m):
    """
    Occultation depth in ppm for a blackbody planet: F_p/F_s = (r1/r2)² · B_λ(T_p)/B_λ(T_s)
    B_λ(T) = 2hc²/λ⁵ · 1/(exp(hc/λkT) − 1)
    rprs2 = (r1/r2)^2
    """
    x_s = H_PLANCK * C_LIGHT / (wavelength_m * K_BOLTZ * Ts)
    x_p = H_PLANCK * C_LIGHT / (wavelength_m * K_BOLTZ * Tp)
    return rprs2 * 1e6 * (np.exp(x_s) - 1.0) / (np.exp(x_p) - 1.0)


def planck_ratio_logderiv(Tp, wavelength_m):
    """d ln(occ_depth) / d ln(Tp), for propagating a temperature uncertainty into an occultation depth."""
    x_p = H_PLANCK * C_LIGHT / (wavelength_m * K_BOLTZ * Tp)
    return x_p * np.exp(x_p) / (np.exp(x_p) - 1.0)


# Wien's displacement law
def wien_lambda_max(T):
    return b / T # max. wavelength [in m]

# Spectral radiance [W m^-2 sr^-1 m^-1] of a blackbody at temperature T, per unit wavelength [m] (Planck function): B_lambda(T) = (2 h c^2 / lambda^5) / (exp(h c / (lambda k_B T)) - 1)
def planck_spectrum(wavelengths, T):
    wavelengths = np.asarray(wavelengths, dtype=float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        exponent = H_PLANCK * C_LIGHT / (wavelengths * K_BOLTZ * T)
        prefactor = 2.0 * H_PLANCK * C_LIGHT ** 2 / wavelengths ** 5
        B = prefactor / np.expm1(exponent)
    B = np.where((wavelengths <= 0) | (exponent > 700), 0.0, B)
    return B if B.ndim > 0 else float(B)

# Percentage of a blackbody's total emitted flux that lies at wavelengths shorter than `lambda_cutoff`, at temperature T.
# lambda_cutoff in m
def planck_fraction_below(T, lambda_cutoff):
    numerator, _ = quad(lambda wl: planck_spectrum(wl, T), 0.0, lambda_cutoff)
    denominator = STEFAN_BOLTZMANN * T ** 4 / np.pi
    return 100.0 * numerator / denominator

# Throughput-weighted mean spectral radiance [W m^-2 sr^-1 m^-1] of a blackbody at temperature T, averaged across the band.
def band_avg_planck(T, lambda_range, filter_path=None, n=300):
    """
    filter_path=None  -> tophat over lambda_range   (matches SPIDERMAN's no-filter mode)
    filter_path given -> real throughput on the file's own grid
    """
    lam, w = load_filter(lambda_range, filter_path=filter_path, n=n)

    # R [counts/photon] -> [counts/energy], so that weighting B_lambda (an energy spectral radiance) below yields a photon-integrated band average.
    w =  w * lam / (H_PLANCK * C_LIGHT)

    B_lambda = planck_spectrum(lam, T)

    return np.trapezoid(B_lambda * w, lam) / np.trapezoid(w, lam)