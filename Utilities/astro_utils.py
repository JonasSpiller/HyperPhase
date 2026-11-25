import numpy as np
from scipy.optimize import newton
from scipy.integrate import quad
from astropy import constants as const
from HItargets_utils import load_filter, full_lambda, filter_lambda, filter_loc
from scipy.optimize import brentq
from functools import lru_cache
import warnings
import newera_utils

au = const.au.value
R_sun = const.R_sun.value
H_PLANCK = const.h.value
C_LIGHT = const.c.value
K_BOLTZ = const.k_B.value
WIEN_B = 2.897771955e-3   # Wien's displacement constant [m K]
STEFAN_BOLTZMANN = (2.0 * np.pi**5 * K_BOLTZ**4) / (15.0 * H_PLANCK**3 * C_LIGHT**2) # Stefan-Boltzmann constant

STELLAR_MODEL = "newera"    # repo-wide stellar SED: "newera" (newera_utils cube), an stsynphot grid ("phoenix", "ck04models", "k93models") or "blackbody". Read as a default argument at import, so change it here, not at runtime
SED_FLUX_TOL = 0.02         # max |pi int(I dlambda) / (sigma T_eff^4) - 1| accepted from a model spectrum (catches wrong units and a truncated wavelength range)


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

# Roche LIMIT - the tidal-disruption distance of the planet, not the Roche LOBE of roche_radius_over_r1. Everything here works in ORBITAL PERIOD,
# because in period the limit depends on the planet alone (Rappaport, Sanchis-Ojeda, Rogers, Levine & Winn 2013, ApJL 773, L15):
# - eq. 1-2, incompressible (uniform-density) fluid body: a_R / R_star = c (rho_star / rho_p)^(1/3) with c = 2.44 (Roche 1849; Chandrasekhar 1963:
#   2.455), so with Kepler III
#   P_R = sqrt(3 pi c^3 / (G rho_p)) = 12.6 h (rho_p / 1 g cm^-3)^(-1/2), independent of the star. The fluid value is the right one for any
#     planet-sized body (self-gravity dwarfs material
#   strength; the rigid-sphere 2^(1/3) is a small-body limit).
# - eq. 5, centrally condensed body (interpolation of Lai et al. 1993 polytropes): P_R = 12.6 h (rho_p / 1 g cm^-3)^(-1/2) (rho_0 / rho_p)^(-0.16),
#   rho_0 the central density; valid for rho_0 / rho_p <~ 6, about 1-2.5 for iron/silicate planets (Earth: 13.1 / 5.51 = 2.37). A real rocky planet
#   therefore survives ~13 % closer in period than the uniform body (KOI-1843.03 at 4.245 h is inside the uniform limit even for rho_p = 8 g/cm^3).
#   roche_period_h is eq. 5; rho_central_ratio = 1 recovers eq. 2.
# On an a/R_star axis the limit needs the host: a_R / R_star = (G rho_star P_R^2 / (3 pi))^(1/3) (kepler_a_rs). A planet's own distance from its limit
# needs no stellar density at all: a / a_R = (P / P_R)^(2/3) (roche_proximity), which is why the period form is immune to errors in the stellar
# parameters.
C_ROCHE = 2.44                                                                   # Rappaport et al. 2013 eq. 1
G_CGS = const.G.cgs.value                                                        # [cm^3 g^-1 s^-2]
RHO_SUN_CGS = (const.M_sun / (4. / 3. * np.pi * const.R_sun ** 3)).cgs.value   # [g/cm^3]
P_ROCHE_UNIFORM_H = np.sqrt(3. * np.pi * C_ROCHE ** 3 / G_CGS) / 3600.          # [h] P_R of a uniform body of 1 g/cm^3: 12.58 h (Rappaport et al. 2013 eq. 2: "12.6 hr")


def roche_period_h(rho_p, rho_central_ratio=1.):
    """Orbital period [h] at the Roche limit of a planet of mean density rho_p [g/cm^3] and central-to-mean density ratio rho_0/rho_p (Rappaport et al. 2013 eq. 5; 1 = uniform body, eq. 2)."""
    return P_ROCHE_UNIFORM_H * np.asarray(rho_p, dtype=float) ** -0.5 * np.asarray(rho_central_ratio, dtype=float) ** -0.16


def roche_proximity(period_h, P_R_h):
    """a / a_R = (P / P_R)^(2/3): a planet's distance from its Roche limit in units of the Roche distance, independent of the host (< 1: inside)."""
    return (np.asarray(period_h, dtype=float) / P_R_h) ** (2. / 3.)


def kepler_period_days(a_over_r2, rho_star):
    """Period [d] of a circular orbit at a/R_star around a star of mean density rho_star [g/cm^3] (Kepler III: P^2 = 3 pi (a/R_star)^3 / (G rho_star))."""
    return np.sqrt(3. * np.pi * np.asarray(a_over_r2, dtype=float) ** 3 / (G_CGS * rho_star)) / 86400.


def kepler_a_rs(period_days, rho_star):
    """The inverse: a/R_star of a circular orbit of period [d] around a star of mean density rho_star [g/cm^3]. With a Roche period it is the Roche limit on an a/R_star axis."""
    return (G_CGS * np.asarray(rho_star, dtype=float) * (86400. * np.asarray(period_days, dtype=float)) ** 2 / (3. * np.pi)) ** (1. / 3.)


# Mean density of a rocky planet of radius R_p from the rock/iron mass-radius fit of Fortney, Marley & Barnes 2007 (ApJ 659, 1661, eq. 8; ~1.5 %
# accuracy, fitted over 0.01-100 M_E): R = (0.0592 rmf + 0.0975)(log M)^2 + (0.2337 rmf + 0.4938) log M + (0.3102 rmf + 0.7932) [R_E, M_E], rmf = 1 -
# iron_fraction the rock (silicate mantle) mass fraction around an iron core. Earth (iron fraction 0.33) comes out at 1.00 R_E for 1 M_E. Compression
# makes larger planets denser: Earth-like 5.5 / 7.7 / 9.0 g/cm^3 at 1.0 / 1.5 / 1.7 R_E. (Zeng, Sasselov & Jacobsen 2016 give 7.3 g/cm^3 for
# Earth-like at 1.5 R_E, i.e. 3 % longer P_R; their fit covers iron fractions <= 0.4 only.)
def fortney_radius(M_earth, iron_fraction):
    """Radius [R_E] of a rock/iron planet of mass M_earth [M_E] (Fortney et al. 2007 eq. 8)."""
    rmf, lm = 1. - iron_fraction, np.log10(M_earth)
    return (0.0592 * rmf + 0.0975) * lm ** 2 + (0.2337 * rmf + 0.4938) * lm + (0.3102 * rmf + 0.7932)


def rocky_mean_density(R_p_earth, iron_fraction):
    """Mean density [g/cm^3] of a rock/iron planet of radius R_p_earth [R_E] and iron mass fraction iron_fraction (fortney_radius solved for the mass)."""
    M = brentq(lambda m: fortney_radius(m, iron_fraction) - R_p_earth, 1e-2, 1e2)
    return float((M * const.M_earth / (4. / 3. * np.pi * (R_p_earth * const.R_earth) ** 3)).cgs.value)


# Main-sequence hosts: (T_eff [K], R [R_sun], M [M_sun]) per spectral type from E. Mamajek's "A Modern Mean Dwarf Stellar Color and Effective
# Temperature Sequence", version 2022.04.16 (https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt; Pecaut & Mamajek 2013, ApJS
# 208, 9), F0V-M6V, every listed subtype. rho_star interpolates log rho linearly in
# T_eff, the other main_sequence_* functions R and M linearly; all clip outside the table, so a T_eff beyond F0V / M6V silently gets the end value.
# rho falls with T_eff except for a 2 % dip at
# K8V/K9V (as tabulated). Main sequence only: a young (contracting) or evolved host is less dense. Transit-derived densities of the PAPER_PLANETS
# hosts scatter by 0.6-1.6x around rho_star(T_eff).
MAIN_SEQUENCE = {"F0V": (7220, 1.728, 1.61), "F1V": (7020, 1.679, 1.50), "F2V": (6820, 1.622, 1.46), "F3V": (6750, 1.578, 1.44), "F4V": (6670, 1.533, 1.38), "F5V": (6550, 1.473, 1.33),
                 "F6V": (6350, 1.359, 1.25), "F7V": (6280, 1.324, 1.21), "F8V": (6180, 1.221, 1.18), "F9V": (6050, 1.167, 1.13), "F9.5V": (5990, 1.142, 1.08), "G0V": (5930, 1.100, 1.06),
                 "G1V": (5860, 1.060, 1.03), "G2V": (5770, 1.012, 1.00), "G3V": (5720, 1.002, 0.99), "G4V": (5680, 0.991, 0.985), "G5V": (5660, 0.977, 0.98), "G6V": (5600, 0.949, 0.97),
                 "G7V": (5550, 0.927, 0.95), "G8V": (5480, 0.914, 0.94), "G9V": (5380, 0.853, 0.90), "K0V": (5270, 0.813, 0.88), "K1V": (5170, 0.797, 0.86), "K2V": (5100, 0.783, 0.82),
                 "K3V": (4830, 0.755, 0.78), "K4V": (4600, 0.713, 0.73), "K5V": (4440, 0.701, 0.70), "K6V": (4300, 0.669, 0.69), "K7V": (4100, 0.630, 0.64), "K8V": (3990, 0.615, 0.62),
                 "K9V": (3930, 0.608, 0.59), "M0V": (3850, 0.588, 0.57), "M0.5V": (3770, 0.544, 0.54), "M1V": (3660, 0.501, 0.50), "M1.5V": (3620, 0.482, 0.47), "M2V": (3560, 0.446, 0.44),
                 "M2.5V": (3470, 0.421, 0.40), "M3V": (3430, 0.361, 0.37), "M3.5V": (3270, 0.300, 0.27), "M4V": (3210, 0.274, 0.23), "M4.5V": (3110, 0.217, 0.184), "M5V": (3060, 0.196, 0.162),
                 "M5.5V": (2930, 0.156, 0.123), "M6V": (2810, 0.137, 0.102)}
_MS_T = np.array([v[0] for v in MAIN_SEQUENCE.values()], dtype=float)[::-1]   # ascending T_eff, as np.interp needs
_MS_R = np.array([v[1] for v in MAIN_SEQUENCE.values()], dtype=float)[::-1]
_MS_M = np.array([v[2] for v in MAIN_SEQUENCE.values()], dtype=float)[::-1]
_MS_RHO = RHO_SUN_CGS * _MS_M / _MS_R ** 3                                       # [g/cm^3]


def rho_star(T_eff):
    """Mean density [g/cm^3] of a main-sequence star of effective temperature T_eff [K]: rho_sun M / R^3 from MAIN_SEQUENCE, log-interpolated in T_eff."""
    return 10. ** np.interp(T_eff, _MS_T, np.log10(_MS_RHO))


def main_sequence_R_star(T_eff):
    """Radius [R_sun] of a main-sequence star of this T_eff [K]."""
    return np.interp(T_eff, _MS_T, _MS_R)


def main_sequence_M_star(T_eff):
    """Mass [M_sun] of a main-sequence star of this T_eff [K]."""
    return np.interp(T_eff, _MS_T, _MS_M)


def main_sequence_logg(T_eff):
    """Surface gravity log10(G M / R^2) [cgs] of a main-sequence star of this T_eff [K]."""
    return np.log10(G_CGS * main_sequence_M_star(T_eff) * const.M_sun.cgs.value / (main_sequence_R_star(T_eff) * const.R_sun.cgs.value) ** 2)


def T_bright(thermal_depth, planet, lambda_range, filter_path=None, stellar_model=STELLAR_MODEL):
    """Brightness temperature [K] of a thermal planet flux [ppm] in the band lambda_range [m] (filter_path: throughput file, None for a top-hat). The
    occultation depth gives the dayside value; it is a blackbody temperature, below the surface temperature for emissivity < 1.

    Solves (r1/r2)^2 S_planet(T) = depth S_star, where S_star is SPIDERMAN's own stellar normalisation (its cubic spline through stellar_spidergrid
    at T2_eff, hpu.spidergrid_value) and S_planet the same band integral of a blackbody (hpu.band_planck: same samples, response and sum_flux). So
    T_bright inverts SPIDERMAN's depths exactly whatever the stellar model and band weighting (a top-hat is energy-weighted there, like SPIDERMAN's
    filter=False); the remaining ~0.15 K of a round trip is SPIDERMAN's own planet quadrature. Raises (check_teff_nodes) outside the T_eff nodes and
    ValueError (brentq) for a flux below the band flux at 10 K."""
    import hyperillumination_phasecurve_utils as hpu   # imported here: hpu imports this module
    grid = hpu.stellar_spidergrid(lambda_range, filter_path, planet.logg, planet.feh, stellar_model=stellar_model)
    hpu.check_teff_nodes(grid, planet.T2_eff)
    tgt = 1e-6 * thermal_depth * hpu.spidergrid_value(grid, planet.T2_eff) / (planet.r1 / planet.r2) ** 2
    return brentq(lambda T: hpu.band_planck(lambda_range, filter_path, T) - tgt, 10., 9000.)


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
    return WIEN_B / T # max. wavelength [in m]

# Spectral radiance [W m^-2 sr^-1 m^-1] of a blackbody at temperature T, per unit wavelength [m] (Planck function): B_lambda(T) = (2 h c^2 / lambda^5)
# / (exp(h c / (lambda k_B T)) - 1)
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
    filter_path=None  -> tophat over lambda_range, photon-weighted (SPIDERMAN's filter=False is energy-weighted, so this does NOT match it; T_bright does not use this function)
    filter_path given -> real throughput on the file's own grid
    """
    lam, w = load_filter(lambda_range, filter_path=filter_path, n=n)

    # R [counts/photon] -> [counts/energy], so that weighting B_lambda (an energy spectral radiance) below yields a photon-integrated band average.
    # Same convention as fspiderman.get_filter (README of github.com/JonasSpiller/fSPIDERMAN, "6. Photon-counting response weighting"). Unused by the
    # pipeline: T_bright integrates through hpu.band_planck.
    w =  w * lam / (H_PLANCK * C_LIGHT)

    B_lambda = planck_spectrum(lam, T)

    return np.trapezoid(B_lambda * w, lam) / np.trapezoid(w, lam)


# --------------------------------------------------------------------------
# Stellar SED
# --------------------------------------------------------------------------

# Model spectrum on its native grid: wavelength [m], surface intensity F_lambda/pi [W m^-2 sr^-1 m^-1] (what B_lambda is for a blackbody). The same
# stsynphot catalog PandExo uses ($PYSYN_CDBS/grid/<stellar_model>). grid_to_spec interpolates linearly between the 8 bracketing models (logg, then
# [Fe/H], then T_eff) and raises ParameterOutOfBounds outside the grid or on a neighbour without data. Checked against sigma T_eff^4, which fails on
# wrong units or a truncated wavelength range. Cached: treat the arrays as read-only.
@lru_cache(maxsize=None)
def stellar_spectrum(T_eff, logg, feh, stellar_model=STELLAR_MODEL):
    if stellar_model == "newera":   # no sigma T^4 check: the JWST product starts at 0.6 um; validated against full-range models in inputs/stellar_model/NewEra/build_newera.ipynb
        wave, I = newera_utils.newera_spectrum(float(T_eff), float(logg), float(feh))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # stsynphot warns about optional reference files it cannot find on import; imported only here, as it needs $PYSYN_CDBS
            import stsynphot
            from synphot import units as syn_units
        s = stsynphot.grid_to_spec(stellar_model, float(T_eff), float(feh), float(logg))   # argument order: (gridname, t_eff, metallicity, log_g)
        w = s.waveset                                                    # the model's own wavelengths [A]
        wave, I = w.to_value("m"), s(w, flux_unit=syn_units.FLAM).value * 1e7 / np.pi   # A -> m; FLAM [erg s^-1 cm^-2 A^-1] -> W m^-3; /pi: surface flux -> surface intensity
        ratio = np.pi * np.trapezoid(I, wave) / (STEFAN_BOLTZMANN * T_eff**4)
        if abs(ratio - 1.) > SED_FLUX_TOL: raise ValueError(f"{stellar_model} (T_eff={T_eff:.0f} K, logg={logg:.2f}, [Fe/H]={feh:+.2f}): pi int(I dlambda) / sigma T_eff^4 = {ratio:.4f}")
    for a in (wave, I): a.setflags(write=False)   # cached and shared by every caller
    return wave, I


# Surface intensity [W m^-2 sr^-1 m^-1] of the star on lam [m]: planck_spectrum for "blackbody", else the model spectrum interpolated.
def stellar_intensity(lam, T_eff, logg, feh, stellar_model=STELLAR_MODEL):
    if stellar_model == "blackbody": return planck_spectrum(lam, T_eff)
    wave, I = stellar_spectrum(T_eff, logg, feh, stellar_model)
    if wave[0] > np.min(lam) or wave[-1] < np.max(lam): raise ValueError(f"{stellar_model} spectrum covers {wave[0]*1e6:.2f}-{wave[-1]*1e6:.2f} um, band needs {np.min(lam)*1e6:.2f}-{np.max(lam)*1e6:.2f} um")
    return np.interp(lam, wave, I)


# lam [m] refined with the model's own samples inside [min(lam), max(lam)], so interpolating the star onto it loses no spectral structure (the catalog
# grid can be coarser or finer than a filter file). Unchanged for "blackbody".
def band_wavelengths(lam, T_eff, logg, feh, stellar_model=STELLAR_MODEL):
    lam = np.asarray(lam, float)
    if stellar_model == "blackbody": return lam
    wave = stellar_spectrum(T_eff, logg, feh, stellar_model)[0]
    return np.union1d(lam, wave[(wave > lam.min()) & (wave < lam.max())])


# --------------------------------------------------------------------------
# Limb darkening (ExoTiC-LD)
# --------------------------------------------------------------------------

# Quadratic limb darkening of the transit, fitted with ExoTiC-LD (Grant & Wakeford 2024) to the specific intensities I(mu) of its stellar grids and
# band-weighted with the throughput files of the forward model (ExoTiC-LD intensities are photon rates, the throughputs electrons per photon: the
# fspiderman weighting). The NewEra cube cannot do this, it has no angular dependence. Model files are downloaded on first use into LD_DATA. exotic_ld
# is imported inside quad_ld, so the rest of astro_utils does not depend on it. Used by inputs/stellar_model/limb_darkening.ipynb.
LD_DATA = str(newera_utils.NEWERA_DIR.parent / "ExoTiC-LD")   # inputs/stellar_model/ExoTiC-LD (gitignored, like NewEra/): model cache, filled on first use
LD_GRID_TEFF_MIN = {"phoenix": 2300., "mps1": 3500., "mps2": 3500., "kurucz": 3500., "stagger": 4000.}   # [K] lower T_eff edge of each ExoTiC-LD grid; below it ExoTiC-LD silently falls back to the nearest model
LD_MU_MIN = 0.1     # ExoTiC-LD default; the steep drop of spherical models at the very limb is not fitted
A_PER_M = 1e10      # ExoTiC-LD takes wavelengths in Angstrom

# Quadratic (u1, u2) of one host star (PlanetParams: T2_eff, logg, feh) per band (instrument keys of HItargets_utils), and the model I(mu)/I(1) each
# fit was made to. Returns (rows, profiles): one dict per band (ld_model, planet, T_eff, logg, feh, band, u1, u2, in_grid, covered) and {band: (mu,
# I_mu)}. in_grid = False: T_eff below the grid (nearest-model fallback). covered = False: the band extends beyond the wavelength range of the stellar
# model (ExoTiC-LD then integrates only the overlap; no overlap gives NaN and a printed note).
def quad_ld(planet, bands, ld_model, interpolate_type="nearest", mu_min=LD_MU_MIN, ld_data_path=LD_DATA):
    from exotic_ld import StellarLimbDarkening
    sld = StellarLimbDarkening(M_H=planet.feh, Teff=planet.T2_eff, logg=planet.logg, ld_model=ld_model,
                               ld_data_path=ld_data_path, interpolate_type=interpolate_type, verbose=0)
    wl_model = sld.stellar_wavelengths.min(), sld.stellar_wavelengths.max()
    rows, profiles = [], {}
    for band in bands:
        lam, T = load_filter(full_lambda(band), filter_loc(band))          # [m], electrons per photon, as in the forward model
        wl_band = filter_lambda(band, planet) * A_PER_M                     # the planet's own part of the band (HItargets_utils.PARTIAL_BANDS)
        try:
            u1, u2 = sld.compute_quadratic_ld_coeffs(wl_band, "custom", custom_wavelengths=lam * A_PER_M, custom_throughput=T, mu_min=mu_min)
            profiles[band] = (sld.mus.copy(), sld.I_mu.copy())              # I_mu is set by the call above
        except ValueError as err:                                           # no overlap with the model's wavelength range
            print(f"{ld_model}, {planet.name}, {band}: {err}")
            u1 = u2 = np.nan
        rows.append(dict(ld_model=ld_model, planet=planet.name, T_eff=planet.T2_eff, logg=planet.logg, feh=planet.feh, band=band, u1=u1, u2=u2,
                         in_grid=planet.T2_eff >= LD_GRID_TEFF_MIN[ld_model], covered=wl_model[0] <= wl_band[0] and wl_band[1] <= wl_model[1]))
    return rows, profiles


# Bolometric intensity profile I_bol(mu)/I_bol(1) of one host star (PlanetParams: T2_eff, logg, feh) from an ExoTiC-LD grid, to compare with the
# linear law of u_bol (the irradiance). ExoTiC-LD intensities are photon rates per Angstrom, so the energy weight is 1/lambda (constants cancel in the
# normalisation); integrated over the whole wavelength range of the grid on its own samples. Returns (mu ascending, I_bol(mu)/I_bol(1), u_lin, f_out):
# u_lin is the least-squares linear coefficient over mu >= mu_min, f_out the fraction of a blackbody's sigma T_eff^4 outside the grid's wavelength
# range (what the profile misses; PHOENIX stops at 5.5 um, MPS-ATLAS reaches 160 um).
def bolometric_ld_profile(planet, ld_model, interpolate_type="nearest", mu_min=LD_MU_MIN, ld_data_path=LD_DATA):
    from exotic_ld import StellarLimbDarkening
    sld = StellarLimbDarkening(M_H=planet.feh, Teff=planet.T2_eff, logg=planet.logg, ld_model=ld_model,
                               ld_data_path=ld_data_path, interpolate_type=interpolate_type, verbose=0)
    wl, mu = sld.stellar_wavelengths, sld.mus                                                 # [A]
    I_bol = np.trapezoid(sld.stellar_intensities / wl[:, None], wl, axis=0)                   # photons -> energy
    o = np.argsort(mu)
    mu, I_bol = mu[o], I_bol[o] / I_bol[o][-1]
    x = -(1. - mu[mu >= mu_min])
    u_lin = np.dot(x, I_bol[mu >= mu_min] - 1.) / np.dot(x, x)
    T = planet.T2_eff
    f_out = 1. - (planck_fraction_below(T, wl.max() / A_PER_M) - planck_fraction_below(T, wl.min() / A_PER_M)) / 100.
    return mu, I_bol, u_lin, f_out
