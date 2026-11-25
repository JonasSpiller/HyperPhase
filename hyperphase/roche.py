"""Roche limit of rocky planets, their mean densities, and main-sequence hosts."""

import numpy as np
from astropy import constants as const
from scipy.optimize import brentq


############################################################################################################################################
#####   Roche cases of the bias maps
############################################################################################################################################

# The Roche limits drawn on the (a/R*, T_eff) maps around main-sequence hosts, as {label: (iron mass fraction, rho_0 / rho_mean)} of a rock/iron
# planet of radius ROCHE_R_P: the mean density follows from rocky_mean_density (Fortney et al. 2007), the Roche period from roche_period_h
# (Rappaport et al. 2013 eq. 5).
# - "Earth-like": Earth's iron fraction 0.33 and central concentration 13.1 / 5.51 = 2.37 (PREM). The limit of a typical rocky planet.
# - "iron-rich": 70 % iron, the most that collisional mantle stripping leaves in planets above ~1 M_E (Marcus et al. 2010, as used by Rappaport
#   et al. 2013), with rho_0 / rho_mean = 2.5, the top of Rappaport's range 1-2.5 (a larger ratio survives closer in, so this is the permissive
#   end). The innermost limit: inside it no rocky planet of radius ROCHE_R_P survives around a main-sequence host.
# ROCHE_R_P = 1.5 R_E, typical of PAPER_PLANETS (TOI-2431 b and K2-141 b are 1.5 R_E). Compression makes larger planets denser, so their limits
# lie further in.
ROCHE_R_P = 1.5   # [R_E]
ROCHE_CASES = {"Earth-like": (0.33, 2.37), "iron-rich (70 % Fe)": (0.70, 2.5)}


############################################################################################################################################
#####   Roche limit and main-sequence hosts
############################################################################################################################################

# Roche LIMIT: the tidal-disruption distance of the planet, not the Roche LOBE of roche_radius_over_r1. Everything here works in orbital period,
# because in period the limit depends on the planet alone (Rappaport, Sanchis-Ojeda, Rogers, Levine & Winn 2013, ApJL 773, L15):
# - eqs. 1-2, incompressible (uniform-density) fluid body: a_R / R_star = c (rho_star / rho_p)^(1/3), c = 2.44 (Roche 1849; Chandrasekhar
#   1963: 2.455), so with Kepler III P_R = sqrt(3 pi c^3 / (G rho_p)) = 12.6 h (rho_p / 1 g cm^-3)^(-1/2), independent of the star. The fluid
#   value holds for any planet-sized body (self-gravity dwarfs material strength; the rigid-sphere 2^(1/3) is a small-body limit).
# - eq. 5, centrally condensed body (interpolation of Lai et al. 1993 polytropes): P_R = 12.6 h (rho_p / 1 g cm^-3)^(-1/2) (rho_0 / rho_p)^(-0.16),
#   rho_0 the central density; valid for rho_0 / rho_p <~ 6, about 1-2.5 for iron/silicate planets.
# On an a/R_star axis the limit needs the host (kepler_a_rs); a planet's distance from its own limit does not (roche_proximity).
C_ROCHE = 2.44                                                                   # Rappaport et al. 2013 eq. 1
G_CGS = const.G.cgs.value                                                        # [cm^3 g^-1 s^-2]
RHO_SUN_CGS = (const.M_sun / (4. / 3. * np.pi * const.R_sun ** 3)).cgs.value   # [g/cm^3]
P_ROCHE_UNIFORM_H = np.sqrt(3. * np.pi * C_ROCHE ** 3 / G_CGS) / 3600.          # [h] P_R of a uniform body of 1 g/cm^3 (eq. 2: 12.6 h)


def roche_period_h(rho_p, rho_central_ratio=1.):
    """Orbital period [h] at the Roche limit of a planet of mean density rho_p [g/cm^3] and central-to-mean density ratio rho_0/rho_p
    (Rappaport et al. 2013 eq. 5; 1 = uniform body, eq. 2)."""
    return P_ROCHE_UNIFORM_H * np.asarray(rho_p, dtype=float) ** -0.5 * np.asarray(rho_central_ratio, dtype=float) ** -0.16


def roche_proximity(period_h, P_R_h):
    """a / a_R = (P / P_R)^(2/3): a planet's distance from its Roche limit in units of the Roche distance, independent of the host
    (< 1: inside)."""
    return (np.asarray(period_h, dtype=float) / P_R_h) ** (2. / 3.)


def kepler_period_days(a_over_r2, rho_star):
    """Period [d] of a circular orbit at a/R_star around a star of mean density rho_star [g/cm^3] (Kepler III:
    P^2 = 3 pi (a/R_star)^3 / (G rho_star))."""
    return np.sqrt(3. * np.pi * np.asarray(a_over_r2, dtype=float) ** 3 / (G_CGS * rho_star)) / 86400.


def kepler_a_rs(period_days, rho_star):
    """The inverse: a/R_star of a circular orbit of period [d] around a star of mean density rho_star [g/cm^3]; with a Roche period, the
    Roche limit on an a/R_star axis."""
    return (G_CGS * np.asarray(rho_star, dtype=float) * (86400. * np.asarray(period_days, dtype=float)) ** 2 / (3. * np.pi)) ** (1. / 3.)


def fortney_radius(M_earth, iron_fraction):
    """Radius [R_E] of a rock/iron planet of mass M_earth [M_E], from the mass-radius fit of Fortney, Marley & Barnes 2007 (ApJ 659, 1661,
    eq. 8; ~1.5 % accuracy, fitted over 0.01-100 M_E): R = (0.0592 rmf + 0.0975)(log M)^2 + (0.2337 rmf + 0.4938) log M + (0.3102 rmf + 0.7932),
    rmf = 1 - iron_fraction the rock (silicate mantle) mass fraction around an iron core. Earth (iron fraction 0.33) comes out at 1.00 R_E."""
    rmf, lm = 1. - iron_fraction, np.log10(M_earth)
    return (0.0592 * rmf + 0.0975) * lm ** 2 + (0.2337 * rmf + 0.4938) * lm + (0.3102 * rmf + 0.7932)


def rocky_mean_density(R_p_earth, iron_fraction):
    """Mean density [g/cm^3] of a rock/iron planet of radius R_p_earth [R_E] and iron mass fraction iron_fraction (fortney_radius solved for
    the mass)."""
    M = brentq(lambda m: fortney_radius(m, iron_fraction) - R_p_earth, 1e-2, 1e2)
    return float((M * const.M_earth / (4. / 3. * np.pi * (R_p_earth * const.R_earth) ** 3)).cgs.value)


# Main-sequence hosts: (T_eff [K], R [R_sun], M [M_sun]) per spectral type, F0V-M6V, every listed subtype, from E. Mamajek's "A Modern Mean
# Dwarf Stellar Color and Effective Temperature Sequence", version 2022.04.16
# (https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt; Pecaut & Mamajek 2013, ApJS 208, 9).
# rho_star interpolates log rho linearly in T_eff, the other main_sequence_* functions R and M linearly; all clip outside the table, so a T_eff
# beyond F0V / M6V silently gets the end value. rho falls with T_eff except for a 2 % dip at K8V/K9V (as tabulated). Main sequence only: a
# young (contracting) or evolved host is less dense.
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
    """Mean density [g/cm^3] of a main-sequence star of effective temperature T_eff [K]: rho_sun M / R^3 from MAIN_SEQUENCE, log-interpolated
    in T_eff."""
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
