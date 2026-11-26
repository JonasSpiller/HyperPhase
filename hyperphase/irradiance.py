"""Irradiance of a planet by its extended host star (hyper-illumination), the point-source and heat-redistribution models, and the
resulting temperature maps.

Equation numbers refer to the paper: Spiller, Zieba & Wordsworth (in prep.), "Hyper-Illumination in Ultra-Short-Period
Rocky Planets: How Finite-Star Geometry Biases Albedos and Heat Redistribution Inferred from JWST Phase Curves"."""
import time
from functools import lru_cache

import numpy as np
from astropy import constants as const

from hyperphase.defaults import DEFAULT_A_B, DEFAULT_EMISSIVITY_BOL, DEFAULT_U_BOL


sigma = const.sigma_sb.value  # 5.670374e-8 W ⋅ m⁻² ⋅ K⁻⁴


############################################################################################################################################
#####   Helpers
############################################################################################################################################
def which_zone(gamma1, planet):
    return np.select(
        [gamma1 < planet.terminator_penumbra(), gamma1 < planet.terminator_nightside()],
        ["fully illuminated", "penumbra"],
        default="night-side")


def alpha_(gamma1, R, r1):
    """(alpha, rho) of Eq. (A1): the zenith angle of the stellar centre at the surface point gamma1, and its distance from it."""
    rho = np.sqrt(R ** 2 + r1 ** 2 - 2 * r1 * R * np.cos(gamma1))
    cos_alpha = (R * np.cos(gamma1) - r1) / rho  # Eq. (A1)
    return np.arccos(np.clip(cos_alpha, -1, 1)), rho


def BBtemp_from_insolation(insol, A_B=DEFAULT_A_B, emissivity_bol=DEFAULT_EMISSIVITY_BOL):
    """Temperature [K] in local radiative equilibrium, emissivity_bol sigma T^4 = (1-A_B) insol. A_B is always the Bond albedo (also for band
    measurements), emissivity_bol the bolometric emissivity. Only (1-A_B)/emissivity_bol enters, so emissivity_bol is exactly degenerate with A_B:
    with emissivity_bol fixed to 1, A_B is the effective albedo 1 - (1-A_B)/emissivity_bol."""
    return ((1-A_B) * insol / (emissivity_bol * sigma))**0.25


def _tabulate_insolation(planet, u_bol, insolation_model, n_tab=3000):
    """(gamma1, insolation) tabulated once on a 1-D grid over [0, pi] (n_tab points per illuminated zone; n_tab over [0, pi] for the point
    source), for 2-D maps and integrals to interpolate."""
    if insolation_model in EXTENDED_SOURCE_MODELS:
        gp1 = planet.terminator_penumbra()
        gp2 = planet.terminator_nightside()
        gp_fi = np.linspace(0., gp1, n_tab)
        F_fi = insolation_fullyilluminated_hyperillum(gp_fi, planet)
        gp_pen = np.linspace(gp1, gp2, n_tab)
        F_pen = insolation_penumbra_hyperillum(gp_pen, planet, u_bol=u_bol)
        gp_grid = np.concatenate([gp_fi, gp_pen, [np.pi]])
        F_grid = np.concatenate([F_fi, F_pen, [0.]])
    elif insolation_model in POINT_SOURCE_MODELS:
        gp_grid = np.linspace(0., np.pi, n_tab)
        F_grid = naive_insolation(gp_grid, planet)
    else:
        raise ValueError(f"insolation_model must be one of {INSOLATION_MODELS}")
    return gp_grid, F_grid


############################################################################################################################################
#####   Core Insolation Models
############################################################################################################################################

# The insolation models: "hyperillum" (extended stellar disk), "naive" (point source), "naive_redist" (point source with the heat redistribution F
# of Kreidberg & Loeb 2016, see Heat Redistribution below) and "hyperillum_redist" (extended disk with the same redistribution). The last is
# the truth of bias.sweeps.true_F_sweep (the hyper-illuminated planet that redistributes heat) and a retrieval model (retrieval.likelihood);
# the noiseless fits of hyperphase.bias fit the two point-source models only.
INSOLATION_MODELS = ("hyperillum", "naive", "naive_redist", "hyperillum_redist")
EXTENDED_SOURCE_MODELS = ("hyperillum", "hyperillum_redist")   # lit by the extended stellar disk (insolation map and reflected light)
POINT_SOURCE_MODELS = ("naive", "naive_redist")                # irradiated by a point source
REDIST_MODELS = ("naive_redist", "hyperillum_redist")          # take the heat redistribution F


def insolation(gamma1, planet, insolation_model, u_bol=DEFAULT_U_BOL):
    """Irradiance [W m^-2] at the angular distance gamma1 [rad] from the substellar point: from the extended star for "hyperillum" and
    "hyperillum_redist" (u_bol: its bolometric linear limb darkening), from a point source for "naive" and "naive_redist". The redistribution models
    share the irradiance and the reflected light of their source and redistribute only the absorbed heat (effective_insolation)."""
    if insolation_model in EXTENDED_SOURCE_MODELS: return insolation_hyperillum(gamma1, planet, u_bol=u_bol)
    elif insolation_model in POINT_SOURCE_MODELS: return naive_insolation(gamma1, planet)
    else: raise ValueError(f"insolation_model must be one of {INSOLATION_MODELS}")


def naive_insolation(gamma1, planet):
    """Point-source irradiance [W m^-2]: S cos(gamma1) with S = (r2/R)^2 sigma T_eff^4, the flux at the planet centre, and a hard terminator at
    90 deg, as in SPIDERMAN's Louden model."""
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    Fp_incident = (r2/R)**2 * sigma * T2_eff**4
    return np.where(gamma1 < np.pi/2, Fp_incident * np.cos(gamma1), 0.) # incident flux on planet (W/m^2)


def insolation_hyperillum(gamma1, planet, u_bol=DEFAULT_U_BOL):
    """Irradiance [W m^-2] from the extended, limb-darkened star (hyper-illumination, after Kopal 1954) at the angular distance gamma1 [rad] from
    the substellar point: the fully illuminated zone, the penumbra, and 0 on the nightside. u_bol: bolometric linear limb darkening."""
    gamma1 = np.atleast_1d(gamma1)
    gp1 = planet.terminator_penumbra()
    gp2 = planet.terminator_nightside()
    fi_mask = gamma1 < gp1
    pen_mask = (gamma1 >= gp1) & (gamma1 < gp2)

    Fp_incident = np.zeros_like(gamma1)
    Fp_incident[fi_mask] = insolation_fullyilluminated_hyperillum(gamma1[fi_mask], planet)
    Fp_incident[pen_mask] = insolation_penumbra_hyperillum(gamma1[pen_mask], planet, u_bol=u_bol)

    return Fp_incident # incident flux on planet (W/m^2)


def insolation_fullyilluminated_hyperillum(gamma1, planet):
    """Irradiance [W m^-2] in the fully illuminated zone: Eq. (3), the closed form of Eq. (A6). Independent of limb darkening: with every ring of
    the disk visible it is cos(alpha) times the radial flux L/(4 pi rho^2)."""
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    mu = np.cos(gamma1)
    Fp_incident =  r2**2 * sigma * T2_eff**4 * (R * mu - r1) / (R**2 + r1**2 - 2. * r1 * R * mu)**1.5   # Eq. (3)
    return Fp_incident # incident flux on planet (W/m^2)


def insolation_penumbra_hyperillum(gamma1, planet, u_bol=DEFAULT_U_BOL, order=64, runtime=False):
    """Irradiance [W m^-2] in the penumbra: Eqs. (A7), (A9) and (A10), exact (no small-disk approximation), by fixed-order Gauss-Legendre
    quadrature vectorized over all gamma1 (validation.insolation_penumbra_hyperillum_simpleint is the adaptive reference)."""
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    t0 = time.perf_counter()
    gamma1 = np.atleast_1d(np.asarray(gamma1, dtype=float))

    nodes, weights = _leggauss_cached(order)   # on [-1, 1]

    alpha, rho = alpha_(gamma1, R, r1)
    I0 = sigma * T2_eff ** 4 / np.pi / (1 - u_bol / 3)

    theta_max = np.arccos(r2 / rho)

    # map nodes to [0, theta_max] per point: theta = theta_max/2 * (x+1)
    theta = (theta_max[:, None] / 2.0) * (nodes[None, :] + 1.0)   # (N, order)
    jac = theta_max[:, None] / 2.0                                # (N, order)
    wq = weights[None, :] * jac

    A, B, cos_tp, Delta, st = _theta_kernels_vec(theta, alpha, rho, r2)

    def Mn(n):
        return 2.0 * r2**2 / np.pi * np.sum(A * Delta * cos_tp**n * st * wq, axis=1) # M_n of Eq. (A10), Gauss-Legendre quadrature

    def Nn(n):
        return 2.0 * r2**2 / np.pi * np.sum(B * np.sin(Delta) * cos_tp**n * st * wq, axis=1) # N_n of Eq. (A10), Gauss-Legendre quadrature

    ca, sa = np.cos(alpha), np.sin(alpha)
    J1 = Mn(1) * ca + Nn(1) * sa                          # Eq. (A9), n = 1
    Jp = I0 * (1.0 - u_bol) * J1
    if u_bol != 0.0:
        J2 = Mn(2) * ca + Nn(2) * sa                      # Eq. (A9), n = 2
        Jp += I0 * u_bol * J2
    result = np.pi * Jp                                   # I = pi B_0 [(1 - u_bol) J_1 + u_bol J_2], Eq. (A9)

    if runtime: print(f"insolation_penumbra_hyperillum runtime: {time.perf_counter() - t0:.4f} s")

    return result  # incident flux on planet (W/m^2)


@lru_cache(maxsize=None)
def _leggauss_cached(order):
    return np.polynomial.legendre.leggauss(order)


def _theta_kernels_vec(theta, alpha, rho, r2):
    """Return (cos(beta)/rho'^2, sin(beta)/rho'^2, cos theta', Delta, sin theta) at polar angle theta --
    vectorized over (point, quadrature-node) grids: theta has shape (N, order), alpha/rho
    shape (N,)."""
    rho = rho[:, None]
    alpha = alpha[:, None]
    ct = np.cos(theta)
    st = np.sin(theta)
    rp2 = rho**2 + r2**2 - 2.0 * rho * r2 * ct          # rho'^2                (Eq. A3)
    rp = np.sqrt(rp2)
    cos_beta = (rho - r2 * ct) / rp                      # cos beta              (Eq. A3)
    sin_beta = r2 * st / rp                              # sin beta
    cos_tp = (rho * ct - r2) / rp                        # cos theta'            (Eq. A3)
    # exact visible azimuthal half-width Delta(theta), Eq. (A7)
    with np.errstate(divide="ignore", invalid="ignore"):
        cot_beta = np.where(st > 0, (rho - r2 * ct) / (r2 * st), np.inf)
        g = np.where(np.tan(alpha) != 0, -cot_beta / np.tan(alpha), 0.0)   # -cot(a) cot(b)
    Delta = np.arccos(np.clip(g, -1.0, 1.0))            # in [0, pi]
    return cos_beta / rp2, sin_beta / rp2, cos_tp, Delta, st


############################################################################################################################################
#####   Heat Redistribution (Kreidberg & Loeb 2016)
############################################################################################################################################

# Kreidberg & Loeb 2016 eq. 1: sigma*T^4 = (1-A_B)*S*[F/2 + (1-2F)*cos(z)] on the dayside and (1-A_B)*S*F/2 on the nightside (z: zenith angle =
# gamma1, S: substellar insolation). In terms of the naive insolation I (= S*cos(z) on the dayside, 0 on the nightside) both read sigma*T^4 =
# (1-A_B)*[(1-2F)*I + F*S/2]: the nightside re-radiates the fraction F of the absorbed power. F=0: no redistribution (identical to "naive"), F=0.5:
# isothermal planet. This is "naive_redist"; SPIDERMAN's Louden model evaluates the same map (with T_int=0).

# "hyperillum_redist" is the same map with the extended-source irradiance: sigma*T^4 = (1-A_B)*[(1-2F)*I + 2F*<I>], <I> the mean insolation over
# the sphere (mean_insolation). For the point source <I> = S/4, so this IS the line above; for the extended source <I> is larger by the power the
# point source misses. The fraction 2F of the absorbed power is spread uniformly over the sphere and the rest re-radiated where it is absorbed, so
# the planet emits exactly what it absorbs for every F. F=0: identical to "hyperillum".

def check_redistribution(insolation_model, F):
    """Raises ValueError if a redistribution model lacks F or F is outside [0, 0.5]; the other models ignore F."""
    if insolation_model in REDIST_MODELS and F is None:
        raise ValueError(f'insolation_model="{insolation_model}" requires the heat redistribution parameter F in [0, 0.5]')
    if F is not None and not 0. <= F <= 0.5:
        raise ValueError(f"heat redistribution parameter F must be in [0, 0.5], got F={F}")


@lru_cache(maxsize=256)   # geometry and u_bol only: a retrieval of "hyperillum_redist" with fixed geometry asks for the same value at every call
def mean_insolation(planet, insolation_model, u_bol=DEFAULT_U_BOL, n_tab=3000):
    """Mean insolation over the whole sphere [W m^-2], <I> = (1/2) int I(gamma1) sin(gamma1) dgamma1: the intercepted stellar power per unit
    surface area. S/4 for the point source (up to the quadrature), larger for the extended source."""
    g, I = _tabulate_insolation(planet, u_bol, insolation_model, n_tab=n_tab)
    return float(np.trapezoid(I * np.sin(g), g) / 2.)


def effective_insolation(insol, planet, insolation_model, F=None, u_bol=DEFAULT_U_BOL):
    """Flux per unit (1-A_B) that a surface element re-radiates: the insolation itself under instant re-radiation ("hyperillum", "naive"), the
    redistributed flux under the redistribution models. u_bol: used only by "hyperillum_redist", for the mean insolation of the extended source."""
    check_redistribution(insolation_model, F)
    if insolation_model not in REDIST_MODELS: return insol
    if insolation_model == "naive_redist":
        S = float(naive_insolation(0., planet))  # substellar insolation
        return (1. - 2. * F) * insol + F * S / 2.
    return (1. - 2. * F) * insol + 2. * F * mean_insolation(planet, insolation_model, u_bol=u_bol)


def temperature_from_insolation(insol, planet, insolation_model, A_B=DEFAULT_A_B, emissivity_bol=DEFAULT_EMISSIVITY_BOL, F=None, u_bol=DEFAULT_U_BOL):
    """Surface temperature [K] from an irradiance map of insolation_model: BBtemp_from_insolation of its effective_insolation."""
    return BBtemp_from_insolation(effective_insolation(insol, planet, insolation_model, F=F, u_bol=u_bol), A_B = A_B, emissivity_bol=emissivity_bol)


def surface_temperature(gamma1, planet, insolation_model, u_bol=DEFAULT_U_BOL, A_B=DEFAULT_A_B, emissivity_bol=DEFAULT_EMISSIVITY_BOL, F=None):
    """Surface temperature [K] at the angular distance gamma1 [rad] from the substellar point, in local radiative equilibrium (Eq. 6), with the heat
    redistribution F of Kreidberg & Loeb (2016) for the redistribution models (Eq. 9, its extension to the extended source, for
    "hyperillum_redist")."""
    return temperature_from_insolation(insolation(gamma1, planet, insolation_model, u_bol=u_bol), planet, insolation_model, A_B = A_B, emissivity_bol=emissivity_bol, F=F, u_bol=u_bol)


def hemisphere_temperatures(planet, insolation_model="hyperillum", u_bol=DEFAULT_U_BOL, A_B=DEFAULT_A_B, emissivity_bol=DEFAULT_EMISSIVITY_BOL, F=None, n=40001):
    """Bolometric hemisphere temperatures [K] of the dayside (gamma1 <= 90 deg) and nightside (>= 90 deg) in local radiative equilibrium (F for the
    redistribution models), under three weights of gamma1. Returns {<weight>_day, <weight>_night} for
      uniform: dgamma1 / (pi/2), the hemisphere averages of Sadh & Gavassino 2026 (ApJ 997, 167) eqs. 15-17. Not an area mean: it over-weights the
        cap about the substellar (antistellar) point (point source: T0 (2/pi)^(1/4));
      area: sin(gamma1) dgamma1, the area mean of the flux over the hemisphere (point source: T0 (1/2)^(1/4));
      disk: 2 |cos gamma1| sin gamma1 dgamma1, the disk-integrated brightness temperature seen edge-on along the star-planet axis (dayside at
        mid-occultation, nightside at mid-transit). The point-source dayside is the T_d,max of Teske et al. 2025, T0 (2/3)^(1/4) (f = 2/3 of
        Cowan & Agol 2011);
    with T0 = T_eff sqrt(R*/a) [(1-A_B)/emissivity_bol]^(1/4). The point-source values are exact up to the quadrature."""
    g = np.linspace(0., np.pi, n)
    I = effective_insolation(insolation(g, planet, insolation_model, u_bol=u_bol), planet, insolation_model, F=F, u_bol=u_bol)
    weights = {"uniform": np.full_like(g, 2. / np.pi), "area": np.sin(g), "disk": 2. * np.abs(np.cos(g)) * np.sin(g)}
    out = {}
    for side, m in (("day", g <= np.pi / 2), ("night", g >= np.pi / 2)):
        for key, w in weights.items():
            out[f"{key}_{side}"] = float(BBtemp_from_insolation(np.trapezoid(I[m] * w[m], g[m]), A_B=A_B, emissivity_bol=emissivity_bol))
    return out
