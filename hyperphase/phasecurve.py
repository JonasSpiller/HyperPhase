"""Phase curves: the thermal curve from a temperature map (fSPIDERMAN), the reflected curve, their sum, and the fast light-curve evaluation.

The irradiance takes the semi-major axis R as the planet-star distance, and fast_lightcurve assumes a circular orbit; an eccentric orbit would need
the instantaneous distance R (1 - ecc^2) / (1 + ecc cos f)."""
import contextlib
import io
import time

import numpy as np
from astropy import constants as const
with contextlib.redirect_stdout(io.StringIO()):   # mute "looking for spidermanrc..." in every process
    import fspiderman as sp

from hyperphase.defaults import (DEFAULT_A, DEFAULT_A_B, DEFAULT_EMISSIVITY, DEFAULT_EMISSIVITY_BOL, DEFAULT_N_LA, DEFAULT_N_LO, DEFAULT_U,
                                 DEFAULT_U_BOL, STELLAR_MODEL)
from hyperphase.irradiance import (EXTENDED_SOURCE_MODELS, INSOLATION_MODELS, POINT_SOURCE_MODELS, check_redistribution, insolation,
                                   insolation_fullyilluminated_hyperillum, insolation_penumbra_hyperillum, naive_insolation,
                                   temperature_from_insolation)
from hyperphase.orbit import occultation_time, true_anomaly
from hyperphase.stellar import T_bright, check_teff_nodes, stellar_spidergrid
from hyperphase.transit import flux_occultation, flux_transit, light_travel_time_correction


sigma = const.sigma_sb.value


############################################################################################################################################
#####   Full phase curve (reflected + thermal)
############################################################################################################################################
def full_PhaseCurve(planet, lambda_range, t=None, t0=0., u=DEFAULT_U, u_bol=DEFAULT_U_BOL, A=DEFAULT_A, n_eta=150, n_phi=80, n_tab=1000, n_lo=DEFAULT_N_LO, n_la=DEFAULT_N_LA, insolation_model="hyperillum", F=None, stellar_model=STELLAR_MODEL, filter_path=False, A_B=DEFAULT_A_B, emissivity_bol=DEFAULT_EMISSIVITY_BOL, emissivity=DEFAULT_EMISSIVITY, N_LAYERS=None, include_transit=True, include_occultation=True, spman="fast", spman_kw=None):
    """Thermal plus reflected phase curve of a bare rock in one band, with the transit and the occultation.

    planet: PlanetParams. lambda_range: band edges [m]; filter_path: throughput file (bands.filter_loc), False for a top-hat over lambda_range.
    t: times [days] (default: 1000 points over one orbit); t0: time of mid-transit [days]. insolation_model: one of irradiance.INSOLATION_MODELS,
    F its heat redistribution. A_B: Bond albedo; A: band albedo of the reflected light; emissivity_bol, emissivity: bolometric and band
    emissivity; u, u_bol: limb darkening of the transit (quadratic) and of the irradiance (linear). The rest as in thermal_PhaseCurve and
    reflected_PhaseCurve.

    Returns (t, flux, depth): the times [days] corrected for the light travel time, the flux relative to the star alone (1 + planet - transit), and
    the occultation depth [ppm] (thermal plus reflected planet flux at full phase)."""
    check_redistribution(insolation_model, F)  # fail before the reflected part is computed
    # Reflected part, with the transit
    t_comb, refl, fp_refl = reflected_PhaseCurve(planet, t=t, t0=t0, u=u, u_bol=u_bol, include_transit=include_transit, include_occultation=include_occultation, A=A, n_eta=n_eta, n_phi=n_phi, n_tab=n_tab, insolation_model=insolation_model)
    # Thermal part, without the transit
    _,  therm, fp_therm, _ = thermal_PhaseCurve(planet, lambda_range, t=t_comb, t0=t0, u=u, u_bol=u_bol, n_lo=n_lo, n_la=n_la, insolation_model=insolation_model, F=F, stellar_model=stellar_model, filter_path=filter_path, A_B = A_B, emissivity_bol=emissivity_bol, emissivity=emissivity, include_transit=False, include_occultation=include_occultation, N_LAYERS=N_LAYERS, spman=spman, spman_kw=spman_kw)

    total = refl + therm
    fp = fp_refl + fp_therm

    return t_comb, total, fp


def nightside_flux(planet, lambda_range, insolation_model="hyperillum", unit="flux", t0=0., u_bol=DEFAULT_U_BOL, A_B=DEFAULT_A_B, A=DEFAULT_A, F=None, emissivity_bol=DEFAULT_EMISSIVITY_BOL, emissivity=DEFAULT_EMISSIVITY, filter_path=False, stellar_model=STELLAR_MODEL, N_LAYERS=None, n_lo=DEFAULT_N_LO, n_la=DEFAULT_N_LA, n_eta=150, n_phi=80, n_tab=1000):
    """Planet flux at mid-transit (t = t0), with the nightside towards the observer, without the transit. It is the visible hemisphere, not the
    antistellar point: for i < 90 deg it includes a sliver of dayside past the pole, all there is under "naive" (exactly 0 edge-on). Parameters
    as in full_PhaseCurve.

    unit="flux": returns (thermal, reflected) F_p/F_star [ppm].
    unit="temperature": returns T_bright [K] of the thermal part, 0 when it is <= 0 or below the band flux at 10 K."""
    if unit not in ("flux", "temperature"): raise ValueError(f"unit must be 'flux' or 'temperature', got {unit!r}")
    t = np.array([t0])
    _, therm, _, _ = thermal_PhaseCurve(planet, lambda_range, t=t, t0=t0, u_bol=u_bol, n_lo=n_lo, n_la=n_la, insolation_model=insolation_model, F=F, stellar_model=stellar_model, filter_path=filter_path, A_B=A_B, emissivity_bol=emissivity_bol, emissivity=emissivity, N_LAYERS=N_LAYERS, include_transit=False, spman="normal")
    therm = 1e6 * float(therm[0])
    if unit == "temperature":
        if therm <= 0.: return 0.
        try: return float(T_bright(therm, planet, lambda_range, filter_path=filter_path or None, stellar_model=stellar_model))
        except ValueError: return 0.   # no sign change for brentq: below the band flux at 10 K
    _, refl, _ = reflected_PhaseCurve(planet, t=t, t0=t0, u_bol=u_bol, include_transit=False, A=A, n_eta=n_eta, n_phi=n_phi, n_tab=n_tab, insolation_model=insolation_model)
    return therm, 1e6 * float(refl[0])


############################################################################################################################################
#####   Reflected light curve
############################################################################################################################################

def reflected_lightcurve(epsilon, planet, u_bol=DEFAULT_U_BOL, unit="normalized", lambda_band_ratio=1., A=DEFAULT_A, n_eta=150, n_phi=80, n_tab=1000, runtime=False, insolation_model="hyperillum"):
    """Reflected light of a hyper-illuminated planet, Eq. (A11) of the paper (Spiller, Zieba & Wordsworth, in prep.): Lambertian (isotropic,
    grey) reflection by a spherical planet.

    epsilon: angle between the substellar point and the line of sight [rad].
    unit: "normalized" returns F_p/F_star, "radiant intensity" the power per solid angle towards the observer [W/sr] (the flux at distance d is
        that / d^2).
    lambda_band_ratio: for unit="radiant intensity" in a band, F_star,band / F_star (for a throughput T(lambda): int T B_lambda(T_eff) dlambda /
        sigma T_eff^4).
    A: spherical albedo, bolometric or band-integrated (not a geometric albedo). For a bolometric measurement, A = 1 gives reflected plus thermal
        light.
    n_eta, n_phi: Gauss-Legendre orders of the inner (eta) and outer (phi) integral; n_tab: gamma1 samples of the tabulated irradiance."""
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    t0 = time.perf_counter()
    epsilon = np.asarray(epsilon, dtype=float)
    eps_flat = epsilon.ravel()

    gp1 = planet.terminator_penumbra()  # arccos((r1+r2)/R)
    gp2 = planet.terminator_nightside()  # arccos((r1-r2)/R)

    def F(gp):
        if insolation_model in EXTENDED_SOURCE_MODELS:
            # tabulate F_fi, F_pen on a gamma1-grid once and interpolate at quadrature nodes
            gp_fi_grid = np.linspace(0., gp1, n_tab)
            Ffi_grid = insolation_fullyilluminated_hyperillum(gp_fi_grid, planet)

            gp_pen_grid = np.linspace(gp1, gp2, n_tab)
            Fpen_grid = insolation_penumbra_hyperillum(gp_pen_grid, planet, u_bol=u_bol)

            return np.where(
                gp <= gp1, np.interp(gp, gp_fi_grid, Ffi_grid),
                np.where(gp <= gp2, np.interp(gp, gp_pen_grid, Fpen_grid), 0.))

        elif insolation_model in POINT_SOURCE_MODELS:
            return naive_insolation(gp, planet)

        else:
            raise ValueError(f"insolation_model must be one of {INSOLATION_MODELS}")

    # fixed Gauss-Legendre nodes, shared across all epsilon values
    psi_nodes, psi_w = np.polynomial.legendre.leggauss(n_phi)
    psi = psi_nodes * (np.pi / 2.)  # [-1,1] -> [-pi/2, pi/2]
    psi_w = psi_w * (np.pi / 2.)

    eta_nodes, eta_w = np.polynomial.legendre.leggauss(n_eta)
    eta = eta_nodes * (np.pi / 2.) + (np.pi / 2.)  # [-1,1] -> [0, pi]
    eta_w = eta_w * (np.pi / 2.)

    # broadcast to (n_epsilon, n_phi, n_eta)
    phi = eps_flat[:, None, None] + psi[None, :, None]
    cos_gp = np.sin(eta)[None, None, :] * np.cos(phi)
    gp = np.arccos(np.clip(cos_gp, -1., 1.))

    integrand = F(gp) * np.sin(eta)[None, None, :] ** 2
    inner = integrand @ eta_w  # (n_epsilon, n_phi)
    outer = inner @ (np.cos(psi) * psi_w)  # (n_epsilon,)

    result = (A * r1 ** 2 / np.pi) * outer
    result = result.reshape(epsilon.shape)

    if runtime: print(f"reflected_lightcurve runtime: {time.perf_counter() - t0:.4f} s")

    if unit == "normalized":
        return result / (sigma * r2 ** 2 * T2_eff ** 4)  # Fp/Fs
    elif unit == "radiant intensity":
        return result * lambda_band_ratio  # [W/sr]
    else:
        raise ValueError("unit must be either 'normalized' or 'radiant intensity'")


def lambertian_reflectance(epsilon, planet, unit="normalized", lambda_band_ratio=1., A=DEFAULT_A):
    """Reflected light of a Lambertian sphere lit by a point source, with the phase function (sin eps + (pi - eps) cos eps) / pi. Parameters and
    units as in reflected_lightcurve."""
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    eps = np.abs(((epsilon + np.pi) % (2 * np.pi)) - np.pi)  # fold into [0, pi]
    phase_function = (np.sin(eps) + (np.pi - eps) * np.cos(eps)) / np.pi

    if unit == "normalized":
        result = 2 / 3 * A * (r1 / R) ** 2 * phase_function
        return result  # Fp/Fs
    elif unit == "radiant intensity":
        Fp_incident = (r2 / R) ** 2 * sigma * T2_eff ** 4
        result = 2 / 3 * A * r1 ** 2 * Fp_incident * phase_function * lambda_band_ratio
        return result  # [W/sr]
    else:
        raise ValueError("unit must be either 'normalized' or 'radiant intensity'")


############################################################################################################################################
#####   Reflected Phase Curve
############################################################################################################################################

def orbital_reflected_lightcurve(t, planet, t0=0., n_eps=300, A=DEFAULT_A, u_bol=DEFAULT_U_BOL, n_eta=150, n_phi=80, n_tab=1000, insolation_model="hyperillum"):
    """reflected_lightcurve (F_p/F_star) at the times t [days], t0 the time of mid-transit."""
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    f = true_anomaly(t, period, ecc, w, t0=t0)
    epsilon = np.arccos(-np.sin(np.radians(incl)) * np.sin(np.radians(w) + f))

    # the expensive integral on a coarse monotonic epsilon grid, interpolated onto t
    eps_grid = np.linspace(epsilon.min(), epsilon.max(), n_eps)
    ref_grid = reflected_lightcurve(eps_grid, planet, u_bol=u_bol, A=A, n_eta=n_eta, n_phi=n_phi, n_tab=n_tab, insolation_model=insolation_model)
    return np.interp(epsilon, eps_grid, ref_grid)


def reflected_PhaseCurve(planet, t=None, t0=0., u=DEFAULT_U, u_bol=DEFAULT_U_BOL, include_transit=True, include_occultation=True, A=DEFAULT_A, n_eta=150, n_phi=80, n_tab=1000, insolation_model="hyperillum"):
    """Reflected-light phase curve (Lambertian, grey) with the transit and the occultation. With the band albedo A it holds in any band: the
    reflected F_p/F_star does not depend on the stellar spectrum, up to the stellar limb darkening and reddening in the penumbra, which are
    neglected (weak in the infrared). Parameters as in full_PhaseCurve.

    Returns (t, flux, depth): the times [days] corrected for the light travel time, the flux relative to the star alone (1 + planet - transit; the
    planet flux only for include_transit=False), and the reflected flux at full phase [ppm]."""
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    if t is None: t = np.linspace(0., period, 1000, endpoint=False)
    t_corr = light_travel_time_correction(t, planet, t0=t0)

    orbital_ref_lightcurve = orbital_reflected_lightcurve(t, planet, t0=t0, A=A, u_bol=u_bol, n_eta=n_eta, n_phi=n_phi, n_tab=n_tab, insolation_model=insolation_model)

    eps_occultation = np.arccos(np.sin(np.radians(incl)))  # epsilon at full phase
    fp = reflected_lightcurve(eps_occultation, planet, A=A, u_bol=u_bol, n_eta=n_eta, n_phi=n_phi, n_tab=n_tab, insolation_model=insolation_model)
    if fp == 0.: return t_corr, include_transit*flux_transit(planet, t, t0=t0, u=u), fp*1e6  # no reflected light (A = 0): transit only

    # The occultation treats the reflected light as a uniformly bright disk: its ingress and egress ignore the brightness distribution
    if include_occultation:
        phase_mod = orbital_ref_lightcurve / fp
        occultation = flux_occultation(planet, fp, t, t0=t0)

        if include_transit:
            transit = flux_transit(planet, t, t0=t0, u=u)
            phase_curve = transit + (occultation - 1) * phase_mod
        else:
            phase_curve = (occultation - 1) * phase_mod
    else:
        if include_transit:
            transit = flux_transit(planet, t, t0=t0, u=u)
            phase_curve = transit + orbital_ref_lightcurve
        else:
            phase_curve = orbital_ref_lightcurve

    return t_corr, phase_curve, fp*1e6


############################################################################################################################################
#####   Thermal Phase Curve (Spiderman)
############################################################################################################################################

def occultation_depth(spider_params, stellar_grid, emissivity=DEFAULT_EMISSIVITY):
    """Occultation depth [ppm] of a SPIDERMAN model: the flux at mid-occultation without minus with the occultation. spider_params.eclipse_depth()
    cannot be used: it segfaults with direct_T. emissivity: the band emissivity of the model (see thermal_PhaseCurve)."""
    set_ecl = spider_params.eclipse
    t_occ = occultation_time(spider_params.per, spider_params.ecc, spider_params.w, t0=spider_params.t0)
    spider_params.eclipse = False; f_unblocked_occ = spider_params.lightcurve([t_occ], stellar_grid=stellar_grid)[0]
    spider_params.eclipse = True;  f_blocked_occ = spider_params.lightcurve([t_occ], stellar_grid=stellar_grid)[0]
    spider_params.eclipse = set_ecl
    return emissivity * (f_unblocked_occ - f_blocked_occ)*1e6 # Fp/Fs(alpha=0) in ppm


def insolation_grid(planet, u_bol=DEFAULT_U_BOL, n_lo=DEFAULT_N_LO, n_la=DEFAULT_N_LA, insolation_model="hyperillum"):
    """The A_B-independent part of custom_temperature_grid: the irradiance map on the (lo, la) grid. It depends on the geometry only, so a fit that
    varies only A_B builds it once (model.insolation_cache, model.model_pcurve(insol=...)).

    Returns (lo, la, I): longitudes [deg] (0 = substellar meridian), latitudes [deg], and the irradiance [W m^-2] with shape (len(la), len(lo))."""
    lo = np.linspace(-180, 180, n_lo)  # longitude [deg], 0 = substellar meridian
    la = np.linspace(-90, 90, n_la)  # latitude [deg], 0 = equator
    LO, LA = np.meshgrid(np.radians(lo), np.radians(la))  # shape (n_la, n_lo), radians

    gamma1_grid = np.arccos(np.cos(LA) * np.cos(LO))  # angular distance from substellar point
    return lo, la, insolation(gamma1_grid, planet, insolation_model, u_bol=u_bol)


def custom_temperature_grid(planet, u_bol=DEFAULT_U_BOL, n_lo=DEFAULT_N_LO, n_la=DEFAULT_N_LA, insolation_model="hyperillum", F=None, A_B=DEFAULT_A_B, emissivity_bol=DEFAULT_EMISSIVITY_BOL, insol=None):
    """Surface-temperature map of insolation_model on a longitude-latitude grid, the input of SPIDERMAN's direct_T model.

    insol: optional precomputed (lo, la, irradiance map) from insolation_grid / model.insolation_cache; then only the
    ((1-A_B)/emissivity_bol)^(1/4) rescale is done. It holds for any F. F: heat redistribution of the redistribution models.

    Returns (lo, la, T): longitudes [deg] (0 = substellar meridian), latitudes [deg], and the temperature [K] with shape (len(la), len(lo))."""
    check_redistribution(insolation_model, F)
    if insol is None:
        lo, la, insol_grid = insolation_grid(planet, u_bol=u_bol, n_lo=n_lo, n_la=n_la, insolation_model=insolation_model)
    else:
        lo, la, insol_grid = insol
        if (len(lo), len(la)) != (n_lo, n_la) or insol_grid.shape != (len(la), len(lo)):
            raise ValueError(f"cached insol is on a {(len(lo), len(la))} grid but n_lo, n_la = {(n_lo, n_la)}")

    temp_grid = temperature_from_insolation(insol_grid, planet, insolation_model, A_B = A_B, emissivity_bol=emissivity_bol, F=F, u_bol=u_bol)

    return lo, la, temp_grid


def thermal_PhaseCurve(planet, lambda_range, t=None, t0=0., u=DEFAULT_U, u_bol=DEFAULT_U_BOL, n_lo=DEFAULT_N_LO, n_la=DEFAULT_N_LA, insolation_model="hyperillum", F=None, stellar_model=STELLAR_MODEL, filter_path=False, A_B=DEFAULT_A_B, emissivity_bol=DEFAULT_EMISSIVITY_BOL, emissivity=DEFAULT_EMISSIVITY, N_LAYERS=None, include_transit=True, include_occultation=True, spman="fast", spman_kw=None):
    """Thermal phase curve from SPIDERMAN's direct_T model with the temperature map of insolation_model ("naive_redist": see Louden_PhaseCurve).

    lambda_range: band edges [m]; SPIDERMAN integrates a top-hat over them for filter_path=False, else the throughput file. F: heat redistribution
    of the redistribution models. emissivity_bol sets the temperature map with A_B. emissivity scales the planet flux and the depth, not the
    transit: SPIDERMAN emits the Planck function, so the thermal flux is exactly linear in it. N_LAYERS: radial layers of SPIDERMAN's map, by
    default 40 for the extended source and 60 for the point source, whose sharp terminator makes the occultation depth converge more slowly (~1/N
    against ~1/N^2); use >= 120 to compare depths between models. spman="fast" interpolates the phase modulation (fast_lightcurve, options in
    spman_kw), "normal" evaluates every time.

    Returns (t, flux, depth, params): the times [days], the flux relative to the star alone (1 + planet - transit; the planet flux only for
    include_transit=False), the occultation depth [ppm], and the SPIDERMAN ModelParams."""
    check_redistribution(insolation_model, F)
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    if N_LAYERS is None:
        if insolation_model in EXTENDED_SOURCE_MODELS: N_LAYERS = 40
        else: N_LAYERS = 60

    if insolation_model == "naive_redist":
        return Louden_PhaseCurve(planet, lambda_range, t=t, t0=t0, F=F, u=u, stellar_model=stellar_model, filter_path=filter_path, A_B = A_B, emissivity_bol=emissivity_bol, emissivity=emissivity, N_LAYERS=N_LAYERS, include_transit=include_transit, include_occultation=include_occultation, spman=spman, spman_kw=spman_kw)

    lo, la, temp_grid = custom_temperature_grid(planet, u_bol=u_bol, n_lo=n_lo, n_la=n_la, insolation_model=insolation_model, F=F, A_B = A_B, emissivity_bol=emissivity_bol)

    # format_grid takes indexing='ij' meshgrids in deg and the transposed temp_grid: a wrong orientation fails silently (rotated planet).
    # No temperature floor: fspiderman clamps the direct_T bicubic to the cell corners (github.com/JonasSpiller/fSPIDERMAN, README "4. Bicubic
    # clamp"), so the 0 K nightside of "naive" goes in as is. Upstream SPIDERMAN overshoots below 0 K at the sharp terminator and exits ("CANNOT
    # DO NEGATIVE TEMPERATURES"), so it would need a floor.
    LO_deg, LA_deg = np.meshgrid(lo, la, indexing='ij')  # longitude/latitude meshgrids [deg]
    grid = sp.format_grid(LO_deg, LA_deg, temp_grid.T)

    spider_params = sp.ModelParams(brightness_model="direct_T")  # the stellar SED enters only through stellar_grid
    spider_params.n_layers = N_LAYERS

    spider_params.t0 = t0  # time of mid-transit [days]
    spider_params.per = period  # [days]
    spider_params.a_abs = a_AU  # semi-major axis [AU], for the light travel time
    spider_params.inc = incl  # [deg]
    spider_params.ecc = ecc
    spider_params.w = w  # argument of periastron [deg]
    spider_params.rp = r1/r2
    spider_params.a = R/r2
    spider_params.p_u1 = spider_params.p_u2 = 0.  # planetary limb darkening

    # Parameters specific to the direct_T model
    spider_params.T_s = T2_eff  # stellar T_eff [K]
    spider_params.grid = grid

    spider_params.l1, spider_params.l2 = lambda_range  # band edges [m]

    spider_params.filter = filter_path
    stellar_grid = stellar_spidergrid(lambda_range, filter_path, planet.logg, planet.feh, stellar_model=stellar_model)
    check_teff_nodes(stellar_grid, T2_eff)

    fp = occultation_depth(spider_params, stellar_grid=stellar_grid, emissivity=emissivity)  # [ppm]

    spider_params.eclipse = include_occultation

    if t is None: t = np.linspace(0., period, 1000, endpoint=False)
    if spman=="fast": light_curve = fast_lightcurve(spider_params, stellar_grid, t, **(spman_kw or {}))  # phase modulation + occultation, no transit
    else: light_curve = spider_params.lightcurve(t, stellar_grid=stellar_grid)  # phase modulation + occultation, no transit

    # light_curve - 1: the planet's thermal flux
    if include_transit: # Full PhaseCurve
        f_transit = flux_transit(planet, t, t0=t0, u=u)
        phase_curve = f_transit + emissivity * (light_curve - 1.)
    else: # Only Lightcurve, no stellar flux
        phase_curve = emissivity * (light_curve - 1.)

    return t, phase_curve, fp, spider_params


def louden_params(planet, lambda_range, F, t0=0., filter_path=False, A_B=DEFAULT_A_B, emissivity_bol=DEFAULT_EMISSIVITY_BOL, N_LAYERS=60):
    """The sp.ModelParams of SPIDERMAN's Louden model with the heat redistribution F (see Louden_PhaseCurve); its eclipse flag is left to the caller."""
    check_redistribution("naive_redist", F)
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    insol = sigma * T2_eff**4 / (R/r2)**2

    sp_louden = sp.ModelParams(brightness_model="Louden")  # the stellar SED enters only through stellar_grid
    sp_louden.thermal = True
    sp_louden.n_layers = N_LAYERS

    sp_louden.t0, sp_louden.per = t0, period
    sp_louden.a_abs = a_AU
    sp_louden.inc, sp_louden.ecc, sp_louden.w = incl, ecc, w
    sp_louden.rp, sp_louden.a = r1/r2, R/r2
    sp_louden.p_u1 = sp_louden.p_u2 = 0. # planetary limb darkening

    sp_louden.T_s = T2_eff
    sp_louden.insol = insol
    # Louden's sigma T^4 = (1-albedo) S [...] has no emissivity: this effective albedo gives emissivity_bol sigma T^4 = (1-A_B) S [...], as in
    # BBtemp_from_insolation
    sp_louden.albedo = 1. - (1. - A_B) / emissivity_bol
    sp_louden.redist = F  # Kreidberg & Loeb 2016 (F=0: instant re-radiation, F=0.5: full redistribution)
    sp_louden.T_int = 0. # no internal heating
    sp_louden.l1, sp_louden.l2 = lambda_range

    sp_louden.filter = filter_path
    return sp_louden


def Louden_PhaseCurve(planet, lambda_range, t=None, t0=0., F=0., u=DEFAULT_U, stellar_model=STELLAR_MODEL, filter_path=False, A_B=DEFAULT_A_B, emissivity_bol=DEFAULT_EMISSIVITY_BOL, emissivity=DEFAULT_EMISSIVITY, N_LAYERS=60, include_transit=True, include_occultation=True, spman="fast", spman_kw=None):
    """Thermal phase curve of "naive_redist": the point-source map with the heat redistribution F in [0, 0.5] (Kreidberg & Loeb 2016, eq. 1) is
    analytic, so SPIDERMAN's Louden model evaluates it instead of a direct_T grid (u_bol, n_lo and n_la do not apply). With F = 0 it equals
    thermal_PhaseCurve(..., insolation_model="naive"), but faster. Parameters and returns as in thermal_PhaseCurve."""
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    sp_louden = louden_params(planet, lambda_range, F, t0=t0, filter_path=filter_path, A_B = A_B, emissivity_bol=emissivity_bol, N_LAYERS=N_LAYERS)

    stellar_grid = stellar_spidergrid(lambda_range, filter_path, planet.logg, planet.feh, stellar_model=stellar_model)
    check_teff_nodes(stellar_grid, T2_eff)

    fp = occultation_depth(sp_louden, stellar_grid=stellar_grid, emissivity=emissivity)  # [ppm]

    sp_louden.eclipse = include_occultation

    if t is None: t = np.linspace(0., period, 1000, endpoint=False)
    if spman=="fast": light_curve = fast_lightcurve(sp_louden, stellar_grid, t, **(spman_kw or {}))  # phase modulation + occultation, no transit
    else: light_curve = sp_louden.lightcurve(t, stellar_grid=stellar_grid)  # phase modulation + occultation, no transit

    # light_curve - 1: the planet's thermal flux
    if include_transit:  # Full PhaseCurve
        f_transit = flux_transit(planet, t, t0=t0, u=u)
        phase_curve = f_transit + emissivity * (light_curve - 1.)
    else:  # Only Lightcurve, no stellar flux
        phase_curve = emissivity * (light_curve - 1.)

    return t, phase_curve, fp, sp_louden


############################################################################################################################################
#####   Fast light curve
############################################################################################################################################
def _trig_interp(y_nodes, per, t, t0=0.):
    """Trigonometric interpolation of a periodic signal sampled uniformly on [t0, t0 + per) (no endpoint); exact if it is band-limited."""
    n = len(y_nodes)
    c = np.fft.rfft(y_nodes) / n
    ph = 2 * np.pi * (t - t0) / per
    out = np.full_like(np.asarray(t, dtype=float), c[0].real)
    for k in range(1, n // 2 + 1):
        amp = 2 * c[k] if k < n / 2 else c[k]
        out = out + amp.real * np.cos(k * ph) - amp.imag * np.sin(k * ph)
    return out


def occultation_durations(spider_params):
    """(T14, T23) of the occultation for a circular orbit: total duration (1st-4th contact) and totality (2nd-3rd contact), Winn 2010 eq. 14.
    T23 = 0 for a grazing geometry, where the planet is never fully hidden."""
    inc = np.radians(spider_params.inc)
    b = spider_params.a * np.cos(inc)
    # sin(inc) as in orbit.transit_duration_t14; min(..., 1.) gives the half-orbit limit instead of NaN when the planet never clears the disk
    dur = lambda x: spider_params.per / np.pi * np.arcsin(min(np.sqrt(max(x ** 2 - b ** 2, 0.)) / (spider_params.a * np.sin(inc)), 1.))
    return dur(1 + spider_params.rp), dur(1 - spider_params.rp)


def occultation_window(spider_params, pad=0.03):
    """(t_ingress, t_egress) of the occultation, widened by pad * T14 on each side."""
    t_occultation = occultation_time(spider_params.per, spider_params.ecc, spider_params.w, t0=spider_params.t0)
    T14, _ = occultation_durations(spider_params)
    return t_occultation - (0.5 + pad) * T14, t_occultation + (0.5 + pad) * T14


def _dt_to_occultation(spider_params, t):
    """Signed offset of each t to the nearest mid-occultation, in any orbit spanned by t."""
    t_occultation = occultation_time(spider_params.per, spider_params.ecc, spider_params.w, t0=spider_params.t0)
    per = spider_params.per
    return np.mod(np.asarray(t, dtype=float) - t_occultation + 0.5 * per, per) - 0.5 * per


def in_occultation(spider_params, t, pad=0.03):
    """Mask of the t inside an occultation window (as occultation_window), in any orbit spanned by t."""
    T14, _ = occultation_durations(spider_params)
    return np.abs(_dt_to_occultation(spider_params, t)) <= (0.5 + pad) * T14


def fast_lightcurve(spider_params, stellar_grid, t, n_phase=40, pad=0.03, split_totality=True):
    """SPIDERMAN's lightcurve, evaluated cheaply (circular orbit). Out of occultation the smooth phase modulation, which has only a few Fourier
    terms, is interpolated from n_phase uniform nodes (_trig_interp). Inside totality the planet is hidden, so the flux is exactly the stellar flux
    (1) and SPIDERMAN is not called; only the ingress and egress, padded by pad * T14, are evaluated exactly. split_totality=False evaluates the
    whole occultation window. A single NaN node spreads over the whole interpolated curve."""
    t = np.asarray(t, dtype=float)
    orig_eclipse = spider_params.eclipse

    try:
        spider_params.eclipse = False # smooth phase modulation, no occultation
        t_nodes = spider_params.t0 + spider_params.per * np.arange(n_phase) / n_phase
        y_nodes = spider_params.lightcurve(t_nodes, stellar_grid=stellar_grid)
        out = _trig_interp(y_nodes, spider_params.per, t, t0=spider_params.t0)

        if orig_eclipse:  # exact inside EVERY occultation spanned by t
            T14, T23 = occultation_durations(spider_params)
            dt = np.abs(_dt_to_occultation(spider_params, t))
            m = dt <= (0.5 + pad) * T14  # padded occultation window
            # totality, pulled back by pad*T14 from the 2nd/3rd contact; <= 0 when grazing: then the whole window is exact
            half_tot = 0.5 * T23 - pad * T14
            if split_totality and half_tot > 0.:
                tot = dt < half_tot
                out[tot] = 1.  # planet hidden: exactly the (normalised) stellar flux
                m &= ~tot  # only the two ingress/egress ramps are left
            if m.any():
                spider_params.eclipse = True
                out[m] = spider_params.lightcurve(t[m], stellar_grid=stellar_grid) # occultation calculated exactly, one call for both ramps
    finally:
        spider_params.eclipse = orig_eclipse # restore even if lightcurve() raised

    return out
