# translating the insolation model with hyperillumination to a light curve signal
import time
import numpy as np
import spiderman as sp
from hyperillumination_utils import *
import kepler_utils
import batman_utils

sigma = 5.670374e-8  # W m^-2 K^-4


###########################################################################################################################
#####
#####   Utilities
#####
###########################################################################################################################


###########################################################################################################################
#####
#####   Reflected Lightcurve
#####
###########################################################################################################################


# Reflected light curve signal accounting for hyperillumination
# Inspired by Kopal eq. 4
def reflected_lightcurve(epsilon, r1, r2, R, T2_eff, u=0.0, unit="normalized", lambda_band_ratio=1., A=0.1, n_eta=150,
                         n_phi=80, n_tab=1000, runtime=False, insolation_model="kopal"):
    """
    epsilon: angle substellar spot to line of sight [rad]
    unit: Whether to return normalized light curve signal (Fp/Fs) or radiant intensity ([W/sr])
    lambda_band_ratio: Only relevant for unit=='radiant intensity'. If measurements are taken in a wavelength band (not bolometric), this is the ratio Fs_band/Fs. Needed to correctly scale the absolute radial intensity in that band.
    A: Bolometric or band integrated albedo. For bolometric measurements, this function is also suitable to calculate reflected + THERMAL light curve signal - for this set A=1.
    n_eta, n_phi: Gauss-Legendre order for the inner (eta) / outer (phi) integral
    n_tab: resolution of the gamma1-grid used to tabulate F_fi/F_pen
    """

    t0 = time.perf_counter()
    epsilon = np.asarray(epsilon, dtype=float)
    eps_flat = epsilon.ravel()

    gp1 = terminator_penumbra(r1, r2, R)  # arccos((r1+r2)/R)
    gp2 = terminator_nightside(r1, r2, R)  # arccos((r1-r2)/R)

    def F(gp):
        if insolation_model == "kopal":
            # tabulate F_fi, F_pen on a gamma1-grid once and interpolate at quadrature nodes
            gp_fi_grid = np.linspace(0., gp1, n_tab)
            Ffi_grid = insolation_fullyilluminated_kopal(gp_fi_grid, r1, r2, R, T2_eff, u=u)

            gp_pen_grid = np.linspace(gp1, gp2, n_tab)
            Fpen_grid = insolation_penumbra_kopal(gp_pen_grid, r1, r2, R, T2_eff, u=u)

            return np.where(
                gp <= gp1, np.interp(gp, gp_fi_grid, Ffi_grid),
                np.where(gp <= gp2, np.interp(gp, gp_pen_grid, Fpen_grid), 0.))

        elif insolation_model == "naive":  # naive (point-like) insolation model:
            return naive_insolation(gp, r2, R, T2_eff)

        else:
            raise ValueError("insolation_model must be either 'kopal' or 'naive'")

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
        return result * lambda_band_ratio  # radiant intensity (power/solid angle emitted towards the observer [W/sr]) - measured is flux [W/m^2]: result/d^2
    else:
        raise ValueError("unit must be either 'normalized' or 'radiant intensity'")


# Naive reflected phase curve signal for a Lambert-scattering sphere (isotropic, angle-independent single-scattering albedo)
def lambertian_reflectance(epsilon, r1, r2, R, T2_eff, unit="normalized", lambda_band_ratio=1., A=0.1):
    """
    epsilon: Angle substellar spot to line of sight
    unit: Whether to return normalized light curve signal (Fp/Fs) or radiant intensity ([W/sr])
    lambda_band_ratio: Only relevant for unit=='radiant intensity'. If measurements are taken in a wavelength band (not bolometric), this is the ratio Fs_band/Fs.
    A: Bolometric or band integrated albedo. For bolometric measurements, this function is also suitable to calculate reflected + THERMAL light curve signal - for this set A=1.
    """
    eps = np.abs(((epsilon + np.pi) % (2 * np.pi)) - np.pi)  # fold into [0, pi]
    phase_function = (np.sin(eps) + (np.pi - eps) * np.cos(eps)) / np.pi

    if unit == "normalized":
        result = 2 / 3 * A * (r1 / R) ** 2 * phase_function
        return result  # Fp/Fs
    elif unit == "radiant intensity":
        Fp_incident = (r2 / R) ** 2 * sigma * T2_eff ** 4
        result = 2 / 3 * A * r1 ** 2 * Fp_incident * phase_function * lambda_band_ratio
        return result  # radiant intensity (power/solid angle emitted towards the observer [W/sr]) - measured is flux [W/m^2]: result/d^2
    else:
        raise ValueError("unit must be either 'normalized' or 'radiant intensity'")


###########################################################################################################################
#####
#####   Reflected Phase Curve
#####
###########################################################################################################################

# Reflected light curve evaluated at times t.
def orbital_reflected_lightcurve(t, period, incl, ecc, w, r1, r2, a, T2_eff, t0=0., n_eps=300,
                                 **kwargs):  #TODO add , u=u but only when hi model quadratic
    f = kepler_utils.true_anomaly(t, period, ecc, w, t0=t0)
    epsilon = np.arccos(-np.sin(np.radians(incl)) * np.sin(np.radians(w) + f))

    # momentary planetary distance from stellar center
    # R = a * (1. - ecc**2) / (1. + ecc * np.cos(f))
    R = a  # THIS NEGLECTS CHANGES IN R THROUGHOUT THE ORBIT! All functions are designed for scalar R, which is why the formula above currently fails. Needs adjustment for ecc>0.

    # evaluating the expensive integral on a coarse monotonic epsilon grid, then interpolate onto t
    eps_grid = np.linspace(epsilon.min(), epsilon.max(), n_eps)
    ref_grid = reflected_lightcurve(eps_grid, r1, r2, R, T2_eff,
                                    **kwargs)  #TODO add , u=u but only when hi model quadratic
    return np.interp(epsilon, eps_grid, ref_grid)


# Returns (t, flux): timestamps and the full reflected phase curve combining transit, secondary eclipse / occultation, and sinusoidal phase modulation.
# Reflected phasecurve valid for any wavelength range as wavelength dependence cancels out in Fp/Fs (Fp linearly dependent on Fs)
def reflected_PhaseCurve(period, incl, ecc, w, r1, r2, a, T2_eff, R_star, t=None, t0=0., u=[0.1, 0.3], **kwargs):
    #  r1, r2, a: planetary radius, stellar radius, semi major axis in consistent units
    # R_star: stellar radius in solar radii
    kwargs.pop("unit", None)  # no unit can be provided, must be "normalized"
    if t is None: t = np.linspace(0., period, 1000)

    t_corr = batman_utils.light_travel_time_correction(t, period, ecc, w, incl, a / r2, R_star, t0=t0)

    orbital_ref_lightcurve = orbital_reflected_lightcurve(t, period, incl, ecc, w, r1, r2, a, T2_eff, t0=t0,
                                                          **kwargs)  #TODO add , u=u but only when hi model quadratic
    eps_occultation = np.arccos(np.sin(np.radians(incl)))  # epsilon at full phase
    fp = reflected_lightcurve(eps_occultation, r1, r2, a, T2_eff, **kwargs)
    if fp == 0.: return t_corr, batman_utils.flux_transit(period, r1 / r2, a / r2, incl, ecc, w, t, t0=t0,
                                                          u=u)  # no reflection (e.g. when albedo=0) -> no phase curve, only transit
    phase_mod = orbital_ref_lightcurve / fp

    # Full Phase Curve Model. This model ignores the exact eclipse geometry during ingress and egress, instead it assumes the integrated flux to be uniformly distributed over the disk
    phase_curve = batman_utils.flux_transit(period, r1 / r2, a / r2, incl, ecc, w, t, t0=t0, u=u) + (
                batman_utils.flux_occultation(period, r1 / r2, a / r2, incl, ecc, w, fp, t, t0=t0, u=u) - 1) * phase_mod

    return t_corr, phase_curve, fp


###########################################################################################################################
#####
#####   Thermal Phase Curve (Spiderman)
#####
###########################################################################################################################


# Generated custom temperature grid from a given insolation model (insolation_model = hyperillumination, naive) as input for spiderman custom forward model
def custom_temperature_grid(r1, r2, R, T2_eff, u=0, n_lo=200, n_la=100, insolation_model="kopal", A_B = 0.1, emissivity=1.):
    lo = np.linspace(-180, 180, n_lo)  # longitude [deg], 0 = substellar meridian
    la = np.linspace(-90, 90, n_la)  # latitude [deg], 0 = equator
    LO, LA = np.meshgrid(np.radians(lo), np.radians(la))  # shape (n_la, n_lo), radians

    gamma1_grid = np.arccos(np.cos(LA) * np.cos(LO))  # angular distance from substellar point
    if insolation_model == "kopal":
        insol_grid = insolation_kopal(gamma1_grid, r1, r2, R, T2_eff)  #TODO add , u=u but only when hi model quadratic
    elif insolation_model == "naive":
        insol_grid = naive_insolation(gamma1_grid, r2, R, T2_eff)
    else:
        raise ValueError("insolation_model must be either 'kopal' or 'naive'")
    temp_grid = BBtemp_from_insolation(insol_grid, A_B = A_B, emissivity=emissivity)

    return lo, la, temp_grid  # Custom temperature map [lo (deg), la (deg), T (K)]


# Generating a thermal phase curve from the spiderman custom forward model (inputting a custom temperature map)
def thermal_PhaseCurve(period, incl, ecc, w, r1, r2, R, T2_eff, R_star, lambda_range, t=None, t0=0., u=[0.1, 0.3], n_lo=200, n_la=100, insolation_model="kopal", stellar_model="blackbody", A_B = 0.1, emissivity=1., N_LAYERS=30):
    #  r1, r2, R: planetary radius, stellar radius, semi major axis in consistent units
    # R_star: stellar radius in solar radii

    lo, la, temp_grid = custom_temperature_grid(r1, r2, R, T2_eff, u=u, n_lo=n_lo, n_la=n_la, insolation_model=insolation_model, A_B = A_B, emissivity=emissivity)

    # pass indexing='ij' meshgrids in deg and the transposed temp_grid (wrong orientation fails silently!)
    # Floor the map: 0 K nightside + sharp terminator makes the bicubic spline overshoot to negative temperatures ("CANNOT DO NEGATIVE TEMPERATURES"); ~100 K adds negligible flux at 1-10 um
    T_floor = 100.  # temperature floor for nightside [K]
    LO_deg, LA_deg = np.meshgrid(lo, la, indexing='ij')  # longitude/latitude meshgrids [deg]
    grid = sp.format_grid(LO_deg, LA_deg, np.maximum(temp_grid, T_floor).T)

    spider_params = sp.ModelParams(brightness_model="direct_T", stellar_model=stellar_model)  # stellar model can also be custom or PHOENIX
    spider_params.n_layers = N_LAYERS  # Number of layers in the brightness map

    spider_params.t0 = t0  # Central time of PRIMARY transit [days]
    spider_params.per = period  # Period [days]
    spider_params.a_abs = R * R_star/r2 * 6.957e8/1.495978707e11  # The absolute value of the semi-major axis [AU] (for light travel time)
    spider_params.inc = incl  # Inclination [degrees]
    spider_params.ecc = ecc  # Eccentricity
    spider_params.w = w  # Argument of periastron [degrees]
    spider_params.rp = r1/r2  # Planet to star radius ratio
    spider_params.a = R/r2  # Semi-major axis scaled by stellar radius
    spider_params.p_u1 = spider_params.p_u2 = 0.  # PLANETARY limb darkening parameters (default=0)

    # Parameters specific to the direct_T model
    spider_params.T_s = T2_eff  # Temperature of the star [K]
    spider_params.grid = grid  # Custom temperature map [lo (deg), la (deg), T (K)]

    spider_params.l1, spider_params.l2 = lambda_range  # The starting & ending wavelength in meters

    # Get occultation depth. In the standard spiderman model spider_params.eclipse_depth() segfaults with direct_T, hence the model has to run explicitly without the eclipse to get the depth.
    t_occultation = kepler_utils.occultation_time(period, ecc, w, t0=t0)
    spider_params.eclipse = False
    f_unblocked_occ = spider_params.lightcurve([t_occultation])[0] # full dayside flux
    spider_params.eclipse = True
    f_blocked_occ = spider_params.lightcurve([t_occultation])[0] # pure stellar flux
    fp = f_unblocked_occ - f_blocked_occ

    # PhaseCurve
    if t is None: t = np.linspace(0., period, 1000)

    light_curve = spider_params.lightcurve(t)  # phase modulation + occultation, no transit
    f_transit = batman_utils.flux_transit(period, r1/r2, R/r2, incl, ecc, w, t, t0=t0, u=u)
    phase_curve = f_transit + (light_curve - 1.)

    return t, phase_curve, fp, spider_params


# Generating a thermal light curve (only the planetary phase modulation without occultation and transit) from the spiderman custom forward model (inputting a custom temperature map)
def thermal_LightCurve(period, incl, ecc, w, r1, r2, R, T2_eff, R_star, lambda_range, t=None, t0=0., u=[0.1, 0.3], n_lo=200, n_la=100, insolation_model="kopal", stellar_model="blackbody", A_B = 0.1, emissivity=1., exclude_occultation=True, N_LAYERS=30):
    #  r1, r2, R: planetary radius, stellar radius, semi major axis in consistent units
    # R_star: stellar radius in solar radii

    lo, la, temp_grid = custom_temperature_grid(r1, r2, R, T2_eff, u=u, n_lo=n_lo, n_la=n_la, insolation_model=insolation_model, A_B = A_B, emissivity=emissivity)

    # pass indexing='ij' meshgrids in deg and the transposed temp_grid (wrong orientation fails silently!)
    # Floor the map: 0 K nightside + sharp terminator makes the bicubic spline overshoot to negative temperatures ("CANNOT DO NEGATIVE TEMPERATURES"); ~100 K adds negligible flux at 1-10 um
    T_floor = 100.  # temperature floor for nightside [K]
    LO_deg, LA_deg = np.meshgrid(lo, la, indexing='ij')  # longitude/latitude meshgrids [deg]
    grid = sp.format_grid(LO_deg, LA_deg, np.maximum(temp_grid, T_floor).T)

    spider_params = sp.ModelParams(brightness_model="direct_T", stellar_model=stellar_model)  # stellar model can also be custom or PHOENIX
    spider_params.n_layers = N_LAYERS  # Number of layers in the brightness map

    spider_params.t0 = t0  # Central time of PRIMARY transit [days]
    spider_params.per = period  # Period [days]
    spider_params.a_abs = R * R_star/r2 * 6.957e8/1.495978707e11  # The absolute value of the semi-major axis [AU] (for light travel time)
    spider_params.inc = incl  # Inclination [degrees]
    spider_params.ecc = ecc  # Eccentricity
    spider_params.w = w  # Argument of periastron [degrees]
    spider_params.rp = r1/r2  # Planet to star radius ratio
    spider_params.a = R/r2  # Semi-major axis scaled by stellar radius
    spider_params.p_u1 = spider_params.p_u2 = 0.  # PLANETARY limb darkening parameters (default=0)

    # Parameters specific to the direct_T model
    spider_params.T_s = T2_eff  # Temperature of the star [K]
    spider_params.grid = grid  # Custom temperature map [lo (deg), la (deg), T (K)]

    spider_params.l1, spider_params.l2 = lambda_range  # The starting & ending wavelength in meters

    if exclude_occultation: spider_params.eclipse = False # Only phase modulation, no occultation
    else: spider_params.eclipse = True

    if t is None: t = np.linspace(0., period, 1000)

    light_curve = spider_params.lightcurve(t)-1  # phase modulation, no transit & occultation
    return t, light_curve, spider_params