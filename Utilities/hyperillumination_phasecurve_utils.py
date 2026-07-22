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
#####   Full PhaseCurve (reflected + thermal)
#####
###########################################################################################################################

def full_PhaseCurve(planet, lambda_range, t=None, t0=0., u=[0.1, 0.3], A=0.1, n_eta=150, n_phi=80, n_tab=1000, n_lo=200, n_la=100, insolation_model="kopal", stellar_model="blackbody", filter_path=False, A_B = 0.1, emissivity=1., N_LAYERS=60):
    # Reflected Part including transit
    t_comb, refl, fp_refl = reflected_PhaseCurve(planet, t=t, t0=t0, u=u, include_transit=True, A=A, n_eta=n_eta, n_phi=n_phi, n_tab=n_tab, insolation_model=insolation_model)
    # Thermal part, without stellar flux (no transit)
    _,  therm, fp_therm, _ = thermal_PhaseCurve(planet, lambda_range, t=t_comb, t0=t0, u=u, n_lo=n_lo, n_la=n_la, insolation_model=insolation_model, stellar_model=stellar_model, filter_path=filter_path, A_B = A_B, emissivity=emissivity, include_transit=False, N_LAYERS=N_LAYERS)

    total = refl + therm
    fp = fp_refl + fp_therm

    return t_comb, total, fp


###########################################################################################################################
#####
#####   Reflected Lightcurve
#####
###########################################################################################################################


# Reflected light curve signal accounting for hyperillumination
# This assumes Lambertian reflectance
# Inspired by Kopal eq. 4
def reflected_lightcurve(epsilon, planet, u=0.0, unit="normalized", lambda_band_ratio=1., A=0.1, n_eta=150, n_phi=80, n_tab=1000, runtime=False, insolation_model="kopal"):
    """
    epsilon: angle substellar spot to line of sight [rad]
    unit: Whether to return normalized light curve signal (Fp/Fs) or radiant intensity ([W/sr])
    lambda_band_ratio: Only relevant for unit=='radiant intensity'. If measurements are taken in a wavelength band (not bolometric), this is the ratio Fs_band/Fs. Needed to correctly scale the absolute radial intensity in that band.
    A: Bolometric or band/throughput-integrated albedo. For bolometric measurements, this function is also suitable to calculate reflected + THERMAL light curve signal - for this set A=1.
    n_eta, n_phi: Gauss-Legendre order for the inner (eta) / outer (phi) integral
    n_tab: resolution of the gamma1-grid used to tabulate F_fi/F_pen
    """
    #TODO: When passing a filter T(λ) one can calculate lambda_band_ratio = ∫T(λ)B_λ(T_eff)dλ / σT_eff⁴
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    t0 = time.perf_counter()
    epsilon = np.asarray(epsilon, dtype=float)
    eps_flat = epsilon.ravel()

    gp1 = planet.terminator_penumbra()  # arccos((r1+r2)/R)
    gp2 = planet.terminator_nightside()  # arccos((r1-r2)/R)

    def F(gp):
        if insolation_model == "kopal":
            # tabulate F_fi, F_pen on a gamma1-grid once and interpolate at quadrature nodes
            gp_fi_grid = np.linspace(0., gp1, n_tab)
            Ffi_grid = insolation_fullyilluminated_kopal(gp_fi_grid, planet, u=u)

            gp_pen_grid = np.linspace(gp1, gp2, n_tab)
            Fpen_grid = insolation_penumbra_kopal(gp_pen_grid, planet, u=u)

            return np.where(
                gp <= gp1, np.interp(gp, gp_fi_grid, Ffi_grid),
                np.where(gp <= gp2, np.interp(gp, gp_pen_grid, Fpen_grid), 0.))

        elif insolation_model == "naive":  # naive (point-like) insolation model:
            return naive_insolation(gp, planet)

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
def lambertian_reflectance(epsilon, planet, unit="normalized", lambda_band_ratio=1., A=0.1):
    """
    epsilon: Angle substellar spot to line of sight
    unit: Whether to return normalized light curve signal (Fp/Fs) or radiant intensity ([W/sr])
    lambda_band_ratio: Only relevant for unit=='radiant intensity'. If measurements are taken in a wavelength band (not bolometric), this is the ratio Fs_band/Fs.
    A: Bolometric or band integrated albedo. For bolometric measurements, this function is also suitable to calculate reflected + THERMAL light curve signal - for this set A=1.
    """
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

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
def orbital_reflected_lightcurve(t, planet, t0=0., n_eps=300, A=0.1, n_eta=150, n_phi=80, n_tab=1000, insolation_model="kopal"):  #TODO add , u=u but only when hi model quadratic
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    f = kepler_utils.true_anomaly(t, period, ecc, w, t0=t0)
    epsilon = np.arccos(-np.sin(np.radians(incl)) * np.sin(np.radians(w) + f))

    # THE CURRENT CODE NEGLECTS CHANGES IN R THROUGHOUT THE ORBIT! All functions are designed for scalar R, which is why the formula below currently fails. Needs adjustment for ecc>0.
    # The current code fixes the planet-star distance to the semi-major axis R
    # R_real = R * (1. - ecc**2) / (1. + ecc * np.cos(f))

    # evaluating the expensive integral on a coarse monotonic epsilon grid, then interpolate onto t
    eps_grid = np.linspace(epsilon.min(), epsilon.max(), n_eps)
    ref_grid = reflected_lightcurve(eps_grid, planet, A=A, n_eta=n_eta, n_phi=n_phi, n_tab=n_tab, insolation_model=insolation_model)  #TODO add , u=u but only when hi model quadratic
    return np.interp(epsilon, eps_grid, ref_grid)


# Returns (t, flux): timestamps and the full reflected phase curve combining transit, secondary eclipse / occultation, and sinusoidal phase modulation.
# Reflected phasecurve valid for any wavelength range/filter and stellar spectrum as wavelength dependence cancels out in Fp/Fs (Fp linearly dependent on Fs)
# Band-integrated albedo A however depends on both detector throughput (wavelength range) and stellar spectrum.
# To assume independence on the stellar spectrum we have to neglect stellar limb reddening - this would have an effect on the penumbral reflection
# Lambertian reflection is assumed
def reflected_PhaseCurve(planet, t=None, t0=0., u=[0.1, 0.3], include_transit=True, A=0.1, n_eta=150, n_phi=80, n_tab=1000, insolation_model="kopal"):
    #  r1, r2, R: planetary radius, stellar radius, semi major axis in consistent units
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    if t is None: t = np.linspace(0., period, 1000)

    t_corr = batman_utils.light_travel_time_correction(t, planet, t0=t0)

    orbital_ref_lightcurve = orbital_reflected_lightcurve(t, planet, t0=t0, A=A, n_eta=n_eta, n_phi=n_phi, n_tab=n_tab, insolation_model=insolation_model)  #TODO add , u=u but only when hi model quadratic
    eps_occultation = np.arccos(np.sin(np.radians(incl)))  # epsilon at full phase
    fp = reflected_lightcurve(eps_occultation, planet, A=A, n_eta=n_eta, n_phi=n_phi, n_tab=n_tab, insolation_model=insolation_model) #TODO add , u=u but only when hi model quadratic
    if fp == 0.: return t_corr, batman_utils.flux_transit(planet, t, t0=t0, u=u), fp*1e6  # no reflection (e.g. when albedo=0) -> no phase curve, only transit

    # Full Phase Curve Model. This model ignores the exact eclipse geometry during ingress and egress, instead it assumes the integrated flux to be uniformly distributed over the disk
    phase_mod = orbital_ref_lightcurve / fp
    occultation = batman_utils.flux_occultation(planet, fp, t, t0=t0, u=u)

    if include_transit:
        transit = batman_utils.flux_transit(planet, t, t0=t0, u=u)
        phase_curve = transit + (occultation - 1) * phase_mod
    else:
        phase_curve = (occultation - 1) * phase_mod

    return t_corr, phase_curve, fp*1e6


###########################################################################################################################
#####
#####   Thermal Phase Curve (Spiderman)
#####
###########################################################################################################################


# Generated custom temperature grid from a given insolation model (insolation_model = hyperillumination, naive) as input for spiderman custom forward model
def custom_temperature_grid(planet, u=0, n_lo=200, n_la=100, insolation_model="kopal", A_B = 0.1, emissivity=1.):
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    lo = np.linspace(-180, 180, n_lo)  # longitude [deg], 0 = substellar meridian
    la = np.linspace(-90, 90, n_la)  # latitude [deg], 0 = equator
    LO, LA = np.meshgrid(np.radians(lo), np.radians(la))  # shape (n_la, n_lo), radians

    gamma1_grid = np.arccos(np.cos(LA) * np.cos(LO))  # angular distance from substellar point
    if insolation_model == "kopal":
        insol_grid = insolation_kopal(gamma1_grid, planet)  #TODO add , u=u but only when hi model quadratic
    elif insolation_model == "naive":
        insol_grid = naive_insolation(gamma1_grid, planet)
    else:
        raise ValueError("insolation_model must be either 'kopal' or 'naive'")
    temp_grid = BBtemp_from_insolation(insol_grid, A_B = A_B, emissivity=emissivity)

    return lo, la, temp_grid  # Custom temperature map [lo (deg), la (deg), T (K)]


# Generating a thermal phase curve from the spiderman custom forward model (inputting a custom temperature map)
def thermal_PhaseCurve(planet, lambda_range, t=None, t0=0., u=[0.1, 0.3], n_lo=200, n_la=100, insolation_model="kopal", stellar_model="blackbody", filter_path=False, A_B = 0.1, emissivity=1., N_LAYERS=60, include_transit=True, include_occultation=True):
    #  r1, r2, R: planetary radius, stellar radius, semi major axis in consistent units
    # For lambda_range, either pass the effective ranges without a filter_path (spiderman assumes tophat) or provide lambda_range and filter_path.
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    lo, la, temp_grid = custom_temperature_grid(planet, u=u, n_lo=n_lo, n_la=n_la, insolation_model=insolation_model, A_B = A_B, emissivity=emissivity)

    # pass indexing='ij' meshgrids in deg and the transposed temp_grid (wrong orientation fails silently!)
    # Floor the map: 0 K nightside + sharp terminator makes the bicubic spline overshoot to negative temperatures ("CANNOT DO NEGATIVE TEMPERATURES"); ~100 K adds negligible flux at 1-10 um
    T_floor = 100.  # temperature floor for nightside [K]
    LO_deg, LA_deg = np.meshgrid(lo, la, indexing='ij')  # longitude/latitude meshgrids [deg]
    grid = sp.format_grid(LO_deg, LA_deg, np.maximum(temp_grid, T_floor).T)

    spider_params = sp.ModelParams(brightness_model="direct_T", stellar_model=stellar_model)  # stellar model can also be custom or PHOENIX
    spider_params.n_layers = N_LAYERS  # Number of layers in the brightness map

    spider_params.t0 = t0  # Central time of PRIMARY transit [days]
    spider_params.per = period  # Period [days]
    spider_params.a_abs = a_AU  # The absolute value of the semi-major axis [AU] (for light travel time)
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

    # Add filter throughput
    spider_params.filter = filter_path
    stellar_grid = sp.stellar_grid.gen_grid(spider_params.l1, spider_params.l2, logg=4.5, response=spider_params.filter, stellar_model=stellar_model)

    # Get occultation depth. In the standard spiderman model spider_params.eclipse_depth() segfaults with direct_T, hence the model has to run explicitly without the eclipse to get the depth.
    t_occultation = kepler_utils.occultation_time(period, ecc, w, t0=t0)
    spider_params.eclipse = False
    f_unblocked_occ = spider_params.lightcurve([t_occultation], stellar_grid=stellar_grid)[0] # full dayside flux
    spider_params.eclipse = True
    f_blocked_occ = spider_params.lightcurve([t_occultation], stellar_grid=stellar_grid)[0] # pure stellar flux
    fp = (f_unblocked_occ - f_blocked_occ)*1e6  # Occultation depth Fp/Fs(alpha=0) in ppm

    # Include Occultation?
    spider_params.eclipse = include_occultation

    # PhaseCurve / Lightcurve
    if t is None: t = np.linspace(0., period, 1000)
    light_curve = spider_params.lightcurve(t, stellar_grid=stellar_grid)  # phase modulation + occultation, no transit

    if include_transit: # Full PhaseCurve
        f_transit = batman_utils.flux_transit(planet, t, t0=t0, u=u)
        phase_curve = f_transit + (light_curve - 1.)
    else: # Only Lightcurve, no stellar flux
        phase_curve = light_curve - 1.

    return t, phase_curve, fp, spider_params


# Calculates the occultation depth of a spiderman model (spider_params.eclipse_depth() does not work with direct_T)
def occultation_depth(spider_params):
    set_ecl = spider_params.eclipse
    stellar_grid = sp.stellar_grid.gen_grid(spider_params.l1, spider_params.l2, logg=4.5, response=spider_params.filter, stellar_model=spider_params.stellar_model)
    t_occ = kepler_utils.occultation_time(spider_params.per, spider_params.ecc, spider_params.w, t0=spider_params.t0)
    spider_params.eclipse = False; f_open = spider_params.lightcurve([t_occ], stellar_grid=stellar_grid)[0]
    spider_params.eclipse = True;  f_closed = spider_params.lightcurve([t_occ], stellar_grid=stellar_grid)[0]
    spider_params.eclipse = set_ecl
    return (f_open - f_closed)*1e6 # Fp/Fs(alpha=0) in ppm