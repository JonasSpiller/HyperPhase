# translating the insolation model with hyperillumination to a light curve signal
import time
import numpy as np
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):   # mute "looking for spidermanrc..." in every process
    import fspiderman as sp
from hyperillumination_utils import *
import astro_utils
import batman_utils
import contextlib, io
from astropy import constants as const
from collections.abc import Callable
from dataclasses import field   # dataclass/replace arrive via hyperillumination_utils; field does not

sigma = const.sigma_sb.value


###########################################################################################################################
#####
#####   Full PhaseCurve (reflected + thermal)
#####
###########################################################################################################################

def full_PhaseCurve(planet, lambda_range, t=None, t0=0., u=[0.1, 0.3], u_B=0., A=0.1, n_eta=150, n_phi=80, n_tab=1000, n_lo=200, n_la=100, insolation_model="hyperillum", stellar_model="blackbody", filter_path=False, A_B = 0.1, emissivity=1., N_LAYERS=60, include_transit=True, include_occultation=True, spman="fast", spman_kw=None):
    # Reflected Part including transit
    t_comb, refl, fp_refl = reflected_PhaseCurve(planet, t=t, t0=t0, u=u, u_B=u_B, include_transit=include_transit, include_occultation=include_occultation, A=A, n_eta=n_eta, n_phi=n_phi, n_tab=n_tab, insolation_model=insolation_model)
    # Thermal part, without stellar flux (no transit)
    _,  therm, fp_therm, _ = thermal_PhaseCurve(planet, lambda_range, t=t_comb, t0=t0, u=u, u_B=u_B, n_lo=n_lo, n_la=n_la, insolation_model=insolation_model, stellar_model=stellar_model, filter_path=filter_path, A_B = A_B, emissivity=emissivity, include_transit=False, include_occultation=include_occultation, N_LAYERS=N_LAYERS, spman=spman, spman_kw=spman_kw)

    total = refl + therm
    fp = fp_refl + fp_therm

    return t_comb, total, fp


###########################################################################################################################
#####
#####   Reflected Lightcurve
#####
###########################################################################################################################


# Reflected light curve signal accounting for hyperillumination
# This assumes Lambertian (isotropic, grey) reflection, circular planet
# Inspired by Kopal eq. 4
def reflected_lightcurve(epsilon, planet, u_B=0., unit="normalized", lambda_band_ratio=1., A=0.1, n_eta=150, n_phi=80, n_tab=1000, runtime=False, insolation_model="hyperillum"):
    """
    epsilon: angle substellar spot to line of sight [rad]
    unit: Whether to return normalized light curve signal (Fp/Fs) or radiant intensity ([W/sr])
    lambda_band_ratio: Only relevant for unit=='radiant intensity'. If measurements are taken in a wavelength band (not bolometric), this is the ratio Fs_band/Fs. Needed to correctly scale the absolute radial intensity in that band.
    A: Spherical bolometric or band/throughput-integrated albedo (NOT geometric albedo). For bolometric measurements, this function is also suitable to calculate reflected + THERMAL light curve signal - for this set A=1.
    n_eta, n_phi: Gauss-Legendre order for the inner (eta) / outer (phi) integral
    n_tab: resolution of the gamma1-grid used to tabulate F_fi/F_pen
    """
    #TODO: When passing a filter T(λ) one can calculate lambda_band_ratio = ∫T(λ)B_λ(T_eff)dλ / σT_eff⁴
    #TODO: For reflected light insolation_penumbra_hyperillum() should actually take u (MIR band), not u_B. For simplicity (since quadratic limb darkening is not implemented in the hyperillum model) we pass the bolometric u_B and assume the impact to be negligible for now.
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    t0 = time.perf_counter()
    epsilon = np.asarray(epsilon, dtype=float)
    eps_flat = epsilon.ravel()

    gp1 = planet.terminator_penumbra()  # arccos((r1+r2)/R)
    gp2 = planet.terminator_nightside()  # arccos((r1-r2)/R)

    def F(gp):
        if insolation_model == "hyperillum":
            # tabulate F_fi, F_pen on a gamma1-grid once and interpolate at quadrature nodes
            gp_fi_grid = np.linspace(0., gp1, n_tab)
            Ffi_grid = insolation_fullyilluminated_hyperillum(gp_fi_grid, planet)

            gp_pen_grid = np.linspace(gp1, gp2, n_tab)
            Fpen_grid = insolation_penumbra_hyperillum(gp_pen_grid, planet, u_B=u_B)

            return np.where(
                gp <= gp1, np.interp(gp, gp_fi_grid, Ffi_grid),
                np.where(gp <= gp2, np.interp(gp, gp_pen_grid, Fpen_grid), 0.))

        elif insolation_model == "naive":  # naive (point-like) insolation model:
            return naive_insolation(gp, planet)

        else:
            raise ValueError("insolation_model must be either 'hyperillum' or 'naive'")

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
def orbital_reflected_lightcurve(t, planet, t0=0., n_eps=300, A=0.1, u_B=0., n_eta=150, n_phi=80, n_tab=1000, insolation_model="hyperillum"):
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    f = astro_utils.true_anomaly(t, period, ecc, w, t0=t0)
    epsilon = np.arccos(-np.sin(np.radians(incl)) * np.sin(np.radians(w) + f))

    # THE CURRENT CODE NEGLECTS CHANGES IN R THROUGHOUT THE ORBIT! All functions are designed for scalar R, which is why the formula below currently fails. Needs adjustment for ecc>0.
    # The current code fixes the planet-star distance to the semi-major axis R
    # R_real = R * (1. - ecc**2) / (1. + ecc * np.cos(f))

    # evaluating the expensive integral on a coarse monotonic epsilon grid, then interpolate onto t
    eps_grid = np.linspace(epsilon.min(), epsilon.max(), n_eps)
    ref_grid = reflected_lightcurve(eps_grid, planet, u_B=u_B, A=A, n_eta=n_eta, n_phi=n_phi, n_tab=n_tab, insolation_model=insolation_model)
    return np.interp(epsilon, eps_grid, ref_grid)


# Returns (t, flux): timestamps and the full reflected phase curve combining transit, secondary eclipse / occultation, and sinusoidal phase modulation.
# Reflected phasecurve valid for any wavelength range/filter and stellar spectrum as wavelength dependence cancels out in Fp/Fs (Fp linearly dependent on Fs)
# Band-integrated albedo A however depends on both detector throughput (wavelength range) and stellar spectrum.
# To assume independence on the stellar spectrum we have to neglect stellar limb darkening + reddening - this would have an effect on the penumbral reflection (this is justified since limb darkening weak in IR)
# This assumes Lambertian (isotropic, grey) reflectance, circular planet
def reflected_PhaseCurve(planet, t=None, t0=0., u=[0.1, 0.3], u_B=0., include_transit=True, include_occultation=True, A=0.1, n_eta=150, n_phi=80, n_tab=1000, insolation_model="hyperillum"):
    #  r1, r2, R: planetary radius, stellar radius, semi major axis in consistent units
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    if t is None: t = np.linspace(0., period, 1000, endpoint=False)
    t_corr = batman_utils.light_travel_time_correction(t, planet, t0=t0)

    orbital_ref_lightcurve = orbital_reflected_lightcurve(t, planet, t0=t0, A=A, u_B=u_B, n_eta=n_eta, n_phi=n_phi, n_tab=n_tab, insolation_model=insolation_model)

    eps_occultation = np.arccos(np.sin(np.radians(incl)))  # epsilon at full phase
    fp = reflected_lightcurve(eps_occultation, planet, A=A, u_B=u_B, n_eta=n_eta, n_phi=n_phi, n_tab=n_tab, insolation_model=insolation_model)
    if fp == 0.: return t_corr, include_transit*batman_utils.flux_transit(planet, t, t0=t0, u=u), fp*1e6  # no reflection (e.g. when albedo=0) -> no phase curve, only transit

    # Full Phase Curve Model. This model ignores the exact eclipse geometry during ingress and egress, instead it assumes the integrated flux to be uniformly distributed over the disk
    if include_occultation:
        phase_mod = orbital_ref_lightcurve / fp
        occultation = batman_utils.flux_occultation(planet, fp, t, t0=t0)

        if include_transit:
            transit = batman_utils.flux_transit(planet, t, t0=t0, u=u)
            phase_curve = transit + (occultation - 1) * phase_mod
        else:
            phase_curve = (occultation - 1) * phase_mod
    else:
        if include_transit:
            transit = batman_utils.flux_transit(planet, t, t0=t0, u=u)
            phase_curve = transit + orbital_ref_lightcurve
        else:
            phase_curve = orbital_ref_lightcurve

    return t_corr, phase_curve, fp*1e6


###########################################################################################################################
#####
#####   Thermal Phase Curve (Spiderman)
#####
###########################################################################################################################

# Calculates the occultation depth of a spiderman model (spider_params.eclipse_depth() does not work with direct_T)
def occultation_depth(spider_params, stellar_grid=None):
    set_ecl = spider_params.eclipse
    if stellar_grid is None:
        with contextlib.redirect_stdout(io.StringIO()): # mute notification: "no stellar models provided, using blackbody approximation"
            stellar_grid = sp.stellar_grid.gen_grid(spider_params.l1, spider_params.l2, logg=4.5, response=spider_params.filter, stellar_model=spider_params.stellar_model)
    t_occ = astro_utils.occultation_time(spider_params.per, spider_params.ecc, spider_params.w, t0=spider_params.t0)
    spider_params.eclipse = False; f_unblocked_occ = spider_params.lightcurve([t_occ], stellar_grid=stellar_grid)[0]
    spider_params.eclipse = True;  f_blocked_occ = spider_params.lightcurve([t_occ], stellar_grid=stellar_grid)[0]
    spider_params.eclipse = set_ecl
    return (f_unblocked_occ - f_blocked_occ)*1e6 # Fp/Fs(alpha=0) in ppm


# The A_B-INDEPENDENT half of custom_temperature_grid: the insolation map on the (lo, la) grid.
# It depends only on geometry, so a fit that varies only A_B computes it ONCE and rescales (ee insolation_cache / model_pcurve(insol=...) in the Fitting section)
def insolation_grid(planet, u_B=0., n_lo=200, n_la=100, insolation_model="hyperillum"):
    # THE CURRENT CODE NEGLECTS CHANGES IN R THROUGHOUT THE ORBIT! All functions are designed for scalar R, which is why the formula below currently fails. Needs adjustment for ecc>0.
    # The current code fixes the planet-star distance to the semi-major axis R
    # R_real = R * (1. - ecc**2) / (1. + ecc * np.cos(f))

    lo = np.linspace(-180, 180, n_lo)  # longitude [deg], 0 = substellar meridian
    la = np.linspace(-90, 90, n_la)  # latitude [deg], 0 = equator
    LO, LA = np.meshgrid(np.radians(lo), np.radians(la))  # shape (n_la, n_lo), radians

    gamma1_grid = np.arccos(np.cos(LA) * np.cos(LO))  # angular distance from substellar point
    return lo, la, insolation(gamma1_grid, planet, insolation_model, u_B=u_B)  # [lo (deg), la (deg), F (W/m^2)]


# Generated custom temperature grid from a given insolation model (insolation_model = hyperillum, naive) as input for spiderman custom forward model
# insol: optional precomputed (lo, la, insol_grid) from insolation_grid / insolation_cache. When given the insolation is NOT recomputed and only the (1-A_B)^(1/4) rescale is done (~0.3 ms instead of ~0.4 s).
def custom_temperature_grid(planet, u_B=0., n_lo=200, n_la=100, insolation_model="hyperillum", A_B = 0.1, emissivity=1., insol=None):
    if insol is None:
        lo, la, insol_grid = insolation_grid(planet, u_B=u_B, n_lo=n_lo, n_la=n_la, insolation_model=insolation_model)
    else:
        lo, la, insol_grid = insol
        if (len(lo), len(la)) != (n_lo, n_la) or insol_grid.shape != (len(la), len(lo)):
            raise ValueError(f"cached insol is on a {(len(lo), len(la))} grid but n_lo, n_la = {(n_lo, n_la)}")

    temp_grid = BBtemp_from_insolation(insol_grid, A_B = A_B, emissivity=emissivity)

    return lo, la, temp_grid  # Custom temperature map [lo (deg), la (deg), T (K)]


# Generating a thermal phase curve from the spiderman custom forward model (inputting a custom temperature map)
def thermal_PhaseCurve(planet, lambda_range, t=None, t0=0., u=[0.1, 0.3], u_B=0., n_lo=200, n_la=100, insolation_model="hyperillum", stellar_model="blackbody", filter_path=False, A_B = 0.1, emissivity=1., N_LAYERS=None, include_transit=True, include_occultation=True, spman="fast", spman_kw=None):
    #  r1, r2, R: planetary radius, stellar radius, semi major axis in consistent units
    # For lambda_range, either pass the effective ranges without a filter_path (spiderman assumes tophat) or provide lambda_range and filter_path.
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    # N_LAYERS = 30-40 usually sufficient for insolation_model="hyperillum", insolation_model="naive" needs N_LAYERS = 60-80
    if N_LAYERS is None:
        if insolation_model=="hyperillum": N_LAYERS = 40
        else: N_LAYERS = 80

    lo, la, temp_grid = custom_temperature_grid(planet, u_B=u_B, n_lo=n_lo, n_la=n_la, insolation_model=insolation_model, A_B = A_B, emissivity=emissivity)

    # pass indexing='ij' meshgrids in deg and the transposed temp_grid (wrong orientation fails silently!)
    # Floor the map: 0 K nightside + sharp terminator makes the bicubic spline overshoot to negative temperatures ("CANNOT DO NEGATIVE TEMPERATURES"); ~100 K adds negligible flux at 1-20 um
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
    with contextlib.redirect_stdout(io.StringIO()): # mute notification: "no stellar models provided, using blackbody approximation"
        stellar_grid = sp.stellar_grid.gen_grid(spider_params.l1, spider_params.l2, logg=4.5, response=spider_params.filter, stellar_model=stellar_model)

    # Get occultation depth. In the standard spiderman model spider_params.eclipse_depth() segfaults with direct_T, hence the model has to run explicitly without the eclipse to get the depth.
    fp = occultation_depth(spider_params, stellar_grid=stellar_grid)  # Occultation depth Fp/Fs(alpha=0) in ppm

    # Include Occultation?
    spider_params.eclipse = include_occultation

    # PhaseCurve / Lightcurve
    if t is None: t = np.linspace(0., period, 1000, endpoint=False)
    if spman=="fast": light_curve = fast_lightcurve(spider_params, stellar_grid, t, **(spman_kw or {}))  # phase modulation + occultation, no transit - faster version with interpolation. spman_kw overrides fast_lightcurve's n_phase / pad / split_totality
    else: light_curve = spider_params.lightcurve(t, stellar_grid=stellar_grid)  # phase modulation + occultation, no transit

    if include_transit: # Full PhaseCurve
        f_transit = batman_utils.flux_transit(planet, t, t0=t0, u=u)
        phase_curve = f_transit + (light_curve - 1.)
    else: # Only Lightcurve, no stellar flux
        phase_curve = light_curve - 1.

    return t, phase_curve, fp, spider_params



# Equivalent to thermal_PhaseCurve(..., insolation_model="naive") but faster.
def Louden_PhaseCurve(planet, lambda_range, t=None, t0=0., u=[0.1, 0.3], stellar_model="blackbody", filter_path=False, A_B = 0.1, N_LAYERS=60, include_transit=True, include_occultation=True, spman="fast", spman_kw=None):
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    insol = sigma * T2_eff**4 / (R/r2)**2

    sp_louden = sp.ModelParams(brightness_model="Louden", stellar_model=stellar_model)
    sp_louden.thermal = True
    sp_louden.n_layers = N_LAYERS

    sp_louden.t0, sp_louden.per = t0, period
    sp_louden.a_abs = a_AU
    sp_louden.inc, sp_louden.ecc, sp_louden.w = incl, ecc, w
    sp_louden.rp, sp_louden.a = r1/r2, R/r2
    sp_louden.p_u1 = sp_louden.p_u2 = 0. # PLANETARY limb darkening parameters (default=0)

    sp_louden.T_s = T2_eff
    sp_louden.insol = insol
    sp_louden.albedo = A_B
    sp_louden.redist = 0.  # instant reradiation
    sp_louden.T_int = 0. # no internal heating
    sp_louden.l1, sp_louden.l2 = lambda_range

    sp_louden.filter = filter_path
    with contextlib.redirect_stdout(io.StringIO()): # mute notification: "no stellar models provided, using blackbody approximation"
        stellar_grid = sp.stellar_grid.gen_grid(sp_louden.l1, sp_louden.l2, logg=4.5, response=sp_louden.filter, stellar_model=stellar_model)

    # Get occultation depth.
    fp = occultation_depth(sp_louden, stellar_grid=stellar_grid)  # Occultation depth Fp/Fs(alpha=0) in ppm

    # Include Occultation?
    sp_louden.eclipse = include_occultation

    # PhaseCurve / Lightcurve
    if t is None: t = np.linspace(0., period, 1000, endpoint=False)
    if spman=="fast": light_curve = fast_lightcurve(sp_louden, stellar_grid, t, **(spman_kw or {}))  # phase modulation + occultation, no transit. spman_kw overrides fast_lightcurve's n_phase / pad / split_totality
    else: light_curve = sp_louden.lightcurve(t, stellar_grid=stellar_grid)  # phase modulation + occultation, no transit

    if include_transit:  # Full PhaseCurve
        f_transit = batman_utils.flux_transit(planet, t, t0=t0, u=u)
        phase_curve = f_transit + (light_curve - 1.)
    else:  # Only Lightcurve, no stellar flux
        phase_curve = light_curve - 1.

    return t, phase_curve, fp, sp_louden



###########################################################################################################################
#####
#####   Fast Lightcurve - SPIDERMAN Patch
#####
###########################################################################################################################

def _trig_interp(y_nodes, per, t, t0=0.):
    # Exact trigonometric interpolation of a band-limited periodic signal sampled uniformly on [t0, t0 + per) (no endpoint).
    n = len(y_nodes)
    c = np.fft.rfft(y_nodes) / n
    ph = 2 * np.pi * (t - t0) / per
    out = np.full_like(np.asarray(t, dtype=float), c[0].real)
    for k in range(1, n // 2 + 1):
        amp = 2 * c[k] if k < n / 2 else c[k]
        out = out + amp.real * np.cos(k * ph) - amp.imag * np.sin(k * ph)
    return out


def occultation_durations(spider_params): # assuming circular orbit
    # (T14, T23) of the occultation: total duration (1st-4th contact) and totality (2nd-3rd contact), Winn 2010 eq. 14. T23 = 0 for a grazing geometry, where the planet is never fully hidden.
    inc = np.radians(spider_params.inc)
    b = spider_params.a * np.cos(inc)
    dur = lambda x: spider_params.per / np.pi * np.arcsin(min(np.sqrt(max(x ** 2 - b ** 2, 0.)) / (spider_params.a * np.sin(inc)), 1.))  # the sin(inc) matches astro_utils.transit_duration_t14; min(..., 1.) gives the half-orbit limit instead of NaN for a geometry so close-in the planet never clears the stellar disk
    return dur(1 + spider_params.rp), dur(1 - spider_params.rp)

def occultation_window(spider_params, pad=0.03):
    # (t_ingress, t_egress) of the occultation, padded by `pad` durations on each side.
    t_occultation = astro_utils.occultation_time(spider_params.per, spider_params.ecc, spider_params.w, t0=spider_params.t0)
    T14, _ = occultation_durations(spider_params)
    return t_occultation - (0.5 + pad) * T14, t_occultation + (0.5 + pad) * T14

def _dt_to_occultation(spider_params, t):
    # Signed offset of each `t` to the NEAREST occultation centre, in ANY orbit spanned by `t`.
    t_occultation = astro_utils.occultation_time(spider_params.per, spider_params.ecc, spider_params.w, t0=spider_params.t0)
    per = spider_params.per
    return np.mod(np.asarray(t, dtype=float) - t_occultation + 0.5 * per, per) - 0.5 * per

def in_occultation(spider_params, t, pad=0.03):
    # Boolean mask: which of `t` fall inside an occultation, in ANY orbit spanned by `t`.
    T14, _ = occultation_durations(spider_params)
    return np.abs(_dt_to_occultation(spider_params, t)) <= (0.5 + pad) * T14


def fast_lightcurve(spider_params, stellar_grid, t, n_phase=40, pad=0.03, split_totality=True):
    # SPIDERMAN lightcurve evaluated cheaply by splitting the orbit in smooth, out-of-occultation part and occulation part.
    # only first few fourier coefficients contribute to lightcurve -> only few interpolation points n_phase needed (Nyquist)
    # inside totality the planet is fully hidden, so the flux is exactly the stellar flux (1) -> only the two ingress/egress ramps still need SPIDERMAN. split_totality=False restores the whole-window behaviour.

    t = np.asarray(t, dtype=float)
    orig_eclipse = spider_params.eclipse

    try:
        spider_params.eclipse = False # smooth phase modulation, no occultation
        t_nodes = spider_params.t0 + spider_params.per * np.arange(n_phase) / n_phase
        y_nodes = spider_params.lightcurve(t_nodes, stellar_grid=stellar_grid)
        out = _trig_interp(y_nodes, spider_params.per, t, t0=spider_params.t0) # signal fully captured in ~11 Fourier coefficients

        if orig_eclipse:  # exact inside EVERY occultation spanned by t
            T14, T23 = occultation_durations(spider_params)
            dt = np.abs(_dt_to_occultation(spider_params, t))
            m = dt <= (0.5 + pad) * T14  # padded occultation window
            half_tot = 0.5 * T23 - pad * T14  # totality, pulled back by pad*T14 from 2nd/3rd contact; <= 0 for a grazing geometry -> keep the whole window exact
            if split_totality and half_tot > 0.:
                tot = dt < half_tot
                out[tot] = 1.  # planet fully occulted: no planetary flux, so the lightcurve is exactly the (normalised) stellar flux
                m &= ~tot  # only the two ingress/egress ramps are left
            if m.any():
                spider_params.eclipse = True
                out[m] = spider_params.lightcurve(t[m], stellar_grid=stellar_grid) # occultation calculated exactly, one call for both ramps
    finally:
        spider_params.eclipse = orig_eclipse # restore even if lightcurve() raised, else spider_params is left dirty

    return out



###########################################################################################################################
#####
#####   Fitting
#####
###########################################################################################################################


# Band-integrated albedo A in the observing band as a function of the Bond albedo A_B — i.e. the FittingConfig.A_ callable.
# Classes rather than a `def` in a notebook cell because dynesty.pool.Pool spawns its workers and dynesty checkpoints with plain pickle which cannot handle notebook functions.
@dataclass(frozen=True)
class ConstantBandAlbedo:
    """Fixed band-integrated albedo, independent of A_B."""
    A: float = 0.1
    def __call__(self, A_B): return self.A
    @property
    def label(self): return f"fixedA{self.A:g}"      # short tag for filenames and plot legends, e.g. 'fixedA0.1'


@dataclass(frozen=True)
class ScaledBandAlbedo:
    """A = scale * A_B; scale=1 sets the band-integrated albedo equal to the Bond albedo."""
    scale: float = 1.0
    def __call__(self, A_B): return self.scale * A_B
    @property
    def label(self): return f"scaledA{self.scale:g}"   # short tag for filenames and plot legends, e.g. 'scaledA1'


@dataclass
class FittingConfig:
    """Fixed physical + instrument context for the bias analysis."""
    planet: object          # PlanetParams for the reference system (e.g. K2-141b); single source of
                            # truth for geometry (r1, r2, R, a_AU), orbit (period, incl, ecc, w) and T2_eff
    stellar_grid: object    # sp.stellar_grid.gen_grid(...) output for the chosen filter
    filter_path: str        # SPIDERMAN throughput file
    l1: float               # filter blue edge [m]
    l2: float               # filter red edge [m]
    A_: object     # band integrated albedo as function of bond albedo (only needed when reflected=True)
    sigma_1min: float = 323.  # detection error [ppm per minute of integration]
    n_orbits: int = 3  # number of orbits observed and phase-folded
    n_bins: int = 27  # bins per period; the width is DERIVED so it tiles exactly
    u_B: float = 0.         # hyperillumination-model limb-darkening coefficient (0 = uniform disk)
    u: tuple = (0.1, 0.3)   # quadratic stellar limb darkening [u1, u2] for the transit
    reflected: bool = True  # True -> full (reflected + thermal); False -> thermal only
    transit: bool = True    # include the primary transit (batman)
    T_floor: float = 200.   # min temperature for the direct_T grid [K]; blocks negative-T spline overshoot

    @property
    def sigma_bin(self):
        minutes_binning = self.planet.period * 24 * 60 / self.n_bins
        return self.sigma_1min / np.sqrt(minutes_binning * self.n_orbits)


# The PlanetParams that model_pcurve actually evaluates, at the swept geometry, in pl's own units.
# Factored out so that a cached insolation map (insolation_cache) is guaranteed to be built at exactly the geometry the corresponding model_pcurve call will use — the caches cannot silently go stale.
def swept_planet(cfg, a_rs=None, T_s=None, rp_rs=None):
    pl = cfg.planet
    a_rs0 = pl.R / pl.r2  # default a/Rs (for the a_AU scaling)
    a_rs = a_rs0 if a_rs is None else a_rs
    T_s = pl.T2_eff if T_s is None else T_s
    rp_rs = pl.r1 / pl.r2 if rp_rs is None else rp_rs
    # rebuild the planet at the swept geometry in pl's own units; a_AU scales with a/Rs (stellar size fixed)
    return replace(pl, r1=rp_rs * pl.r2, R=a_rs * pl.r2, T2_eff=T_s, a_AU=pl.a_AU * a_rs / a_rs0)


# Ready-made `insol` argument for model_pcurve, built at exactly the geometry model_pcurve would use.
# ONLY valid while a/r2, r1/r2, T2_eff, cfg.u_B, n_lo and n_la are held fixed — see dynesty_utils.GaussianLogL, which builds this and invalidates it automatically when the corresponding parameter is sampled.
def insolation_cache(cfg, insolation_model, n_lo=300, n_la=150, **geom):
    planet = swept_planet(cfg, **geom)   # **geom rather than named args, so adding a parameter to swept_planet is enough
    return insolation_grid(planet, u_B=cfg.u_B, n_lo=n_lo, n_la=n_la, insolation_model=insolation_model)


# Reflected phase-curve shape at unit albedo (A = 1)
def reflected_unit(cfg, insolation_model, a_rs=None, T_s=None, rp_rs=None, n_t=None, **_):
    pl = cfg.planet
    a_rs = pl.R / pl.r2  if a_rs is None else a_rs   # default a/Rs from PlanetParams
    T_s = pl.T2_eff      if T_s is None else T_s
    rp_rs = pl.r1 / pl.r2  if rp_rs is None else rp_rs      # default Rp/Rs from PlanetParams
    n_t = 50 * cfg.n_bins if n_t is None else n_t

    t_ = np.linspace(0., pl.period, int(n_t), endpoint=False)
    # set the ratios in the planet's own units (reflected is geometric -> a_AU unused, left as-is)
    planet = replace(pl, r1=rp_rs * pl.r2, R=a_rs * pl.r2, T2_eff=T_s)
    _, refl, _ = reflected_PhaseCurve(planet, t=t_, include_transit=False, A=1., u_B=cfg.u_B, insolation_model=insolation_model)
    return refl * 1e6


# The sp.ModelParams that model_pcurve feeds to SPIDERMAN, factored out exactly like swept_planet so the OCCULTATION DEPTH can be taken at the fitted geometry with the existing occultation_depth(p, stellar_grid=cfg.stellar_grid) instead of duplicating this setup.
def pcurve_params(cfg, insolation_model, A_B_, a_rs=None, T_s=None, rp_rs=None, n_layers=None, n_lo=300, n_la=150, insol=None):
    # n_layers = 30-40 usually sufficient for insolation_model="hyperillum", insolation_model="naive" needs n_layers = 60-80
    if n_layers is None:
        if insolation_model == "hyperillum":
            n_layers = 40
        else:
            n_layers = 60

    planet = swept_planet(cfg, a_rs=a_rs, T_s=T_s, rp_rs=rp_rs)

    # insol=... reuses the A_B-independent insolation map, leaving only the exact T(A_B) = T(A_B=0)*(1-A_B)^(1/4) rescale (valid because insol depends on geometry alone).
    lo, la, Tg = custom_temperature_grid(planet, u_B=cfg.u_B, n_lo=n_lo, n_la=n_la, insolation_model=insolation_model, A_B=A_B_, insol=insol)
    LO, LA = np.meshgrid(lo, la, indexing="ij")
    p = sp.ModelParams(brightness_model="direct_T", stellar_model="blackbody")
    p.n_layers = n_layers
    p.t0, p.per = 0., planet.period
    p.a_abs = planet.a_AU
    p.inc, p.ecc, p.w = planet.incl, planet.ecc, planet.w
    p.rp, p.a = planet.r1 / planet.r2, planet.R / planet.r2
    p.p_u1 = p.p_u2 = 0.
    p.T_s, p.l1, p.l2 = planet.T2_eff, cfg.l1, cfg.l2
    p.grid = sp.format_grid(LO, LA, np.maximum(Tg, cfg.T_floor).T)
    p.filter, p.eclipse = cfg.filter_path, True
    return p


# Fast, stripped-down phase curve [ppm] for the fitting loop
# refl_unit, insol: precomputed geometry-only pieces (reflected_unit / insolation_cache). Both are optional and purely a speed-up: passing them is bit-identical to letting them be rebuilt here.
def model_pcurve(cfg, insolation_model, A_B_, a_rs=None, T_s=None, rp_rs=None, n_t=None, n_layers=None, n_lo=300, n_la=150, refl_unit=None, insol=None, spman_kw=None):
    pl = cfg.planet
    n_t = 50 * cfg.n_bins if n_t is None else n_t

    t_ = np.linspace(0., pl.period, int(n_t), endpoint=False)
    planet = swept_planet(cfg, a_rs=a_rs, T_s=T_s, rp_rs=rp_rs)

    # Thermal Part
    p = pcurve_params(cfg, insolation_model, A_B_, a_rs=a_rs, T_s=T_s, rp_rs=rp_rs, n_layers=n_layers, n_lo=n_lo, n_la=n_la, insol=insol)
    thermal = (fast_lightcurve(p, cfg.stellar_grid, t_, **{"n_phase": 24, **(spman_kw or {})}) - 1) * 1e6  # spman_kw overrides n_phase / pad / split_totality; it reaches here through **model_kw / **kw in the fitting loop

    # Transit. Pure geometry: independent of insolation_model and A_B, so it cancels exactly in a truth - model residual.
    f_transit = (batman_utils.flux_transit(planet, t_, t0=0., u=cfg.u) - 1.) * 1e6 if cfg.transit else 0.

    if not cfg.reflected:
        return t_, thermal + f_transit

    # Reflected Part (linear in A)
    A = cfg.A_(A_B_)
    if refl_unit is None:
        refl_unit = reflected_unit(cfg, insolation_model, a_rs=planet.R / planet.r2, T_s=planet.T2_eff, rp_rs=planet.r1 / planet.r2, n_t=n_t)

    return t_, thermal + A * refl_unit + f_transit