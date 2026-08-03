# translating the insolation model with hyperillumination to a light curve signal
import time
import numpy as np
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):   # mute "looking for spidermanrc..." in every process
    import fspiderman as sp
from hyperillumination_utils import *
import kepler_utils
import batman_utils
import contextlib, io

sigma = 5.670374e-8  # W m^-2 K^-4


###########################################################################################################################
#####
#####   Full PhaseCurve (reflected + thermal)
#####
###########################################################################################################################

def full_PhaseCurve(planet, lambda_range, t=None, t0=0., u=[0.1, 0.3], u_B=0., A=0.1, n_eta=150, n_phi=80, n_tab=1000, n_lo=200, n_la=100, insolation_model="hyperillum", stellar_model="blackbody", filter_path=False, A_B = 0.1, emissivity=1., N_LAYERS=60, include_transit=True):
    # Reflected Part including transit
    t_comb, refl, fp_refl = reflected_PhaseCurve(planet, t=t, t0=t0, u=u, u_B=u_B, include_transit=include_transit, A=A, n_eta=n_eta, n_phi=n_phi, n_tab=n_tab, insolation_model=insolation_model)
    # Thermal part, without stellar flux (no transit)
    _,  therm, fp_therm, _ = thermal_PhaseCurve(planet, lambda_range, t=t_comb, t0=t0, u=u, u_B=u_B, n_lo=n_lo, n_la=n_la, insolation_model=insolation_model, stellar_model=stellar_model, filter_path=filter_path, A_B = A_B, emissivity=emissivity, include_transit=False, N_LAYERS=N_LAYERS)

    total = refl + therm
    fp = fp_refl + fp_therm

    return t_comb, total, fp


###########################################################################################################################
#####
#####   Reflected Lightcurve
#####
###########################################################################################################################


# Reflected light curve signal accounting for hyperillumination
# This assumes Lambertian (isotropic, grey) reflectance, circular planet
# Inspired by Kopal eq. 4
def reflected_lightcurve(epsilon, planet, u_B=0., unit="normalized", lambda_band_ratio=1., A=0.1, n_eta=150, n_phi=80, n_tab=1000, runtime=False, insolation_model="hyperillum"):
    """
    epsilon: angle substellar spot to line of sight [rad]
    unit: Whether to return normalized light curve signal (Fp/Fs) or radiant intensity ([W/sr])
    lambda_band_ratio: Only relevant for unit=='radiant intensity'. If measurements are taken in a wavelength band (not bolometric), this is the ratio Fs_band/Fs. Needed to correctly scale the absolute radial intensity in that band.
    A: Lambertian bolometric or band/throughput-integrated albedo (NOT geometric albedo). For bolometric measurements, this function is also suitable to calculate reflected + THERMAL light curve signal - for this set A=1.
    n_eta, n_phi: Gauss-Legendre order for the inner (eta) / outer (phi) integral
    n_tab: resolution of the gamma1-grid used to tabulate F_fi/F_pen
    """
    #TODO: When passing a filter T(λ) one can calculate lambda_band_ratio = ∫T(λ)B_λ(T_eff)dλ / σT_eff⁴
    # TODO: For reflected light insolation_penumbra_hyperillum() should actually take u (MIR band), not u_B. For simplicity (since quadratic limb darkening is not implemented in the hyperillum model) we pass the bolometric u_B and assume the impact to be negligible for now.
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

    f = kepler_utils.true_anomaly(t, period, ecc, w, t0=t0)
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
def reflected_PhaseCurve(planet, t=None, t0=0., u=[0.1, 0.3], u_B=0., include_transit=True, A=0.1, n_eta=150, n_phi=80, n_tab=1000, insolation_model="hyperillum"):
    #  r1, r2, R: planetary radius, stellar radius, semi major axis in consistent units
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    if t is None: t = np.linspace(0., period, 1000)

    t_corr = batman_utils.light_travel_time_correction(t, planet, t0=t0)

    orbital_ref_lightcurve = orbital_reflected_lightcurve(t, planet, t0=t0, A=A, u_B=u_B, n_eta=n_eta, n_phi=n_phi, n_tab=n_tab, insolation_model=insolation_model)
    eps_occultation = np.arccos(np.sin(np.radians(incl)))  # epsilon at full phase
    fp = reflected_lightcurve(eps_occultation, planet, A=A, u_B=u_B, n_eta=n_eta, n_phi=n_phi, n_tab=n_tab, insolation_model=insolation_model)
    if fp == 0.: return t_corr, batman_utils.flux_transit(planet, t, t0=t0, u=u), fp*1e6  # no reflection (e.g. when albedo=0) -> no phase curve, only transit

    # Full Phase Curve Model. This model ignores the exact eclipse geometry during ingress and egress, instead it assumes the integrated flux to be uniformly distributed over the disk
    phase_mod = orbital_ref_lightcurve / fp
    occultation = batman_utils.flux_occultation(planet, fp, t, t0=t0)

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

# Calculates the occultation depth of a spiderman model (spider_params.eclipse_depth() does not work with direct_T)
def occultation_depth(spider_params, stellar_grid=None):
    set_ecl = spider_params.eclipse
    if stellar_grid is None:
        with contextlib.redirect_stdout(io.StringIO()): # mute notification: "no stellar models provided, using blackbody approximation"
            stellar_grid = sp.stellar_grid.gen_grid(spider_params.l1, spider_params.l2, logg=4.5, response=spider_params.filter, stellar_model=spider_params.stellar_model)
    t_occ = kepler_utils.occultation_time(spider_params.per, spider_params.ecc, spider_params.w, t0=spider_params.t0)
    spider_params.eclipse = False; f_unblocked_occ = spider_params.lightcurve([t_occ], stellar_grid=stellar_grid)[0]
    spider_params.eclipse = True;  f_blocked_occ = spider_params.lightcurve([t_occ], stellar_grid=stellar_grid)[0]
    spider_params.eclipse = set_ecl
    return (f_unblocked_occ - f_blocked_occ)*1e6 # Fp/Fs(alpha=0) in ppm


# Generated custom temperature grid from a given insolation model (insolation_model = hyperillum, naive) as input for spiderman custom forward model
def custom_temperature_grid(planet, u_B=0., n_lo=200, n_la=100, insolation_model="hyperillum", A_B = 0.1, emissivity=1.):
    # THE CURRENT CODE NEGLECTS CHANGES IN R THROUGHOUT THE ORBIT! All functions are designed for scalar R, which is why the formula below currently fails. Needs adjustment for ecc>0.
    # The current code fixes the planet-star distance to the semi-major axis R
    # R_real = R * (1. - ecc**2) / (1. + ecc * np.cos(f))

    lo = np.linspace(-180, 180, n_lo)  # longitude [deg], 0 = substellar meridian
    la = np.linspace(-90, 90, n_la)  # latitude [deg], 0 = equator
    LO, LA = np.meshgrid(np.radians(lo), np.radians(la))  # shape (n_la, n_lo), radians

    gamma1_grid = np.arccos(np.cos(LA) * np.cos(LO))  # angular distance from substellar point
    insol_grid = insolation(gamma1_grid, planet, insolation_model, u_B=u_B)
    temp_grid = BBtemp_from_insolation(insol_grid, A_B = A_B, emissivity=emissivity)

    return lo, la, temp_grid  # Custom temperature map [lo (deg), la (deg), T (K)]


# Generating a thermal phase curve from the spiderman custom forward model (inputting a custom temperature map)
def thermal_PhaseCurve(planet, lambda_range, t=None, t0=0., u=[0.1, 0.3], u_B=0., n_lo=200, n_la=100, insolation_model="hyperillum", stellar_model="blackbody", filter_path=False, A_B = 0.1, emissivity=1., N_LAYERS=None, include_transit=True, include_occultation=True, spman="fast"):
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
    if t is None: t = np.linspace(0., period, 1000)
    if spman=="fast": light_curve = fast_lightcurve(spider_params, stellar_grid, t)  # phase modulation + occultation, no transit - faster version with interpolation
    else: light_curve = spider_params.lightcurve(t, stellar_grid=stellar_grid)  # phase modulation + occultation, no transit

    if include_transit: # Full PhaseCurve
        f_transit = batman_utils.flux_transit(planet, t, t0=t0, u=u)
        phase_curve = f_transit + (light_curve - 1.)
    else: # Only Lightcurve, no stellar flux
        phase_curve = light_curve - 1.

    return t, phase_curve, fp, spider_params



# Equivalent to thermal_PhaseCurve(..., insolation_model="naive") but faster.
def Louden_PhaseCurve(planet, lambda_range, t=None, t0=0., u=[0.1, 0.3], stellar_model="blackbody", filter_path=False, A_B = 0.1, N_LAYERS=60, include_transit=True, include_occultation=True, spman="fast"):
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
    if t is None: t = np.linspace(0., period, 1000)
    if spman=="fast": light_curve = fast_lightcurve(sp_louden, stellar_grid, t)  # phase modulation + occultation, no transit
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


def occultation_window(spider_params, pad=0.03):
    # (t_ingress, t_egress) of the occultation, padded by `pad` durations on each side.
    t_occultation = kepler_utils.occultation_time(spider_params.per, spider_params.ecc, spider_params.w, t0=spider_params.t0)
    b = spider_params.a * np.cos(np.radians(spider_params.inc))
    dur = spider_params.per / np.pi * np.arcsin(np.sqrt(max((1 + spider_params.rp) ** 2 - b ** 2, 0.)) / spider_params.a)
    return t_occultation - (0.5 + pad) * dur, t_occultation + (0.5 + pad) * dur

def in_occultation(spider_params, t, pad=0.03):
    # Boolean mask: which of `t` fall inside an occultation, in ANY orbit spanned by `t`.
    t_ingress, t_egress = occultation_window(spider_params, pad=pad)
    t_centre = 0.5 * (t_ingress + t_egress)
    half = 0.5 * (t_egress - t_ingress)
    per = spider_params.per
    dt = np.mod(np.asarray(t, dtype=float) - t_centre + 0.5 * per, per) - 0.5 * per  # signed offset to nearest occultation
    return np.abs(dt) <= half


def fast_lightcurve(spider_params, stellar_grid, t, n_phase=40):
    # SPIDERMAN lightcurve evaluated cheaply by splitting the orbit in smooth, out-of-occultation part and occulation part.
    # only first few fourier coefficients contribute to lightcurve -> only few interpolation points n_phase needed (Nyquist)

    t = np.asarray(t, dtype=float)
    orig_eclipse = spider_params.eclipse

    try:
        spider_params.eclipse = False # smooth phase modulation, no occultation
        t_nodes = spider_params.t0 + spider_params.per * np.arange(n_phase) / n_phase
        y_nodes = spider_params.lightcurve(t_nodes, stellar_grid=stellar_grid)
        out = _trig_interp(y_nodes, spider_params.per, t, t0=spider_params.t0) # signal fully captured in ~11 Fourier coefficients

        if orig_eclipse:  # exact inside EVERY occultation spanned by t
            m = in_occultation(spider_params, t)
            if m.any():
                spider_params.eclipse = True
                out[m] = spider_params.lightcurve(t[m], stellar_grid=stellar_grid) # occultation calculated exactly
    finally:
        spider_params.eclipse = orig_eclipse # restore even if lightcurve() raised, else spider_params is left dirty

    return out

