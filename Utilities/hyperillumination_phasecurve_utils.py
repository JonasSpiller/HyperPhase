# translating the insolation model with hyperillumination to a light curve signal
import time
import numpy as np
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):   # mute "looking for spidermanrc..." in every process
    import fspiderman as sp
from hyperillumination_utils import *
from HItargets_utils import DEFAULT_A_B, DEFAULT_A, DEFAULT_EMISSIVITY_BOL, DEFAULT_EMISSIVITY, DEFAULT_U_BOL, DEFAULT_U, DEFAULT_N_ORBITS, DEFAULT_N_BINS   # baseline settings
import astro_utils
import batman_utils
import contextlib, io
from astropy import constants as const
from collections.abc import Callable
from dataclasses import field   # dataclass/replace arrive via hyperillumination_utils; field does not
from functools import lru_cache

sigma = const.sigma_sb.value

# Resolution (longitude, latitude nodes) of the insolation / temperature map of every forward-model evaluation; 300/150 differs from 600/300 by ~0.001 ppm (see model_times).
DEFAULT_N_LO, DEFAULT_N_LA = 300, 150


###########################################################################################################################
#####
#####   Full PhaseCurve (reflected + thermal)
#####
###########################################################################################################################

# F: Kreidberg & Loeb 2016 heat redistribution in [0, 0.5], required by (and only used for) insolation_model="naive_redist"
# N_LAYERS=None: the per-model default of thermal_PhaseCurve (40 for hyperillum, 80 for naive / naive_redist)
def full_PhaseCurve(planet, lambda_range, t=None, t0=0., u=DEFAULT_U, u_bol=DEFAULT_U_BOL, A=DEFAULT_A, n_eta=150, n_phi=80, n_tab=1000, n_lo=DEFAULT_N_LO, n_la=DEFAULT_N_LA, insolation_model="hyperillum", F=None, stellar_model=astro_utils.STELLAR_MODEL, filter_path=False, A_B=DEFAULT_A_B, emissivity_bol=DEFAULT_EMISSIVITY_BOL, emissivity=DEFAULT_EMISSIVITY, N_LAYERS=None, include_transit=True, include_occultation=True, spman="fast", spman_kw=None):
    check_redistribution(insolation_model, F)  # fail before the reflected part is computed
    # Reflected Part including transit ("naive_redist" reflects exactly like "naive")
    t_comb, refl, fp_refl = reflected_PhaseCurve(planet, t=t, t0=t0, u=u, u_bol=u_bol, include_transit=include_transit, include_occultation=include_occultation, A=A, n_eta=n_eta, n_phi=n_phi, n_tab=n_tab, insolation_model=insolation_model)
    # Thermal part, without stellar flux (no transit)
    _,  therm, fp_therm, _ = thermal_PhaseCurve(planet, lambda_range, t=t_comb, t0=t0, u=u, u_bol=u_bol, n_lo=n_lo, n_la=n_la, insolation_model=insolation_model, F=F, stellar_model=stellar_model, filter_path=filter_path, A_B = A_B, emissivity_bol=emissivity_bol, emissivity=emissivity, include_transit=False, include_occultation=include_occultation, N_LAYERS=N_LAYERS, spman=spman, spman_kw=spman_kw)

    total = refl + therm
    fp = fp_refl + fp_therm

    return t_comb, total, fp


# Nightside signal at mid-transit (t = t0: the nightside faces the observer), transit dip excluded. It is the hemisphere the observer sees, not the antistellar point:
# under "naive" the nightside is 0 K, so for i < 90 deg the value is only the sliver of dayside visible past the pole (exactly 0 edge-on); "hyperillum" adds the real emission of the
# penumbra band past 90 deg and "naive_redist" the uniform Kreidberg & Loeb nightside (F required). A single time point, so SPIDERMAN is evaluated exactly (spman="normal", no phase interpolation).
#   unit="flux":        (thermal, reflected) planet signal F_p/F_s [ppm]
#   unit="temperature": nightside brightness temperature [K] of the THERMAL part, astro_utils.T_bright on the band flux exactly like the dayside value from the occultation depth
#                       (a blackbody brightness temperature: below the surface temperature for emissivity < 1).
#                       Reflected light is not thermal, so it is neither included nor computed. 0 K when the thermal signal vanishes (<= 0, or below the band flux at T_bright's 10 K lower bracket).
# No limb-darkening argument: the transit is excluded, so u has no effect. Lambda_range / filter_path as in thermal_PhaseCurve (tophat without a filter_path).
def nightside_flux(planet, lambda_range, insolation_model="hyperillum", unit="flux", t0=0., u_bol=DEFAULT_U_BOL, A_B=DEFAULT_A_B, A=DEFAULT_A, F=None, emissivity_bol=DEFAULT_EMISSIVITY_BOL, emissivity=DEFAULT_EMISSIVITY, filter_path=False, stellar_model=astro_utils.STELLAR_MODEL, N_LAYERS=None, n_lo=DEFAULT_N_LO, n_la=DEFAULT_N_LA, n_eta=150, n_phi=80, n_tab=1000):
    if unit not in ("flux", "temperature"): raise ValueError(f"unit must be 'flux' or 'temperature', got {unit!r}")
    t = np.array([t0])
    _, therm, _, _ = thermal_PhaseCurve(planet, lambda_range, t=t, t0=t0, u_bol=u_bol, n_lo=n_lo, n_la=n_la, insolation_model=insolation_model, F=F, stellar_model=stellar_model, filter_path=filter_path, A_B=A_B, emissivity_bol=emissivity_bol, emissivity=emissivity, N_LAYERS=N_LAYERS, include_transit=False, spman="normal")
    therm = 1e6 * float(therm[0])
    if unit == "temperature":
        if therm <= 0.: return 0.
        try: return float(astro_utils.T_bright(therm, planet, lambda_range, filter_path=filter_path or None, stellar_model=stellar_model))
        except ValueError: return 0.   # brentq has no sign change: the signal is below the band flux of the 10 K lower bracket
    _, refl, _ = reflected_PhaseCurve(planet, t=t, t0=t0, u_bol=u_bol, include_transit=False, A=A, n_eta=n_eta, n_phi=n_phi, n_tab=n_tab, insolation_model=insolation_model)
    return therm, 1e6 * float(refl[0])


###########################################################################################################################
#####
#####   Reflected Lightcurve
#####
###########################################################################################################################


# Reflected light curve signal accounting for hyperillumination
# This assumes Lambertian (isotropic, grey) reflection, circular planet
# Inspired by Kopal eq. 4
def reflected_lightcurve(epsilon, planet, u_bol=DEFAULT_U_BOL, unit="normalized", lambda_band_ratio=1., A=DEFAULT_A, n_eta=150, n_phi=80, n_tab=1000, runtime=False, insolation_model="hyperillum"):
    """
    epsilon: angle substellar spot to line of sight [rad]
    unit: Whether to return normalized light curve signal (Fp/Fs) or radiant intensity ([W/sr])
    lambda_band_ratio: Only relevant for unit=='radiant intensity'. If measurements are taken in a wavelength band (not bolometric), this is the ratio Fs_band/Fs. Needed to correctly scale the absolute radial intensity in that band.
    A: Spherical bolometric or band/throughput-integrated albedo (NOT geometric albedo). For bolometric measurements, this function is also suitable to calculate reflected + THERMAL light curve signal - for this set A=1.
    n_eta, n_phi: Gauss-Legendre order for the inner (eta) / outer (phi) integral
    n_tab: resolution of the gamma1-grid used to tabulate F_fi/F_pen
    """
    #TODO: When passing a filter T(λ) one can calculate lambda_band_ratio = ∫T(λ)B_λ(T_eff)dλ / σT_eff⁴
    #TODO: For reflected light insolation_penumbra_hyperillum() should actually take u (MIR band), not u_bol. For simplicity (since quadratic limb darkening is not implemented in the hyperillum model) we pass the bolometric u_bol and assume the impact to be negligible for now (reflected light only minor contribution).
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
            Fpen_grid = insolation_penumbra_hyperillum(gp_pen_grid, planet, u_bol=u_bol)

            return np.where(
                gp <= gp1, np.interp(gp, gp_fi_grid, Ffi_grid),
                np.where(gp <= gp2, np.interp(gp, gp_pen_grid, Fpen_grid), 0.))

        elif insolation_model in ("naive", "naive_redist"):  # naive (point-like) insolation model; heat redistribution does not change the reflected light
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
        return result * lambda_band_ratio  # radiant intensity (power/solid angle emitted towards the observer [W/sr]) - measured is flux [W/m^2]: result/d^2
    else:
        raise ValueError("unit must be either 'normalized' or 'radiant intensity'")


# Naive reflected phase curve signal for a Lambert-scattering sphere (isotropic, angle-independent single-scattering albedo)
def lambertian_reflectance(epsilon, planet, unit="normalized", lambda_band_ratio=1., A=DEFAULT_A):
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
def orbital_reflected_lightcurve(t, planet, t0=0., n_eps=300, A=DEFAULT_A, u_bol=DEFAULT_U_BOL, n_eta=150, n_phi=80, n_tab=1000, insolation_model="hyperillum"):
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    f = astro_utils.true_anomaly(t, period, ecc, w, t0=t0)
    epsilon = np.arccos(-np.sin(np.radians(incl)) * np.sin(np.radians(w) + f))

    # THE CURRENT CODE NEGLECTS CHANGES IN R THROUGHOUT THE ORBIT! All functions are designed for scalar R, which is why the formula below currently fails. Needs adjustment for ecc>0.
    # The current code fixes the planet-star distance to the semi-major axis R
    # R_real = R * (1. - ecc**2) / (1. + ecc * np.cos(f))

    # evaluating the expensive integral on a coarse monotonic epsilon grid, then interpolate onto t
    eps_grid = np.linspace(epsilon.min(), epsilon.max(), n_eps)
    ref_grid = reflected_lightcurve(eps_grid, planet, u_bol=u_bol, A=A, n_eta=n_eta, n_phi=n_phi, n_tab=n_tab, insolation_model=insolation_model)
    return np.interp(epsilon, eps_grid, ref_grid)


# Returns (t, flux): timestamps and the full reflected phase curve combining transit, secondary eclipse / occultation, and sinusoidal phase modulation.
# Reflected phasecurve valid for any wavelength range/filter and stellar spectrum as wavelength dependence cancels out in Fp/Fs (Fp linearly dependent on Fs)
# Band-integrated albedo A however depends on both detector throughput (wavelength range) and stellar spectrum.
# To assume independence on the stellar spectrum we have to neglect stellar limb darkening + reddening - this would have an effect on the penumbral reflection (this is justified since limb darkening weak in IR)
# This assumes Lambertian (isotropic, grey) reflectance, circular planet
def reflected_PhaseCurve(planet, t=None, t0=0., u=DEFAULT_U, u_bol=DEFAULT_U_BOL, include_transit=True, include_occultation=True, A=DEFAULT_A, n_eta=150, n_phi=80, n_tab=1000, insolation_model="hyperillum"):
    #  r1, r2, R: planetary radius, stellar radius, semi major axis in consistent units
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    if t is None: t = np.linspace(0., period, 1000, endpoint=False)
    t_corr = batman_utils.light_travel_time_correction(t, planet, t0=t0)

    orbital_ref_lightcurve = orbital_reflected_lightcurve(t, planet, t0=t0, A=A, u_bol=u_bol, n_eta=n_eta, n_phi=n_phi, n_tab=n_tab, insolation_model=insolation_model)

    eps_occultation = np.arccos(np.sin(np.radians(incl)))  # epsilon at full phase
    fp = reflected_lightcurve(eps_occultation, planet, A=A, u_bol=u_bol, n_eta=n_eta, n_phi=n_phi, n_tab=n_tab, insolation_model=insolation_model)
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

# SPIDERMAN stellar grid [teffs, totals]: totals = in-band surface intensity of the star (astro_utils.stellar_intensity) at each T_eff node, integrated exactly like fspiderman's gen_grid (sum_flux over the closed band [l1, l2], response R*lambda/hc from get_filter), on the filter's samples refined with the model's.
# SPIDERMAN cubic-splines totals at p.T_s and extrapolates silently outside the nodes (check_teff_nodes). The grid belongs to ONE star (logg, feh) and ONE band, which is why FittingConfig derives it.
# [K] the whole NewEra cube range on a UNIFORM 100 K grid: SPIDERMAN's spline (spline.c) swaps the interval widths of its tridiagonal rows, which only matters for non-uniform knots, so the nodes
# must stay uniform. Every node is a NewEra model except 7100 K (the grid steps by 200 K above 7000 K), which newera_spectrum interpolates linearly in flux between 7000 and 7200 K.
TEFF_NODES = np.arange(2300., 7201., 100.)

def stellar_spidergrid(lambda_range, filter_path, logg, feh, teffs=TEFF_NODES, stellar_model=astro_utils.STELLAR_MODEL):
    l1, l2 = map(float, lambda_range)
    return _stellar_spidergrid(l1, l2, filter_path or False, float(logg), float(feh), tuple(float(T) for T in teffs), stellar_model)

# The band integration shared by the star (stellar_spidergrid) and the planet (band_planck): the filter's own samples (or gen_grid's top-hat sampling) plus the band edges, and the
# per-energy response from fspiderman.get_filter (False for a top-hat). Cached and read-only, since every caller shares the arrays.
@lru_cache(maxsize=None)
def _band_setup(l1, l2, filter_path):
    filt = sp.get_filter(filter_path) if filter_path else False
    base = np.union1d(np.asarray(filt[0], float) if filt else np.linspace(l1, l2, 1000), [l1, l2])
    base = base[(base >= l1) & (base <= l2)]
    for a in [base] + (list(filt) if filt else []): a.setflags(write=False)
    return base, filt

@lru_cache(maxsize=None)   # hashable arguments only, hence the wrapper above
def _stellar_spidergrid(l1, l2, filter_path, logg, feh, teffs, stellar_model):
    if len(teffs) < 3 or np.any(np.diff(teffs) <= 0): raise ValueError("teffs: >= 3 strictly increasing T_eff nodes (SPIDERMAN's cubic spline)")
    base, filt = _band_setup(l1, l2, filter_path)
    totals = []
    for T in teffs:
        lam = astro_utils.band_wavelengths(base, T, logg, feh, stellar_model)
        totals.append(sp.stellar_grid.sum_flux(lam, astro_utils.stellar_intensity(lam, T, logg, feh, stellar_model), l1, l2, filt))
    grid = [np.array(teffs), np.array(totals)]
    for a in grid: a.setflags(write=False)   # cached and shared by every FittingConfig of this star and band
    return grid

# In-band intensity of a blackbody at T [K], integrated exactly like the stellar grid's totals (same samples, response and sum_flux). The planet side of T_bright.
def band_planck(lambda_range, filter_path, T):
    l1, l2 = map(float, lambda_range)
    base, filt = _band_setup(l1, l2, filter_path or False)
    return sp.stellar_grid.sum_flux(base, astro_utils.planck_spectrum(base, T), l1, l2, filt)

# The stellar surface intensity SPIDERMAN uses for a T_s: its cubic spline through the grid (spline_cubic_set with ibcbeg = ibcend = 0, "quadratic over the first/last interval", and
# spline_cubic_val in c_src/spline.c), replicated line by line, including the swapped interval widths of the interior rows (a2[i] = A(i,i-1) holds t[i+1]-t[i]), so that T_bright
# inverts SPIDERMAN's depths exactly. Interpolating the spectrum in T instead differs from this by up to ~3e-3 below 3500 K, where the NewEra node integrals are not smooth.
def spidergrid_value(stellar_grid, T_s):
    t, y = (np.asarray(a, float) for a in stellar_grid); n = t.size
    A, b = np.zeros((n, n)), np.zeros(n)
    A[0, 0], A[0, 1] = 1., -1.
    A[-1, -2], A[-1, -1] = -1., 1.
    for i in range(1, n - 1):
        b[i] = (y[i+1] - y[i]) / (t[i+1] - t[i]) - (y[i] - y[i-1]) / (t[i] - t[i-1])
        A[i, i-1], A[i, i], A[i, i+1] = (t[i+1] - t[i]) / 6., (t[i+1] - t[i-1]) / 3., (t[i] - t[i-1]) / 6.
    ypp = np.linalg.solve(A, b)
    i = int(np.clip(np.searchsorted(t, T_s, side="right") - 1, 0, n - 2))   # the interval spline_cubic_val's bisection picks
    dt, h = T_s - t[i], t[i+1] - t[i]
    return float(y[i] + dt * ((y[i+1] - y[i]) / h - (ypp[i+1] / 6. + ypp[i] / 3.) * h + dt * (0.5 * ypp[i] + dt * ((ypp[i+1] - ypp[i]) / (6. * h)))))

def check_teff_nodes(stellar_grid, T_s):
    T_nodes = stellar_grid[0]
    if not T_nodes[0] <= T_s <= T_nodes[-1]: raise ValueError(f"T_s = {T_s:.0f} K outside the stellar grid nodes [{T_nodes[0]:.0f}, {T_nodes[-1]:.0f}] K: SPIDERMAN's spline would extrapolate silently")


# Calculates the occultation depth of a spiderman model (spider_params.eclipse_depth() does not work with direct_T)
# SPIDERMAN emits as a blackbody, so the band-integrated emissivity scales the depth (the thermal Fp/Fs is linear in it). Pass the emissivity of the model the depth belongs to.
def occultation_depth(spider_params, stellar_grid, emissivity=DEFAULT_EMISSIVITY):
    set_ecl = spider_params.eclipse
    t_occ = astro_utils.occultation_time(spider_params.per, spider_params.ecc, spider_params.w, t0=spider_params.t0)
    spider_params.eclipse = False; f_unblocked_occ = spider_params.lightcurve([t_occ], stellar_grid=stellar_grid)[0]
    spider_params.eclipse = True;  f_blocked_occ = spider_params.lightcurve([t_occ], stellar_grid=stellar_grid)[0]
    spider_params.eclipse = set_ecl
    return emissivity * (f_unblocked_occ - f_blocked_occ)*1e6 # Fp/Fs(alpha=0) in ppm


# The A_B-INDEPENDENT half of custom_temperature_grid: the insolation map on the (lo, la) grid.
# It depends only on geometry, so a fit that varies only A_B computes it ONCE and rescales (ee insolation_cache / model_pcurve(insol=...) in the Fitting section)
def insolation_grid(planet, u_bol=DEFAULT_U_BOL, n_lo=DEFAULT_N_LO, n_la=DEFAULT_N_LA, insolation_model="hyperillum"):
    # THE CURRENT CODE NEGLECTS CHANGES IN R THROUGHOUT THE ORBIT! All functions are designed for scalar R, which is why the formula below currently fails. Needs adjustment for ecc>0.
    # The current code fixes the planet-star distance to the semi-major axis R
    # R_real = R * (1. - ecc**2) / (1. + ecc * np.cos(f))

    lo = np.linspace(-180, 180, n_lo)  # longitude [deg], 0 = substellar meridian
    la = np.linspace(-90, 90, n_la)  # latitude [deg], 0 = equator
    LO, LA = np.meshgrid(np.radians(lo), np.radians(la))  # shape (n_la, n_lo), radians

    gamma1_grid = np.arccos(np.cos(LA) * np.cos(LO))  # angular distance from substellar point
    return lo, la, insolation(gamma1_grid, planet, insolation_model, u_bol=u_bol)  # [lo (deg), la (deg), F (W/m^2)]


# Generated custom temperature grid from a given insolation model (insolation_model = hyperillum, naive, naive_redist) as input for spiderman custom forward model
# insol: optional precomputed (lo, la, insol_grid) from insolation_grid / insolation_cache. When given the insolation is NOT recomputed and only the ((1-A_B)/emissivity_bol)^(1/4) rescale is done (~0.3 ms instead of ~0.4 s).
# F: heat redistribution, required by insolation_model="naive_redist" (Kreidberg & Loeb 2016 eq. 1 map, built from the same F-independent naive insolation, so insol stays valid)
def custom_temperature_grid(planet, u_bol=DEFAULT_U_BOL, n_lo=DEFAULT_N_LO, n_la=DEFAULT_N_LA, insolation_model="hyperillum", F=None, A_B=DEFAULT_A_B, emissivity_bol=DEFAULT_EMISSIVITY_BOL, insol=None):
    check_redistribution(insolation_model, F)
    if insol is None:
        lo, la, insol_grid = insolation_grid(planet, u_bol=u_bol, n_lo=n_lo, n_la=n_la, insolation_model=insolation_model)
    else:
        lo, la, insol_grid = insol
        if (len(lo), len(la)) != (n_lo, n_la) or insol_grid.shape != (len(la), len(lo)):
            raise ValueError(f"cached insol is on a {(len(lo), len(la))} grid but n_lo, n_la = {(n_lo, n_la)}")

    temp_grid = temperature_from_insolation(insol_grid, planet, insolation_model, A_B = A_B, emissivity_bol=emissivity_bol, F=F)

    return lo, la, temp_grid  # Custom temperature map [lo (deg), la (deg), T (K)]


# Generating a thermal phase curve from the spiderman custom forward model (inputting a custom temperature map; for "naive_redist" the analytic Louden model instead)
def thermal_PhaseCurve(planet, lambda_range, t=None, t0=0., u=DEFAULT_U, u_bol=DEFAULT_U_BOL, n_lo=DEFAULT_N_LO, n_la=DEFAULT_N_LA, insolation_model="hyperillum", F=None, stellar_model=astro_utils.STELLAR_MODEL, filter_path=False, A_B=DEFAULT_A_B, emissivity_bol=DEFAULT_EMISSIVITY_BOL, emissivity=DEFAULT_EMISSIVITY, N_LAYERS=None, include_transit=True, include_occultation=True, spman="fast", spman_kw=None):
    #  r1, r2, R: planetary radius, stellar radius, semi major axis in consistent units
    # For lambda_range, either pass the effective ranges without a filter_path (spiderman assumes tophat) or provide lambda_range and filter_path.
    # F: Kreidberg & Loeb 2016 heat redistribution in [0, 0.5], required by (and only used for) insolation_model="naive_redist"
    # emissivity_bol sets the temperature map with A_B; emissivity (band-integrated) scales the emitted flux, since SPIDERMAN emits B_lambda(T) of a blackbody
    check_redistribution(insolation_model, F)
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    # N_LAYERS = 30-40 usually sufficient for insolation_model="hyperillum", insolation_model="naive" (and "naive_redist", whose F=0 limit has the same sharp terminator) needs N_LAYERS = 60-80
    if N_LAYERS is None:
        if insolation_model=="hyperillum": N_LAYERS = 40
        else: N_LAYERS = 80

    # "naive_redist": the Kreidberg & Loeb 2016 map is analytic, so SPIDERMAN's Louden model evaluates it instead of a direct_T grid (u_bol, n_lo, n_la unused)
    if insolation_model == "naive_redist":
        return Louden_PhaseCurve(planet, lambda_range, t=t, t0=t0, F=F, u=u, stellar_model=stellar_model, filter_path=filter_path, A_B = A_B, emissivity_bol=emissivity_bol, emissivity=emissivity, N_LAYERS=N_LAYERS, include_transit=include_transit, include_occultation=include_occultation, spman=spman, spman_kw=spman_kw)

    lo, la, temp_grid = custom_temperature_grid(planet, u_bol=u_bol, n_lo=n_lo, n_la=n_la, insolation_model=insolation_model, A_B = A_B, emissivity_bol=emissivity_bol)

    # pass indexing='ij' meshgrids in deg and the transposed temp_grid (wrong orientation fails silently!)
    # The map goes in as is, nightside at exactly 0 K for "naive": the installed fspiderman clamps the direct_T bicubic to the cell corners (fork README, patch 4). Upstream SPIDERMAN
    # overshoots below 0 K at the sharp terminator and exit()s ("CANNOT DO NEGATIVE TEMPERATURES"), which is why older versions of this code floored the map at 100-200 K.
    LO_deg, LA_deg = np.meshgrid(lo, la, indexing='ij')  # longitude/latitude meshgrids [deg]
    grid = sp.format_grid(LO_deg, LA_deg, temp_grid.T)

    spider_params = sp.ModelParams(brightness_model="direct_T")  # the stellar SED enters only through stellar_grid
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
    stellar_grid = stellar_spidergrid(lambda_range, filter_path, planet.logg, planet.feh, stellar_model=stellar_model)
    check_teff_nodes(stellar_grid, T2_eff)

    # Get occultation depth. In the standard spiderman model spider_params.eclipse_depth() segfaults with direct_T, hence the model has to run explicitly without the eclipse to get the depth.
    fp = occultation_depth(spider_params, stellar_grid=stellar_grid, emissivity=emissivity)  # Occultation depth Fp/Fs(alpha=0) in ppm

    # Include Occultation?
    spider_params.eclipse = include_occultation

    # PhaseCurve / Lightcurve
    if t is None: t = np.linspace(0., period, 1000, endpoint=False)
    if spman=="fast": light_curve = fast_lightcurve(spider_params, stellar_grid, t, **(spman_kw or {}))  # phase modulation + occultation, no transit - faster version with interpolation. spman_kw overrides fast_lightcurve's n_phase / pad / split_totality
    else: light_curve = spider_params.lightcurve(t, stellar_grid=stellar_grid)  # phase modulation + occultation, no transit

    # light_curve - 1 is the planet's thermal flux, scaled by the band emissivity (the transit is not)
    if include_transit: # Full PhaseCurve
        f_transit = batman_utils.flux_transit(planet, t, t0=t0, u=u)
        phase_curve = f_transit + emissivity * (light_curve - 1.)
    else: # Only Lightcurve, no stellar flux
        phase_curve = emissivity * (light_curve - 1.)

    return t, phase_curve, fp, spider_params



# The sp.ModelParams of SPIDERMAN's Louden model (analytic Kreidberg & Loeb 2016 eq. 1 map), shared by Louden_PhaseCurve and pcurve_params. eclipse is left to the caller.
def louden_params(planet, lambda_range, F, t0=0., filter_path=False, A_B=DEFAULT_A_B, emissivity_bol=DEFAULT_EMISSIVITY_BOL, N_LAYERS=60):
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
    sp_louden.p_u1 = sp_louden.p_u2 = 0. # PLANETARY limb darkening parameters (default=0)

    sp_louden.T_s = T2_eff
    sp_louden.insol = insol
    sp_louden.albedo = 1. - (1. - A_B) / emissivity_bol   # Louden's sigma T^4 = (1-albedo) S [...] has no emissivity: this effective albedo gives emissivity_bol sigma T^4 = (1-A_B) S [...], as in BBtemp_from_insolation
    sp_louden.redist = F  # heat redistribution Kreidberg-Loeb-2016 (F=0: instant reradiation, F=0.5: full redistribution)
    sp_louden.T_int = 0. # no internal heating
    sp_louden.l1, sp_louden.l2 = lambda_range

    sp_louden.filter = filter_path
    return sp_louden


# Thermal phase curve of insolation_model="naive_redist": heat redistribution F in [0, 0.5] (Kreidberg & Loeb 2016).
# With F=0 equivalent to thermal_PhaseCurve(..., insolation_model="naive") but faster.
def Louden_PhaseCurve(planet, lambda_range, t=None, t0=0., F=0., u=DEFAULT_U, stellar_model=astro_utils.STELLAR_MODEL, filter_path=False, A_B=DEFAULT_A_B, emissivity_bol=DEFAULT_EMISSIVITY_BOL, emissivity=DEFAULT_EMISSIVITY, N_LAYERS=60, include_transit=True, include_occultation=True, spman="fast", spman_kw=None):
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    sp_louden = louden_params(planet, lambda_range, F, t0=t0, filter_path=filter_path, A_B = A_B, emissivity_bol=emissivity_bol, N_LAYERS=N_LAYERS)

    stellar_grid = stellar_spidergrid(lambda_range, filter_path, planet.logg, planet.feh, stellar_model=stellar_model)
    check_teff_nodes(stellar_grid, T2_eff)

    # Get occultation depth.
    fp = occultation_depth(sp_louden, stellar_grid=stellar_grid, emissivity=emissivity)  # Occultation depth Fp/Fs(alpha=0) in ppm

    # Include Occultation?
    sp_louden.eclipse = include_occultation

    # PhaseCurve / Lightcurve
    if t is None: t = np.linspace(0., period, 1000, endpoint=False)
    if spman=="fast": light_curve = fast_lightcurve(sp_louden, stellar_grid, t, **(spman_kw or {}))  # phase modulation + occultation, no transit. spman_kw overrides fast_lightcurve's n_phase / pad / split_totality
    else: light_curve = sp_louden.lightcurve(t, stellar_grid=stellar_grid)  # phase modulation + occultation, no transit

    # light_curve - 1 is the planet's thermal flux, scaled by the band emissivity (the transit is not)
    if include_transit:  # Full PhaseCurve
        f_transit = batman_utils.flux_transit(planet, t, t0=t0, u=u)
        phase_curve = f_transit + emissivity * (light_curve - 1.)
    else:  # Only Lightcurve, no stellar flux
        phase_curve = emissivity * (light_curve - 1.)

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
    A: float = DEFAULT_A
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
                            # truth for geometry (r1, r2, R, a_AU), orbit (period, incl, ecc, w) and the star (T2_eff, logg, feh)
    filter_path: str = None   # SPIDERMAN throughput file. Left out (with l1, l2) it is a band-less template, which simple_bias_analysis.instrument_cfg completes with the band(s) of an instrument
    l1: float = np.nan      # filter blue edge [m]
    l2: float = np.nan      # filter red edge [m]
    A_: object = ConstantBandAlbedo()   # band-integrated albedo A as a function of the Bond albedo A_B (only needed when reflected=True)
    emissivity_bol: float = DEFAULT_EMISSIVITY_BOL   # bolometric emissivity: sets the temperature map with A_B (exactly degenerate with it)
    emissivity: float = DEFAULT_EMISSIVITY       # band-integrated emissivity: scales the thermal planet flux
    sigma_1min: float = 323.  # detection error [ppm per minute of integration]
    n_orbits: int = DEFAULT_N_ORBITS  # number of orbits observed and phase-folded
    n_bins: int = DEFAULT_N_BINS  # bins per period; the width is DERIVED so it tiles exactly
    u_bol: float = DEFAULT_U_BOL   # bolometric linear limb darkening of the irradiance (0 = uniform disk)
    u: tuple = DEFAULT_U       # quadratic stellar limb darkening [u1, u2] of the transit
    reflected: bool = True  # True -> full (reflected + thermal); False -> thermal only
    transit: bool = True    # include the primary transit (batman)
    stellar_model: str = astro_utils.STELLAR_MODEL   # stellar SED of stellar_grid (and of dynesty_utils.brightness_temperature): "newera", an stsynphot grid name or "blackbody"
    stellar_grid: object = field(init=False, repr=False, default=None)   # DERIVED from planet (logg, feh) and the band in __post_init__, so every replace(cfg, planet=..., l1=..., ...) rebuilds it (cached in stellar_spidergrid) and it can never belong to another star or band

    def __post_init__(self):
        if np.isfinite(self.l1) and np.isfinite(self.l2): self.stellar_grid = stellar_spidergrid((self.l1, self.l2), self.filter_path, self.planet.logg, self.planet.feh, stellar_model=self.stellar_model)   # l1 = l2 = nan: a band-less template cfg, instrument_cfg fills the band

    @property
    def sigma_bin(self):
        minutes_binning = self.planet.period * 24 * 60 / self.n_bins
        return self.sigma_1min / np.sqrt(minutes_binning * self.n_orbits)

    # Noise of a fitted phase-curve semi-amplitude (first harmonic over n_orbits): sigma_1min sqrt(2 / t_obs[min]), i.e. sigma(a) = sqrt(2/N) sigma per point (Montgomery & O'Donoghue 1999,
    # DSSN 13, 28; Kallinger et al. 2008, A&A 481, 571, eq. 1). Binning-independent, so it is the reference for a flux level
    # such as the nightside signal at mid-transit (sigma_bin is not: it scales with sqrt(n_bins)). A lower bound: it ignores the compensation by A_B that dchi2_F / z_F includes.
    @property
    def sigma_amp(self):
        return self.sigma_1min * np.sqrt(2. / (self.planet.period * 24 * 60 * self.n_orbits))


# The PlanetParams that model_pcurve actually evaluates, at the swept geometry, in pl's own units.
# Factored out so that a cached insolation map (insolation_cache) is guaranteed to be built at exactly the geometry the corresponding model_pcurve call will use — the caches cannot silently go stale.
# A T_s sweep changes T_eff only: log g and [Fe/H] stay those of cfg.planet (cfg.stellar_grid belongs to that star), so the swept stars are hypothetical. A real M dwarf has log g ~ 5 rather than the
# reference star's 4.5, which changes its mid-IR flux by ~2 % at 3000 K and ~6 % at 2600 K (LRS); bias maps are insensitive to it (the bias does not depend on k), their significances are not.
def swept_planet(cfg, a_rs=None, T_s=None, rp_rs=None):
    pl = cfg.planet
    a_rs0 = pl.R / pl.r2  # default a/Rs (for the a_AU scaling)
    a_rs = a_rs0 if a_rs is None else a_rs
    T_s = pl.T2_eff if T_s is None else T_s
    rp_rs = pl.r1 / pl.r2 if rp_rs is None else rp_rs
    check_teff_nodes(cfg.stellar_grid, T_s)
    # rebuild the planet at the swept geometry in pl's own units; a_AU scales with a/Rs (stellar size fixed)
    return replace(pl, r1=rp_rs * pl.r2, R=a_rs * pl.r2, T2_eff=T_s, a_AU=pl.a_AU * a_rs / a_rs0)


# Ready-made `insol` argument for model_pcurve, built at exactly the geometry model_pcurve would use.
# ONLY valid while a/r2, r1/r2, T2_eff, cfg.u_bol, n_lo and n_la are held fixed — see dynesty_utils.GaussianLogL, which builds this and invalidates it automatically when the corresponding parameter is sampled.
def insolation_cache(cfg, insolation_model, n_lo=DEFAULT_N_LO, n_la=DEFAULT_N_LA, **geom):
    planet = swept_planet(cfg, **geom)   # **geom rather than named args, so adding a parameter to swept_planet is enough
    return insolation_grid(planet, u_bol=cfg.u_bol, n_lo=n_lo, n_la=n_la, insolation_model=insolation_model)


# Uniform time grid for a model that is going to be BIN-AVERAGED: the midpoints of the n_t sub-intervals, not their left edges.
# binned() averages these samples per bin, so left edges would make that a left Riemann sum - first order, with error
# ~ (step/2) x slope, i.e. ~ transit_depth / (2 * n_t/n_bins) in the bins the transit ingress/egress falls into. At the usual
# n_t = 50 per bin that is several ppm (measured 4.62 ppm for a 454 ppm transit), orders of magnitude above every other
# discretisation in the forward model (n_layers 40 vs 80: 0.05 ppm; n_lo/n_la 300/150 vs 600/300: 0.001 ppm). The midpoint
# rule removes it for free: same number of evaluations, ~70x smaller error, and it keeps converging with n_t.
def model_times(period, n_t):
    return np.linspace(0., period, int(n_t), endpoint=False) + 0.5 * period / int(n_t)


# Reflected phase-curve shape at unit albedo (A = 1)
def reflected_unit(cfg, insolation_model, a_rs=None, T_s=None, rp_rs=None, n_t=None, **_):
    pl = cfg.planet
    a_rs = pl.R / pl.r2  if a_rs is None else a_rs   # default a/Rs from PlanetParams
    T_s = pl.T2_eff      if T_s is None else T_s
    rp_rs = pl.r1 / pl.r2  if rp_rs is None else rp_rs      # default Rp/Rs from PlanetParams
    n_t = 50 * cfg.n_bins if n_t is None else n_t

    t_ = model_times(pl.period, n_t)
    # set the ratios in the planet's own units (reflected is geometric -> a_AU unused, left as-is)
    planet = replace(pl, r1=rp_rs * pl.r2, R=a_rs * pl.r2, T2_eff=T_s)
    _, refl, _ = reflected_PhaseCurve(planet, t=t_, include_transit=False, A=1., u_bol=cfg.u_bol, insolation_model=insolation_model)
    return refl * 1e6


# The sp.ModelParams that model_pcurve feeds to SPIDERMAN, factored out exactly like swept_planet so the OCCULTATION DEPTH can be taken at the fitted geometry with the existing occultation_depth(p, stellar_grid=cfg.stellar_grid) instead of duplicating this setup.
# F: heat redistribution, required by insolation_model="naive_redist"
def pcurve_params(cfg, insolation_model, A_B_, F=None, a_rs=None, T_s=None, rp_rs=None, n_layers=None, n_lo=DEFAULT_N_LO, n_la=DEFAULT_N_LA, insol=None):
    check_redistribution(insolation_model, F)
    # n_layers = 30-40 usually sufficient for insolation_model="hyperillum", insolation_model="naive" (and "naive_redist") needs n_layers = 60-80
    # Measured (TOI-2431 b/NIRSpec and K2-141 b/MIRI-LRS, vs n_layers=400): on the light CURVE the two models converge alike
    # (max |d| at n_layers=40: naive 0.09-0.17 ppm, hyperillum 0.08-0.14 ppm; <0.02 sigma_bin once binned), and the naive-vs-truth
    # A_B bias is stable to ~1e-4 from n_layers=20 up. The 60-80 matters for the OCCULTATION DEPTH, where naive converges only
    # ~N^-1 against hyperillum's ~N^-2 (K2-141 b: 0.33 / 0.20 / 0.08 ppm at n_layers 40 / 60 / 120, vs 0.13 / 0.07 / 0.02).
    # Since the hyperillum-naive depth difference is itself only ~1 ppm, use n_layers >= 120 when COMPARING depths between models.
    if n_layers is None:
        if insolation_model == "hyperillum":
            n_layers = 40
        else:
            n_layers = 60

    planet = swept_planet(cfg, a_rs=a_rs, T_s=T_s, rp_rs=rp_rs)

    # "naive_redist": SPIDERMAN's analytic Louden map, as in thermal_PhaseCurve (insol, n_lo and n_la are unused)
    if insolation_model == "naive_redist":
        p = louden_params(planet, (cfg.l1, cfg.l2), F, t0=0., filter_path=cfg.filter_path, A_B=A_B_, emissivity_bol=cfg.emissivity_bol, N_LAYERS=n_layers)
        p.eclipse = True
        return p

    # insol=... reuses the A_B-independent insolation map, leaving only the exact T = T(A_B=0, emissivity_bol=1)*((1-A_B)/emissivity_bol)^(1/4) rescale (valid because insol depends on geometry alone).
    lo, la, Tg = custom_temperature_grid(planet, u_bol=cfg.u_bol, n_lo=n_lo, n_la=n_la, insolation_model=insolation_model, A_B=A_B_, emissivity_bol=cfg.emissivity_bol, insol=insol)
    LO, LA = np.meshgrid(lo, la, indexing="ij")
    p = sp.ModelParams(brightness_model="direct_T")   # the stellar SED enters only through cfg.stellar_grid
    p.n_layers = n_layers
    p.t0, p.per = 0., planet.period
    p.a_abs = planet.a_AU
    p.inc, p.ecc, p.w = planet.incl, planet.ecc, planet.w
    p.rp, p.a = planet.r1 / planet.r2, planet.R / planet.r2
    p.p_u1 = p.p_u2 = 0.
    p.T_s, p.l1, p.l2 = planet.T2_eff, cfg.l1, cfg.l2
    p.grid = sp.format_grid(LO, LA, Tg.T)   # no temperature floor, see thermal_PhaseCurve
    p.filter, p.eclipse = cfg.filter_path, True
    return p


# Fast, stripped-down phase curve [ppm] for the fitting loop
# refl_unit, insol: precomputed geometry-only pieces (reflected_unit / insolation_cache). Both are optional and purely a speed-up: passing them is bit-identical to letting them be rebuilt here.
# F: heat redistribution, required by insolation_model="naive_redist" (whose thermal part is the Louden model, so insol is unused there; its reflected part is that of "naive")
def model_pcurve(cfg, insolation_model, A_B_, F=None, a_rs=None, T_s=None, rp_rs=None, n_t=None, n_layers=None, n_lo=DEFAULT_N_LO, n_la=DEFAULT_N_LA, refl_unit=None, insol=None, spman_kw=None):
    check_redistribution(insolation_model, F)
    pl = cfg.planet
    n_t = 50 * cfg.n_bins if n_t is None else n_t

    t_ = model_times(pl.period, n_t)
    planet = swept_planet(cfg, a_rs=a_rs, T_s=T_s, rp_rs=rp_rs)

    # Thermal Part
    p = pcurve_params(cfg, insolation_model, A_B_, F=F, a_rs=a_rs, T_s=T_s, rp_rs=rp_rs, n_layers=n_layers, n_lo=n_lo, n_la=n_la, insol=insol)
    thermal = cfg.emissivity * (fast_lightcurve(p, cfg.stellar_grid, t_, **{"n_phase": 24, **(spman_kw or {})}) - 1) * 1e6  # SPIDERMAN emits as a blackbody, the band emissivity scales it. spman_kw overrides n_phase / pad / split_totality; it reaches here through **model_kw / **kw in the fitting loop

    # Transit. Pure geometry: independent of insolation_model and A_B, so it cancels exactly in a truth - model residual.
    f_transit = (batman_utils.flux_transit(planet, t_, t0=0., u=cfg.u) - 1.) * 1e6 if cfg.transit else 0.

    if not cfg.reflected:
        return t_, thermal + f_transit

    # Reflected Part (linear in A)
    A = cfg.A_(A_B_)
    if refl_unit is None:
        refl_unit = reflected_unit(cfg, insolation_model, a_rs=planet.R / planet.r2, T_s=planet.T2_eff, rp_rs=planet.r1 / planet.r2, n_t=n_t)

    return t_, thermal + A * refl_unit + f_transit