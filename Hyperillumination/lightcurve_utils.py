# translating the insolation model with hyperillumination to a light curve signal
import time
import numpy as np
from hyperillumination_utils import *

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
def reflected_lightcurve(epsilon, r1, r2, R, T2_eff, u=0.0, unit="normalized", lambda_band_ratio=1., A=0.1, n_eta=150, n_phi=80, n_tab=1000, runtime=False, insolation_model="kopal"):
    """
    epsilon: angle substellar spot to line of sight
    unit: Whether to return normalized light curve signal (Fp/Fs) or radiant intensity ([W/sr])
    lambda_band_ratio: Only relevant for unit=='radiant intensity'. If measurements are taken in a wavelength band (not bolometric), this is the ratio Fs_band/Fs.
    A: Bolometric or band integrated albedo. For bolometric measurements, this function is also suitable to calculate reflected + THERMAL light curve signal - for this set A=1.
    n_eta, n_phi: Gauss-Legendre order for the inner (eta) / outer (phi) integral
    n_tab: resolution of the gamma1-grid used to tabulate F_fi/F_pen
    """

    t0 = time.perf_counter()
    epsilon = np.asarray(epsilon, dtype=float)
    eps_flat = epsilon.ravel()

    gp1 = terminator_penumbra(r1, r2, R)     # arccos((r1+r2)/R)
    gp2 = terminator_nightside(r1, r2, R)    # arccos((r1-r2)/R)

    def F(gp):
        if insolation_model=="kopal":
            # tabulate F_fi, F_pen on a gamma1-grid once and interpolate at quadrature nodes
            gp_fi_grid = np.linspace(0., gp1, n_tab)
            Ffi_grid = insolation_fullyilluminated_kopal(gp_fi_grid, r1, r2, R, T2_eff, u=u)

            gp_pen_grid = np.linspace(gp1, gp2, n_tab)
            Fpen_grid = insolation_penumbra_kopal(gp_pen_grid, r1, r2, R, T2_eff, u=u)

            return np.where(
                gp <= gp1, np.interp(gp, gp_fi_grid, Ffi_grid),
                np.where(gp <= gp2, np.interp(gp, gp_pen_grid, Fpen_grid), 0.))

        elif insolation_model=="naive": # naive (point-like) insolation model:
            return naive_insolation(gp, r2, R, T2_eff)

        else: raise ValueError("insolation_model must be either 'kopal' or 'naive'")

    # fixed Gauss-Legendre nodes, shared across all epsilon values
    psi_nodes, psi_w = np.polynomial.legendre.leggauss(n_phi)
    psi = psi_nodes * (np.pi / 2.)                   # [-1,1] -> [-pi/2, pi/2]
    psi_w = psi_w * (np.pi / 2.)

    eta_nodes, eta_w = np.polynomial.legendre.leggauss(n_eta)
    eta = eta_nodes * (np.pi / 2.) + (np.pi / 2.)     # [-1,1] -> [0, pi]
    eta_w = eta_w * (np.pi / 2.)

    # broadcast to (n_epsilon, n_phi, n_eta)
    phi = eps_flat[:, None, None] + psi[None, :, None]
    cos_gp = np.sin(eta)[None, None, :] * np.cos(phi)
    gp = np.arccos(np.clip(cos_gp, -1., 1.))

    integrand = F(gp) * np.sin(eta)[None, None, :] ** 2
    inner = integrand @ eta_w                                          # (n_epsilon, n_phi)
    outer = inner @ (np.cos(psi) * psi_w)                               # (n_epsilon,)

    result = (A * r1 ** 2 / np.pi) * outer
    result = result.reshape(epsilon.shape)

    if runtime: print(f"reflected_lightcurve runtime: {time.perf_counter() - t0:.4f} s")

    if unit == "normalized":
        return result / (sigma * r2**2 * T2_eff**4) # Fp/Fs
    elif unit == "radiant intensity":
        return  result * lambda_band_ratio # radiant intensity (power/solid angle emitted towards the observer [W/sr]) - measured is flux [W/m^2]: result/d^2
    else: raise ValueError("unit must be either 'normalized' or 'radiant intensity'")


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
        result = 2/3*A * (r1/R)**2 * phase_function
        return result # Fp/Fs
    elif unit == "radiant intensity":
        Fp_incident = (r2 / R) ** 2 * sigma * T2_eff ** 4
        result = 2/3*A * r1**2 * Fp_incident * phase_function * lambda_band_ratio
        return result  # radiant intensity (power/solid angle emitted towards the observer [W/sr]) - measured is flux [W/m^2]: result/d^2
    else: raise ValueError("unit must be either 'normalized' or 'radiant intensity'")



###########################################################################################################################
#####
#####   Thermal Lightcurve
#####
###########################################################################################################################
