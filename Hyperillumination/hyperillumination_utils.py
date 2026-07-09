import time
import numpy as np
from scipy.integrate import quad

sigma = 5.670374e-8  # W ⋅ m⁻² ⋅ K⁻⁴

def terminator_nightside(r1, r2, R, unit="rad"):
    if unit=="rad": return np.arccos((r1-r2)/R)
    elif unit=="deg": return np.degrees(np.arccos((r1-r2)/R))

def terminator_penumbra(r1, r2, R, unit="rad"):
    if unit=="rad": return np.arccos((r1 + r2) / R)
    elif unit=="deg": return np.degrees(np.arccos((r1 + r2) / R))

def which_zone(gamma1, r1, r2, R):
    return np.select(
        [gamma1 < terminator_penumbra(r1, r2, R), gamma1 < terminator_nightside(r1, r2, R)],
        ["fully illuminated", "penumbra"],
        default="night-side")

def alpha_(gamma1, R, r1):
    # returns alpha, rho
    rho = np.sqrt(R ** 2 + r1 ** 2 - 2 * r1 * R * np.cos(gamma1))
    cos_alpha = (R * np.cos(gamma1) - r1) / rho  # corrected eq. (23): r1, not r2
    return np.arccos(np.clip(cos_alpha, -1, 1)), rho

def fraction_illuminated(r1, r2, R, extra=None):
    fraction_fullyilluminated = (1. - (r1 + r2) / R) / 2.
    fraction_penumbra = r2 / R

    if extra is not None:
        return {"fully illuminated": fraction_fullyilluminated, "penumbra": fraction_penumbra}
    return fraction_fullyilluminated + fraction_penumbra

def BBtemp_from_insolation(insol):
    return (insol/sigma)**0.25

# Insolation Model without Hyperillumination
def naive_insolation(gamma1, r2, R, T2_eff):
    I0 = sigma * T2_eff**4 * (r2/R)**2
    return np.where(gamma1 < np.pi/2, I0 * np.cos(gamma1), 0.)


# computes illumination on the surface of the exoplanet accounting for hyper-illumination
# Using models from Kopal 1954
def insolation_kopal(gamma1, r1, r2, R, T2_eff, u=0):
    # gamma1 is angle from substellar point; u is star's limb-darkening coefficient
    gamma1 = np.atleast_1d(gamma1)
    zone = which_zone(gamma1, r1, r2, R)

    result = np.zeros_like(gamma1)
    result[zone == "fully illuminated"] = insolation_fullyilluminated_kopal(gamma1[zone == "fully illuminated"], r1, r2, R, T2_eff, u=u)
    result[zone == "penumbra"] = insolation_penumbra_kopal(gamma1[zone == "penumbra"], r1, r2, R, T2_eff, u=u)
    result[zone == "night-side"] = 0.

    return result


# Fully illuminated zone: Kopal implementation (eq. 33-35)
def insolation_fullyilluminated_kopal(gamma1, r1, r2, R, T2_eff, u=0):
    mu = np.cos(gamma1)
    L2 = r2**2 * sigma * T2_eff**4 # Kopal: L2 = np.pi * r2**2 * I0 * (1 - u/3) - Kopal (corrected by the pi's)
    return L2 * (R * mu - r1) / (R**2 + r1**2 - 2. * r1 * R * mu)**1.5   # eq (34) Kopal1954


# Penumbra: Kopal implementation (eq. 36-41) without small-disk approximation and with corrections (lambda inconsistency, ...), see Goodnotes derivation
# Evaluated with fixed-order Gauss-Legendre quadrature, vectorized across all gamma1 points at once (instead of looping scipy.integrate.quad per point)
# 1e4 time improvement on insolation_penumbra_kopal_simpleint
def insolation_penumbra_kopal(gamma1, r1, r2, R, T2_eff, u=0.0, order=32):
    t0 = time.perf_counter()
    gamma1 = np.atleast_1d(np.asarray(gamma1, dtype=float))

    nodes, weights = np.polynomial.legendre.leggauss(order)   # on [-1, 1]

    alpha, rho = alpha_(gamma1, R, r1)
    I0 = sigma * T2_eff ** 4 / np.pi / (1 - u / 3)
    theta_max = np.arccos(r2 / rho)

    # map nodes to [0, theta_max] per point: theta = theta_max/2 * (x+1)
    theta = (theta_max[:, None] / 2.0) * (nodes[None, :] + 1.0)   # (N, order)
    jac = theta_max[:, None] / 2.0                                # (N, order)
    w = weights[None, :] * jac

    A, B, cos_tp, Delta = _theta_kernels_vec(theta, alpha, rho, r2)
    st = np.sin(theta)

    def Mn(n):
        return 2.0 * r2**2 / np.pi * np.sum(A * Delta * cos_tp**n * st * w, axis=1) # gauss-legendre integration scheme

    def Nn(n):
        return 2.0 * r2**2 / np.pi * np.sum(B * np.sin(Delta) * cos_tp**n * st * w, axis=1) # gauss-legendre integration scheme

    ca, sa = np.cos(alpha), np.sin(alpha)
    J1 = Mn(1) * ca + Nn(1) * sa                          # eq. 39, n=1
    Jp = I0 * (1.0 - u) * J1
    if u != 0.0:
        J2 = Mn(2) * ca + Nn(2) * sa                      # eq. 39, n=2
        Jp += I0 * u * J2
    result = np.pi * Jp                                   # F = pi * I'   (eq. 36)

    print(f"insolation_penumbra_kopal runtime: {time.perf_counter() - t0:.4f} s")
    return result
def _theta_kernels_vec(theta, alpha, rho, r2):
    """Return (cosb/rp^2, sinb/rp^2, cos theta', Delta) at polar angle theta -- vectorized
    over (point, quadrature-node) grids: theta has shape (N, order), alpha/rho shape (N,)."""
    rho = rho[:, None]
    alpha = alpha[:, None]
    ct = np.cos(theta)
    st = np.sin(theta)
    rp2 = rho**2 + r2**2 - 2.0 * rho * r2 * ct          # rho'^2                (eq. 20)
    rp = np.sqrt(rp2)
    cos_beta = (rho - r2 * ct) / rp                      # cos beta              (eq. 18)
    sin_beta = r2 * st / rp                              # sin beta
    cos_tp = (rho * ct - r2) / rp                        # cos theta'            (eq. 19)
    # exact visible azimuthal half-width Delta(theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        cot_beta = np.where(st > 0, (rho - r2 * ct) / (r2 * st), np.inf)
        g = np.where(np.tan(alpha) != 0, -cot_beta / np.tan(alpha), 0.0)   # -cot(a) cot(b)
    Delta = np.arccos(np.clip(g, -1.0, 1.0))            # in [0, pi]
    return cos_beta / rp2, sin_beta / rp2, cos_tp, Delta






###########################################################################################################################
#####
#####   Other Penumbra Implementations
#####
###########################################################################################################################


# Penumbra: Kopal implementation (eq. 36-41) without small-disk approximation and with corrections (lambda inconsistency, ...), see Goodnotes derivation
# Penumbra: baseline/ground-truth implementation via scipy.integrate.quad, per-point adaptive quadrature
def insolation_penumbra_kopal_simpleint(gamma1, r1, r2, R, T2_eff, u=0.0, **quad_kw):
    t0 = time.perf_counter()
    result = np.array([_insolation_penumbra_kopal_simpleint_scalar(g, r1, r2, R, T2_eff, u, **quad_kw) for g in gamma1])
    print(f"insolation_penumbra_kopal_simpleint runtime: {time.perf_counter() - t0:.4f} s")
    return result
def _insolation_penumbra_kopal_simpleint_scalar(gamma1, r1, r2, R, T2_eff, u, **quad_kw):
    alpha, rho = alpha_(gamma1, R, r1)
    I0 = sigma * T2_eff ** 4 / np.pi / (1 - u / 3)

    # integrands for M_n, N_n (n = 1, 2)
    def integ_M(theta, n):
        A, _, cos_tp, Delta = _theta_kernels_simpleint(theta, alpha, rho, r2)
        return A * Delta * cos_tp**n * np.sin(theta)

    def integ_N(theta, n):
        _, B, cos_tp, Delta = _theta_kernels_simpleint(theta, alpha, rho, r2)
        return B * np.sin(Delta) * cos_tp**n * np.sin(theta)

    kw = dict(epsabs=1e-12, epsrel=1e-10, limit=200)
    kw.update(quad_kw)

    Mn = lambda n: 2.0 * r2**2 / np.pi * quad(integ_M, 0.0, np.arccos(r2 / rho), args=(n,), **kw)[0]
    Nn = lambda n: 2.0 * r2**2 / np.pi * quad(integ_N, 0.0, np.arccos(r2 / rho), args=(n,), **kw)[0]

    ca, sa = np.cos(alpha), np.sin(alpha)
    J1 = Mn(1) * ca + Nn(1) * sa                          # eq. 39, n=1
    Jp = I0 * (1.0 - u) * J1
    if u != 0.0:
        J2 = Mn(2) * ca + Nn(2) * sa                      # eq. 39, n=2
        Jp += I0 * u * J2
    return np.pi * Jp                                     # F = pi * I'   (eq. 36)
def _theta_kernels_simpleint(theta, alpha, rho, r2):
    """Return (cosb/rp^2, sinb/rp^2, cos theta', Delta) at polar angle theta."""
    ct = np.cos(theta)
    st = np.sin(theta)
    rp2 = rho**2 + r2**2 - 2.0 * rho * r2 * ct          # rho'^2                (eq. 20)
    rp = np.sqrt(rp2)
    cos_beta = (rho - r2 * ct) / rp                      # cos beta              (eq. 18)
    sin_beta = r2 * st / rp                              # sin beta
    cos_tp = (rho * ct - r2) / rp                        # cos theta'            (eq. 19)
    # exact visible azimuthal half-width Delta(theta)
    cot_beta = (rho - r2 * ct) / (r2 * st) if st > 0 else np.inf
    g = -cot_beta / np.tan(alpha) if np.tan(alpha) != 0 else 0.0   # -cot(a) cot(b)
    Delta = np.arccos(np.clip(g, -1.0, 1.0))            # in [0, pi]
    return cos_beta / rp2, sin_beta / rp2, cos_tp, Delta



# Penumbra: Quick solution through masking for integral validation (long runtimes, only accurate for relatively high nth, nom)
def insolation_penumbra_mask(gamma1, r1, r2, R, T2_eff, u=0.0, nth=400, nom=800):
    """Brute-force exact 2-D disk integral with cos(zeta)>0 mask (ground truth).
    Accepts scalar or array gamma1 (dispatches per-point internally -- the (nth x nom)
    grid is already large at default resolution, so batching all points into one grid
    at once is not memory-practical)."""
    t0 = time.perf_counter()
    result = np.array([_insolation_penumbra_mask_scalar(g, r1, r2, R, T2_eff, u, nth, nom) for g in gamma1])
    print(f"insolation_penumbra_mask runtime: {time.perf_counter() - t0:.4f} s")
    return result
def _insolation_penumbra_mask_scalar(gamma1, r1, r2, R, T2_eff, u, nth, nom):
    alpha, rho = alpha_(gamma1, R, r1)
    I0 = sigma * T2_eff ** 4 / np.pi / (1 - u / 3)

    theta_max = np.arccos(r2 / rho)
    th = np.linspace(1e-7, theta_max, nth)
    om = np.linspace(0.0, 2 * np.pi, nom, endpoint=False)
    TH, OM = np.meshgrid(th, om, indexing="ij")
    ct = np.cos(TH)
    rp2 = rho**2 + r2**2 - 2 * rho * r2 * ct
    rp = np.sqrt(rp2)
    cb = (rho - r2 * ct) / rp
    sb = r2 * np.sin(TH) / rp
    cos_tp = (rho * ct - r2) / rp
    cz = np.cos(alpha) * cb + np.sin(alpha) * sb * np.cos(OM)     # cos zeta (eq.17)
    Iem = I0 * (1 - u + u * cos_tp)
    integ = np.where((cz > 0) & (cos_tp > 0),
                     Iem * cz * cos_tp * (r2**2 * np.sin(TH)) / rp2, 0.0)
    return np.trapz(np.trapz(integ, om, axis=1), th)


# Penumbra: Kopal's analytic penumbral approximation, lambda sign-corrected (only valid for r2<<R!, inaccurate for close-in Exoplanets)
def insolation_penumbra_kopal_approx(gamma1, r1, r2, R, T2_eff, u=0.0):
    """Small-disk analytic form from eqs. (43)-(48), using the SELF-CONSISTENT sign
    x = sin(lambda) = -(rho/r2)cos(alpha)  (opposite to printed eq. 56).
    Returns incident flux = pi * I0 * [(1-u)J1' + u J2'] with I0 chosen s.t. L2=pi r2^2."""
    t0 = time.perf_counter()
    alpha, rho = alpha_(gamma1, R, r1)
    I0 = sigma * T2_eff ** 4 / np.pi / (1 - u / 3)

    x = np.clip(-(rho / r2) * np.cos(alpha), -1.0, 1.0)
    lam = np.arcsin(x)
    cl, sl, ca, sa = np.cos(lam), np.sin(lam), np.cos(alpha), np.sin(alpha)
    J1 = (r2 ** 2 / rho ** 2) * (0.5 - (lam + sl * cl) / np.pi) * ca \
         + (2 * r2 ** 3 / (3 * np.pi * rho ** 3)) * cl ** 3 * sa
    J2 = (r2 ** 2 / (3 * rho ** 2)) * (1 - sl * (1 + cl ** 2 / 2)) * ca \
         + (r2 ** 3 / (8 * rho ** 3)) * cl ** 4 * sa  # corrected: r2^3 (eq.48 prints r1^3)
    result = np.pi * I0 * ((1 - u) * J1 + u * J2)
    print(f"insolation_penumbra_kopal_approx runtime: {time.perf_counter() - t0:.4f} s")
    return result
