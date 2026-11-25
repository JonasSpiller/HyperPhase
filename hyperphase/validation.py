"""Alternative implementations of the penumbral irradiance, for validating irradiance.insolation_penumbra_hyperillum: direct adaptive
quadrature, a pixel mask, a small-disk approximation, brute-force integration over the stellar surface, and the method of Kang et al."""
import time

import numpy as np
from scipy.integrate import quad

from hyperphase.defaults import DEFAULT_U_BOL
from hyperphase.irradiance import alpha_, sigma


############################################################################################################################################
#####   Other penumbra implementations
############################################################################################################################################

def insolation_penumbra_hyperillum_simpleint(gamma1, planet, u_bol=DEFAULT_U_BOL, runtime=False, **quad_kw):
    """Penumbra, Eqs. (A7), (A9) and (A10), by per-point adaptive quadrature (scipy.integrate.quad): the reference implementation."""
    t0 = time.perf_counter()
    result = np.array([_insolation_penumbra_hyperillum_simpleint_scalar(g, planet, u_bol, **quad_kw) for g in gamma1])
    if runtime: print(f"insolation_penumbra_hyperillum_simpleint runtime: {time.perf_counter() - t0:.4f} s")
    return result


def _insolation_penumbra_hyperillum_simpleint_scalar(gamma1, planet, u_bol, **quad_kw):
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    alpha, rho = alpha_(gamma1, R, r1)
    I0 = sigma * T2_eff ** 4 / np.pi / (1 - u_bol / 3)

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
    J1 = Mn(1) * ca + Nn(1) * sa                          # Eq. (A9), n = 1
    Jp = I0 * (1.0 - u_bol) * J1
    if u_bol != 0.0:
        J2 = Mn(2) * ca + Nn(2) * sa                      # Eq. (A9), n = 2
        Jp += I0 * u_bol * J2
    return np.pi * Jp                                     # I = pi B_0 [(1 - u_bol) J_1 + u_bol J_2], Eq. (A9)


def _theta_kernels_simpleint(theta, alpha, rho, r2):
    """Return (cosb/rp^2, sinb/rp^2, cos theta', Delta) at polar angle theta."""
    ct = np.cos(theta)
    st = np.sin(theta)
    rp2 = rho**2 + r2**2 - 2.0 * rho * r2 * ct          # rho'^2                (Eq. A3)
    rp = np.sqrt(rp2)
    cos_beta = (rho - r2 * ct) / rp                      # cos beta              (Eq. A3)
    sin_beta = r2 * st / rp                              # sin beta
    cos_tp = (rho * ct - r2) / rp                        # cos theta'            (Eq. A3)
    # exact visible azimuthal half-width Delta(theta), Eq. (A7)
    cot_beta = (rho - r2 * ct) / (r2 * st) if st > 0 else np.inf
    g = -cot_beta / np.tan(alpha) if np.tan(alpha) != 0 else 0.0   # -cot(a) cot(b)
    Delta = np.arccos(np.clip(g, -1.0, 1.0))            # in [0, pi]
    return cos_beta / rp2, sin_beta / rp2, cos_tp, Delta


def insolation_penumbra_mask(gamma1, planet, u_bol=DEFAULT_U_BOL, nth=400, nom=800, runtime=False):
    """Penumbra by masking the visible part of the star on a (theta, omega) grid, a simple check of the integrals. Slow, and accurate only for
    large nth, nom."""
    t0 = time.perf_counter()
    result = np.array([_insolation_penumbra_mask_scalar(g, planet, u_bol, nth, nom) for g in gamma1])
    if runtime: print(f"insolation_penumbra_mask runtime: {time.perf_counter() - t0:.4f} s")
    return result


def _insolation_penumbra_mask_scalar(gamma1, planet, u_bol, nth, nom):
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    alpha, rho = alpha_(gamma1, R, r1)
    I0 = sigma * T2_eff ** 4 / np.pi / (1 - u_bol / 3)

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
    cz = np.cos(alpha) * cb + np.sin(alpha) * sb * np.cos(OM)     # cos zeta (Eq. A4)
    Iem = I0 * (1 - u_bol + u_bol * cos_tp)
    integ = np.where((cz > 0) & (cos_tp > 0),
                     Iem * cz * cos_tp * (r2**2 * np.sin(TH)) / rp2, 0.0)
    return np.trapezoid(np.trapezoid(integ, om, axis=1), th)


def insolation_penumbra_hyperillum_approx(gamma1, planet, u_bol=DEFAULT_U_BOL, runtime=False):
    """Penumbra in Kopal's small-disk approximation (lowest order in r2/rho; valid only for r2 << R, inaccurate for close-in planets), with the
    self-consistent sign x = sin(lambda) = -(rho/r2) cos(alpha) (opposite to the printed one)."""
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    t0 = time.perf_counter()
    alpha, rho = alpha_(gamma1, R, r1)
    I0 = sigma * T2_eff ** 4 / np.pi / (1 - u_bol / 3)

    x = np.clip(-(rho / r2) * np.cos(alpha), -1.0, 1.0)
    lam = np.arcsin(x)
    cl, sl, ca, sa = np.cos(lam), np.sin(lam), np.cos(alpha), np.sin(alpha)
    J1 = (r2 ** 2 / rho ** 2) * (0.5 - (lam + sl * cl) / np.pi) * ca \
         + (2 * r2 ** 3 / (3 * np.pi * rho ** 3)) * cl ** 3 * sa
    J2 = (r2 ** 2 / (3 * rho ** 2)) * (1 - sl * (1 + cl ** 2 / 2)) * ca \
         + (r2 ** 3 / (8 * rho ** 3)) * cl ** 4 * sa  # corrected: r2^3 (printed as r1^3)
    result = np.pi * I0 * ((1 - u_bol) * J1 + u_bol * J2)
    if runtime: print(f"insolation_penumbra_hyperillum_approx runtime: {time.perf_counter() - t0:.4f} s")
    return result


def insolation_hyperillum_bruteforce(gamma1, planet, u_bol=DEFAULT_U_BOL, n_theta=3000, n_phi=6000, runtime=False):
    """Irradiance [W m^-2] at any gamma1 by brute-force integration over the stellar sphere, independent of the geometry of irradiance (no alpha_,
    theta kernels or Delta). Stellar surface points Q lie on a midpoint grid about the star -> planet-centre axis; Q counts where both its
    emission cosine (Q faces the surface point P) and the incidence cosine at P (Q above P's horizon) are positive. Linear limb darkening. Slow
    (~1 s per point at the default grid, accurate to ~1e-6 of the substellar insolation): for spot checks, parallelised across points."""
    t0 = time.perf_counter()
    gamma1 = np.atleast_1d(np.asarray(gamma1, dtype=float))
    result = np.array([_insolation_hyperillum_bruteforce_scalar(g, planet, u_bol, n_theta, n_phi) for g in gamma1])
    if runtime: print(f"insolation_hyperillum_bruteforce runtime: {time.perf_counter() - t0:.4f} s")
    return result # incident flux on planet (W/m^2)


def _insolation_hyperillum_bruteforce_scalar(gamma1, planet, u_bol, n_theta, n_phi, rows_per_chunk=50):
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    # lengths in units of the stellar radius: star centre at the origin, planet centre at (a, 0, 0), substellar point towards -x
    rp, a = r1 / r2, R / r2
    n_hat = np.array([-np.cos(gamma1), np.sin(gamma1), 0.])     # surface normal at P
    P = np.array([a, 0., 0.]) + rp * n_hat                      # surface point P

    # the visible cap lies within arccos(1/rho) <= arccos(1/(a+rp)) of the star -> P axis,
    # which is tilted from the star -> planet-centre axis by at most arcsin(rp/a)
    t_max = min(np.pi, np.arccos(1. / (a + rp)) + np.arcsin(rp / a) + 1e-3)
    dt, dp = t_max / n_theta, 2. * np.pi / n_phi
    t = (np.arange(n_theta) + 0.5) * dt                         # polar angle of Q about the x axis
    p = (np.arange(n_phi) + 0.5) * dp                           # azimuth of Q about the x axis
    cos_p, sin_p = np.cos(p)[None, :], np.sin(p)[None, :]

    B0 = sigma * T2_eff ** 4 / np.pi / (1 - u_bol / 3)            # central surface brightness, fixes L = 4 pi r2^2 sigma T^4
    total = 0.
    for rows in np.array_split(np.arange(n_theta), max(1, n_theta // rows_per_chunk)):
        cos_t, sin_t = np.cos(t[rows])[:, None], np.sin(t[rows])[:, None]
        Qx, Qy, Qz = cos_t * np.ones_like(cos_p), sin_t * cos_p, sin_t * sin_p   # on the unit sphere, Q is also its outward normal
        dx, dy, dz = P[0] - Qx, P[1] - Qy, P[2] - Qz            # Q -> P
        dist2 = dx ** 2 + dy ** 2 + dz ** 2
        dist = np.sqrt(dist2)
        cos_emit = (Qx * dx + Qy * dy + Qz * dz) / dist         # emission angle at Q
        cos_inc = -(n_hat[0] * dx + n_hat[1] * dy) / dist       # incidence angle at P (n_hat has no z component)
        visible = (cos_emit > 0.) & (cos_inc > 0.)
        integrand = np.where(visible, (1. - u_bol + u_bol * cos_emit) * cos_emit * cos_inc / dist2, 0.)
        total += np.sum(integrand * sin_t)                      # dA = r2^2 sin(t) dt dphi; r2^2 cancels against dist^2 in units of r2
    return B0 * total * dt * dp


def insolation_kang(gamma1, planet, runtime=False):
    """Irradiance [W m^-2] at any gamma1 from the closed form of Kang, Nimmo & Ding 2023 (ApJL 949, L20), Appendix eqs. (A1)-(A2) and
    (A5)-(A12), as printed. Uniformly bright stellar disk only (their "isotropic approximation", no limb darkening): the exact reference for
    u_bol = 0. Their notation -> ours: d (star-planet separation) = R, a (planet radius) = r1, R_star = r2, theta_TL = gamma1, theta_d =
    terminator."""
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    t0 = time.perf_counter()
    gamma1 = np.atleast_1d(np.asarray(gamma1, dtype=float))
    d, a, R_star = R, r1, r2

    theta_f = np.arccos((a + R_star) / d)                                   # (A1) end of the fully illuminated zone
    theta_d = np.arccos((a - R_star) / d)                                   # (A2) start of complete darkness
    rho = np.sqrt(d ** 2 + a ** 2 - 2 * d * a * np.cos(gamma1))             # below (A3): distance P -> stellar centre
    alpha = np.arccos((d * np.cos(gamma1) - a) / rho)                       # (A5) zenith angle of the stellar centre
    phi = np.arcsin(R_star / rho)                                           # (A6) angular radius of the star
    t = rho * np.cos(phi)                                                   # below (A7): radius of the horizon circle
    l, s = t * np.sin(phi), t * np.sin(phi) * np.cos(alpha)                 # (A7) semi-axes of the projected disk
    beta = np.pi / 2 - alpha                                                # below (A12): elevation of the stellar centre

    J_f = np.pi * l * s / (np.pi * t ** 2)                                  # (A8) = sin^2(phi) cos(alpha)
    with np.errstate(invalid="ignore"):                                     # (A11)/(A12) are NaN outside the penumbra (masked below)
        delta1 = np.arccos(np.cos(phi) / np.cos(beta))                      # (A11)
        delta2 = np.arccos(np.tan(beta) / np.tan(phi))                      # (A12)
        J_p = l * s / (np.pi * t ** 2) * (np.pi - delta2 + np.sin(delta2) * np.cos(delta2)) \
              + t ** 2 / (np.pi * t ** 2) * (delta1 - np.sin(delta1) * np.cos(delta1))   # (A10)

    J = np.select([gamma1 < theta_f, gamma1 < theta_d], [J_f, J_p], default=0.)
    result = sigma * T2_eff ** 4 * J                                        # (A9): F = sigma T_star^4 J
    if runtime: print(f"insolation_kang runtime: {time.perf_counter() - t0:.4f} s")
    return result # incident flux on planet (W/m^2)
