# ---------------------------------------------------------------------------------------------------------------------
# The integrand of instellca_E is transcribed from InstellCa_3.0.py (InstellCa, Exoplanet Instellation Calculator,
# https://github.com/Mradumay137/InstellCa, commit 99225f4), Copyright (c) 2021 Mradumay137 (Mradumay Sadh), MIT License;
# reference: Sadh & Gavassino 2026, ApJ 997, 167. The MIT permission notice is reproduced in InstellCa_3.0_exported.py.
# ---------------------------------------------------------------------------------------------------------------------
"""
Independent re-implementation of the InstellCa_3.0 irradiance integral, written for this repository and used by
instellca_comparison.ipynb to locate the origin of InstellCa's offset from the exact irradiance. Nothing here is
imported from InstellCa; `instellca_E` is a
line-for-line transcription of the integration block of InstellCa_3.0.py (same integrand incl. its abs(), same
rectangular limits, same scipy.integrate.quad calls, same P/(4 pi^2) normalisation, same 1.0573 Teff factor), and
the notebook checks it against the output of the actual script.

Geometry (as in InstellCa): star centre at the origin, planet centre at (al, 0, 0), surface point
P = (al - rp cos la, 0, rp sin la) with la the angle from the substellar point; stellar surface element at
rs (cos th cos x, cos th sin x, sin th) with th = stellar latitude, x = stellar longitude from the sub-planet meridian.
The irradiance at P is  E = int I(mu_s) mu_s cos_i dA / |d|^2  over the elements that face P (mu_s > 0) AND lie
above P's horizon (cos_i > 0).  dA = rs^2 cos th dth dx.
"""
import math
import numpy as np
from scipy.integrate import quad

SIGMA_INSTELLCA = 5.67e-8   # hard-coded in InstellCa_3.0.py


class Geo:
    """Planet geometry in InstellCa's variables (metres, Kelvin). Only a/R* and rp/R* matter for the geometry."""
    def __init__(self, a_over_rs, rp_over_rs, teff, rs=6.955e8):
        self.rs, self.al, self.rp, self.fa1 = rs, a_over_rs * rs, rp_over_rs * rs, float(teff)

    @property
    def symp(self):          # InstellCa's "critical point of symmetry" = end of the fully illuminated zone
        return math.acos((self.rs + self.rp) / self.al)

    @property
    def terminator(self):    # end of the penumbra
        return math.acos((self.rp - self.rs) / self.al)


# ---------------------------------------------------------------------------------------------------------------------
# 1. Transcription of InstellCa_3.0.py (one latitude point). u = 0.6 is the code's default branch, u = 0 the other one.
# ---------------------------------------------------------------------------------------------------------------------
def instellca_E(la, g, u=0.6, quad_kw=None):
    """InstellCa's `value2` [W/m^2] at angle la [rad] from the substellar point (la > 0)."""
    quad_kw = quad_kw or {}
    rs, rp, al, fa1 = g.rs, g.rp, g.al, g.fa1
    symp = math.acos((rs + rp) / al)
    fa = fa1 * 1.0573 if u == 0.6 else fa1                       # InstellCa: "if para==1 and u==0.6: fa=fa1*1.0573"
    P = SIGMA_INSTELLCA * fa ** 4 * 4 * np.pi * rs ** 2
    y3 = math.acos(rs / math.sqrt(al ** 2 + rp ** 2 - 2 * al * rp * math.cos(la)))

    def function(x, th, la):                                     # verbatim structure of InstellCa's integrand
        a = -rs * math.sin(th) * math.cos(th) * math.sin(np.pi - la) + al * math.cos(np.pi - la) * math.cos(th) + rp * math.cos(th)
        b = rs * math.cos(np.pi - la) * (math.cos(th)) ** 2
        c = rs ** 2 + al ** 2 + rp ** 2 - 2 * rs * rp * math.sin(th) * math.sin(la) + 2 * al * rp * math.cos(np.pi - la)
        e = -2 * (al + rp * math.cos(np.pi - la)) * rs * math.cos(th)
        mu = abs(-rs + (al + rp * math.cos(np.pi - la)) * math.cos(th) * math.cos(x) + rp * math.sin(th) * math.sin(la)) / (c + e * math.cos(x)) ** (1 / 2)
        lf = 1 - u * (1 - mu)
        return abs(a - b * math.cos(x)) * lf * mu / (c + e * math.cos(x)) ** (3 / 2)

    def integration(th, la):                                     # inner integral over stellar longitude
        return quad(function, -y3, y3, args=(th, la), **quad_kw)[0]

    ll, ul, _ = instellca_box_limits(la, g)                      # outer limits in stellar latitude (InstellCa's branches)
    value = quad(lambda th: integration(th, la), ll, ul, **quad_kw)[0]
    return value * P / (4 * np.pi * np.pi)


def instellca_box_limits(la, g):
    """InstellCa's rectangle for la > 0: th in [ll, ul], x in [-y3, y3]. Fully illuminated zone: ll, ul = latitudes of the
    lower/upper limb on the central meridian (eps -/+ alpha); penumbra: ll = horizon crossing on the central meridian."""
    rs, rp, al = g.rs, g.rp, g.al
    symp = math.acos((rs + rp) / al)
    rho2 = al ** 2 + rp ** 2 - 2 * al * rp * math.cos(la)
    y3 = math.acos(rs / math.sqrt(rho2))
    sq = np.sqrt(rho2 - rs ** 2)
    ul = math.acos((rs * (al - rp * math.cos(la)) - rp * math.sin(la) * sq) / rho2)
    if la < symp:
        ll = -math.acos((rs * (al - rp * math.cos(la)) + rp * math.sin(la) * sq) / rho2)
    else:
        ll = -la + math.acos((al * math.cos(la) - rp) / rs)
    return ll, ul, y3


# ---------------------------------------------------------------------------------------------------------------------
# 2. The same physics in vector form, with the SIGNED visibility factors and the TRUE integration domain
# ---------------------------------------------------------------------------------------------------------------------
def intensity_scale(g, u):
    """I0 of InstellCa's star: B = P/(4 pi^2 rs^2) with its 1.0573 Teff factor for u = 0.6."""
    fa = g.fa1 * 1.0573 if u == 0.6 else g.fa1
    return SIGMA_INSTELLCA * fa ** 4 / math.pi


def signed_kernel(x, th, la, g):
    """(mu_s, cos_i, 1/|d|^2, cos th) for stellar element (th, x) and the surface point at angle la; mu_s, cos_i SIGNED."""
    rs, rp, al = g.rs, g.rp, g.al
    cth, sth, cx, sx = np.cos(th), np.sin(th), np.cos(x), np.sin(x)
    Px, Pz = al - rp * math.cos(la), rp * math.sin(la)
    dx, dy, dz = Px - rs * cth * cx, -rs * cth * sx, Pz - rs * sth          # d = P - S
    d2 = dx * dx + dy * dy + dz * dz
    dn = np.sqrt(d2)
    mu_s = (cth * cx * dx + cth * sx * dy + sth * dz) / dn                   # element faces P if > 0
    cos_i = -(-math.cos(la) * dx + math.sin(la) * dz) / dn                   # element above P's horizon if > 0
    return mu_s, cos_i, 1.0 / d2, cth


def exact_xlimits(th, la, g):
    """Admissible stellar-longitude half-interval [x_lo, x_hi] (x >= 0, mirrored for x < 0) at latitude th:
    the visible cap (mu_s > 0) intersected with the horizon condition (cos_i > 0). None if empty."""
    rs, rp, al = g.rs, g.rp, g.al
    cth, sth = math.cos(th), math.sin(th)
    if cth <= 0:
        return None
    c_lo, c_hi = -1.0, 1.0
    c_lo = max(c_lo, (rs - rp * math.sin(la) * sth) / ((al - rp * math.cos(la)) * cth))   # cap: s_hat . P > rs
    q = rs * math.sin(la) * sth - rp + al * math.cos(la)                                 # horizon: rs cos la cos th cos x < q
    cl = math.cos(la)
    if abs(cl) < 1e-15:
        if q <= 0:
            return None
    elif cl > 0:
        c_hi = min(c_hi, q / (rs * cl * cth))
    else:
        c_lo = max(c_lo, q / (rs * cl * cth))
    if c_lo >= c_hi:
        return None
    return math.acos(c_hi), math.acos(c_lo)


def exact_E_quad(la, g, u=0.6, order=96, epsrel=1e-11):
    """Exact irradiance for InstellCa's star: adaptive quad in latitude, Gauss-Legendre in longitude over the exact
    admissible interval (smooth integrand). Independent of hyperphase.irradiance."""
    I0 = intensity_scale(g, u)
    rs, rp, al = g.rs, g.rp, g.al
    rho = math.sqrt(al ** 2 + rp ** 2 - 2 * al * rp * math.cos(la))
    alpha, eps = math.acos(rs / rho), math.asin(rp * math.sin(la) / rho)
    xg, wg = np.polynomial.legendre.leggauss(order)

    def inner(th):
        lim = exact_xlimits(th, la, g)
        if lim is None or lim[1] - lim[0] < 1e-14:
            return 0.0
        x = 0.5 * (lim[1] - lim[0]) * xg + 0.5 * (lim[1] + lim[0])
        mu, ci, inv_d2, cth = signed_kernel(x, th, la, g)
        f = (1 - u * (1 - np.clip(mu, 0, None))) * np.clip(mu, 0, None) * np.clip(ci, 0, None) * inv_d2 * cth
        return (lim[1] - lim[0]) * np.sum(wg * f)                 # factor 2 (x < 0 mirror) x 0.5 (interval map)

    pts = []
    arg = (al * math.cos(la) - rp) / rs
    if abs(arg) <= 1 and eps - alpha < -la + math.acos(arg) < eps + alpha:
        pts.append(-la + math.acos(arg))                          # horizon crossing on the central meridian: kink
    val = quad(inner, eps - alpha, eps + alpha, points=pts or None, limit=500, epsabs=0, epsrel=epsrel)[0]
    return I0 * rs ** 2 * val


def analytic_FZ(la, g, u=0.6):
    """L cos(alpha) / (4 pi rho^2) for InstellCa's star (luminosity P (1 - u/3)), with alpha and rho the zenith angle and distance of the
    stellar centre: exact in the fully illuminated zone."""
    rs, rp, al = g.rs, g.rp, g.al
    fa = g.fa1 * 1.0573 if u == 0.6 else g.fa1
    L = SIGMA_INSTELLCA * fa ** 4 * 4 * math.pi * rs ** 2 * (1 - u / 3)
    rho2 = al ** 2 + rp ** 2 - 2 * al * rp * math.cos(la)
    return L * (al * math.cos(la) - rp) / math.sqrt(rho2) / (4 * math.pi * rho2)


# ---------------------------------------------------------------------------------------------------------------------
# 3. Dissection: InstellCa's integrand (vector form, signed_kernel) and nested quad, with the domain / abs() switched one at a time
# ---------------------------------------------------------------------------------------------------------------------
def instellca_variant(la, g, u=0.6, domain="box", use_abs=True, quad_kw=None):
    """domain = 'box'        : InstellCa's rectangle (with abs() as in InstellCa if use_abs, else signed factors)
                'box_masked' : the rectangle, but elements outside the true domain contribute zero
                'exact'      : the true domain (visible cap cut by the horizon)"""
    quad_kw = quad_kw or dict(epsabs=0, epsrel=1e-10, limit=200)
    I0 = intensity_scale(g, u)
    ll, ul, y3 = instellca_box_limits(la, g)

    def f_point(x, th):
        mu, ci, inv_d2, cth = signed_kernel(np.array(x), np.array(th), la, g)
        mu, ci = float(mu), float(ci)
        if domain in ("box_masked", "exact"):
            if mu <= 0 or ci <= 0:
                return 0.0
        elif use_abs:
            mu, ci = abs(mu), abs(ci)
        return (1 - u * (1 - mu)) * mu * ci * float(inv_d2) * float(cth)

    if domain in ("box", "box_masked"):
        val = quad(lambda th: quad(f_point, -y3, y3, args=(th,), **quad_kw)[0], ll, ul, **quad_kw)[0]
    else:
        rs, rp, al = g.rs, g.rp, g.al
        rho = math.sqrt(al ** 2 + rp ** 2 - 2 * al * rp * math.cos(la))
        alpha, eps = math.acos(rs / rho), math.asin(rp * math.sin(la) / rho)

        def inner(th):
            lim = exact_xlimits(th, la, g)
            return 0.0 if lim is None else 2 * quad(f_point, lim[0], lim[1], args=(th,), **quad_kw)[0]
        val = quad(inner, eps - alpha, eps + alpha, **quad_kw)[0]
    return I0 * g.rs ** 2 * val


def domain_masks(la, g, n=600, pad_deg=8.):
    """Grid in (stellar longitude x, stellar latitude th) around InstellCa's rectangle with boolean masks:
    'true' (visible cap and above horizon), 'box' (InstellCa's rectangle), 'cap' (mu_s > 0), 'above' (cos_i > 0)."""
    ll, ul, y3 = instellca_box_limits(la, g)
    pad = math.radians(pad_deg)
    x = np.linspace(-y3 - pad, y3 + pad, n)
    th = np.linspace(ll - pad, ul + pad, n)
    X, TH = np.meshgrid(x, th)
    mu, ci, _, _ = signed_kernel(X, TH, la, g)
    box = (np.abs(X) <= y3) & (TH >= ll) & (TH <= ul)
    return dict(x=x, th=th, cap=mu > 0, above=ci > 0, true=(mu > 0) & (ci > 0), box=box, ll=ll, ul=ul, y3=y3)
