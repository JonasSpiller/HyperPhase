import io
import time
from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from scipy.integrate import quad
from PIL import Image
from dataclasses import dataclass, replace
from astropy import constants as const
from dataclasses import fields

sigma = const.sigma_sb.value  # 5.670374e-8 W ⋅ m⁻² ⋅ K⁻⁴
h = const.h.value
c = const.c.value
kB = const.k_B.value


###########################################################################################################################
#####
#####   Planetary Parameter Class
#####
###########################################################################################################################

@dataclass(frozen=True)
class PlanetParams:
    name: str                # Exoplanet Name
    period: float       # orbital period [days]
    incl:   float       # inclination [deg]
    ecc:    float       # eccentricity
    w:      float       # argument of periastron [deg]
    r1:     float       # planet radius   [consistent units]
    r2:     float       # stellar radius  [consistent units]
    R:      float       # semi-major axis [consistent units]
    T2_eff: float       # stellar Teff [K]
    a_AU: float         # semi-major axis [AU] (for light travel time correction)
    T14: float  # Transit Duration [h]
    logg: float = 4.5  # stellar surface gravity [log10 cgs]; only the stellar SED uses it (astro_utils.stellar_spectrum)
    feh: float = 0.0  # stellar metallicity [Fe/H] [dex]; only the stellar SED uses it

    @property
    def args(self):
        return (self.period, self.incl, self.ecc, self.w,
                self.r1, self.r2, self.R, self.T2_eff, self.a_AU)

    def terminator_penumbra(self, unit="rad"):  # arccos(x1): end of the fully-illuminated zone
        gp = np.arccos((self.r1 + self.r2) / self.R)
        return np.degrees(gp) if unit == "deg" else gp

    def terminator_nightside(self, unit="rad"):  # arccos(x2): end of the penumbra / start of the night-side
        gp = np.arccos((self.r1 - self.r2) / self.R)
        return np.degrees(gp) if unit == "deg" else gp

    def fraction_illuminated(self, extra=False): # Carter 2019
        fraction_fullyilluminated = (1. - (self.r1 + self.r2) / self.R) / 2.
        fraction_penumbra = self.r2 / self.R
        if extra:
            return {"fully illuminated": fraction_fullyilluminated, "penumbra": fraction_penumbra}
        return fraction_fullyilluminated + fraction_penumbra

    # Fractional change in substellar insolation/temperature through hyperillumination (compared to naive point-source approximation)
    def substellar_boost(self, insolation=True):
        # the boost is not effected by hyperillumination at all, it is purely the stellar distance correction a-rp: the boost is (R/(R − r₁))² − 1 ≈ 2 r₁/R
        if insolation: return (insolation_fullyilluminated_hyperillum(0, self)/naive_insolation(0, self)) - 1
        else: return (insolation_fullyilluminated_hyperillum(0, self)/naive_insolation(0, self))**0.25 - 1

    # Approximate (naive) occultation depth [ppm] for a grey planet and a blackbody star: delta_occ = emissivity * (r1/r2)^2 * <B_l(T1_day)> / <B_l(T2_eff)>  +  A_g * (r1/R)^2
    # (<.> is the bandpass average weighted by stellar photon flux (B_l * l).)
    def approx_occ_depth(planet, eff_wavelength_range, A_B=None, f=2/3, emissivity_B=None, emissivity=None, A_g=0.0, n_wl=400):
        # f is Cowan & Agol (2011) redistribution parameter: 1/4 full redistribution, 1/2 uniform dayside, 2/3 no redistribution (disk-integrated), 1 substellar.
        # resolved here, not in the signature: the baseline settings are imported only after this class (see below)
        if A_B is None: A_B = DEFAULT_A_B
        if emissivity_B is None: emissivity_B = DEFAULT_EMISSIVITY_B
        if emissivity is None: emissivity = DEFAULT_EMISSIVITY

        wl_min, wl_max = eff_wavelength_range # [m]

        # Point-source value ignoring hyper-illumination.
        T1_day = planet.T2_eff * np.sqrt(planet.r2 / planet.R) * (f * (1.0 - A_B) / emissivity_B) ** 0.25

        wl = np.linspace(wl_min, wl_max, n_wl)
        B = lambda T: (2 * h * c ** 2 / wl ** 5) / np.expm1(h * c / (wl * kB * T))

        contrast = np.trapezoid(wl * B(T1_day), wl) / np.trapezoid(wl * B(planet.T2_eff), wl)
        return 1e6 * (emissivity * (planet.r1 / planet.r2) ** 2 * contrast + A_g * (planet.r1 / planet.R) ** 2)

    def summary(self, print_params=False, eff_wavelength_range=None):
        print(f"=== {self.name} ===")
        if print_params:
            for f in fields(self):
                print(f"{f.name:>8} = {getattr(self, f.name)}")
            print("--- derived ---")
        print(f"rp = r1/r2           = {self.r1 / self.r2:.4f}")
        print(f"a  = R/r2            = {self.R / self.r2:.3f}")
        print(f"terminator_penumbra  = {self.terminator_penumbra('deg'):.1f} deg")
        print(f"terminator_nightside = {self.terminator_nightside('deg'):.1f} deg")
        illuminated = self.fraction_illuminated(extra=True)
        print(f"fraction_illuminated = {100*self.fraction_illuminated():.1f} %  (fully illuminated: {100*illuminated['fully illuminated']:.1f} %, penumbra: {100*illuminated['penumbra']:.1f} %)")
        print(f"substellar_boost     = {100*self.substellar_boost():.5f} % (insolation), {100*self.substellar_boost(insolation=False):.5f} % (temperature)")
        if eff_wavelength_range is not None:
            print(f"approx_occ_depth     = {self.approx_occ_depth(eff_wavelength_range):.1f} ppm")


# Baseline settings (HItargets_utils.py). Imported here, after PlanetParams, because HItargets_utils imports PlanetParams from this module and defines the
# settings before doing so: whichever of the two modules is imported first, the other then finds what it needs already defined. PlanetParams itself therefore
# cannot use them as argument defaults and resolves them at call time.
from HItargets_utils import DEFAULT_A_B, DEFAULT_U_B, DEFAULT_EMISSIVITY_B, DEFAULT_EMISSIVITY


###########################################################################################################################
#####
#####   Utilities
#####
###########################################################################################################################

def which_zone(gamma1, planet):
    return np.select(
        [gamma1 < planet.terminator_penumbra(), gamma1 < planet.terminator_nightside()],
        ["fully illuminated", "penumbra"],
        default="night-side")

def alpha_(gamma1, R, r1):
    # returns alpha, rho
    rho = np.sqrt(R ** 2 + r1 ** 2 - 2 * r1 * R * np.cos(gamma1))
    cos_alpha = (R * np.cos(gamma1) - r1) / rho  # corrected eq. (23): r1, not r2
    return np.arccos(np.clip(cos_alpha, -1, 1)), rho

# Local radiative equilibrium emissivity_B sigma T^4 = (1-A_B) insol. A_B is always the Bond albedo (even for lambda-band measurements), emissivity_B the bolometric emissivity.
# Only (1-A_B)/emissivity_B enters, so emissivity_B is exactly degenerate with A_B: with emissivity_B fixed to 1, A_B is really the effective albedo 1 - (1-A_B)/emissivity_B.
def BBtemp_from_insolation(insol, A_B=DEFAULT_A_B, emissivity_B=DEFAULT_EMISSIVITY_B):
    return ((1-A_B) * insol / (emissivity_B * sigma))**0.25


# tabulates insolation(gamma1) on a 1D grid once, so 2D pixel grids
def _tabulate_insolation(planet, u_B, insolation_model, n_tab=3000):
    if insolation_model == "hyperillum":
        gp1 = planet.terminator_penumbra()
        gp2 = planet.terminator_nightside()
        gp_fi = np.linspace(0., gp1, n_tab)
        F_fi = insolation_fullyilluminated_hyperillum(gp_fi, planet)
        gp_pen = np.linspace(gp1, gp2, n_tab)
        F_pen = insolation_penumbra_hyperillum(gp_pen, planet, u_B=u_B)
        gp_grid = np.concatenate([gp_fi, gp_pen, [np.pi]])
        F_grid = np.concatenate([F_fi, F_pen, [0.]])
    elif insolation_model in ("naive", "naive_redist"):  # naive_redist redistributes the heat, not the incident flux
        gp_grid = np.linspace(0., np.pi, n_tab)
        F_grid = naive_insolation(gp_grid, planet)
    else:
        raise ValueError(f"insolation_model must be one of {INSOLATION_MODELS}")
    return gp_grid, F_grid



###########################################################################################################################
#####
#####   Core Insolation Models
#####
###########################################################################################################################

# "hyperillum": extended stellar disk, "naive": point source, "naive_redist": point source + Kreidberg & Loeb 2016 heat redistribution F (see Heat Redistribution below)
INSOLATION_MODELS = ("hyperillum", "naive", "naive_redist")

def insolation(gamma1, planet, insolation_model, u_B=DEFAULT_U_B):
    if insolation_model == "hyperillum": return insolation_hyperillum(gamma1, planet, u_B=u_B)
    elif insolation_model in ("naive", "naive_redist"): return naive_insolation(gamma1, planet)  # naive_redist redistributes the heat, not the incident flux
    else: raise ValueError(f"insolation_model must be one of {INSOLATION_MODELS}")

# Insolation Model without Hyperillumination
# point source at the planet centre with a hard terminator at 90°. Equivalent to SPIDERMAN Louden model
def naive_insolation(gamma1, planet):
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    Fp_incident = (r2/R)**2 * sigma * T2_eff**4 # Fp_incident = L2 / (4*np.pi*R**2), L2 = 4*np.pi * r2**2 * sigma * T2_eff**4
    return np.where(gamma1 < np.pi/2, Fp_incident * np.cos(gamma1), 0.) # incident flux on planet (W/m^2)



# computes illumination on the surface of the exoplanet accounting for hyper-illumination
# Inspired by Kopal 1954
def insolation_hyperillum(gamma1, planet, u_B=DEFAULT_U_B):
    # gamma1 is angle from substellar point; u_B is the bolometric stellar limb-darkening coefficient

    gamma1 = np.atleast_1d(gamma1)
    gp1 = planet.terminator_penumbra()
    gp2 = planet.terminator_nightside()
    fi_mask = gamma1 < gp1
    pen_mask = (gamma1 >= gp1) & (gamma1 < gp2)

    Fp_incident = np.zeros_like(gamma1)
    Fp_incident[fi_mask] = insolation_fullyilluminated_hyperillum(gamma1[fi_mask], planet)
    Fp_incident[pen_mask] = insolation_penumbra_hyperillum(gamma1[pen_mask], planet, u_B=u_B)

    return Fp_incident # incident flux on planet (W/m^2)


# Fully illuminated zone: Kopal implementation (eq. 33-35)
# fully illuminated zone is independent of limb darkening (integral term vanishes through symmetry)
def insolation_fullyilluminated_hyperillum(gamma1, planet):
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    mu = np.cos(gamma1)
    Fp_incident =  r2**2 * sigma * T2_eff**4 * (R * mu - r1) / (R**2 + r1**2 - 2. * r1 * R * mu)**1.5   # eq (34) Kopal1954
    return Fp_incident # incident flux on planet (W/m^2)


# Penumbra: inspired by Kopal (eq. 36-41) without small-disk approximation and with corrections (lambda inconsistency, ...), see Goodnotes derivation
# Evaluated with fixed-order Gauss-Legendre quadrature, vectorized across all gamma1 points at once (instead of looping scipy.integrate.quad per point)
# 1e4 time improvement on insolation_penumbra_hyperillum_simpleint
def insolation_penumbra_hyperillum(gamma1, planet, u_B=DEFAULT_U_B, order=64, runtime=False):
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    t0 = time.perf_counter()
    gamma1 = np.atleast_1d(np.asarray(gamma1, dtype=float))

    nodes, weights = _leggauss_cached(order)   # on [-1, 1]

    alpha, rho = alpha_(gamma1, R, r1)
    I0 = sigma * T2_eff ** 4 / np.pi / (1 - u_B / 3)

    theta_max = np.arccos(r2 / rho)

    # map nodes to [0, theta_max] per point: theta = theta_max/2 * (x+1)
    theta = (theta_max[:, None] / 2.0) * (nodes[None, :] + 1.0)   # (N, order)
    jac = theta_max[:, None] / 2.0                                # (N, order)
    w = weights[None, :] * jac

    A, B, cos_tp, Delta, st = _theta_kernels_vec(theta, alpha, rho, r2)

    def Mn(n):
        return 2.0 * r2**2 / np.pi * np.sum(A * Delta * cos_tp**n * st * w, axis=1) # gauss-legendre integration scheme

    def Nn(n):
        return 2.0 * r2**2 / np.pi * np.sum(B * np.sin(Delta) * cos_tp**n * st * w, axis=1) # gauss-legendre integration scheme

    ca, sa = np.cos(alpha), np.sin(alpha)
    J1 = Mn(1) * ca + Nn(1) * sa                          # eq. 39, n=1
    Jp = I0 * (1.0 - u_B) * J1
    if u_B != 0.0:
        J2 = Mn(2) * ca + Nn(2) * sa                      # eq. 39, n=2
        Jp += I0 * u_B * J2
    result = np.pi * Jp                                   # F = pi * I'   (eq. 36)

    if runtime: print(f"insolation_penumbra_hyperillum runtime: {time.perf_counter() - t0:.4f} s")

    return result  # incident flux on planet (W/m^2)
@lru_cache(maxsize=None)
def _leggauss_cached(order):
    return np.polynomial.legendre.leggauss(order)

def _theta_kernels_vec(theta, alpha, rho, r2):
    """Return (cosb/rp^2, sinb/rp^2, cos theta', Delta, sin theta) at polar angle theta --
    vectorized over (point, quadrature-node) grids: theta has shape (N, order), alpha/rho
    shape (N,)."""
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
    return cos_beta / rp2, sin_beta / rp2, cos_tp, Delta, st



###########################################################################################################################
#####
#####   Heat Redistribution (Kreidberg & Loeb 2016)
#####
###########################################################################################################################

# Kreidberg & Loeb 2016 eq. 1: sigma*T^4 = (1-A_B)*S*[F/2 + (1-2F)*cos(z)] on the dayside and (1-A_B)*S*F/2 on the nightside (z: zenith angle = gamma1, S: substellar insolation).
# In terms of the naive insolation I (= S*cos(z) on the dayside, 0 on the nightside) both read sigma*T^4 = (1-A_B)*[(1-2F)*I + F*S/2]: the nightside re-radiates the fraction F of the absorbed power.
# F=0: no redistribution (identical to "naive"), F=0.5: isothermal planet. SPIDERMAN's Louden model evaluates the same map (with T_int=0), which is what the "naive_redist" thermal phase curve uses.

# F is required by insolation_model="naive_redist" and ignored by the other models (like u_B is by "naive").
def check_redistribution(insolation_model, F):
    if insolation_model == "naive_redist" and F is None:
        raise ValueError('insolation_model="naive_redist" requires the heat redistribution parameter F in [0, 0.5]')
    if F is not None and not 0. <= F <= 0.5:
        raise ValueError(f"heat redistribution parameter F must be in [0, 0.5], got F={F}")


# Flux per unit (1-A_B) a surface element re-radiates: the insolation itself under instant re-radiation ("hyperillum", "naive"), the redistributed flux under "naive_redist".
def effective_insolation(insol, planet, insolation_model, F=None):
    check_redistribution(insolation_model, F)
    if insolation_model != "naive_redist": return insol
    S = float(naive_insolation(0., planet))  # substellar insolation
    return (1. - 2. * F) * insol + F * S / 2.


# Surface temperature from an insolation map of the given insolation_model (local radiative equilibrium, redistributed for "naive_redist").
# emissivity_B is degenerate with A_B (see BBtemp_from_insolation): with emissivity_B = 1, A_B is an effective albedo.
def temperature_from_insolation(insol, planet, insolation_model, A_B=DEFAULT_A_B, emissivity_B=DEFAULT_EMISSIVITY_B, F=None):
    return BBtemp_from_insolation(effective_insolation(insol, planet, insolation_model, F=F), A_B = A_B, emissivity_B=emissivity_B)


# T(gamma1): surface temperature at an angular distance gamma1 from the substellar point (the T(z) profile of Kreidberg & Loeb 2016 for "naive_redist")
def surface_temperature(gamma1, planet, insolation_model, u_B=DEFAULT_U_B, A_B=DEFAULT_A_B, emissivity_B=DEFAULT_EMISSIVITY_B, F=None):
    return temperature_from_insolation(insolation(gamma1, planet, insolation_model, u_B=u_B), planet, insolation_model, A_B = A_B, emissivity_B=emissivity_B, F=F)



###########################################################################################################################
#####
#####   Plotting
#####
###########################################################################################################################


# Dotted horizontal line marking the stellar flux on plots of the observed signal with the transit included (the "phase curve" views): y = 0 for F/F_s - 1 in ppm,
# y = 1 (or 1e6) for normalised flux. Draws on ax (default: current axes); kw overrides the style. Shows up in the legend as "Stellar Flux".
def stellar_flux_line(ax=None, y=0., **kw):
    ax = ax or plt.gca()
    return ax.axhline(y, **{"color": "k", "ls": ":", "lw": 1., "label": "Stellar Flux", "zorder": 1, **kw})


def _plot_iso_gamma1_line(ax, gamma1_val, x_obs, y_obs, z_obs, color, linewidth=1.5, zorder=5):
    """Draw the locus of points at constant angle `gamma1_val` from the substellar point, clipped to the observer-visible hemisphere."""
    if not np.isfinite(gamma1_val) or gamma1_val <= 0. or gamma1_val >= np.pi:
        return
    theta_t = np.linspace(0., 2. * np.pi, 720)
    nx_val, r_val = np.cos(gamma1_val), np.sin(gamma1_val)
    n_t = np.stack([np.full_like(theta_t, nx_val), r_val * np.cos(theta_t), r_val * np.sin(theta_t)])
    Xt = n_t[0] * x_obs[0] + n_t[1] * x_obs[1] + n_t[2] * x_obs[2]
    Yt = n_t[0] * y_obs[0] + n_t[1] * y_obs[1] + n_t[2] * y_obs[2]
    visible_t = (n_t[0] * z_obs[0] + n_t[1] * z_obs[1] + n_t[2] * z_obs[2]) >= 0.15
    ax.plot(np.where(visible_t, Xt, np.nan), np.where(visible_t, Yt, np.nan),
            color=color, linewidth=linewidth, zorder=zorder)


# unit="flux": re-radiated (bolometric) flux normalized to the substellar point - the insolation itself, except for "naive_redist" (needs F for both units)
# incl [deg]: orbital inclination of the view (90 = edge-on, the default; pass planet.incl for the view as observed). The map is symmetric about the star-planet axis,
# so only the angle between the line of sight and the substellar direction matters, cos = sin(incl) cos(2 pi phase): quadrature looks the same at every inclination.
def plot_insolmap(phase, planet, u_B=DEFAULT_U_B, A_B=DEFAULT_A_B, emissivity_B=DEFAULT_EMISSIVITY_B, min_temp=None, max_temp=None, n_grid=1000, ax=None, theme='black', show_cax=True, show_axes=False, terminator=False, show_penumbra=None, insolation_model="hyperillum", F=None, unit="flux", n_tab=3000, gp_table=None, incl=90.):

    if unit not in ("temperature", "flux"):
        raise ValueError("unit must be either 'temperature' or 'flux'")
    check_redistribution(insolation_model, F)

    mycmap = plt.cm.inferno   # same colormap spiderman's plot_planet defaults to
    bg, tc = ("black", "#04d9ff") if theme == "black" else ("white", "black")

    # body frame: x = substellar direction, z = spin axis (= orbit normal). The sub-observer direction lies at longitude `phase` from the substellar
    # direction and 90 - incl above the orbital plane (incl = 90: the equatorial, edge-on view); "up" in the image is the spin axis projected onto the sky
    sin_i, cos_i = np.sin(np.radians(incl)), np.cos(np.radians(incl))
    z_obs = np.array([sin_i * np.cos(2*np.pi*phase), -sin_i * np.sin(2*np.pi*phase), cos_i])    # direction toward the observer
    y_obs = np.array([-cos_i * np.cos(2*np.pi*phase), cos_i * np.sin(2*np.pi*phase), sin_i])    # projected spin axis -- unit and perp to z_obs
    x_obs = np.cross(y_obs, z_obs)

    xg = np.linspace(-1., 1., n_grid)
    X, Y = np.meshgrid(xg, xg)
    disk = X**2 + Y**2 <= 1.0
    Zdepth = np.where(disk, np.sqrt(np.clip(1.0 - X**2 - Y**2, 0., None)), 0.)

    nx = X * x_obs[0] + Y * y_obs[0] + Zdepth * z_obs[0]   # body-frame x-component (substellar direction)
    gamma1 = np.where(disk, np.arccos(np.clip(nx, -1., 1.)), np.nan)

    if gp_table is None:
        gp_table = _tabulate_insolation(planet, u_B, insolation_model, n_tab=n_tab)
    gp_grid, F_grid = gp_table

    F_interp = np.interp(gamma1[disk], gp_grid, F_grid)
    mapval = np.full_like(X, np.nan)
    if unit == "temperature":
        mapval[disk] = temperature_from_insolation(F_interp, planet, insolation_model, A_B = A_B, emissivity_B=emissivity_B, F=F)
    else:   # flux, normalized to 1 at the substellar point (gp_grid[0] == 0)
        mapval[disk] = effective_insolation(F_interp, planet, insolation_model, F=F) / effective_insolation(F_grid[0], planet, insolation_model, F=F)

    if min_temp is None or max_temp is None:
        if unit == "temperature":
            t_hot = temperature_from_insolation(F_grid[0], planet, insolation_model, A_B = A_B, emissivity_B=emissivity_B, F=F)
            if min_temp is None: min_temp = -t_hot*0.1
            if max_temp is None: max_temp = 1.1 * t_hot
        else:   # flux, normalized to 1 at the substellar point
            if min_temp is None: min_temp = -0.1
            if max_temp is None: max_temp = 1.0

    if ax is None: _, ax = plt.subplots(facecolor=bg)
    im = ax.imshow(mapval, extent=[-1, 1, -1, 1], origin="lower", cmap=mycmap, vmin=min_temp, vmax=max_temp,
                    interpolation="bilinear")
    im.set_clip_path(patches.Circle((0, 0), 1.0, transform=ax.transData))

    # add terminator line / penumbra border lines
    if show_penumbra not in (None, False, "both", "inner", "outer"):
        raise ValueError("show_penumbra must be one of None, False, 'both', 'inner', 'outer'")

    on_axis = np.isclose(sin_i, 1.) and (np.isclose(phase % 1.0, 0.) or np.isclose(phase % 1.0, 0.5))   # edge-on occultation / transit: no lines, as before
    if (terminator or show_penumbra) and not on_axis:
        if terminator:
            _plot_iso_gamma1_line(ax, np.pi / 2., x_obs, y_obs, z_obs, color="grey")
        if show_penumbra in ("both", "inner"):
            _plot_iso_gamma1_line(ax, planet.terminator_penumbra(), x_obs, y_obs, z_obs, color="green")
        if show_penumbra in ("both", "outer"):
            _plot_iso_gamma1_line(ax, planet.terminator_nightside(), x_obs, y_obs, z_obs, color="green")

    ax.set_facecolor(bg)
    ax.set(aspect=1, xlim=(1.05, -1.05), ylim=(-1.05, 1.05))
    for spine in ax.spines.values():
        spine.set_color(tc if show_axes else bg)
    ax.get_xaxis().set_visible(show_axes)
    ax.get_yaxis().set_visible(show_axes)

    if show_axes: ax.tick_params(colors=tc)

    if show_cax:
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(colors=tc)
        cbar.set_label("T (K)" if unit == "temperature" else "Relative brightness", color=tc)

    return ax


def plot_quad_insolmap(planet, u_B=DEFAULT_U_B, A_B=DEFAULT_A_B, emissivity_B=DEFAULT_EMISSIVITY_B, min_temp=None, max_temp=None, n_grid=500,
                       theme='black', show_cax=True, show_axes=False, terminator=False, show_penumbra=None,
                       insolation_model="hyperillum", F=None, unit="flux", phase1=0., phase2=0.25, phase3=0.5, phase4=0.75, n_tab=3000, incl=90.):

    if unit not in ("temperature", "flux"):
        raise ValueError("unit must be either 'temperature' or 'flux'")
    check_redistribution(insolation_model, F)

    bg, tc = ("black", "#04d9ff") if theme == "black" else ("white", "black")

    gp_table = _tabulate_insolation(planet, u_B, insolation_model, n_tab=n_tab)

    if min_temp is None or max_temp is None:
        if unit == "temperature":
            t_hot = temperature_from_insolation(gp_table[1][0], planet, insolation_model, A_B = A_B, emissivity_B=emissivity_B, F=F)
            if min_temp is None: min_temp = -t_hot*0.1
            if max_temp is None: max_temp = 1.1 * t_hot
        else:   # flux, normalized to 1 at the substellar point
            if min_temp is None: min_temp = -0.1
            if max_temp is None: max_temp = 1.0

    fig, axs = plt.subplots(2, 2, figsize=(8 / 0.865, 8), facecolor=bg)

    for ax, phase in zip(axs.ravel(), [phase1, phase2, phase3, phase4]):
        plot_insolmap(phase, planet, u_B=u_B, A_B = A_B, emissivity_B=emissivity_B, min_temp=min_temp, max_temp=max_temp,
                      n_grid=n_grid, ax=ax, theme=theme, show_cax=False, show_axes=show_axes,
                      terminator=terminator, show_penumbra=show_penumbra, insolation_model=insolation_model, F=F, unit=unit, gp_table=gp_table, incl=incl)

    if show_cax:
        im = axs[0, 0].images[0]
        fig.subplots_adjust(wspace=0, hspace=0, right=0.80)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.tick_params(colors=tc)
        cbar.set_label("T (K)" if unit == "temperature" else "Relative brightness", color=tc)
    else:
        fig.subplots_adjust(wspace=0, hspace=0)

    return fig


# Several planets side by side, one column each, one row per insolation model, all at the same phase: the surface maps the models disagree on (e.g. the Roche-limit planets of
# HypotheticalExtremePlanet.ipynb). Each column has its own temperature scale, 0 to 1.1 x its substellar temperature (the same in every model: there is no substellar boost), so the
# panels of one planet are comparable; the colour bar of the last column labels the rows. The terminator is drawn in grey, the penumbra borders in green on the hyperillum panels only.
# planets: {name: PlanetParams} or a list. F is passed to "naive_redist" only. Returns fig, axs (shape (len(models), len(planets))).
def plot_model_insolmaps(planets, phase=0.25, models=("hyperillum", "naive"), u_B=DEFAULT_U_B, A_B=DEFAULT_A_B, emissivity_B=DEFAULT_EMISSIVITY_B, F=None, n_grid=400, theme="black", terminator=True, show_penumbra="both", title=None, incl=90.):
    planets = list(planets.values()) if isinstance(planets, dict) else list(planets)
    bg, tc = ("black", "#04d9ff") if theme == "black" else ("white", "black")
    fig, axs = plt.subplots(len(models), len(planets), figsize=(4.3 * len(planets), 4.3 * len(models)), facecolor=bg, squeeze=False)
    for j, pl in enumerate(planets):
        T_hot = float(np.atleast_1d(surface_temperature(np.array([0.]), pl, "naive", A_B=A_B, emissivity_B=emissivity_B))[0])
        for i, model in enumerate(models):
            plot_insolmap(phase, pl, u_B=u_B, A_B=A_B, emissivity_B=emissivity_B, min_temp=0., max_temp=1.1 * T_hot, n_grid=n_grid, ax=axs[i, j], theme=theme, show_cax=(j == len(planets) - 1),
                          terminator=terminator, show_penumbra=show_penumbra if model == "hyperillum" else None, insolation_model=model, F=F if model == "naive_redist" else None, unit="temperature", incl=incl)
            axs[i, j].set_title((f"{pl.name}\n" if i == 0 else "") + rf"{model}, $a/R_\star$ = {pl.R / pl.r2:.2f}, $T_{{\rm eff}}$ = {pl.T2_eff:.0f} K", color=tc, fontsize=9)
    fig.suptitle(title or rf"Surface temperature at phase {phase:g}  ($A_B$ = {A_B:.2f}, $u_B$ = {u_B:g})", color=tc)
    fig.tight_layout()
    return fig, axs



def animation_insolmap(planet, A_B=DEFAULT_A_B, emissivity_B=DEFAULT_EMISSIVITY_B, filename="output_map", u_B=DEFAULT_U_B, min_temp=None, max_temp=None, n_grid=500, theme='black', show_cax=True,
                       show_axes=False, terminator=False, show_penumbra=None, insolation_model="hyperillum", F=None, unit="flux", n_frames=1000, fps=50, phase_start=0., phase_end=1., n_tab=3000, incl=90.):

    if unit not in ("temperature", "flux"):
        raise ValueError("unit must be either 'temperature' or 'flux'")
    check_redistribution(insolation_model, F)

    bg, tc = ("black", "#04d9ff") if theme == "black" else ("white", "black")
    bg_rgb = (0, 0, 0) if theme == "black" else (255, 255, 255)
    tc_rgb = (4, 217, 255) if theme == "black" else (0, 0, 0)   # matches "#04d9ff"

    gp_table = _tabulate_insolation(planet, u_B, insolation_model, n_tab=n_tab)

    if unit == "temperature":
        t_hot = temperature_from_insolation(gp_table[1][0], planet, insolation_model, A_B = A_B, emissivity_B=emissivity_B, F=F)
        if min_temp is None: min_temp = -t_hot*0.1
        if max_temp is None: max_temp = 1.1 * t_hot
    else:   # flux, normalized to 1 at the substellar point
        if min_temp is None: min_temp = -0.1
        if max_temp is None: max_temp = 1.0


    # seed palette: the full inferno gradient (every color any frame's heatmap could possibly use) plus the background, terminator-line, penumbra-border, and colorbar-text colors
    grad = (plt.cm.inferno(np.linspace(0., 1., 200))[:, :3] * 255).astype(np.uint8)
    extra = np.array([bg_rgb, (128, 128, 128), (0, 128, 0), tc_rgb], dtype=np.uint8)   # background, terminator grey, penumbra green, colorbar text
    seed = np.concatenate([grad, extra], axis=0).reshape(1, -1, 3)
    palette_img = Image.fromarray(seed, mode="RGB").quantize(colors=256)

    fig, ax = plt.subplots(facecolor=bg)
    phases = np.linspace(phase_start, phase_end, n_frames, endpoint=False)

    plot_insolmap(phases[0], planet, u_B=u_B, A_B = A_B, emissivity_B=emissivity_B, min_temp=min_temp, max_temp=max_temp,
                  n_grid=n_grid, ax=ax, theme=theme, show_cax=False, show_axes=show_axes,
                  terminator=terminator, show_penumbra=show_penumbra, insolation_model=insolation_model, F=F, unit=unit, gp_table=gp_table, incl=incl)
    if show_cax:
        cbar = fig.colorbar(ax.images[0], ax=ax)
        cbar.ax.tick_params(colors=tc)
        cbar.set_label("T (K)" if unit == "temperature" else "Relative brightness", color=tc)

    w, h = fig.canvas.get_width_height()


    mp4_writer = animation.FFMpegWriter(fps=fps)
    mp4_writer.setup(fig, filename+".mp4", dpi=fig.dpi)

    pil_frames = []

    def capture_frame():
        mp4_writer.grab_frame(facecolor=bg)
        buf = io.BytesIO()
        fig.savefig(buf, format="rgba", facecolor=bg)
        buf.seek(0)
        frame = Image.frombuffer("RGBA", (w, h), buf.getbuffer(), "raw", "RGBA", 0, 1).convert("RGB")
        pil_frames.append(frame.quantize(palette=palette_img, dither=Image.FLOYDSTEINBERG))

    capture_frame()
    for phase in phases[1:]:
        ax.clear()
        plot_insolmap(phase, planet, u_B=u_B, A_B = A_B, emissivity_B=emissivity_B, min_temp=min_temp, max_temp=max_temp,
                      n_grid=n_grid, ax=ax, theme=theme, show_cax=False, show_axes=show_axes,
                      terminator=terminator, show_penumbra=show_penumbra, insolation_model=insolation_model, F=F, unit=unit, gp_table=gp_table, incl=incl)
        capture_frame()

    plt.close(fig)

    mp4_writer.finish()
    pil_frames[0].save(filename+".gif", save_all=True, append_images=pil_frames[1:], duration=int(1000 / fps), loop=0, disposal=2)





###########################################################################################################################
#####
#####   Other Penumbra Implementations (unused, for validation)
#####
###########################################################################################################################


# Penumbra: Inspired by Kopal (eq. 36-41) without small-disk approximation and with corrections (lambda inconsistency, ...), see Goodnotes derivation
# Penumbra: baseline/ground-truth implementation via scipy.integrate.quad, per-point adaptive quadrature
def insolation_penumbra_hyperillum_simpleint(gamma1, planet, u_B=DEFAULT_U_B, runtime=False, **quad_kw):

    t0 = time.perf_counter()
    result = np.array([_insolation_penumbra_hyperillum_simpleint_scalar(g, planet, u_B, **quad_kw) for g in gamma1])
    if runtime: print(f"insolation_penumbra_hyperillum_simpleint runtime: {time.perf_counter() - t0:.4f} s")
    return result
def _insolation_penumbra_hyperillum_simpleint_scalar(gamma1, planet, u_B, **quad_kw):
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    alpha, rho = alpha_(gamma1, R, r1)
    I0 = sigma * T2_eff ** 4 / np.pi / (1 - u_B / 3)

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
    Jp = I0 * (1.0 - u_B) * J1
    if u_B != 0.0:
        J2 = Mn(2) * ca + Nn(2) * sa                      # eq. 39, n=2
        Jp += I0 * u_B * J2
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
def insolation_penumbra_mask(gamma1, planet, u_B=DEFAULT_U_B, nth=400, nom=800, runtime=False):

    t0 = time.perf_counter()
    result = np.array([_insolation_penumbra_mask_scalar(g, planet, u_B, nth, nom) for g in gamma1])
    if runtime: print(f"insolation_penumbra_mask runtime: {time.perf_counter() - t0:.4f} s")
    return result
def _insolation_penumbra_mask_scalar(gamma1, planet, u_B, nth, nom):
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    alpha, rho = alpha_(gamma1, R, r1)
    I0 = sigma * T2_eff ** 4 / np.pi / (1 - u_B / 3)

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
    Iem = I0 * (1 - u_B + u_B * cos_tp)
    integ = np.where((cz > 0) & (cos_tp > 0),
                     Iem * cz * cos_tp * (r2**2 * np.sin(TH)) / rp2, 0.0)
    return np.trapezoid(np.trapezoid(integ, om, axis=1), th)


# Penumbra: Kopal's analytic penumbral approximation, lambda sign-corrected (only valid for r2<<R!, inaccurate for close-in Exoplanets)
def insolation_penumbra_hyperillum_approx(gamma1, planet, u_B=DEFAULT_U_B, runtime=False):
    """Small-disk analytic form from eqs. (43)-(48), using the self consistent sign
    x = sin(lambda) = -(rho/r2)cos(alpha)  (opposite to printed eq. 56)."""
    period, incl, ecc, w, r1, r2, R, T2_eff, a_AU = planet.args

    t0 = time.perf_counter()
    alpha, rho = alpha_(gamma1, R, r1)
    I0 = sigma * T2_eff ** 4 / np.pi / (1 - u_B / 3)

    x = np.clip(-(rho / r2) * np.cos(alpha), -1.0, 1.0)
    lam = np.arcsin(x)
    cl, sl, ca, sa = np.cos(lam), np.sin(lam), np.cos(alpha), np.sin(alpha)
    J1 = (r2 ** 2 / rho ** 2) * (0.5 - (lam + sl * cl) / np.pi) * ca \
         + (2 * r2 ** 3 / (3 * np.pi * rho ** 3)) * cl ** 3 * sa
    J2 = (r2 ** 2 / (3 * rho ** 2)) * (1 - sl * (1 + cl ** 2 / 2)) * ca \
         + (r2 ** 3 / (8 * rho ** 3)) * cl ** 4 * sa  # corrected: r2^3 (eq.48 prints r1^3)
    result = np.pi * I0 * ((1 - u_B) * J1 + u_B * J2)
    if runtime: print(f"insolation_penumbra_hyperillum_approx runtime: {time.perf_counter() - t0:.4f} s")
    return result


# Full surface (all gamma1): brute-force integration over the stellar sphere in vector form, for validation.
# Independent of the Kopal-style geometry above (no alpha_, no theta kernels, no Delta): stellar surface points Q are sampled
# on a midpoint grid about the star -> planet-centre axis, and the visibility of each Q is decided by the signs of the
# emission cosine (Q faces P) and the incidence cosine (Q is above the horizon of P). Includes linear limb darkening.
# Slow (~1 s per point at the default grid, accurate to ~1e-6 of the substellar insolation): use for spot checks, parallelise across points.
def insolation_hyperillum_bruteforce(gamma1, planet, u_B=DEFAULT_U_B, n_theta=3000, n_phi=6000, runtime=False):

    t0 = time.perf_counter()
    gamma1 = np.atleast_1d(np.asarray(gamma1, dtype=float))
    result = np.array([_insolation_hyperillum_bruteforce_scalar(g, planet, u_B, n_theta, n_phi) for g in gamma1])
    if runtime: print(f"insolation_hyperillum_bruteforce runtime: {time.perf_counter() - t0:.4f} s")
    return result # incident flux on planet (W/m^2)
def _insolation_hyperillum_bruteforce_scalar(gamma1, planet, u_B, n_theta, n_phi, rows_per_chunk=50):
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

    B0 = sigma * T2_eff ** 4 / np.pi / (1 - u_B / 3)            # central surface brightness, fixes L = 4 pi r2^2 sigma T^4
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
        integrand = np.where(visible, (1. - u_B + u_B * cos_emit) * cos_emit * cos_inc / dist2, 0.)
        total += np.sum(integrand * sin_t)                      # dA = r2^2 sin(t) dt dphi; r2^2 cancels against dist^2 in units of r2
    return B0 * total * dt * dp


# Full surface (all gamma1): closed form of Kang, Nimmo & Ding 2023 (ApJL 949, L20), Appendix eqs. (A1)-(A2) and (A5)-(A12), as printed.
# Uniformly bright stellar disk only ("isotropic approximation", no limb darkening): exact reference for u_B = 0.
# Kang -> our notation: d (star-planet separation) = R, a (planet radius) = r1, R_star = r2, theta_TL = gamma1, theta_d = terminator.
def insolation_kang(gamma1, planet, runtime=False):
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
