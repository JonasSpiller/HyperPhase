import io
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from scipy.integrate import quad
from PIL import Image

sigma = 5.670374e-8  # W ⋅ m⁻² ⋅ K⁻⁴

def terminator_nightside(r1, r2, R, unit="rad"): # x2
    if unit=="rad": return np.arccos((r1-r2)/R)
    elif unit=="deg": return np.degrees(np.arccos((r1-r2)/R))

def terminator_penumbra(r1, r2, R, unit="rad"): # x1
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
def insolation_penumbra_kopal(gamma1, r1, r2, R, T2_eff, u=0.0, order=32, runtime=False):
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

    if runtime: print(f"insolation_penumbra_kopal runtime: {time.perf_counter() - t0:.4f} s")

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
def insolation_penumbra_kopal_simpleint(gamma1, r1, r2, R, T2_eff, u=0.0, runtime=False, **quad_kw):
    t0 = time.perf_counter()
    result = np.array([_insolation_penumbra_kopal_simpleint_scalar(g, r1, r2, R, T2_eff, u, **quad_kw) for g in gamma1])
    if runtime: print(f"insolation_penumbra_kopal_simpleint runtime: {time.perf_counter() - t0:.4f} s")
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
def insolation_penumbra_mask(gamma1, r1, r2, R, T2_eff, u=0.0, nth=400, nom=800, runtime=False):
    t0 = time.perf_counter()
    result = np.array([_insolation_penumbra_mask_scalar(g, r1, r2, R, T2_eff, u, nth, nom) for g in gamma1])
    if runtime: print(f"insolation_penumbra_mask runtime: {time.perf_counter() - t0:.4f} s")
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
    return np.trapezoid(np.trapezoid(integ, om, axis=1), th)


# Penumbra: Kopal's analytic penumbral approximation, lambda sign-corrected (only valid for r2<<R!, inaccurate for close-in Exoplanets)
def insolation_penumbra_kopal_approx(gamma1, r1, r2, R, T2_eff, u=0.0, runtime=False):
    """Small-disk analytic form from eqs. (43)-(48), using the self consistent sign
    x = sin(lambda) = -(rho/r2)cos(alpha)  (opposite to printed eq. 56)."""
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
    if runtime: print(f"insolation_penumbra_kopal_approx runtime: {time.perf_counter() - t0:.4f} s")
    return result



###########################################################################################################################
#####
#####   Plotting
#####
###########################################################################################################################


def plot_tempmap(phase, r1, r2, R, T2_eff, u=0, min_temp=None, max_temp=None, n_grid=1000, ax=None, theme='black', show_cax=True, show_axes=False, terminator=False, insolation_model="kopal"):

    mycmap = plt.cm.inferno   # same colormap spiderman's plot_planet defaults to
    bg, tc = ("black", "#04d9ff") if theme == "black" else ("white", "black")

    # equatorial (edge-on) view: sub-observer direction lies in the orbital plane, at
    # longitude `phase` from the substellar direction (1,0,0); "up" in the image is the spin axis
    z_obs = np.array([np.cos(2*np.pi*phase), -np.sin(2*np.pi*phase), 0.])   # direction toward the observer
    y_obs = np.array([0., 0., 1.])                          # spin axis -- already unit and perp to z_obs
    x_obs = np.cross(y_obs, z_obs)

    xg = np.linspace(-1., 1., n_grid)
    X, Y = np.meshgrid(xg, xg)
    disk = X**2 + Y**2 <= 1.0
    Zdepth = np.where(disk, np.sqrt(np.clip(1.0 - X**2 - Y**2, 0., None)), 0.)

    nx = X * x_obs[0] + Y * y_obs[0] + Zdepth * z_obs[0]   # body-frame x-component (substellar direction)
    gamma1 = np.where(disk, np.arccos(np.clip(nx, -1., 1.)), np.nan)

    temp = np.full_like(X, np.nan)
    if insolation_model=="kopal": temp[disk] = BBtemp_from_insolation(insolation_kopal(gamma1[disk], r1, r2, R, T2_eff, u=u))
    elif insolation_model == "naive": temp[disk] = BBtemp_from_insolation(naive_insolation(gamma1[disk], r2, R, T2_eff))
    else: raise ValueError("Insolation model must be either 'kopal' or 'naive'")

    if min_temp is None: min_temp = np.nanmin(temp) * 0.9
    if max_temp is None: max_temp = np.nanmax(temp) * 1.1

    if ax is None: _, ax = plt.subplots(facecolor=bg)
    im = ax.imshow(temp, extent=[-1, 1, -1, 1], origin="lower", cmap=mycmap, vmin=min_temp, vmax=max_temp,
                    interpolation="bilinear")
    im.set_clip_path(patches.Circle((0, 0), 1.0, transform=ax.transData))

    # add terminator line
    if terminator and not np.isclose(phase % 1.0, 0.) and not np.isclose(phase % 1.0, 0.5):
        theta_t = np.linspace(0., 2. * np.pi, 720)
        n_t = np.stack([np.zeros_like(theta_t), np.cos(theta_t), np.sin(theta_t)])
        Xt = n_t[0] * x_obs[0] + n_t[1] * x_obs[1] + n_t[2] * x_obs[2]
        Yt = n_t[0] * y_obs[0] + n_t[1] * y_obs[1] + n_t[2] * y_obs[2]
        visible_t = (n_t[0] * z_obs[0] + n_t[1] * z_obs[1] + n_t[2] * z_obs[2]) >= 0.15
        ax.plot(np.where(visible_t, Xt, np.nan), np.where(visible_t, Yt, np.nan),
                color="grey", linewidth=1.5, zorder=5)

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
        cbar.set_label("T (K)", color=tc)

    return ax


def plot_quad_tempmap(r1, r2, R, T2_eff, u=0, min_temp=None, max_temp=None, n_grid=500,
                     theme='black', show_cax=True, show_axes=False, terminator=False,
                     insolation_model="kopal", phase1=0., phase2=0.25, phase3=0.5, phase4=0.75):
    bg, tc = ("black", "#04d9ff") if theme == "black" else ("white", "black")

    if min_temp is None or max_temp is None:
        # the substellar point (gamma1=0) is always this model's global hottest point, and
        # the night side (temp=0) is always its coldest -- reuse these directly instead of
        # re-rendering all four maps just to find their combined range
        if insolation_model == "kopal":
            t_hot = BBtemp_from_insolation(insolation_kopal(np.array([0.]), r1, r2, R, T2_eff, u=u))[0]
        elif insolation_model == "naive":
            t_hot = BBtemp_from_insolation(naive_insolation(np.array([0.]), r2, R, T2_eff))[0]
        else:
            raise ValueError("Insolation model must be either 'kopal' or 'naive'")
        if min_temp is None: min_temp = 0.
        if max_temp is None: max_temp = 1.1 * t_hot

    fig, axs = plt.subplots(2, 2, figsize=(8 / 0.865, 8), facecolor=bg)

    for ax, phase in zip(axs.ravel(), [phase1, phase2, phase3, phase4]):
        plot_tempmap(phase, r1, r2, R, T2_eff, u=u, min_temp=min_temp, max_temp=max_temp,
                           n_grid=n_grid, ax=ax, theme=theme, show_cax=False, show_axes=show_axes,
                           terminator=terminator, insolation_model=insolation_model)

    if show_cax:
        im = axs[0, 0].images[0]
        fig.subplots_adjust(wspace=0, hspace=0, right=0.80)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.tick_params(colors=tc)
        cbar.set_label("T (K)", color=tc)
    else:
        fig.subplots_adjust(wspace=0, hspace=0)

    return fig



def animation_tempmap(r1, r2, R, T2_eff, filename="output_tempmap", u=0, min_temp=None, max_temp=None, n_grid=500, theme='black', show_cax=True,
                      show_axes=False, terminator=False, insolation_model="kopal", n_frames=1000, fps=20, phase_start=0., phase_end=1.):

    bg, tc = ("black", "#04d9ff") if theme == "black" else ("white", "black")
    bg_rgb = (0, 0, 0) if theme == "black" else (255, 255, 255)
    tc_rgb = (4, 217, 255) if theme == "black" else (0, 0, 0)   # matches "#04d9ff"

    if insolation_model == "kopal": t_hot = BBtemp_from_insolation(insolation_kopal(np.array([0.]), r1, r2, R, T2_eff, u=u))[0]
    elif insolation_model == "naive": t_hot = BBtemp_from_insolation(naive_insolation(np.array([0.]), r2, R, T2_eff))[0]
    else: raise ValueError("insolation_model must be either 'kopal' or 'naive'")
    if min_temp is None: min_temp = 0.
    if max_temp is None: max_temp = 1.1 * t_hot


    # seed palette: the full inferno gradient (every color any frame's heatmap could possibly use) plus the background, terminator-line, and colorbar-text colors
    grad = (plt.cm.inferno(np.linspace(0., 1., 200))[:, :3] * 255).astype(np.uint8)
    extra = np.array([bg_rgb, (128, 128, 128), tc_rgb], dtype=np.uint8)   # background, terminator grey, colorbar text
    seed = np.concatenate([grad, extra], axis=0).reshape(1, -1, 3)
    palette_img = Image.fromarray(seed, mode="RGB").quantize(colors=256)

    fig, ax = plt.subplots(facecolor=bg)
    phases = np.linspace(phase_start, phase_end, n_frames, endpoint=False)

    plot_tempmap(phases[0], r1, r2, R, T2_eff, u=u, min_temp=min_temp, max_temp=max_temp,
                 n_grid=n_grid, ax=ax, theme=theme, show_cax=False, show_axes=show_axes,
                 terminator=terminator, insolation_model=insolation_model)
    if show_cax:
        cbar = fig.colorbar(ax.images[0], ax=ax)
        cbar.ax.tick_params(colors=tc)
        cbar.set_label("T (K)", color=tc)

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
        plot_tempmap(phase, r1, r2, R, T2_eff, u=u, min_temp=min_temp, max_temp=max_temp,
                     n_grid=n_grid, ax=ax, theme=theme, show_cax=False, show_axes=show_axes,
                     terminator=terminator, insolation_model=insolation_model)
        capture_frame()

    plt.close(fig)

    mp4_writer.finish()
    pil_frames[0].save(filename+".gif", save_all=True, append_images=pil_frames[1:], duration=int(1000 / fps), loop=0, disposal=2)
