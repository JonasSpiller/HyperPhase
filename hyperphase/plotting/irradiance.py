"""Maps of the irradiance and the surface temperature, and their animation."""
import io

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from hyperphase.defaults import DEFAULT_A_B, DEFAULT_EMISSIVITY_BOL, DEFAULT_U_BOL
from hyperphase.irradiance import _tabulate_insolation, check_redistribution, effective_insolation, surface_temperature, temperature_from_insolation


def stellar_flux_line(ax=None, y=0., **kw):
    """Dotted line at the stellar flux, labelled "Stellar Flux": y = 0 for F/F_s - 1 in ppm, 1 (or 1e6) for normalised flux; kw sets the style."""
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


def plot_insolmap(phase, planet, u_bol=DEFAULT_U_BOL, A_B=DEFAULT_A_B, emissivity_bol=DEFAULT_EMISSIVITY_BOL, min_temp=None, max_temp=None, n_grid=1000, ax=None, theme='black', show_cax=True, show_axes=False, terminator=False, show_penumbra=None, insolation_model="hyperillum", F=None, unit="flux", n_tab=3000, gp_table=None, incl=90.):
    """The planet's visible disk at orbital `phase`, coloured by surface temperature or flux. Returns ax.
    phase: 0 faces the dayside to the observer (occultation), 0.5 the nightside (transit); the light curves put the transit at t = 0 instead.
    unit: "temperature", or "flux": the re-radiated bolometric flux normalised to the substellar point (the insolation itself, except for the
    redistribution models, which need F for both units).
    incl [deg]: inclination of the view (90 = edge-on; planet.incl for the view as observed). The map is symmetric about the star-planet axis,
    so only the angle between the line of sight and the substellar direction matters, cos = sin(incl) cos(2 pi phase): quadrature looks the
    same at every inclination."""
    if unit not in ("temperature", "flux"):
        raise ValueError("unit must be either 'temperature' or 'flux'")
    check_redistribution(insolation_model, F)

    mycmap = plt.cm.inferno   # same colormap spiderman's plot_planet defaults to
    bg, tc = ("black", "#04d9ff") if theme == "black" else ("white", "black")

    # Body frame: x = substellar direction, z = spin axis (= orbit normal). The sub-observer direction lies at longitude `phase` from the
    # substellar direction and 90 - incl above the orbital plane; "up" in the image is the spin axis projected onto the sky.
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
        gp_table = _tabulate_insolation(planet, u_bol, insolation_model, n_tab=n_tab)
    gp_grid, F_grid = gp_table

    F_interp = np.interp(gamma1[disk], gp_grid, F_grid)
    mapval = np.full_like(X, np.nan)
    if unit == "temperature":
        mapval[disk] = temperature_from_insolation(F_interp, planet, insolation_model, A_B = A_B, emissivity_bol=emissivity_bol, F=F, u_bol=u_bol)
    else:   # flux, normalized to 1 at the substellar point (gp_grid[0] == 0)
        mapval[disk] = effective_insolation(F_interp, planet, insolation_model, F=F, u_bol=u_bol) / effective_insolation(F_grid[0], planet, insolation_model, F=F, u_bol=u_bol)

    if min_temp is None or max_temp is None:
        if unit == "temperature":
            t_hot = temperature_from_insolation(F_grid[0], planet, insolation_model, A_B = A_B, emissivity_bol=emissivity_bol, F=F, u_bol=u_bol)
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

    on_axis = np.isclose(sin_i, 1.) and (np.isclose(phase % 1.0, 0.) or np.isclose(phase % 1.0, 0.5))   # edge-on occultation / transit: no lines
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


def plot_quad_insolmap(planet, u_bol=DEFAULT_U_BOL, A_B=DEFAULT_A_B, emissivity_bol=DEFAULT_EMISSIVITY_BOL, min_temp=None, max_temp=None, n_grid=500,
                       theme='black', show_cax=True, show_axes=False, terminator=False, show_penumbra=None,
                       insolation_model="hyperillum", F=None, unit="flux", phase1=0., phase2=0.25, phase3=0.5, phase4=0.75, n_tab=3000, incl=90.):
    """plot_insolmap at phase1..phase4 in a 2 x 2 grid with one colour scale. Returns fig."""
    if unit not in ("temperature", "flux"):
        raise ValueError("unit must be either 'temperature' or 'flux'")
    check_redistribution(insolation_model, F)

    bg, tc = ("black", "#04d9ff") if theme == "black" else ("white", "black")

    gp_table = _tabulate_insolation(planet, u_bol, insolation_model, n_tab=n_tab)

    if min_temp is None or max_temp is None:
        if unit == "temperature":
            t_hot = temperature_from_insolation(gp_table[1][0], planet, insolation_model, A_B = A_B, emissivity_bol=emissivity_bol, F=F, u_bol=u_bol)
            if min_temp is None: min_temp = -t_hot*0.1
            if max_temp is None: max_temp = 1.1 * t_hot
        else:   # flux, normalized to 1 at the substellar point
            if min_temp is None: min_temp = -0.1
            if max_temp is None: max_temp = 1.0

    fig, axs = plt.subplots(2, 2, figsize=(8 / 0.865, 8), facecolor=bg)

    for ax, phase in zip(axs.ravel(), [phase1, phase2, phase3, phase4]):
        plot_insolmap(phase, planet, u_bol=u_bol, A_B = A_B, emissivity_bol=emissivity_bol, min_temp=min_temp, max_temp=max_temp,
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


def plot_model_insolmaps(planets, phase=0.25, models=("hyperillum", "naive"), u_bol=DEFAULT_U_BOL, A_B=DEFAULT_A_B, emissivity_bol=DEFAULT_EMISSIVITY_BOL, F=None, n_grid=400, theme="black", terminator=True, show_penumbra="both", title=None, incl=90.):
    """Surface temperature maps at one phase: one column per planet, one row per insolation model.
    Each column has its own scale, 0 to 1.1 x the point-source substellar temperature (the extended source is ~0.5 % hotter there, from the
    finite planet radius; the fit with redistribution cooler), so the panels of one planet are comparable. The 90 deg terminator of the point
    source in grey; the penumbra borders gamma_f and gamma_t (the extended-source terminator) in green (hyperillum panels only).
    planets: {name: PlanetParams} or a list.
    F goes to "naive_redist" only. Returns fig, axs (len(models) x len(planets))."""
    planets = list(planets.values()) if isinstance(planets, dict) else list(planets)
    bg, tc = ("black", "#04d9ff") if theme == "black" else ("white", "black")
    fig, axs = plt.subplots(len(models), len(planets), figsize=(4.3 * len(planets), 4.3 * len(models)), facecolor=bg, squeeze=False)
    for j, pl in enumerate(planets):
        T_hot = float(np.atleast_1d(surface_temperature(np.array([0.]), pl, "naive", A_B=A_B, emissivity_bol=emissivity_bol))[0])
        for i, model in enumerate(models):
            plot_insolmap(phase, pl, u_bol=u_bol, A_B=A_B, emissivity_bol=emissivity_bol, min_temp=0., max_temp=1.1 * T_hot, n_grid=n_grid, ax=axs[i, j], theme=theme, show_cax=(j == len(planets) - 1),
                          terminator=terminator, show_penumbra=show_penumbra if model == "hyperillum" else None, insolation_model=model, F=F if model == "naive_redist" else None, unit="temperature", incl=incl)
            axs[i, j].set_title((f"{pl.name}\n" if i == 0 else "") + rf"{model}, $a/R_\star$ = {pl.R / pl.r2:.2f}, $T_{{\rm eff}}$ = {pl.T2_eff:.0f} K", color=tc, fontsize=9)
    fig.suptitle(title or rf"Surface temperature at phase {phase:g}  ($A_B$ = {A_B:.2f}, $u_{{\rm bol}}$ = {u_bol:g})", color=tc)
    fig.tight_layout()
    return fig, axs


def animation_insolmap(planet, A_B=DEFAULT_A_B, emissivity_bol=DEFAULT_EMISSIVITY_BOL, filename="output_map", u_bol=DEFAULT_U_BOL, min_temp=None, max_temp=None, n_grid=500, theme='black', show_cax=True,
                       show_axes=False, terminator=False, show_penumbra=None, insolation_model="hyperillum", F=None, unit="flux", n_frames=1000, fps=50, phase_start=0., phase_end=1., n_tab=3000, incl=90.):
    """plot_insolmap animated from phase_start to phase_end, written to <filename>.mp4 and <filename>.gif."""
    if unit not in ("temperature", "flux"):
        raise ValueError("unit must be either 'temperature' or 'flux'")
    check_redistribution(insolation_model, F)

    bg, tc = ("black", "#04d9ff") if theme == "black" else ("white", "black")
    bg_rgb = (0, 0, 0) if theme == "black" else (255, 255, 255)
    tc_rgb = (4, 217, 255) if theme == "black" else (0, 0, 0)   # matches "#04d9ff"

    gp_table = _tabulate_insolation(planet, u_bol, insolation_model, n_tab=n_tab)

    if unit == "temperature":
        t_hot = temperature_from_insolation(gp_table[1][0], planet, insolation_model, A_B = A_B, emissivity_bol=emissivity_bol, F=F, u_bol=u_bol)
        if min_temp is None: min_temp = -t_hot*0.1
        if max_temp is None: max_temp = 1.1 * t_hot
    else:   # flux, normalized to 1 at the substellar point
        if min_temp is None: min_temp = -0.1
        if max_temp is None: max_temp = 1.0


    # GIF palette: the full inferno gradient (every colour a frame can use) plus the line and text colours
    grad = (plt.cm.inferno(np.linspace(0., 1., 200))[:, :3] * 255).astype(np.uint8)
    extra = np.array([bg_rgb, (128, 128, 128), (0, 128, 0), tc_rgb], dtype=np.uint8)   # background, terminator grey, penumbra green, colorbar text
    seed = np.concatenate([grad, extra], axis=0).reshape(1, -1, 3)
    palette_img = Image.fromarray(seed, mode="RGB").quantize(colors=256)

    fig, ax = plt.subplots(facecolor=bg)
    phases = np.linspace(phase_start, phase_end, n_frames, endpoint=False)

    plot_insolmap(phases[0], planet, u_bol=u_bol, A_B = A_B, emissivity_bol=emissivity_bol, min_temp=min_temp, max_temp=max_temp,
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
        plot_insolmap(phase, planet, u_bol=u_bol, A_B = A_B, emissivity_bol=emissivity_bol, min_temp=min_temp, max_temp=max_temp,
                      n_grid=n_grid, ax=ax, theme=theme, show_cax=False, show_axes=show_axes,
                      terminator=terminator, show_penumbra=show_penumbra, insolation_model=insolation_model, F=F, unit=unit, gp_table=gp_table, incl=incl)
        capture_frame()

    plt.close(fig)

    mp4_writer.finish()
    pil_frames[0].save(filename+".gif", save_all=True, append_images=pil_frames[1:], duration=int(1000 / fps), loop=0, disposal=2)
