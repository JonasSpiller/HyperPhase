# Paper / presentation versions of the notebook figures.
#
# Convention in every notebook cell that draws a figure:
#     PAPER_MODE = False                       # first line of the cell
#     ...plotting as usual...
#     paper_figure("short_descriptive_name", PAPER_MODE)   # before plt.show(); once per figure, with a distinct name per figure
#     plt.show()
# With PAPER_MODE = False the call is a no-op and the cell behaves exactly as before. With PAPER_MODE = True the figure is turned into
# a paper/presentation version: all font sizes are enlarged, the (sup)title is removed, the background is set to white, heavy meshes are
# rasterized, the margins are trimmed, and the figure is saved as a vector PDF <PAPER_STORAGE>/<name>.pdf. Temperature maps take their
# colour theme at call time: pass theme=paper_theme(PAPER_MODE) (white in paper mode, black otherwise) to plot_insolmap & co.
import os
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
from matplotlib.contour import ContourSet
from matplotlib.font_manager import FontProperties

PAPER_STORAGE = "paper_figures"
PAPER_SCALE = 1.6      # every font size is multiplied by this ...
PAPER_MIN_SIZE = 13.   # ... and is at least this many points


def paper_theme(paper_mode, default="black"):
    """Colour theme for plot_insolmap / plot_quad_insolmap / plot_model_insolmaps: white background in paper mode, `default` otherwise."""
    return "white" if paper_mode else default


def _points(size):
    """Font size spec (points or a name like 'medium') -> points."""
    return FontProperties(size=size).get_size_in_points()


def _grow(text, scale, min_size):
    if text is None or text.get_text() == "": return
    text.set_fontsize(max(_points(text.get_fontsize()) * scale, min_size))


def _relegend(owner, leg, scale, min_size):
    """Re-create a legend (of an Axes or a Figure) with enlarged fonts: a legend's handle lengths and paddings are fixed in font-size units when it is
    created, so only re-creating it keeps the entries properly spaced. Position, columns, title, frame and anchor are carried over."""
    if not leg.texts: return
    size = max(_points(leg.texts[0].get_fontsize()) * scale, min_size)
    title = leg.get_title().get_text()
    kw = dict(loc=leg._loc, ncols=leg._ncols, fontsize=size, frameon=leg.get_frame_on(), framealpha=leg.get_frame().get_alpha(),
              columnspacing=leg.columnspacing, handlelength=leg.handlelength, handletextpad=leg.handletextpad, borderaxespad=leg.borderaxespad, labelspacing=leg.labelspacing)
    if title: kw.update(title=title, title_fontsize=max(_points(leg.get_title().get_fontsize()) * scale, size))
    labels = [t.get_text() for t in leg.texts]
    if labels == owner.get_legend_handles_labels()[1] if isinstance(owner, mpl.axes.Axes) else labels == _fig_labels(owner):
        new = owner.legend(**kw)   # the default entries: let matplotlib collect the original artists again (keeps e.g. the error bars of an errorbar entry)
    else:
        new = owner.legend(leg.legend_handles, labels, **kw)   # explicit handles/labels: re-use the legend's proxy artists
    new._bbox_to_anchor = leg._bbox_to_anchor   # anchor (with its transform) as set by the original call, e.g. bbox_to_anchor=(1.01, 1.)
    new.set_zorder(leg.get_zorder())
    if owner is not leg.axes and leg in getattr(owner, "legends", []): owner.legends.remove(leg)   # fig.legend appends; drop the old one


def _fig_labels(fig):
    """The labels fig.legend() would collect from all axes."""
    return mpl.legend._get_legend_handles_labels(fig.axes)[1]


def _has_layout_engine(fig):
    """True for a figure with an active layout engine (constrained layout). A cell's own fig.tight_layout() leaves a PlaceHolderLayoutEngine
    (or a TightLayoutEngine) behind, which does not count: such a figure still has to be laid out again after the fonts were enlarged."""
    from matplotlib.layout_engine import PlaceHolderLayoutEngine, TightLayoutEngine
    return not isinstance(fig.get_layout_engine(), (type(None), PlaceHolderLayoutEngine, TightLayoutEngine))


def _hand_placed(fig):
    """True if an axes of the figure was placed by hand (fig.add_axes, e.g. a colour bar next to a grid of maps, or an axes_grid1 divider):
    tight_layout would then move the gridded axes underneath it."""
    return any(ax.get_subplotspec() is None for ax in fig.axes)


def _xtick_clash(ax, renderer, pad=2.):
    labels = [t for t in ax.xaxis.get_ticklabels() if t.get_visible() and t.get_text() and t.get_rotation() == 0]
    boxes = sorted((t.get_window_extent(renderer) for t in labels), key=lambda b: b.x0)
    return any(b.x0 < a.x1 + pad for a, b in zip(boxes, boxes[1:]))


def _unclash_xticks(fig, original):
    """Shrink horizontal x tick labels that run into each other after the enlargement, 1 pt at a time, but never below their original size.
    Returns True if anything changed."""
    changed = False
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, floor in original.items():
        labels = ax.xaxis.get_ticklabels()
        if not labels: continue
        size = labels[0].get_fontsize()
        while size > floor and _xtick_clash(ax, renderer):
            size = max(floor, size - 1.)
            ax.xaxis.set_tick_params(which="both", labelsize=size)
            fig.canvas.draw(); changed = True
    return changed


def _wrap_once(text):
    """The text broken at the space closest to its middle that lies outside $...$ math, or None if there is no such space."""
    spaces, in_math = [], False
    for i, ch in enumerate(text):
        if ch == "$" and (i == 0 or text[i - 1] != "\\"): in_math = not in_math
        elif ch == " " and not in_math: spaces.append(i)
    if "\n" in text or not spaces: return None
    i = min(spaces, key=lambda k: abs(k - len(text) / 2))
    return text[:i] + "\n" + text[i + 1:]


def _fit_axis_labels(fig, original):
    """An axis label (colour bar labels included) longer than its axes would be cut at the figure edge by bbox_inches="tight". Such a label is first
    broken onto two lines (at a space outside math), and if it is still too long, shrunk until it fits, never below its original size.
    Returns True if anything changed."""
    changed = False
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in _all_axes(fig):
        box = ax.get_window_extent(renderer)
        for lab, room in ((ax.xaxis.label, box.width), (ax.yaxis.label, box.height)):
            if not lab.get_visible() or not lab.get_text() or lab not in original or room <= 0: continue
            length = lambda: (lambda e: e.height if lab is ax.yaxis.label else e.width)(lab.get_window_extent(renderer))
            if length() <= room: continue
            wrapped = _wrap_once(lab.get_text())
            if wrapped is not None: lab.set_text(wrapped); changed = True
            if length() > room:
                lab.set_fontsize(max(original[lab], lab.get_fontsize() * room / length() * 0.97)); changed = True
    return changed


def _all_axes(fig):
    """fig.axes plus the child axes matplotlib keeps outside it (secondary_xaxis / secondary_yaxis, inset axes)."""
    out = []
    for ax in fig.axes:
        out.append(ax); out.extend(a for a in ax.child_axes if a not in out)
    return out


def _is_colorbar(ax):
    return ax.get_label() == "<colorbar>"


def enlarge_fonts(fig, scale=PAPER_SCALE, min_size=PAPER_MIN_SIZE):
    """Multiply every font size in the figure (titles, axis labels, tick labels, legends, annotations, colour bars) by `scale`, with a floor of `min_size` points.
    Works on the finished figure, so explicit fontsize= arguments and rcParams-driven sizes are treated alike. Returns ({axes: original x tick label size}, {axis label: original size})."""
    original_xtick, original_labels = {}, {}
    for ax in _all_axes(fig):
        _grow(ax.title, scale, min_size)
        for lab in (ax.xaxis.label, ax.yaxis.label):
            original_labels[lab] = _points(lab.get_fontsize()); _grow(lab, scale, min_size)
        for axis, key in ((ax.xaxis, "xtick.labelsize"), (ax.yaxis, "ytick.labelsize")):
            labels = axis.get_ticklabels()
            size = _points(labels[0].get_fontsize() if labels else mpl.rcParams[key])
            if axis is ax.xaxis: original_xtick[ax] = size
            axis.set_tick_params(which="both", labelsize=max(size * scale, min_size))
            _grow(axis.get_offset_text(), scale, min_size)
        for t in ax.texts: _grow(t, scale, min_size)
        if ax.get_legend() is not None: _relegend(ax, ax.get_legend(), scale, min_size)
    for leg in list(fig.legends): _relegend(fig, leg, scale, min_size)
    for t in fig.texts: _grow(t, scale, min_size)
    if fig._suptitle is not None: _grow(fig._suptitle, scale, min_size)
    return original_xtick, original_labels


def paper_figure(name, paper_mode=True, fig=None, titles="auto", tight="auto", scale=PAPER_SCALE, min_size=PAPER_MIN_SIZE, dpi=300, pad=0.02, storage=None):
    """Paper/presentation version of a figure, saved as a vector PDF. A no-op (returns the figure) when paper_mode is False.

    name     : file name without extension, saved to <storage>/<name>.pdf (default plot_utils.PAPER_STORAGE, read at call time; the folder is created if needed)
    fig      : the Figure (or one of its Axes); default plt.gcf(). Call it BEFORE plt.show(): the inline backend closes figures on show.
    titles   : "auto" removes the suptitle and, for single-panel figures, the axes title; multi-panel figures keep their panel titles (they usually label the panels).
               "remove" drops every title, "keep" only the suptitle.
    tight    : run fig.tight_layout(). "auto" (default) does so unless the figure has a layout engine (constrained_layout) or an axes placed by hand
               (fig.add_axes, axes_grid1). Pass False for figures laid out with fig.subplots_adjust (not detectable), True to force it.
    Horizontal x tick labels that collide after the enlargement, are shrunk again, never below their original size. Axis labels longer than their axes are broken onto two lines, and shrunk if still too long.
    The figure background is set to white, QuadMesh / filled-contour artists are rasterized at `dpi` (an imshow map is an embedded image anyway),
    and the PDF is written with bbox_inches="tight" and `pad` inches of margin.
    """
    if fig is None: fig = plt.gcf()
    if isinstance(fig, mpl.axes.Axes): fig = fig.figure
    if not paper_mode: return fig
    if titles not in ("auto", "remove", "keep"): raise ValueError(f"titles must be 'auto', 'remove' or 'keep', got {titles!r}")

    data_axes = [ax for ax in fig.axes if not _is_colorbar(ax)]
    if fig._suptitle is not None: fig._suptitle.set_visible(False)
    if titles == "remove" or (titles == "auto" and len(data_axes) <= 1):
        for ax in fig.axes: ax.set_title("")

    original_xtick, original_labels = enlarge_fonts(fig, scale=scale, min_size=min_size)

    fig.patch.set_facecolor("white")
    for art in fig.findobj(QuadMesh): art.set_rasterized(True)
    for art in fig.findobj(ContourSet):
        if getattr(art, "filled", False): art.set_rasterized(True)

    if tight == "auto": tight = not _has_layout_engine(fig) and not _hand_placed(fig)
    def _tight():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.tight_layout(pad=0.3)
    if tight: _tight()
    changed = _unclash_xticks(fig, original_xtick)
    changed = _fit_axis_labels(fig, original_labels) or changed
    if changed and tight: _tight()

    storage = storage or PAPER_STORAGE
    os.makedirs(storage, exist_ok=True)
    path = os.path.join(storage, f"{name}.pdf")
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=pad, dpi=dpi, facecolor="white")
    print(f"paper figure saved: {path}")
    return fig
