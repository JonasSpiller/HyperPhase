"""Colours, line styles and labels shared by the figures."""
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from hyperphase.bias.detectors import detectors


############################################################################################################################################
#####   Labels
############################################################################################################################################

def _fmt_esm(esm):
    """The Emission Spectroscopy Metric (Kempton et al. 2018) for a legend entry; "n/a" where the archive has none."""
    return f"{esm:.1f}" if np.isfinite(esm) else "n/a"


############################################################################################################################################
#####   Colours and style
############################################################################################################################################

# F is ordered, so a one-hue (blue) ramp; the models keep fixed colours
F_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
C_REDIST, C_HI, C_NAIVE = "#2a78d6", "#eb6834", "#52514e"   # joint A_B + F fit (blue), hyperillumination truth (orange), A_B-only fit (grey)
INK, INK_2, AXIS, MUTED = "#0b0b0b", "#52514e", "#c3c2b7", "#898781"
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
TARGET = dict(marker="*", ls="none", color="#0cff0c", ms=18, mec="k", zorder=6)   # the target planet (green star)
MOCK_KW = dict(fmt="o", ms=3, color="0.25", ecolor="0.6", elinewidth=.8, capsize=2, zorder=5)   # errorbar style of mock data

# The fitted A_B = 0 reference (contour on the maps, line elsewhere): a thin dash-dot line, apart from the solid level contours and the dashed truth
# lines; white over dark map cells, like the level contours.
FIT_ZERO = dict(color=INK, lw=1.2, ls="-.")


def set_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "font.size": 11, "axes.titlesize": 12,
    })


def F_axis(ax, F_max):
    """Linear F axis from 0 to F_max."""
    ax.set_xlim(0., F_max)
    ax.set_xlabel("Heat redistribution $F$")


def cell_edges(x, lo, hi):
    """pcolormesh cell edges halfway between grid points, clipped to [lo, hi]."""
    return np.concatenate([[lo], 0.5 * (x[1:] + x[:-1]), [hi]])


def _tick_name(name, width=18):
    """A planet name wrapped at spaces onto lines of <= width characters, so long names do not collide as tick labels."""
    return textwrap.fill(str(name), width, break_on_hyphens=False, break_long_words=False)


# Instrument colours (the palette of Ramirez et al. 2026, arXiv:2610.22338, Fig. 4) and short tick labels of the band keys
INSTRUMENT_COLORS = ["#264653", "#2a9d8f", "#e9c46a", "#b5400f", "#6d597a", "#8ab17d", "#e76f51"]
INSTRUMENT_LABELS = {"JWST_NIRSpec.G395H_NRS1": "NIRSpec G395H\nNRS1", "JWST_NIRSpec.G395H_NRS2": "NIRSpec G395H\nNRS2", "MIRI/LRS": "MIRI LRS",
                     "MIRI/F1280W": "MIRI F1280W", "MIRI/F1500W": "MIRI F1500W", "MIRI/F1800W": "MIRI F1800W", "Spitzer/IRAC2": "Spitzer IRAC2"}


def instrument_label(instrument, oneline=False, short=False):
    """Tick label of an instrument key. A joint key ("A+B") joins the detector labels, under one head line where they share it
    ("NIRSpec G395H\\nNRS1 + NRS2").
    short=True drops that head line ("NRS1 + NRS2", for legends); oneline=True puts the label on one line."""
    labels = [INSTRUMENT_LABELS.get(d, d) for d in detectors(instrument)]
    heads = {l.split("\n")[0] for l in labels}
    if len(labels) > 1 and len(heads) == 1 and all("\n" in l for l in labels):
        label = ("" if short else heads.pop() + "\n") + " + ".join(l.split("\n", 1)[1] for l in labels)
    else:
        label = " + ".join(labels)
    return label.replace("\n", " ") if oneline else label
