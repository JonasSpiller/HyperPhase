"""Colours, line styles and labels shared by the figures."""
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from hyperphase.bias.detectors import detectors


############################################################################################################################################
#####   Labels
############################################################################################################################################

# Emission Spectroscopy Metric (Kempton et al. 2018) formatted for a legend entry; the archive column is NaN for a few planets.
def _fmt_esm(esm):
    return f"{esm:.1f}" if np.isfinite(esm) else "n/a"


############################################################################################################################################
#####   Colours and style
############################################################################################################################################

# Colours as in forward_model/05_heat_redistribution.ipynb: F is ordered -> one-hue (blue) ramp; the models keep fixed colours
F_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
C_REDIST, C_HI, C_NAIVE = "#2a78d6", "#eb6834", "#52514e"   # joint A_B + F fit (blue), hyperillumination truth (orange), A_B-only fit (grey)
INK, INK_2, AXIS, MUTED = "#0b0b0b", "#52514e", "#c3c2b7", "#898781"
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
TARGET = dict(marker="*", ls="none", color="#0cff0c", ms=18, mec="k", zorder=6)   # the notebook's planet (green star), on the maps via target_marker and in the line plots
MOCK_KW = dict(fmt="o", ms=3, color="0.25", ecolor="0.6", elinewidth=.8, capsize=2, zorder=5)   # errorbar style of mock data

# The fitted A_B = 0 reference line on every bias figure (contour on the maps, line in the sweeps and retrieval plots): beyond it the point-source fit
# needs A_B < 0, i.e. more absorbed energy than the star delivers to a point, a value a physical prior (A_B_BOUNDS, as in the dynesty fits) would cut
# off at 0. A thin dash-dot line, set apart from the solid level contours and the dashed truth lines; on the maps it switches to white over dark
# cells, like the level contours.
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


# Linear F axis, restricted to the F range that matters for the plot
def F_axis(ax, F_max):
    ax.set_xlim(0., F_max)
    ax.set_xlabel("Heat redistribution $F$")


# pcolormesh cell edges halfway between grid points, clipped to [lo, hi]
def cell_edges(x, lo, hi):
    return np.concatenate([[lo], 0.5 * (x[1:] + x[:-1]), [hi]])


# The exact value written at the end of each bar; below the bar for negative values, skipped where the value is NaN.
def _tick_name(name, width=18):
    """A planet name wrapped at spaces onto lines of <= width characters, for the tick labels of the per-planet plots (long hypothetical names collide otherwise)."""
    return textwrap.fill(str(name), width, break_on_hyphens=False, break_long_words=False)


# Instrument colours of plot_instrument_retrievals (the palette of Ramirez et al. 2026, arXiv:2610.22338, Fig. 4), and short tick labels for the
# FILTER_LAMBDA_UM keys
INSTRUMENT_COLORS = ["#264653", "#2a9d8f", "#e9c46a", "#b5400f", "#6d597a", "#8ab17d", "#e76f51"]
INSTRUMENT_LABELS = {"JWST_NIRSpec.G395H_NRS1": "NIRSpec G395H\nNRS1", "JWST_NIRSpec.G395H_NRS2": "NIRSpec G395H\nNRS2", "MIRI/LRS": "MIRI LRS",
                     "MIRI/F1280W": "MIRI F1280W", "MIRI/F1500W": "MIRI F1500W", "MIRI/F1800W": "MIRI F1800W", "Spitzer/IRAC2": "Spitzer IRAC2"}


# Tick label of an instrument key; a joint key ("A+B") joins the detector labels, on one head line where they share it ("NIRSpec G395H\nNRS1 + NRS2").
# short=True drops that shared head line ("NRS1 + NRS2", for legends); oneline=True puts the label on one line.
def instrument_label(instrument, oneline=False, short=False):
    labels = [INSTRUMENT_LABELS.get(d, d) for d in detectors(instrument)]
    heads = {l.split("\n")[0] for l in labels}
    if len(labels) > 1 and len(heads) == 1 and all("\n" in l for l in labels):
        label = ("" if short else heads.pop() + "\n") + " + ".join(l.split("\n", 1)[1] for l in labels)
    else:
        label = " + ".join(labels)
    return label.replace("\n", " ") if oneline else label
