"""The figure style for every committed figure in this repo (validation/figures
and validation/parity/figs). Not used by the app, which keeps its own GUI style.

    import sys; sys.path.insert(0, "<repo>/validation")
    import figstyle as fs; fs.use()

Serif science.mplstyle, Okabe-Ito cycle, square panels, axis labels + legend
only. save() embeds the generating script in the PNG (tEXt chunks) so
tests/unit/test_validation_figures.py can prove each committed figure was made
by the committed code.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
DATA = HERE / "data"

# Okabe-Ito / Wong 2011, yellow dropped, ordered for lines on white.
CYC = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
INK = "#2b2b2b"          # data points; the reference curve of a two-code comparison
RED = "#CC3311"          # the model curve of a two-code comparison
MS = 3.5                 # marker size for data points
DASH_KW = dict(ls="--")  # matplotlib's own dash, never a custom pattern
DOT_KW = dict(ls=":")


def use():
    plt.style.use(str(HERE / "science.mplstyle"))


def pastel(c, keep=0.6):
    """Blend toward white; 0.6 is the reference-curve underlay, 0.3 a band."""
    return tuple(1 - keep + keep * v for v in to_rgb(c))


def square():
    return plt.subplots(figsize=(6.5, 6.5))


def panels(nrows=1, ncols=3):
    """Grid of square panels: 1x3 -> (16, 5.2), 2x2 -> (10.7, 10.4)."""
    return plt.subplots(nrows, ncols, figsize=(5.33 * ncols, 5.2 * nrows),
                        constrained_layout=True)


def overlay(ax, x, ref, mod, c, labels, lws=(2.5, 1.2)):
    """One of several coincident pairs: tint of c under, c dashed on top."""
    ax.plot(x, ref, color=pastel(c), lw=lws[0], solid_capstyle="round", label=labels[0])
    ax.plot(x, mod, color=c, lw=lws[1], label=labels[1], **DASH_KW)


def overlay2(ax, x, ref, mod, labels, colors=(INK, RED), lws=(2.0, 1.2)):
    """The one coincident pair of a panel: black solid reference, red dashed model."""
    ax.plot(x, ref, color=colors[0], lw=lws[0], solid_capstyle="round", label=labels[0])
    ax.plot(x, mod, color=colors[1], lw=lws[1], label=labels[1], **DASH_KW)


def lollipop(ax, names, a, b, labels, colors=(INK, RED), fontsize=8):
    """Two values per category on a log x axis: guide line, circles for a, squares for b."""
    y = np.arange(len(names))
    ax.hlines(y, 0, np.maximum(a, b), color=pastel(INK, 0.3), lw=0.8, zorder=1)
    ax.plot(a, y + 0.15, "o", ms=MS, color=colors[0], ls="", label=labels[0], zorder=3)
    ax.plot(b, y - 0.15, "s", ms=MS, color=colors[1], ls="", label=labels[1], zorder=3)
    ax.set_xscale("log")
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_yticks(y, labels=names, fontsize=fontsize)
    ax.tick_params(axis="y", which="minor", left=False, right=False)


def legend(ax, **kw):
    """Legend in the corner covering the fewest plotted points, one column then
    two; matplotlib's 'best' if every corner sits on data. Never widens the axes."""
    fig = ax.figure

    def covered(loc, **k2):
        leg = ax.legend(loc=loc, **kw, **k2)
        fig.canvas.draw()
        x0, y0, x1, y1 = leg.get_window_extent().extents
        n = 0
        for line in ax.lines:
            xy = ax.transData.transform(np.column_stack([line.get_xdata(), line.get_ydata()]))
            n += int(np.sum((xy[:, 0] > x0) & (xy[:, 0] < x1) & (xy[:, 1] > y0) & (xy[:, 1] < y1)))
        return n

    corners = ("upper right", "upper left", "lower right", "lower left")
    for ncol in (1, 2):
        scores = {loc: covered(loc, ncol=ncol) for loc in corners}
        loc = min(scores, key=scores.get)
        if scores[loc] == 0:
            return ax.legend(loc=loc, ncol=ncol, **kw)
    return ax.legend(loc="best", **kw)


def save(fig, name, out_dir=FIGURES, script=None):
    """Write out_dir/name at 300 dpi with the generating script embedded.

    tEXt chunks: Source = the script's file name, Generator-Source = its full
    text, Creation Time. Read back with embedded_source()."""
    src = Path(script or sys._getframe(1).f_code.co_filename).resolve()
    meta = {"Title": name, "Source": src.name, "Generator-Source": src.read_text(),
            "Creation Time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    path = Path(out_dir) / name
    fig.savefig(path, dpi=300, metadata=meta)
    print("wrote", path)
    return path


def embedded_source(png_path):
    """The tEXt chunks save() wrote, as a dict (empty for a PNG without them)."""
    from PIL import Image
    with Image.open(png_path) as im:
        return dict(im.text)
