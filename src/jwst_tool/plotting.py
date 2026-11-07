"""Pure figure builders + the process-wide render lock.

Method and contract: docs/gui_and_sharing.md, section `plotting`.
"""

from __future__ import annotations

import itertools
import threading
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter, NullLocator

from jwst_tool.model_defs import T_WINDOW

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Reentrant so a build_* call nested inside a caller's `with render_lock:`
# block does not deadlock.
render_lock = threading.RLock()

# Tick-label crowding: a decade-per-tick log axis overruns a small square
# panel, so cap the number of labelled decades and let the locator thin to
# every 2nd/3rd/... decade. Minor labels are suppressed outright.
MAX_LOG_TICKS = 7

# Linear axes (temperature): 4-digit labels on a ~3 in box collide at the
# matplotlib default. Cap the count and let the locator pick round values.
MAX_LIN_TICKS = 5


def display_smooth(wl_um: np.ndarray, y_ppm: np.ndarray, r_bin: int) -> np.ndarray:
    """Convolve a native model curve to a constant DISPLAY resolving power.

    At the model's own resolving power (R = 1000 on the correlated-k band
    grid) the unresolved line forest renders as one-sample spikes, so the PLOT
    is convolved to >= 3x the analysis R (floor 300) with the SAME tested LSF
    operator the science path uses (flat weight). That operator no-ops when the
    model grid cannot resolve the kernel, so past an analysis R of about 140
    this is already the native curve. No score touches it -- the native model
    is what the "Native model (CSV)" download exports -- and the caller's array
    is never modified.
    """
    from . import binning

    wl_um = np.asarray(wl_um, float)
    r = float(max(300, 3 * int(r_bin)))
    return binning.smooth_to_native_r(
        wl_um, y_ppm, np.array([wl_um[0], wl_um[-1]]), np.array([r, r]), float(wl_um[0]), float(wl_um[-1])
    )


def _thin_log_axis(ax: Axes, which: str) -> None:
    """Label at most ``MAX_LOG_TICKS`` decades on a log axis; no minor ticks.

    ``LogLocator(numticks=)`` picks the decade stride itself, so the labels
    stay on round decades at any span instead of being decimated by position.

    Minor ticks are removed entirely rather than merely unlabelled: over a
    10-14 decade span the 8 subdecade marks per decade merge into a solid
    black band along the spine (the vendored style draws ticks on all four
    sides), which reads as a rendering artifact. The grid already carries the
    decade structure.
    """
    axis = ax.xaxis if which == "x" else ax.yaxis
    axis.set_major_locator(LogLocator(base=10.0, numticks=MAX_LOG_TICKS))
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())


# Run caveats on the canvas: an exported figure leaves its page behind, so the
# st.warning above it and the CSV `# caveat:` header do not travel with a
# pasted PNG. One small footnote line at the foot of the canvas carries the
# SHORT form of each caveat with the image; the full text stays on the page
# and in the CSV headers.
CAVEAT_FONTSIZE = 6.5
# Character budget for the footnote, SHARED equally by the caveats: a long
# unmodeled-species list must not push a second caveat off the line, and the
# whole line has to stay inside the narrowest canvas (11 in: ~0.047 in per
# character at CAVEAT_FONTSIZE).
CAVEAT_MAX_CHARS = 180
# The footnote gets its OWN strip below the axes: the bottom margin of a
# builder's canvas is already full of tick labels and an axis label (the
# summary figure's x label sits 0.004 in above where a footnote at the canvas
# edge would end), so the strip is ADDED to the figure height and every axes
# is put back at the inch it occupied. The panels are therefore identical
# with and without caveats -- only the canvas grows.
CAVEAT_STRIP_IN = 0.20


def caveat_line(caveats: Sequence[str]) -> str:
    """Return the footnote text.

    Each caveat cut at its first sentence or first colon, whichever comes first, then to its share of
    CAVEAT_MAX_CHARS, and the pieces joined into one line.
    """
    caveats = list(caveats)
    budget = CAVEAT_MAX_CHARS // max(1, len(caveats))
    parts = []
    for c in caveats:
        s = " ".join(str(c).split())
        ends = [i for i in (s.find(". "), s.find(": ")) if i > 0]
        s = s[: min(ends)] if ends else s.rstrip(".")
        parts.append(s if len(s) <= budget else s[: budget - 3].rstrip() + "...")
    return "Model caveats: " + "; ".join(parts) + "."


def draw_caveats(fig: Figure, caveats: Sequence[str]) -> None:
    """Footnote ``fig`` with the run's caveats; no-op when there are none.

    Grows the canvas by ``CAVEAT_STRIP_IN`` and re-seats every axes at the
    inch it already occupied, so the panels are unchanged and the footnote
    has clear space of its own.
    """
    caveats = [c for c in (caveats or []) if str(c).strip()]
    if not caveats:
        return
    with render_lock:
        w, h = fig.get_size_inches()
        h2 = h + CAVEAT_STRIP_IN
        fig.set_size_inches(w, h2, forward=False)
        for ax in fig.axes:
            # original=True: with set_box_aspect the ACTIVE position is the
            # box matplotlib squeezed into the cell, and writing that back
            # would make the next draw squeeze it a second time.
            p = ax.get_position(original=True)
            ax.set_position((p.x0, (p.y0 * h + CAVEAT_STRIP_IN) / h2, p.width, p.height * h / h2))
        fig.text(
            0.5, 0.5 * CAVEAT_STRIP_IN / h2, caveat_line(caveats), ha="center", va="center", fontsize=CAVEAT_FONTSIZE
        )


TP_XLIM_DEFAULT = (1.0, 3000.0)
VMR_XLIM_DEFAULT = (1e-12, 1.0)


# Raster resolution for every PNG this app hands out or displays. Matplotlib
# takes the dpi from the SAVE, not from the figure, so the figure-construction
# value only matters for interactive backends -- gui_common.fig_bytes is what
# actually sets the resolution of the bytes a user gets.
FIG_DPI = 300
# Structure figure: ONE canvas, two square panels side by side plus the
# mixing-ratio legend strip on the right. Both panels get the SAME allocated
# rect and set_box_aspect(1.0), so matplotlib squares them identically.
STRUCT_FIG_W_IN = 11.0
STRUCT_FIG_H_IN = 4.8
STRUCT_AXES_RECT = dict(left=0.078, right=0.863, bottom=0.13, top=0.97, wspace=0.20)
# Profiles read at a glance rather than as hairlines.
STRUCT_LW = 2.6
# T-P profile in black; mixing ratios in Paul Tol's colour-blind-safe
# "vibrant" scheme, the SAME colour per species as the research note's
# model-structure figure (jwst_note/scripts/fig_model_structure.py), with
# Tol "muted" colours for any further RT molecule. All distinct from black.
TP_COLOR = "black"
VMR_COLORS = {
    "H2O": "#0077BB",
    "CO": "#33BBEE",
    "CO2": "#009988",
    "CH4": "#EE7733",
    "H2S": "#BBBBBB",
    "SO2": "#CC3311",
    "NH3": "#EE3377",
}
VMR_EXTRA_COLORS = ("#332288", "#117733", "#882255", "#AA4499", "#999933", "#44AA99", "#DDCC77", "#661100")


def build_structure_figure(
    p_bar: np.ndarray, T_K: np.ndarray, columns: list[tuple[str, np.ndarray]], caveats: Sequence[str] = ()
) -> Figure:
    """T-P profile and mixing-ratio profiles as ONE two-panel figure.

    Left: pressure (log, inverted) vs temperature. Right: VMR (log) vs the same pressure axis, through ``sharey``.
    ``columns`` is ``[(molecule, vmr_array), ...]`` sorted by peak abundance, so the legend reads top-down. ``caveats``
    are footnoted in the bottom margin, so a download keeps them.
    """
    p = np.asarray(p_bar, dtype=float)
    T = np.asarray(T_K, dtype=float)
    if p.ndim != 1 or p.shape != T.shape or p.size == 0:
        raise ValueError(
            f"build_structure_figure: p_bar {p.shape} and T_K {T.shape} must be matching non-empty 1-D arrays"
        )
    cols = [(m, np.asarray(y, dtype=float)) for m, y in columns]
    if not cols:
        raise ValueError("build_structure_figure: columns is empty")
    for m, ya in cols:  # validated before any figure exists
        if ya.shape != p.shape:
            raise ValueError(
                f"build_structure_figure: {m} column {ya.shape} does not match the pressure grid {p.shape}"
            )
    with render_lock:
        fig, (ax_t, ax_v) = plt.subplots(1, 2, figsize=(STRUCT_FIG_W_IN, STRUCT_FIG_H_IN), dpi=FIG_DPI, sharey=True)
        fig.subplots_adjust(**STRUCT_AXES_RECT)
        for ax in (ax_t, ax_v):
            ax.set_box_aspect(1.0)

        ax_t.plot(T, p, color=TP_COLOR, lw=STRUCT_LW)
        # the chemistry grid's validated temperature span
        for tlim in T_WINDOW:
            ax_t.axvline(tlim, color="#cccccc", lw=0.8, ls=":")
        ax_t.set_xlim(*TP_XLIM_DEFAULT)
        # 4-digit temperature labels collide on a small box at the default
        # locator; cap the count instead of rotating them
        ax_t.xaxis.set_major_locator(MaxNLocator(nbins=MAX_LIN_TICKS, steps=[1, 2, 5, 10]))
        ax_t.set_xlabel("temperature (K)")
        ax_t.set_ylabel("pressure (bar)")

        extra = itertools.cycle(VMR_EXTRA_COLORS)
        for m, ya in cols:
            ax_v.plot(
                np.clip(ya, 1e-14, None), p, lw=STRUCT_LW, label=str(m), color=VMR_COLORS.get(str(m)) or next(extra)
            )
        ax_v.set_xscale("log")
        ax_v.set_xlim(*VMR_XLIM_DEFAULT)
        _thin_log_axis(ax_v, "x")
        ax_v.set_xlabel("volume mixing ratio")

        # One pressure axis, inverted once. Tick labels on both panels (sharey hides the right panel's), the "pressure
        # (bar)" label once, at the far left (maintainer decision).
        ax_t.set_yscale("log")
        ax_t.invert_yaxis()
        _thin_log_axis(ax_t, "y")
        ax_v.tick_params(axis="y", labelleft=True)
        for ax in (ax_t, ax_v):
            ax.grid(alpha=0.25)
        # Legend in the strip STRUCT_AXES_RECT leaves free to the RIGHT, so
        # it can never cover a profile and needs no y-limit padding. One
        # column preserves the caller's peak-abundance order.
        ax_v.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=False,
            fontsize=7,
            ncol=1,
            handletextpad=0.5,
            borderaxespad=0.0,
            labelspacing=0.35,
        )
        draw_caveats(fig, caveats)
        return fig
