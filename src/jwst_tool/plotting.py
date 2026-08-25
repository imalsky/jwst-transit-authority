"""Pure figure builders + the process-wide render lock.

Importable WITHOUT streamlit: every figure the GUI shows is built here from
plain arrays, so the same code the app renders is the code the tests render.

WHY THE LOCK (the ParseException crash on the Space)
----------------------------------------------------
Matplotlib's mathtext parser is process-global state. ``Figure.draw()``
holds matplotlib's own ``Figure._render_lock``, but ``Figure.tight_layout()``
runs the layout engine OUTSIDE it, and layout measures every tick label --
mathtext on a log axis. Two Streamlit sessions (each its own script-runner
thread) laying out figures at once can therefore be inside the shared parser
concurrently; the loser raises ``ValueError: ParseException`` out of
``tight_layout``. VERIFIED on the deployed pin with a threaded stress test
(the unlocked path raises in 7-8 of 8 threads, the locked builders in 0;
measurements: notes.md); bare ``Figure`` objects do NOT help -- the race is
in the shared parser, not pyplot's figure registry.

RULE: every figure lifecycle -- construct, lay out, draw, export, close --
happens inside ``render_lock``. Do not lay out, save, or draw a figure
outside it, and do not narrow the span to just the export: the layout call is
the one that crashed. ``build_*`` acquire it themselves (it is reentrant), so
a caller that wraps a build + export in one ``with render_lock:`` block is
correct and is the intended pattern. The lock is process-wide and
in-process; it says nothing about other processes (the ETC/forward
subprocesses render nothing).

Pinned by tests/unit/test_plotting.py (threaded stress + a structural check
that no unlocked layout/export call survives in app.py).
"""
from __future__ import annotations

import itertools
import threading

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (LogLocator, MaxNLocator, NullFormatter,
                               NullLocator)

# Reentrant so a build_* call nested inside a caller's `with render_lock:`
# block does not deadlock.
render_lock = threading.RLock()

# Tick-label crowding: a decade-per-tick log axis overruns a small square
# panel, so cap the number of labelled decades and let the locator thin to
# every 2nd/3rd/... decade. Minor labels are suppressed outright.
MAX_LOG_TICKS = 7

# Linear axes (temperature): 4-digit labels on a ~3 in box collide at the
# matplotlib default, which is what "the temperature ticks overlap a ton"
# was. Cap the count and let the locator pick round values.
MAX_LIN_TICKS = 5


def _thin_log_axis(ax, which: str) -> None:
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


TP_XLIM_DEFAULT = (1.0, 3000.0)
VMR_XLIM_DEFAULT = (1e-12, 1.0)


# Structure figure: ONE canvas, two square panels side by side plus the
# mixing-ratio legend strip on the right. Both panels get the SAME allocated
# rect and set_box_aspect(1.0), so matplotlib squares them identically -- the
# two can never drift to different sizes the way two separate figures did.
STRUCT_FIG_DPI = 200
STRUCT_FIG_W_IN = 11.0
STRUCT_FIG_H_IN = 4.8
STRUCT_AXES_RECT = dict(left=0.055, right=0.863, bottom=0.13, top=0.97,
                        wspace=0.20)
# Profiles read at a glance rather than as hairlines.
STRUCT_LW = 2.6
# T-P profile in black; mixing ratios in Paul Tol's colour-blind-safe
# "vibrant" scheme, the SAME colour per species as the research note's
# model-structure figure (jwst_note/scripts/fig_model_structure.py), with
# Tol "muted" colours for any further RT molecule. All distinct from black.
TP_COLOR = "black"
VMR_COLORS = {"H2O": "#0077BB", "CO": "#33BBEE", "CO2": "#009988",
              "CH4": "#EE7733", "H2S": "#BBBBBB", "SO2": "#CC3311",
              "NH3": "#EE3377"}
VMR_EXTRA_COLORS = ("#332288", "#117733", "#882255", "#AA4499", "#999933",
                    "#44AA99", "#DDCC77", "#661100")


def build_structure_figure(p_bar, T_K, columns):
    """T-P profile and mixing-ratio profiles as ONE two-panel figure.

    Left: pressure (log, inverted) vs temperature. Right: VMR (log) vs the
    SAME pressure axis, shared through ``sharey`` so the two panels cannot
    disagree about the vertical scale. ``columns`` is the ordered
    ``[(molecule, vmr_array), ...]`` the caller sorted by peak abundance, so
    the legend reads top-down.

    Axis windows are the module defaults; this figure carries no per-axis
    controls (the panel it belongs to has no figure-settings block).
    """
    p = np.asarray(p_bar, dtype=float)
    T = np.asarray(T_K, dtype=float)
    if p.ndim != 1 or p.shape != T.shape or p.size == 0:
        raise ValueError(f"build_structure_figure: p_bar {p.shape} and T_K "
                         f"{T.shape} must be matching non-empty 1-D arrays")
    cols = list(columns)
    if not cols:
        raise ValueError("build_structure_figure: columns is empty")
    with render_lock:
        fig, (ax_t, ax_v) = plt.subplots(
            1, 2, figsize=(STRUCT_FIG_W_IN, STRUCT_FIG_H_IN), dpi=STRUCT_FIG_DPI,
            sharey=True)
        fig.subplots_adjust(**STRUCT_AXES_RECT)
        for ax in (ax_t, ax_v):
            ax.set_box_aspect(1.0)

        ax_t.plot(T, p, color=TP_COLOR, lw=STRUCT_LW)
        # the chemistry grid's validated temperature span
        for tlim in (320.0, 2980.0):
            ax_t.axvline(tlim, color="#cccccc", lw=0.8, ls=":")
        ax_t.set_xlim(*TP_XLIM_DEFAULT)
        # 4-digit temperature labels collide on a small box at the default
        # locator; cap the count instead of rotating them
        ax_t.xaxis.set_major_locator(MaxNLocator(nbins=MAX_LIN_TICKS,
                                                 steps=[1, 2, 5, 10]))
        ax_t.set_xlabel("temperature (K)")
        ax_t.set_ylabel("pressure (bar)")

        extra = itertools.cycle(VMR_EXTRA_COLORS)
        for m, y in cols:
            ya = np.asarray(y, dtype=float)
            if ya.shape != p.shape:
                raise ValueError(
                    f"build_structure_figure: {m} column {ya.shape} does not "
                    f"match the pressure grid {p.shape}")
            ax_v.plot(np.clip(ya, 1e-14, None), p, lw=STRUCT_LW, label=str(m),
                      color=VMR_COLORS.get(str(m)) or next(extra))
        ax_v.set_xscale("log")
        ax_v.set_xlim(*VMR_XLIM_DEFAULT)
        _thin_log_axis(ax_v, "x")
        ax_v.set_xlabel("volume mixing ratio")

        # ONE pressure axis for both panels (sharey), inverted once -- but
        # drawn on BOTH: plt.subplots(sharey=True) hides the right panel's
        # tick labels, which left the mixing-ratio panel with no visible
        # pressure axis across the gap between the panels.
        ax_t.set_yscale("log")
        ax_t.invert_yaxis()
        _thin_log_axis(ax_t, "y")
        ax_v.tick_params(axis="y", labelleft=True)
        ax_v.set_ylabel("pressure (bar)")
        for ax in (ax_t, ax_v):
            ax.grid(alpha=0.25)
        # Legend in the strip STRUCT_AXES_RECT leaves free to the RIGHT, so
        # it can never cover a profile and needs no y-limit padding. One
        # column preserves the caller's peak-abundance order.
        ax_v.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                    frameon=False, fontsize=7, ncol=1,
                    handletextpad=0.5, borderaxespad=0.0, labelspacing=0.35)
        return fig
