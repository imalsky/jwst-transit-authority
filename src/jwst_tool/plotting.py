"""Pure figure builders + the process-wide render lock.

Importable WITHOUT streamlit: every figure the GUI shows is built here from
plain arrays, so the same code the app renders is the code the tests render.

WHY THE LOCK (2026-08-13, the ParseException crash on the Space)
---------------------------------------------------------------
Matplotlib's mathtext parser is process-global state. Matplotlib 3.10 guards
it with a class-level ``Figure._render_lock`` held for the duration of
``Figure.draw()`` -- but ``Figure.tight_layout()`` runs the layout engine
directly, OUTSIDE that lock, and the layout engine measures every tick label.
On a log axis those labels are mathtext (``$\\mathdefault{10^{-8}}$``), so two
Streamlit sessions laying out figures at the same time can be inside the
shared parser concurrently. Streamlit runs each session in its own
script-runner thread, so this is reachable with two users on one instance:
the loser raises ``ValueError: ParseException: exception raised in parse
action`` out of ``tight_layout``.

VERIFIED on the deployed pin, matplotlib 3.10.0 + pyparsing 3.2.1
(deploy/requirements-app-lock.txt): ``Figure._render_lock`` exists and
``Figure.draw`` takes it, while ``Figure.tight_layout`` does not. Eight
threads running the OLD inline T-P code (subplots -> tight_layout -> savefig)
raise in 7/8; the same load through these locked builders raises in 0/8.
Locally on matplotlib 3.11.0 it is 8/8 versus 0/8. Dropping pyplot for bare
``Figure`` objects does NOT help (also 8/8) -- the race is in the shared
parser reached through layout/measurement, not in pyplot's figure registry.

``ParserElement.reset_cache()`` in ``_mathtext.Parser.parse`` plausibly
participates (it clears a global packrat cache mid-parse), but it is not
claimed as the proven mechanism; upstream describes mathtext as generally
thread-unsafe through its shared singleton parser.

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

import threading

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (LogLocator, MaxNLocator, NullFormatter,
                               NullLocator)

# Reentrant so a build_* call nested inside a caller's `with render_lock:`
# block does not deadlock.
render_lock = threading.RLock()

# IDENTICAL GEOMETRY for the T-P and mixing-ratio figures (2026-08-13): same
# canvas, same axes rectangle, so the two render at exactly the same size
# side by side. The canvas is WIDER than the square plot box on purpose --
# the surplus on the right is the mixing-ratio legend strip, and the T-P
# figure simply leaves it blank rather than growing its plot box to fill it.
#
# The rect below is chosen so the allocated box is EXACTLY square
# (0.6135 * 7.4 in = 4.54 in wide, 0.8566 * 5.3 in = 4.54 in tall), which means
# set_box_aspect(1.0) is satisfied with no shrink-to-fit -- if these numbers
# drift apart, matplotlib silently shrinks the axes and the two figures stop
# matching. Recompute BOTH if you change the canvas: the margins are held at
# their original inch sizes so the labels keep the same physical room.
#
# NOTE the figure CANVAS is deliberately NOT square here (5.6 x 4.0); only
# the axes box is. An earlier revision made the whole canvas square, so do
# not "restore" that -- the surplus width is the legend strip, and equal
# canvases are what make the two panels the same on-page size.
# test_plotting.py::test_tp_and_vmr_panels_share_one_geometry pins it.
# Canvas enlarged 5.6x4.0 -> 7.4x5.3 in (maintainer, 2026-08-13: "make the
# figure size larger for the physical structure figures, i.e. make the text
# smaller relative to the figures"). Font sizes are UNCHANGED house-style
# points, so growing the canvas is exactly what shrinks the text relative to
# the plot: the square plot box goes 3.24 -> 4.54 in on a side (1.40x linear,
# 1.96x area) while a tick label stays the same number of points. Streamlit
# stretches both panels to the column width, so they still render at equal
# on-page size -- and the PNG/PDF downloads gain the detail.
FIG_W_IN = 7.4
FIG_H_IN = 5.3
FIG_DPI = 200
AXES_RECT = dict(left=0.0984, right=0.7119, bottom=0.1208, top=0.9774)

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


def _new_panel():
    """A figure + axes at the shared geometry: identical canvas and identical
    axes rectangle for every panel built here.

    Uses an EXPLICIT rect rather than tight_layout, because tight_layout sizes
    the axes from the labels each figure happens to carry -- which is exactly
    how the T-P and mixing-ratio plots drifted to different sizes.
    """
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=FIG_DPI)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(**AXES_RECT)
    ax.set_box_aspect(1.0)
    return fig, ax


TP_XLIM_DEFAULT = (1.0, 3000.0)
VMR_XLIM_DEFAULT = (1e-12, 1.0)


def _axis_window(name, v, positive):
    """Validate an optional ``(lo, hi)`` axis window. None means the default."""
    if v is None:
        return None
    try:
        lo, hi = float(v[0]), float(v[1])
    except (TypeError, ValueError, IndexError, KeyError):
        raise ValueError(f"{name} must be a (lo, hi) pair or None, got {v!r}")
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError(f"{name} needs finite lo < hi, got {(lo, hi)}")
    if positive and lo <= 0.0:
        raise ValueError(f"{name} is plotted on a log axis, so lo must be > 0, "
                         f"got {lo}")
    return (lo, hi)


def build_tp_figure(p_bar, T_K, xlim=None, ylim=None):
    """T-P profile: pressure (log, inverted) vs temperature. Square.

    ``xlim``: temperature window in K, None for TP_XLIM_DEFAULT. ``ylim``:
    pressure window in bar, None to fit the profile. Both are drawn INVERTED
    on the pressure axis, so the caller passes (p_min, p_max) in the natural
    order and this function orients it.

    Returns ``(fig, ylim)``; the returned ``ylim`` is the pressure axis range
    in matplotlib's (inverted) order so the mixing-ratio panel beside it can
    share the vertical scale verbatim.
    """
    p = np.asarray(p_bar, dtype=float)
    T = np.asarray(T_K, dtype=float)
    if p.ndim != 1 or p.shape != T.shape or p.size == 0:
        raise ValueError(f"build_tp_figure: p_bar {p.shape} and T_K {T.shape} "
                         "must be matching non-empty 1-D arrays")
    xlim = _axis_window("build_tp_figure: xlim", xlim, positive=False)
    ylim = _axis_window("build_tp_figure: ylim", ylim, positive=True)
    with render_lock:
        fig, ax = _new_panel()
        ax.plot(T, p, color="#2a78d6", lw=1.6)
        # the chemistry grid's validated temperature span
        for tlim in (320.0, 2980.0):
            ax.axvline(tlim, color="#cccccc", lw=0.8, ls=":")
        ax.set_xlim(*(xlim or TP_XLIM_DEFAULT))
        ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(ylim[1], ylim[0])      # deep at the bottom
        else:
            ax.invert_yaxis()
        _thin_log_axis(ax, "y")
        # 4-digit temperature labels collide on a ~3 in box at the default
        # locator; cap the count instead of rotating them
        ax.xaxis.set_major_locator(MaxNLocator(nbins=MAX_LIN_TICKS,
                                               steps=[1, 2, 5, 10]))
        ax.set_xlabel("temperature (K)")
        ax.set_ylabel("pressure (bar)")
        ax.grid(alpha=0.25)
        return fig, ax.get_ylim()


def build_vmr_figure(p_bar, columns, ylim=None, xlim=None):
    """Mixing-ratio profiles: VMR (log) vs pressure (log, inverted). Square.

    ``columns``: ordered ``[(molecule, vmr_array), ...]`` (the caller sorts;
    peak-abundance order makes the legend read top-down). ``ylim``: pressure
    range to share with the T-P panel, in MATPLOTLIB order (i.e. exactly what
    ``build_tp_figure`` returned). ``xlim``: mixing-ratio window, None for
    VMR_XLIM_DEFAULT.

    The legend sits OUTSIDE the axes, in the strip to the RIGHT, so it can
    never cover a profile and needs no y-limit padding.
    """
    p = np.asarray(p_bar, dtype=float)
    cols = list(columns)
    if not cols:
        raise ValueError("build_vmr_figure: columns is empty")
    xlim = _axis_window("build_vmr_figure: xlim", xlim, positive=True)
    with render_lock:
        fig, ax = _new_panel()
        for m, y in cols:
            ya = np.asarray(y, dtype=float)
            if ya.shape != p.shape:
                raise ValueError(
                    f"build_vmr_figure: {m} column {ya.shape} does not match "
                    f"the pressure grid {p.shape}")
            ax.plot(np.clip(ya, 1e-14, None), p, lw=1.4, label=str(m))
        ax.set_xscale("log")
        ax.set_xlim(*(xlim or VMR_XLIM_DEFAULT))
        ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.invert_yaxis()
        _thin_log_axis(ax, "x")
        _thin_log_axis(ax, "y")
        ax.set_xlabel("volume mixing ratio")
        ax.set_ylabel("pressure (bar)")
        ax.grid(alpha=0.25)
        # Legend to the RIGHT of the axes (maintainer, 2026-08-13), in the
        # strip AXES_RECT leaves free. Species read top-down in the same
        # peak-abundance order the caller sorted them into, which a single
        # column preserves and a multi-column block does not. Anchored just
        # outside the axes' right edge, so it cannot touch a profile and
        # needs no y-limit padding. Placed WITHOUT tight_layout: the whole
        # point of the shared rect is that neither panel resizes itself.
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  frameon=False, fontsize=7, ncol=1,
                  handletextpad=0.5, borderaxespad=0.0, labelspacing=0.35)
        return fig
