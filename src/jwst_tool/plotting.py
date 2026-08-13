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
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator

# Reentrant so a build_* call nested inside a caller's `with render_lock:`
# block does not deadlock.
render_lock = threading.RLock()

# Square figures, house convention (2026-08-13): every figure the tool shows
# is square, so panels compare like-for-like across the page.
FIG_SIDE_IN = 4.0
FIG_DPI = 200

# Tick-label crowding: a decade-per-tick log axis overruns a small square
# panel, so cap the number of labelled decades and let the locator thin to
# every 2nd/3rd/... decade. Minor labels are suppressed outright.
MAX_LOG_TICKS = 7


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


def build_tp_figure(p_bar, T_K):
    """T-P profile: pressure (log, inverted) vs temperature. Square.

    Returns ``(fig, ylim)``; ``ylim`` is the pressure axis range so the
    mixing-ratio panel beside it can share the vertical scale.
    """
    p = np.asarray(p_bar, dtype=float)
    T = np.asarray(T_K, dtype=float)
    if p.ndim != 1 or p.shape != T.shape or p.size == 0:
        raise ValueError(f"build_tp_figure: p_bar {p.shape} and T_K {T.shape} "
                         "must be matching non-empty 1-D arrays")
    with render_lock:
        fig, ax = plt.subplots(figsize=(FIG_SIDE_IN, FIG_SIDE_IN), dpi=FIG_DPI)
        ax.set_box_aspect(1.0)
        ax.plot(T, p, color="#2a78d6", lw=1.6)
        # the chemistry grid's validated temperature span
        for tlim in (320.0, 2980.0):
            ax.axvline(tlim, color="#cccccc", lw=0.8, ls=":")
        ax.set_xlim(1.0, 3000.0)
        ax.set_yscale("log")
        ax.invert_yaxis()
        _thin_log_axis(ax, "y")
        ax.set_xlabel("temperature (K)")
        ax.set_ylabel("pressure (bar)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig, ax.get_ylim()


def build_vmr_figure(p_bar, columns, ylim=None):
    """Mixing-ratio profiles: VMR (log) vs pressure (log, inverted). Square.

    ``columns``: ordered ``[(molecule, vmr_array), ...]`` (the caller sorts;
    peak-abundance order makes the legend read top-down). ``ylim``: pressure
    range to share with the T-P panel.

    The legend sits OUTSIDE the axes (below), so it can never cover a profile
    and needs no y-limit padding.
    """
    p = np.asarray(p_bar, dtype=float)
    cols = list(columns)
    if not cols:
        raise ValueError("build_vmr_figure: columns is empty")
    with render_lock:
        fig, ax = plt.subplots(figsize=(FIG_SIDE_IN, FIG_SIDE_IN), dpi=FIG_DPI)
        ax.set_box_aspect(1.0)
        for m, y in cols:
            ya = np.asarray(y, dtype=float)
            if ya.shape != p.shape:
                raise ValueError(
                    f"build_vmr_figure: {m} column {ya.shape} does not match "
                    f"the pressure grid {p.shape}")
            ax.plot(np.clip(ya, 1e-14, None), p, lw=1.4, label=str(m))
        ax.set_xscale("log")
        ax.set_xlim(1e-12, 1.0)
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
        # Legend below the axes. The anchor must clear the x tick labels AND
        # the x axis label -- tight_layout does not know about an artist
        # anchored outside the axes, so it is placed in FIGURE coordinates
        # after the layout runs (bbox_inches="tight" then includes it,
        # because the legend is registered on the axes).
        ncol = min(4, max(2, int(np.ceil(len(cols) / 2))))
        fig.tight_layout()
        leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02),
                        bbox_transform=fig.transFigure,
                        frameon=False, fontsize=7, ncol=ncol,
                        handletextpad=0.5, columnspacing=1.0,
                        borderaxespad=0.0)
        # make room so the saved figure is not just cropped larger
        fig.canvas.draw()
        lh = leg.get_window_extent(fig.canvas.get_renderer()).height
        fig.subplots_adjust(bottom=(lh / fig.bbox.height) + 0.16)
        return fig
