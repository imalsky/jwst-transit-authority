"""What every GUI panel shares: widget keys, figure export and display, the axis-range widget."""

from __future__ import annotations

import io
import re
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import streamlit as st

from jwst_tool import plotting, provenance

if TYPE_CHECKING:
    import pandas as pd


def slug(s: str) -> str:
    """Return a label in lower case with runs of other characters as underscores, for a file name."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_").lower()


# On-screen width of every rendered figure, in CSS pixels: FIXED, not
# "stretch", so figures stop rescaling as the window changes (see show_fig).
# Streamlit clamps it to the container width on a narrower screen.
_FIG_DISPLAY_PX = 1100


def fig_bytes(fig: Any, fmt: str, tight: bool = True) -> bytes:
    """Export a figure for download.

    Raster PNG at plotting.FIG_DPI or vector PDF (proposal-ready; text and lines stay vector).

    This savefig is what sets the resolution of every PNG the app produces:
    matplotlib takes dpi from the save, not from the Figure, so the dpi passed
    to plt.figure()/subplots() does not reach the output.

    ``tight=True`` crops to the figure's OWN ink, so two figures on the same
    canvas come out different sizes when one carries a legend the other lacks;
    the paired T-P / mixing-ratio panels therefore export with ``tight=False``.
    bbox_inches=None would NOT mean "no crop" -- matplotlib reads it as "use
    rcParams['savefig.bbox']", which science.mplstyle sets to `tight` -- so the
    figure's own bbox is the only way to force the full canvas.

    Under plotting.render_lock: savefig measures text through the
    process-global mathtext parser (see plotting.py).
    """
    buf = io.BytesIO()
    with plotting.render_lock:
        fig.savefig(
            buf,
            format=fmt,
            facecolor="white",
            bbox_inches=("tight" if tight else fig.bbox_inches),
            **({"dpi": plotting.FIG_DPI} if fmt == "png" else {}),
        )
    return buf.getvalue()


def show_fig(fig: Any, png: bytes) -> None:
    """Show the figure's PNG bytes.

    (the ones its download button carries) at a fixed pixel width, then close the figure under the render lock.

    The bytes are the only way to display above 200 dpi (st.pyplot hardcodes
    it). A fixed width stops the figure rescaling on every window resize while
    its text stays at fixed points; Streamlit still caps it at the container
    width on a narrower screen.
    """
    st.image(png, width=_FIG_DISPLAY_PX)
    with plotting.render_lock:
        plt.close(fig)


def K(name: str) -> str:
    """Return the widget key of ``name`` under the session's reset nonce."""
    return f"n{st.session_state['reset_nonce']}_{name}"


def axis_range(
    container: Any,
    label: str,
    key: str,
    warn: Any,
    *,
    unit: str = "",
    positive: bool = False,
    positive_reason: str = "it is drawn on a log axis",
    step: float | None = None,
) -> tuple | None:
    """Draw a plain min/max pair of number boxes for one plot axis.

    Every axis control in this app is this widget:
    two typed numbers, no slider. Both boxes start BLANK, meaning "fit the
    data", rather than prefilled with the current automatic values -- the
    automatic fit tracks the run and the mode selection, so a prefilled number
    would silently pin a stale window the moment either changed.

    Returns ``(lo, hi)`` or None. Both boxes or neither: a one-sided window has
    no second edge to fall back on, and the automatic fit lives inside the
    figure builders, not here, so a half-specified range is refused VISIBLY
    through ``warn`` rather than half-applied.

    ``positive``: a bound at or below zero is refused here, with
    ``positive_reason`` naming WHY for this axis (the log-scale wavelength
    axis; the C/O axis, a ratio with no values at or below zero). The figure
    builders raise on it -- correctly, they are the API backstop -- but that
    exception reaches Streamlit uncaught and kills the whole results page. A
    typed number is a user choice, not a defect: warn and fall back to the
    automatic fit, the same way a one-sided window does.
    """
    u = f" ({unit})" if unit else ""
    c_lo, c_hi = container.columns(2)
    lo = c_lo.number_input(f"{label} min{u}", value=None, step=step, format="%.6g", key=f"{key}_min")
    hi = c_hi.number_input(f"{label} max{u}", value=None, step=step, format="%.6g", key=f"{key}_max")
    if (lo is None) != (hi is None):
        warn(f"{label} range needs both boxes filled. Fitting to the data.")
        return None
    if lo is None:
        return None
    if float(lo) >= float(hi):
        warn(f"{label} range needs min below max. Fitting to the data.")
        return None
    if positive and float(lo) <= 0.0:
        warn(f"{label} min must be above 0 ({positive_reason}). Fitting to the data.")
        return None
    return (float(lo), float(hi))


def csv_bytes(df: pd.DataFrame, caveats: list, producer: list | None) -> bytes:
    """CSV with identity and run caveats (`pd.read_csv(..., comment="#")`)."""
    p = provenance.snapshot()
    # The commits that SOLVED the column, not the ones installed now: a cache
    # hit can be many engine versions old. Unstamped results (written before
    # the stamp) fall back to the installed ones, SAID so rather than implied.
    repos = " ".join(producer or [f"{k}={p['repositories'][k]['commit'][:12]}" for k in provenance.PRODUCER_REPOS])
    if not producer:
        repos += " (installed now; result carries no producer stamp)"
    ps, cs = p["pandeia_stack"], p["cache_schema"]
    head = (
        f"# provenance: exported by jwst-transit-authority "
        f"{p['software']['jwst-transit-authority']} | solved by: {repos} | "
        f"pandeia engine {ps['engine']} refdata {ps['refdata']['version']} "
        f"psf {ps['psf']['version']} | cache model v{cs['model']} "
        f"worker v{cs['pandeia_worker']}\n"
    )
    head += "".join(f"# caveat: {c}\n" for c in caveats)
    return (head + df.to_csv(index=False)).encode()
