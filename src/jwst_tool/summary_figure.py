"""One-figure proposal summary: spectra + marginalized forecast posteriors.

Pure matplotlib + numpy, importable and renderable without Streamlit (the GUI
builds the input dicts; tests render headless). The composition takes plain
dicts/arrays so it is unit-testable without any engine or noise backend.

Three equal, square panels side by side answer the collaborator's three
questions in one graphic:

* LEFT  -- can the hypothesis be tested, and which mode is best at what
  precision? The model spectrum with each mode's simulated data points
  (per-mode color + marker; optionally one seeded mock noise realization).
  Each legend entry carries that mode's expected performance number (the
  caller appends it to the point-series label: a conditional template S/N
  for a detection goal, an expected +/- for a constraint goal).
* CENTER and RIGHT -- what would the measurement look like? Up to two 1D
  marginalized
  Fisher-Gaussian forecast curves (linearized Cramer-Rao forecasts, never
  sampled retrieval posteriors). An unconstrained direction renders as an
  explicit annotation, never a fake finite curve.

House style: the vendored science.mplstyle with the GUI's serif/STIX
overrides, so the standalone render matches the in-app figures.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jwst_tool import plotting

_STYLE_FILE = Path(__file__).resolve().parent / "science.mplstyle"

# The GUI's overrides on top of the vendored style (app.py applies the same
# set globally); repeated here so a headless render matches the app.
# One typography scale across the three square panels (the vendored style is
# sized for a single full-width axes; a third-width panel needs its own).
_AX_LBL, _TICK, _LEG = 9.0, 8.0, 7.0

_STYLE_OVERRIDES = {
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def _req(d: dict, key: str, where: str):
    if key not in d:
        raise ValueError(f"{where}: missing required entry {key!r}")
    return d[key]


def _finite_1d(x, name: str, where: str) -> np.ndarray:
    a = np.asarray(x, float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{where}: {name} must be a non-empty 1-D array, "
                         f"got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{where}: {name} contains non-finite values")
    return a


def _validate_spectrum(spectrum: dict) -> dict:
    if not isinstance(spectrum, dict):
        raise ValueError("compose_summary_figure: spectrum must be a dict")
    wl = _finite_1d(_req(spectrum, "wl_um", "spectrum"), "wl_um", "spectrum")
    depth = _finite_1d(_req(spectrum, "depth_ppm", "spectrum"), "depth_ppm",
                       "spectrum")
    if wl.shape != depth.shape:
        raise ValueError(f"spectrum: wl_um {wl.shape} and depth_ppm "
                         f"{depth.shape} shapes differ")
    order = np.argsort(wl)
    points = []
    for i, p in enumerate(spectrum.get("points") or []):
        where = f"spectrum.points[{i}]"
        pw = _finite_1d(_req(p, "wl_um", where), "wl_um", where)
        pd = _finite_1d(_req(p, "depth_ppm", where), "depth_ppm", where)
        ps = _finite_1d(_req(p, "sigma_ppm", where), "sigma_ppm", where)
        if not (pw.shape == pd.shape == ps.shape):
            raise ValueError(f"{where}: wl_um/depth_ppm/sigma_ppm shapes "
                             f"differ ({pw.shape}/{pd.shape}/{ps.shape})")
        points.append(dict(label=str(_req(p, "label", where)),
                           color=str(_req(p, "color", where)),
                           marker=str(p.get("marker", "o")),
                           wl_um=pw, depth_ppm=pd, sigma_ppm=ps))
    depth2 = spectrum.get("depth2_ppm")
    if depth2 is not None:
        depth2 = _finite_1d(depth2, "depth2_ppm", "spectrum")
        if depth2.shape != wl.shape:
            raise ValueError(f"spectrum: depth2_ppm {depth2.shape} and "
                             f"wl_um {wl.shape} shapes differ")
        depth2 = depth2[order]
    _lt = spectrum.get("legend_title")

    def _pair(key, positive):
        """Optional (lo, hi) axis window -- validated, never silently ignored."""
        v = spectrum.get(key)
        if v is None:
            return None
        try:
            lo, hi = (float(v[0]), float(v[1]))
        except (TypeError, ValueError, IndexError, KeyError):
            raise ValueError(f"spectrum: {key} must be a (lo, hi) pair, "
                             f"got {v!r}")
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError(f"spectrum: {key} must be finite, got {(lo, hi)}")
        if lo >= hi:
            raise ValueError(f"spectrum: {key} needs lo < hi, got {(lo, hi)}")
        if positive and lo <= 0.0:
            raise ValueError(f"spectrum: {key} is plotted on a log axis, so "
                             f"lo must be > 0, got {lo}")
        return (lo, hi)

    return dict(wl_um=wl[order], depth_ppm=depth[order],
                # optional caller-chosen axis windows; depth_range=None means
                # "fit to whatever is visible inside wl_range"
                wl_range=_pair("wl_range", positive=True),
                depth_range=_pair("depth_range", positive=False),
                depth2_ppm=depth2,
                depth2_label=str(spectrum.get("depth2_label", "comparison")),
                depth_label=str(spectrum.get("depth_label",
                                             "transit depth (ppm)")),
                model_label=str(spectrum.get("model_label", "model")),
                # what the per-entry numbers mean -- ONE short line as the
                # legend's title, never folded into an entry label (a
                # multi-line label wrecks the legend's row spacing)
                legend_title=(None if _lt is None else str(_lt)),
                points=points)


def _validate_panels(posterior_panels) -> list[dict]:
    panels = list(posterior_panels or [])
    if len(panels) > 2:
        raise ValueError("compose_summary_figure: at most two posterior "
                         f"panels are supported, got {len(panels)}")
    out = []
    for i, pan in enumerate(panels):
        where = f"posterior_panels[{i}]"
        if not isinstance(pan, dict):
            raise ValueError(f"{where}: must be a dict")
        curves = []
        for j, c in enumerate(pan.get("curves") or []):
            cw = f"{where}.curves[{j}]"
            theta = _finite_1d(_req(c, "theta", cw), "theta", cw)
            pdf = _finite_1d(_req(c, "pdf", cw), "pdf", cw)
            if theta.shape != pdf.shape:
                raise ValueError(f"{cw}: theta {theta.shape} and pdf "
                                 f"{pdf.shape} shapes differ")
            curves.append(dict(label=str(_req(c, "label", cw)),
                               theta=theta, pdf=pdf,
                               kind=(None if c.get("kind") is None
                                     else str(c["kind"])),
                               color=str(c.get("color", "#333333")),
                               ls=str(c.get("ls", "-")),
                               lw=float(c.get("lw", 1.6))))
        notes = [str(n) for n in (pan.get("notes") or [])]
        if not curves and not notes:
            raise ValueError(f"{where}: needs curves or notes (an empty "
                             "panel says nothing; annotate unconstrained "
                             "directions explicitly)")
        center = pan.get("center")
        out.append(dict(axis_label=str(_req(pan, "axis_label", where)),
                        curves=curves, notes=notes,
                        center=(None if center is None else float(center))))
    return out


def _wl_ticks(lo: float, hi: float, max_n: int = 7) -> list[float]:
    """"Nice" wavelength ticks inside [lo, hi] on a log axis.

    The old fixed list (1, 1.5, 2, 3, ... 12) is right for a full-range
    spectrum and wrong for a zoom: a 3.0-3.5 um window landed a single tick.
    This falls back to progressively finer steps until the window carries
    enough of them, so a user-chosen range is always readable.
    """
    for step in (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
        first = np.ceil(lo / step) * step
        cand = [round(float(t), 4)
                for t in np.arange(first, hi + step * 0.5, step)
                if lo <= t <= hi]
        if len(cand) >= 3:
            # thin from the middle out rather than truncating the range
            while len(cand) > max_n:
                cand = cand[::2]
            return cand
    return [round(float(lo), 4), round(float(hi), 4)]


def _visible_ylim(spec: dict, lo: float, hi: float):
    """Depth range of the data actually INSIDE the wavelength window.

    Without this a zoom keeps the full-range y limits, so a narrow window
    renders as a flat line in the middle of mostly empty axes. Error bars are
    included so a point's whisker is never clipped.
    """
    vals = []
    m = (spec["wl_um"] >= lo) & (spec["wl_um"] <= hi)
    if m.any():
        vals.append(spec["depth_ppm"][m])
        if spec["depth2_ppm"] is not None:
            vals.append(spec["depth2_ppm"][m])
    for p in spec["points"]:
        pm = (p["wl_um"] >= lo) & (p["wl_um"] <= hi)
        if pm.any():
            vals.append(p["depth_ppm"][pm] - p["sigma_ppm"][pm])
            vals.append(p["depth_ppm"][pm] + p["sigma_ppm"][pm])
    if not vals:
        return None                       # nothing visible; leave autoscale
    allv = np.concatenate(vals)
    y0, y1 = float(np.min(allv)), float(np.max(allv))
    pad = 0.06 * (y1 - y0) if y1 > y0 else max(1.0, abs(y0) * 0.01)
    return (y0 - pad, y1 + pad)


def _plot_spectrum(ax, spec: dict) -> None:
    """Render the model spectrum and each mode's simulated points."""
    ax.plot(spec["wl_um"], spec["depth_ppm"], color="#444444", lw=1.1,
            alpha=0.9, zorder=2, label=spec["model_label"])
    if spec["depth2_ppm"] is not None:
        ax.plot(spec["wl_um"], spec["depth2_ppm"], color="#888888",
                lw=1.0, ls="--", zorder=1, label=spec["depth2_label"])
    for p in spec["points"]:
        ax.errorbar(p["wl_um"], p["depth_ppm"], yerr=p["sigma_ppm"],
                    fmt=p["marker"], ms=3.6, lw=0.9, color=p["color"],
                    ecolor=p["color"], elinewidth=0.7, capsize=0,
                    zorder=3, label=p["label"])
    ax.set_xscale("log")
    if spec["wl_range"] is not None:
        # Caller-chosen window (the GUI defaults it to the span the SELECTED
        # modes actually cover, so the figure is not mostly empty spectrum).
        lo, hi = spec["wl_range"]
    else:
        lo = float(min(spec["wl_um"].min(),
                       min((p["wl_um"].min() for p in spec["points"]),
                           default=spec["wl_um"].min())))
        hi = float(max(spec["wl_um"].max(),
                       max((p["wl_um"].max() for p in spec["points"]),
                           default=spec["wl_um"].max())))
        lo, hi = lo * 0.97, hi * 1.03
    ticks = _wl_ticks(lo, hi)
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_xlim(lo, hi)
    ylim = spec["depth_range"] or _visible_ylim(spec, lo, hi)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("wavelength (µm)", fontsize=_AX_LBL)
    ax.set_ylabel(spec["depth_label"], fontsize=_AX_LBL)
    ax.tick_params(labelsize=_TICK)
    ax.grid(alpha=0.25)
    if spec["points"]:
        # Legend OUTSIDE the axes, below (2026-08-13). It used to sit inside
        # at "upper left", which forced a y-limit inflation to keep it off the
        # data -- that padding distorted the visible depth range purely for
        # the legend's benefit. Outside the axes needs no padding and can
        # never overlap a series, so the data range is the data range.
        _n_leg = len(spec["points"]) + 1 + (spec["depth2_ppm"] is not None)
        _leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
                         frameon=False, fontsize=_LEG,
                         ncol=(1 if _n_leg <= 2 else 2), handletextpad=0.5,
                         borderaxespad=0.0, columnspacing=1.0,
                         labelspacing=0.3,
                         title=spec["legend_title"])
        if spec["legend_title"]:
            _leg.get_title().set_fontsize(_LEG)



# One color per PARAMETER in the merged forecast box. Chosen to stay
# distinguishable in grayscale and for the common forms of color blindness
# (blue vs vermillion, Okabe-Ito).
_PARAM_COLORS = ("#0072b2", "#d55e00")


def _plot_posterior_box(axp, panels: list[dict]) -> None:
    """Both marginalized forecast posteriors in ONE box, twin x-axes.

    Maintainer, 2026-08-13: replaces the two separate square panels. The two
    posteriors are DIFFERENT PARAMETERS in different units (metallicity in
    dex, C/O a bare ratio), so they cannot share one x-axis without putting
    unlike quantities on the same numbers. Instead the first parameter reads
    off the BOTTOM axis and the second off the TOP, with each axis, its tick
    labels and its curves drawn in that parameter's color -- so the
    axis-to-curve mapping is unambiguous without a lookup.

    Y AXIS: each curve is scaled to unit peak, so the axis is a RELATIVE
    density and is dimensionless. That is deliberate. A marginalized
    posterior density carries units of 1/[parameter], i.e. dex^-1 for
    metallicity and (C/O)^-1 for the ratio, so with two parameters in one box
    there is no single honest y unit to label. Normalizing removes the unit
    question and keeps the readable content -- the WIDTH of each curve, which
    is the forecast -- fully intact.
    """
    axes = [axp, axp.twiny()]
    for i, (pan, ax_i) in enumerate(zip(panels, axes)):
        col = _PARAM_COLORS[i % len(_PARAM_COLORS)]
        for c in pan["curves"]:
            pdf = np.asarray(c["pdf"], dtype=float)
            peak = float(np.max(pdf)) if pdf.size else 0.0
            if peak > 0.0:
                pdf = pdf / peak          # unit peak: see the y-axis note
            ax_i.plot(c["theta"], pdf, color=col, ls=c["ls"], lw=c["lw"],
                      label=f"{pan['axis_label']} -- {c['label']}")
        if pan["center"] is not None:
            ax_i.axvline(pan["center"], color=col, lw=0.7, ls=":", alpha=0.6)
        ax_i.set_xlabel(pan["axis_label"], fontsize=_AX_LBL, color=col)
        ax_i.tick_params(axis="x", labelsize=_TICK, colors=col)
        ax_i.xaxis.label.set_color(col)
        ax_i.set_ylim(0.0, 1.10)

    axp.set_yticks([])
    axp.set_ylabel("relative forecast density", fontsize=_AX_LBL)
    axp.grid(alpha=0.15)

    # ONE legend for the whole box, outside and below, carrying every curve
    # from both axes (twiny() otherwise gives each axis its own).
    handles, labels = [], []
    for ax_i in axes:
        h, l = ax_i.get_legend_handles_labels()
        handles += h
        labels += l
    if handles:
        axp.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, -0.16), frameon=False,
                   fontsize=_LEG, ncol=1, handletextpad=0.5,
                   borderaxespad=0.0, labelspacing=0.3)
    notes = [n for pan in panels for n in pan["notes"]]
    for k, note in enumerate(notes):
        axp.text(0.5, 0.5 - 0.18 * k, note, transform=axp.transAxes,
                 ha="center", va="center", fontsize=7,
                 color="#883333", wrap=True)



def compose_summary_figure(spectrum: dict, posterior_panels=None,
                           title: str | None = None,
                           footnote: str | None = None):
    """Compose the three-panel proposal summary figure; returns the Figure.

    ``spectrum``: dict(wl_um, depth_ppm, depth_label, model_label,
    points=[dict(label, color, marker, wl_um, depth_ppm, sigma_ppm), ...]).
    Depths in ppm; the model curve is plotted as given (pre-smooth it for
    display upstream if desired -- this function never alters the data).
    Per-mode expected performance belongs IN each point label (the caller
    appends the number), so the legend carries the ranking.

    ``posterior_panels``: up to two dicts, one per parameter:
    dict(axis_label, curves=[dict(label, theta, pdf, color, ls, lw), ...],
    notes=[...], center=float|None). An unconstrained direction belongs in
    ``notes`` (rendered as an in-panel annotation), never as a curve. A
    panel must carry curves or notes -- silence is not allowed.

    ``footnote``: honesty caption under the figure (pass
    posteriors.FORECAST_LABEL-based wording from the caller).

    All inputs are validated loudly (house style); the caller's arrays are
    never mutated. The caller owns the Figure (close it after saving).
    """
    spec = _validate_spectrum(spectrum)
    panels = _validate_panels(posterior_panels)

    # plotting.render_lock covers BOTH hazards here: the style context mutates
    # global rcParams, and the layout/legend placement below measures mathtext
    # through matplotlib's process-global parser (plotting.py has the full
    # argument). Reentrant, so a caller holding it already is fine.
    with plotting.render_lock, \
            plt.style.context([str(_STYLE_FILE), _STYLE_OVERRIDES]):
        # TWO panels (maintainer, 2026-08-13): the spectrum is the subject of
        # this figure, so it gets 2x the width of the forecast box beside it.
        # The two marginalized posteriors that used to occupy separate square
        # panels are now ONE box with twin x-axes (see _plot_posterior_box).
        #
        # The spectrum is no longer square -- a 2:1 wavelength panel is the
        # right shape for a spectrum and matches how these appear in papers.
        # The forecast box stays square.
        fig = plt.figure(figsize=(12.0, 5.0), dpi=200)
        gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.0],
                              wspace=0.20,
                              left=0.065, right=0.985,
                              top=(0.93 if title else 0.97),
                              bottom=(0.30 if footnote else 0.26))

        # -- LEFT: spectrum with per-mode simulated points (2x width) -------
        ax = fig.add_subplot(gs[0, 0])
        _plot_spectrum(ax, spec)

        # -- RIGHT: both marginalized forecast posteriors in ONE box --------
        axp = fig.add_subplot(gs[0, 1])
        axp.set_box_aspect(1.0)
        if panels:
            _plot_posterior_box(axp, panels)
        else:
            axp.set_axis_off()

        if title:
            fig.suptitle(str(title), fontsize=11)
        if footnote:
            fig.text(0.075, 0.03, str(footnote), fontsize=6.5,
                     color="#555555", ha="left", va="bottom", wrap=True)
    return fig