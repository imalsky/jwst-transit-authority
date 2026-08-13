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
    return dict(wl_um=wl[order], depth_ppm=depth[order],
                depth2_ppm=depth2,
                depth2_label=str(spectrum.get("depth2_label", "comparison")),
                depth_label=str(spectrum.get("depth_label",
                                             "transit depth (ppm)")),
                model_label=str(spectrum.get("model_label", "model")),
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
    lo = float(min(spec["wl_um"].min(),
                   min((p["wl_um"].min() for p in spec["points"]),
                       default=spec["wl_um"].min())))
    hi = float(max(spec["wl_um"].max(),
                   max((p["wl_um"].max() for p in spec["points"]),
                       default=spec["wl_um"].max())))
    ticks = [t for t in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 12.0)
             if lo * 0.97 <= t <= hi * 1.03]
    # 10 and 12 collide on a square panel; keep the decade
    if 10.0 in ticks and 12.0 in ticks:
        ticks.remove(12.0)
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_xlim(lo * 0.97, hi * 1.03)
    ax.set_xlabel("wavelength (µm)", fontsize=_AX_LBL)
    ax.set_ylabel(spec["depth_label"], fontsize=_AX_LBL)
    ax.tick_params(labelsize=_TICK)
    ax.grid(alpha=0.25)
    if spec["points"]:
        # headroom for the legend: 1 row per entry (2-column layout
        # halves it), so the legend never lands on the data
        _n_leg = len(spec["points"]) + 1 + (spec["depth2_ppm"] is not None)
        _ncol = 1 if _n_leg <= 3 else 2
        _rows = int(np.ceil(_n_leg / _ncol))
        _y0, _y1 = ax.get_ylim()
        ax.set_ylim(_y0, _y0 + (_y1 - _y0) / (1.0 - 0.075 * _rows))
        ax.legend(loc="upper left", frameon=False, fontsize=_LEG,
                  ncol=_ncol, handletextpad=0.5, borderaxespad=0.4,
                  columnspacing=1.0, labelspacing=0.3)



def _plot_posterior_panel(axp, pan: dict) -> None:
    """Render one marginalized forecast posterior panel."""
    for c in pan["curves"]:
        axp.plot(c["theta"], c["pdf"], color=c["color"], ls=c["ls"],
                 lw=c["lw"], label=c["label"])
    if pan["center"] is not None:
        axp.axvline(pan["center"], color="#999999", lw=0.7, ls=":")
    axp.set_xlabel(pan["axis_label"], fontsize=_AX_LBL)
    axp.set_yticks([])
    axp.tick_params(labelsize=_TICK)
    axp.set_ylabel("forecast density", fontsize=_AX_LBL)
    if pan["curves"]:
        _y0, _y1 = axp.get_ylim()
        axp.set_ylim(_y0, _y0 + (_y1 - _y0)
                     / (1.0 - 0.085 * len(pan["curves"])))
        axp.legend(loc="upper right", frameon=False, fontsize=_LEG,
                   handletextpad=0.5, borderaxespad=0.4,
                   labelspacing=0.3)
    for k, note in enumerate(pan["notes"]):
        axp.text(0.5, 0.5 - 0.18 * k, note, transform=axp.transAxes,
                 ha="center", va="center", fontsize=7,
                 color="#883333", wrap=True)
    axp.grid(alpha=0.15)



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

    with plt.style.context([str(_STYLE_FILE), _STYLE_OVERRIDES]):
        # three EQUAL, SQUARE panels side by side: spectrum, then up to two
        # marginalized forecast posteriors
        fig = plt.figure(figsize=(12.0, 4.3), dpi=200)
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0],
                              wspace=0.16,
                              left=0.075, right=0.99,
                              top=(0.88 if title else 0.94),
                              bottom=(0.18 if footnote else 0.13))

        # -- LEFT: spectrum with per-mode simulated points ------------------
        ax = fig.add_subplot(gs[0, 0])
        ax.set_box_aspect(1.0)
        _plot_spectrum(ax, spec)

        # -- CENTER/RIGHT: up to two marginalized forecast posteriors -------
        for i in range(2):
            axp = fig.add_subplot(gs[0, i + 1])
            axp.set_box_aspect(1.0)
            if i >= len(panels):
                axp.set_axis_off()
                continue
            _plot_posterior_panel(axp, panels[i])

        if title:
            fig.suptitle(str(title), fontsize=11)
        if footnote:
            fig.text(0.075, 0.015, str(footnote), fontsize=6.5,
                     color="#555555", ha="left", va="bottom", wrap=True)
    return fig