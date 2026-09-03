"""One-figure proposal summary: spectra + marginalized forecast densities.

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
* CENTER and RIGHT -- what would the measurement look like? Up to three 1D
  marginalized Fisher-Gaussian forecast curves (linearized Cramer-Rao
  forecasts, never sampled retrieval posteriors). An unconstrained direction
  renders as an explicit annotation, never a fake finite curve.

House style: the vendored science.mplstyle plus the serif/STIX overrides
this module owns (_STYLE_OVERRIDES, which the GUI applies globally), so a
standalone render matches the in-app figures.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import (FuncFormatter, LogLocator, MaxNLocator,
                              NullFormatter, NullLocator, ScalarFormatter)

from jwst_tool import plotting

_STYLE_FILE = Path(__file__).resolve().parent / "science.mplstyle"

# ONE typography scale across the three square panels: the vendored style is
# sized for a single full-width axes, so a third-width panel needs its own.
_AX_LBL, _TICK, _LEG = 9.0, 8.0, 7.0

# The overrides on top of the vendored style, applied here per figure and
# globally by the GUI (app.py imports this dict), so a headless render and an
# in-app figure match.
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
                # axis scales (GUI checkboxes). x defaults to log, which is
                # the wavelength convention here; y defaults to linear,
                # because transit depth spans a narrow range.
                x_log=bool(spectrum.get("x_log", True)),
                y_log=bool(spectrum.get("y_log", False)),
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


# The figure solves its height so each panel is square at a fixed total width,
# so every extra panel shrinks all of them; 3 is where one stops being
# readable. The GUI's multiselect reads this, so the two cannot drift.
MAX_POST_PANELS = 3


def _validate_panels(posterior_panels) -> list[dict]:
    panels = list(posterior_panels or [])
    if len(panels) > MAX_POST_PANELS:
        raise ValueError(f"compose_summary_figure: at most {MAX_POST_PANELS} "
                         f"posterior panels are supported, got {len(panels)}")
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
            # mu/sigma are the curve's center and 1-sigma width in DISPLAY
            # units. Optional, but when present the panel shades the 1-sigma
            # interval and quotes the width -- which is how a marginalized
            # posterior is conventionally reported (corner.py annotates the
            # median and 68% interval above each histogram; papers mark the
            # 1-sigma range with shading or vertical lines). Without them the
            # curve's SHAPE carries no width information at all, because each
            # parameter's axis is auto-scaled to its own +/-5 sigma.
            _mu, _sg = c.get("mu"), c.get("sigma")
            if _sg is not None:
                _sg = float(_sg)
                if not (np.isfinite(_sg) and _sg > 0.0):
                    raise ValueError(f"{cw}: sigma must be finite and > 0, "
                                     f"got {_sg!r}")
            if _mu is not None:
                _mu = float(_mu)
                if not np.isfinite(_mu):
                    raise ValueError(f"{cw}: mu must be finite, got {_mu!r}")
            # curve_family "ln_gaussian" marks a multiplicative width (C/O):
            # the panel goes to a log x axis and the automatic window is taken
            # in ln theta, at the width the curve was BUILT with (sigma_ln),
            # never sigma/mu -- those differ once a mock draw shifts mu.
            _fam = c.get("curve_family")
            _sln = c.get("sigma_ln")
            if _sln is not None:
                _sln = float(_sln)
                if not (np.isfinite(_sln) and _sln > 0.0):
                    raise ValueError(f"{cw}: sigma_ln must be finite and > 0, "
                                     f"got {_sln!r}")
            if _fam == "ln_gaussian" and _sln is None:
                raise ValueError(f"{cw}: an ln_gaussian curve must carry "
                                 "sigma_ln (its multiplicative width)")
            _wt = c.get("width_text")
            curves.append(dict(label=str(_req(c, "label", cw)),
                               theta=theta, pdf=pdf,
                               mu=_mu, sigma=_sg, sigma_ln=_sln,
                               width_text=(None if _wt is None else str(_wt)),
                               curve_family=(None if _fam is None
                                             else str(_fam)),
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
        # density_label overrides the y axis text. The validator REBUILDS each
        # panel dict, so any key not listed here is silently dropped -- which is
        # why this has to be threaded explicitly rather than left to pan.get()
        # in the plotting function.
        _dl = pan.get("density_label")
        out.append(dict(axis_label=str(_req(pan, "axis_label", where)),
                        curves=curves, notes=notes,
                        density_label=(None if _dl is None else str(_dl)),
                        center=(None if center is None else float(center))))

    return out


def _wl_ticks(lo: float, hi: float, max_n: int = 7) -> list[float]:
    """"Nice" wavelength ticks inside [lo, hi] on a log axis.

    A fixed list (1, 1.5, 2, 3, ... 12) is right for a full-range spectrum and
    wrong for a zoom: a 3.0-3.5 um window lands a single tick. This falls back
    to progressively finer steps until the window carries enough of them, so a
    user-chosen range is always readable.
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
    renders as a flat line in the middle of mostly empty axes.

    The DEFAULT window fits the MODEL curves only, padded 10% of their span
    on each side. The simulated observation points
    are deliberately excluded: a low-S/N MIRI point's whisker can span
    several times the spectrum's own range and compressed the curve to a
    near-flat line. Points still set the limits when no model sample is
    visible, and an explicit depth_range always wins upstream. The padding
    is MULTIPLICATIVE when the axis is logarithmic (additive padding over a
    wide span reaches below zero and trips the positive-range guard).
    """
    m = (spec["wl_um"] >= lo) & (spec["wl_um"] <= hi)
    model = []
    if m.any():
        model.append(spec["depth_ppm"][m])
        if spec["depth2_ppm"] is not None:
            model.append(spec["depth2_ppm"][m])
    if model:
        allv = np.concatenate(model)
        y0, y1 = float(np.min(allv)), float(np.max(allv))
        if spec["y_log"]:
            # A log axis silently DROPS a non-positive point center (a
            # whisker merely clips). Poison y0 so the caller's existing
            # positive-range guard refuses the axis, exactly as before;
            # positive centers stay excluded from the model-only fit.
            for p in spec["points"]:
                pm = (p["wl_um"] >= lo) & (p["wl_um"] <= hi)
                if pm.any() and float(np.min(p["depth_ppm"][pm])) <= 0.0:
                    y0 = min(y0, float(np.min(p["depth_ppm"][pm])))
        if spec["y_log"] and y0 > 0.0:
            f = (y1 / y0) ** 0.10 if y1 > y0 else 1.10
            return (y0 / f, y1 * f)
        pad = 0.10 * (y1 - y0) if y1 > y0 else max(1.0, abs(y0) * 0.01)
        return (y0 - pad, y1 + pad)
    # Fallback (no model inside the window): a points-inclusive fit, whiskers
    # included so a point's whisker is never clipped.
    centers, whiskers = [], []
    for p in spec["points"]:
        pm = (p["wl_um"] >= lo) & (p["wl_um"] <= hi)
        if pm.any():
            centers.append(p["depth_ppm"][pm])
            whiskers.append(p["depth_ppm"][pm] - p["sigma_ppm"][pm])
            whiskers.append(p["depth_ppm"][pm] + p["sigma_ppm"][pm])
    if not centers and not whiskers:
        return None                       # nothing visible; leave autoscale
    if spec["y_log"]:
        # A negative WHISKER may be dropped from the limits; a negative CENTER
        # may not. An eclipse depth of 50 +/- 340 ppm has a 1-sigma lower bound
        # below zero -- physically ordinary at low S/N -- and letting that set
        # y0 trips the positive-range guard below, refusing a log axis on data
        # whose depths are entirely positive and span 1.4 decades (exactly
        # where a log axis earns its keep). The whisker is then clipped at the
        # spine and visibly runs off the bottom, which costs far less than
        # refusing the axis.
        #
        # Model depths are NOT filtered: silently dropping part of the plotted
        # curve is the failure the guard exists to catch. Filtering everything
        # renders a -400..900 ppm model as its positive 69% with no
        # indication, so a mixed-sign model must still reach the guard and be
        # refused.
        whiskers = [w[w > 0.0] for w in whiskers]
    allv = np.concatenate([a for a in centers + whiskers if a.size])
    if allv.size == 0:
        return None
    y0, y1 = float(np.min(allv)), float(np.max(allv))
    # ASYMMETRIC padding: 2% below is enough to keep a whisker off the spine,
    # and the top gap is not padded here at all -- the caller adds the legend
    # headroom, sized from its row count.
    if spec["y_log"] and y0 > 0.0:
        f = (y1 / y0) ** 0.02 if y1 > y0 else 1.02
        return (y0 / f, y1 * f)
    pad = 0.02 * (y1 - y0) if y1 > y0 else max(1.0, abs(y0) * 0.01)
    return (y0 - pad, y1 + pad)


def _plot_spectrum(ax, spec: dict) -> None:
    """Render the model spectrum and each mode's simulated points."""
    # zorder 4: ABOVE the points (zorder 3). The model is the thing the points
    # are being compared against, so it must stay continuous and unbroken --
    # with the points on top it was chopped into fragments wherever a mode had
    # coverage, which is exactly the occlusion the smaller markers address.
    ax.plot(spec["wl_um"], spec["depth_ppm"], color="#444444", lw=1.1,
            alpha=0.9, zorder=4, label=spec["model_label"])
    if spec["depth2_ppm"] is not None:
        # zorder 3.5: ABOVE the points (3), below the model (4). It is identical
        # to the model everywhere outside the target molecule's own bands, so
        # under the points it reads as absent; above the model it would hide the
        # solid line wherever the two coincide.
        ax.plot(spec["wl_um"], spec["depth2_ppm"], color="#888888",
                lw=1.0, ls="--", zorder=3.5, label=spec["depth2_label"])
    for p in spec["points"]:
        # markeredgecolor MUST be set: science.mplstyle leaves it "auto" with
        # markeredgewidth 1.0, and on a 3.6 pt marker a 1 pt black edge
        # swallows the fill -- every mode's marker renders black and the
        # per-mode color is invisible. That color is the series identity
        # shared with the forecast panels, so it has to read.
        # ms 3.6 is the smallest size at which the marker shapes still
        # separate (D/^/v and P/X/*); below it they collapse to a dot and the
        # per-mode marker encoding is silently lost.
        ax.errorbar(p["wl_um"], p["depth_ppm"], yerr=p["sigma_ppm"],
                    fmt=p["marker"], ms=3.6, lw=0.9, color=p["color"],
                    markerfacecolor=p["color"], markeredgecolor=p["color"],
                    markeredgewidth=0.4,
                    ecolor=p["color"], elinewidth=0.7, capsize=0,
                    zorder=3, label=p["label"])
    ax.set_xscale("log" if spec["x_log"] else "linear")
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
    if spec["x_log"]:
        # _wl_ticks picks "nice" values for a LOG axis; on a linear axis
        # matplotlib's own locator is already even-spaced and correct.
        ticks = _wl_ticks(lo, hi)
        if ticks:
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:g}" for t in ticks])
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 5, 10]))
    ax.set_xlim(lo, hi)
    ylim = spec["depth_range"] or _visible_ylim(spec, lo, hi)
    if ylim is not None:
        if spec["depth_range"] is None and spec["points"]:
            # Headroom for the IN-AXES legend, applied to the VISIBLE data
            # range and sized from the legend's actual row count, so it
            # scales with what is drawn instead of being a fixed fudge -- and
            # only when a legend will be drawn at all. An explicit
            # depth_range from the caller is never overridden.
            _rows = len(spec["points"]) + 1 + (spec["depth2_ppm"] is not None)
            # ~5.5% of the axis per legend row, matched to the data padding.
            # Two columns above 4 entries.
            _frac = min(0.38, 0.055 * (_rows if _rows <= 4
                                       else np.ceil(_rows / 2) + 1))
            y0, y1 = ylim
            ylim = (y0, y0 + (y1 - y0) / (1.0 - _frac))
        ax.set_ylim(*ylim)
    if spec["y_log"]:
        _y0, _y1 = ax.get_ylim()
        if _y0 <= 0.0:
            # Fail loudly rather than render an empty panel: matplotlib drops
            # non-positive data on a log axis without saying so. Eclipse depths
            # or a jitter draw can go negative at low S/N.
            raise ValueError(
                f"spectrum: y_log=True needs a positive depth range, but the "
                f"visible range starts at {_y0:.4g} ppm. Use a linear depth "
                "axis for this data, or set depth_range explicitly.")
        ax.set_yscale("log")
        # LOG-SPACED ticks at every span (LogLocator with 1-2-5 subdivisions),
        # formatted as plain numbers rather than powers of ten. A transit depth
        # spans a small fraction of a decade, where a bare decade locator puts
        # every tick outside the view and the axis draws with NO labels;
        # linearly spaced ticks are not the answer either, since equal label
        # steps would sit at unequal distances.
        ax.yaxis.set_major_locator(
            LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_xlabel("wavelength (µm)", fontsize=_AX_LBL)
    ax.set_ylabel(spec["depth_label"], fontsize=_AX_LBL)
    ax.tick_params(labelsize=_TICK)
    ax.grid(alpha=0.25)
    if spec["points"]:
        # Legend INSIDE the axes. loc="best" lets matplotlib score the
        # candidate corners against the plotted artists, so it lands where
        # the data is not -- there is NO y-limit inflation to make room, which
        # would distort the visible depth range purely for the legend's
        # benefit.
        # A translucent frame keeps it readable if it must sit over gridlines.
        # TWO COLUMNS BY GROUP: model curves in the first, instrument modes in
        # the second. matplotlib fills a legend COLUMN-MAJOR, so the handle list
        # is ordered models-then-modes and the SHORTER group is padded with
        # invisible entries -- otherwise the column break lands mid-group and
        # the grouping is lost.
        _h, _l = ax.get_legend_handles_labels()
        _mode_labels = {str(p["label"]) for p in spec["points"]}
        _models = [(h, l) for h, l in zip(_h, _l) if l not in _mode_labels]
        _modes = [(h, l) for h, l in zip(_h, _l) if l in _mode_labels]
        if _models and _modes:
            _rows = max(len(_models), len(_modes))
            _blank = Line2D([], [], linestyle="none", marker="none")
            _models += [(_blank, " ")] * (_rows - len(_models))
            _modes += [(_blank, " ")] * (_rows - len(_modes))
            _h = [h for h, _ in _models + _modes]
            _l = [l for _, l in _models + _modes]
            _ncol = 2
        else:
            _ncol = 1
        # "upper left" into the reserved headroom, not "best": with a wide
        # spectrum every corner touches data at some wavelength.
        # No legend TITLE: the entries carry their own numbers, so a title is
        # a second caption inside the legend.
        _leg = ax.legend(_h, _l, loc="upper left", frameon=True,
                         framealpha=0.82,
                         edgecolor="none", fontsize=_LEG,
                         ncol=_ncol, handletextpad=0.5,
                         borderaxespad=0.4, columnspacing=1.0,
                         labelspacing=0.3)
        _leg.set_zorder(5)



# Golden ratio for the spectrum panel, and the figure's fixed total width.
# The HEIGHT is solved from these plus the panel count (compose_summary_figure).
_PHI = 1.618
_FIG_W_IN = 13.0

# Forecast-panel x window: +/-N sigma about the widest drawn curve. The
# grid's own +/-5 sigma leaves the curve visually flat across most of the
# axis. 3.5 sigma, not 3: at 3 sigma the curve is still at 1.1% of peak when
# it meets the spine, so it reads as CLIPPED rather than as a tail going to
# zero. At 3.5 it is 0.2% -- visually on the axis -- and >99.95% of the mass
# is in frame either way.
_XLIM_SIGMA = 3.5


def _fmt_val(v: float) -> str:
    """Round a value/uncertainty for a legend entry: 3 significant figures,
    no exponent for the ranges these parameters live in."""
    a = abs(float(v))
    if a == 0.0:
        return "0"
    if a < 1e-3 or a >= 1e4:
        return f"{v:.2e}"
    return f"{float(v):.3g}"


def _plot_posterior_panel(axp, pan: dict,
                          xlim: tuple | None = None) -> None:
    """One marginalized forecast density: its curve, and the width QUOTED in
    the panel title.

    ONE PANEL PER PARAMETER, never a merged box with twin x-axes: the curves
    are peak-normalized and each grid spans its own center +/- 5 sigma, so
    every parameter draws the SAME outline and one curve hides exactly under
    the other. The width therefore cannot be read from a curve's shape and is
    reported the way marginalized posteriors are published -- as a number
    above the panel (corner.py annotates the median and 68% interval over each
    histogram) against real tick values.
    """
    for c in pan["curves"]:
        theta = np.asarray(c["theta"], dtype=float)
        pdf = np.asarray(c["pdf"], dtype=float)
        peak = float(np.max(pdf)) if pdf.size else 0.0
        if peak > 0.0:
            pdf = pdf / peak              # unit peak: see the y-axis note
        # The colour IS the series identity, shared with that series on the
        # spectrum, so it always comes from the curve (_validate_panels
        # guarantees one). No shaded 1-sigma band: with one curve per selected
        # series the overlapping fills muddy the panel. With several sources
        # the title cannot quote one width without silently picking a source,
        # so each ENTRY carries its own.
        _lab = str(c["label"])
        _wt = c.get("width_text")
        if _wt is None and c["sigma"] is not None:
            _wt = f"±{_fmt_val(c['sigma'])}"          # generic panels
        if _wt and len(pan["curves"]) > 1:
            _lab = f"{_lab}: {_wt}"
        axp.plot(theta, pdf, color=c["color"], ls=c["ls"], lw=c["lw"],
                 zorder=2, label=_lab)
    if pan["center"] is not None:
        # the input value the forecast is centered on: with a jitter draw the
        # curve sits OFF it, and that offset is the realization's luck
        axp.axvline(pan["center"], color="#666666", lw=0.8, ls=":", zorder=3,
                    label=f"input value {_fmt_val(pan['center'])}")
    # width in the TITLE, where corner.py puts it
    _sized = [c for c in pan["curves"] if c["sigma"] is not None]
    _q = _sized[0] if len(_sized) == 1 else None
    if _q is not None:
        _mu = _q["mu"] if _q["mu"] is not None else pan["center"]
        _wt = _q.get("width_text") or f"± {_fmt_val(_q['sigma'])}"
        axp.set_title(f"{pan['axis_label']} = {_fmt_val(_mu)}; {_wt}"
                      if _mu is not None
                      else f"{pan['axis_label']}: {_wt}", fontsize=_AX_LBL)
    # posteriors.gaussian_curve builds its grid as center +/- 5 sigma, where
    # the curve is visually zero across most of the axis, so the automatic
    # window clips to the widest drawn curve at +/-_XLIM_SIGMA (still >99.7% of
    # every curve's mass). The grid itself is untouched, so this is a WINDOW,
    # never a resampling: an explicit ``xlim`` from the caller shows the same
    # curve and is used verbatim.
    if xlim is not None:
        axp.set_xlim(float(xlim[0]), float(xlim[1]))
    else:
        _spans = []
        for c in pan["curves"]:
            if c["mu"] is None or c["sigma"] is None:
                continue
            if c.get("curve_family") == "ln_gaussian":
                # multiplicative width: window taken in ln theta at the width
                # the curve was BUILT with, never sigma/mu (equal only when
                # the mock draw left the center unmoved)
                _s_ln = _XLIM_SIGMA * c["sigma_ln"]
                _spans.append((c["mu"] * float(np.exp(-_s_ln)),
                               c["mu"] * float(np.exp(_s_ln))))
            else:
                _spans.append((c["mu"] - _XLIM_SIGMA * c["sigma"],
                               c["mu"] + _XLIM_SIGMA * c["sigma"]))
        if _spans:
            _lo = min(s[0] for s in _spans)
            _hi = max(s[1] for s in _spans)
            if pan["center"] is not None:
                # never crop the input-value line out of frame
                _lo, _hi = min(_lo, pan["center"]), max(_hi, pan["center"])
            if _hi > _lo:
                axp.set_xlim(_lo, _hi)
    # A multiplicative-width panel shows PHYSICAL values on a LOG axis (house
    # style: never logged values on a linear axis). The default log locator
    # emits no tick inside a sub-decade window and dozens across a wide one,
    # so the subdivisions follow the span. Otherwise: a narrow window needs a
    # tick count that fits the panel's width.
    if pan["curves"] and all(c.get("curve_family") == "ln_gaussian"
                             for c in pan["curves"]):
        axp.set_xscale("log")
        _lo, _hi = axp.get_xlim()
        _dec = np.log10(_hi / _lo)
        _subs = "all" if _dec <= 1.2 else ((1., 2., 3., 5.) if _dec <= 2.5
                                           else (1.,))
        axp.xaxis.set_major_locator(LogLocator(base=10.0, subs=_subs))
        axp.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axp.xaxis.set_minor_formatter(NullFormatter())
    else:
        axp.xaxis.set_major_locator(MaxNLocator(nbins=5,
                                                steps=[1, 2, 2.5, 5, 10]))
    axp.set_xlabel(pan["axis_label"], fontsize=_AX_LBL)
    axp.set_yticks([])
    axp.tick_params(labelsize=_TICK)
    # The y label distinguishes a FORECAST (centered on the input by
    # construction, "relative forecast density") from a fit to the jitter
    # realization (whose center moves, plain "relative density"): a forecast
    # centered anywhere but the input value would be a bug, so the two must
    # never share a label. The caller passes density_label; the forecast
    # wording is the default.
    axp.set_ylabel(pan.get("density_label")
                   or "relative forecast density", fontsize=_AX_LBL)
    # Headroom for the in-axes legend, sized from the legend's ROW COUNT --
    # the same rule the spectrum panel uses.
    # Never a hardcoded fraction: with one curve per selected series the
    # legend grows with the selection, so a constant only holds by coincidence
    # (the legend sits upper-left while the curves peak centrally). Curves are
    # peak-normalized to 1.0, so the top is 1.0 + the legend's share.
    _rows = len(pan["curves"]) + (pan["center"] is not None)
    axp.set_ylim(0.0, 1.0 + min(0.85, 0.13 * max(1, _rows)))
    axp.grid(alpha=0.15)
    if pan["curves"]:
        leg = axp.legend(loc="upper left", frameon=True, framealpha=0.82,
                         edgecolor="none", fontsize=_LEG, ncol=1,
                         handletextpad=0.5, borderaxespad=0.4,
                         labelspacing=0.3)
        leg.set_zorder(5)
    for k, note in enumerate(pan["notes"]):
        axp.text(0.5, 0.5 - 0.18 * k, note, transform=axp.transAxes,
                 ha="center", va="center", fontsize=7,
                 color="#883333", wrap=True)


def compose_summary_figure(spectrum: dict, posterior_panels=None,
                           panel_xlims=None):
    """Compose the three-panel proposal summary figure; returns the Figure.

    ``spectrum``: dict(wl_um, depth_ppm, depth_label, model_label,
    points=[dict(label, color, marker, wl_um, depth_ppm, sigma_ppm), ...]).
    Depths in ppm; the model curve is plotted as given (pre-smooth it for
    display upstream if desired -- this function never alters the data).
    Per-mode expected performance belongs IN each point label (the caller
    appends the number), so the legend carries the ranking.

    ``posterior_panels``: up to three dicts, one per parameter:
    dict(axis_label, curves=[dict(label, theta, pdf, color, ls, lw), ...],
    notes=[...], center=float|None). An unconstrained direction belongs in
    ``notes`` (rendered as an in-panel annotation), never as a curve. A
    panel must carry curves or notes -- silence is not allowed.

    ``panel_xlims``: one entry per forecast panel, each ``(lo, hi)`` in that
    panel's own parameter units or None for the automatic +/-_XLIM_SIGMA
    window. A WINDOW on the existing curves, never a resampling. Shorter than
    the panel list is allowed (the rest stay automatic).

    The spectrum's own axis windows are ``spectrum['wl_range']`` and
    ``spectrum['depth_range']``; both default to None, meaning auto-fit.

    All inputs are validated loudly (house style); the caller's arrays are
    never mutated. The caller owns the Figure (close it after saving).
    """
    spec = _validate_spectrum(spectrum)
    panels = _validate_panels(posterior_panels)
    xlims = list(panel_xlims or [])
    for i, v in enumerate(xlims):
        if v is None:
            continue
        try:
            lo, hi = float(v[0]), float(v[1])
        except (TypeError, ValueError, IndexError, KeyError):
            raise ValueError(f"compose_summary_figure: panel_xlims[{i}] must "
                             f"be a (lo, hi) pair or None, got {v!r}")
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f"compose_summary_figure: panel_xlims[{i}] needs "
                             f"finite lo < hi, got {(lo, hi)}")
        xlims[i] = (lo, hi)

    # plotting.render_lock covers BOTH hazards here: the style context mutates
    # global rcParams, and the layout/legend placement below measures mathtext
    # through matplotlib's process-global parser (plotting.py has the full
    # argument). Reentrant, so a caller holding it already is fine.
    with plotting.render_lock, \
            plt.style.context([str(_STYLE_FILE), _STYLE_OVERRIDES]):
        # Spectrum at 2x width, then ONE PANEL PER PARAMETER -- never a
        # single merged twin-axis box (_plot_posterior_panel says why). A 2:1
        # wavelength panel is the right shape for a spectrum and matches how
        # these appear in papers; the posterior panels stay square.
        # ASPECT RATIOS: the spectrum is golden
        # ratio (PHI:1 wide) and every forecast panel is SQUARE. Both are
        # enforced twice over -- the figure size is SOLVED so each gridspec
        # cell already has the target shape, and set_box_aspect pins the axes
        # regardless of margin drift. Without the solve, box_aspect would
        # shrink an axes inside an ill-shaped cell and leave dead whitespace;
        # without box_aspect, a margin tweak would silently skew the ratios.
        _npan = max(1, len(panels))
        _top, _bottom = 0.965, 0.135
        # The y tick labels on each posterior panel set the wspace floor here:
        # below ~0.13 they start colliding with the panel to their left.
        _left, _right, _wspace = 0.055, 0.988, 0.16
        # fig_w = axes_h * sum(width_ratios) * (1 + npan*wspace/ncols)
        #                / (right - left);  solve for the height that makes
        # the panels square at a fixed total width.
        _ncols = 1 + _npan
        _k = ((_PHI + _npan) * (1.0 + _npan * _wspace / _ncols)
              / (_right - _left))
        _axes_h = _FIG_W_IN / _k
        fig = plt.figure(figsize=(_FIG_W_IN, _axes_h / (_top - _bottom)),
                         dpi=200)
        gs = fig.add_gridspec(1, _ncols,
                              width_ratios=[_PHI] + [1.0] * _npan,
                              wspace=_wspace,
                              left=_left, right=_right,
                              top=_top, bottom=_bottom)

        # -- LEFT: spectrum with per-mode simulated points (2x width) -------
        ax = fig.add_subplot(gs[0, 0])
        ax.set_box_aspect(1.0 / _PHI)      # PHI:1 wide, golden ratio
        _plot_spectrum(ax, spec)

        # -- RIGHT: one square panel per marginalized forecast posterior ----
        for i in range(_npan):
            axp = fig.add_subplot(gs[0, i + 1])
            # SQUARE. The figure height is solved so a square panel exactly
            # fills the row, so square and same-height-as-the-spectrum are the
            # same thing here.
            axp.set_box_aspect(1.0)
            if i >= len(panels):
                axp.set_axis_off()
                continue
            _plot_posterior_panel(
                axp, panels[i],
                xlim=(xlims[i] if i < len(xlims) else None))
    return fig