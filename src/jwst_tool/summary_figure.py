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
from matplotlib.lines import Line2D
from matplotlib.ticker import (LogLocator, MaxNLocator, NullLocator,
                              ScalarFormatter)

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
            curves.append(dict(label=str(_req(c, "label", cw)),
                               theta=theta, pdf=pdf,
                               mu=_mu, sigma=_sg,
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

    The padding is MULTIPLICATIVE when the axis is logarithmic. Additive
    padding is 6% of the linear range, which over a wide span reaches below
    zero (measured: a 300-75000 ppm emission range padded to -4224 ppm) and
    then trips the positive-range guard on data that is entirely positive.
    """
    # CENTERS (model curves and point depths) and WHISKERS are kept apart,
    # because a log axis treats them differently -- see below.
    centers, whiskers = [], []
    m = (spec["wl_um"] >= lo) & (spec["wl_um"] <= hi)
    if m.any():
        centers.append(spec["depth_ppm"][m])
        if spec["depth2_ppm"] is not None:
            centers.append(spec["depth2_ppm"][m])
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
        # y0 tripped the positive-range guard below, refusing a log axis on
        # data whose depths were entirely positive and spanned 1.4 decades
        # (exactly where a log axis earns its keep). The whisker is then
        # clipped at the spine and visibly runs off the bottom, which costs far
        # less than refusing the axis.
        #
        # Model depths are NOT filtered: silently dropping part of the plotted
        # curve is the failure the guard exists to catch. A first version of
        # this fix filtered everything and rendered a -400..900 ppm model as
        # its positive 69% with no indication -- so a mixed-sign model must
        # still reach the guard and be refused.
        whiskers = [w[w > 0.0] for w in whiskers]
    allv = np.concatenate([a for a in centers + whiskers if a.size])
    if allv.size == 0:
        return None
    y0, y1 = float(np.min(allv)), float(np.max(allv))
    # ASYMMETRIC padding (maintainer, 2026-08-13: "make the y scale fit a
    # little better by auto"). The old symmetric 6% left the data filling only
    # 76% of the axis -- 4.6% dead space below and 19.6% above, and the top
    # gap is separately reserved for the in-axes legend anyway. 2% below is
    # enough to keep a whisker off the spine; the legend headroom is added by
    # the caller, sized from its row count.
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
        ax.plot(spec["wl_um"], spec["depth2_ppm"], color="#888888",
                lw=1.0, ls="--", zorder=1, label=spec["depth2_label"])
    for p in spec["points"]:
        # markeredgecolor MUST be set: science.mplstyle leaves it "auto" with
        # markeredgewidth 1.0, and on a 3.6 pt marker a 1 pt black edge
        # swallows the fill -- every mode's marker rendered black and the
        # per-mode color was invisible. That color is now the series identity
        # shared with the forecast panels, so it has to read.
        # ms 3.6, down from 5.0 (maintainer, 2026-08-13: the points were
        # blocking the model). Two changes address that, and the size is the
        # smaller of them: the model line is also raised ABOVE the points
        # (zorder 4, see _plot_spectrum) so it can no longer be chopped into
        # fragments. That is what allows the markers to stay large enough to
        # READ -- rendered at 3.0 / 3.6 / 4.2 / 5.0 over the model line, and at
        # 3.0 the eight shapes all collapse to a dot, which would silently undo
        # the per-mode marker encoding. 3.6 is the smallest size where D/^/v
        # and P/X/* still separate.
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
            # Headroom for the IN-AXES legend (maintainer, 2026-08-13).
            # Unlike the old y-limit inflation this replaces, it is applied
            # to the VISIBLE data range and sized from the legend's actual
            # row count, so it scales with what is drawn instead of being a
            # fixed fudge -- and only when a legend will be drawn at all.
            # An explicit depth_range from the caller is never overridden.
            _rows = len(spec["points"]) + 1 + (spec["depth2_ppm"] is not None)
            # ~5.5% of the axis per legend row (was 7.5%, which overshot now
            # that the data padding is tighter). Two columns above 4 entries.
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
        # formatted as plain numbers rather than powers of ten.
        #
        # This replaces a MaxNLocator branch that placed LINEARLY spaced ticks
        # whenever the span was under 1.5 decades. That was written to fix a
        # real problem -- a transit depth spans a tiny fraction of a decade
        # (measured: 0.019 for a 19.6-20.5 kppm range), and a bare decade
        # locator puts every tick outside the view, so the axis drew with NO
        # labels at all. But linear ticks on a log axis are misleading: equal
        # label steps sit at unequal distances. Measured tick counts inside the
        # view with subs=(1, 2, 5): 8 at 0.019 decades, 7 at 0.48, 5 at 1.38,
        # 6 at 1.70, 9 at 2.60 -- so the subdivided log locator solves the
        # no-labels case that motivated the linear branch, at every span.
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
        # Legend INSIDE the axes (maintainer, 2026-08-13), superseding the
        # below-the-axes placement. loc="best" lets matplotlib score the
        # candidate corners against the plotted artists, so it lands where
        # the data is not -- there is NO y-limit inflation to make room (that
        # padding distorted the visible depth range purely for the legend's
        # benefit, and is what the outside placement originally fixed).
        # A translucent frame keeps it readable if it must sit over gridlines.
        # TWO COLUMNS BY GROUP (maintainer, 2026-08-13): model curves in the
        # first, instrument modes in the second. matplotlib fills a legend
        # COLUMN-MAJOR (verified: with ncol=2 and 6 entries, entries 0-2 land
        # in column 1), so the handle list is ordered models-then-modes and the
        # SHORTER group is padded with invisible entries -- otherwise the
        # column break lands mid-group and the grouping is lost.
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
        # No legend TITLE (maintainer, 2026-08-13): the entries carry their own
        # numbers, so the title was a second caption inside the legend.
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

# Forecast-panel x window: +/-N sigma about the widest drawn curve. 3 keeps
# >99.7% of the mass and fills the panel; the grid's own +/-5 sigma left the
# curve visually flat across most of the axis.
# 3.5 sigma, not 3: at 3 sigma the curve is still at 1.1% of peak when it
# meets the spine, so it reads as CLIPPED rather than as a tail going to zero.
# At 3.5 it is 0.2% -- visually on the axis -- and >99.95% of the mass is in
# frame either way.
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


# One color per PARAMETER in the merged forecast box. Chosen to stay
# distinguishable in grayscale and for the common forms of color blindness
# (blue vs vermillion, Okabe-Ito).
_PARAM_COLORS = ("#0072b2", "#d55e00")


def _plot_posterior_panel(axp, pan: dict, color: str) -> None:
    """One marginalized forecast posterior: curve, shaded 1-sigma band, and
    the width QUOTED in the panel title.

    2026-08-13: this replaces a single merged box with twin x-axes. That merge
    was geometrically self-defeating -- ``gaussian_curve`` builds its grid as
    center +/- 5 sigma, so each parameter's axis auto-scales to its own width;
    peak-normalizing then leaves exactly ONE possible outline. Measured: FWHM
    = 0.230 of the axis width for sigma 0.001, 0.035 and 5.0 alike, and the
    two curves landed within 2e-12 px of each other -- one hidden exactly
    under the other. Separate panels give each parameter its own axis back.

    The width therefore cannot be read from the curve's shape and is reported
    the way marginalized posteriors are published: as a number above the panel
    (corner.py annotates the median and 68% interval over each histogram) and
    as a shaded 1-sigma interval against real tick values.
    """
    for c in pan["curves"]:
        theta = np.asarray(c["theta"], dtype=float)
        pdf = np.asarray(c["pdf"], dtype=float)
        peak = float(np.max(pdf)) if pdf.size else 0.0
        if peak > 0.0:
            pdf = pdf / peak              # unit peak: see the y-axis note
        # each curve's OWN color when the caller set one: with the sources
        # following the spectrum's series, the color IS the series identity, so
        # a per-parameter color would erase the thing the panel is showing.
        # ``color`` (the per-parameter fallback) applies only when there is one
        # unlabelled curve and no caller color.
        _c = c.get("color") or color
        # No shaded 1-sigma band (maintainer, 2026-08-13): with one curve per
        # selected series the overlapping fills muddied the panel, and every
        # entry already quotes its sigma as a number.
        # With several sources the title cannot quote one width without
        # silently picking a source, so each ENTRY carries its own.
        _lab = str(c["label"])
        if c["sigma"] is not None and len(pan["curves"]) > 1:
            _lab = f"{_lab}: ±{_fmt_val(c['sigma'])}"
        axp.plot(theta, pdf, color=_c, ls=c["ls"], lw=c["lw"], zorder=2,
                 label=_lab)
    if pan["center"] is not None:
        # the input value the forecast is centered on: with a jitter draw the
        # curve sits OFF it, and that offset is the realization's luck
        axp.axvline(pan["center"], color="#666666", lw=0.8, ls=":", zorder=3,
                    label="input value")
    # width in the TITLE, where corner.py puts it
    _sized = [c for c in pan["curves"] if c["sigma"] is not None]
    _q = _sized[0] if len(_sized) == 1 else None
    if _q is not None:
        _mu = _q["mu"] if _q["mu"] is not None else pan["center"]
        axp.set_title(
            f"{pan['axis_label']} = {_fmt_val(_mu)} ± {_fmt_val(_q['sigma'])}"
            if _mu is not None
            else f"{pan['axis_label']}: ± {_fmt_val(_q['sigma'])}",
            fontsize=_AX_LBL)
    # NARROWER x bounds (maintainer, 2026-08-13: "hard to read right now").
    # posteriors.gaussian_curve builds its grid as center +/- 5 sigma, and at
    # +/-5 sigma the curve is visually zero across most of the axis, which
    # squeezes the informative part into the middle fifth. Clip to the widest
    # drawn curve at +/-_XLIM_SIGMA, which still contains >99.7% of every
    # curve's mass. The grid itself is untouched (it is MC-pinned elsewhere).
    _spans = [(c["mu"], c["sigma"]) for c in pan["curves"]
              if c["mu"] is not None and c["sigma"] is not None]
    if _spans:
        _lo = min(m - _XLIM_SIGMA * sg for m, sg in _spans)
        _hi = max(m + _XLIM_SIGMA * sg for m, sg in _spans)
        if pan["center"] is not None:
            # never crop the input-value line out of frame
            _lo, _hi = min(_lo, pan["center"]), max(_hi, pan["center"])
        if _hi > _lo:
            axp.set_xlim(_lo, _hi)
    # a narrow window needs a tick count that fits the panel's width
    axp.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    axp.set_xlabel(pan["axis_label"], fontsize=_AX_LBL)
    axp.set_yticks([])
    axp.tick_params(labelsize=_TICK)
    axp.set_ylabel("relative forecast density", fontsize=_AX_LBL)
    # Headroom for the in-axes legend, sized from the legend's ROW COUNT --
    # the same rule the spectrum panel uses, and what CLAUDE.md documents.
    # This replaced a hardcoded 1.42: with one curve per selected series the
    # legend grows with the selection, so a constant only held by coincidence
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
        # Spectrum at 2x width (maintainer), then ONE PANEL PER PARAMETER.
        # The single merged twin-axis box this replaces could not work: see
        # _plot_posterior_panel for the measurement. A 2:1 wavelength panel is
        # the right shape for a spectrum and matches how these appear in
        # papers; the posterior panels stay square.
        # ASPECT RATIOS (maintainer, 2026-08-13): the spectrum is golden
        # ratio (PHI:1 wide) and every forecast panel is SQUARE. Both are
        # enforced twice over -- the figure size is SOLVED so each gridspec
        # cell already has the target shape, and set_box_aspect pins the axes
        # regardless of margin drift. Without the solve, box_aspect would
        # shrink an axes inside an ill-shaped cell and leave dead whitespace;
        # without box_aspect, a margin tweak would silently skew the ratios.
        _npan = max(1, len(panels))
        _top = 0.965 if title is None else 0.905
        _bottom = 0.165 if footnote else 0.135
        # wspace 0.16, was 0.24 (maintainer, 2026-08-13): a narrower gap. The
        # y tick labels on each posterior panel are what set the floor here --
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
            # SQUARE (maintainer, 2026-08-13). The equal-height rule this
            # replaces is preserved anyway: the figure height is solved so a
            # square panel exactly fills the row, so square and
            # same-height-as-the-spectrum are the same thing here -- which is
            # why the earlier attempt needed set_box_aspect(None) and this one
            # does not.
            axp.set_box_aspect(1.0)
            if i >= len(panels):
                axp.set_axis_off()
                continue
            _plot_posterior_panel(axp, panels[i],
                                  _PARAM_COLORS[i % len(_PARAM_COLORS)])

        if title:
            fig.suptitle(str(title), fontsize=11)
        if footnote:
            fig.text(0.075, 0.03, str(footnote), fontsize=6.5,
                     color="#555555", ha="left", va="bottom", wrap=True)
    return fig