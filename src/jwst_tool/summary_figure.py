"""One-figure proposal summary: spectra + marginalized forecast densities.

Method and contract: docs/gui_and_sharing.md, section `summary_figure`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter, NullLocator, ScalarFormatter

from jwst_tool import fisher, plotting

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_STYLE_FILE = Path(__file__).resolve().parent / "science.mplstyle"

# ONE typography scale across the three square panels: the house style is
# sized for a single full-width axes, so a third-width panel needs its own.
_AX_LBL, _TICK, _LEG = 9.0, 8.0, 7.0

# The overrides on top of the house style, applied here per figure and
# globally by the GUI (app.py imports this dict), so a headless render and an
# in-app figure match.
_STYLE_OVERRIDES = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def _req(d: dict, key: str, where: str) -> Any:
    if key not in d:
        raise ValueError(f"{where}: missing required entry {key!r}")
    return d[key]


def _finite_1d(x: Any, name: str, where: str) -> np.ndarray:
    a = np.asarray(x, float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{where}: {name} must be a non-empty 1-D array, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{where}: {name} contains non-finite values")
    return a


def _validate_spectrum(spectrum: dict) -> dict:
    if not isinstance(spectrum, dict):
        raise ValueError("compose_summary_figure: spectrum must be a dict")
    wl = _finite_1d(_req(spectrum, "wl_um", "spectrum"), "wl_um", "spectrum")
    depth = _finite_1d(_req(spectrum, "depth_ppm", "spectrum"), "depth_ppm", "spectrum")
    if wl.shape != depth.shape:
        raise ValueError(f"spectrum: wl_um {wl.shape} and depth_ppm {depth.shape} shapes differ")
    order = np.argsort(wl)
    points = []
    for i, p in enumerate(spectrum.get("points") or []):
        where = f"spectrum.points[{i}]"
        pw = _finite_1d(_req(p, "wl_um", where), "wl_um", where)
        pd = _finite_1d(_req(p, "depth_ppm", where), "depth_ppm", where)
        ps = _finite_1d(_req(p, "sigma_ppm", where), "sigma_ppm", where)
        if not (pw.shape == pd.shape == ps.shape):
            raise ValueError(f"{where}: wl_um/depth_ppm/sigma_ppm shapes differ ({pw.shape}/{pd.shape}/{ps.shape})")
        points.append(
            dict(
                label=str(_req(p, "label", where)),
                color=str(_req(p, "color", where)),
                marker=str(p.get("marker", "o")),
                wl_um=pw,
                depth_ppm=pd,
                sigma_ppm=ps,
            )
        )
    depth2 = spectrum.get("depth2_ppm")
    if depth2 is not None:
        depth2 = _finite_1d(depth2, "depth2_ppm", "spectrum")
        if depth2.shape != wl.shape:
            raise ValueError(f"spectrum: depth2_ppm {depth2.shape} and wl_um {wl.shape} shapes differ")
        depth2 = depth2[order]

    def _pair(key: str, positive: bool) -> tuple | None:
        """Return optional (lo, hi) axis window -- validated, never silently ignored."""
        v = spectrum.get(key)
        if v is None:
            return None
        try:
            lo, hi = (float(v[0]), float(v[1]))
        except (TypeError, ValueError, IndexError, KeyError) as e:
            raise ValueError(f"spectrum: {key} must be a (lo, hi) pair, got {v!r}") from e
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError(f"spectrum: {key} must be finite, got {(lo, hi)}")
        if lo >= hi:
            raise ValueError(f"spectrum: {key} needs lo < hi, got {(lo, hi)}")
        if positive and lo <= 0.0:
            raise ValueError(f"spectrum: {key} is plotted on a log axis, so lo must be > 0, got {lo}")
        return (lo, hi)

    return dict(
        wl_um=wl[order],
        depth_ppm=depth[order],
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
        depth_label=str(spectrum.get("depth_label", "transit depth (ppm)")),
        model_label=str(spectrum.get("model_label", "model")),
        points=points,
    )


# The figure solves its height so each panel is square at a fixed total width,
# so every extra panel shrinks all of them; 3 is where one stops being
# readable. The GUI's multiselect reads this, so the two cannot drift.
MAX_POST_PANELS = 3


def _validate_curve(c: dict, cw: str) -> dict:
    theta = _finite_1d(_req(c, "theta", cw), "theta", cw)
    pdf = _finite_1d(_req(c, "pdf", cw), "pdf", cw)
    if theta.shape != pdf.shape:
        raise ValueError(f"{cw}: theta {theta.shape} and pdf {pdf.shape} shapes differ")
    # mu/sigma: the curve's center and 1-sigma width in DISPLAY units. Optional; when present the legend quotes the
    # width (_width_text) and the automatic x window is sized from them.
    mu, sg = c.get("mu"), c.get("sigma")
    if sg is not None:
        sg = float(sg)
        if not (np.isfinite(sg) and sg > 0.0):
            raise ValueError(f"{cw}: sigma must be finite and > 0, got {sg!r}")
    if mu is not None:
        mu = float(mu)
        if not np.isfinite(mu):
            raise ValueError(f"{cw}: mu must be finite, got {mu!r}")
    # curve_family "ln_gaussian" marks a multiplicative width (C/O): the panel goes to a log x axis and the automatic
    # window is taken in ln theta, at the width the curve was BUILT with (sigma_ln), never sigma/mu -- those differ
    # once a mock draw shifts mu.
    fam = c.get("curve_family")
    sln = c.get("sigma_ln")
    if sln is not None:
        sln = float(sln)
        if not (np.isfinite(sln) and sln > 0.0):
            raise ValueError(f"{cw}: sigma_ln must be finite and > 0, got {sln!r}")
    if fam == "ln_gaussian" and sln is None:
        raise ValueError(f"{cw}: an ln_gaussian curve must carry sigma_ln (its multiplicative width)")
    wt = c.get("width_text")
    return dict(
        label=str(_req(c, "label", cw)),
        theta=theta,
        pdf=pdf,
        mu=mu,
        sigma=sg,
        sigma_ln=sln,
        width_text=(None if wt is None else str(wt)),
        curve_family=(None if fam is None else str(fam)),
        color=str(c.get("color", "#333333")),
        ls=str(c.get("ls", "-")),
        lw=float(c.get("lw", 1.6)),
    )


def _validate_panels(posterior_panels: list | None) -> list[dict]:
    """Rebuild each panel from the keys read here; any other key a caller passes is dropped."""
    panels = list(posterior_panels or [])
    if len(panels) > MAX_POST_PANELS:
        raise ValueError(
            f"compose_summary_figure: at most {MAX_POST_PANELS} posterior panels are supported, got {len(panels)}"
        )
    out = []
    for i, pan in enumerate(panels):
        where = f"posterior_panels[{i}]"
        if not isinstance(pan, dict):
            raise ValueError(f"{where}: must be a dict")
        curves = [_validate_curve(c, f"{where}.curves[{j}]") for j, c in enumerate(pan.get("curves") or [])]
        center = pan.get("center")
        dl = pan.get("density_label")
        au = pan.get("axis_unit")
        out.append(
            dict(
                axis_label=str(_req(pan, "axis_label", where)),
                # the unit alone, so the legend can quote it after the error ("[M/H] = 0.88 +- 0.37 dex")
                axis_unit=("" if au is None else str(au)),
                curves=curves,
                density_label=(None if dl is None else str(dl)),
                center=(None if center is None else float(center)),
            )
        )
    return out


def _wl_ticks(lo: float, hi: float, max_n: int = 7) -> list[float]:
    """Return nice wavelength ticks inside [lo, hi] on a log axis.

    A fixed list (1, 1.5, 2, 3, ... 12) is right for a full-range spectrum and
    wrong for a zoom: a 3.0-3.5 um window lands a single tick. This falls back
    to progressively finer steps until the window carries enough of them, so a
    user-chosen range is always readable.
    """
    for step in (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
        first = np.ceil(lo / step) * step
        cand = [round(float(t), 4) for t in np.arange(first, hi + step * 0.5, step) if lo <= t <= hi]
        if len(cand) >= 3:
            # thin from the middle out rather than truncating the range
            while len(cand) > max_n:
                cand = cand[::2]
            return cand
    return [round(float(lo), 4), round(float(hi), 4)]


def _visible_ylim(spec: dict, lo: float, hi: float) -> tuple | None:
    """Depth range of the data actually INSIDE the wavelength window.

    Without this a zoom keeps the full-range y limits, so a narrow window
    renders as a flat line in the middle of mostly empty axes.

    The DEFAULT window fits the MODEL curves only, padded 10% of their span
    on each side. The simulated observation points
    are deliberately excluded: a low-S/N MIRI point's whisker can span
    several times the spectrum's own range and would compress the curve to a
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
            # whisker merely clips). Poison y0 so the caller's
            # positive-range guard refuses the axis; positive centers stay
            # excluded from the model-only fit.
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
        return None  # nothing visible; leave autoscale
    if spec["y_log"]:
        # A negative WHISKER may be dropped from the limits (it clips at the spine), since 50 +/- 340 ppm is ordinary at
        # low S/N. A negative CENTER may not, and model depths are never filtered: a mixed-sign model must reach the
        # positive-range guard and be refused rather than render silently cut.
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


def _spectrum_window(spec: dict) -> tuple[float, float]:
    """Return the x window: the caller's, or the data's with a 3 % margin."""
    if spec["wl_range"] is not None:
        return spec["wl_range"]
    wl = spec["wl_um"]
    lo = float(min(wl.min(), min((p["wl_um"].min() for p in spec["points"]), default=wl.min())))
    hi = float(max(wl.max(), max((p["wl_um"].max() for p in spec["points"]), default=wl.max())))
    return lo * 0.97, hi * 1.03


def _depth_limits(spec: dict, lo: float, hi: float) -> tuple | None:
    """Return the y limits, with headroom for the in-axes legend sized from its row count."""
    ylim = spec["depth_range"] or _visible_ylim(spec, lo, hi)
    if ylim is None or spec["depth_range"] is not None or not spec["points"]:
        return ylim
    rows = len(spec["points"]) + 1 + (spec["depth2_ppm"] is not None)
    # ~5.5% of the axis per legend row, matched to the data padding. Two columns above 4 entries.
    frac = min(0.38, 0.055 * (rows if rows <= 4 else np.ceil(rows / 2) + 1))
    y0, y1 = ylim
    return (y0, y0 + (y1 - y0) / (1.0 - frac))


def _spectrum_legend(ax: Axes, spec: dict) -> None:
    """Draw the legend in the reserved headroom: model curves in one column, modes in the other.

    matplotlib fills a legend column-major, so the handles run models then modes and the shorter group is padded with
    invisible entries, or the column break lands mid-group.
    """
    handles, labels = ax.get_legend_handles_labels()
    mode_labels = {str(p["label"]) for p in spec["points"]}
    models = [(h, lab) for h, lab in zip(handles, labels, strict=True) if lab not in mode_labels]
    modes = [(h, lab) for h, lab in zip(handles, labels, strict=True) if lab in mode_labels]
    ncol = 1
    if models and modes:
        rows = max(len(models), len(modes))
        blank = Line2D([], [], linestyle="none", marker="none")
        models += [(blank, " ")] * (rows - len(models))
        modes += [(blank, " ")] * (rows - len(modes))
        handles = [h for h, _ in models + modes]
        labels = [lab for _, lab in models + modes]
        ncol = 2
    # "upper left" into the reserved headroom, not "best": with a wide spectrum every corner touches data.
    leg = ax.legend(
        handles,
        labels,
        loc="upper left",
        frameon=True,
        framealpha=0.82,
        edgecolor="none",
        fontsize=_LEG,
        ncol=ncol,
        handletextpad=0.5,
        borderaxespad=0.4,
        columnspacing=1.0,
        labelspacing=0.3,
    )
    leg.set_zorder(5)


def _plot_spectrum(ax: Axes, spec: dict) -> None:
    """Render the model spectrum and each mode's simulated points."""
    # zorder 4: above the points (3), so the model stays continuous wherever a mode has coverage.
    ax.plot(spec["wl_um"], spec["depth_ppm"], color="#444444", lw=1.1, alpha=0.9, zorder=4, label=spec["model_label"])
    if spec["depth2_ppm"] is not None:
        # zorder 3.5: identical to the model outside the target's bands, so above it the curve would hide the model.
        ax.plot(
            spec["wl_um"], spec["depth2_ppm"], color="#888888", lw=1.0, ls="--", zorder=3.5, label=spec["depth2_label"]
        )
    for p in spec["points"]:
        # The explicit edge color keeps the fill readable: the color is the series identity shared with the forecast
        # panels, and the rc default's dark edge swallows a 3 pt marker.
        ax.errorbar(
            p["wl_um"],
            p["depth_ppm"],
            yerr=p["sigma_ppm"],
            fmt=p["marker"],
            ms=3.0,
            lw=0.9,
            color=p["color"],
            markerfacecolor=p["color"],
            markeredgecolor=p["color"],
            markeredgewidth=0.4,
            ecolor=p["color"],
            elinewidth=0.7,
            capsize=0,
            zorder=3,
            label=p["label"],
        )
    ax.set_xscale("log" if spec["x_log"] else "linear")
    lo, hi = _spectrum_window(spec)
    if spec["x_log"]:
        ticks = _wl_ticks(lo, hi)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())  # a lone minor tick would otherwise get a label
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 5, 10]))
    ax.set_xlim(lo, hi)
    ylim = _depth_limits(spec, lo, hi)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if spec["y_log"]:
        y0, _ = ax.get_ylim()
        if y0 <= 0.0:
            # matplotlib drops non-positive data on a log axis without saying so
            raise ValueError(
                f"spectrum: y_log=True needs a positive depth range, but the "
                f"visible range starts at {y0:.4g} ppm. Use a linear depth "
                "axis for this data, or set depth_range explicitly."
            )
        ax.set_yscale("log")
        # 1-2-5 subdivisions as plain numbers: a depth spans a fraction of a decade, where a decade locator draws none.
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_xlabel("wavelength (µm)", fontsize=_AX_LBL)
    ax.set_ylabel(spec["depth_label"], fontsize=_AX_LBL)
    ax.tick_params(labelsize=_TICK)
    ax.grid(alpha=0.25)
    if spec["points"]:
        _spectrum_legend(ax, spec)


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
    """Round a value/uncertainty for a legend entry.

    3 significant figures, no exponent for the ranges these parameters live in.
    """
    a = abs(float(v))
    if a == 0.0:
        return "0"
    if a < 1e-3 or a >= 1e4:
        return f"{v:.2e}"
    return f"{float(v):.3g}"


def _width_text(pan: dict, c: dict) -> str | None:
    """One curve's width statement for its LEGEND entry.

    No panel carries a title, so this is the only place the number appears on the figure.

    Reads "symbol = value +- error unit" ("[M/H] = 0.88 +- 0.37 dex"). The
    symbol is the axis label with its unit token removed, because the unit
    belongs AFTER the error, not between the symbol and the "=". C/O supplies
    its own text (fisher.format_co_width): its width is in log10 while its
    axis is physical C/O, so it cannot be built from the axis label.
    """
    if c.get("width_text"):
        return c["width_text"]
    if c.get("sigma") is None:
        return None
    unit = pan.get("axis_unit") or ""
    sym = str(pan["axis_label"])
    if unit and sym.endswith(f" [{unit}]"):
        sym = sym[: -len(unit) - 3]
    mu = c["mu"] if c.get("mu") is not None else pan.get("center")
    return f"{sym} = {fisher.format_pm(mu, c['sigma'])}" + (f" {unit}" if unit else "")


def _curve_window(c: dict) -> tuple[float, float]:
    """Return one curve's +/-_XLIM_SIGMA window."""
    if c.get("curve_family") != "ln_gaussian":
        return (c["mu"] - _XLIM_SIGMA * c["sigma"], c["mu"] + _XLIM_SIGMA * c["sigma"])
    # Multiplicative width, at the sigma_ln the curve was built with. Never past the drawn grid: mu*exp(3.5*sigma_ln)
    # can overflow to inf (a hard matplotlib error) well before the grid's clip to the range the model solves.
    with np.errstate(over="ignore"):
        s_ln = _XLIM_SIGMA * c["sigma_ln"]
        lo, hi = c["mu"] * float(np.exp(-s_ln)), c["mu"] * float(np.exp(s_ln))
    th = np.asarray(c["theta"], float)
    return (max(lo, float(th[0])), min(hi, float(th[-1])))


def _auto_xlim(axp: Axes, pan: dict) -> None:
    """Window the panel on the widest drawn curve, keeping the input-value line in frame. Never a resampling."""
    spans = [_curve_window(c) for c in pan["curves"] if c["mu"] is not None and c["sigma"] is not None]
    if not spans:
        return
    lo, hi = min(s[0] for s in spans), max(s[1] for s in spans)
    if pan["center"] is not None:
        lo, hi = min(lo, pan["center"]), max(hi, pan["center"])
    if hi > lo:
        axp.set_xlim(lo, hi)


def _posterior_x_axis(axp: Axes, pan: dict) -> None:
    """Physical values on a log axis for a multiplicative width, subdivided by span; otherwise a count that fits."""
    if pan["curves"] and all(c.get("curve_family") == "ln_gaussian" for c in pan["curves"]):
        axp.set_xscale("log")
        lo, hi = axp.get_xlim()
        dec = np.log10(hi / lo)
        subs: Any = "all" if dec <= 1.2 else ((1.0, 2.0, 3.0, 5.0) if dec <= 2.5 else (1.0,))
        axp.xaxis.set_major_locator(LogLocator(base=10.0, subs=subs))
        axp.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axp.xaxis.set_minor_formatter(NullFormatter())
    else:
        axp.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))


def _plot_posterior_panel(axp: Axes, pan: dict, xlim: tuple | None = None) -> None:
    """One marginalized forecast density per curve, each width QUOTED in its legend entry.

    ONE PANEL PER PARAMETER, never a merged box with twin x-axes: the curves are peak-normalized and each grid spans
    its own center +/- 5 sigma, so every parameter draws the same outline and the width is reported as a number.
    """
    for c in pan["curves"]:
        theta = np.asarray(c["theta"], dtype=float)
        pdf = np.asarray(c["pdf"], dtype=float)
        peak = float(np.max(pdf))
        if peak > 0.0:
            pdf = pdf / peak
        lab = str(c["label"])
        wt = _width_text(pan, c)
        if wt:
            lab = f"{lab}: {wt}"
        axp.plot(theta, pdf, color=c["color"], ls=c["ls"], lw=c["lw"], zorder=2, label=lab)
    if pan["center"] is not None:
        # with a jitter draw the curve sits off the input value, and that offset is the realization's luck
        axp.axvline(
            pan["center"], color="#666666", lw=0.8, ls=":", zorder=3, label=f"input value {_fmt_val(pan['center'])}"
        )
    if xlim is not None:
        axp.set_xlim(float(xlim[0]), float(xlim[1]))
    else:
        _auto_xlim(axp, pan)
    _posterior_x_axis(axp, pan)
    axp.set_xlabel(pan["axis_label"], fontsize=_AX_LBL)
    axp.set_yticks([])
    axp.tick_params(labelsize=_TICK)
    # A fit to the jitter realization passes its own density_label: a forecast is centered on the input by
    # construction, so the two must never share a label.
    axp.set_ylabel(pan.get("density_label") or "relative forecast density", fontsize=_AX_LBL)
    # Headroom for the in-axes legend, from its row count: curves peak at 1.0, and the legend grows with the selection.
    rows = len(pan["curves"]) + (pan["center"] is not None)
    axp.set_ylim(0.0, 1.0 + min(0.85, 0.13 * max(1, rows)))
    axp.grid(alpha=0.15)
    if pan["curves"]:
        leg = axp.legend(
            loc="upper left",
            frameon=True,
            framealpha=0.82,
            edgecolor="none",
            fontsize=_LEG,
            ncol=1,
            handletextpad=0.5,
            borderaxespad=0.4,
            labelspacing=0.3,
        )
        leg.set_zorder(5)


def _validate_xlims(panel_xlims: list | None) -> list:
    xlims = list(panel_xlims or [])
    for i, v in enumerate(xlims):
        if v is None:
            continue
        try:
            lo, hi = float(v[0]), float(v[1])
        except (TypeError, ValueError, IndexError, KeyError) as e:
            raise ValueError(
                f"compose_summary_figure: panel_xlims[{i}] must be a (lo, hi) pair or None, got {v!r}"
            ) from e
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f"compose_summary_figure: panel_xlims[{i}] needs finite lo < hi, got {(lo, hi)}")
        xlims[i] = (lo, hi)
    return xlims


def compose_summary_figure(
    spectrum: dict, posterior_panels: list | None = None, panel_xlims: list | None = None, caveats: Sequence[str] = ()
) -> Figure:
    """Compose the proposal summary figure; returns the Figure, which the caller closes.

    Inputs are validated loudly and never mutated; a figure that fails part-way is closed before the error propagates.
    The panels carry no prose: ``caveats`` go in the canvas's bottom margin, so a download keeps them. Input formats:
    docs/gui_and_sharing.md, section `summary_figure`.
    """
    spec = _validate_spectrum(spectrum)
    panels = _validate_panels(posterior_panels)
    xlims = _validate_xlims(panel_xlims)

    # render_lock covers both hazards: the style context mutates global rcParams, and layout measures mathtext through
    # the process-global parser (docs/gui_and_sharing.md, section `plotting`). Reentrant.
    with plotting.render_lock, plt.style.context([str(_STYLE_FILE), _STYLE_OVERRIDES]):
        # The spectrum is PHI:1 and each panel square. The figure size is solved so each gridspec cell has that shape,
        # and set_box_aspect pins the axes against margin drift.
        npan = max(1, len(panels))
        top, bottom = 0.965, 0.135
        # The y tick labels on each posterior panel set the wspace floor: below ~0.13 they collide with their neighbor.
        left, right, wspace = 0.055, 0.988, 0.16
        # fig_w = axes_h * sum(width_ratios) * (1 + npan*wspace/ncols) / (right - left), solved for the height.
        ncols = 1 + npan
        k = (_PHI + npan) * (1.0 + npan * wspace / ncols) / (right - left)
        axes_h = _FIG_W_IN / k
        fig = plt.figure(figsize=(_FIG_W_IN, axes_h / (top - bottom)), dpi=plotting.FIG_DPI)
        try:
            gs = fig.add_gridspec(
                1,
                ncols,
                width_ratios=[_PHI] + [1.0] * npan,
                wspace=wspace,
                left=left,
                right=right,
                top=top,
                bottom=bottom,
            )
            ax = fig.add_subplot(gs[0, 0])
            ax.set_box_aspect(1.0 / _PHI)
            _plot_spectrum(ax, spec)
            for i in range(npan):
                axp = fig.add_subplot(gs[0, i + 1])
                axp.set_box_aspect(1.0)
                if i >= len(panels):
                    axp.set_axis_off()
                    continue
                _plot_posterior_panel(axp, panels[i], xlim=(xlims[i] if i < len(xlims) else None))
            plotting.draw_caveats(fig, caveats)
        except BaseException:
            plt.close(fig)
            raise
    return fig
