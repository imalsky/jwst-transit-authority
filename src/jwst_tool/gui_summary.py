"""The summary figure: series and axis controls, the spectrum with its points, and the figure's downloads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import streamlit as st

from jwst_tool import detect, fisher as fisher_mod, instruments as ins, plotting, posteriors, summary_figure
from jwst_tool.gui_common import K, axis_range, fig_bytes, show_fig
from jwst_tool.gui_forecast import ALL_USABLE, COMBO_COLORS, ComboRecords, width_in_row_units

if TYPE_CHECKING:
    from jwst_tool.gui_results import RunView


def _binned_series(r: dict, mock: dict | None = None) -> tuple:
    """One evaluated mode as (wl_eff um, depth ppm, sigma ppm).

    The plotted x coordinate (it differs from wl_um near a detector gap), depths in ppm. ``mock`` substitutes that
    mode's seeded draw for the noiseless depth.
    """
    depth = mock["modes"][r["mode_key"]]["depth_mock"] if mock is not None else r["depth"]
    return (
        np.asarray(r["wl_eff"], dtype=float),
        np.asarray(depth, dtype=float) * 1e6,
        np.asarray(r["sigma"], dtype=float) * 1e6,
    )


def _legend_scores(view: RunView, fig_box: Any) -> dict:
    """Return each mode's expected performance for its legend label."""
    results, usable_jac, fisher_names, tsig, meta = (
        view.results,
        view.usable_jac,
        view.fisher_names,
        view.tsig,
        view.meta,
    )
    goal_r, have_fisher = view.goal, view.have_fisher
    # Per-mode expected performance, rendered IN the legend label of each
    # point series: the conditional template S/N for a detection goal, the
    # expected ± on a chosen parameter for a constraint/Fisher run.
    leg_num: dict = {}
    if goal_r == "detect":
        # saturated modes carry no usable data anywhere else (rankings,
        # combinations, forecasts); they get no score in the legend either
        for r in results:
            score, _ = detect.detection_metric(r)
            if not r["saturated"] and np.isfinite(score):
                leg_num[r["mode_key"]] = f"S/N {score:.1f}σ"
    elif have_fisher:
        rk_key = K("sum_rank_param_" + "_".join(fisher_names))
        if st.session_state.get(rk_key) not in fisher_names:
            st.session_state.pop(rk_key, None)
        gp_default = meta.get("goal_param")
        with fig_box:
            rk_param = st.selectbox(
                "Legend parameter (per-mode Fisher ±)",
                fisher_names,
                index=(fisher_names.index(gp_default) if gp_default in fisher_names else 0),
                key=rk_key,
                format_func=lambda n: fisher_mod.PARAM_LABELS[n],
            )
        for r in usable_jac:
            # width_in_row_units, not display_sigma: this number sits on the same page as the
            # constraint table and must be in the same coordinate (dex for C/O).
            v = tsig * width_in_row_units(rk_param, fisher_mod.mode_forecast(r, fisher_names)[rk_param])
            if np.isfinite(v):
                leg_num[r["mode_key"]] = f"±{v:.3g}"
    return leg_num


def _series_controls(view: RunView, combos: ComboRecords, fig_ctx: Any) -> tuple[list, dict]:
    """Draw the series selector; return the selection and each combination's usable members."""
    ok, combos_cfg = view.ok, combos.cfg
    # Figure controls
    # The figure re-renders from the CACHED run: which series are drawn and what
    # wavelength window is shown are display choices, so none of this recomputes
    # a spectrum or an ETC job. model_cache.params_key excludes the mode selection
    # and the Pandeia cache is per-mode, so re-rendering is free.
    series_opts = [("mode", r["mode_key"], r["label"]) for r in ok]
    # custom sets from the combination builder: scored through Fisher, with no
    # spectrum of their own, so a combo's series is its members' points merged
    # onto one line (disclosed in the label).
    combo_members = {str(c["name"]): [m for m in c["modes"] if any(r["mode_key"] == m for r in ok)] for c in combos_cfg}
    series_opts += [("combo", n, f"{n} (combined)") for n, mem in combo_members.items() if mem]
    # The all-usable row is a FORECAST source, not a distinct spectrum: selecting
    # it draws every usable mode's points (which is what those modes already
    # draw), and its value is that its combined posterior can appear in the
    # panels. Labelled so that redundancy is not a surprise.
    if len(ok) >= 2:
        series_opts.append(("allusable", "all", f"{ALL_USABLE} (forecast only)"))
    series_ids = [f"{kind}:{key}" for kind, key, _ in series_opts]
    series_lbl = {f"{kind}:{key}": lbl for kind, key, lbl in series_opts}

    with fig_ctx:
        sel_series = st.multiselect(
            "Modes",
            series_ids,
            default=[i for i in series_ids if i.startswith("mode:")],
            format_func=lambda i: series_lbl[i],
            key=K("sum_series"),
        )
    return sel_series, combo_members


def _default_window(view: RunView, sel_series: list, combo_members: dict, wl_s: np.ndarray) -> tuple[float, float]:
    """Return the span the selected modes cover, so the figure is not mostly empty spectrum."""
    ok = view.ok

    def _series_modes(i: str) -> list:
        """Mode keys a series id covers (the all-usable entry: every usable mode)."""
        kind, key = i.split(":", 1)
        if kind == "mode":
            return [key]
        if kind == "allusable":
            return [r["mode_key"] for r in ok]
        return combo_members.get(key, [])

    cov = [
        (ins.MODES[k]["wl_min"], ins.MODES[k]["wl_max"]) for i in sel_series for k in _series_modes(i) if k in ins.MODES
    ]
    grid_lo, grid_hi = float(wl_s[0]), float(wl_s[-1])
    if cov:
        fit = (max(grid_lo, min(c[0] for c in cov) * 0.97), min(grid_hi, max(c[1] for c in cov) * 1.03))
    else:
        fit = (grid_lo, grid_hi)
    return fit


def _figure_settings(fig_ctx: Any, fig_box: Any, post_panels: list, fit: tuple) -> tuple:
    """Draw the axis controls; return (wavelength range, depth range, log x, log y, panel x limits)."""
    # Figure settings. These move the WINDOW and the SCALES the figure is drawn
    # with and never the data, so
    # nothing downstream reads them -- the scores, the CSVs and the forecast are
    # unchanged by anything in here. One row per axis: typed min and max, blank
    # for automatic (see axis_range), with that axis's log toggle on the same
    # row. x defaults to log (the wavelength convention here); y to linear,
    # since transit depth spans a narrow range where log adds nothing.
    with fig_ctx.expander("Figure settings"):
        wx, wlg = st.columns([2.6, 0.9], vertical_alignment="bottom")
        wl_range = axis_range(wx, "Wavelength", K("sum_x"), fig_box.warning, unit="um", step=0.1, positive=True)
        x_log = wlg.checkbox("Log x", value=True, key=K("sum_xlog"))
        dx, dlg = st.columns([2.6, 0.9], vertical_alignment="bottom")
        depth_range = axis_range(
            dx,
            "Depth",
            K("sum_y"),
            fig_box.warning,
            unit="ppm",
            step=1.0,
        )
        y_log = dlg.checkbox("Log y", value=False, key=K("sum_ylog"))
        # no unit= here: fisher_mod.param_axis already carries it inside the label
        # ("[M/H] [dex]"), so passing one again reads "[M/H] [dex] min (dex)"
        post_xlims = [
            axis_range(
                st,
                p["axis_label"],
                K("sum_post_" + p["param"]),
                fig_box.warning,
                # C/O is the one panel on a positive-only axis: a
                # ratio has no values at or below zero, and its curves
                # are drawn on a log x axis (summary_figure)
                positive=(p["param"] == "dlnCO"),
                positive_reason="C/O is a ratio, so only positive values exist",
            )
            for p in post_panels
        ]
    # Blank wavelength boxes fall back to the span the SELECTED modes cover, so
    # choosing modes remains a wavelength control on its own.
    if wl_range is None:
        wl_range = fit
    return wl_range, depth_range, x_log, y_log, post_xlims


def _points(
    view: RunView, combos: ComboRecords, sel_series: list, combo_members: dict, leg_num: dict, mock: dict | None
) -> list:
    """Return the summary figure's point series: each selected mode, then each combination's merged points."""
    results, ok, combo_names_cfg = view.results, view.ok, combos.names
    sum_points = []
    for r in results:
        # saturated modes are excluded from every ranking, combination and
        # forecast; plotting simulated points for one would give unusable data
        # scientific weight. The saturation itself is disclosed in the
        # excluded-modes warning above.
        if r["saturated"]:
            continue
        if f"mode:{r['mode_key']}" not in sel_series and "allusable:all" not in sel_series:
            continue  # deselected in the controls above
        pwl, pdep, psig = _binned_series(r, mock)
        lbl = r["label"]
        if r["mode_key"] in leg_num:
            lbl += f": {leg_num[r['mode_key']]}"
        sum_points.append(
            dict(
                label=lbl,
                color=ins.MODE_COLOR[r["mode_key"]],
                # Per-mode marker from the registry. At the size these points
                # render at the SHAPES do not separate; they read in the legend,
                # where the handle is drawn full size. On the panel itself modes
                # are distinguished by colour plus the legend entry.
                marker=ins.MODE_MARKER[r["mode_key"]],
                wl_um=pwl,
                depth_ppm=pdep,
                sigma_ppm=psig,
            )
        )
    for cname, members in combo_members.items():
        if not members or f"combo:{cname}" not in sel_series:
            continue
        cw, cd, cs = (
            np.concatenate(a)
            for a in zip(
                *(_binned_series(next(x for x in ok if x["mode_key"] == mk), mock) for mk in members), strict=True
            )
        )
        o = np.argsort(cw)
        sum_points.append(
            dict(
                label=f"{cname} (combined)",
                marker="d",
                color=COMBO_COLORS[combo_names_cfg.index(cname) % len(COMBO_COLORS)],
                wl_um=cw[o],
                depth_ppm=cd[o],
                sigma_ppm=cs[o],
            )
        )
    return sum_points


def _downloads(
    view: RunView,
    fig_sum: Any,
    fig_box: Any,
    wl_s: np.ndarray,
    d_s: np.ndarray,
    d_wo_s: np.ndarray | None,
    mock: dict | None,
    seed: int,
) -> None:
    """Show the figure and offer it and its numbers for download."""
    results, meta, fname_base = view.results, view.meta, view.fname_base
    # plotted/binned numbers for the downloads under the summary figure
    bin_rows = []
    for r in results:
        bwl, bd, bs = _binned_series(r)
        bin_rows.append(
            pd.DataFrame(
                {
                    "mode": r["mode_key"],
                    "label": r["label"],
                    "wl_um": np.asarray(r["wl"], dtype=float),
                    "wl_eff_um": bwl,
                    "depth_ppm": bd,
                    "sigma_ppm": bs,
                    "saturated": bool(r["saturated"]),
                }
            )
        )
    bin_df = pd.concat(bin_rows, ignore_index=True)
    native = {"wl_um": wl_s, "depth_ppm": d_s}
    if d_wo_s is not None:
        native[f"depth_{meta['target']}_opacity_removed_ppm"] = d_wo_s
    sum_png = fig_bytes(fig_sum, "png")
    sum_pdf = fig_bytes(fig_sum, "pdf")
    with fig_box:
        show_fig(fig_sum, sum_png)
    s1, s2, s3, s4, s5 = fig_box.columns([1.5, 1.2, 1.5, 1.5, 1.9])
    s1.download_button(
        "Figure (PDF, vector)",
        sum_pdf,
        f"{fname_base}_proposal_summary.pdf",
        "application/pdf",
        key=K("dl_summary_pdf"),
        on_click="ignore",
    )
    s2.download_button(
        "Figure (PNG)",
        sum_png,
        f"{fname_base}_proposal_summary.png",
        "image/png",
        key=K("dl_summary_png"),
        on_click="ignore",
    )
    s3.download_button(
        "Binned points (CSV)",
        view.csv(bin_df),
        f"{fname_base}_binned_points.csv",
        "text/csv",
        key=K("dl_spec_bins"),
        on_click="ignore",
    )
    s4.download_button(
        "Native model (CSV)",
        view.csv(pd.DataFrame(native)),
        f"{fname_base}_model_spectrum.csv",
        "text/csv",
        key=K("dl_spec_native"),
        on_click="ignore",
    )
    if mock is not None:
        _mock_download(view, s5, mock, seed)


def _mock_download(view: RunView, s5: Any, mock: dict | None, seed: int) -> None:
    """Offer the seeded mock realization, only under a name that says what it is; the result CSVs stay noiseless."""
    results, fname_base = view.results, view.fname_base
    # is (a seeded mock realization) -- the result CSVs above stay noiseless
    mock_rows = []
    for r in results:
        # wl_eff is the PLOTTED x coordinate: the figure must be reproducible
        # from this file alone
        mwl, md, ms = _binned_series(r, mock)
        mock_rows.append(
            pd.DataFrame(
                {
                    "mode": r["mode_key"],
                    "label": r["label"],
                    "wl_um": np.asarray(r["wl"], dtype=float),
                    "wl_eff_um": mwl,
                    "depth_mock_ppm": md,
                    "sigma_ppm": ms,
                    "seed": seed,
                    "disclosure": mock["label"],
                    "seed_scheme": mock["seed_scheme"],
                    "numpy_version": mock["numpy_version"],
                }
            )
        )
    mock_df = pd.concat(mock_rows, ignore_index=True)
    s5.download_button(
        "Mock observation (CSV)",
        view.csv(mock_df),
        f"{fname_base}_mock_realization_seed{seed}.csv",
        "text/csv",
        key=K("dl_spec_mock"),
        on_click="ignore",
        help=posteriors.MOCK_SHORT_LABEL,
    )


def render_summary(view: RunView, combos: ComboRecords, post_panels: list, mock: dict | None, seed: int) -> None:
    """Draw the summary figure from the stored run; nothing here recomputes a spectrum or an ETC job."""
    model, meta, goal_r, emis, ev = view.model, view.meta, view.goal, view.emission, view.event
    wl = model["wl_um"]
    order = np.argsort(wl)
    wl_s, d_s = wl[order], model["depth"][order] * 1e6

    def _display_smooth(y_ppm: np.ndarray) -> np.ndarray:
        return plotting.display_smooth(wl_s, y_ppm, int(meta["r_bin"]))

    d_plot = _display_smooth(d_s)
    d_wo_s = None
    if goal_r == "detect":
        # depth_wo rows align with the model's wo_mols set, not mols
        wo_mols_r = [str(x) for x in model["wo_mols"]]
        d_wo_s = model["depth_wo"][wo_mols_r.index(meta["target"])][order] * 1e6
    depth_lbl = f"{ev} depth (ppm)"
    # An EXPANDER like every other results section. Expanded by DEFAULT: it
    # holds the figure the whole page builds toward, so it should be visible on
    # arrival, unlike the supporting sections above it.
    fig_box = st.expander(
        "Simulated eclipse emission spectrum & forecast summary"
        if emis
        else "Simulated transmission spectrum & forecast summary",
        expanded=True,
    )
    leg_num = _legend_scores(view, fig_box)
    fig_ctx = fig_box.container()
    sel_series, combo_members = _series_controls(view, combos, fig_ctx)
    fit = _default_window(view, sel_series, combo_members, wl_s)
    wl_range, depth_range, x_log, y_log, post_xlims = _figure_settings(fig_ctx, fig_box, post_panels, fit)
    sum_points = _points(view, combos, sel_series, combo_members, leg_num, mock)

    sum_spectrum = dict(
        wl_um=wl_s,
        depth_ppm=d_plot,
        depth_label=depth_lbl,
        model_label="model",
        wl_range=wl_range,
        depth_range=depth_range,
        x_log=x_log,
        y_log=y_log,
        points=sum_points,
    )
    if d_wo_s is not None:
        # detect goal: the without-target comparison curve, smoothed identically
        # to the model curve for display
        sum_spectrum["depth2_ppm"] = _display_smooth(d_wo_s)
        sum_spectrum["depth2_label"] = f"No {meta['target']}"

    def _compose(spec: dict) -> Any:
        # No in-figure title: the section header above already names the figure;
        # the exported PNG/PDF carry the planet name in their FILENAME.
        return summary_figure.compose_summary_figure(
            spec, posterior_panels=post_panels or None, panel_xlims=post_xlims, caveats=view.caveats
        )

    try:
        fig_sum = _compose(sum_spectrum)
    except ValueError as _e:
        # A log DEPTH axis is refused when the visible range reaches zero or below
        # (eclipse depths and low-S/N jitter draws can). That is a widget choice,
        # not a broken run: say why, fall back to linear, and keep the page alive.
        # Any other ValueError is a real defect and must not be swallowed.
        if "y_log=True" not in str(_e):
            raise
        fig_box.warning("Log depth axis needs a positive visible range; showing a linear axis.")
        sum_spectrum["y_log"] = False
        fig_sum = _compose(sum_spectrum)
    _downloads(view, fig_sum, fig_box, wl_s, d_s, d_wo_s, mock, seed)
