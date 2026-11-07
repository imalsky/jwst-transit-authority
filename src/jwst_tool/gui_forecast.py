"""The constraint forecast: named mode combinations, the Fisher table and the forecast panels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import streamlit as st

from jwst_tool import fisher as fisher_mod, instruments as ins, posteriors, share_config, summary_figure
from jwst_tool.gui_common import K

if TYPE_CHECKING:
    from collections.abc import Callable

    from jwst_tool.gui_results import RunView
    from jwst_tool.tool_types import Slot

# Display label for the every-usable-mode combination. A saturated mode is
# unusable data and is excluded from every combination -- that exclusion is
# disclosed per combo, not encoded in this label.
# Display only: never a stored config key.
ALL_USABLE = share_config.ALL_USABLE


# Custom-combination palette, deliberately disjoint from instruments.MODE_COLOR
# so a combo line never collides with a member mode's color.
COMBO_COLORS = ("#6a3d9a", "#117733", "#882255", "#88ccee")


# A forecast this wide carries no information; say so rather than print a
# number the Fisher linearization cannot support.
_UNCONSTRAINED_ABOVE = 1e4


_LN10 = np.log(10.0)  # ln-coordinate sigma -> dex


def _row_label(n: str) -> str:
    # C/O is fitted in ln and reported in log10, so its row is named for that; fisher.param_axis stays the physical C/O
    # the panel's x axis draws, a different thing from the coordinate its width is quoted in.
    return "log10 C/O" if n == "dlnCO" else fisher_mod.param_axis(n)


def width_in_row_units(n: str, s: float) -> float:
    """Convert an internal sigma to that row's coordinate: dex for log10 C/O (internal is ln), else display units."""
    return s / _LN10 if n == "dlnCO" else fisher_mod.display_sigma(n, s)


def _combo_add() -> None:
    """Add a named mode combination (results-area builder).

    Callbacks run before widgets instantiate, so writing the name widget's key back to "" here is allowed. The note
    key is un-namespaced on purpose: a reset's session_state.clear() must kill it (same pattern as the archive
    fill).
    """
    name = str(st.session_state.get(K("cb_name")) or "").strip()
    modes = list(st.session_state.get(K("cb_modes")) or [])
    if not name:
        st.session_state["_combo_note"] = ("error", "The combination needs a name.")
        return
    combos = st.session_state.setdefault(K("combos"), [])
    if any(c["name"] == name for c in combos) or name in share_config.reserved_combo_names():
        st.session_state["_combo_note"] = (
            "error",
            f"The name {name!r} is already taken by a combination or a mode; choose another name.",
        )
        return
    if not modes:
        st.session_state["_combo_note"] = ("error", "Select at least one instrument mode for the combination.")
        return
    combos.append(dict(name=name, modes=modes))
    st.session_state[K("cb_name")] = ""
    st.session_state["_combo_note"] = ("success", f"Added the combination {name!r}.")


def _combo_remove(i: int) -> None:
    combos = st.session_state.get(K("combos")) or []
    if 0 <= i < len(combos):
        removed = combos.pop(i)
        st.session_state["_combo_note"] = ("info", f"Removed the combination {removed['name']!r}.")


@dataclass(frozen=True)
class ComboRecords:
    """The named mode combinations in the session, evaluated on the stored run."""

    by_mode: dict
    cfg: list
    names: list
    recs: list
    errs: list


def combo_records(view: RunView) -> ComboRecords:
    """Evaluate the named combinations; one that cannot be evaluated is an error beside the builder."""
    # Evaluated through posteriors.combo_forecast (the SAME combination math as
    # the all-usable-modes row -- never reimplemented here). A combo that
    # cannot be evaluated (mode not run, all modes saturated) is reported as an
    # error next to the builder; the others still render.
    results_by_mode = {r["mode_key"]: r for r in view.with_jac}
    combos_cfg = st.session_state.get(K("combos")) or []
    combo_names_cfg = [str(c["name"]) for c in combos_cfg]
    combo_recs, combo_errs = [], []
    if combos_cfg and view.fisher_names and results_by_mode:
        for _c in combos_cfg:
            try:
                combo_recs.append(
                    posteriors.combo_forecast(
                        str(_c["name"]), list(_c["modes"]), results_by_mode, view.fisher_names, co_eval=view.co_eval
                    )
                )
            except ValueError as _e:
                combo_errs.append((str(_c["name"]), str(_e)))
    elif combos_cfg:
        combo_errs = [
            (
                str(_c["name"]),
                (
                    "no parameter constraints in this run: combinations "
                    "need a run with free parameters (a Fisher forecast)"
                ),
            )
            for _c in combos_cfg
        ]
    return ComboRecords(results_by_mode, combos_cfg, combo_names_cfg, combo_recs, combo_errs)


def _combo_builder(view: RunView, combos: ComboRecords) -> None:
    # Custom mode sets (builder)
    # Named combinations of the modes that were run, evaluated through
    # posteriors.combo_forecast (the same combination math as the
    # all-usable-modes row). They add rows to the table below and series
    # to the summary figure.
    cb_opts = [r["mode_key"] for r in view.with_jac]
    # a stored selection can reference a mode absent from this run's results
    # (Streamlit crashes at widget instantiation on off-menu session state)
    cb_stale = st.session_state.get(K("cb_modes"))
    if cb_stale is not None:
        cb_kept = [m for m in cb_stale if m in cb_opts]
        if cb_kept != list(cb_stale):
            st.session_state[K("cb_modes")] = cb_kept
    note = st.session_state.pop("_combo_note", None)
    if note is not None:
        getattr(st, note[0])(note[1])
    cbc1, cbc2 = st.columns([1.6, 2.2])
    cbc1.text_input("Combination name", key=K("cb_name"), placeholder="e.g. SOSS + G395H")
    cbc2.multiselect(
        "Modes in the combination", cb_opts, key=K("cb_modes"), format_func=lambda k: ins.MODES[k]["label"]
    )
    # The button on its OWN row, not in a second column beside the inputs:
    # the inputs carry labels, so a column-mounted button floats to the
    # top of the row and lines up with nothing.
    cba1, _ = st.columns([1.2, 3.6])
    cba1.button("Add combination", key=K("cb_add"), on_click=_combo_add, width="stretch")
    for i, c in enumerate(combos.cfg):
        cc1, cc2 = st.columns([4.0, 1.0])
        cc1.markdown(
            f"- **{c['name']}**: " + ", ".join(ins.MODES[m]["label"] if m in ins.MODES else m for m in c["modes"])
        )
        cc2.button("Remove", key=K(f"cb_rm_{i}"), on_click=_combo_remove, args=(i,))
    for cname, cerr in combos.errs:
        st.error(f"Combination {cname!r} could not be forecast: {cerr}")
    for rec in combos.recs:
        if rec["excluded"]:
            st.warning(
                f"Combination {rec['name']!r}: "
                + ", ".join(
                    ins.MODES[e["mode_key"]]["label"] if e["mode_key"] in ins.MODES else e["mode_key"]
                    for e in rec["excluded"]
                )
                + (" saturate; uses " if len(rec["excluded"]) > 1 else " saturates; uses ")
                + " + ".join(ins.MODES[m]["label"] for m in rec["usable_modes"])
                + "."
            )
    st.divider()


def _constraint_rows(view: RunView, combos: ComboRecords, mode_notes: dict) -> tuple[list, set]:
    """Return the table rows, one per mode and parameter with combinations first, and the combination names."""
    row_center = {
        n: (
            None
            if (c := posteriors.param_center(n, view.canon)) is None
            else float(np.log10(c))
            if n == "dlnCO"
            else float(c)
        )
        for n in view.fisher_names
    }

    def _cell(w: float, center: float | None = None) -> str:
        """One cell at the target significance.

        ``w`` is the 1-sigma width in the row's own coordinate. Both columns lead with the input-model value so
        each cell reads as an interval; they share that center and differ only in width.
        """
        w = view.tsig * float(w)
        if not np.isfinite(w) or w > _UNCONSTRAINED_ABOVE:
            return "unconstrained"
        return fisher_mod.format_pm(center, w)

    # long format, one row per mode x parameter, marginalized and conditional
    # side by side -- both read off the SAME nuisance-augmented Fisher matrix
    marg_col = f"marginalized at {view.tsig:g}σ"
    cond_col = f"conditional at {view.tsig:g}σ (others fixed)"

    def _param_rows(mode_label: str, sig: dict, cond: dict, note: str = "") -> list:
        # the mode's risk notes ride on its first row only
        return [
            {
                "mode": mode_label,
                "parameter": _row_label(n),
                marg_col: _cell(width_in_row_units(n, sig[n]), row_center[n]),
                cond_col: _cell(width_in_row_units(n, cond[n]), row_center[n]),
                "notes": note if i == 0 else "",
            }
            for i, n in enumerate(view.fisher_names)
        ]

    frows = []
    for r in view.with_jac:
        if r["saturated"]:
            # shown for completeness; a saturated mode contributes no usable
            # data (same exclusion policy as the verdict + combined)
            frows.append(
                {
                    "mode": r["label"],
                    "parameter": "(saturated, excluded)",
                    marg_col: "",
                    cond_col: "",
                    "notes": mode_notes.get(r["mode_key"], ""),
                }
            )
            continue
        cond: dict = {}
        sig = fisher_mod.mode_forecast(r, view.fisher_names, conditional=cond)
        frows.extend(_param_rows(r["label"], sig, cond, note=mode_notes.get(r["mode_key"], "")))
    # the all-usable row, then each named combination under the user's own
    # name, unprefixed: the same combined_forecast over its usable modes
    comb_rows = [(ALL_USABLE, view.usable_jac)] if len(view.usable_jac) >= 2 else []
    comb_rows += [(str(rec["name"]), [combos.by_mode[k] for k in rec["usable_modes"]]) for rec in combos.recs]
    for lbl, rl in comb_rows:
        cond = {}
        sig = fisher_mod.combined_forecast(rl, view.fisher_names, conditional=cond)
        frows.extend(_param_rows(lbl, sig, cond))
    # Custom combinations FIRST: they are what the user built, so they
    # lead the table. Order within each group is preserved.
    combo_names = {str(rec["name"]) for rec in combos.recs}
    frows = [_r for _r in frows if _r["mode"] in combo_names] + [_r for _r in frows if _r["mode"] not in combo_names]
    return frows, combo_names


def _constraint_table(frows: list, combo_names: set) -> None:
    # Repeated mode names blanked for READING only; the CSV below is
    # built from the UNBLANKED rows -- a machine-readable file must carry
    # the mode on every row. _is_combo rides along for the highlight and
    # is dropped before display.
    disp, prev = [], None
    for r in frows:
        r2 = dict(r)
        r2["_is_combo"] = r2["mode"] in combo_names
        if r2["mode"] == prev:
            r2["mode"] = ""
        else:
            prev = r2["mode"]
        disp.append(r2)
    disp_df = pd.DataFrame(disp)
    combo_mask = disp_df.pop("_is_combo")
    # st.table, NOT st.dataframe: st.dataframe is interactive (a column-
    # header click re-sorts), and this table cannot survive that -- the
    # mode column is blanked on repeat rows, so a re-sort detaches every
    # blank row from its mode and silently attributes numbers to the
    # wrong instrument. st.table is static by definition.
    if combo_mask.any():
        # tint the custom-set rows so they read as a group distinct from
        # the per-mode rows. TEXT color, not a background tint (a filled
        # row reads as a highlight/alert); #5b3a8e is the combo
        # palette's purple, dark enough for body text on white.
        st.table(
            disp_df.style.apply(
                lambda row: ["color: #5b3a8e; font-weight: 600" if combo_mask.loc[row.name] else "" for _ in row],
                axis=1,
            )
        )
    else:
        st.table(disp_df)


def render_forecast_panel(view: RunView, combos: ComboRecords, mode_notes: dict) -> Slot | None:
    """Draw the constraint-forecast panel; return the slot reserved for the forecast-panel selector, if any."""
    post_slot = None
    with st.expander("Parameter constraint forecast (local Fisher)"):
        if view.have_fisher:
            _combo_builder(view, combos)
            frows, combo_names = _constraint_rows(view, combos, mode_notes)
            _constraint_table(frows, combo_names)
            st.download_button(
                "Constraint forecast (CSV)",
                view.csv(pd.DataFrame(frows)),
                f"{view.fname_base}_fisher_forecast.csv",
                "text/csv",
                key=K("dl_fisher_csv"),
                on_click="ignore",
            )
            st.divider()
            # Filled by forecast_panels, which needs the records built after this panel renders.
            post_slot = st.container()
        else:
            # Saved combinations outlive the run that could evaluate them; without
            # this they would vanish from the page with no word about why.
            for cname, cerr in combos.errs:
                st.info(f"Combination {cname!r} could not be forecast: {cerr}")
    return post_slot


def _panel_curves(
    p: str, src_labels_drawn: list, recs_by_src: dict, mock_rec: dict, series_color: Callable[[str], str]
) -> list:
    """Return one parameter's curves, one per drawn series, recentred on the mock fit when there is one."""
    curves = []
    for lbl in src_labels_drawn:
        # one curve per SELECTED SERIES, colored to match its series on the spectrum
        pr = recs_by_src[lbl]["params"].get(p)
        col = series_color(lbl)
        if pr is None:
            continue  # no center: no curve, no prose
        if pr["constrained"]:
            mr = mock_rec.get(lbl)
            mc = None
            if mr is not None and mr["recovered"].get(p):
                # THE JITTERED CURVE is the one drawn: same width as
                # the forecast (it is the forecast, recentered on
                # what this realization recovers), so no width
                # information is lost by dropping the unshifted twin.
                # The dotted center line marks the input value.
                if p == "dlnCO":
                    # C/O lives on (0, inf): shift multiplicatively by
                    # the internal ln-space draw and keep the
                    # ln_gaussian family. Off-scale draws return None
                    # (posteriors.mock_center_co) and draw unshifted.
                    mu_d = posteriors.mock_center_co(pr["center"], pr["sigma_display"], mr["delta"][p])
                    if mu_d is not None:
                        mc = posteriors.ln_gaussian_curve(mu_d, pr["sigma_ln"], bounds=posteriors.co_curve_bounds())
                else:
                    mu_d = pr["center"] + float(mr["delta_display"][p])
                    mc = posteriors.gaussian_curve(mu_d, pr["sigma_display"])
            # the recentred fit keeps the series' plain name: the
            # panel's y axis ("relative density") says it is a fit.
            # mu/sigma make the figure REPORT the width, which the
            # outline cannot (each axis auto-scales to +/-5 sigma);
            # mu is the RECOVERED center for a fit, else the input.
            src, mu = (mc, mu_d) if mc is not None else (pr, pr["center"])
            curves.append(
                dict(
                    label=str(lbl),
                    theta=src["theta"],
                    pdf=src["pdf"],
                    mu=mu,
                    sigma=pr["sigma_display"],
                    sigma_ln=pr["sigma_ln"],
                    width_text=(fisher_mod.format_co_width(mu, pr["sigma_ln"]) if p == "dlnCO" else None),  # type: ignore[arg-type]  # a curve has a center
                    curve_family=pr["curve_family"],
                    color=col,
                    ls="-",
                    lw=1.8,
                    kind=(posteriors.MOCK_RECOVERY_KIND if mc is not None else None),
                )
            )
    return curves


def _forecast_sources(view: RunView, combos: ComboRecords) -> dict:
    """Return the forecast sources: each usable mode, the all-usable combination, and every named combination."""
    # forecast sources: each usable mode, the all-usable combination, and
    # every named combination from the builder above
    sources: dict[str, list] = {}
    for r in view.usable_jac:
        sources[r["label"]] = [r]
    if len(view.usable_jac) >= 2:
        sources[ALL_USABLE] = list(view.usable_jac)
    for rec in combos.recs:
        sources[str(rec["name"])] = [combos.by_mode[k] for k in rec["usable_modes"]]
    return sources


def _series_color(view: RunView, combos: ComboRecords, label: str) -> str:
    """Return one color per series, shared by the spectrum and the forecast panels."""
    if label == ALL_USABLE:
        return "#444444"
    if label in combos.names:
        return COMBO_COLORS[combos.names.index(label) % len(COMBO_COLORS)]
    return ins.MODE_COLOR[next(r["mode_key"] for r in view.results if r["label"] == label)]


def _series_shown(view: RunView, sources: dict) -> dict:
    """Keep the sources the figure's series multiselect shows, so a panel never shows a mode the spectrum does not.

    The multiselect renders later in the script, so this reads its committed state; absent means every source.
    """
    sel_ids = st.session_state.get(K("sum_series"))
    if not sel_ids:
        return sources
    by_key = {r["mode_key"]: r["label"] for r in view.usable_jac}
    want = []
    for i in sel_ids:
        kind, key = i.split(":", 1)
        if kind == "allusable":
            want.append(ALL_USABLE)
        elif kind == "mode" and key in by_key:
            want.append(by_key[key])  # saturated/Jacobian-less
        elif kind == "combo":  # modes drop out here
            want.append(key)
    return {k: v for k, v in sources.items() if k in want}


def forecast_panels(view: RunView, combos: ComboRecords, mock: dict | None, post_slot: Slot | None) -> list:
    """Draw the forecast-panel selector and return the summary figure's panels."""
    post_panels: list[dict] = []
    if not view.have_fisher:
        return post_panels
    sources = _forecast_sources(view, combos)
    if not sources:
        return post_panels
    centers_all = {n: posteriors.param_center(n, view.canon) for n in view.fisher_names}
    pp_key = K("post_params_" + "_".join(view.fisher_names))
    # a stored selection can go stale across runs (different free set)
    if any(p not in view.fisher_names for p in st.session_state.get(pp_key, [])):
        st.session_state.pop(pp_key, None)
    with post_slot:  # type: ignore[union-attr]  # None only for detect runs, which have no fisher_names
        post_sel = st.multiselect(
            "Marginalized parameters",
            view.fisher_names,
            default=view.fisher_names[:2],
            key=pp_key,
            max_selections=summary_figure.MAX_POST_PANELS,
            format_func=lambda n: fisher_mod.PARAM_LABELS[n],
        )
    # record per source (sigmas always; curves for centered params)
    curve_params = [p for p in post_sel if centers_all.get(p) is not None]
    centers = {p: centers_all[p] for p in curve_params}
    recs_by_src = {
        lbl: posteriors.marginalized_posteriors(
            rl, view.fisher_names, centers, params=curve_params, co_eval=view.co_eval
        )
        for lbl, rl in sources.items()
    }
    sources = _series_shown(view, sources)
    labels = list(sources)
    mock_rec = {}
    if mock is not None and post_sel:
        # linearized single-realization recovery on the same stacked system as the forecast
        for lbl in labels:
            mock_rec[lbl] = posteriors.mock_recovery(sources[lbl], view.fisher_names, mock, co_eval=view.co_eval)
    for p in post_sel:
        curves = _panel_curves(p, labels, recs_by_src, mock_rec, lambda lbl: _series_color(view, combos, lbl))
        # A fit to the jitter draw is not a forecast: its center moves with the realization. The y axis says which.
        fitted = any(c.get("kind") == posteriors.MOCK_RECOVERY_KIND for c in curves)
        post_panels.append(
            dict(
                param=str(p),  # the NAME, so the axis-range widget keys stay stable when the selection reorders
                axis_label=fisher_mod.param_axis(p),
                axis_unit=fisher_mod.PARAM_UNITS.get(p, ""),
                density_label=("relative density" if fitted else "relative forecast density"),
                curves=curves,
                center=centers_all.get(p),
            )
        )
    return post_panels
