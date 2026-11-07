"""GUI smoke tests of the result page, on synthetic cached runs. No forward-model run is launched."""

from __future__ import annotations

import re

import numpy as np
import pytest

st = pytest.importorskip("streamlit")
pytest.importorskip("pandas")
from app_helpers import APP, run_app, run_with_result, synthetic_out
from streamlit.testing.v1 import AppTest


def test_results_render_and_below_target_is_warning_not_error(monkeypatch):
    """Full post-Run render path on a synthetic result: every figure and
    table offers a download, and a run that works but finds no signal is a
    scientific outcome, not a software failure -- a warning, never an
    error.

    A download leaves the page behind, so the run caveats must ride the
    bytes: the CSVs carry the full text in their header and every result
    FIGURE carries the short form as a footnote in its own bottom strip --
    a pasted PNG has neither the st.warning nor the CSV header.

    The mixing-ratio panel selects species by NAME: ymix is the FULL network
    state, model["mols"] the short RT list, and zipping them positionally
    labels H2 (0.847 at 1 bar on Tsai et al. 2023's W39b run) as CO2.
    """
    from streamlit.delta_generator import DeltaGenerator

    from jwst_tool import gui_common, plotting, summary_figure as _sf

    downloads, png_downloads, shown = {}, [], []
    original = DeltaGenerator.download_button

    def capture(self, label, data, *args, **kwargs):
        if isinstance(data, bytes) and "CSV" in label:
            downloads[label] = data.decode()
        if isinstance(data, bytes) and "PNG" in label:
            png_downloads.append(data)
        return original(self, label, data, *args, **kwargs)

    monkeypatch.setattr(DeltaGenerator, "download_button", capture)
    monkeypatch.setattr(st, "download_button", capture.__get__(st.download_button.__self__))
    real_image = st.image

    def capture_image(png, **kwargs):
        shown.append((png, kwargs.get("width")))
        return real_image(png, **kwargs)

    monkeypatch.setattr(st, "image", capture_image)

    # The app closes every Figure it builds, so record what each builder
    # DREW and whether it landed on a panel.
    drawn, columns = {}, []

    def _spy(name, real):
        def _f(*a, **kw):
            if name == "structure":  # (p_bar, T_K, columns, ...)
                columns.extend(a[2])
            fig = real(*a, **kw)
            with plotting.render_lock:
                fig.canvas.draw()
                r = fig.canvas.get_renderer()
                boxes = [ax.get_window_extent() for ax in fig.axes]
                drawn[name] = [
                    (t.get_text(), any(t.get_window_extent(r).overlaps(b) for b in boxes)) for t in fig.texts
                ]
            return fig

        return _f

    monkeypatch.setattr(_sf, "compose_summary_figure", _spy("summary", _sf.compose_summary_figure))
    monkeypatch.setattr(plotting, "build_structure_figure", _spy("structure", plotting.build_structure_figure))
    out, out_meta = synthetic_out(with_jac=True)  # sigma_detect=0.0
    # an escalated column must SAY so on the page: the flag is in the artifact
    # precisely because a cache hit never solves and never logs (forward.py)
    out["model"]["photo_escalated"] = np.array(["FD dlnCO row"], dtype="U48")
    out["model"]["unmodeled"] = np.array(["C6H6|1e-5"])
    at = run_with_result(out, out_meta)
    assert not at.exception, at.exception
    assert any("FD dlnCO row" in w.value for w in at.warning), (
        "the photolysis-cadence escalation reached no user-facing surface"
    )
    dl_labels = {b.label for b in at.get("download_button")}
    assert {
        "Figure (PDF, vector)",
        "Figure (PNG)",
        "Binned points (CSV)",
        "Native model (CSV)",
        "T-P values (CSV)",
        "Mixing ratios (CSV)",
        "Mock observation (CSV)",
    } <= dl_labels
    assert any("No signal" in w.value for w in at.warning)
    assert not any("Best mode" in e.value for e in at.error)
    assert "Constraint forecast (CSV)" in downloads
    for label, payload in downloads.items():
        assert "FD dlnCO row" in payload, label
        assert "refreshed every accepted step" in payload, label
        assert "No opacity table for C6H6" in payload, label
    for name in ("summary", "structure"):
        assert len(drawn[name]) == 1, (name, drawn[name])
        text, on_a_panel = drawn[name][0]
        assert "No opacity table for C6H6" in text, (name, text)
        assert "FD dlnCO row" in text, (name, text)
        assert not on_a_panel, f"the {name} caveat footnote covers a panel"
    # a figure displays the very PNG its download carries, at the fixed width: st.pyplot hardcodes 200 dpi, and
    # "stretch" makes the figure wiggle on resize
    assert len(shown) == 2, len(shown)
    for png, width in shown:
        assert png in png_downloads
        assert width == gui_common._FIG_DISPLAY_PX
    # the static table blanks repeated mode names, which a header-click re-sort would misattribute; the CSV is the
    # unblanked rows
    table = next(t.value for t in at.table if "parameter" in getattr(t.value, "columns", []))
    assert "" in {str(m).strip() for m in table["mode"]}
    rows = [ln for ln in downloads["Constraint forecast (CSV)"].splitlines() if ln and not ln.startswith("#")]
    head = rows[0].split(",")
    assert all(r.split(",")[head.index("mode")].strip() for r in rows[1:])
    # the fixture puts H2 (not an RT molecule) first at 0.85; by NAME, not
    # by position
    got = {n: float(np.asarray(y)[0]) for n, y in columns}
    assert set(got) == {"H2O", "CO2", "CO", "CH4", "SO2"}, got
    assert (got["CO"], got["H2O"], got["CO2"]) == pytest.approx((1e-3, 1e-4, 1e-6))


def test_the_rate_caveat_reaches_the_page_and_every_export(monkeypatch):
    """A cache hit never solves and never logs, so the stored counts are the only way a run can say so."""
    from streamlit.delta_generator import DeltaGenerator

    csvs = {}
    original = DeltaGenerator.download_button

    def capture(self, label, data, *args, **kwargs):
        if isinstance(data, bytes) and "CSV" in label:
            csvs[label] = data.decode()
        return original(self, label, data, *args, **kwargs)

    monkeypatch.setattr(DeltaGenerator, "download_button", capture)
    monkeypatch.setattr(st, "download_button", capture.__get__(st.download_button.__self__))
    out, out_meta = synthetic_out(with_jac=True)
    out["model"]["rate_rows_outside"] = np.array([3])
    out["model"]["rate_rows_total"] = np.array([10])
    at = run_with_result(out, out_meta)
    assert not at.exception, at.exception
    assert any("3 of 10 reaction rates" in w.value for w in at.warning)
    assert csvs
    for label, payload in csvs.items():
        assert "3 of 10 reaction rates" in payload, label


def test_emission_results_use_eclipse_terms():
    """An emission run says "eclipse" throughout, never "transit"; an
    above-target result renders NO verdict line and NO banner (maintainer:
    the figure and table carry the number).
    """
    out, out_meta = synthetic_out(science_mode="emission", sigma_detect=8.0, n_transits=3)
    at = run_with_result(out, out_meta)
    assert not at.exception, at.exception
    assert not at.success, f"an above-target result must render no banner: {[s.value for s in at.success]}"
    assert not [m.value for m in at.markdown if "template S/N 8.0" in m.value]
    # the figure section is an EXPANDER, not a subheader
    _exps = [e.label for e in at.get("expander")]
    assert any("eclipse emission spectrum" in e for e in _exps), _exps
    assert not any(e == "Proposal summary figure" for e in _exps)
    # scope to the RESULTS text: the sidebar/intro copy legitimately says
    # "transit" for the general case
    results = " ".join(
        [w.value for w in at.warning]
        + [e.value for e in at.error]
        + [c.value for c in at.get("caption") if "eclipse" in c.value.lower() or "transit" in c.value.lower()]
        + [s.value for s in at.subheader]
        + _exps
    )
    assert "eclipse" in results.lower(), results[:300]
    assert not re.search(r"\b\d+ transits?\b", results), results[:300]


def test_all_saturated_run_has_no_best_mode_score_or_points():
    """All modes saturated: no "best" mode, no error alert for a valid
    calculation -- and no detection score or simulated points in the figure,
    matching the exclusion every ranking, combination and forecast already
    applies. Saturation shows as the absence of a ranked mode; the page says
    nothing about it.
    """
    import jwst_tool.summary_figure as sf

    seen = {}
    _real = sf.compose_summary_figure

    def _spy(spectrum, **kw):
        seen["labels"] = [p["label"] for p in (spectrum.get("points") or [])]
        return _real(spectrum, **kw)

    out, out_meta = synthetic_out(saturated=True, sigma_detect=5.0)
    sf.compose_summary_figure = _spy
    try:
        at = run_with_result(out, out_meta)
    finally:
        sf.compose_summary_figure = _real
    assert not at.exception, at.exception
    warns = [w.value for w in at.warning]
    assert not at.success
    assert not any("Best mode" in w for w in warns)
    assert not any("Best mode" in e.value for e in at.error)
    # no plotted series at all, in particular no "<mode>: 5.0σ" legend entry
    assert seen.get("labels") == [], seen


def test_custom_archive_fill_and_uv_menu_never_moves():
    """Custom planet, one boot: a typed Teff never moves the UV MENU, and
    the archive Fill button writes the
    snapshot row into the form (pending-then-apply path) -- still without
    touching the UV menu (no substitute spectra, standing maintainer
    rule).
    """
    at = run_app()
    at.selectbox(key="n0_planet").set_value("custom").run()
    assert not at.exception, at.exception
    at.number_input(key="n0_custom_teff").set_value(3100.0).run()
    assert not at.exception, at.exception
    assert at.selectbox(key="n0_custom_sflux").value == "sflux-W39b_Tsai2023.txt"

    at.selectbox(key="n0_custom_arch_name").set_value("HD 189733 b").run()
    at.button(key="n0_custom_arch_fill").click().run()
    assert not at.exception, at.exception
    from jwst_tool import archive

    values = archive.custom_fill(archive.lookup("HD 189733 b"))
    assert at.number_input(key="n0_custom_teff").value == values["teff"]
    assert at.number_input(key="n0_custom_g").value == pytest.approx(values["g"])
    # the UV menu is NEVER written by the fill
    assert at.selectbox(key="n0_custom_sflux").value == "sflux-W39b_Tsai2023.txt"


def test_combo_builder_fisher_table_naming_and_reset():
    """One booted app with Jacobians carries the whole combo story: every
    results section is an expander (renames pinned); a primed
    combination LEADS the STATIC constraint table under the user's own name
    ('All usable modes', no 'COMBO: ' prefix; st.table so a header click
    cannot detach the blanked mode names); the builder adds and removes
    combos; the confirm-step reset clears the combo/posterior session keys
    (nonce-namespaced keys + un-namespaced pending notes).
    """
    import pandas as pd

    primed = dict(name="My set", modes=["nirspec_g395h", "nirspec_prism"])
    out, out_meta = synthetic_out(sigma_detect=8.0, with_jac=True)
    at = run_with_result(out, out_meta, n0_combos=[dict(primed)])
    assert not at.exception, at.exception
    subs = [s.value for s in at.subheader]
    exps = [e.label for e in at.get("expander")]
    # the combo builder and the marginalized-forecast control live in the
    # constraint-forecast panel; neither is a section of its own
    assert "Add a custom mode set" not in exps, exps
    assert "Mode combinations" not in exps, exps
    assert "Marginalized Fisher forecasts" not in exps, exps
    assert "Marginalized Fisher forecasts" not in subs, subs
    assert "Mode details" not in exps, exps
    assert "Run summary & configuration" not in exps, exps
    assert "Parameter constraint forecast (local Fisher)" in exps, exps
    assert "Physical structure (T-P profile, mixing ratios)" in exps, exps
    assert any("forecast summary" in e for e in exps), exps
    assert not any("forecast summary" in s for s in subs), subs
    assert "Figure (PDF, vector)" in {b.label for b in at.get("download_button")}

    # identify the CONSTRAINT table by its parameter column
    tables = [
        t.value
        for t in at.table
        if "mode" in getattr(t.value, "columns", []) and "parameter" in getattr(t.value, "columns", [])
    ]
    assert tables, "no constraint-forecast table rendered"
    assert not [df.value for df in at.dataframe if "parameter" in getattr(df.value, "columns", [])], (
        "the constraint forecast is an interactive dataframe again"
    )
    modes = [str(v) for v in tables[0]["mode"]]
    assert not any("ALL USABLE" in m for m in modes), modes[:8]
    assert not any(m.startswith("COMBO:") for m in modes), modes[:8]
    assert any("All usable modes" == m for m in modes), modes[:8]
    first = next(m for m in modes if m.strip())
    assert "My set" in first, f"combos should lead the table, got {first!r}"
    # C/O is reported in log10(C/O), the coordinate the Fisher Gaussian is
    # symmetric in: the row names it and each marginalized cell is the value
    # and its error there (a 3-sigma cell in linear C/O can cross zero)
    df = tables[0]
    col = next(c for c in df.columns if str(c).startswith("marginalized"))
    cells = [str(r[col]).strip() for _, r in df.iterrows() if str(r["parameter"]) == "log10 C/O"]
    cells = [c for c in cells if c not in ("unconstrained", "")]
    assert cells, "no constrained C/O row to check"
    for cell in cells:
        assert "log10" not in cell and "(C/O" not in cell, cell
        centre, width = cell.split("±")
        assert float(centre) == pytest.approx(np.log10(0.55), abs=5e-3), cell
        assert float(width) > 0.0, cell

    # add a named combination through the builder widgets
    at.text_input(key="n0_cb_name").set_value("PRISM + G395H")
    at.multiselect(key="n0_cb_modes").set_value(["nirspec_prism", "nirspec_g395h"])
    at.button(key="n0_cb_add").click().run()
    assert not at.exception, at.exception
    added = dict(name="PRISM + G395H", modes=["nirspec_prism", "nirspec_g395h"])
    combos = at.session_state["n0_combos"]
    assert added in combos and primed in combos and len(combos) == 2, combos
    assert "PRISM + G395H" in " ".join(m.value for m in at.markdown)
    tables = [pd.DataFrame(d.value) for d in list(at.get("table")) + list(at.get("dataframe"))]
    assert any(t.astype(str).apply(lambda c: c.str.contains("PRISM \\+ G395H")).any().any() for t in tables), (
        "combo rows missing from the Fisher table"
    )
    # a name that would replace another series is refused
    at.text_input(key="n0_cb_name").set_value("All usable modes")
    at.button(key="n0_cb_add").click().run()
    assert at.session_state["n0_combos"] == combos
    assert any("already taken" in e.value for e in at.error)
    # remove works (index 0 = the first listed combo)
    removed_first = combos[0]
    at.button(key="n0_cb_rm_0").click().run()
    assert not at.exception, at.exception
    assert at.session_state["n0_combos"] == [c for c in combos if c != removed_first]

    # confirm-step reset (labels, not keys: the reset buttons are keyless)
    next(b for b in at.button if (b.label or "") == "Reset all settings").click().run()
    next(b for b in at.button if (b.label or "") == "Confirm reset").click().run()
    assert not at.exception, at.exception
    assert "n0_combos" not in at.session_state
    assert "_combo_note" not in at.session_state


def test_emission_mode_archive_fill_skips_transit_duration():
    """The archive duration is the PRIMARY-TRANSIT duration; in emission
    mode the fill leaves the event-duration widget alone, silently.
    """
    at = run_app()
    at.selectbox(key="n0_planet").set_value("custom").run()
    at.radio(key="n0_scimode").set_value("emission").run()
    t14_before = at.number_input(key="n0_custom_t14").value
    at.selectbox(key="n0_custom_arch_name").set_value("HD 189733 b").run()
    at.button(key="n0_custom_arch_fill").click().run()
    assert not at.exception, at.exception
    from jwst_tool import archive

    values = archive.custom_fill(archive.lookup("HD 189733 b"))
    assert at.number_input(key="n0_custom_t14").value == t14_before
    assert at.number_input(key="n0_custom_teff").value == values["teff"]


@pytest.mark.parametrize(
    ("widget", "value", "field"),
    # the molecule selectbox keys on the extra-molecule selection, so the key
    # carries the default extra set (jwst_tool.app: K("mol_<provider>_<mols>"));
    # CS2/C2H6 are out of the default (no ExoMolOP table); SH+SO are in
    # (measured 10.0 and 48.1 ppm on W39b -- forward.EXTRA_MOLECULES_DEFAULT)
    [
        ("n0_mol_vulcan_sncho2025_C2H2_C2H4_H2S_HCN_NH3_OCS_SH_SO", "CO2", "target_mol"),  # not a canonical param
        ("n0_noisescale", 2.0, "noise_scale"),  # an observation-block field
        ("n0_provider", "picaso", "chem_provider"),  # the chemistry engine
        # DISPLAY-ONLY: the seed redraws the mock from the cached run and
        # recomputes nothing, so it must never read as stale (field=None)
        ("n0_seed", 7, None),
    ],
)
def test_changing_any_run_input_marks_the_result_stale(widget, value, field, monkeypatch):
    """The staleness guard compares the WHOLE non-canonical input set minus
    the display-only fields, not a hand-picked subset. The detection target
    is absent from the canonical model params, so without the guard switching
    SO2 to CO2 without pressing Run leaves the curve, legend and CSV on the
    previous target with no notice. The noise scale represents the
    observation block because a subset guard misses the next such field; the
    seed represents the display-only fields, which change nothing that was
    computed.
    """
    from jwst_tool import share_config as _sc

    seen = {}
    _real = _sc.build_share

    def _spy(*, canon, goal, observation, **kw):
        # the app hands the same two blocks to the shareable config and to the
        # staleness guard, minus the display-only observation fields, which the guard never compares
        seen["run_sig"] = dict(
            goal=goal, observation={k: v for k, v in observation.items() if k not in ("show_noise", "seed", "combos")}
        )
        seen["canon"] = canon
        return _real(canon=canon, goal=goal, observation=observation, **kw)

    monkeypatch.setattr(_sc, "build_share", _spy)
    out, out_meta = synthetic_out(sigma_detect=8.0, with_jac=True)
    at = run_with_result(out, out_meta)
    assert not at.exception, at.exception
    # Adopt the page's OWN canonical params and run signature as the cached
    # run's, so the ONLY difference below is the widget under test.
    import json as _json

    at.session_state["out"] = dict(out, model=dict(out["model"], params_json=_json.dumps(seen["canon"], default=str)))
    at.session_state["out_meta"] = dict(out_meta, run_sig=seen["run_sig"])
    at.run()

    def _named():
        return [w.value for w in at.warning if "previous run" in w.value]

    assert not _named(), f"an unchanged sidebar must not read as stale: {_named()}"
    at.session_state[widget] = value
    at.run()
    assert not at.exception, at.exception
    if field is None:
        assert not _named(), f"a display-only input must not mark the run stale: {_named()}"
    else:
        assert _named() and field in _named()[0], _named()


def test_a_run_that_produces_nothing_clears_the_previous_result(monkeypatch):
    """A refused or failed Run must not leave the previous run's verdict and
    figures on screen under a stale banner: the page shows only the error.
    Exercised through the one refusal that keeps the Run button enabled (a
    cache MISS with no free concurrency slot); the forward-model failure
    branch pops the same keys.
    """
    from jwst_tool import model_cache, runlimit as _rl

    monkeypatch.setattr(_rl, "acquire", lambda *_a, **_k: None)
    # the miss is pinned here: only a heavy launch is declined now, so the
    # refusal must not depend on what this machine happens to have cached
    monkeypatch.setattr(model_cache, "load_result", lambda *_a, **_k: None)
    out, out_meta = synthetic_out(sigma_detect=8.0, with_jac=True)
    at = run_with_result(out, out_meta)
    assert not at.exception, at.exception
    next(b for b in at.button if b.label == "Run").click()
    at.run()
    assert not at.exception, at.exception
    assert "out" not in at.session_state
    assert any("already running" in e.value for e in at.error), [e.value for e in at.error]
    assert not [w.value for w in at.warning if "previous run" in w.value or "reach it" in w.value]


def test_a_fully_cached_run_renders_while_every_slot_is_busy(monkeypatch):
    """The refusal promises previously computed results stay instant, so a
    Run whose model AND ETC entries are all cached -- it starts no
    subprocess -- must render with every heavy slot held. The slot is taken
    at the launch, not at the click, and the second half pins the other side:
    a cache MISS on the same busy instance is still refused, so the limiter
    (never the cache) gates every heavy call.
    """
    from jwst_tool import detect as _detect, model_cache, noise as _noise, runlimit as _rl

    out, _ = synthetic_out(sigma_detect=8.0, with_jac=True)
    by_mode = {r["mode_key"]: r for r in out["results"]}
    monkeypatch.setattr(_rl, "acquire", lambda *_a, **_k: None)  # slots held
    monkeypatch.setattr(model_cache, "load_result", lambda *_a, **_k: out["model"])
    monkeypatch.setattr(_noise, "missing_modes", lambda _star, keys, **_k: [k for k in keys if k not in by_mode])
    monkeypatch.setattr(
        _noise, "run_modes", lambda _star, keys, **_k: {**{k: {"wl": [1.0]} for k in keys}, "__provenance__": None}
    )
    monkeypatch.setattr(_detect, "evaluate_mode", lambda k, *_a, **_kw: by_mode[k])

    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    at.multiselect(key="n0_modes").set_value(list(by_mode)).run()
    assert not at.exception, at.exception
    next(b for b in at.button if b.label == "Run").click()
    at.run()
    assert not at.exception, at.exception
    assert not [e.value for e in at.error if "already running" in e.value]
    assert {r["mode_key"] for r in at.session_state["out"]["results"]} == set(by_mode)

    monkeypatch.setattr(model_cache, "load_result", lambda *_a, **_k: None)
    next(b for b in at.button if b.label == "Run").click()
    at.run()
    assert not at.exception, at.exception
    assert any("already running" in e.value for e in at.error), [e.value for e in at.error]
    assert "out" not in at.session_state

    # an ETC entry that vanished between the cache probe and the read: the
    # model is cached, missing_modes says nothing is missing, run_modes finds
    # a miss -- still refused, never launched without a slot
    monkeypatch.setattr(model_cache, "load_result", lambda *_a, **_k: out["model"])
    monkeypatch.setattr(_noise, "run_modes", lambda _s, _keys, launch=True, **_k: None if not launch else {})
    next(b for b in at.button if b.label == "Run").click()
    at.run()
    assert not at.exception, at.exception
    assert any("already running" in e.value for e in at.error)
    assert "out" not in at.session_state


def test_a_nonpositive_bound_on_a_log_axis_warns_instead_of_killing_the_page():
    """A min at or below zero has no logarithm on a log axis. The builders
    raise on it -- right for an API backstop, but uncaught it takes the
    ENTIRE results page down. A typed number is a user choice, not a defect:
    warn, fall back to the automatic fit, keep the page alive.

    Only the summary figure carries axis controls; the structure figure has
    none (it draws at the module defaults), so there is no typed bound to
    poison there.
    """
    key, label = "n0_sum_x", "Wavelength"
    out, out_meta = synthetic_out(sigma_detect=8.0, with_jac=True)
    at = run_with_result(out, out_meta, **{f"{key}_min": 0.0, f"{key}_max": 5.0})
    assert not at.exception, at.exception
    assert any("log axis" in w.value and label in w.value for w in at.warning), [w.value for w in at.warning]


def test_axis_bounds_refuse_one_sided_and_reach_the_figure(monkeypatch):
    """One bound alone has no second edge: the page must SAY it fell back
    rather than invent the other bound -- and a COMPLETE pair must actually
    reach the figure (the reviewer's 3.0 to 5.5 um zoom), not merely be
    accepted by the widget. The app closes every Figure it builds, so this
    records the ARGUMENTS of the call the page makes.
    """
    from jwst_tool import summary_figure as _sf

    seen = {}
    _real = _sf.compose_summary_figure

    def _spy(spectrum, **kw):
        seen["wl_range"] = spectrum.get("wl_range")
        seen["depth_range"] = spectrum.get("depth_range")
        seen["panel_xlims"] = kw.get("panel_xlims")
        return _real(spectrum, **kw)

    monkeypatch.setattr(_sf, "compose_summary_figure", _spy)
    out, out_meta = synthetic_out(sigma_detect=8.0, with_jac=True)
    at = run_with_result(out, out_meta, n0_sum_y_min=20500.0)  # max left blank
    assert not at.exception, at.exception
    # Axis-control contract (maintainer): every axis is a
    # typed min/max number-box pair that STARTS blank, no sliders anywhere,
    # no wavelength-range radio.
    assert not [r for r in at.radio if r.key == "n0_sum_wlmode"]
    assert not at.slider, [s.key for s in at.slider]
    for _k in ("sum_x", "sum_post_lnZ"):
        for _end in ("min", "max"):
            assert at.number_input(key=f"n0_{_k}_{_end}").value is None, f"{_k}_{_end} should start blank"
    assert any("both boxes" in w.value for w in at.warning), [w.value for w in at.warning]
    # the COMPLETE pairs are accepted silently (so the warning above is a
    # real refusal) and the typed windows reach the figure call
    at.session_state["n0_sum_y_max"] = 21500.0
    at.session_state["n0_sum_x_min"] = 3.0
    at.session_state["n0_sum_x_max"] = 5.5
    at.session_state["n0_sum_post_lnZ_min"] = -0.4
    at.session_state["n0_sum_post_lnZ_max"] = 0.4
    at.run()
    assert not at.exception, at.exception
    assert not [w for w in at.warning if "Depth range" in w.value], [w.value for w in at.warning]
    assert seen["wl_range"] == pytest.approx((3.0, 5.5))
    assert seen["depth_range"] == pytest.approx((20500.0, 21500.0))
    assert seen["panel_xlims"][0] == pytest.approx((-0.4, 0.4))


def _render_order(at):
    """Main-area elements in render order (AppTest exposes children per
    block, so the tree has to be flattened to compare positions).
    """

    def walk(node, out):
        ch = getattr(node, "children", None)
        for c in ch.values() if isinstance(ch, dict) else (ch or []):
            out.append(c)
            walk(c, out)
        return out

    return walk(at.main, [])


def test_the_verdict_renders_above_the_explainers():
    """The shortfall line is the ANSWER -- which mode, what significance,
    against the target -- so it renders with the Run button that produced it,
    not hundreds of lines down beside the figure it summarizes. Anything
    below Validation is reference material the user reads once.
    """
    at = run_with_result(*synthetic_out())
    assert not at.exception, at.exception
    order = _render_order(at)
    verdict = next(
        i
        for i, e in enumerate(order)
        if type(e).__name__ == "Warning" and "template S/N" in str(getattr(e, "value", ""))
    )
    validation = next(
        i for i, e in enumerate(order) if type(e).__name__ == "Expander" and getattr(e, "label", "") == "Validation"
    )
    assert verdict < validation, f"the verdict renders at {verdict}, below Validation at {validation}"


@pytest.mark.parametrize("case", ["all_unusable", "mixed", "saturated_only"])
def test_exclusions_are_reported_once_even_without_usable_results(case):
    from jwst_tool import instruments as ins

    out, meta = synthetic_out(saturated=case == "saturated_only")
    if case != "saturated_only":
        out["unusable"] = [("nirspec_prism", "no usable pixels")]
    if case == "all_unusable":
        out["results"] = []
    if case == "saturated_only":
        out["results"][0]["warnings"] = {"full_saturated": "duplicate risk note"}
    at = run_with_result(out, meta)
    assert not at.exception, at.exception
    excluded = [w.value for w in at.warning if "excluded from every ranking" in w.value]
    assert len(excluded) == 1
    key = "nirspec_g395h" if case == "saturated_only" else "nirspec_prism"
    assert f"{ins.MODES[key]['label']} ({ins.config_label(key)})" in excluded[0]
    assert ("saturated" if case == "saturated_only" else "no usable pixels") in excluded[0]
    assert not any("duplicate risk note" in w.value for w in at.warning)
    if case == "mixed":
        assert any(b.label == "Binned points (CSV)" for b in at.get("download_button"))


def test_combined_forecast_is_reported_when_no_single_mode_constrains():
    """Complementary degeneracies: each mode alone puts the goal parameter on
    a null direction (lnZ = +dlnCO in one, -dlnCO in the other), the two
    jointly do not. The joint width IS the answer, so it is reported and the
    page keeps rendering -- an empty per-mode set must not st.stop() and take
    the structure panel, the constraint forecast and the figure with it. The
    verdict is a SHORTFALL warning: it speaks for a joint width that misses
    the target and stays silent for one that meets it.
    """
    out, out_meta = synthetic_out(sigma_detect=8.0, with_jac=True)
    nb = out["results"][0]["jac_bins"].shape[1]
    ramp = np.linspace(1.0, 2.0, nb)
    quad = np.linspace(2.0, 1.0, nb) ** 2
    r0 = np.cos(np.linspace(0.0, 2.0, nb))  # the shared lnR0 row
    # amplitude scaled so the joint width lands at ~0.2 dex, i.e. WIDE of the
    # 0.1 dex target below; the null directions are unchanged by a scale
    out["results"][0]["jac_bins"] = np.vstack([ramp, ramp, r0]) * 1.0e-3
    out["results"][1]["jac_bins"] = np.vstack([quad, -quad, r0]) * 1.0e-3
    out_meta.update(goal="constrain", target=None, goal_param="lnZ", target_prec=0.1)
    at = run_with_result(out, out_meta)
    assert not at.exception, at.exception
    warns = [w.value for w in at.warning]
    hit = [w for w in warns if "No single mode constrains" in w]
    assert len(hit) == 1, warns
    assert "Metallicity" in hit[0] and "Combined modes: ±" in hit[0], hit[0]
    assert "dex at 3σ in 1 transit (target ±0.1 dex)" in hit[0], hit[0]
    exps = [e.label for e in at.get("expander")]
    assert "Parameter constraint forecast (local Fisher)" in exps, exps
    assert "Physical structure (T-P profile, mixing ratios)" in exps, exps
    # same run, a target the joint width meets: no shortfall, no warning
    out_meta.update(target_prec=0.5)
    at = run_with_result(out, out_meta)
    assert not at.exception, at.exception
    assert not any("No single mode constrains" in w.value for w in at.warning)
