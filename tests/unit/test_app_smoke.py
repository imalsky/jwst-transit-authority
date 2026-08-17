"""GUI smoke tests: the app renders end-to-end with no exception.

Needs the GUI extras (streamlit + pandas); the dependency-light CI skips it.
Uses Streamlit's AppTest; no forward-model run is launched. Pruned 2026-08-15
(maintainer: fewer, stronger tests): AppTest boots are the expensive part, so
every assertion that examines the same booted app lives in one test.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

import pytest

st = pytest.importorskip("streamlit")
pytest.importorskip("pandas")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "src" / "jwst_tool" / "app.py"


def _run_app():
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    return at


def _synthetic_out(science_mode="transmission", saturated=False,
                   sigma_detect=0.0, n_transits=1, with_jac=False):
    """Minimal cached-result pair (out, out_meta) for post-run rendering.

    sigma_detect must stay 0 or above target so the render never calls
    detect.transits_to_target, which needs the full evaluate_mode payload.
    ``with_jac=True`` adds a two-parameter Jacobian (lnZ, dlnCO + lnR0) and a
    SECOND mode, so the Fisher table, combo builder, forecast posteriors and
    summary figure all render. ymix is the FULL network state, wider than the
    RT molecule list and in a different order -- the shape that exposed the
    mixing-ratio mislabelling; it also drives the one log-x panel.
    """
    import json

    n = 40
    wl = np.linspace(1.0, 5.0, n)
    model = {
        "wl_um": wl,
        "depth": np.full(n, 0.021) + 1e-4 * np.sin(wl),
        "mols": np.array(["H2O", "CO2", "CO", "CH4", "SO2"], dtype="U8"),
        # depth_wo rows align with wo_mols, not mols (v32)
        "wo_mols": np.array(["H2O", "CO2", "CO", "CH4", "SO2"], dtype="U8"),
        "depth_wo": np.tile(np.full(n, 0.0208), (5, 1)),
        "T": np.full(30, 1100.0),
        "p_bar": np.logspace(-7, 0.8, 30),
        "ymix": np.tile(np.array([0.85, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 0.14]),
                        (30, 1)),
        "ymix_species": np.array(["H2", "CO", "H2O", "CH4", "CO2", "SO2", "He"],
                                 dtype="U16"),
        "science_mode": science_mode,
        "params_json": json.dumps({"co_ratio": 0.549348,
                                   "tp_mode": "guillot",
                                   "met_x_solar": 10.0,
                                   "science_mode": science_mode}),
    }
    nb = 12
    rng = np.random.default_rng(0)

    def _mode(key, label, wl_lo, wl_hi):
        return {
            "mode_key": key, "label": label,
            "saturated": saturated, "sat_frac": 0.97,
            "sigma_detect": sigma_detect,
            "sigma_detect_proj": float("nan"),
            "wl": np.linspace(wl_lo, wl_hi, nb),
            "wl_eff": np.linspace(wl_lo, wl_hi, nb),
            "depth": np.full(nb, 0.021), "sigma": np.full(nb, 1.5e-4),
            "floor": np.zeros(nb), "seg": np.zeros(nb, int),
            "median_sigma_ppm": 150.0, "n_bins": nb, "ngroup": 12,
            "t_cycle_s": 11.0, "warnings": (),
            "jac_bins": (rng.standard_normal((3, nb)) * 1e-4
                         if with_jac else None),
        }

    results = [_mode("nirspec_g395h", "NIRSpec G395H", 2.9, 5.1)]
    fisher_names = []
    if with_jac:
        results.append(_mode("nirspec_prism", "NIRSpec PRISM", 0.7, 5.2))
        model["jac"] = np.zeros((3, n))
        model["jac_names"] = np.array(["lnZ", "dlnCO", "lnR0"], dtype="U8")
        fisher_names = ["lnZ", "dlnCO"]
    out = dict(model=model, results=results, failed=[], unusable=[],
               fisher_names=fisher_names, provenance=None)
    out_meta = dict(
        goal="detect", target="SO2", goal_param=None, target_prec=None,
        target_sig=3.0, n_transits=n_transits, show_noise=False, seed=0,
        r_bin=100, planet="WASP-39 b",
        floor_mode="constant")
    return out, out_meta


def _run_with_result(out, out_meta, **state):
    """AppTest primed with a cached result (plus optional session state), run
    once. Post-run rendering only; no forward model is launched."""
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["out"] = out
    at.session_state["out_meta"] = out_meta
    for key, val in state.items():
        at.session_state[key] = val
    at.run()
    return at


def _deferred_labels(at):
    """(live download buttons, dead busy stand-ins) among the deferred set."""
    deferred = {"Download configuration (JSON)",
                "Download this example (edit and re-upload)"}
    live = {b.label for b in at.get("download_button")} & deferred
    dead = {b.label for b in at.get("button")} & deferred
    return live, dead


def test_fresh_boot_pre_run_contract():
    """One default boot: no intro gate, the 0.27.0 speed-first mode trio (the
    ETC computes only the selected modes), the data-status panel + annotated
    molecules, noise multiplier at 1.0, and the mock controls -- the 'New
    draw' button was removed 2026-08-13 but the seed field keeps a
    realization reproducible; widget keys unchanged so share_config
    round-trips."""
    from jwst_tool import instruments as ins

    at = _run_app()
    assert not at.exception, at.exception
    assert at.selectbox(key="n0_planet").value is not None
    assert not [b for b in at.button if "I understand" in (b.label or "")]
    assert set(at.multiselect(key="n0_modes").value) == set(ins.DEFAULT_MODES)
    assert set(ins.DEFAULT_MODES) == {"nirspec_prism", "nirspec_g395h",
                                      "miri_lrs"}
    labels = " ".join(e.label for e in at.expander)
    assert "Data status" in labels
    ms = [w for w in at.multiselect if w.label == "Extra opacity molecules"]
    assert ms, "extra-molecule multiselect missing"
    # (the molecule_linelist_status contract itself is pinned in
    # test_datacheck.py; here only the annotated widget's presence matters)
    sld = at.number_input(key="n0_noisescale")
    assert sld.value == 1.0 and sld.label == "Noise multiplier", \
        (sld.label, sld.value)
    assert at.checkbox(key="n0_shownoise").value is True   # ON by default
    assert at.number_input(key="n0_seed").value == 0
    assert not [b for b in at.get("button") if b.key == "n0_reroll"], \
        "the 'New draw' button was removed (maintainer, 2026-08-13)"
    # editing the seed still redraws without recomputing
    at.number_input(key="n0_seed").set_value(7).run()
    assert not at.exception, at.exception
    assert at.number_input(key="n0_seed").value == 7


def test_gui_structure_defaults_match_canonical_params():
    """GUI and API defaults must mean the same atmosphere on EVERY planet
    (a hard-coded API T_irr once drifted 470 K from the widget on HD 209458 b).
    """
    from jwst_tool import forward, planets

    at = _run_app()
    assert not at.exception, at.exception
    for key in planets.PLANETS:
        at.selectbox(key="n0_planet").set_value(key).run()
        assert not at.exception, (key, at.exception)
        cp = forward.canonical_params(dict(planet=key))
        assert at.selectbox(key=f"n0_{key}_tp").value == cp["tp_mode"], key
        if cp["tp_mode"] == "file":
            assert at.selectbox(key=f"n0_{key}_kzzmode").value == \
                cp["kzz_mode"] == "file", key
        # Force Guillot so the T_irr widget renders; compare against
        # canonical_params, never the helper the widget itself calls.
        at.selectbox(key=f"n0_{key}_tp").set_value("guillot").run()
        assert not at.exception, (key, at.exception)
        api_tirr = forward.canonical_params(
            dict(planet=key, tp_mode="guillot"))["Tirr"]
        assert at.number_input(key=f"n0_{key}_tirr").value == api_tirr, key
        at.selectbox(key=f"n0_{key}_tp").set_value(cp["tp_mode"]).run()


def test_deferred_downloads_are_live_outside_a_run():
    """Clicking a download widget mid-run cancels the run (the
    ScriptControlException class, fixed 0.21.2, 2026-08-04), so the config
    JSON and the sidebar T-P example downloads render through deferred
    placeholder slots. Outside a run both must be LIVE download buttons,
    never dead busy stand-ins."""
    at = _run_app()
    assert not at.exception, at.exception
    live, dead = _deferred_labels(at)
    assert "Download configuration (JSON)" in live
    assert not dead, "a busy stand-in must not render outside a run"

    at2 = AppTest.from_file(str(APP), default_timeout=60)
    at2.session_state["n0_wasp39b_tp"] = "file"
    at2.session_state["n0_wasp39b_tpsrc"] = "upload"   # forward.TP_FILE_UPLOAD
    at2.run()
    assert not at2.exception, at2.exception
    live, dead = _deferred_labels(at2)
    assert "Download this example (edit and re-upload)" in live
    assert not dead


def test_sidebar_gating_geometry_boxes_ad_lock_and_floor():
    """One fresh boot, three gating behaviors.

    1. Each geometry keeps its OWN column-bottom box (shipped key contract),
       and the DEFAULT is structure-aware (v32): W39b transmission defaults
       to its measured table's own bottom, emission (Guillot everywhere)
       to the round parametric default.
    2. AD under a constrain goal forces the photochemistry checkbox ON and
       disabled; switching back to detect releases it (the photo widget
       renders before the method menu, so the lock reads session state).
    3. The noise floor defaults to "constant" (maintainer, 2026-08-13) -- the
       CONSERVATIVE side, since it claims LESS precision than "No floor";
       "none" stays an explicit choice; "Wavelength table" with no upload is
       the one state that still BLOCKS the run.
    """
    from jwst_tool import forward

    at = _run_app()
    assert not at.exception, at.exception
    # 1. per-geometry column bottoms, structure-aware defaults
    assert at.number_input(key="n0_pbtm_transmission").value == \
        forward.P_BTM_FILE_BAR
    at.radio(key="n0_scimode").set_value("emission").run()
    assert not at.exception, at.exception
    assert at.number_input(key="n0_pbtm_emission").value == \
        forward.P_BTM_PARAMETRIC_BAR
    at.radio(key="n0_scimode").set_value("transmission").run()
    assert at.number_input(key="n0_pbtm_transmission").value == \
        forward.P_BTM_FILE_BAR

    # 2. AD photo-lock
    at.radio(key="n0_goal").set_value("constrain").run()
    at.selectbox(key="n0_jacm").set_value("ad").run()
    assert not at.exception, at.exception
    photo = at.checkbox(key="n0_photo")
    assert photo.value is True and photo.disabled
    at.radio(key="n0_goal").set_value("detect").run()
    assert not at.exception, at.exception
    assert not at.checkbox(key="n0_photo").disabled

    # 3. noise-floor gate
    def _floor_error(app):
        return [e.value for e in app.error
                if "minimum noise floor" in (e.value or "").lower()]

    assert at.radio(key="n0_floormode").value == "constant"
    assert not _floor_error(at), [e.value for e in at.error]
    assert any((w.key or "").startswith("n0_floor_")
               for w in at.get("number_input")), "per-mode floor inputs gone"
    at.radio(key="n0_floormode").set_value("none").run()
    assert not at.exception, at.exception
    assert not _floor_error(at)
    at.radio(key="n0_floormode").set_value("file").run()
    assert not at.exception, at.exception
    assert _floor_error(at), [e.value for e in at.error]


def test_source_pins_fig_width_fisher_table_and_noise_recording():
    """Source-level pins for three 2026-08-13 maintainer decisions: (1) the
    global noise multiplier COMPOSES with the per-mode multipliers and the
    config records the two factors APART (the product would re-multiply on
    restore); (2) every figure gets an explicit pixel width -- st.pyplot
    defaults to "stretch", so omitting the argument does not stop the resize
    wiggle, and the constant must be defined before first use; (3) the Fisher
    table is st.table, never st.dataframe -- its mode column is BLANKED on
    repeat rows, so a header-click re-sort would attribute numbers to the
    wrong instrument, while the CSV is built from the UNBLANKED rows."""
    src = APP.read_text()
    assert "infl = {k: float(noise_scale) * float(_infl_mode[k])" in src, \
        "the global multiplier no longer composes with the per-mode ones"
    assert "noise_infl={k: float(_infl_mode[k]) for k in mode_keys}" in src, \
        "noise_infl must record the PER-MODE widget values, not the product"
    assert "noise_scale=float(noise_scale)," in src, \
        "the global scale is not recorded, so a run cannot be reproduced"
    assert "st.pyplot(fig, width=_FIG_DISPLAY_PX)" in src, \
        "the tight branch no longer pins the display width"
    assert "st.pyplot(fig, width=_FIG_DISPLAY_PX,\n" in src, \
        "the full-canvas branch no longer pins the display width"
    assert 'st.pyplot(fig, width="stretch"' not in src
    lines = src.splitlines()
    define = next(i for i, l in enumerate(lines)
                  if l.startswith("_FIG_DISPLAY_PX"))
    use = next(i for i, l in enumerate(lines)
               if "st.pyplot(fig, width=_FIG_DISPLAY_PX)" in l)
    assert define < use, (define, use)
    i = src.index('with st.expander("Parameter constraint forecast (Fisher)")')
    j = src.index('st.download_button("Constraint forecast (CSV)"', i)
    block = src[i:j]
    assert "st.table(" in block and "st.dataframe(" not in block, \
        "the Fisher table must stay a static st.table (blanked mode names)"
    assert '_r2["mode"] = ""' in block, \
        "mode names are no longer blanked; re-check whether st.table is needed"
    assert "_csv_bytes(pd.DataFrame(frows))" in src, \
        "the Fisher CSV is no longer built from the unblanked rows"


def test_removed_gui_prose_sections_and_tooltips_stay_gone():
    """2026-08-13 GUI cleanup (maintainer): the removal IS the requirement,
    so an edit that restores any of it fails here. Gone: the Condensation and
    boundary-condition expanders (and their widgets), step-3 and
    More-settings tooltips, the table guidance, the long-form intro and the
    honesty paragraphs. Displaced content lives in README.md (GUI prose
    policy). Kept: engine names and the LIVE, interpolated Pandeia version
    (the backend-label rule)."""
    at = _run_app()
    assert not at.exception, at.exception
    labels = [e.label for e in at.get("expander")]
    for gone in ("Condensation (detection goals only)",
                 "Boundary conditions & escape"):
        assert gone not in labels, f"{gone!r} expander is back"
    keys = {w.key for w in at.get("checkbox")} | {
        w.key for w in at.get("multiselect")}
    assert "n0_conden" not in keys and "n0_settle" not in keys
    assert "n0_descape" not in keys
    # n0_seed joined the list in the 2026-08-16 trim (label says it all);
    # tooltips that stay are one line of simplified technical English
    for key in ("n0_nz", "n0_yconv", "n0_rtptop", "n0_rtint", "n0_rtdit",
                "n0_seed"):
        found = [w for w in at.get("number_input") + at.get("selectbox")
                 if w.key == key]
        if not found:            # nz_pic under the picaso provider
            continue
        assert not getattr(found[0], "help", None), \
            f"{key} regained a help tooltip"

    page = " ".join(m.value for m in at.markdown)
    caps = " ".join(c.value for c in at.get("caption"))
    body = page + " " + caps
    for gone in ("modeling choices that can move the spectrum",
                 "Read the results as optimistic",
                 "Treat mode rankings as more robust"):
        assert gone not in body
    assert "and its limits" not in page
    for gone in ("steady-state photochemical kinetics",
                 "radiative-convective climate profile",
                 "conditional template S/N per molecule",
                 "Beta", "full retrieval",
                 "numerical quality checks", "uncertified spectrum"):
        assert gone not in page, f"long-form intro phrase survived: {gone!r}"
    # The "How the model works" expander went 2026-08-17 (maintainer), and it
    # carried the engine names and the interpolated Pandeia version. Both are
    # still disclosed -- the backend label on the run status line, the engine
    # list in README.md -- so what this now pins is that the BLOCK stays gone
    # and that no one hardcodes a release number in its place.
    assert "Each run computes a spectrum for the atmosphere you configure" \
        not in page, "the 'How the model works' block is back"
    assert not re.search(r"Pandeia \d{4}\.\d", page), \
        "a Pandeia release is hardcoded into the landing page"
    assert re.search(r"Pandeia \d{4}\.\d",
                     "\n".join(l for l in APP.read_text().splitlines()
                               if "BACKEND_STATUS" in l)) is None, \
        "the backend label must interpolate ins.BACKEND_STATUS, never a literal"

    # Strip comment lines first: the removals left comments NAMING what was
    # removed, and matching those would make this test unfixable.
    src = "\n".join(l for l in APP.read_text().splitlines()
                    if not l.lstrip().startswith("#"))
    for phrase in ("Width of the final analysis bins",
                   "How to read this table",
                   "One panel per parameter in the summary figure",
                   "linearized best case",
                   "Detect: compare the full spectrum with one that omits",
                   "The forecast is a local Fisher (Cramer-Rao) bound from",
                   "AD (the default) differentiates the numerical model",
                   "Scales the reported bounds"):
        assert phrase not in src, f"removed GUI prose is back: {phrase!r}"
    lines = src.splitlines()
    lo = next(i for i, l in enumerate(lines)
              if 'st.markdown("### 3 · Science goal")' in l)
    hi = next(i for i, l in enumerate(lines)
              if i > lo and 'st.markdown("### 4 · Observation")' in l)
    offenders = [i + 1 for i in range(lo, hi) if "help=" in lines[i]]
    assert not offenders, f"step 3 regained tooltips at lines {offenders}"
    # 2026-08-16 resolution notes (maintainer request): the landing list and
    # the analysis-R caption state the resolution chain, and the three
    # line-by-line-only widgets stay OUT of a default (no-Mie-deck) boot --
    # they render only when a Mie deck forces line-by-line mode (companion
    # test below).
    assert "computed at R = 1000" in page, "the step-2 resolution note is gone"
    assert "blurred to each mode's native resolution" in page, \
        "the step-4 resolution note is gone"
    assert "not change the instrument's native resolution" in caps, \
        "the analysis-R caption is gone"
    _widget_keys = {w.key for w in at.get("number_input")} | {
        w.key for w in at.get("selectbox")}
    for key in ("n0_broad", "n0_nupts", "n0_rtdit"):
        assert key not in _widget_keys, \
            f"{key} renders without a Mie deck (line-by-line widgets are " \
            "Mie-branch only)"
    readme = (APP.parent.parent.parent / "README.md").read_text()
    assert "Analysis resolving power (the R control)" in readme
    assert "not the instrument's resolving power" in readme
    assert "PHOENIX" in readme, "the stellar model is now disclosed nowhere"


def test_lbl_widgets_render_only_with_mie_deck():
    """The broadening gas, native grid points, and line-wing grid widgets act
    only when a Mie deck forces line-by-line mode, so they render only then
    (2026-08-16); the default-boot absence is pinned in the removals test
    above. Keys are the shipped contract and must not change. Also pins the
    mode picker's measured native-R labels (r_native_med: display metadata
    measured from the 2026.7 refdata dispersion files -- re-measure on any
    refdata change, never edit the numbers freehand)."""
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["n0_miec"] = "MgSiO3"
    at.run()
    assert not at.exception, at.exception
    keys = {w.key for w in at.get("number_input")} | {
        w.key for w in at.get("selectbox")}
    for key in ("n0_broad", "n0_nupts", "n0_rtdit"):
        assert key in keys, f"{key} missing with a Mie deck selected"
    caps = " ".join(c.value for c in at.get("caption"))
    assert "line-by-line" in caps, "the Mie line-by-line caption is gone"

    from jwst_tool import instruments as ins
    for k, m in ins.MODES.items():
        assert int(m["r_native_med"]) > 0, f"{k} lacks r_native_med"
    assert ins.MODES["nirspec_prism"]["r_native_med"] == 100
    assert ins.MODES["nirspec_g395h"]["r_native_med"] == 2700


def test_results_render_and_below_target_is_warning_not_error():
    """Full post-Run render path on a synthetic result: every figure and
    table offers a download, and a run that works but finds no signal is a
    scientific outcome, not a software failure -- a warning, never an
    error."""
    out, out_meta = _synthetic_out()          # sigma_detect=0.0
    at = _run_with_result(out, out_meta)
    assert not at.exception, at.exception
    dl_labels = {b.label for b in at.get("download_button")}
    assert {"Figure (PDF, vector)", "Figure (PNG)", "Binned points (CSV)",
            "Native model (CSV)", "Values (CSV)",
            "Mode details (CSV)"} <= dl_labels
    assert any("No signal" in w.value for w in at.warning)
    assert not any("Best mode" in e.value for e in at.error)


def test_emission_results_use_eclipse_terms():
    """An emission run says "eclipse" throughout, never "transit"; an
    above-target result renders NO banner (maintainer, 2026-08-13: the figure
    and mode table already carry the number)."""
    out, out_meta = _synthetic_out(science_mode="emission",
                                   sigma_detect=8.0, n_transits=3)
    at = _run_with_result(out, out_meta)
    assert not at.exception, at.exception
    assert not at.success, \
        f"an above-target result must render no banner: {[s.value for s in at.success]}"
    # the figure section is an EXPANDER (2026-08-13), not a subheader
    _exps = [e.label for e in at.get("expander")]
    assert any("eclipse emission spectrum" in e for e in _exps), _exps
    assert not any(e == "Proposal summary figure" for e in _exps)
    # scope to the RESULTS text: the sidebar/intro copy legitimately says
    # "transit" for the general case
    results = " ".join(
        [w.value for w in at.warning] + [e.value for e in at.error]
        + [c.value for c in at.get("caption") if "eclipse" in c.value.lower()
           or "transit" in c.value.lower()]
        + [s.value for s in at.subheader] + _exps)
    assert "eclipse" in results.lower(), results[:300]
    assert not re.search(r"\b\d+ transits?\b", results), results[:300]


def test_all_saturated_run_has_no_best_mode_score_or_points():
    """All modes saturated: warning verdict, no "best" mode, no error alert
    for a valid calculation -- and no detection score or simulated points in
    the figure, matching the exclusion every ranking, combination and
    forecast already applies."""
    import jwst_tool.summary_figure as sf

    seen = {}
    _real = sf.compose_summary_figure

    def _spy(spectrum, **kw):
        seen["labels"] = [p["label"] for p in (spectrum.get("points") or [])]
        return _real(spectrum, **kw)

    out, out_meta = _synthetic_out(saturated=True, sigma_detect=5.0)
    sf.compose_summary_figure = _spy
    try:
        at = _run_with_result(out, out_meta)
    finally:
        sf.compose_summary_figure = _real
    assert not at.exception, at.exception
    warns = [w.value for w in at.warning]
    assert any("all selected modes saturate" in w for w in warns), warns
    assert not at.success
    assert not any("Best mode" in w for w in warns)
    assert not any("Best mode" in e.value for e in at.error)
    # no plotted series at all, in particular no "<mode>: 5.0σ" legend entry
    assert seen.get("labels") == [], seen


def test_picaso_provider_paths_render(monkeypatch):
    """Under the experimental gate, every PICASO switch order must render:
    the provider, the detect goal's Fisher checkbox, the constrain-goal
    Fisher multiselect in BOTH switch orders (regression: default lnKzz not
    in the menu crashed it), and picaso_climate. canonical_params failures
    surface through the params_error caption, never an exception."""
    monkeypatch.setenv("JWST_TOOL_ENABLE_UNCERTIFIED_PICASO", "1")

    at = _run_app()
    at.selectbox(key="n0_provider").set_value("picaso")
    at.run()
    assert not at.exception, at.exception
    at.checkbox(key="n0_dofish").check()
    at.run()
    assert not at.exception, at.exception
    at.radio(key="n0_goal").set_value("constrain")
    at.run()
    assert not at.exception, at.exception

    # constrain goal first, then the provider switch
    at2 = _run_app()
    at2.radio(key="n0_goal").set_value("constrain")
    at2.run()
    at2.selectbox(key="n0_provider").set_value("picaso")
    at2.run()
    assert not at2.exception, at2.exception

    at3 = _run_app()
    at3.selectbox(key="n0_wasp39b_tp").set_value("picaso_climate")
    at3.run()
    assert not at3.exception, at3.exception


def test_display_smoothing_is_nondestructive_and_actually_smooths():
    """Display smoothing must not touch the caller's native array (the CSV
    download exports it) and must measurably reduce spikiness at the app's
    display-R rule. Core smooth_to_native_r semantics live in
    test_binning.py; these two properties are the app-side contract."""
    from jwst_tool import binning

    wl = np.geomspace(1.0, 12.0, 4000)          # the real native grid shape
    rng = np.random.default_rng(0)
    native = (21000.0 + 200.0 * np.sin(8.0 * np.log(wl))
              + rng.choice([0.0, 0.0, 0.0, 600.0], size=wl.size))
    untouched = native.copy()
    r_disp = float(max(300, 3 * 100))           # the app's rule at R_bin=100
    smoothed = binning.smooth_to_native_r(
        wl, native, np.array([wl[0], wl[-1]]), np.array([r_disp, r_disp]),
        float(wl[0]), float(wl[-1]))
    assert np.array_equal(native, untouched), \
        "display smoothing mutated the caller's native model"
    # RMS of point-to-point differences, NOT the median: most consecutive
    # pairs are exactly equal, so the median jump is ~0 and unbeatable.
    rms_native = float(np.diff(untouched).std())
    rms_disp = float(np.diff(smoothed).std())
    assert rms_disp < 0.2 * rms_native, (rms_native, rms_disp)


def test_custom_archive_fill_and_uv_menu_never_moves():
    """Custom planet, one boot: the nearest-Teff caption tracks a typed Teff
    while the UV MENU never moves, and the archive Fill button writes the
    snapshot row into the form (pending-then-apply path) -- still without
    touching the UV menu (no substitute spectra, maintainer rule
    2026-08-09)."""
    at = _run_app()
    at.selectbox(key="n0_planet").set_value("custom").run()
    assert not at.exception, at.exception
    caps = " | ".join(c.value for c in at.caption)
    assert "Nearest-Teff shipped UV template" in caps and "WASP-39" in caps
    at.number_input(key="n0_custom_teff").set_value(3100.0).run()
    assert not at.exception, at.exception
    caps = " | ".join(c.value for c in at.caption)
    assert "GJ 1214" in caps
    assert "A different spectrum is currently selected" in caps
    assert at.selectbox(key="n0_custom_sflux").value == \
        "sflux-W39b_Tsai2023.txt"

    at.selectbox(key="n0_custom_arch_name").set_value("HD 189733 b").run()
    at.button(key="n0_custom_arch_fill").click().run()
    assert not at.exception, at.exception
    from jwst_tool import archive
    values, _ = archive.custom_fill(archive.lookup("HD 189733 b"))
    assert at.number_input(key="n0_custom_teff").value == values["teff"]
    assert at.number_input(key="n0_custom_g").value == \
        pytest.approx(values["g"])
    # the UV menu is NEVER written by the fill
    assert at.selectbox(key="n0_custom_sflux").value == \
        "sflux-W39b_Tsai2023.txt"
    assert any("archive snapshot" in s.value for s in at.success)


def test_mock_observation_disclosure_and_recovery_overlay():
    """Mock layer ON + Jacobians: the seeded mock CSV appears, the noiseless
    result downloads stay, and the mock_recovery posterior overlay renders.
    NOT "display-only" -- the draw is fitted; the CSV split pins that the
    draw rides in its own clearly-named export and never contaminates the
    noiseless result CSVs."""
    out, out_meta = _synthetic_out(sigma_detect=8.0, with_jac=True)
    at = _run_with_result(out, out_meta, n0_shownoise=True, n0_seed=7)
    assert not at.exception, at.exception
    dl = {b.label for b in at.get("download_button")}
    assert "Mock observation (CSV)" in dl
    assert {"Binned points (CSV)", "Native model (CSV)"} <= dl
    _exps = [e.label for e in at.get("expander")]
    assert any("forecast summary" in e for e in _exps), _exps


def test_combo_builder_fisher_table_naming_and_reset():
    """One booted app with Jacobians carries the whole combo story: every
    results section is an expander (2026-08-13 renames pinned); a primed
    combination LEADS the STATIC constraint table under the user's own name
    ('All usable modes', no 'COMBO: ' prefix; st.table so a header click
    cannot detach the blanked mode names); the builder adds and removes
    combos; the confirm-step reset clears the combo/posterior session keys
    (nonce-namespaced keys + un-namespaced pending notes)."""
    import pandas as pd

    primed = dict(name="My set", modes=["nirspec_g395h", "nirspec_prism"])
    out, out_meta = _synthetic_out(sigma_detect=8.0, with_jac=True)
    at = _run_with_result(out, out_meta, n0_combos=[dict(primed)])
    assert not at.exception, at.exception
    subs = [s.value for s in at.subheader]
    exps = [e.label for e in at.get("expander")]
    assert "Add a custom mode set" in exps, exps
    assert "Mode combinations" not in exps, exps
    assert "Parameter constraint forecast (Fisher)" in exps, exps
    assert "Physical structure (T-P profile, mixing ratios)" in exps, exps
    assert "Marginalized forecast posteriors" in exps, exps
    assert "Marginalized forecast posteriors" not in subs, subs
    assert any("forecast summary" in e for e in exps), exps
    assert not any("forecast summary" in s for s in subs), subs
    assert "Figure (PDF, vector)" in {b.label
                                      for b in at.get("download_button")}

    # the CONSTRAINT table specifically -- Mode details also has a "mode"
    # column and renders first, so identify by the parameter column
    tables = [t.value for t in at.table
              if "mode" in getattr(t.value, "columns", [])
              and "parameter" in getattr(t.value, "columns", [])]
    assert tables, "no constraint-forecast table rendered"
    assert not [df.value for df in at.dataframe
                if "parameter" in getattr(df.value, "columns", [])], \
        "the constraint forecast is an interactive dataframe again"
    modes = [str(v) for v in tables[0]["mode"]]
    assert not any("ALL USABLE" in m for m in modes), modes[:8]
    assert not any(m.startswith("COMBO:") for m in modes), modes[:8]
    assert any("All usable modes" == m for m in modes), modes[:8]
    first = next(m for m in modes if m.strip())
    assert "My set" in first, f"combos should lead the table, got {first!r}"

    # add a named combination through the builder widgets
    at.text_input(key="n0_cb_name").set_value("PRISM + G395H")
    at.multiselect(key="n0_cb_modes").set_value(
        ["nirspec_prism", "nirspec_g395h"])
    at.button(key="n0_cb_add").click().run()
    assert not at.exception, at.exception
    added = dict(name="PRISM + G395H", modes=["nirspec_prism", "nirspec_g395h"])
    combos = at.session_state["n0_combos"]
    assert added in combos and primed in combos and len(combos) == 2, combos
    assert "PRISM + G395H" in " ".join(m.value for m in at.markdown)
    tables = [pd.DataFrame(d.value)
              for d in list(at.get("table")) + list(at.get("dataframe"))]
    assert any(t.astype(str)
               .apply(lambda c: c.str.contains("PRISM \\+ G395H"))
               .any().any() for t in tables), \
        "combo rows missing from the Fisher table"
    # remove works (index 0 = the first listed combo)
    removed_first = combos[0]
    at.button(key="n0_cb_rm_0").click().run()
    assert not at.exception, at.exception
    assert at.session_state["n0_combos"] == \
        [c for c in combos if c != removed_first]

    # confirm-step reset (labels, not keys: the reset buttons are keyless)
    [b for b in at.button if (b.label or "") == "Reset all settings"][0] \
        .click().run()
    [b for b in at.button if (b.label or "") == "Confirm reset"][0] \
        .click().run()
    assert not at.exception, at.exception
    assert "n0_combos" not in at.session_state
    assert "_combo_note" not in at.session_state


def test_emission_mode_archive_fill_skips_transit_duration():
    """The archive duration is the PRIMARY-TRANSIT duration; in emission
    mode the fill must leave the event-duration widget alone and say why."""
    at = _run_app()
    at.selectbox(key="n0_planet").set_value("custom").run()
    at.radio(key="n0_scimode").set_value("emission").run()
    t14_before = at.number_input(key="n0_custom_t14").value
    at.selectbox(key="n0_custom_arch_name").set_value("HD 189733 b").run()
    at.button(key="n0_custom_arch_fill").click().run()
    assert not at.exception, at.exception
    from jwst_tool import archive
    values, _ = archive.custom_fill(archive.lookup("HD 189733 b"))
    assert at.number_input(key="n0_custom_t14").value == t14_before
    assert at.number_input(key="n0_custom_teff").value == values["teff"]
    warns = " | ".join(w.value for w in at.warning)
    assert "secondary-eclipse duration can differ" in warns


def test_every_axis_control_is_a_typed_min_max_pair():
    """No slider anywhere near an axis (maintainer, 2026-08-14): every axis
    on every figure is two number boxes, min and max, blank for automatic.
    The wavelength window came BACK in that pass (a reviewer asked to zoom
    into the PRISM/G395H regions); what was wrong before was the RADIO +
    range-slider pair. Blank boxes must START blank -- prefilled bounds go
    stale the moment the run or mode selection changes."""
    out, out_meta = _synthetic_out(sigma_detect=8.0, with_jac=True)
    at = _run_with_result(out, out_meta)
    assert not at.exception, at.exception
    assert not [r for r in at.radio if r.key == "n0_sum_wlmode"], \
        "the wavelength-range radio should be gone"
    for _k in ("sum_x", "sum_y", "struct_T", "struct_p", "struct_vmr",
               "sum_post_lnZ"):
        for _end in ("min", "max"):
            _w = at.number_input(key=f"n0_{_k}_{_end}")
            assert _w.value is None, f"{_k}_{_end} should start blank"
    # NO sliders anywhere (maintainer, 2026-08-15): every numeric input is a
    # typed number box, including the noise multiplier that used to be the
    # one exception
    assert not at.slider, [s.key for s in at.slider]
    _x = at.checkbox(key="n0_sum_xlog")
    _y = at.checkbox(key="n0_sum_ylog")
    assert _x.value is True and _x.label == "Log x", _x.label
    assert _y.value is False and _y.label == "Log y", _y.label


@pytest.mark.parametrize(
    "widget,value,field",
    # the molecule selectbox keys on the extra-molecule selection, so the key
    # carries the default extra set (jwst_tool.app: K("mol_<provider>_<mols>"));
    # CS2/C2H6 left the default at v31 (no ExoMolOP table), SH+SO JOINED it at
    # v34 (measured 10.0 and 48.1 ppm on W39b -- forward.EXTRA_MOLECULES_DEFAULT)
    [("n0_mol_vulcan_C2H2_C2H4_H2S_HCN_NH3_OCS_SH_SO", "CO2",
      "target_mol"),                                # the reported bug
     ("n0_noisescale", 2.0, "noise_scale")])        # an observation-block field
def test_changing_any_run_input_marks_the_result_stale(widget, value, field,
                                                       monkeypatch):
    """The staleness guard compares the WHOLE non-canonical input set, not a
    hand-picked subset. The bug this pins: the detection target is absent
    from the canonical model params, so switching SO2 to CO2 without pressing
    Run left the curve, legend and CSV on the previous target with no notice
    (reported as a caching bug). The noise scale represents the observation
    block because a subset guard is exactly how the next one gets missed."""
    from jwst_tool import share_config as _sc
    seen = {}
    _real = _sc.build_share

    def _spy(*, canon, goal, observation, **kw):
        # the app hands the SAME two blocks to the shareable config and to
        # the staleness guard
        seen["run_sig"] = dict(goal=goal, observation=observation)
        seen["canon"] = canon
        return _real(canon=canon, goal=goal, observation=observation, **kw)

    monkeypatch.setattr(_sc, "build_share", _spy)
    out, out_meta = _synthetic_out(sigma_detect=8.0, with_jac=True)
    at = _run_with_result(out, out_meta)
    assert not at.exception, at.exception
    # Adopt the page's OWN canonical params and run signature as the cached
    # run's, so the ONLY difference below is the widget under test.
    import json as _json
    at.session_state["out"] = dict(
        out, model=dict(out["model"],
                        params_json=_json.dumps(seen["canon"], default=str)))
    at.session_state["out_meta"] = dict(out_meta, run_sig=seen["run_sig"])
    at.run()

    def _named():
        return [w.value for w in at.warning if "previous run" in w.value]

    assert not _named(), \
        f"an unchanged sidebar must not read as stale: {_named()}"
    at.session_state[widget] = value
    at.run()
    assert not at.exception, at.exception
    assert _named() and field in _named()[0], _named()


@pytest.mark.parametrize("key,label", [("n0_sum_x", "Wavelength"),
                                       ("n0_struct_vmr", "Mixing ratio")])
def test_a_nonpositive_bound_on_a_log_axis_warns_instead_of_killing_the_page(
        key, label):
    """A min at or below zero has no logarithm on a log axis (one summary
    axis + one structure-panel axis cover both figure builders). The builders
    raise on it -- right for an API backstop, but uncaught it takes the
    ENTIRE results page down. A typed number is a user choice, not a defect:
    warn, fall back to the automatic fit, keep the page alive."""
    out, out_meta = _synthetic_out(sigma_detect=8.0, with_jac=True)
    at = _run_with_result(out, out_meta,
                          **{f"{key}_min": 0.0, f"{key}_max": 5.0})
    assert not at.exception, at.exception
    assert any("log axis" in w.value and label in w.value for w in at.warning), \
        [w.value for w in at.warning]


def test_axis_bounds_refuse_one_sided_and_reach_the_figure(monkeypatch):
    """One bound alone has no second edge: the page must SAY it fell back
    rather than invent the other bound -- and a COMPLETE pair must actually
    reach the figure (the reviewer's 3.0 to 5.5 um zoom), not merely be
    accepted by the widget. The app closes every Figure it builds, so this
    records the ARGUMENTS of the call the page makes."""
    from jwst_tool import summary_figure as _sf
    seen = {}
    _real = _sf.compose_summary_figure

    def _spy(spectrum, **kw):
        seen["wl_range"] = spectrum.get("wl_range")
        seen["depth_range"] = spectrum.get("depth_range")
        seen["panel_xlims"] = kw.get("panel_xlims")
        return _real(spectrum, **kw)

    monkeypatch.setattr(_sf, "compose_summary_figure", _spy)
    out, out_meta = _synthetic_out(sigma_detect=8.0, with_jac=True)
    at = _run_with_result(out, out_meta,
                          n0_sum_y_min=20500.0)     # max left blank
    assert not at.exception, at.exception
    assert any("both boxes" in w.value for w in at.warning), \
        [w.value for w in at.warning]
    # the COMPLETE pairs are accepted silently (so the warning above is a
    # real refusal) and the typed windows reach the figure call
    at.session_state["n0_sum_y_max"] = 21500.0
    at.session_state["n0_sum_x_min"] = 3.0
    at.session_state["n0_sum_x_max"] = 5.5
    at.session_state["n0_sum_post_lnZ_min"] = -0.4
    at.session_state["n0_sum_post_lnZ_max"] = 0.4
    at.run()
    assert not at.exception, at.exception
    assert not [w for w in at.warning if "Depth range" in w.value], \
        [w.value for w in at.warning]
    assert seen["wl_range"] == pytest.approx((3.0, 5.5))
    assert seen["depth_range"] == pytest.approx((20500.0, 21500.0))
    assert seen["panel_xlims"][0] == pytest.approx((-0.4, 0.4))


def test_mixing_ratio_panel_selects_species_by_name(monkeypatch):
    """ymix is the FULL network state (89 species for SNCHO); model["mols"]
    is the short RT list. Zipping them positionally read the wrong species:
    found 2026-08-14 against Tsai et al. 2023's published WASP-39 b VULCAN
    run -- the curve labelled CO2 sat at 0.847 at 1 bar, which is H2. The
    panel and its CSV were mislabelled for every species."""
    from jwst_tool import plotting
    seen = {}
    _real = plotting.build_vmr_figure

    def _spy(p_bar, columns, **kw):
        seen["columns"] = list(columns)
        return _real(p_bar, columns, **kw)

    monkeypatch.setattr(plotting, "build_vmr_figure", _spy)
    out, out_meta = _synthetic_out(sigma_detect=8.0, with_jac=True)
    at = _run_with_result(out, out_meta)
    assert not at.exception, at.exception
    got = {name: float(np.asarray(y)[0]) for name, y in seen["columns"]}
    # the fixture puts H2 first at 0.85; it is NOT an RT molecule and must
    # not appear, and no RT molecule may take its column
    assert "H2" not in got
    assert set(got) == {"H2O", "CO2", "CO", "CH4", "SO2"}, got
    assert got["CO"] == pytest.approx(1e-3)     # by NAME, not by position
    assert got["H2O"] == pytest.approx(1e-4)
    assert got["CO2"] == pytest.approx(1e-6)
    assert 0.85 not in got.values()
