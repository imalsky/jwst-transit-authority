"""GUI smoke tests: the app renders end-to-end with no exception.

Needs the GUI extras (streamlit + pandas); the dependency-light CI skips it.
Uses Streamlit's AppTest; no forward-model run is launched.
"""
from __future__ import annotations

import re
from pathlib import Path

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
    ``with_jac=True`` adds a two-parameter Jacobian (lnZ, dlnCO + lnR0) to
    the model and a SECOND mode, so the Fisher table, the combo builder,
    the forecast posteriors, and the summary figure all render.
    """
    import json
    import numpy as np

    n = 40
    wl = np.linspace(1.0, 5.0, n)
    model = {
        "wl_um": wl,
        "depth": np.full(n, 0.021) + 1e-4 * np.sin(wl),
        "mols": np.array(["H2O", "CO2", "CO", "CH4", "SO2"], dtype="U8"),
        "depth_wo": np.tile(np.full(n, 0.0208), (5, 1)),
        "T": np.full(30, 1100.0),
        "p_bar": np.logspace(-7, 0.8, 30),
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


def test_app_renders_without_exception():
    at = _run_app()
    assert not at.exception, at.exception


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
        # Force Guillot mode so the T_irr widget renders; compare against
        # canonical_params, never the helper the widget itself calls.
        at.selectbox(key=f"n0_{key}_tp").set_value("guillot").run()
        assert not at.exception, (key, at.exception)
        api_tirr = forward.canonical_params(
            dict(planet=key, tp_mode="guillot"))["Tirr"]
        assert at.number_input(key=f"n0_{key}_tirr").value == api_tirr, key
        at.selectbox(key=f"n0_{key}_tp").set_value(cp["tp_mode"]).run()


def test_default_instrument_modes_are_the_speed_first_trio():
    """0.27.0 default: PRISM + G395H + MIRI LRS (full span + both SO2
    bands); the ETC computes only the selected modes, so the trio is what a
    default run pays for."""
    from jwst_tool import instruments as ins

    at = _run_app()
    assert not at.exception, at.exception
    assert set(at.multiselect(key="n0_modes").value) == set(ins.DEFAULT_MODES)
    assert set(ins.DEFAULT_MODES) == {"nirspec_prism", "nirspec_g395h",
                                      "miri_lrs"}


def test_data_status_panel_present():
    at = _run_app()
    # the availability expander renders on the main page, pre-run
    labels = " ".join(e.label for e in at.expander)
    assert "Data status" in labels


def test_sidebar_molecule_annotations_present():
    at = _run_app()
    ms = [w for w in at.multiselect if w.label == "Extra opacity molecules"]
    assert ms, "extra-molecule multiselect missing"
    # format_func must annotate availability, one of the three states
    from jwst_tool import datacheck, forward
    status = datacheck.molecule_linelist_status(forward.EXTRA_MOLECULES)
    assert set(status) == set(forward.EXTRA_MOLECULES)


def test_results_render_with_synthetic_run():
    """Full post-Run render path on a synthetic result (no forward model)."""
    out, out_meta = _synthetic_out()
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["out"] = out
    at.session_state["out_meta"] = out_meta
    at.run()
    assert not at.exception, at.exception
    # every figure and table must offer a download
    dl_labels = {b.label for b in at.get("download_button")}
    assert {"Figure (PDF, vector)", "Figure (PNG)", "Binned points (CSV)",
            "Native model (CSV)", "Values (CSV)",
            "Mode details (CSV)"} <= dl_labels


def _deferred_labels(at):
    """(live download buttons, dead busy stand-ins) among the deferred set."""
    deferred = {"Download configuration (JSON)",
                "Download this example (edit and re-upload)"}
    live = {b.label for b in at.get("download_button")} & deferred
    dead = {b.label for b in at.get("button")} & deferred
    return live, dead


def test_config_download_is_live_when_no_run_is_in_flight():
    """Outside a run the placeholders hold live download buttons, not dead
    stand-ins (clicking a download widget mid-run cancels the run)."""
    at = _run_app()
    assert not at.exception, at.exception
    live, dead = _deferred_labels(at)
    assert "Download configuration (JSON)" in live
    assert not dead, "a busy stand-in must not render outside a run"


def test_tp_example_download_is_deferred_too():
    """The sidebar's T-P example download uses the same deferred slot."""
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["n0_wasp39b_tp"] = "file"
    at.session_state["n0_wasp39b_tpsrc"] = "upload"   # forward.TP_FILE_UPLOAD
    at.run()
    assert not at.exception, at.exception
    live, dead = _deferred_labels(at)
    assert "Download this example (edit and re-upload)" in live
    assert not dead


def test_no_intro_gate():
    """The first render is the tool itself: sidebar widgets exist immediately,
    with no acknowledgment button blocking them."""
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert at.selectbox(key="n0_planet").value is not None
    assert not [b for b in at.button if "I understand" in (b.label or "")]


def test_noise_floor_defaults_to_constant_and_file_still_blocks():
    """The floor type defaults to "constant" (maintainer, 2026-08-13),
    superseding the earlier no-preselection gate. A 15-40 ppm minimum SETS the
    reported precision and no floor ignores 1/f and visit-long systematics, so
    the default is the CONSERVATIVE choice, not a neutral one -- and it is
    still recorded with the run and editable per mode. "Wavelength table" with
    no file uploaded is the remaining state that blocks the run."""
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception, at.exception

    def _floor_error(app):
        return [e.value for e in app.error
                if "minimum noise floor" in (e.value or "").lower()]

    # 2026-08-13 (maintainer): "constant" is PRESELECTED, so the floor gate is
    # satisfied on first load and no floor error is shown. A default is
    # acceptable in this direction only -- a constant floor claims LESS
    # precision than "No floor" would, so it is the conservative side of the
    # choice rather than a flattering one.
    assert at.radio(key="n0_floormode").value == "constant"
    assert not _floor_error(at), [e.value for e in at.error]
    # the per-mode floor inputs render with it
    assert any((w.key or "").startswith("n0_floor_")
               for w in at.get("number_input"))

    # "No floor" stays an explicit choice and also clears the gate
    at.radio(key="n0_floormode").set_value("none").run()
    assert not at.exception, at.exception
    assert not _floor_error(at)

    # "Wavelength table" with no upload is the one state that still BLOCKS
    at.radio(key="n0_floormode").set_value("file").run()
    assert not at.exception, at.exception
    assert _floor_error(at), [e.value for e in at.error]


def test_constant_floor_prefill_renders_per_mode_inputs():
    """Choosing the constant floor renders one editable input per mode
    (the 2026-08-12 caption trim removed the prose; the widgets stay)."""
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    at.radio(key="n0_floormode").set_value("constant").run()
    assert not at.exception, at.exception
    keys = {w.key for w in at.get("number_input")}
    assert any(k and k.startswith("n0_floor_") for k in keys)


def test_removed_tooltips_and_table_guidance_are_gone():
    """Prose the maintainer removed 2026-08-13 must not come back.

    Every "?" tooltip in step 3 (Science goal), the R-binning tooltip, the
    "How to read this table" expander, and the parameter-panel tooltip. The
    displaced content that changes how a number is READ lives in README.md
    (the R section and the Fisher-bounds section), which the house policy
    prefers over GUI prose.
    """
    # Strip comment lines first: the removals left behind comments that NAME
    # what was removed (deliberately -- they explain why the code is absent),
    # and matching those would make this test unfixable.
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

    # step 3 (Science goal) carries NO tooltips at all
    lines = src.splitlines()
    lo = next(i for i, l in enumerate(lines)
              if 'st.markdown("### 3 \u00b7 Science goal")' in l)
    hi = next(i for i, l in enumerate(lines)
              if i > lo and 'st.markdown("### 4 \u00b7 Observation")' in l)
    offenders = [i + 1 for i in range(lo, hi) if "help=" in lines[i]]
    assert not offenders, f"step 3 regained tooltips at lines {offenders}"

    # the README keeps what the R tooltip used to say
    readme = (APP.parent.parent.parent / "README.md").read_text()
    assert "Analysis resolving power (the R control)" in readme
    assert "not the instrument's resolving power" in readme


def test_emission_results_use_eclipse_terms():
    """An emission run says "eclipse" throughout, never "transit".

    An above-target result renders NO banner as of 2026-08-13 (maintainer):
    the figure and the mode table already carry the number, so a green bar
    restating it was redundant. The eclipse/transit wording is what this test
    is for, and it is checked across the whole rendered page.
    """
    out, out_meta = _synthetic_out(science_mode="emission",
                                   sigma_detect=8.0, n_transits=3)
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["out"] = out
    at.session_state["out_meta"] = out_meta
    at.run()
    assert not at.exception, at.exception
    assert not at.success, \
        f"an above-target result must render no banner: {[s.value for s in at.success]}"
    # the figure section is an EXPANDER now (maintainer, 2026-08-13): it was
    # the last st.subheader on the results page
    _exps = [e.label for e in at.get("expander")]
    assert any("eclipse emission spectrum" in e for e in _exps), _exps
    assert not any(e == "Proposal summary figure" for e in _exps)
    # Scope to the RESULTS text: the sidebar/intro copy is written for the
    # general case and legitimately says "transit" there.
    results = " ".join(
        [w.value for w in at.warning] + [e.value for e in at.error]
        + [c.value for c in at.get("caption") if "eclipse" in c.value.lower()
           or "transit" in c.value.lower()]
        + [s.value for s in at.subheader] + _exps)
    assert "eclipse" in results.lower(), results[:300]
    assert not re.search(r"\b\d+ transits?\b", results), results[:300]


def test_all_saturated_state_has_no_best_mode():
    """All modes saturated: warning verdict, no "best" mode, and no error
    alert for what is a valid calculation."""
    out, out_meta = _synthetic_out(saturated=True, sigma_detect=5.0)
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["out"] = out
    at.session_state["out_meta"] = out_meta
    at.run()
    assert not at.exception, at.exception
    warns = [w.value for w in at.warning]
    assert any("all selected modes saturate" in w for w in warns), warns
    assert not at.success
    assert not any("Best mode" in w for w in warns)
    assert not any("Best mode" in e.value for e in at.error)


def test_saturated_mode_gets_no_score_and_no_plotted_points():
    """A saturated mode carries no usable data: it must not appear in the
    results figure with a detection score or simulated points, matching the
    exclusion every ranking, combination and forecast already applies."""
    import jwst_tool.summary_figure as sf

    seen = {}
    _real = sf.compose_summary_figure

    def _spy(spectrum, **kw):
        seen["labels"] = [p["label"] for p in (spectrum.get("points") or [])]
        return _real(spectrum, **kw)

    out, out_meta = _synthetic_out(saturated=True, sigma_detect=5.0)
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["out"] = out
    at.session_state["out_meta"] = out_meta
    sf.compose_summary_figure = _spy
    try:
        at.run()
    finally:
        sf.compose_summary_figure = _real
    assert not at.exception, at.exception
    # every mode in this run saturates -> no plotted series at all, and in
    # particular no "<mode>: 5.0σ" legend entry
    assert seen.get("labels") == [], seen
    assert not any("σ" in lbl for lbl in seen.get("labels", []))


def test_valid_below_target_result_is_warning_not_error():
    """A run that works but finds no signal is a scientific outcome, not a
    software failure: it must render as a warning, never an error."""
    out, out_meta = _synthetic_out(sigma_detect=0.0)
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["out"] = out
    at.session_state["out_meta"] = out_meta
    at.run()
    assert not at.exception, at.exception
    assert any("No signal" in w.value for w in at.warning)
    assert not any("Best mode" in e.value for e in at.error)


def test_picaso_provider_renders(monkeypatch):
    # Switching the engine to PICASO must render; canonical_params failures
    # surface through the params_error caption, never an exception.
    at = _run_app()
    at.selectbox(key="n0_provider").set_value("picaso")
    at.run()
    assert not at.exception, at.exception


def test_picaso_climate_mode_renders():
    at = _run_app()
    at.selectbox(key="n0_wasp39b_tp").set_value("picaso_climate")
    at.run()
    assert not at.exception, at.exception


def test_picaso_constrain_goal_renders():
    # Regression: the constrain-goal Fisher multiselect crashed under PICASO
    # (default lnKzz not in the menu) -- both switch orders must render.
    at = _run_app()
    at.radio(key="n0_goal").set_value("constrain")
    at.run()
    at.selectbox(key="n0_provider").set_value("picaso")
    at.run()
    assert not at.exception, at.exception

    at2 = _run_app()
    at2.selectbox(key="n0_provider").set_value("picaso")
    at2.run()
    at2.radio(key="n0_goal").set_value("constrain")
    at2.run()
    assert not at2.exception, at2.exception


def test_display_smoothing_is_nondestructive_and_actually_smooths():
    """Display smoothing must not touch the caller's native array (the CSV
    download exports it) and must measurably reduce spikiness."""
    import numpy as np

    from jwst_tool import binning

    wl = np.geomspace(1.0, 12.0, 4000)          # the real native grid shape
    rng = np.random.default_rng(0)
    # smooth continuum plus isolated upward spikes (an unresolved line forest)
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
    # the convolution conserves the continuum level
    assert abs(smoothed.mean() / untouched.mean() - 1.0) < 1e-4


def test_ad_selection_locks_photochemistry_on():
    """AD under a constrain goal forces the photochemistry checkbox ON and
    disabled; switching back to detect releases it. (The photo widget renders
    before the method menu, so the lock reads session state.)"""
    at = _run_app()
    at.radio(key="n0_goal").set_value("constrain")
    at.run()
    at.selectbox(key="n0_jacm").set_value("ad")
    at.run()
    assert not at.exception, at.exception
    photo = at.checkbox(key="n0_photo")
    assert photo.value is True
    assert photo.disabled
    at.radio(key="n0_goal").set_value("detect")
    at.run()
    assert not at.exception, at.exception
    assert not at.checkbox(key="n0_photo").disabled


def test_picaso_detect_fisher_checkbox_renders():
    # same defect on the detect goal's "Compute parameter constraints too"
    at = _run_app()
    at.selectbox(key="n0_provider").set_value("picaso")
    at.run()
    at.checkbox(key="n0_dofish").check()
    at.run()
    assert not at.exception, at.exception


def test_custom_archive_fill_updates_the_form():
    """The archive Fill button writes the snapshot row into the custom form
    via the pending-then-apply-before-widgets path, with a success message."""
    at = _run_app()
    at.selectbox(key="n0_planet").set_value("custom").run()
    assert not at.exception, at.exception
    at.selectbox(key="n0_custom_arch_name").set_value("HD 189733 b")
    at.button(key="n0_custom_arch_fill").click().run()
    assert not at.exception, at.exception
    from jwst_tool import archive
    values, _ = archive.custom_fill(archive.lookup("HD 189733 b"))
    assert at.number_input(key="n0_custom_teff").value == values["teff"]
    assert at.number_input(key="n0_custom_g").value == \
        pytest.approx(values["g"])
    # the UV menu is NEVER written by the fill (no substitute spectra):
    # it stays at the custom default even though HD 189733 b was filled
    assert at.selectbox(key="n0_custom_sflux").value == \
        "sflux-W39b_Tsai2023.txt"
    assert any("archive snapshot" in s.value for s in at.success)


def test_custom_sflux_nearest_caption_discloses_and_never_flips():
    """The nearest-type caption tracks a typed Teff; the MENU does not move
    (only the fill path applies the nearest-type default)."""
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
    # the menu itself stayed on the WASP-39 default
    assert at.selectbox(key="n0_custom_sflux").value == \
        "sflux-W39b_Tsai2023.txt"


def test_intro_carries_no_beta_or_quality_check_prose():
    """Both paragraphs were REMOVED 2026-08-13 (maintainer): the beta
    disclaimer and the numerical-quality-checks sentence.

    Inverted rather than deleted -- the removal is the requirement now, and a
    future edit that reintroduces either should fail here. The caveats they
    carried live in README.md, per the standing GUI prose policy.
    """
    at = _run_app()
    assert not at.exception, at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "Beta" not in md
    assert "full retrieval" not in md
    assert "numerical quality checks" not in md
    assert "uncertified spectrum" not in md


def test_mock_observation_controls():
    """The Jitter toggle + seed render; the 'New draw' BUTTON is gone.

    Removed 2026-08-13 (maintainer): the seed field stays, so a realization is
    still reproducible and still selectable by typing, but there is no
    one-click re-roll. Widget keys are unchanged, so share_config round-trips.
    """
    at = _run_app()
    assert not at.exception, at.exception
    assert at.checkbox(key="n0_shownoise").value is True   # ON by default
    assert at.number_input(key="n0_seed").value == 0
    # editing the seed still redraws without recomputing
    at.number_input(key="n0_seed").set_value(7).run()
    assert not at.exception, at.exception
    assert at.number_input(key="n0_seed").value == 7
    assert not [b for b in at.get("button") if b.key == "n0_reroll"], \
        "the 'New draw' button was removed"


def test_mock_observation_render_disclosure_and_download():
    """With the mock layer ON, the mock CSV download (named with its seed)
    appears and the noiseless result downloads stay.

    NOT "display-only": the draw is fitted (mock_recovery overlays the
    recovered parameters on the posterior panels). What the CSV split pins is
    the one-directional invariant -- the draw rides in its own clearly-named
    export and never contaminates the noiseless result CSVs.
    """
    out, out_meta = _synthetic_out(sigma_detect=8.0)
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["out"] = out
    at.session_state["out_meta"] = out_meta
    at.session_state["n0_shownoise"] = True
    at.session_state["n0_seed"] = 7
    at.run()
    assert not at.exception, at.exception
    dl = {b.label for b in at.get("download_button")}
    # the mock CSV is the disclosure: it is named with its seed and is the
    # ONLY export carrying the draw (no UI prose, per the GUI prose policy)
    assert "Mock observation (CSV)" in dl
    assert {"Binned points (CSV)", "Native model (CSV)"} <= dl


def test_combo_builder_and_summary_figure_render_with_jacobians():
    """With Jacobians present: the combo builder renders, adding a combo
    puts its rows in the Fisher table,
    and the proposal summary figure offers PDF + PNG downloads."""
    out, out_meta = _synthetic_out(sigma_detect=8.0, with_jac=True)
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["out"] = out
    at.session_state["out_meta"] = out_meta
    at.run()
    assert not at.exception, at.exception
    subs = [s.value for s in at.subheader]
    # EVERY results section is a collapsible expander (maintainer,
    # 2026-08-13): the combo builder is "Add a custom mode set" (renamed from
    # "Mode combinations"), and the forecast posteriors moved from the one
    # remaining st.subheader into an expander like the rest.
    exps = [e.label for e in at.get("expander")]
    assert "Add a custom mode set" in exps, exps
    assert "Mode combinations" not in exps, exps
    assert "Parameter constraint forecast (Fisher)" in exps, exps
    assert "Physical structure (T-P profile, mixing ratios)" in exps, exps
    assert "Marginalized forecast posteriors" in exps, exps
    assert "Marginalized forecast posteriors" not in subs, subs
    assert any("forecast summary" in e for e in exps), exps
    assert not any("forecast summary" in s for s in subs), subs
    dl = {b.label for b in at.get("download_button")}
    assert "Figure (PDF, vector)" in dl
    # add a named combination through the builder widgets
    at.text_input(key="n0_cb_name").set_value("PRISM + G395H")
    at.multiselect(key="n0_cb_modes").set_value(
        ["nirspec_prism", "nirspec_g395h"])
    at.button(key="n0_cb_add").click().run()
    assert not at.exception, at.exception
    assert at.session_state["n0_combos"] == [
        dict(name="PRISM + G395H",
             modes=["nirspec_prism", "nirspec_g395h"])]
    md = " ".join(m.value for m in at.markdown)
    assert "PRISM + G395H" in md
    # its rows are in the Fisher long-format table
    import pandas as pd
    tables = [pd.DataFrame(d.value) for d in at.get("dataframe")]
    assert any(t.astype(str)
               .apply(lambda c: c.str.contains("PRISM \\+ G395H"))
               .any().any() for t in tables), \
        "combo rows missing from the Fisher table"
    # remove works
    at.button(key="n0_cb_rm_0").click().run()
    assert not at.exception, at.exception
    assert at.session_state["n0_combos"] == []


def test_combo_and_posterior_widgets_survive_reset():
    """The confirm-step reset clears combos and the posterior selections
    (nonce-namespaced keys + un-namespaced pending notes)."""
    out, out_meta = _synthetic_out(sigma_detect=8.0, with_jac=True)
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["out"] = out
    at.session_state["out_meta"] = out_meta
    at.session_state["n0_combos"] = [dict(name="X",
                                          modes=["nirspec_g395h"])]
    at.run()
    assert not at.exception, at.exception
    # arm + confirm the reset (labels, not keys: the reset buttons are keyless)
    [b for b in at.button if (b.label or "") == "Reset all settings"][0] \
        .click().run()
    [b for b in at.button if (b.label or "") == "Confirm reset"][0] \
        .click().run()
    assert not at.exception, at.exception
    assert "n0_combos" not in at.session_state
    assert "_combo_note" not in at.session_state


def test_posterior_panel_mock_recovery_overlay_renders():
    """Mock layer ON + Jacobians: the results render with the mock layer and
    the seeded mock export, with no exception from the recovery overlay."""
    out, out_meta = _synthetic_out(sigma_detect=8.0, with_jac=True)
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["out"] = out
    at.session_state["out_meta"] = out_meta
    at.session_state["n0_shownoise"] = True
    at.session_state["n0_seed"] = 3
    at.run()
    assert not at.exception, at.exception
    dl = {b.label for b in at.get("download_button")}
    assert "Mock observation (CSV)" in dl
    _exps = [e.label for e in at.get("expander")]
    assert any("forecast summary" in e for e in _exps), _exps


def test_emission_mode_archive_fill_skips_transit_duration():
    """The archive duration is the PRIMARY-TRANSIT duration; in emission
    mode the fill must leave the event-duration widget alone and say why."""
    at = _run_app()
    at.selectbox(key="n0_planet").set_value("custom").run()
    at.radio(key="n0_scimode").set_value("emission").run()
    t14_before = at.number_input(key="n0_custom_t14").value
    at.selectbox(key="n0_custom_arch_name").set_value("HD 189733 b")
    at.button(key="n0_custom_arch_fill").click().run()
    assert not at.exception, at.exception
    from jwst_tool import archive
    values, _ = archive.custom_fill(archive.lookup("HD 189733 b"))
    assert at.number_input(key="n0_custom_t14").value == t14_before
    assert at.number_input(key="n0_custom_teff").value == values["teff"]
    warns = " | ".join(w.value for w in at.warning)
    assert "secondary-eclipse duration can differ" in warns


def test_removed_sections_and_prose_are_gone():
    """2026-08-13 UI cleanup (maintainer): the Condensation and 'Boundary
    conditions & escape' expanders are removed, More settings carries no help
    tooltips, and two prose blocks are deleted.

    Kept structural, not cosmetic: each assertion names something a future
    edit could silently reintroduce.
    """
    at = _run_app()
    assert not at.exception, at.exception
    labels = [e.label for e in at.get("expander")]
    for gone in ("Condensation (detection goals only)",
                 "Boundary conditions & escape"):
        assert gone not in labels, f"{gone!r} expander is back"
    # the widgets behind them must not exist either
    keys = {w.key for w in at.get("checkbox")} | {
        w.key for w in at.get("multiselect")}
    assert "n0_conden" not in keys and "n0_settle" not in keys
    assert "n0_descape" not in keys
    # ExoJAX caption + the intro honesty paragraph
    md = " ".join(m.value for m in at.markdown)
    caps = " ".join(c.value for c in at.get("caption"))
    body = md + " " + caps
    assert "modeling choices that can move the spectrum" not in body
    assert "Read the results as optimistic" not in body
    assert "Treat mode rankings as more robust" not in body


def test_more_settings_widgets_carry_no_help_tooltips():
    """No '?' icons under More settings: the labels stand alone and the
    reference material lives in README.md (GUI prose policy)."""
    at = _run_app()
    assert not at.exception, at.exception
    for key in ("n0_nz", "n0_yconv", "n0_rtptop", "n0_rtint", "n0_rtdit"):
        found = [w for w in at.get("number_input") + at.get("selectbox")
                 if w.key == key]
        if not found:            # nz_pic under the picaso provider
            continue
        assert not getattr(found[0], "help", None), \
            f"{key} regained a help tooltip"


def test_intro_prose_is_short_and_carries_no_methodology():
    """2026-08-13 (maintainer): the model expander is retitled and its five
    steps rewritten in STE; the Fisher caveat block drops to four bullets.

    Inverted assertions -- the REMOVAL is the requirement, so a future edit
    that restores the long form fails here. The displaced methodology (the
    differentiation method, the lnR0/per-segment nuisance design with its
    citations, the elemental-set provenance) lives in README.md.
    """
    at = _run_app()
    assert not at.exception, at.exception
    page = " ".join(m.value for m in at.markdown)
    assert "and its limits" not in page
    # cut phrasings
    for gone in ("steady-state photochemical kinetics",
                 "radiative-convective climate profile",
                 "conditional template S/N per molecule"):
        assert gone not in page, f"long-form intro phrase survived: {gone!r}"
    # kept: engine names and the LIVE Pandeia version (never hardcoded)
    for kept in ("VULCAN", "PICASO", "ExoJAX", "PHOENIX", "Pandeia",
                 "PandExo"):
        assert kept in page, f"engine name lost: {kept!r}"
    import re as _re
    # ins.BACKEND_STATUS.split(" /")[0] == "Pandeia 2026.7": a live version,
    # never a hardcoded one. Match the year so a literal string would fail.
    assert _re.search(r"Pandeia \d{4}\.\d", page), \
        "the live Pandeia version is no longer interpolated"


def test_summary_axis_controls_replace_the_custom_slider():
    """The wavelength radio carries two options and the Custom slider is gone;
    log/linear checkboxes for both spectrum axes take its place."""
    out, out_meta = _synthetic_out(sigma_detect=8.0, with_jac=True)
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["out"] = out
    at.session_state["out_meta"] = out_meta
    at.run()
    assert not at.exception, at.exception
    radio = at.radio(key="n0_sum_wlmode")
    assert list(radio.options) == ["Fit to selected modes", "Full model"], \
        radio.options
    assert not [s for s in at.slider if s.key == "n0_sum_wlrange"], \
        "the Custom wavelength slider should be gone"
    assert at.checkbox(key="n0_sum_xlog").value is True    # log wavelength
    assert at.checkbox(key="n0_sum_ylog").value is False   # linear depth


def test_all_usable_row_is_renamed_and_combos_lead_the_table():
    """'ALL USABLE (combined)' reads 'All usable modes', a custom set shows the
    user's own name with no 'COMBO: ' prefix, and combination rows sort to the
    TOP of the constraint table."""
    out, out_meta = _synthetic_out(sigma_detect=8.0, with_jac=True)
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["out"] = out
    at.session_state["out_meta"] = out_meta
    at.session_state["n0_combos"] = [dict(name="My set",
                                          modes=["nirspec_g395h",
                                                 "nirspec_prism"])]
    at.run()
    assert not at.exception, at.exception
    # the CONSTRAINT table specifically -- Mode details also has a "mode"
    # column and renders first, so identify by the parameter column
    tables = [df.value for df in at.dataframe
              if "mode" in getattr(df.value, "columns", [])
              and "parameter" in getattr(df.value, "columns", [])]
    assert tables, "no constraint-forecast table rendered"
    modes = [str(v) for v in tables[0]["mode"]]
    assert not any("ALL USABLE" in m for m in modes), modes[:8]
    assert not any(m.startswith("COMBO:") for m in modes), modes[:8]
    assert any("All usable modes" == m for m in modes), modes[:8]
    # combos lead: the first non-blank mode label is the combination
    first = next(m for m in modes if m.strip())
    assert "My set" in first, f"combos should lead the table, got {first!r}"
