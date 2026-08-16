"""GUI smoke tests: the app renders end-to-end with no exception.

Needs the GUI extras (streamlit + pandas); the dependency-light CI skips it.
Uses Streamlit's AppTest; no forward-model run is launched.
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
    ``with_jac=True`` adds a two-parameter Jacobian (lnZ, dlnCO + lnR0) to
    the model and a SECOND mode, so the Fisher table, the combo builder,
    the forecast posteriors, and the summary figure all render.
    """
    import json

    n = 40
    wl = np.linspace(1.0, 5.0, n)
    model = {
        "wl_um": wl,
        "depth": np.full(n, 0.021) + 1e-4 * np.sin(wl),
        "mols": np.array(["H2O", "CO2", "CO", "CH4", "SO2"], dtype="U8"),
        "depth_wo": np.tile(np.full(n, 0.0208), (5, 1)),
        "T": np.full(30, 1100.0),
        "p_bar": np.logspace(-7, 0.8, 30),
        # the mixing-ratio panel only renders when this is present, and it is
        # the one figure with its own log x axis -- without it a whole panel
        # (and the axis controls that drive it) was untested
        # the FULL network state, wider than the RT molecule list and in a
        # different order -- which is the shape that exposed the mislabelling
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
    at = _run_with_result(out, out_meta)
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
    at = _run_app()
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
    at = _run_app()
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


def test_global_noise_multiplier_composes_and_round_trips():
    """One "Noise multiplier" knob beside Jitter scales EVERY mode's random
    noise (maintainer, 2026-08-13: "a simple knob to increase jitter").

    It scales the NOISE MODEL, not the draw: posteriors.mock_realization
    refuses a draw-only factor, because a 2x draw beside 1x error bars is not a
    realization of the plotted model. Two things this pins:

    1. the global knob COMPOSES with the per-mode multipliers rather than
       overriding them (neither control is a hidden override), and
    2. the recorded config stores the PER-MODE values plus the global scale
       separately -- writing the product into noise_infl would re-multiply by
       the global scale on every restore, since share_config restores
       noise_infl into the per-mode widgets.
    """
    at = _run_app()
    assert not at.exception, at.exception
    sld = at.slider(key="n0_noisescale")
    assert sld.value == 1.0, sld.value
    assert sld.label == "Noise multiplier", sld.label

    src = APP.read_text()
    # composition, not replacement
    assert "infl = {k: float(noise_scale) * float(_infl_mode[k])" in src, \
        "the global multiplier no longer composes with the per-mode ones"
    # the recorded config keeps the two factors apart
    assert "noise_infl={k: float(_infl_mode[k]) for k in mode_keys}" in src, \
        "noise_infl must record the PER-MODE widget values, not the product"
    assert "noise_scale=float(noise_scale)," in src, \
        "the global scale is not recorded, so a run cannot be reproduced"


def test_figures_render_at_a_fixed_width_not_stretched():
    """Every figure gets an explicit pixel width (maintainer, 2026-08-13: "the
    whole thing wiggles when I change the size of the screen").

    st.pyplot's width DEFAULTS to "stretch", so simply omitting the argument
    does NOT stop the rescaling -- the value has to be passed. Each figure's
    text is baked in at fixed points, so a stretched raster changes its label
    sizes relative to everything else on every resize. Streamlit clamps an
    explicit width to the container on a narrower window, so nothing overflows.
    """
    src = APP.read_text()
    assert "st.pyplot(fig, width=_FIG_DISPLAY_PX)" in src, \
        "the tight branch no longer pins the display width"
    assert "st.pyplot(fig, width=_FIG_DISPLAY_PX,\n" in src, \
        "the full-canvas branch no longer pins the display width"
    assert 'st.pyplot(fig, width="stretch"' not in src, src[:0]
    # defined BEFORE _show_fig uses it (a later definition is a NameError on
    # the first figure -- the same ordering bug _COMBO_COLORS hit)
    lines = src.splitlines()
    define = next(i for i, l in enumerate(lines)
                  if l.startswith("_FIG_DISPLAY_PX"))
    use = next(i for i, l in enumerate(lines)
               if "st.pyplot(fig, width=_FIG_DISPLAY_PX)" in l)
    assert define < use, (define, use)


def test_fisher_table_is_static_and_cannot_be_resorted():
    """The Fisher table renders through st.table, never st.dataframe
    (maintainer, 2026-08-13: "if you sort by something else you can't see which
    mode does what. I just don't want to allow that reordering").

    st.dataframe sorts on a header click. This table cannot survive that: the
    mode column is BLANKED on repeat rows so the mode boundaries read clearly,
    so half the rows carry no mode name and only mean anything in their emitted
    order. A re-sort detaches every blank row from its mode and silently
    attributes numbers to the wrong instrument. st.table is static by
    definition, which removes the interaction instead of trying to make the
    blanks survive it.

    The other results tables are deliberately LEFT sortable -- they carry one
    self-contained row per mode, so re-sorting them loses nothing.
    """
    src = APP.read_text()
    i = src.index('with st.expander("Parameter constraint forecast (Fisher)")')
    j = src.index('st.download_button("Constraint forecast (CSV)"', i)
    block = src[i:j]
    assert "st.table(" in block, \
        "the Fisher table no longer renders through st.table"
    assert "st.dataframe(" not in block, \
        "the Fisher table is interactive again -- a header sort would " \
        "detach the blanked mode names from their rows"
    # the blanking that makes a re-sort unsafe is still what it protects
    assert '_r2["mode"] = ""' in block, \
        "mode names are no longer blanked; re-check whether st.table is needed"
    # the flip side: the DISPLAY blanks repeated mode names, the CSV must not
    # (a machine-readable file with an empty key column on half its rows is
    # unusable), so the download is built from the UNBLANKED rows
    assert "_csv_bytes(pd.DataFrame(frows))" in src, \
        "the Fisher CSV is no longer built from the unblanked rows"


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
    at = _run_with_result(out, out_meta)
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
    at = _run_with_result(out, out_meta)
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
    sf.compose_summary_figure = _spy
    try:
        at = _run_with_result(out, out_meta)
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
    at = _run_with_result(out, out_meta)
    assert not at.exception, at.exception
    assert any("No signal" in w.value for w in at.warning)
    assert not any("Best mode" in e.value for e in at.error)


def test_picaso_provider_renders(monkeypatch):
    # Switching the engine to PICASO must render; canonical_params failures
    # surface through the params_error caption, never an exception.
    monkeypatch.setenv("JWST_TOOL_ENABLE_UNCERTIFIED_PICASO", "1")
    at = _run_app()
    at.selectbox(key="n0_provider").set_value("picaso")
    at.run()
    assert not at.exception, at.exception


def test_picaso_climate_mode_renders(monkeypatch):
    monkeypatch.setenv("JWST_TOOL_ENABLE_UNCERTIFIED_PICASO", "1")
    at = _run_app()
    at.selectbox(key="n0_wasp39b_tp").set_value("picaso_climate")
    at.run()
    assert not at.exception, at.exception


def test_picaso_constrain_goal_renders(monkeypatch):
    # Regression: the constrain-goal Fisher multiselect crashed under PICASO
    # (default lnKzz not in the menu) -- both switch orders must render.
    monkeypatch.setenv("JWST_TOOL_ENABLE_UNCERTIFIED_PICASO", "1")
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


def test_picaso_detect_fisher_checkbox_renders(monkeypatch):
    # same defect on the detect goal's "Compute parameter constraints too"
    monkeypatch.setenv("JWST_TOOL_ENABLE_UNCERTIFIED_PICASO", "1")
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
    at.selectbox(key="n0_custom_arch_name").set_value("HD 189733 b").run()
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
    at = _run_with_result(out, out_meta, n0_shownoise=True, n0_seed=7)
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
    at = _run_with_result(out, out_meta)
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
    # at.TABLE, not at.dataframe: the Fisher table renders through st.table so
    # a header click cannot re-sort it away from its blanked mode names
    tables = [pd.DataFrame(d.value)
              for d in list(at.get("table")) + list(at.get("dataframe"))]
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
    at = _run_with_result(out, out_meta,
                          n0_combos=[dict(name="X", modes=["nirspec_g395h"])])
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
    at = _run_with_result(out, out_meta, n0_shownoise=True, n0_seed=3)
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
    at.selectbox(key="n0_custom_arch_name").set_value("HD 189733 b").run()
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
    # no '?' icons under More settings: the labels stand alone and the
    # reference material lives in README.md (GUI prose policy)
    for key in ("n0_nz", "n0_yconv", "n0_rtptop", "n0_rtint", "n0_rtdit"):
        found = [w for w in at.get("number_input") + at.get("selectbox")
                 if w.key == key]
        if not found:            # nz_pic under the picaso provider
            continue
        assert not getattr(found[0], "help", None), \
            f"{key} regained a help tooltip"


def test_intro_prose_is_short_and_carries_no_methodology():
    """2026-08-13 (maintainer): the model expander is retitled and its five
    steps rewritten in STE; the Fisher caveat block drops to four bullets;
    the beta disclaimer and the numerical-quality-checks sentence are gone.

    Inverted assertions -- the REMOVAL is the requirement, so a future edit
    that restores the long form fails here. The displaced methodology (the
    differentiation method, the lnR0/per-segment nuisance design with its
    citations, the elemental-set provenance) lives in README.md.
    """
    at = _run_app()
    assert not at.exception, at.exception
    page = " ".join(m.value for m in at.markdown)
    assert "and its limits" not in page
    # cut phrasings (incl. the beta + quality-check paragraphs)
    for gone in ("steady-state photochemical kinetics",
                 "radiative-convective climate profile",
                 "conditional template S/N per molecule",
                 "Beta", "full retrieval",
                 "numerical quality checks", "uncertified spectrum"):
        assert gone not in page, f"long-form intro phrase survived: {gone!r}"
    # kept: engine names and the LIVE Pandeia version (never hardcoded).
    # PHOENIX left this list 2026-08-13 when the maintainer rewrote the five
    # steps: "compute a spectrum using ExoJAX" no longer names the stellar
    # model. It is still disclosed in README.md, which is where the house
    # policy puts detail that does not change a control's effect.
    for kept in ("VULCAN", "PICASO", "ExoJAX", "Pandeia", "PandExo"):
        assert kept in page, f"engine name lost: {kept!r}"
    _readme = (APP.parent.parent.parent / "README.md").read_text()
    assert "PHOENIX" in _readme, "the stellar model is now disclosed nowhere"
    # ins.BACKEND_STATUS.split(" /")[0] == "Pandeia 2026.7": a live version,
    # never a hardcoded one. Match the year so a literal string would fail.
    assert re.search(r"Pandeia \d{4}\.\d", page), \
        "the live Pandeia version is no longer interpolated"


def test_every_axis_control_is_a_typed_min_max_pair():
    """No slider anywhere near an axis (maintainer, 2026-08-14). Every axis on
    every figure is two number boxes, min and max, blank for automatic.

    The wavelength window came BACK as a control in that pass, after being
    removed on 2026-08-13. The thing that was wrong with it then was the
    RADIO + range-slider pair, not the idea: a reviewer specifically asked to
    zoom into the PRISM and G395H regions. Blank boxes still fit the selected
    modes, so selecting modes remains a wavelength control on its own.
    """
    out, out_meta = _synthetic_out(sigma_detect=8.0, with_jac=True)
    at = _run_with_result(out, out_meta)
    assert not at.exception, at.exception
    assert not [r for r in at.radio if r.key == "n0_sum_wlmode"], \
        "the wavelength-range radio should be gone"
    # every axis pair: the results figure, the two structure panels. Each must
    # exist and START BLANK -- prefilled bounds would go stale the moment the
    # run or the mode selection changed, which is why blank is the default.
    for _k in ("sum_x", "sum_y", "struct_T", "struct_p", "struct_vmr",
               "sum_post_lnZ"):
        for _end in ("min", "max"):
            _w = at.number_input(key=f"n0_{_k}_{_end}")
            assert _w.value is None, f"{_k}_{_end} should start blank"
    # NO slider drives any axis. The only slider left on the page is the noise
    # scale, which is a model input.
    assert {s.key for s in at.slider} <= {"n0_noisescale"}, \
        [s.key for s in at.slider]
    _x = at.checkbox(key="n0_sum_xlog")
    _y = at.checkbox(key="n0_sum_ylog")
    assert _x.value is True and _x.label == "Log x", _x.label
    assert _y.value is False and _y.label == "Log y", _y.label


@pytest.mark.parametrize(
    "widget,value,field",
    # the molecule selectbox keys on the extra-molecule selection, so the key
    # carries the default extra set (jwst_tool.app: K("mol_<provider>_<mols>"));
    # CS2/C2H6 left the default at v31 (no ExoMolOP table)
    [("n0_mol_vulcan_C2H2_C2H4_H2S_HCN_NH3_OCS", "CO2",
      "target_mol"),                                # the reported bug
     ("n0_ntr", 4, "n_transits"),
     ("n0_rbin", 60, "r_bin"),
     ("n0_noisescale", 2.0, "noise_scale")])
def test_changing_any_run_input_marks_the_result_stale(widget, value, field,
                                                       monkeypatch):
    """The staleness guard compares the WHOLE non-canonical input set, not a
    hand-picked subset.

    The bug this pins: the detection target is deliberately absent from the
    canonical model params (one run computes the removed-molecule curve for
    every RT molecule), so switching SO2 to CO2 without pressing Run left the
    curve, the legend and the CSV column on the previous target with no notice.
    It was reported as a caching bug. The noise scale and R are here because a
    subset guard is exactly how the next one of these gets missed.
    """
    from jwst_tool import share_config as _sc
    seen = {}
    _real = _sc.build_share

    def _spy(*, canon, goal, observation, **kw):
        # the app builds ONE description of the run and hands the same two
        # blocks to the shareable config and to the staleness guard
        seen["run_sig"] = dict(goal=goal, observation=observation)
        seen["canon"] = canon
        return _real(canon=canon, goal=goal, observation=observation, **kw)

    monkeypatch.setattr(_sc, "build_share", _spy)
    out, out_meta = _synthetic_out(sigma_detect=8.0, with_jac=True)
    at = _run_with_result(out, out_meta)
    assert not at.exception, at.exception
    # Adopt the page's OWN canonical params and run signature as the cached
    # run's, so an untouched sidebar reads as fresh and the ONLY difference
    # below is the widget under test.
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
                                       ("n0_struct_p", "Pressure"),
                                       ("n0_struct_vmr", "Mixing ratio")])
def test_a_nonpositive_bound_on_a_log_axis_warns_instead_of_killing_the_page(
        key, label):
    """Three of the axes are LOG, and a min at or below zero has no logarithm.

    The figure builders raise on it, which is right for an API backstop, but
    that exception reaches Streamlit uncaught and takes the ENTIRE results page
    down -- spectrum, tables, downloads. A typed number is a user choice, not a
    defect. Warn, fall back to the automatic fit, keep the page alive.
    """
    out, out_meta = _synthetic_out(sigma_detect=8.0, with_jac=True)
    at = _run_with_result(out, out_meta,
                          **{f"{key}_min": 0.0, f"{key}_max": 5.0})
    assert not at.exception, at.exception
    assert any("log axis" in w.value and label in w.value for w in at.warning), \
        [w.value for w in at.warning]


def test_summary_refuses_a_one_sided_depth_range():
    """One bound alone has no second edge, and the auto fit lives in
    summary_figure. The page must SAY it fell back rather than invent the
    other bound -- and it must still render the figure."""
    out, out_meta = _synthetic_out(sigma_detect=8.0, with_jac=True)
    at = _run_with_result(out, out_meta,
                          n0_sum_y_min=20500.0)     # max left blank
    assert not at.exception, at.exception
    assert any("both boxes" in w.value for w in at.warning), \
        [w.value for w in at.warning]
    # and the COMPLETE pair is accepted silently, so the warning above is a
    # real refusal and not something the page says either way
    at.session_state["n0_sum_y_max"] = 21500.0
    at.run()
    assert not at.exception, at.exception
    assert not [w for w in at.warning if "Depth range" in w.value], \
        [w.value for w in at.warning]


def test_typed_axis_windows_reach_the_figure(monkeypatch):
    """The zoom the reviewer asked for: typing 3.0 to 5.5 um must reach the
    figure, not merely be accepted by the widget. A control that validates and
    is then dropped looks identical from the widget side, so this intercepts
    the call the page actually makes. The app closes every Figure it builds,
    which is why this records the ARGUMENTS rather than inspecting axes.
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
    out, out_meta = _synthetic_out(sigma_detect=8.0, with_jac=True)
    at = _run_with_result(out, out_meta,
                          n0_sum_x_min=3.0, n0_sum_x_max=5.5,
                          n0_sum_y_min=20500.0, n0_sum_y_max=21500.0,
                          n0_sum_post_lnZ_min=-0.4, n0_sum_post_lnZ_max=0.4)
    assert not at.exception, at.exception
    assert seen["wl_range"] == pytest.approx((3.0, 5.5))
    assert seen["depth_range"] == pytest.approx((20500.0, 21500.0))
    assert seen["panel_xlims"][0] == pytest.approx((-0.4, 0.4))


def test_all_usable_row_is_renamed_and_combos_lead_the_table():
    """'ALL USABLE (combined)' reads 'All usable modes', a custom set shows the
    user's own name with no 'COMBO: ' prefix, and combination rows sort to the
    TOP of the constraint table."""
    out, out_meta = _synthetic_out(sigma_detect=8.0, with_jac=True)
    at = _run_with_result(out, out_meta,
                          n0_combos=[dict(name="My set",
                                          modes=["nirspec_g395h",
                                                 "nirspec_prism"])])
    assert not at.exception, at.exception
    # the CONSTRAINT table specifically -- Mode details also has a "mode"
    # column and renders first, so identify by the parameter column
    # at.TABLE: the constraint forecast is a STATIC table (st.table), so a
    # reader cannot re-sort it and detach the blanked mode names from their rows
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
    # combos lead: the first non-blank mode label is the combination
    first = next(m for m in modes if m.strip())
    assert "My set" in first, f"combos should lead the table, got {first!r}"


def test_column_bottom_follows_the_geometry():
    """The emission column has to be an order of magnitude deeper than the
    transmission one, and a Streamlit widget takes its default only on FIRST
    render. One shared key would therefore have kept 7.6 bar after a switch to
    emission and refused the run for being transparent -- the exact failure
    this release exists to remove. Each geometry keeps its own box."""
    from jwst_tool import forward

    at = _run_app()
    assert not at.exception, at.exception
    t = at.number_input(key="n0_pbtm_transmission")
    assert t.value == forward.P_BTM_TRANSMISSION_BAR
    at.radio(key="n0_scimode").set_value("emission").run()
    assert not at.exception, at.exception
    e = at.number_input(key="n0_pbtm_emission")
    assert e.value == forward.P_BTM_EMISSION_BAR
    # and the transmission box is untouched when we go back
    at.radio(key="n0_scimode").set_value("transmission").run()
    assert at.number_input(key="n0_pbtm_transmission").value == \
        forward.P_BTM_TRANSMISSION_BAR


def test_mixing_ratio_panel_selects_species_by_name(monkeypatch):
    """ymix is the FULL network state (89 species for SNCHO); model["mols"] is
    the short RT list. Zipping them positionally read the wrong species.

    Found 2026-08-14 by comparing this tool's WASP-39 b chemistry against Tsai
    et al. 2023's published VULCAN run for the same planet: the curve the panel
    labelled CO2 sat at 0.847 at 1 bar, which is H2. The panel and its CSV were
    mislabelled for every species.
    """
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
    # the fixture puts H2 first at 0.85; it is NOT an RT molecule and must not
    # appear, and no RT molecule may take its column
    assert "H2" not in got
    assert set(got) == {"H2O", "CO2", "CO", "CH4", "SO2"}, got
    assert got["CO"] == pytest.approx(1e-3)     # by NAME, not by position
    assert got["H2O"] == pytest.approx(1e-4)
    assert got["CO2"] == pytest.approx(1e-6)
    assert 0.85 not in got.values()
