"""GUI smoke tests: the app renders end-to-end with no exception.

Needs the GUI extras (streamlit + pandas); the dependency-light CI skips it.
Uses Streamlit's AppTest; no forward-model run is launched.
"""
from __future__ import annotations

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
                   sigma_detect=0.0, n_transits=1):
    """Minimal cached-result pair (out, out_meta) for post-run rendering.

    sigma_detect must stay 0 or above target so the render never calls
    detect.transits_to_target, which needs the full evaluate_mode payload.
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
                                   "science_mode": science_mode}),
    }
    nb = 12
    result = {
        "mode_key": "nirspec_g395h", "label": "NIRSpec G395H",
        "saturated": saturated, "sat_frac": 0.97,
        "sigma_detect": sigma_detect,
        "sigma_detect_proj": float("nan"),
        "wl": np.linspace(2.9, 5.1, nb), "wl_eff": np.linspace(2.9, 5.1, nb),
        "depth": np.full(nb, 0.021), "sigma": np.full(nb, 1.5e-4),
        "floor": np.zeros(nb),
        "median_sigma_ppm": 150.0, "n_bins": nb, "ngroup": 12,
        "t_cycle_s": 11.0, "warnings": (), "jac_bins": None,
    }
    out = dict(model=model, results=[result], failed=[], unusable=[],
               fisher_names=[], provenance=None)
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
    assert {"Figure (PNG)", "Binned points (CSV)", "Native model (CSV)",
            "Values (CSV)", "Mode details (CSV)"} <= dl_labels


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


def test_noise_floor_has_no_default_and_blocks_the_run():
    """No default floor: neither candidate is neutral, so the run is blocked
    until the user picks one (a 15-40 ppm minimum SETS the reported precision;
    no floor ignores 1/f and visit-long systematics)."""
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception, at.exception

    def _floor_error(app):
        return [e.value for e in app.error
                if "minimum noise floor" in (e.value or "").lower()]

    # nothing preselected, run blocked, and the block is explained
    assert at.radio(key="n0_floormode").value is None
    assert [b for b in at.button if (b.label or "") == "Run"][0].disabled
    assert _floor_error(at), [e.value for e in at.error]

    # "No floor" is a valid EXPLICIT choice: the floor gate clears. (Run is
    # also gated on params_error/modes, which depend on installed engine data.)
    at.radio(key="n0_floormode").set_value("none").run()
    assert not at.exception, at.exception
    assert not _floor_error(at)


def test_constant_floor_prefill_is_labeled_illustrative():
    """The 15-40 ppm prefill must never read as a calibration for the program."""
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    at.radio(key="n0_floormode").set_value("constant").run()
    assert not at.exception, at.exception
    caps = " ".join((c.value or "") for c in at.caption).lower()
    assert "illustrative" in caps
    assert "not in-flight calibrations" in caps


def test_emission_results_use_eclipse_terms():
    """An emission run's verdict and spectrum header say "eclipse", never
    "transit", and an above-target result renders as success."""
    out, out_meta = _synthetic_out(science_mode="emission",
                                   sigma_detect=8.0, n_transits=3)
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["out"] = out
    at.session_state["out_meta"] = out_meta
    at.run()
    assert not at.exception, at.exception
    succ = [s.value for s in at.success]
    assert succ, "expected a success verdict for an above-target score"
    assert any("3 eclipses" in s for s in succ), succ
    assert not any("transit" in s for s in succ), succ
    subs = [s.value for s in at.subheader]
    assert any("eclipse emission spectrum" in s for s in subs), subs


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
