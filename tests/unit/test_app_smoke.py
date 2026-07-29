"""GUI smoke test: the app renders end-to-end with no exception.

Runs only where the GUI extras (streamlit + pandas) are installed -- the
dependency-light CI skips it. Uses Streamlit's AppTest; the pre-run render
path exercises the data-status panel (datacheck.full_report) and every
sidebar widget, without launching a forward-model run. (The mandatory intro
gate was removed in the 2026-07-29 UX pass: the first render IS the tool.)
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
    """A minimal cached-result pair (out, out_meta) for post-run rendering.

    sigma_detect stays either 0 (no-signal branch) or above the 3-sigma
    target (success branch) so the render never calls
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
        r_bin=100, planet="WASP-39 b", scenario="random",
        floor_mode="constant")
    return out, out_meta


def test_app_renders_without_exception():
    at = _run_app()
    assert not at.exception, at.exception


def test_gui_structure_defaults_match_canonical_params():
    """The GUI and the API must mean the SAME atmosphere by "the defaults".

    Checked on EVERY planet, not just the reference one: the drift this test
    exists to catch (a hard-coded API T_irr vs a T_eq-derived widget) was
    invisible on WASP-39 b and 470 K wide on HD 209458 b. Structure mode, Kzz
    mode, and T_irr must all agree, or a "default" GUI run and a "default"
    API run silently model different atmospheres.
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
        # T_irr: drive the GUI into Guillot mode so the widget renders, and
        # compare it to what the API would default to for THIS planet. Compare
        # against canonical_params, never against the same helper the widget
        # calls -- that would pass even while the API drifted.
        at.selectbox(key=f"n0_{key}_tp").set_value("guillot").run()
        assert not at.exception, (key, at.exception)
        api_tirr = forward.canonical_params(
            dict(planet=key, tp_mode="guillot"))["Tirr"]
        assert at.number_input(key=f"n0_{key}_tirr").value == api_tirr, key
        at.selectbox(key=f"n0_{key}_tp").set_value(cp["tp_mode"]).run()


def test_default_instrument_modes_are_the_observed_ones():
    from jwst_tool import instruments as ins

    at = _run_app()
    assert not at.exception, at.exception
    assert set(at.multiselect(key="n0_modes").value) == set(ins.DEFAULT_MODES)


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
    """Drive the full post-Run render path (spectrum + ranking + T-P figures,
    download buttons, mode table) on a synthetic result -- no forward model."""
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


def test_no_intro_gate():
    """The first render is the tool itself: sidebar widgets exist immediately,
    with no acknowledgment button blocking them (2026-07-29 UX pass)."""
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert at.selectbox(key="n0_planet").value is not None
    assert not [b for b in at.button if "I understand" in (b.label or "")]


def test_emission_results_use_eclipse_terms():
    """An emission run's verdict and spectrum header say "eclipse", never
    "transit" (2026-07-29 UX review, item 1.5), and a valid result that
    MEETS the target renders as success."""
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
    """When every selected mode saturates, the verdict is a warning that says
    so; no mode is presented as "best", and no error alert fires for what is
    a valid calculation (2026-07-29 UX review, items 1.7 and 1.10)."""
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
    # v18: switching the forward-model engine to PICASO must render (the
    # kinetics sections collapse to the provider caption; composition swaps
    # to the picaso ranges; the detect menu loses SO2). Works with or
    # without the reference tree: canonical_params failures surface through
    # the params_error caption, never an exception.
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
    # v18.1 regression: the constrain-goal Fisher multiselect crashed with
    # StreamlitAPIException under the PICASO engine (default lnKzz not in
    # the provider's menu) -- both switch orders must render cleanly.
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
    """The spectrum figure plots the model convolved to a display resolution
    (2026-07-29): at native sampling the unresolved line forest renders as
    one-sample spikes. Two properties the app relies on: the operator must
    not touch the caller's array (the "Native model (CSV)" download exports
    that same un-smoothed array), and at the app's display resolution it
    must measurably reduce the point-to-point spikiness -- otherwise the
    plot change is cosmetically inert and the caption would be wrong.
    """
    import numpy as np

    from jwst_tool import binning

    wl = np.geomspace(1.0, 12.0, 4000)          # the real native grid shape
    rng = np.random.default_rng(0)
    # smooth continuum plus an unresolved "line forest": isolated samples
    # spiking upward, which is what a saturated line core looks like once
    # its strength is conserved onto a grid ~200x coarser than the line
    native = (21000.0 + 200.0 * np.sin(8.0 * np.log(wl))
              + rng.choice([0.0, 0.0, 0.0, 600.0], size=wl.size))
    untouched = native.copy()

    r_disp = float(max(300, 3 * 100))           # the app's rule at R_bin=100
    smoothed = binning.smooth_to_native_r(
        wl, native, np.array([wl[0], wl[-1]]), np.array([r_disp, r_disp]),
        float(wl[0]), float(wl[-1]))

    assert np.array_equal(native, untouched), \
        "display smoothing mutated the caller's native model"
    # RMS of the point-to-point differences, NOT the median: three quarters
    # of the native samples share the same offset, so 62% of consecutive
    # pairs are exactly equal and the median jump is ~0 -- a baseline no
    # smoothed curve can beat. The RMS measures the spikiness that matters.
    rms_native = float(np.diff(untouched).std())
    rms_disp = float(np.diff(smoothed).std())
    assert rms_disp < 0.2 * rms_native, (rms_native, rms_disp)
    # and the convolution conserves the continuum level it is drawn on top of
    assert abs(smoothed.mean() / untouched.mean() - 1.0) < 1e-4


def test_ad_selection_locks_photochemistry_on():
    """Kzz/photochemistry moved into the Atmosphere step (2026-07-29), which
    renders BEFORE the differentiation method: the AD photo-lock now reads
    the effective method from session state. Selecting AD under a constrain
    goal must still force the photochemistry checkbox ON and disable it, and
    switching back to detect (no constraints requested) must release it."""
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
