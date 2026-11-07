"""GUI smoke tests: the app renders end-to-end with no exception.

Needs the GUI extras (streamlit + pandas); the CI unit lane skips it and the
gui-smoke lane runs it. Uses Streamlit's AppTest; no forward-model run is
launched. AppTest boots are the expensive part, so every assertion that
examines the same booted app lives in one test.
"""

from __future__ import annotations

import ast
import re

import pytest

from jwst_tool import model_defs

st = pytest.importorskip("streamlit")
pytest.importorskip("pandas")
from app_helpers import APP, GUI_SOURCES, run_app
from streamlit.testing.v1 import AppTest


def test_fresh_boot_pre_run_contract():
    """One default boot: no intro gate, the speed-first mode trio (the ETC
    computes only the selected modes), the "Data" panel with its opacity
    sources, noise multiplier at 1.0, and the mock controls -- the seed
    field keeps a realization reproducible; widget keys are the shipped
    contract, so share_config round-trips.
    """
    from jwst_tool import instruments as ins

    at = run_app()
    assert not at.exception, at.exception
    assert at.selectbox(key="n0_planet").value is not None
    assert not [b for b in at.button if "I understand" in (b.label or "")]
    assert set(at.multiselect(key="n0_modes").value) == set(ins.DEFAULT_MODES)
    assert set(ins.DEFAULT_MODES) == {"nirspec_prism", "nirspec_g395h", "miri_lrs"}
    labels = [e.label for e in at.expander]
    assert any(label == "Data" or label.startswith("Data (") for label in labels)
    ms = [w for w in at.multiselect if w.label == "Extra opacity molecules"]
    assert ms, "extra-molecule multiselect missing"
    # Opacity data: exactly one of {the sources table, a loud provenance
    # warning} exists, depending on whether the fetcher's record is
    # installed; the picker's values and options are bare species tokens and
    # the Data table names the dataset behind each one.
    from jwst_tool import datacheck, forward, planets

    pp = datacheck.exomolop_provenance_path()
    have = bool(pp and pp.is_file())
    srcs = [
        d
        for d in at.dataframe
        if "component" in list(d.value.columns) and "used in this setup" in list(d.value.columns)
    ]
    warns = [w for w in at.warning if w.value.startswith("Opacity sources unavailable: ")]
    assert (len(srcs) == 1) == have and bool(warns) == (not have), (have, len(srcs), [w.value for w in warns])
    assert set(ms[0].value) == set(forward.EXTRA_MOLECULES_DEFAULT)
    if have:
        df = srcs[0].value
        offered = set(forward.MOLECULES) | (set(forward.EXTRA_MOLECULES) - forward._NO_EXOMOLOP_TABLE)
        assert offered <= set(df["component"])
        assert (df.loc[df["component"].isin(offered), "data set"] != "").all()
        # The picker shows the bare formula; the line list, isotopologue, DOI
        # and page each have a column in the Data table, checked just above.
        assert set(ms[0].options) <= set(forward.EXTRA_MOLECULES), ms[0].options
        # Every row is citable: a real DOI AND a real page, on the opacity
        # rows and the non-opacity data files alike. The regex also rejects
        # the four ExoMolOP placeholder headers -- note the LOWERCASE x in
        # 'x.xxxx/yyyyy' and 'xxxxxxx/xxxxxxxxx/xxxxxx', which a "10.xxxx"
        # filter would miss. This is the whole no-blank-cells contract.
        bad = [
            (c, d, u)
            for c, d, u in zip(df["component"], df["source DOI"], df["source page"], strict=True)
            if not re.fullmatch(r"10\.\d{4,9}/\S+", str(d)) or not str(u).startswith("http")
        ]
        assert not bad, bad
        # A new shipped UV spectrum must bring its citation with it, or the
        # table would KeyError in front of a user.
        assert set(planets.SFLUX_SOURCES) == set(planets.SFLUX_CHOICES)
    sld = at.number_input(key="n0_noisescale")
    assert sld.value == 1.0 and sld.label == "Global noise multiplier", (sld.label, sld.value)
    assert at.checkbox(key="n0_shownoise").value is True  # ON by default
    assert at.number_input(key="n0_seed").value == 0
    # editing the seed still redraws without recomputing
    at.number_input(key="n0_seed").set_value(7).run()
    assert not at.exception, at.exception
    assert at.number_input(key="n0_seed").value == 7


def test_gui_structure_defaults_match_canonical_params():
    """GUI and API defaults must mean the same atmosphere on EVERY planet
    (a hard-coded API T_irr would sit 470 K off the widget on HD 209458 b).
    """
    from jwst_tool import forward, planets

    at = run_app()
    assert not at.exception, at.exception
    for key in planets.PLANETS:
        at.selectbox(key="n0_planet").set_value(key).run()
        assert not at.exception, (key, at.exception)
        cp = forward.canonical_params(dict(planet=key))
        assert at.selectbox(key=f"n0_{key}_tp").value == cp["tp_mode"], key
        if cp["tp_mode"] == "file":
            assert at.selectbox(key=f"n0_{key}_kzzmode").value == cp["kzz_mode"] == "file", key
        # Force Guillot so the T_irr widget renders; compare against
        # canonical_params, never the helper the widget itself calls.
        at.selectbox(key=f"n0_{key}_tp").set_value("guillot").run()
        assert not at.exception, (key, at.exception)
        api_tirr = forward.canonical_params(dict(planet=key, tp_mode="guillot"))["Tirr"]
        assert at.number_input(key=f"n0_{key}_tirr").value == api_tirr, key
        at.selectbox(key=f"n0_{key}_tp").set_value(cp["tp_mode"]).run()


def test_every_download_button_is_plain_and_click_safe():
    """Clicking a download widget queues a rerun, and a queued rerun cancels
    the script run in flight -- on the Run pass, the forward model itself.
    on_click="ignore" is what keeps the buttons live during a run instead of
    dead stand-ins, so EVERY download call must carry it (AppTest cannot
    click one, hence the per-call scan). Both pre-run downloads are plain,
    always-live download buttons.
    """
    calls = [
        node
        for path in GUI_SOURCES
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "download_button"
    ]
    assert len(calls) > 5, f"only {len(calls)} download calls found: the scan is not looking where it thinks it is"
    for call in calls:
        on_click = next((k.value for k in call.keywords if k.arg == "on_click"), None)
        assert isinstance(on_click, ast.Constant) and on_click.value == "ignore", (
            f"the download call at line {call.lineno} cancels a run in flight"
        )
    at = run_app()
    assert not at.exception, at.exception
    assert "Download configuration (JSON)" in {b.label for b in at.get("download_button")}
    # The upload applies only through the Populate button (a callback), and
    # the button waits for a file.
    _pop = [b for b in at.button if b.key == "n0_cfg_populate"]
    assert _pop and _pop[0].proto.disabled, "Populate button missing or live without a file"

    at2 = AppTest.from_file(str(APP), default_timeout=60)
    at2.session_state["n0_wasp39b_tp"] = "file"
    at2.session_state["n0_wasp39b_tpsrc"] = "upload"  # model_defs.TP_FILE_UPLOAD
    at2.run()
    assert not at2.exception, at2.exception
    assert "Download this example" in {b.label for b in at2.get("download_button")}


def test_sidebar_gating_geometry_boxes_ad_lock_and_floor():
    """One fresh boot, three gating behaviors.

    1. Each geometry keeps its OWN column-bottom box (shipped key contract),
       and the DEFAULT is structure-aware: W39b transmission defaults
       to its measured table's own bottom, emission (Guillot everywhere)
       to the round parametric default.
    2. Photochemistry is always settable: forcing it ON whenever a Jacobian
       is requested makes the photolysis-off carbon-rich path unreachable
       from the shipped defaults. AD with it off is refused at the Run block
       instead, by the message canonical_params raises.
    3. The noise floor defaults to "constant" -- the CONSERVATIVE side,
       since it claims LESS precision than "No floor"; "none" stays an
       explicit choice; "Wavelength table" with no upload is the one state
       that BLOCKS the run.
    """

    at = run_app()
    assert not at.exception, at.exception
    # 1. per-geometry column bottoms, structure-aware defaults
    assert at.number_input(key="n0_pbtm_transmission").value == model_defs.P_BTM_FILE_BAR
    at.radio(key="n0_scimode").set_value("emission").run()
    assert not at.exception, at.exception
    assert at.number_input(key="n0_pbtm_emission").value == model_defs.P_BTM_PARAMETRIC_BAR
    at.radio(key="n0_scimode").set_value("transmission").run()
    assert at.number_input(key="n0_pbtm_transmission").value == model_defs.P_BTM_FILE_BAR

    # 2. photochemistry stays settable, and AD photo-off is refused loudly
    at.radio(key="n0_goal").set_value("constrain").run()
    at.selectbox(key="n0_jacm").set_value("ad").run()
    assert not at.exception, at.exception
    # Baseline: AD + photo ON is ALLOWED, so whatever errors show here are
    # environment, not gating (CI has no data root and always shows the
    # required-data banner). Compare against this rather than against "no
    # error at all", which is unsatisfiable in CI.
    allowed_errors = {e.value for e in at.error}
    assert not at.checkbox(key="n0_photo").disabled
    at.checkbox(key="n0_photo").set_value(False).run()
    assert not at.exception, at.exception
    # the value STICKS across reruns (nothing overwrites it) ...
    assert at.checkbox(key="n0_photo").value is False
    at.radio(key="n0_scimode").set_value("transmission").run()  # plain rerun
    assert at.checkbox(key="n0_photo").value is False
    # ... and the run is blocked with one sentence naming the fix
    assert any("no self-check" in (e.value or "") for e in at.error), [e.value for e in at.error]
    # ... but the refusal covers CHEMISTRY rows only: an all-RT-only request
    # (a cloud-deck row, plus the auto-appended lnR0) carries no through-solver
    # tangent, so it must stay RUNNABLE with photolysis off and AD on. Assert
    # the error set is back to the ALLOWED baseline -- excluding just this one
    # message would pass while some other error blocked the Run.
    at.checkbox(key="n0_cloud").set_value(True).run()
    at.checkbox(key="n0_marg").set_value(False).run()
    at.selectbox(key="n0_gp_vulcan_file_1").set_value("log_kappa_cloud").run()
    assert not at.exception, at.exception
    assert {e.value for e in at.error} == allowed_errors, [e.value for e in at.error]
    at.checkbox(key="n0_cloud").set_value(False).run()
    at.checkbox(key="n0_marg").set_value(True).run()
    at.checkbox(key="n0_photo").set_value(True).run()
    at.radio(key="n0_goal").set_value("detect").run()
    assert not at.exception, at.exception

    # 3. noise-floor gate
    def _floor_error(app):
        return [e.value for e in app.error if "minimum noise floor" in (e.value or "").lower()]

    assert at.radio(key="n0_floormode").value == "constant"
    assert not _floor_error(at), [e.value for e in at.error]
    assert any((w.key or "").startswith("n0_floor_") for w in at.get("number_input")), "per-mode floor inputs gone"
    at.radio(key="n0_floormode").set_value("none").run()
    assert not at.exception, at.exception
    assert not _floor_error(at)
    at.radio(key="n0_floormode").set_value("file").run()
    assert not at.exception, at.exception
    assert _floor_error(at), [e.value for e in at.error]

    # 4. the out-of-transit baseline FOLLOWS T14 (PandExo convention) until
    #    the user edits it; after that the link is broken for good
    at.number_input(key="n0_wasp39b_t14").set_value(4.0).run()
    assert not at.exception, at.exception
    assert at.number_input(key="n0_wasp39b_tbase").value == 4.0
    at.number_input(key="n0_wasp39b_tbase").set_value(5.0).run()
    at.number_input(key="n0_wasp39b_t14").set_value(6.0).run()
    assert not at.exception, at.exception
    assert at.number_input(key="n0_wasp39b_tbase").value == 5.0


def test_engine_switch_keeps_each_engines_settings(monkeypatch):
    """PICASO is the second engine beside VULCAN (the default). Switching
    shows PICASO's own widgets (climate T-P with heat redistribution, a
    quench box instead of the Kzz profile menu, FD only, no lnKzz row, H2O as
    the detection target) and hides the VULCAN-only ones. Streamlit drops the
    state of a widget it does not render, so a round trip must find every
    setting of BOTH engines as it was, and the VULCAN differentiation method
    is never written by PICASO's FD-only widget. That includes the goal
    selections (each engine's own keys) and the Guillot boxes PICASO's climate
    profile hides. A missing PICASO setup blocks a PICASO Run with one
    sentence and leaves VULCAN alone.
    """
    from jwst_tool import datacheck, fisher, forward

    monkeypatch.setattr(
        datacheck,
        "picaso_missing",
        lambda: [
            datacheck.Item(
                key="picaso:python",
                label="PICASO python",
                status=datacheck.MISSING,
                required=False,
                detail="JWST_TOOL_PICASO_PYTHON not set",
            )
        ],
    )

    def _pic_error(app):
        return [e.value for e in app.error if "PICASO engine" in e.value]

    at = run_app()
    for widget, key, value in (
        ("selectbox", "n0_network", "ncho"),
        ("number_input", "n0_sza", 60.0),
        ("selectbox", "n0_wasp39b_tp", "guillot"),
        ("selectbox", "n0_wasp39b_kzzmode", "const"),
        ("number_input", "n0_wasp39b_kzz", 10.0),
        ("number_input", "n0_wasp39b_tirr", 1500.0),
        ("number_input", "n0_wasp39b_lk", -1.5),
        ("multiselect", "n0_fp_vulcan_guillot_0", ["lnZ", "Tirr"]),
    ):
        getattr(at, widget)(key=key).set_value(value).run()
    mol_key = next(w.key for w in at.selectbox if w.key.startswith("n0_mol_vulcan_ncho_"))
    at.selectbox(key=mol_key).set_value("CO2").run()
    assert not at.exception, at.exception
    vulcan = {
        k: at.session_state[k]
        for k in (
            "n0_network",
            "n0_sza",
            "n0_wasp39b_tp",
            "n0_wasp39b_kzz",
            "n0_jacm",
            "n0_wasp39b_tirr",
            "n0_wasp39b_lk",
            "n0_fp_vulcan_guillot_0",
            mol_key,
        )
    }
    assert vulcan["n0_jacm"] == "ad" and not _pic_error(at)

    at.selectbox(key="n0_provider").set_value("picaso").run()
    assert not at.exception, at.exception
    assert "Photochemistry & transport" not in [e.label for e in at.expander]
    assert (
        at.selectbox(key="n0_wasp39b_tp_picaso").value
        == forward.canonical_params(dict(planet="wasp39b", chem_provider="picaso"))["tp_mode"]
    )
    assert at.number_input(key="n0_rfacv_picaso_transmission").value == 0.5
    jm = at.selectbox(key="n0_jacm_picaso")
    assert jm.disabled and jm.options == ["Finite differences"]
    assert at.selectbox(key="n0_wasp39b_sflux").disabled
    assert "N2O" not in at.multiselect(key="n0_xmols_picaso").options
    assert at.selectbox(key="n0_mol_picaso_C2H2_C2H4_H2S_HCN_NH3_OCS_SH_SO").value == "H2O"
    assert _pic_error(at) and "PICASO python" not in _pic_error(at)[0]
    assert next(b for b in at.button if b.label == "Run").proto.disabled
    assert not [w for w in at.number_input if w.key == "n0_wasp39b_kzz"]
    for quench in (False, True):
        at.checkbox(key="n0_quench_picaso").set_value(quench).run()
        assert not at.exception, at.exception
        assert fisher.PARAM_LABELS["lnKzz"] not in at.multiselect(key="n0_fp_picaso_picaso_climate_0").options
    assert at.number_input(key="n0_wasp39b_kzz").value == 10.0  # shared box
    at.number_input(key="n0_rfacv_picaso_transmission").set_value(0.7).run()
    at.multiselect(key="n0_fp_picaso_picaso_climate_0").set_value(["lnZ", "dlnCO"]).run()
    at.radio(key="n0_scimode").set_value("emission").run()
    assert at.number_input(key="n0_rfacv_picaso_emission").value == 1.0
    at.radio(key="n0_scimode").set_value("transmission").run()
    assert at.session_state["n0_jacm"] == "ad"

    at.selectbox(key="n0_provider").set_value("vulcan").run()
    assert not at.exception, at.exception
    assert {k: at.session_state[k] for k in vulcan} == vulcan
    assert not _pic_error(at)
    at.selectbox(key="n0_provider").set_value("picaso").run()
    assert at.number_input(key="n0_rfacv_picaso_transmission").value == 0.7
    assert at.checkbox(key="n0_quench_picaso").value is True
    assert at.multiselect(key="n0_fp_picaso_picaso_climate_0").value == ["lnZ", "dlnCO"]
    # an uploaded-table choice survives a detour, never the shipped table instead
    at.selectbox(key="n0_provider").set_value("vulcan").run()
    at.selectbox(key="n0_wasp39b_tp").set_value("file").run()
    at.radio(key="n0_wasp39b_tpsrc").set_value(model_defs.TP_FILE_UPLOAD).run()
    for engine in ("picaso", "vulcan"):
        at.selectbox(key="n0_provider").set_value(engine).run()
    assert not at.exception, at.exception
    assert at.session_state["n0_wasp39b_tpsrc"] == model_defs.TP_FILE_UPLOAD


def test_the_global_noise_multiplier_composes_and_the_config_records_the_factors_apart(monkeypatch):
    """The run applies global x per-mode; the share file records the two apart, since the product would re-multiply
    on restore.
    """
    from jwst_tool import gui_run, share_config

    seen = {}
    real_request, real_share = gui_run.build_request, share_config.build_share

    def spy_request(t, a, g, o, s):
        seen["infl"] = dict(o.noise.infl)
        return real_request(t, a, g, o, s)

    def spy_share(*, canon, goal, observation, **kw):
        seen["observation"] = observation
        return real_share(canon=canon, goal=goal, observation=observation, **kw)

    monkeypatch.setattr(gui_run, "build_request", spy_request)
    monkeypatch.setattr(share_config, "build_share", spy_share)
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["n0_noisescale"] = 2.0
    at.session_state["n0_infl_nirspec_prism"] = 1.5
    at.run()
    assert not at.exception, at.exception
    assert seen["infl"]["nirspec_prism"] == pytest.approx(3.0)
    assert seen["observation"]["noise_infl"]["nirspec_prism"] == pytest.approx(1.5)
    assert seen["observation"]["noise_scale"] == pytest.approx(2.0)


def test_no_figure_goes_through_st_pyplot():
    """st.pyplot hardcodes 200 dpi; every figure is shown from the PNG bytes its download carries."""
    calls = [
        (path.name, node.lineno)
        for path in GUI_SOURCES
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "pyplot"
    ]
    assert not calls, calls
