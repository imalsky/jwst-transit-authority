"""GUI smoke tests: the app renders end-to-end with no exception.

Needs the GUI extras (streamlit + pandas); the dependency-light CI skips it.
Uses Streamlit's AppTest; no forward-model run is launched. AppTest boots are
the expensive part, so every assertion that examines the same booted app lives
in one test (maintainer: fewer, stronger tests).
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
        # depth_wo rows align with wo_mols, not mols
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
            "t_cycle_s": 11.0,
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


def test_fresh_boot_pre_run_contract():
    """One default boot: no intro gate, the speed-first mode trio (the ETC
    computes only the selected modes), the data-status panel + annotated
    molecules, noise multiplier at 1.0, and the mock controls -- there is no
    'New draw' button, the seed field keeps a realization reproducible;
    widget keys are the shipped contract, so share_config round-trips."""
    from jwst_tool import instruments as ins

    at = _run_app()
    assert not at.exception, at.exception
    assert at.selectbox(key="n0_planet").value is not None
    assert not [b for b in at.button if "I understand" in (b.label or "")]
    assert set(at.multiselect(key="n0_modes").value) == set(ins.DEFAULT_MODES)
    assert set(ins.DEFAULT_MODES) == {"nirspec_prism", "nirspec_g395h",
                                      "miri_lrs"}
    labels = [e.label for e in at.expander]
    assert any(label == "Data" or label.startswith("Data (")
               for label in labels)
    assert "Opacity sources" not in labels
    ms = [w for w in at.multiselect if w.label == "Extra opacity molecules"]
    assert ms, "extra-molecule multiselect missing"
    # Opacity data: exactly one of {the sources table, a loud provenance
    # warning} exists, depending on whether the fetcher's record is
    # installed; the picker's VALUES stay species tokens while its OPTIONS
    # name the dataset behind each one.
    from jwst_tool import datacheck, forward, planets
    pp = datacheck.exomolop_provenance_path()
    have = bool(pp and pp.is_file())
    srcs = [d for d in at.dataframe if "component" in list(d.value.columns)
            and "used in this setup" in list(d.value.columns)]
    warns = [w for w in at.warning
             if w.value.startswith("Opacity sources unavailable: ")]
    assert (len(srcs) == 1) == have and bool(warns) == (not have), \
        (have, len(srcs), [w.value for w in warns])
    assert set(ms[0].value) == set(forward.EXTRA_MOLECULES_DEFAULT)
    if have:
        df = srcs[0].value
        offered = (set(forward.MOLECULES)
                   | (set(forward.EXTRA_MOLECULES) - forward._NO_EXOMOLOP_TABLE))
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
        bad = [(c, d, u) for c, d, u in
               zip(df["component"], df["source DOI"], df["source page"])
               if not re.fullmatch(r"10\.\d{4,9}/\S+", str(d))
               or not str(u).startswith("http")]
        assert not bad, bad
        # A new shipped UV spectrum must bring its citation with it, or the
        # table would KeyError in front of a user.
        assert set(planets.SFLUX_SOURCES) == set(planets.SFLUX_CHOICES)
    sld = at.number_input(key="n0_noisescale")
    assert sld.value == 1.0 and sld.label == "Global noise multiplier", \
        (sld.label, sld.value)
    assert at.checkbox(key="n0_shownoise").value is True   # ON by default
    assert at.number_input(key="n0_seed").value == 0
    assert not [b for b in at.get("button") if b.key == "n0_reroll"], \
        "the 'New draw' button was removed (maintainer decision)"
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


def test_every_download_button_is_plain_and_click_safe():
    """Clicking a download widget queues a rerun, and a queued rerun cancels
    the script run in flight -- on the Run pass, the forward model itself.
    on_click="ignore" is what keeps the buttons live during a run instead of
    dead stand-ins, so EVERY download button must carry it (AppTest cannot
    click one, hence the source count). Both pre-run downloads are plain,
    always-live download buttons."""
    src = APP.read_text()
    assert src.count("download_button(") == src.count('on_click="ignore"'), \
        "a download button without on_click=ignore cancels a run in flight"
    at = _run_app()
    assert not at.exception, at.exception
    assert "Download configuration (JSON)" in {b.label
                                               for b in at.get("download_button")}
    # The upload applies only through the Populate button (a callback), and
    # the button waits for a file.
    _pop = [b for b in at.button if b.key == "n0_cfg_populate"]
    assert _pop and _pop[0].proto.disabled, "Populate button missing or live without a file"

    at2 = AppTest.from_file(str(APP), default_timeout=60)
    at2.session_state["n0_wasp39b_tp"] = "file"
    at2.session_state["n0_wasp39b_tpsrc"] = "upload"   # forward.TP_FILE_UPLOAD
    at2.run()
    assert not at2.exception, at2.exception
    assert "Download this example (edit and re-upload)" in \
        {b.label for b in at2.get("download_button")}


def test_sidebar_gating_geometry_boxes_ad_lock_and_floor():
    """One fresh boot, three gating behaviors.

    1. Each geometry keeps its OWN column-bottom box (shipped key contract),
       and the DEFAULT is structure-aware: W39b transmission defaults
       to its measured table's own bottom, emission (Guillot everywhere)
       to the round parametric default.
    2. Photochemistry is always settable -- it used to be force-locked ON
       whenever a Jacobian was requested, which made the photolysis-off
       carbon-rich path unreachable from the shipped defaults. AD with it off
       is refused at the Run block instead, by the message canonical_params
       already raises.
    3. The noise floor defaults to "constant" (maintainer) -- the
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

    # 2. photochemistry stays settable, and AD photo-off is refused loudly
    at.radio(key="n0_goal").set_value("constrain").run()
    at.selectbox(key="n0_jacm").set_value("ad").run()
    assert not at.exception, at.exception
    assert not at.checkbox(key="n0_photo").disabled
    at.checkbox(key="n0_photo").set_value(False).run()
    assert not at.exception, at.exception
    # the value STICKS (the lock used to overwrite it on every rerun) ...
    assert at.checkbox(key="n0_photo").value is False
    at.radio(key="n0_scimode").set_value("transmission").run()   # plain rerun
    assert at.checkbox(key="n0_photo").value is False
    # ... and the run is blocked with one sentence naming the fix
    assert any("photo-on regime" in (e.value or "") for e in at.error), \
        [e.value for e in at.error]
    at.checkbox(key="n0_photo").set_value(True).run()
    at.radio(key="n0_goal").set_value("detect").run()
    assert not at.exception, at.exception

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

    # 4. the out-of-transit baseline FOLLOWS T14 (PandExo convention) until
    #    the user edits it; after that the link is broken for good
    at.number_input(key="n0_wasp39b_t14").set_value(4.0).run()
    assert not at.exception, at.exception
    assert at.number_input(key="n0_wasp39b_tbase").value == 4.0
    at.number_input(key="n0_wasp39b_tbase").set_value(5.0).run()
    at.number_input(key="n0_wasp39b_t14").set_value(6.0).run()
    assert not at.exception, at.exception
    assert at.number_input(key="n0_wasp39b_tbase").value == 5.0


def test_source_pins_fig_width_fisher_table_and_noise_recording():
    """Source-level pins for three maintainer decisions: (1) the
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
    assert 'width=_FIG_DISPLAY_PX)' in src.split("def _show_fig")[1], \
        "the full-canvas branch no longer pins the display width"
    assert "_show_fig(fig3, tight=False, png=_struct_png)" in src, \
        "the structure figure must reuse the PNG the download button already " \
        "rasterized, not pay a second dpi-200 savefig of the same figure"
    assert 'st.pyplot(fig, width="stretch"' not in src
    assert not [ln for ln in src.splitlines()
                if "st.pyplot(" in ln and "bbox_inches" in ln], \
        "st.pyplot takes no savefig keywords (deprecated in Streamlit)"
    lines = src.splitlines()
    define = next(i for i, l in enumerate(lines)
                  if l.startswith("_FIG_DISPLAY_PX"))
    use = next(i for i, l in enumerate(lines)
               if "st.pyplot(fig, width=_FIG_DISPLAY_PX)" in l)
    assert define < use, (define, use)
    i = src.index('with st.expander("Parameter constraint forecast (local Fisher)")')
    j = src.index('st.download_button("Constraint forecast (CSV)"', i)
    block = src[i:j]
    assert "st.table(" in block and "st.dataframe(" not in block, \
        "the Fisher table must stay a static st.table (blanked mode names)"
    assert '_r2["mode"] = ""' in block, \
        "mode names are no longer blanked; re-check whether st.table is needed"
    assert "_csv_bytes(pd.DataFrame(frows))" in src, \
        "the Fisher CSV is no longer built from the unblanked rows"


def test_gap_band_labels_match_the_registry_bands():
    """The mode picker's hand-written H-grating band strings must keep the
    registry's own endpoints.

    A duplicated band statement is how the G395M red edge drifted to 5.10
    while its source (Birkmann et al. 2022 Table 2 / jwst-docs BOTS Table 1)
    said 5.18. These three strings restate a band the registry already owns,
    so pin the outer endpoints to it; the inner pair is the measured NRS1/NRS2
    gap and has no registry counterpart."""
    from jwst_tool import instruments as ins
    src = APP.read_text()
    i = src.index("_MODE_BAND_DISPLAY = {")
    block = src[i:src.index("}", i)]
    labels = dict(re.findall(r'"([a-z0-9_]+)":\s*"([^"]+)"', block))
    assert labels, "the band-label table moved or changed shape"
    for key, text in labels.items():
        assert key in ins.MODES, f"{key} is not a registry mode"
        edges = [float(x) for x in re.findall(r"\d+\.\d+", text)]
        assert len(edges) == 4, f"{key}: expected two sub-bands, got {text!r}"
        assert edges == sorted(edges), f"{key}: band edges out of order"
        m = ins.MODES[key]
        assert (edges[0], edges[-1]) == (m["wl_min"], m["wl_max"]), (
            f"{key}: label {text!r} disagrees with the registry band "
            f"{m['wl_min']}-{m['wl_max']}")


def test_mode_picker_native_r_labels_are_measured():
    """Pins the mode picker's measured native-R labels (r_native_med: display
    metadata measured from the 2026.7 refdata dispersion files -- re-measure
    on any refdata change, never edit the numbers freehand)."""
    from jwst_tool import instruments as ins
    for k, m in ins.MODES.items():
        assert int(m["r_native_med"]) > 0, f"{k} lacks r_native_med"
    # PRISM's median is taken over its registry band, which starts at the
    # model's 1.0 um short edge (not the instrument's 0.6 um)
    assert ins.MODES["nirspec_prism"]["r_native_med"] == 110
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
            "Native model (CSV)", "T-P values (CSV)",
            "Mixing ratios (CSV)"} <= dl_labels
    assert any("No signal" in w.value for w in at.warning)
    assert not any("Best mode" in e.value for e in at.error)


def test_emission_results_use_eclipse_terms():
    """An emission run says "eclipse" throughout, never "transit"; an
    above-target result renders NO verdict line and NO banner (maintainer:
    the figure and table carry the number)."""
    out, out_meta = _synthetic_out(science_mode="emission",
                                   sigma_detect=8.0, n_transits=3)
    at = _run_with_result(out, out_meta)
    assert not at.exception, at.exception
    assert not at.success, \
        f"an above-target result must render no banner: {[s.value for s in at.success]}"
    assert not [m.value for m in at.markdown if "template S/N 8.0" in m.value]
    # the figure section is an EXPANDER, not a subheader
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
    assert any("saturated at the shortest ramp" in w for w in warns), warns
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
    rule)."""
    at = _run_app()
    at.selectbox(key="n0_planet").set_value("custom").run()
    assert not at.exception, at.exception
    at.number_input(key="n0_custom_teff").set_value(3100.0).run()
    assert not at.exception, at.exception
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
    results section is an expander (renames pinned); a primed
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
    # the combo builder and the marginalized-forecast control were MERGED
    # into the constraint-forecast panel; neither is a section of its own
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
    assert "Figure (PDF, vector)" in {b.label
                                      for b in at.get("download_button")}

    # identify the CONSTRAINT table by its parameter column
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


@pytest.mark.parametrize(
    "widget,value,field",
    # the molecule selectbox keys on the extra-molecule selection, so the key
    # carries the default extra set (jwst_tool.app: K("mol_<provider>_<mols>"));
    # CS2/C2H6 are out of the default (no ExoMolOP table); SH+SO are in
    # (measured 10.0 and 48.1 ppm on W39b -- forward.EXTRA_MOLECULES_DEFAULT)
    [("n0_mol_vulcan_C2H2_C2H4_H2S_HCN_NH3_OCS_SH_SO", "CO2",
      "target_mol"),                                # the reported bug
     ("n0_noisescale", 2.0, "noise_scale"),         # an observation-block field
     # DISPLAY-ONLY: the seed redraws the mock from the cached run and
     # recomputes nothing, so it must never read as stale (field=None)
     ("n0_seed", 7, None)])
def test_changing_any_run_input_marks_the_result_stale(widget, value, field,
                                                       monkeypatch):
    """The staleness guard compares the WHOLE non-canonical input set minus
    the display-only fields, not a hand-picked subset. The bug this pins: the
    detection target is absent from the canonical model params, so switching
    SO2 to CO2 without pressing Run left the curve, legend and CSV on the
    previous target with no notice (reported as a caching bug). The noise
    scale represents the observation block because a subset guard is exactly
    how the next one gets missed; the seed represents the display-only
    fields, which change nothing that was computed."""
    from jwst_tool import share_config as _sc
    seen = {}
    _real = _sc.build_share

    def _spy(*, canon, goal, observation, **kw):
        # the app hands the same two blocks to the shareable config and to the
        # staleness guard, minus the display-only observation fields (app.py
        # _DISPLAY_ONLY), which the guard never compares
        seen["run_sig"] = dict(
            goal=goal,
            observation={k: v for k, v in observation.items()
                         if k not in ("show_noise", "seed", "combos")})
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
    if field is None:
        assert not _named(), \
            f"a display-only input must not mark the run stale: {_named()}"
    else:
        assert _named() and field in _named()[0], _named()


def test_a_run_that_produces_nothing_clears_the_previous_result(monkeypatch):
    """A refused or failed Run must not leave the previous run's verdict and
    figures on screen under a stale banner: the page shows only the error.
    Exercised through the one refusal that keeps the Run button enabled (no
    free concurrency slot); the forward-model failure branch pops the same
    keys."""
    from jwst_tool import runlimit as _rl

    monkeypatch.setattr(_rl, "acquire", lambda *_a, **_k: None)
    out, out_meta = _synthetic_out(sigma_detect=8.0, with_jac=True)
    at = _run_with_result(out, out_meta)
    assert not at.exception, at.exception
    next(b for b in at.button if b.label == "Run").click()
    at.run()
    assert not at.exception, at.exception
    assert "out" not in at.session_state
    assert any("already running" in e.value for e in at.error), [e.value for e in at.error]
    assert not [w.value for w in at.warning if "previous run" in w.value or "reach it" in w.value]


def test_a_nonpositive_bound_on_a_log_axis_warns_instead_of_killing_the_page():
    """A min at or below zero has no logarithm on a log axis. The builders
    raise on it -- right for an API backstop, but uncaught it takes the
    ENTIRE results page down. A typed number is a user choice, not a defect:
    warn, fall back to the automatic fit, keep the page alive.

    Only the summary figure carries axis controls now; the structure figure
    has none (it draws at the module defaults), so there is no typed bound
    left to poison there."""
    key, label = "n0_sum_x", "Wavelength"
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
    # Axis-control contract (maintainer): every axis is a
    # typed min/max number-box pair that STARTS blank, no sliders anywhere,
    # no wavelength-range radio.
    assert not [r for r in at.radio if r.key == "n0_sum_wlmode"]
    assert not at.slider, [s.key for s in at.slider]
    for _k in ("sum_x", "sum_post_lnZ"):
        for _end in ("min", "max"):
            assert at.number_input(key=f"n0_{_k}_{_end}").value is None, \
                f"{_k}_{_end} should start blank"
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
    is the short RT list. Zipping them positionally reads the wrong species:
    measured against Tsai et al. 2023's published WASP-39 b VULCAN run, the
    curve labelled CO2 sat at 0.847 at 1 bar, which is H2 -- the panel and
    its CSV were mislabelled for every species."""
    from jwst_tool import plotting
    seen = {}
    _real = plotting.build_structure_figure

    def _spy(p_bar, T_K, columns, **kw):
        seen["columns"] = list(columns)
        return _real(p_bar, T_K, columns, **kw)

    monkeypatch.setattr(plotting, "build_structure_figure", _spy)
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
