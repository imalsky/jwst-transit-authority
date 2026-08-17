"""share_config: the download/upload configuration round-trip.

The mapping must restore exactly the widget state the download described, and
an invalid file must raise before anything could be applied. No Streamlit, no
pandeia, no JAX: the mapping is pure dict work over canonical params. Pruned
2026-08-15 (maintainer: fewer, stronger tests): refusal micro-tests are
merged into consolidated fail-closed tests; every assertion survives.
"""
import pytest

from jwst_tool import forward, share_config


def _key(name: str) -> str:              # the app's K() with nonce 0
    return f"n0_{name}"


def _canon(**over):
    base = dict(planet="wasp39b", tp_mode="guillot")
    base.update(over)
    return forward.canonical_params(base)


def test_round_trip_restores_the_full_run():
    """Constrain goal, detect goal, and a bare canonical dict all restore
    exactly the widget state the download described (widget KEYS are a
    shipped contract)."""
    canon = _canon(met_x_solar=7.0, co_ratio=0.6, kzz_mode="const",
                   kzz_const=1.0e9, extra_mols=["HCN", "NH3"],
                   cloud_on=True, log_kappa_cloud=-2.0, alpha_cloud=1.0,
                   fisher_params=["lnZ", "lnKzz"])
    share = share_config.build_share(
        canon,
        goal=dict(goal="constrain", goal_param="lnZ", target_prec=0.1,
                  target_sig=3.0, marginalize=True, do_fisher=False,
                  fisher_params=["lnZ", "lnKzz"], jac_method="fd"),
        observation=dict(ks_mag=9.0, t14=2.8, t_base=2.8, sat_limit=0.8,
                         star_teff=5485.0, star_logg=4.47, star_feh=0.0,
                         modes=["nirspec_g395h"], n_transits=2, r_bin=100,
                         floor_mode="constant",
                         floors={"nirspec_g395h": 15.0},
                         noise_infl={"nirspec_g395h": 1.05},
                         show_noise=False, seed=0))
    state, notes = share_config.widget_state(share, _key)
    assert not notes
    # target + system (per-planet keys); the Pandeia star restores from the
    # observation block (the transmission canonical block zeroes it)
    assert state["n0_planet"] == "wasp39b"
    assert state["n0_wasp39b_teff"] == 5485.0
    assert state["n0_wasp39b_g"] == pytest.approx(canon["gs_cgs"] / 100.0)
    assert state["n0_wasp39b_ks"] == 9.0 and state["n0_wasp39b_t14"] == 2.8
    # atmosphere
    assert state["n0_wasp39b_tp"] == "guillot"
    assert state["n0_wasp39b_tirr"] == canon["Tirr"]
    assert state["n0_met"] == 7.0 and state["n0_co"] == 0.6
    assert state["n0_wasp39b_kzz"] == pytest.approx(9.0)   # log10(kzz_const)
    assert state["n0_xmols_vulcan"] == ["HCN", "NH3"]
    assert state["n0_cloud"] is True and state["n0_ck"] == -2.0
    # science goal (dynamic widget-key suffixes)
    assert state["n0_goal"] == "constrain"
    assert state["n0_gp_vulcan_guillot_1_0"] == "lnZ"
    assert state["n0_tgt_lnZ"] == 0.1
    assert state["n0_fx_vulcan_guillot_1_0"] == ["lnKzz"]
    # observation
    assert state["n0_modes"] == ["nirspec_g395h"]
    assert state["n0_ntr"] == 2 and state["n0_rbin"] == 100
    assert state["n0_floormode"] == "constant"
    assert state["n0_floor_nirspec_g395h"] == 15.0
    assert state["n0_infl_nirspec_g395h"] == 1.05

    # a detect goal maps the per-selection molecule widget key
    share = share_config.build_share(
        _canon(extra_mols=["C2H2", "HCN"]),
        goal=dict(goal="detect", target_mol="SO2", target_sig=3.0,
                  marginalize=True, do_fisher=False, fisher_params=[],
                  jac_method="fd"),
        observation=dict(modes=["nirspec_prism"], n_transits=1,
                         star_teff=5485.0, star_logg=4.47, star_feh=0.0))
    state, notes = share_config.widget_state(share, _key)
    assert not notes
    assert state["n0_mol_vulcan_C2H2_HCN"] == "SO2"
    assert state["n0_dofish"] is False

    # a bare canonical dict is accepted
    state, _ = share_config.widget_state(_canon(), _key)
    assert state["n0_planet"] == "wasp39b"
    assert "n0_modes" not in state          # no observation section to restore


def test_invalid_input_is_refused_before_anything_applies():
    """Fail closed: malformed files, unknown parameters, a missing uploaded
    table, and an unsupported format version all raise before any state
    could be applied; recoverable unknown entries are dropped WITH a note,
    never silently."""
    with pytest.raises(ValueError):
        share_config.widget_state({"foo": 1}, _key)
    with pytest.raises(ValueError):
        share_config.widget_state(
            {"canonical_params": {"planet": "wasp39b", "nz": 5}}, _key)
    with pytest.raises(ValueError):
        share_config.widget_state([1, 2, 3], _key)
    # a config referencing an uploaded table this machine does not hold
    cfg = {"canonical_params": dict(
        planet="wasp39b", tp_mode="file", tp_file=forward.TP_FILE_UPLOAD,
        tp_file_sha1="0000000000000000")}
    with pytest.raises(ValueError, match="upload the table again"):
        share_config.widget_state(cfg, _key)
    # unsupported format version; the marker written is the marker read
    share = share_config.build_share(_canon(), goal={}, observation={})
    share["jwst_tool_config"] = 999
    with pytest.raises(ValueError, match="format 999"):
        share_config.widget_state(share, _key)
    assert share_config.build_share(_canon(), {}, {})["jwst_tool_config"] \
        == share_config.SHARE_FORMAT
    # unknown entries are dropped with a note, not applied
    share = share_config.build_share(
        _canon(), goal={},
        observation=dict(modes=["nirspec_g395h", "not_a_mode"]))
    state, notes = share_config.widget_state(share, _key)
    assert state["n0_modes"] == ["nirspec_g395h"]
    assert any("not_a_mode" in n for n in notes)


def test_embedded_tp_table_is_all_or_nothing(tmp_path):
    """All-or-nothing includes the filesystem: a config that fails validation
    must not deposit its embedded table in the uploads archive; a valid
    embedded table is archived and its restored path round-trips."""
    from pathlib import Path

    up = forward._uploads_dir()
    before = set(up.glob("*")) if up.exists() else set()
    cfg = {"canonical_params": dict(planet="wasp39b", tp_mode="file",
                                    tp_file=forward.TP_FILE_UPLOAD),
           "tp_table_text": "this is not a T-P table\n"}
    with pytest.raises((ValueError, RuntimeError)):
        share_config.widget_state(cfg, _key)
    after = set(up.glob("*")) if up.exists() else set()
    assert after == before, "invalid config left files in the uploads archive"

    text = forward._shipped_tp_file("wasp39b").read_text()
    src = tmp_path / "table.txt"
    src.write_text(text)
    canon = forward.canonical_params(dict(
        planet="wasp39b", tp_mode="file", tp_file=forward.TP_FILE_UPLOAD,
        tp_file_path=str(src)))
    state, _notes = share_config.widget_state(
        {"canonical_params": canon, "tp_table_text": text}, _key)
    p = Path(state["restored_tp_path"])
    assert p.parent == forward._uploads_dir()
    assert p.suffix == ".txt" and p.exists()
    assert p.read_text() == text


def test_provenance_block_is_complete_and_portable(tmp_path):
    """Release provenance carries no machine paths and reports EVERY declared
    repository -- the first version keyed on the GitHub name (`jax-vulcan`),
    missed the working copy (`VULCAN-JAX`), and silently dropped the solver
    from every export. The identity helper must find the solver under either
    directory name, report an unversioned copy as such, and make a missing
    repository explicit."""
    from jwst_tool import provenance
    share = share_config.build_share(_canon(), {}, {"seed": 73})
    prov = share["provenance"]
    assert prov["random_seed"] == 73
    assert set(prov["cache_schema"]) == {"model", "pandeia_worker"}
    assert "vulcan-jwst-tool" in prov["software"]
    assert "pandeia_stack" in prov and "datasets" in prov
    serialized = __import__("json").dumps(prov)
    assert "/Users/" not in serialized and "\\Users\\" not in serialized
    repos = prov["repositories"]
    assert set(repos) == set(provenance.REPOSITORIES), (
        "every declared repository must appear, even when absent")
    for name, rec in repos.items():
        assert rec["commit"], f"{name} carries no commit field"

    # VULCAN-JAX and jax-vulcan are the same repository
    for directory in ("VULCAN-JAX", "jax-vulcan"):
        workspace = tmp_path / directory
        (workspace / directory / ".git").mkdir(parents=True)
        rec = provenance._repo_identity(
            workspace, provenance.REPOSITORIES["vulcan-jax"])
        assert rec["directory"] == directory, rec
    # unversioned copy reported as such; genuinely missing repo explicit
    plain = tmp_path / "plain"
    (plain / "VULCAN-master").mkdir(parents=True)
    rec = provenance._repo_identity(
        plain, provenance.REPOSITORIES["VULCAN-master"])
    assert rec["commit"] == "unversioned", rec
    rec = provenance._repo_identity(plain, ("nope-not-here",))
    assert rec["commit"] == "absent" and rec["directory"] is None, rec


def test_combos_round_trip_and_invalid_entries_are_noted():
    """Named mode combinations restore to the K('combos') session key;
    unknown modes are dropped with a note, nameless/empty/duplicate
    combinations are never applied silently."""
    share = share_config.build_share(
        _canon(),
        goal=dict(goal="constrain", goal_param="lnZ", target_prec=0.1,
                  target_sig=3.0, marginalize=True, do_fisher=False,
                  fisher_params=["lnZ"], jac_method="fd"),
        observation=dict(modes=["nirspec_g395h", "niriss_soss", "miri_lrs"],
                         star_teff=5485.0, star_logg=4.47, star_feh=0.0,
                         n_transits=1, r_bin=100, floor_mode="none",
                         floors={}, noise_infl={}, show_noise=True, seed=42,
                         combos=[
                             dict(name="SOSS + G395H",
                                  modes=["niriss_soss", "nirspec_g395h"]),
                             dict(name="SOSS + G395H + MIRI",
                                  modes=["niriss_soss", "nirspec_g395h",
                                         "miri_lrs"])]))
    state, notes = share_config.widget_state(share, _key)
    assert not notes
    assert state["n0_seed"] == 42 and state["n0_shownoise"] is True
    assert state["n0_combos"] == [
        dict(name="SOSS + G395H", modes=["niriss_soss", "nirspec_g395h"]),
        dict(name="SOSS + G395H + MIRI",
             modes=["niriss_soss", "nirspec_g395h", "miri_lrs"])]

    # unknown mode dropped with a note; all-unknown combo not restored;
    # nameless and duplicate-name combos noted, first duplicate wins
    share["observation"]["combos"] = [
        dict(name="partial", modes=["nirspec_g395h", "not_a_mode"]),
        dict(name="hollow", modes=["ghost_mode"]),
        dict(name="", modes=["nirspec_g395h"]),
        dict(name="partial", modes=["miri_lrs"]),
    ]
    state, notes = share_config.widget_state(share, _key)
    assert state["n0_combos"] == [
        dict(name="partial", modes=["nirspec_g395h"])]
    joined = " | ".join(notes)
    assert "not_a_mode" in joined and "hollow" in joined
    assert "without a name" in joined and "duplicate" in joined.lower()


def test_noise_model_config_evolution_round_trips():
    """The global "Noise multiplier" and the per-mode multipliers round-trip
    as SEPARATE factors (the app composes them; recording the product would
    re-multiply on restore); a file predating the global knob restores the
    OLD behavior (no scale invented, no failure); and the 0.28.0-removed
    correlated-floor scenarios load with a note (they never changed the
    model) while scenario="random", the old default, loads silently."""
    goal = dict(goal="detect", goal_mol="H2O", target_sig=3.0)

    def _obs(**over):
        base = dict(ks_mag=9.0, t14=2.8, t_base=2.8, sat_limit=0.8,
                    star_teff=5485.0, star_logg=4.47, star_feh=0.0,
                    modes=["nirspec_g395h"], n_transits=1, r_bin=100,
                    floor_mode="constant", floors={"nirspec_g395h": 15.0},
                    noise_infl={"nirspec_g395h": 1.5},
                    show_noise=True, seed=0)
        base.update(over)
        return base

    share = share_config.build_share(
        _canon(), goal=goal, observation=_obs(noise_scale=2.0))
    state, notes = share_config.widget_state(share, _key)
    assert not notes, notes
    assert state[_key("infl_nirspec_g395h")] == 1.5, state
    assert state[_key("noisescale")] == 2.0, state

    # a config written before the global knob existed
    share = share_config.build_share(_canon(), goal=goal, observation=_obs())
    share["observation"].pop("noise_scale", None)
    state, notes = share_config.widget_state(share, _key)
    assert not notes, notes
    assert state[_key("infl_nirspec_g395h")] == 1.5, state
    assert _key("noisescale") not in state, state

    # unknown floor type: noted, floor settings keep their current values
    share2 = share_config.build_share(_canon(), goal=goal,
                                      observation=_obs(floor_mode="bogus"))
    state2, notes2 = share_config.widget_state(share2, _key)
    assert _key("floormode") not in state2
    assert any("noise-floor type" in n for n in notes2)

    # removed correlated-floor scenario: noted, never restored
    share["observation"]["scenario"] = "conservative"
    state, notes = share_config.widget_state(share, _key)
    assert not any("scenario" in k for k in state)
    assert any("scenario" in n for n in notes)
    share["observation"]["scenario"] = "random"
    state, notes = share_config.widget_state(share, _key)
    assert not any("scenario" in k for k in state)
    assert not any("scenario" in n for n in notes)


def test_out_of_range_values_refuse_loudly():
    """Streamlit does not reject an out-of-range session-state value -- it
    silently discards it and the widget falls back to its default, running a
    different model than the file describes. The restore must therefore
    refuse loudly BEFORE anything applies. Covers the observation family,
    the per-mode families, and a widget-only bound the engine itself never
    validates (Guillot T_irr)."""
    def _share(**obs_over):
        obs = dict(ks_mag=9.0, t14=2.8, t_base=2.8, sat_limit=0.8,
                   star_teff=5485.0, star_logg=4.47, star_feh=0.0,
                   modes=["nirspec_g395h"], n_transits=1, r_bin=100,
                   floor_mode="constant", floors={"nirspec_g395h": 15.0},
                   noise_infl={"nirspec_g395h": 1.05},
                   show_noise=True, seed=0)
        obs.update(obs_over)
        return share_config.build_share(_canon(), goal={}, observation=obs)

    cases = [
        (dict(r_bin=2700), "rbin"),                       # the pixel-level ask
        (dict(noise_scale=0.2), "noisescale"),
        (dict(floors={"nirspec_g395h": 500.0}), "floor"),
        (dict(star_teff=50000.0), "teff"),
        (dict(sat_limit=0.99), "sat"),
        (dict(noise_infl={"nirspec_g395h": 5.0}), "multiplier"),
    ]
    for over, needle in cases:
        with pytest.raises(ValueError, match=needle):
            share_config.widget_state(_share(**over), _key)
    with pytest.raises(ValueError, match="tirr"):
        share_config.widget_state(
            share_config.build_share(_canon(Tirr=5000.0), {}, {}), _key)


def test_noise_star_round_trips_and_legacy_files_get_a_note():
    """The transmission canonical block carries a zeroed star (cache
    hygiene), so the observation block records the Pandeia star. The zero
    sentinel must never reach the widget keys; a file without the record
    says so instead of silently keeping the session's values; an emission
    file restores the star from the canonical block itself."""
    canon = _canon()
    assert canon["star_teff"] == 0.0        # the transmission sentinel
    share = share_config.build_share(
        canon, goal={}, observation=dict(
            ks_mag=9.0, star_teff=5485.0, star_logg=4.47, star_feh=0.0))
    state, notes = share_config.widget_state(share, _key)
    assert state["n0_wasp39b_teff"] == 5485.0
    assert state["n0_wasp39b_logg"] == 4.47
    assert not any("stellar parameters" in n for n in notes)
    # legacy file: an observation block without the star record
    share = share_config.build_share(canon, goal={},
                                     observation=dict(ks_mag=9.0))
    state, notes = share_config.widget_state(share, _key)
    assert "n0_wasp39b_teff" not in state
    assert any("stellar parameters" in n for n in notes)
    # emission: the canonical block carries the real star
    canon_em = forward.canonical_params(dict(
        planet="wasp39b", tp_mode="guillot", science_mode="emission",
        star_teff=5485.0, star_logg=4.47, star_feh=0.0))
    state, notes = share_config.widget_state(
        share_config.build_share(canon_em, {}, {}), _key)
    assert state["n0_wasp39b_teff"] == 5485.0


def test_restore_bounds_match_the_widgets():
    """The range tables mirror app.py's widget bounds; this AST check pins
    the mirror so a widget-range edit cannot silently diverge from the
    restore path. Bounds shared through a constant (forward.*_RANGE,
    planets.CUSTOM_FIELD_RANGES) cannot drift and are excluded."""
    import ast
    import importlib.util
    from pathlib import Path

    from jwst_tool import planets

    src = Path(importlib.util.find_spec("jwst_tool.app").origin).read_text()

    def _const(node):
        if isinstance(node, ast.Constant) and isinstance(node.value,
                                                         (int, float)):
            return float(node.value)
        if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
                and isinstance(node.operand, ast.Constant)):
            return -float(node.operand.value)
        return None

    found: dict = {}
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "number_input"):
            continue
        key_arg = next((kw.value for kw in node.keywords
                        if kw.arg == "key"), None)
        if not (isinstance(key_arg, ast.Call) and key_arg.args):
            continue
        ka = key_arg.args[0]
        if isinstance(ka, ast.Constant):
            suffix = str(ka.value)
        elif (isinstance(ka, ast.JoinedStr) and ka.values
                and isinstance(ka.values[0], ast.Constant)):
            suffix = str(ka.values[0].value) + "*"       # dynamic family
        else:
            continue
        if len(node.args) >= 3:
            lo, hi = _const(node.args[1]), _const(node.args[2])
            if lo is not None and hi is not None:
                found.setdefault(suffix, set()).add((lo, hi))

    shared = ({"mierg", "miesg", "miemmr", "nupts", "nz"}
              | set(planets.CUSTOM_FIELD_RANGES))
    for table in (share_config._GLOBAL_BOUNDS, share_config._PLANET_BOUNDS):
        for widget, (lo, hi) in table.items():
            if widget in shared:
                continue
            assert widget in found, (
                f"share_config restores widget {widget!r} but app.py has no "
                "number_input with that key and literal bounds")
            assert (float(lo), float(hi)) in found[widget], (
                f"bounds for {widget!r} differ: share_config has "
                f"({lo}, {hi}), app.py has {found[widget]}")
    assert found.get("floor_*") == {share_config._FLOOR_BOUNDS}
    assert found.get("infl_*") == {share_config._INFL_BOUNDS}
    assert found.get("tgt_*") == {share_config._TGT_BOUNDS_K,
                                  share_config._TGT_BOUNDS}


def test_gui_removed_physics_defaults_load_and_nondefaults_refuse():
    """Condensation, settling, escape and BC fluxes are API-only
    (2026-08-13). Defaults load normally, but a configuration that ENABLES
    any of them must RAISE, not load with a note: these switches change the
    atmosphere the model computes, so pinning them off would show a
    successful restore while Run computed something the file does not
    describe -- the class of silent behavior change the fail-fast rule
    forbids. (The removed noise scenarios only get a note: those never
    changed the model.) The settings remain reachable through the API."""
    canon = _canon()
    assert canon["use_condense"] is False and canon["use_settling"] is False
    assert not canon["diff_esc"] and not canon["top_flux"]
    state, notes = share_config.widget_state(canon, _key)
    assert not notes
    assert state["n0_planet"] == "wasp39b"

    cases = [
        (dict(use_condense=True, use_photo=True, use_moldiff=True),
         "condensation"),
        (dict(use_settling=True, use_moldiff=True), "gravitational settling"),
        (dict(diff_esc=["H2"], use_moldiff=True), "diffusion-limited escape"),
        (dict(top_flux=[["H2O", "1.0e8"]]), "top-boundary"),
        (dict(bot_flux=[["SO2", "1.0e9", "0.1"]]), "bottom-boundary"),
    ]
    for over, needle in cases:
        canon = _canon(**over)
        with pytest.raises(ValueError, match=needle):
            share_config.widget_state(canon, _key)
        with pytest.raises(ValueError, match="programmatic interface"):
            share_config.widget_state(canon, _key)
