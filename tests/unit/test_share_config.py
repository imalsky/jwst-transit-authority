"""share_config: the download/upload configuration round-trip.

The mapping must restore exactly the widget state the download described, and
an invalid file must raise before anything could be applied. No Streamlit, no
pandeia, no JAX: the mapping is pure dict work over canonical params.
Refusals are merged into consolidated fail-closed tests (maintainer: fewer,
stronger tests).
"""
import pytest

from jwst_tool import forward, planets, share_config


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
    state = share_config.widget_state(share, _key)
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
    assert state["n0_gp_vulcan_guillot_1"] == "lnZ"
    assert state["n0_tgt_lnZ"] == 0.1
    assert state["n0_fx_vulcan_guillot_1"] == ["lnKzz"]
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
    state = share_config.widget_state(share, _key)
    assert state["n0_mol_vulcan_C2H2_HCN"] == "SO2"
    assert state["n0_dofish"] is False

    # a bare canonical dict is accepted
    state = share_config.widget_state(_canon(), _key)
    assert state["n0_planet"] == "wasp39b"
    assert "n0_modes" not in state          # no observation section to restore


def test_a_gray_cloud_round_trips_on_the_pressure_it_was_entered_as():
    """Gray is a GUI shape, not physics: the file carries only the opacity the
    engine actually uses, and the restore inverts it back to the pressure that
    was entered. The slope is pinned at 0 and is NOT offered as a free
    parameter, so the goal widgets carry their own key suffix. A file with no
    cloud_mode is the power-law haze -- which is what every configuration
    written before this mode existed was."""
    # the app's own T_eq: derived from the star and orbit, not from the
    # canonical block, which zeroes star_teff in transmission
    w39 = planets.PLANETS["wasp39b"]
    teq = planets.system_teq(w39["star"]["teff"], w39["rstar_rsun"],
                             w39["orbit_au"])
    canon = _canon(cloud_on=True, alpha_cloud=0.0,
                   log_kappa_cloud=planets.gray_cloud_log_kappa(
                       1.0e-3, w39["gs_cgs"], w39["rp_rjup"], teq),
                   fisher_params=["lnZ", "log_kappa_cloud"])
    goal = dict(goal="constrain", goal_param="log_kappa_cloud",
                target_prec=0.3, target_sig=3.0, marginalize=True,
                do_fisher=False, fisher_params=["lnZ", "log_kappa_cloud"],
                jac_method="fd")
    share = share_config.build_share(canon, goal=goal, observation={},
                                     cloud_top_bar=1.0e-3)
    assert share["cloud_top_bar"] == 1.0e-3
    # the opacity is what the run is keyed on; the pressure is display only
    assert "cloud_top_bar" not in share["canonical_params"]

    state = share_config.widget_state(share, _key)
    assert state["n0_cmode"] == "gray"
    assert state["n0_cpt"] == pytest.approx(-3.0)     # 1 mbar, as entered
    assert "n0_ck" not in state and "n0_ca" not in state
    # gray asserts the slope, so it moves off the freeable list and the goal
    # widgets key on their own suffix (off 0 / haze 1 / gray 2)
    assert state["n0_gp_vulcan_guillot_2"] == "log_kappa_cloud"
    assert state["n0_fx_vulcan_guillot_2"] == ["lnZ"]

    # no marker -> haze, restoring the opacity widgets exactly as before
    share.pop("cloud_top_bar")
    state = share_config.widget_state(share, _key)
    assert state["n0_cmode"] == "haze"
    assert state["n0_ck"] == canon["log_kappa_cloud"]
    assert "n0_cpt" not in state
    assert "n0_gp_vulcan_guillot_1" in state


@pytest.mark.parametrize("network", ["ncho", "sncho2025"])
def test_a_non_default_network_round_trips_on_its_own_widget_keys(network):
    """The molecule widgets are re-keyed per network so a selection from one
    never strands in the other's options; only the default network's keys are
    unsuffixed. That suffixed branch had no coverage, and it is a shipped
    widget-KEY contract."""
    co = 2.0 if forward.CO_MAX[network] > 1.0 else 0.6
    canon = _canon(network=network, co_ratio=co, extra_mols=["HCN", "NH3"])
    share = share_config.build_share(
        canon,
        goal=dict(goal="detect", target_mol="H2O", do_fisher=False,
                  jac_method="fd"),
        observation=dict(ks_mag=9.0, t14=2.8, t_base=2.8, sat_limit=0.8,
                         star_teff=5485.0, star_logg=4.47, star_feh=0.0,
                         modes=["nirspec_g395h"], n_transits=1, r_bin=100,
                         floor_mode="none", floors={}, noise_infl={},
                         show_noise=True, seed=0))
    state = share_config.widget_state(share, _key)
    assert state["n0_network"] == network and state["n0_co"] == co
    sfx = f"_{network}"
    assert state[f"n0_xmols_vulcan{sfx}"] == ["HCN", "NH3"]
    assert "n0_xmols_vulcan" not in state          # the sncho keys stay clean
    assert any(k.startswith(f"n0_mol_vulcan{sfx}_") for k in state)


def test_a_carbon_rich_config_is_refused_on_a_network_that_cannot_hold_it():
    """The restore range for C/O follows the network, so a file written on
    sncho2025 at C/O 2 must not silently restore onto the default network's
    widget (Streamlit would discard it and run C/O 0.55 instead)."""
    canon = _canon(network="sncho2025", co_ratio=2.0)
    state = {"n0_network": "sncho", "n0_co": float(canon["co_ratio"])}
    with pytest.raises(ValueError, match="co_ratio=2 outside"):
        share_config._check_widget_ranges(state, "n0_{}".format,
                                          "n0_{}".format, "transmission")


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
    with pytest.raises(ValueError):
        share_config.widget_state(cfg, _key)
    # unsupported format version; the marker written is the marker read
    share = share_config.build_share(_canon(), goal={}, observation={})
    share["jwst_tool_config"] = 999
    with pytest.raises(ValueError):
        share_config.widget_state(share, _key)
    assert share_config.build_share(_canon(), {}, {})["jwst_tool_config"] \
        == share_config.SHARE_FORMAT
    # unknown entries are dropped with a note, not applied
    share = share_config.build_share(
        _canon(), goal={},
        observation=dict(modes=["nirspec_g395h", "not_a_mode"]))
    state = share_config.widget_state(share, _key)
    assert state["n0_modes"] == ["nirspec_g395h"]


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
    state = share_config.widget_state(
        {"canonical_params": canon, "tp_table_text": text}, _key)
    p = Path(state["restored_tp_path"])
    assert p.parent == forward._uploads_dir()
    assert p.suffix == ".txt" and p.exists()
    assert p.read_text() == text

    # ...and a failure LATER in the restore, after the payload itself
    # validates, must leave nothing behind either: the upload is staged and
    # committed only once the whole restore has validated.
    before = set(up.glob("*"))
    other = tmp_path / "other.txt"
    other.write_text(text + "# distinct content so the sha differs\n")
    canon2 = forward.canonical_params(dict(
        planet="wasp39b", tp_mode="file", tp_file=forward.TP_FILE_UPLOAD,
        tp_file_path=str(other)))
    def _boom(*a, **k):
        raise ValueError("a later restore step refused")

    real = share_config._check_widget_ranges
    share_config._check_widget_ranges = _boom
    try:
        with pytest.raises(ValueError, match="later restore step"):
            share_config.widget_state(
                {"canonical_params": canon2, "tp_table_text": other.read_text()},
                _key)
    finally:
        share_config._check_widget_ranges = real
    assert set(up.glob("*")) == before, "a late failure left a staged upload"


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
    assert "jwst-transit-authority" in prov["software"]
    assert "pandeia_stack" in prov and "datasets" in prov
    assert "ktables" in prov["datasets"]
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
    state = share_config.widget_state(share, _key)
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
    state = share_config.widget_state(share, _key)
    assert state["n0_combos"] == [
        dict(name="partial", modes=["nirspec_g395h"])]


def test_noise_model_config_evolution_round_trips():
    """The global "Noise multiplier" and the per-mode multipliers round-trip
    as SEPARATE factors (the app composes them; recording the product would
    re-multiply on restore); a file predating the global knob restores the
    same behavior (no scale invented, no failure); and the removed
    correlated-floor scenarios load with a note (they never changed the
    model) while scenario="random" loads silently."""
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
    state = share_config.widget_state(share, _key)
    assert state[_key("infl_nirspec_g395h")] == 1.5, state
    assert state[_key("noisescale")] == 2.0, state

    # a config written before the global knob existed
    share = share_config.build_share(_canon(), goal=goal, observation=_obs())
    share["observation"].pop("noise_scale", None)
    state = share_config.widget_state(share, _key)
    assert state[_key("infl_nirspec_g395h")] == 1.5, state
    assert _key("noisescale") not in state, state

    # unknown floor type: noted, floor settings keep their current values
    share2 = share_config.build_share(_canon(), goal=goal,
                                      observation=_obs(floor_mode="bogus"))
    state2 = share_config.widget_state(share2, _key)
    assert _key("floormode") not in state2

    # removed correlated-floor scenario: noted, never restored
    share["observation"]["scenario"] = "conservative"
    state = share_config.widget_state(share, _key)
    assert not any("scenario" in k for k in state)
    share["observation"]["scenario"] = "random"
    state = share_config.widget_state(share, _key)
    assert not any("scenario" in k for k in state)


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
    state = share_config.widget_state(share, _key)
    assert state["n0_wasp39b_teff"] == 5485.0
    assert state["n0_wasp39b_logg"] == 4.47
    # legacy file: an observation block without the star record
    share = share_config.build_share(canon, goal={},
                                     observation=dict(ks_mag=9.0))
    state = share_config.widget_state(share, _key)
    assert "n0_wasp39b_teff" not in state
    # emission: the canonical block carries the real star
    canon_em = forward.canonical_params(dict(
        planet="wasp39b", tp_mode="guillot", science_mode="emission",
        star_teff=5485.0, star_logg=4.47, star_feh=0.0))
    state = share_config.widget_state(
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

    shared = {"nz"} | set(planets.CUSTOM_FIELD_RANGES)
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


def test_provenance_snapshot_records_the_ktables(tmp_path, monkeypatch):
    """The exported provenance names the k-tables a run actually reads
    (dataset, isotopologue, file, curated + header DOI per species),
    path-free; an absent
    provenance record is an explicit 'absent' entry with the remedy, never
    a silent omission."""
    import json
    from jwst_tool import provenance
    from vulcan_forward import exomolop
    root = tmp_path / "data"
    (root / "exomolop").mkdir(parents=True)
    (root / "exomolop" / "provenance.json").write_text(json.dumps({
        "H2O": {"dataset": "POKAZATEL", "iso": "1H2-16O",
                "natural_abundance": False, "url": "https://x/H2O",
                "file": "1H2-16O__POKAZATEL.h5"},
        "CO2": {"dataset": "Dozen", "iso": "12C-16O2",
                "natural_abundance": True, "url": "https://x/CO2",
                "file": "CO2-all__Dozen.h5",
                "doi": "10.1093/mnras/staf2135"}}))
    monkeypatch.setenv("VULCAN_FORWARD_DATA", str(root))
    monkeypatch.setattr(exomolop, "table_info", lambda m: {
        "doi": "x.xxxx/yyyyy", "date_id": None, "ngauss": 16,
        "t_range_k": [100.0, 3400.0], "p_range_bar": [1e-5, 100.0],
        "wl_range_um": [0.3, 50.0], "grid_sha256": "a" * 64,
        "band_resolution": 1000.0})
    provenance._base_snapshot.cache_clear()
    try:
        snap = provenance.snapshot()
        kt = snap["datasets"]["ktables"]
        assert kt["status"] == "ok" and set(kt["tables"]) == {"H2O", "CO2"}
        assert kt["tables"]["CO2"]["natural_abundance"] is True
        # DOI resolution, both branches: the fetcher's curated value wins
        # where it exists (four ExoMolOP headers ship a placeholder), the
        # header is the fallback, and the header value is always kept
        # alongside so nothing is silently substituted.
        assert kt["tables"]["H2O"]["doi"] == "x.xxxx/yyyyy"   # header fallback
        assert kt["tables"]["CO2"]["doi"] == "10.1093/mnras/staf2135"
        assert (kt["tables"]["CO2"]["header_doi"]
                == kt["tables"]["H2O"]["header_doi"] == "x.xxxx/yyyyy")
        assert kt["tables"]["H2O"]["file"] == "1H2-16O__POKAZATEL.h5"
        assert str(tmp_path) not in json.dumps(snap)
        monkeypatch.setenv("VULCAN_FORWARD_DATA", str(tmp_path / "empty"))
        (tmp_path / "empty").mkdir()
        provenance._base_snapshot.cache_clear()
        kt = provenance.snapshot()["datasets"]["ktables"]
        assert kt["status"] == "absent" and "fetch_exomolop" in kt["remedy"]
        assert str(tmp_path) not in json.dumps(kt)
    finally:
        provenance._base_snapshot.cache_clear()


def test_gui_removed_physics_defaults_load_and_nondefaults_refuse():
    """Condensation is API-only. Defaults load normally, but a configuration
    that ENABLES it must RAISE, not load with a note: the switch changes the
    atmosphere the model computes, so pinning it off would show a
    successful restore while Run computed something the file does not
    describe -- the class of silent behavior change the fail-fast rule
    forbids. (The removed noise scenarios only get a note: those never
    changed the model.) The setting remains reachable through the API. The
    retired boundary-flux/escape/settling keys still round-trip pinned off."""
    canon = _canon()
    assert canon["use_condense"] is False and canon["use_settling"] is False
    assert not canon["diff_esc"] and not canon["top_flux"]
    assert not canon["bot_flux"]
    state = share_config.widget_state(canon, _key)
    assert state["n0_planet"] == "wasp39b"

    canon = _canon(use_condense=True, use_photo=True, use_moldiff=True)
    with pytest.raises(ValueError, match="condensation"):
        share_config.widget_state(canon, _key)
    # a saved configuration carrying the deleted Mie deck / line-by-line
    # mode refuses with the removal message (never loads under correlated-k)
    for stale in (dict(mie_condensate="MgSiO3"), dict(opacity_mode="lbl")):
        canon = dict(_canon(), **stale)
        with pytest.raises(ValueError, match="removed in 0.48.0"):
            share_config.widget_state(canon, _key)
