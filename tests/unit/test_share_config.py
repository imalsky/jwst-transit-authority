"""share_config: the download/upload configuration round-trip.

The mapping must restore exactly the widget state the download described, and
an invalid file must raise before anything could be applied. No Streamlit, no
pandeia, no JAX: the mapping is pure dict work over canonical params.
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
                         modes=["nirspec_g395h"], n_transits=2, r_bin=100,
                         floor_mode="constant",
                         floors={"nirspec_g395h": 15.0},
                         noise_infl={"nirspec_g395h": 1.05},
                         scenario="random", show_noise=False, seed=0))
    state, notes = share_config.widget_state(share, _key)
    assert not notes

    # target + system (per-planet keys)
    assert state["n0_planet"] == "wasp39b"
    assert state["n0_wasp39b_teff"] == canon["star_teff"]
    assert state["n0_wasp39b_g"] == pytest.approx(canon["gs_cgs"] / 100.0)
    assert state["n0_wasp39b_ks"] == 9.0
    assert state["n0_wasp39b_t14"] == 2.8
    # atmosphere
    assert state["n0_wasp39b_tp"] == "guillot"
    assert state["n0_wasp39b_tirr"] == canon["Tirr"]
    assert state["n0_met"] == 7.0
    assert state["n0_co"] == 0.6
    assert state["n0_wasp39b_kzz"] == pytest.approx(9.0)   # log10(kzz_const)
    assert state["n0_xmols_vulcan"] == ["HCN", "NH3"]
    assert state["n0_cloud"] is True
    assert state["n0_ck"] == -2.0
    # science goal (dynamic widget-key suffixes)
    assert state["n0_goal"] == "constrain"
    assert state["n0_gp_vulcan_guillot_1_0"] == "lnZ"
    assert state["n0_tgt_lnZ"] == 0.1
    assert state["n0_fx_vulcan_guillot_1_0"] == ["lnKzz"]
    # observation
    assert state["n0_modes"] == ["nirspec_g395h"]
    assert state["n0_ntr"] == 2
    assert state["n0_rbin"] == 100
    assert state["n0_floormode"] == "constant"
    assert state["n0_floor_nirspec_g395h"] == 15.0
    assert state["n0_infl_nirspec_g395h"] == 1.05


def test_detect_goal_maps_the_molecule_widget_key():
    canon = _canon(extra_mols=["C2H2", "HCN"])
    share = share_config.build_share(
        canon,
        goal=dict(goal="detect", target_mol="SO2", target_sig=3.0,
                  marginalize=True, do_fisher=False, fisher_params=[],
                  jac_method="fd"),
        observation=dict(modes=["nirspec_prism"], n_transits=1))
    state, notes = share_config.widget_state(share, _key)
    assert not notes
    assert state["n0_mol_vulcan_C2H2_HCN"] == "SO2"
    assert state["n0_dofish"] is False


def test_bare_canonical_dict_is_accepted():
    state, _ = share_config.widget_state(_canon(), _key)
    assert state["n0_planet"] == "wasp39b"
    assert "n0_modes" not in state          # no observation section to restore


def test_unknown_entries_are_dropped_with_a_note_not_applied():
    share = share_config.build_share(
        _canon(), goal={},
        observation=dict(modes=["nirspec_g395h", "not_a_mode"]))
    state, notes = share_config.widget_state(share, _key)
    assert state["n0_modes"] == ["nirspec_g395h"]
    assert any("not_a_mode" in n for n in notes)


def test_invalid_file_raises_before_anything_applies():
    with pytest.raises(ValueError):
        share_config.widget_state({"foo": 1}, _key)
    with pytest.raises(ValueError):
        share_config.widget_state(
            {"canonical_params": {"planet": "wasp39b", "nz": 5}}, _key)
    with pytest.raises(ValueError):
        share_config.widget_state([1, 2, 3], _key)


def test_missing_uploaded_tp_table_is_a_loud_error():
    cfg = {"canonical_params": dict(
        planet="wasp39b", tp_mode="file", tp_file=forward.TP_FILE_UPLOAD,
        tp_file_sha1="0000000000000000")}
    with pytest.raises(ValueError, match="upload the table again"):
        share_config.widget_state(cfg, _key)
