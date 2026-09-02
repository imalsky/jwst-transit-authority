"""Pure-Python tests of adjoint_diag's light path (no JAX / no chemistry):
the cache-key semantics and the detailed-balance pairing rules the reaction
table is built from (the validated jax_paper pairing logic, inlined)."""
from types import SimpleNamespace

import numpy as np

from jwst_tool import adjoint_diag


def _p(**kw):
    base = dict(planet="wasp39b", tp_mode="guillot",
                kzz_mode="const", kzz_const=1.0e9)
    base.update(kw)
    return base


def _fake_network():
    # slots 1..6: (1,2) reversible pair, (3,4) reversible pair, 5 photo, 6 pad
    nr = 6
    is_photo = np.zeros(nr + 1, dtype=bool)
    is_photo[5] = True
    return SimpleNamespace(
        is_photo=is_photo, is_ion=np.zeros(nr + 1, dtype=bool),
        stop_rev_indx=5, conden_indx=7,
        Rf={1: "A + B -> C + D", 3: "E + F -> G", 5: "X -> Y + Z"})


def test_pairing_sums_reversible_and_keeps_photo_single():
    g = np.array([0.0, 2.0, -1.5, 0.3, 0.1, -0.9, 0.0])
    rows = adjoint_diag._pair_physical(g, _fake_network())
    by_fwd = {r["fwd"]: r for r in rows}
    # reversible: SIGNED SUM of forward + detailed-balance reverse
    assert by_fwd[1]["S"] == 2.0 - 1.5 and by_fwd[1]["kind"] == "reversible"
    assert "<->" in by_fwd[1]["label"]
    assert abs(by_fwd[3]["S"] - 0.4) < 1e-15
    # photolysis: single directional row, no pairing with the next slot
    assert by_fwd[5]["S"] == -0.9 and by_fwd[5]["kind"] == "photolysis"
    assert "->" in by_fwd[5]["label"] and "<->" not in by_fwd[5]["label"]
    # sorted by |S| descending
    assert [r["fwd"] for r in rows] == [5, 1, 3]


def test_adjoint_key_ignores_rt_only_knobs():
    # the adjoint runs on the chemistry state alone: spectra-only settings
    # must not fragment the cache
    k0 = adjoint_diag.adjoint_key(_p(), "SO2")
    assert adjoint_diag.adjoint_key(
        _p(cloud_on=True,
           extra_mols=["HCN"], fisher_params=["lnZ"], jac_method="ad",
           use_photo=True), "SO2") == k0
    # RT-only knobs are stripped too; rt_ptop_bar is NOT one of them any more
    # (the chemistry top follows it, so the adjoint state depends on it)
    assert adjoint_diag.adjoint_key(_p(rt_integration="trapezoid"), "SO2") == k0
    assert adjoint_diag.adjoint_key(_p(rt_ptop_bar=1.0e-8), "SO2") != k0


def test_adjoint_key_tracks_chemistry_and_species():
    k0 = adjoint_diag.adjoint_key(_p(), "SO2")
    assert adjoint_diag.adjoint_key(_p(), "CH4") != k0
    assert adjoint_diag.adjoint_key(_p(Tirr=1100.0), "SO2") != k0
    assert adjoint_diag.adjoint_key(_p(co_ratio=1.5), "SO2") != k0
    assert adjoint_diag.adjoint_key(_p(use_photo=False), "SO2") != k0
    assert adjoint_diag.adjoint_key(_p(use_vm_mol=True), "SO2") != k0


def test_load_result_missing_is_none():
    assert adjoint_diag.load_result(_p(Tirr=871.23), "SO2") is None


def test_run_adjoint_refuses_condensing_states():
    # condensing states are refused up front, before any heavy import
    import pytest

    with pytest.raises(RuntimeError, match="condensing state"):
        adjoint_diag.run_adjoint(_p(use_condense=True), "SO2",
                                 log=lambda *a: None)
