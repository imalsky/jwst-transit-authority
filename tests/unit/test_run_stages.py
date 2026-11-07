"""The forward model's stages against fake chemistry and RT: analytic rows, refused caches, stored arrays.

A row built from a polynomial has a known answer, so these pin the arithmetic and the escalation rules without a solve.
"""

from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from jwst_tool import certify, forward, instruments as ins, jacobian, model_defs, spectrum


def _quiet(_message):
    return None


def _baseline(depth_of, co_ratio=0.55, run_frq=None, **cp):
    """A Baseline whose every build is a record of its abundances and cadence; depth_of(x, frq) is the spectrum."""
    builds = []

    def build_chem(extra_abun=None, tag="", photo_frq=None):
        chem = SimpleNamespace(x=(extra_abun or {}).get("x", 0.0), frq=photo_frq, tag=tag, y=np.ones((2, 2)))
        builds.append(chem)
        return chem

    def abundance_overrides(met, co):
        return {"x": float(np.log(co / co_ratio)) if met == 10.0 else float(np.log(met / 10.0))}

    return jacobian.Baseline(
        cp={"met_x_solar": 10.0, "co_ratio": co_ratio, "network": "sncho", "jac_method": "fd", **cp},
        pic=False,
        theta=np.zeros(4),
        theta_names=["a", "b", "lnKzz", "Tirr"],
        th0=np.zeros(4),
        build_chem=build_chem,
        abundance_overrides=abundance_overrides,
        assemble=None,
        setup=SimpleNamespace(emis=None, tp_eval=None),
        chem=SimpleNamespace(),
        column=SimpleNamespace(depth=lambda *_a, **_k: depth_of(0.0, run_frq)),
        y_sol=np.ones((2, 2)),
        species=["H2", "H2O"],
        run_frq=run_frq,
        escalated=[],
        bz_build=float("nan"),
        advance=lambda: None,
        log=_quiet,
    ), builds


def _spectrum(x, frq):
    """Known spectrum: slope 3 and 5 per unit x, with an offset that differs by cadence.

    A row mixing cadences would divide that offset by h and report it as derivative.
    """
    return np.array([1.0 + 3.0 * x, 2.0 + 5.0 * x]) + (0.4 if frq == 1 else 0.0)


@pytest.mark.parametrize(
    ("co_ratio", "method"), [(0.55, "fd-central"), (0.9, "fd-onesided")], ids=["central", "one-sided"]
)
@pytest.mark.parametrize("stalls", [False, True], ids=["certified", "escalated"])
def test_an_fd_row_recovers_the_slope_and_never_mixes_cadences(co_ratio, method, stalls, monkeypatch):
    b, builds = _baseline(_spectrum, co_ratio=co_ratio)

    def certified_depth(_b, chem, _theta, _stage, _tp=None):
        if stalls and chem.frq is None:
            raise certify.NotCertified("stalled")
        return _spectrum(chem.x, chem.frq)

    monkeypatch.setattr(jacobian, "_certified_depth", certified_depth)
    rows = jacobian.Rows(["dlnCO"], np.zeros((2, 2)))
    jacobian._fd_row(b, "dlnCO", rows, 0, 0.0)

    np.testing.assert_allclose(rows.jac[0], [3.0, 5.0], rtol=1e-9)
    assert rows.method == [method]
    assert b.escalated == (["FD dlnCO row"] if stalls else [])
    # h/2 on the one-sided stencil, so its reach equals the central stencil's 2h
    assert rows.fd_h == [model_defs.FD_STEPS["dlnCO"] / (2.0 if method == "fd-onesided" else 1.0)]
    if stalls:
        # the solve stops at its first stall, then the whole row and its anchor run at cadence 1
        assert builds[0].frq is None
        assert {c.frq for c in builds[1:]} == {1}


def test_only_a_stall_escalates_and_never_past_cadence_one(monkeypatch):
    def backend_fault(*_a, **_k):
        raise RuntimeError("backend: RESOURCE_EXHAUSTED")

    b, builds = _baseline(_spectrum)
    monkeypatch.setattr(jacobian, "_certified_depth", backend_fault)
    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        jacobian._fd_row(b, "dlnCO", jacobian.Rows(["dlnCO"], np.zeros((2, 2))), 0, 0.0)
    assert b.escalated == [] and {c.frq for c in builds} == {None}

    def stalled(*_a, **_k):
        raise certify.NotCertified("stalled")

    b, builds = _baseline(_spectrum, run_frq=1)
    monkeypatch.setattr(jacobian, "_certified_depth", stalled)
    with pytest.raises(certify.NotCertified):
        jacobian._fd_row(b, "dlnCO", jacobian.Rows(["dlnCO"], np.zeros((2, 2))), 0, 0.0)
    assert b.escalated == []


def test_an_ad_row_is_divided_by_its_step_and_a_stall_moves_every_later_row(monkeypatch):
    b, builds = _baseline(_spectrum, jac_method="ad", fisher_params=["lnZ", "lnKzz"])
    b.theta_names = ["lnZ", "dlnCO", "lnKzz", "Tirr"]
    slope = {0: np.array([3.0, 5.0]), 2: np.array([-1.0, 2.0])}
    used = []

    def ad_row(_b, chem_x, e):
        i = int(np.flatnonzero(e)[0])
        used.append((i, chem_x.frq))
        # a tangent along e carries the step: the row divides it back out
        return np.ones(2), chem_x, None, np.ones((2, 2)), slope[i] * e[i]

    def check_converged(diag, _stage, _species, _chem, _log):
        if diag.frq is None:
            raise certify.NotCertified("stalled")

    monkeypatch.setattr(jacobian, "_ad_row", ad_row)
    monkeypatch.setattr(jacobian, "check_converged", check_converged)
    monkeypatch.setattr(jacobian, "check_elements", lambda *_a: 0.0)
    rows = jacobian.Rows(["lnZ", "lnKzz"], np.zeros((3, 2)))
    cols = jacobian._ad_chemistry_rows(b, rows)

    np.testing.assert_allclose(cols["lnZ"], slope[0])
    np.testing.assert_allclose(cols["lnKzz"], slope[2])
    # the first row stalled at the config cadence; the shared cadence-1 build carries both
    assert used == [(0, None), (0, 1), (2, 1)]
    assert b.escalated == ["AD lnZ row", "AD lnKzz row"]
    assert builds[1].frq == 1


def test_the_lnr0_row_is_the_central_difference_of_the_depth():
    b, _ = _baseline(_spectrum)
    b.column = SimpleNamespace(depth=lambda _y, _th, lnR0=0.0: np.array([1.0, 2.0]) + np.array([7.0, 11.0]) * lnR0)
    rows = jacobian.Rows([], np.zeros((1, 2)))
    jacobian._lnr0_row(b, rows, 0)
    np.testing.assert_allclose(rows.jac[0], [7.0, 11.0])
    assert rows.method == ["fd-rt"] and np.isnan(rows.fd_err[0])


def _cache_entry(**over):
    entry = {
        "species": np.array(["H2", "H2O"]),
        "theta": np.zeros(4),
        "y_raw": np.ones((2, 2)),
        "conv_accept": np.array([10]),
        "conv_longdy": np.array([0.05]),
        "photo_escalated": np.array([0]),
    }
    return {**entry, **over}


@pytest.mark.parametrize(
    ("over", "chem_gate", "match"),
    [
        ({"species": np.array(["H2", "CO"])}, 0.1, "species set does not match"),
        ({"theta": np.ones(4)}, 0.1, "stored theta differs"),
        ({"y_raw": np.full((2, 2), np.nan)}, 0.1, "non-finite stored column"),
        ({"conv_longdy": np.array([0.2])}, 0.1, "fails the current gate"),
    ],
    ids=["species", "theta", "non-finite", "certificate"],
)
def test_a_chem_cache_entry_that_does_not_fit_is_refused_loudly(over, chem_gate, match, monkeypatch, tmp_path):
    monkeypatch.setattr(forward, "check_elements", lambda *_a: 0.0)
    monkeypatch.setattr(forward, "ColumnSpectrum", lambda *_a: None)
    chem = SimpleNamespace(yconv_min=chem_gate)
    assembly = SimpleNamespace(theta=np.zeros(4), build_chem=None)
    with pytest.raises(RuntimeError, match=match):
        forward._column_from_cache(
            _cache_entry(**over), tmp_path / "chem.npz", assembly, ["H2", "H2O"], chem, None, _quiet
        )


def test_a_chem_cache_entry_runs_its_element_gate_and_its_stored_cadence(monkeypatch, tmp_path):
    gated = []
    monkeypatch.setattr(forward, "check_elements", lambda y, chem, stage, log: gated.append(stage))
    monkeypatch.setattr(forward, "ColumnSpectrum", lambda *_a: None)
    built = []

    def build_chem(tag, photo_frq):
        built.append(photo_frq)
        return SimpleNamespace(yconv_min=0.1, rebuilt=True)

    chem = SimpleNamespace(yconv_min=0.1, rebuilt=False)
    assembly = SimpleNamespace(theta=np.zeros(4), build_chem=build_chem)
    plain = forward._column_from_cache(_cache_entry(), tmp_path / "c.npz", assembly, ["H2", "H2O"], chem, None, _quiet)
    assert gated == ["chem-cache column"]
    assert plain.run_frq is None and plain.escalated == [] and not plain.chem.rebuilt and built == []

    esc = forward._column_from_cache(
        _cache_entry(photo_escalated=np.array([1])), tmp_path / "c.npz", assembly, ["H2", "H2O"], chem, None, _quiet
    )
    assert esc.run_frq == 1 and esc.chem.rebuilt and built == [1] and len(esc.escalated) == 1


def test_the_opacity_advisory_refuses_a_bulk_absorber_with_no_table():
    config = SimpleNamespace(MOLECULES={"H2O": {"vulcan": "H2O"}})
    p_bar = np.geomspace(1e-5, 1.0, 12)
    y = np.ones((12, 3)) * np.array([1.0, 1e-4, 1e-1])
    col = SimpleNamespace(species=["H2", "H2O", "CS2"], y_sol=y, chem=SimpleNamespace(p_bar=p_bar))
    with pytest.raises(RuntimeError, match="CS2"):
        forward._opacity_advisory(col, config, _quiet)


def test_the_stored_entry_holds_gas_mixing_ratios_and_the_flux_its_depth_implies(monkeypatch, tmp_path):
    monkeypatch.setattr(ins, "MODEL_CACHE", tmp_path)
    monkeypatch.setattr(forward, "cache_path", lambda _params: tmp_path / "entry.npz")
    chem = SimpleNamespace(
        p_bar=np.array([1.0, 0.1]), species_masses=np.ones(3), sidx={"H2": 0, "H2O": 1, "H2O_l_s": 2}
    )
    y = np.array([[0.5, 0.25, 0.25], [0.5, 0.25, 0.25]])
    col = forward._Column(y, y, chem, None, ["H2", "H2O", "H2O_l_s"], [], None, False)
    fs = np.array([2.0, 4.0])
    setup = SimpleNamespace(rt=SimpleNamespace(wl_um=np.array([1.0, 2.0])), emis=object(), fs_j=fs)
    evidence = SimpleNamespace(
        depth_norm=0.5, tau_bottom=np.ones(2), tau_bottom_wo=np.ones((0, 2)), thin_flux_frac_wo=np.ones(0)
    )
    assembly = SimpleNamespace(theta=np.zeros(4), theta_names=["a"], profile={"molecules": ["H2O"]})
    depth = np.array([1e-3, 2e-3])
    res = forward._RunResult(
        np.array([900.0, 900.0]),
        depth,
        None,
        evidence,
        None,
        [("CS2", 1e-5)],
        {"any_outside": 1, "thermal_rows": 2},
        float("nan"),
        [],
    )
    cp = {"science_mode": "emission", "chem_provider": "vulcan"}
    out = np.load(forward._write_result({}, cp, assembly, setup, col, res), allow_pickle=False)

    # the condensate column carries no gas: 0.5/0.75 and 0.25/0.75, as the RT normalizes
    np.testing.assert_allclose(out["ymix"][0], [2.0 / 3.0, 1.0 / 3.0, 0.0])
    # depth = (Fp / Fs) * norm, so Fp = depth * Fs / norm
    np.testing.assert_allclose(out["fp_flux"], depth * fs / 0.5)
    assert json.loads(str(out["params_json"])) == cp
    assert list(out["unmodeled"]) == ["CS2|1.000e-05"]


def test_an_eclipse_depth_carries_the_emission_radius_and_a_transit_refuses_tau():
    import jax.numpy as jnp

    emis = SimpleNamespace(
        eclipse_flux_tau=lambda *_a, **_k: (jnp.array([3.0, 6.0]), jnp.array([9.0, 9.0])),
        emission_radius=lambda *_a, **_k: 2.0,
    )
    setup = SimpleNamespace(emis=emis, rt=None, cloud_vec=None, fs_j=jnp.array([12.0, 12.0]), rstar_cm_sq=8.0)
    column = spectrum.ColumnSpectrum.__new__(spectrum.ColumnSpectrum)
    column._setup = setup
    column.art_profiles = lambda _y, _th: (None, None, None, None, None)
    np.testing.assert_allclose(column.depth(None, None), np.array([3.0, 6.0]) / 12.0 * (4.0 / 8.0))
    with pytest.raises(ValueError, match="transmission-only"):
        column.depth(None, None, wo_mols=["H2O"])

    setup.emis = None
    with pytest.raises(ValueError, match="emission-only"):
        column.depth(None, None, return_tau=True)


def test_the_gas_mask_drops_only_condensate_columns():
    chem = SimpleNamespace(species_masses=np.ones(5), sidx={"H2": 0, "H2O_l_s": 1, "S8_l": 2, "S2_s": 3, "S8": 4})
    np.testing.assert_array_equal(spectrum.gas_mask(chem), [1, 0, 0, 0, 1])


_ECHOED = {
    "art_ptop_bar": 1e-7,
    "art_pbtm_bar": 7.0,
    "rt_integration": "simpson",
    "p_ref_bar": 1.0,
    "opacity_mode": "exomolop",
}


@pytest.mark.parametrize("wrong", [None, *_ECHOED], ids=["all-honored", *_ECHOED])
def test_an_engine_that_does_not_echo_a_setting_is_refused(wrong, monkeypatch):
    """An engine ignoring a profile key must not cache a spectrum under a key describing physics it skipped."""
    import vulcan_forward

    echo = dict(_ECHOED)
    if wrong is not None:
        echo[wrong] = "not honored"
    rt = SimpleNamespace(p_art_bar=np.array([1.0, 0.1]), **echo)
    fake = SimpleNamespace(build_rt_model=lambda profile: rt)
    monkeypatch.setitem(sys.modules, "vulcan_forward.exojax_rt", fake)
    monkeypatch.setattr(vulcan_forward, "exojax_rt", fake, raising=False)
    cp = {
        "rt_ptop_bar": 1e-7,
        "rt_integration": "simpson",
        "p_ref_bar": 1.0,
        "rstar_rsun": 1.0,
        "science_mode": "transmission",
        "cloud_on": False,
    }
    profile = {"art_pbtm_bar": 7.0, "opacity_mode": "exomolop"}
    tp_eval = object()  # parametric: no tabulated column to map
    if wrong is None:
        setup = spectrum.build_rt_setup(cp, profile, tp_eval, None, None, lambda: None, _quiet)
        assert setup.rt is rt
    else:
        with pytest.raises(RuntimeError, match=f"did not honor {wrong}"):
            spectrum.build_rt_setup(cp, profile, tp_eval, None, None, lambda: None, _quiet)


def _cfg_like_the_constants(**over):
    return SimpleNamespace(
        **{
            **model_defs.CFG_ABUNDANCES,
            "P_t": model_defs.CHEM_P_SPAN_DYN[0],
            "P_b": model_defs.CHEM_P_SPAN_DYN[1],
            **over,
        }
    )


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({}, None),
        ({"O_H": 0.006}, "CFG_ABUNDANCES"),
        ({"C_H": 0.003}, "CFG_ABUNDANCES"),
        ({"N_H": 0.0008}, "CFG_ABUNDANCES"),
        ({"S_H": 0.0002}, "CFG_ABUNDANCES"),
        ({"P_t": 1.0}, "CHEM_P_SPAN_DYN"),
        ({"P_b": 1.0e6}, "CHEM_P_SPAN_DYN"),
    ],
    ids=["same", "O_H", "C_H", "N_H", "S_H", "P_t", "P_b"],
)
def test_a_cfg_that_drifted_from_the_constants_is_refused(over, match, monkeypatch):
    """The constants label every C/O display and export and gate the light path's table span."""
    import vulcan3

    from jwst_tool import chem_build

    cfg = _cfg_like_the_constants(**over)
    monkeypatch.setattr(vulcan3, "load_config", lambda name: cfg)
    config = SimpleNamespace(W39B_CFG_NAME="W39b")
    cp = {"tp_mode": "guillot"}
    if match is None:
        assert chem_build._check_cfg_constants(cp, config, None, _quiet) is cfg
    else:
        with pytest.raises(RuntimeError, match=match):
            chem_build._check_cfg_constants(cp, config, None, _quiet)


def test_a_table_that_changed_after_canonicalization_is_refused(tmp_path, monkeypatch):
    """The cached spectrum is keyed to the content hash, so re-resolving a table whose bytes moved must refuse."""
    from jwst_tool import tp_tables

    monkeypatch.setattr(tp_tables, "uploads_dir", lambda: tmp_path)
    data = b"table bytes"
    sha = hashlib.sha1(data, usedforsecurity=False).hexdigest()[:16]
    path = tmp_path / f"{sha}.txt"
    cp = {"tp_file": model_defs.TP_FILE_UPLOAD, "tp_file_sha1": sha}
    with pytest.raises(RuntimeError, match="missing"):
        tp_tables.tp_file_from_cp(cp)
    path.write_bytes(data)
    assert tp_tables.tp_file_from_cp(cp) == path
    path.write_bytes(b"other bytes")
    with pytest.raises(RuntimeError, match="content drifted"):
        tp_tables.tp_file_from_cp(cp)


def test_the_producer_stamp_names_a_dirty_checkout_and_omits_an_unversioned_one(monkeypatch):
    from jwst_tool import provenance

    monkeypatch.setattr(provenance, "PRODUCER_REPOS", ("a", "b", "c"))
    monkeypatch.setattr(
        provenance,
        "_repos",
        lambda: {
            "a": {"commit": "1" * 40, "dirty": False},
            "b": {"commit": "2" * 40, "dirty": True},
            "c": {"commit": "unversioned", "dirty": False},
        },
    )
    assert provenance.producer_commits() == ["a=" + "1" * 12, "b=" + "2" * 12 + "+dirty"]
