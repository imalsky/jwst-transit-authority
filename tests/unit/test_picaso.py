"""PICASO engine, host side: composition, certificates, species map, worker
protocol. numpy only: the worker is replaced by a stand-in script."""
import json
import sys
import textwrap

import numpy as np
import pytest

from jwst_tool import forward
from jwst_tool import instruments as ins
from jwst_tool import picaso_engine as pe


@pytest.mark.parametrize("met,co", [(1.0, 0.3), (10.0, 0.55), (100.0, 1.5)])
def test_composition_is_the_vulcan_convention(met, co):
    xh = pe.composition(met, co)
    ab = forward.elemental_abundances(met, co)
    assert {k: xh[k] for k in "CNOS"} == {k: ab[f"{k}_H"] for k in "CNOS"}
    assert xh["C"] / xh["O"] == pytest.approx(co, rel=1e-12)   # C moves, O fixed
    assert xh["He"] == pe.HE_H
    assert xh["Fe"] == pytest.approx(met * 10 ** (7.45 - 12), rel=1e-12)
    assert set(xh) == {"He", "C", "N", "O", "S"} | set(pe.LODDERS2020_PD)


def _result(n=4, climate=True):
    atoms_names = np.array(["H", "He", "C", "N", "O", "S"])
    target = np.array([0.8, 0.1, 0.03, 0.01, 0.05, 0.01])
    res = {"converged": np.ones(n, bool), "atoms_names": atoms_names,
           "atoms": np.tile(target, (n, 1)), "target": target, "chem_fail": np.array(0),
           "vmr": np.full((n, 2), 0.5), "T": np.full(n, 1000.0)}
    if climate:
        P = np.logspace(-6, 2, 20)
        res.update(clim_p=P, clim_T=1000.0 * (1 + 0.3 * np.log10(P / 1e-6) / 8),
                   conv_flag=np.array(1), flux_net=np.array([1e-6]),
                   tidal=np.array([-1.0]), cvz=np.array([0, 15, 18, 0, 0, 0]))
    return res


def _req(n=4, climate=True):
    return {"p_bar": list(np.logspace(-5, 1, n)), "climate": {} if climate else None}


def test_certificate_passes_and_refuses_each_gate():
    assert "flux_toa" in pe.certify(_req(), _result(), "t")
    assert set(pe.certify(_req(climate=False), _result(climate=False), "t")) == {"element_err"}

    def bad(**kw):
        r = _result()
        for k, v in kw.items():
            if k == "layer":
                r["converged"][v] = False
            elif k == "carbon":
                r["atoms"][1, 2] *= 1.0 + v
            elif k == "hot":
                r["clim_T"] = r["clim_T"] * v
            else:
                r[k] = v
        return r
    # a failed column layer is also counted in chem_fail; the layer gate names it
    for r, msg in [(bad(layer=2, chem_fail=np.array(1)), "did not converge in 1 of 4"),
                   (bad(carbon=2e-3), "element budget"),
                   (bad(conv_flag=np.array(0)), "did not converge"),
                   (bad(flux_net=np.array([1e-2])), "flux balance"),
                   (bad(cvz=np.array([0, 3, 18, 0, 0, 0])), "model top"),
                   (bad(hot=4.5), "correlated-k grid"),
                   (bad(chem_fail=np.array(2)), "solves did not converge inside picaso"),
                   (bad(flux_net=np.array([np.nan])), "flux balance"),
                   (bad(clim_T=500.0 * np.logspace(0, 8, 20) ** 0.6), "dlnT/dlnP"),
                   (bad(clim_T=np.r_[np.nan, np.ones(19)]), "not finite and ordered"),
                   (bad(carbon=np.nan), "non-finite atoms")]:
        with pytest.raises(RuntimeError, match=msg):
            pe.certify(_req(), r, "t")


def test_column_maps_names_drops_ions_and_runs_bottom_to_top():
    names = np.array(["H2", "He", "HCHO", "SN", "H2O", "e-", "H+", "H-"])
    vmr = np.arange(5 * 8, dtype=float).reshape(5, 8) + 1.0
    res = {"gas_names": names, "vmr": vmr, "p_eval": np.logspace(-6, 1, 5),
           "T": np.arange(5.0), "gas_mass": np.arange(8.0), "cert": {},
           "quench_json": np.array("{}"), "versions_json": np.array("{}")}
    col = pe.column(res, 4)            # the 5th level is quench-search depth
    assert list(col.sidx) == ["H2", "He", "H2CO", "NS", "H2O"]
    assert col.p_bar[0] > col.p_bar[-1] and col.p_bar.size == 4
    assert np.array_equal(col.y[0], vmr[3, :5]) and np.array_equal(col.T_base, [3, 2, 1, 0])
    assert np.array_equal(col.species_masses, np.arange(5.0))


_FAKE_WORKER = textwrap.dedent("""
    import json, sys, numpy as np
    job = json.load(open(sys.argv[1]))
    open(job["refdata"] + "/calls", "a").write("x")
    if job["request"]["metallicity"] < 0:
        raise RuntimeError("picaso exploded on purpose")
    t = np.array([0.8, 0.1, 0.03, 0.01, 0.05, 0.01])
    n = len(job["request"]["p_bar"])
    print("[pic] climate solve: 30 s", flush=True)
    np.savez(sys.argv[2], versions_json=np.array("{}"), converged=np.ones(n, bool),
             target=t, atoms=np.tile(t, (n, 1)),
             atoms_names=np.array(["H", "He", "C", "N", "O", "S"]),
             vmr=np.ones((n, 2)), gas_names=np.array(["H2", "He"]),
             gas_mass=np.array([2.0, 4.0]), T=np.full(n, 900.0),
             quench_json=np.array("{}"), chem_fail=np.array(0))
""")


def test_solve_runs_the_worker_once_caches_and_surfaces_failures(tmp_path, monkeypatch):
    (tmp_path / "picaso_worker.py").write_text(_FAKE_WORKER)
    monkeypatch.setattr(ins, "PICASO_PYTHON", sys.executable)
    monkeypatch.setattr(ins, "TOOL_DIR", tmp_path)
    monkeypatch.setattr(ins, "PICASO_REFDATA", str(tmp_path))
    monkeypatch.setattr(pe, "PICASO_CACHE", tmp_path / "cache")
    monkeypatch.setattr(pe, "LOCK_DIR", tmp_path / "locks")
    req = {"metallicity": 10.0, "p_bar": [1e-3, 1e-1, 1.0], "climate": None}
    lines = []
    first = pe.solve(req, "probe", lines.append)
    again = pe.solve(req, "probe", lines.append)
    assert (tmp_path / "calls").read_text() == "x"          # one worker launch
    assert first["cert"] == again["cert"] and "[fwd] [pic] climate solve: 30 s" in lines
    with pytest.raises(RuntimeError, match="PICASO probe: picaso exploded on purpose"):
        pe.solve(dict(req, metallicity=-1.0), "probe", lines.append)
    assert [p.suffix for p in (tmp_path / "cache").iterdir()] == [".npz"]   # failures never cached
    json.dumps(first["cert"])        # the certificate is stored as JSON in the npz
    # a cached result that no longer certifies refuses at once, with no re-solve
    monkeypatch.setattr(pe, "ELEMENT_TOL", -1.0)
    with pytest.raises(RuntimeError, match="element budget"):
        pe.solve(req, "probe", lines.append)
    assert (tmp_path / "calls").read_text() == "xx"   # only the exploding launch


def test_no_picaso_interpreter_is_one_actionable_error(monkeypatch):
    monkeypatch.setattr(ins, "PICASO_PYTHON", None)
    with pytest.raises(RuntimeError, match="JWST_TOOL_PICASO_PYTHON"):
        pe.require_picaso_python()
