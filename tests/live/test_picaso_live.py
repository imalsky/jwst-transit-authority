"""PICASO engine, live: the real worker in the PICASO environment.

Gated like the other slow tests (JWST_TOOL_RUN_SLOW=1), plus a clean skip
without the PICASO environment or its reference data.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from jwst_tool import datacheck
from jwst_tool import instruments as ins
from jwst_tool import picaso_engine as pe

pytestmark = pytest.mark.skipif(
    os.environ.get("JWST_TOOL_RUN_SLOW") != "1",
    reason="PICASO live tests: set JWST_TOOL_RUN_SLOW=1")

if any(i.status != datacheck.OK for i in datacheck.check_picaso()) or \
        importlib.util.find_spec("vulcan_jax") is None:
    pytest.skip("PICASO environment, its data or the VULCAN stack not installed",
                allow_module_level=True)

_SP = ("H2O", "CH4", "CO", "CO2", "NH3", "N2", "HCN", "H2S")


@pytest.mark.parametrize("met", [1.0, 10.0, 100.0])
def test_picaso_chemistry_is_the_vulcan_seed_on_the_same_column(met, tmp_path,
                                                                 monkeypatch):
    """Same T-P (the shipped WASP-39 b table), same C/N/O/S/He: PICASO's
    on-the-fly equilibrium and VULCAN's ExoGibbs seed agree to 0.01 dex on
    every major carrier over the photosphere. This pins the composition
    injection AND the thermochemical data of both codes; the production
    metals (silicate oxygen sink) are deliberately zeroed here."""
    from jwst_tool import forward
    monkeypatch.setattr(pe, "PICASO_CACHE", tmp_path)
    cp = forward.canonical_params(dict(planet="wasp39b", tp_mode="file",
                                       use_photo=False, met_x_solar=met))
    chem = forward._assemble_chem(cp, lambda s: None).build_chem()
    p = np.asarray(chem.p_bar)[::-1]
    T = np.asarray(chem.T_base, float)[::-1]
    xh = {k: (v if k in ("He", "C", "N", "O", "S") else 1e-30)
          for k, v in pe.composition(met, cp["co_ratio"]).items()}
    res = pe.solve({"xh": xh, "metallicity": met, "g_cgs": 500.0,
                    "p_bar": p.tolist(), "T": T.tolist(), "quench_kzz": None,
                    "climate": None}, "seed comparison", lambda s: None)
    names = [str(s) for s in res["gas_names"]]
    y_pic = np.asarray(res["vmr"])
    y_vul = np.asarray(chem.y0, float)[::-1]
    y_vul = y_vul / y_vul.sum(axis=1, keepdims=True)
    band = (p >= 1e-5) & (p <= 1.0)
    for sp in _SP:
        a = y_vul[band, chem.sidx[sp]]
        b = y_pic[band, names.index(sp)]
        m = np.maximum(a, b) > 1e-8
        assert np.abs(np.log10(b[m] / a[m])).max() < 0.01, sp


def test_picaso_worker_refuses_a_missing_climate_table(tmp_path, monkeypatch):
    """picaso drops a missing correlated-k table with only a warning; the
    worker must refuse instead of solving the climate without it."""
    src = ins.PICASO_REFDATA
    root = tmp_path / "ref"
    for rel in datacheck.PICASO_REFDATA_MD5:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).symlink_to(os.path.join(src, rel))
    ck = root / datacheck.PICASO_CK_DIR
    ck.mkdir(parents=True)
    for f in sorted(datacheck.PICASO_CK_MD5)[1:]:          # one table missing
        (ck / f).symlink_to(os.path.join(src, datacheck.PICASO_CK_DIR, f))
    monkeypatch.setattr(ins, "PICASO_REFDATA", str(root))
    monkeypatch.setattr(pe, "PICASO_CACHE", tmp_path / "cache")
    p = np.logspace(-6, 1, 20)
    req = {"xh": pe.composition(10.0, 0.55), "metallicity": 10.0, "g_cgs": 500.0,
           "p_bar": p.tolist(), "T": np.full(p.size, 1200.0).tolist(),
           "quench_kzz": 1.0e9, "climate": None}
    with pytest.raises(RuntimeError, match="correlated-k tables"):
        pe.solve(req, "missing table", lambda s: None)
