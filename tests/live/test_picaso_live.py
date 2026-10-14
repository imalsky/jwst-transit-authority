"""PICASO engine, live: the real worker in the PICASO environment.

Gated like the other slow tests (JWST_TOOL_RUN_SLOW=1), plus a clean skip
without the PICASO environment or its reference data.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess

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


# Stock picaso: atmosphere(chem_method='on-the-fly') solves the column with its
# own EquilibriumChemistry (layer-to-layer warm start). Its composition is the
# Lodders 2020 protosolar table with every atom but H and He scaled by mh, and
# C/O moved at fixed C+O to C/O_rel = cto_absolute / LODDERS2020_C_TO_O
# (picaso/photochem.py LODDERS2020_SUN_COMP, picaso/justdoit.py
# chemeq_on_the_fly, photochem solve_metallicity); at picaso's solar C/O the
# move is zero.
_STOCK = r"""
import json, os, sys
os.environ["picaso_refdata"], os.environ["PYSYN_CDBS"] = sys.argv[1], sys.argv[2]
import picaso.justdoit as jdi
from picaso.photochem import LODDERS2020_SUN_COMP as L
a = json.load(open(sys.argv[3]))
b = jdi.inputs(calculation="planet", climate=True)
b.add_pt(T=a["T"], P=a["p"])
b.atmosphere(mh=a["mh"], cto_absolute=jdi.LODDERS2020_C_TO_O, chem_method="on-the-fly")
s = b.inputs["climate"]["chemeq_solver"]
prof = b.inputs["atmosphere"]["profile"]
json.dump({"xh": {k: L[k] / L["H"] * (1.0 if k == "He" else a["mh"])
                  for k in s.atoms_names if k != "H"},
           "names": list(s.gas_names),
           "vmr": [prof[k].tolist() for k in s.gas_names]}, open(sys.argv[4], "w"))
"""


@pytest.mark.parametrize("met", [1.0, 10.0])
def test_picaso_worker_reproduces_stock_picaso_at_its_own_composition(met, tmp_path,
                                                                      monkeypatch):
    """Given the atom fractions stock picaso uses for (mh, solar C/O), the
    composition-locked worker returns stock picaso's gas column to 1e-6 dex
    over the photosphere (measured ~1e-11 at 1x and 10x on a WASP-39 b
    climate column). This pins the injected solver itself -- atom order,
    normalization, the cold per-layer solves -- against unmodified picaso,
    over all 20 atoms including the metals the seed comparison zeroes."""
    monkeypatch.setattr(pe, "PICASO_CACHE", tmp_path)
    p = np.logspace(-6, 2, 61)
    T = np.interp(np.log10(p), [-6, -3, -1, 0, 2], [950, 980, 1150, 1400, 2250])
    args, out = tmp_path / "in.json", tmp_path / "stock.json"
    args.write_text(json.dumps({"p": p.tolist(), "T": T.tolist(), "mh": met}))
    env = dict(os.environ, OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
               NUMBA_CACHE_DIR=str(ins.OUTPUT_DIR / "numba_cache" / "picaso"))
    subprocess.run([ins.PICASO_PYTHON, "-c", _STOCK, str(ins.PICASO_REFDATA),
                    ins.PYSYN_CDBS, str(args), str(out)], check=True, env=env)
    stock = json.loads(out.read_text())
    res = pe.solve({"xh": stock["xh"], "metallicity": met, "g_cgs": 500.0,
                    "p_bar": p.tolist(), "T": T.tolist(), "quench_kzz": None,
                    "climate": None}, "stock identity", lambda s: None)
    assert [str(s) for s in res["gas_names"]] == stock["names"]
    a = np.asarray(stock["vmr"]).T
    b = np.asarray(res["vmr"], float)
    band = (p >= 1e-4) & (p <= 1.0)
    a, b = a[band], b[band]
    m = np.maximum(a, b) > 1e-8
    assert m.any(axis=0).sum() > 20
    assert np.abs(np.log10(b[m] / a[m])).max() < 1e-6


# FastChem 4.0.3 (pyfastchem 4.0.3 with logK.dat and logK_condensates.dat of the
# v4.0.3 source tag, SiO(s) removed: an amorphous-solid entry from Gail et al.
# 2013 that picaso's thermo file does not carry), local equilibrium
# condensation, at pe.composition(Z, 0.55) without Li, Rb and Cs (no species in
# that set; X/H < 2e-8 at 10x). log10 VMR of _SP.
# Every point is settled in FastChem: Mg and Si either > 99% condensed (MgSiO3 +
# Mg2SiO4 hold 18% of the oxygen, H2O -0.23 dex in the CO regime) or < 1%.
_P = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)
_T = (1100.0, 2000.0, 1200.0, 1750.0, 1250.0, 2200.0, 1000.0, 2000.0, 900.0)
_SP = ("H2O", "CO", "CO2", "CH4", "NH3", "H2S")
_FASTCHEM = {
    1.0: ((-3.6113, -3.2963, -6.8419, -10.3650, -9.6998, -4.6189),
          (-3.4695, -3.3147, -7.3359, -14.5675, -10.4921, -6.1219),
          (-3.6110, -3.2963, -6.9740, -9.2630, -8.9197, -4.6196),
          (-3.4438, -3.2971, -7.2292, -11.6076, -9.2158, -4.8807),
          (-3.6109, -3.2963, -7.0310, -7.6592, -8.0170, -4.6187),
          (-3.4535, -3.3028, -7.3958, -11.0294, -8.5772, -5.2153),
          (-3.3684, -3.4936, -6.6330, -3.7337, -6.4391, -4.6171),
          (-3.4430, -3.2969, -7.3311, -8.4575, -7.4239, -4.7646),
          (-3.1263, -6.1154, -8.8075, -3.2964, -5.1363, -4.6168)),
    10.0: ((-2.6159, -2.2996, -4.8464, -10.3741, -9.2061, -3.6227),
           (-2.4738, -2.3179, -5.3388, -14.5803, -10.0004, -5.1390),
           (-2.6153, -2.2993, -4.9779, -9.2721, -8.4260, -3.6226),
           (-2.4478, -2.3005, -5.2319, -11.6207, -8.7241, -3.9015),
           (-2.6150, -2.2992, -5.0347, -7.6683, -7.5233, -3.6212),
           (-2.4574, -2.3059, -5.3982, -11.0425, -8.0855, -4.2280),
           (-2.5549, -2.3359, -4.6581, -3.4002, -5.9450, -3.6192),
           (-2.4467, -2.3001, -5.3335, -8.4708, -6.9321, -3.7707),
           (-2.1289, -4.1033, -5.7916, -2.3010, -4.6358, -3.6152)),
}


@pytest.mark.parametrize("met", [1.0, 10.0])
def test_picaso_metal_oxygen_sink_matches_fastchem(met, tmp_path, monkeypatch):
    """The shipped composition with its metals, through the worker: every
    carrier within 0.05 dex of FastChem wherever either VMR exceeds 1e-8. This
    pins the metals' oxygen sink (0.23 dex in H2O, so a missing or doubled sink
    fails) against an independent thermochemical database. picaso's MgO(c) and
    SiO2(c) proxies take the same oxygen per Mg and Si as the silicates once
    condensed; the points avoid their condensation fronts, where they differ."""
    monkeypatch.setattr(pe, "PICASO_CACHE", tmp_path)
    res = pe.solve({"xh": pe.composition(met, 0.55), "metallicity": met, "g_cgs": 500.0,
                    "p_bar": list(_P), "T": list(_T), "quench_kzz": None,
                    "climate": None}, "FastChem comparison", lambda s: None)
    names = [str(s) for s in res["gas_names"]]
    pic = np.log10(np.asarray(res["vmr"])[:, [names.index(s) for s in _SP]])
    ref = np.asarray(_FASTCHEM[met])
    err = np.where(np.maximum(pic, ref) > -8.0, np.abs(pic - ref), 0.0)
    i, j = np.unravel_index(int(np.argmax(err)), err.shape)
    assert err[i, j] < 0.05, (f"{_SP[j]} at {_P[i]:g} bar, {_T[i]:g} K: log10 VMR "
                              f"{pic[i, j]:.3f}, FastChem {ref[i, j]:.3f}")
