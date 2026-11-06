"""PICASO engine vs the published PICASO best fit to WASP-39 b (Rustamkulov et al. 2023).

Why, cost and tolerances: docs/validation.md, section `test_picaso_published`.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from jwst_tool import model_cache

pytestmark = pytest.mark.skipif(
    os.environ.get("JWST_TOOL_RUN_SLOW") != "1",
    reason="PICASO vs published spectrum (~1 min): set JWST_TOOL_RUN_SLOW=1",
)

if importlib.util.find_spec("exojax") is None or importlib.util.find_spec("vulcan3") is None:
    pytest.skip("RT stack not installed", allow_module_level=True)

DATA = Path(__file__).parent / "data" / "w39b_picaso_prism2023_r100.npz"
SO2_BAND_UM = (3.95, 4.15)
RMS_MAX_PPM = 60.0


def _cell_means(wl, y, lo, hi):
    """Cell means of the piecewise-linear model on the data file's cells (the
    operator its published side was binned with; test_e2e_run_model.bin_r100
    builds its own fixed edges, so it cannot take these).
    """
    o = np.argsort(wl)
    wl, y = np.asarray(wl, float)[o], np.asarray(y, float)[o]
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(wl))])
    return (np.interp(hi, wl, cum) - np.interp(lo, wl, cum)) / (hi - lo)


def test_picaso_engine_reproduces_the_published_prism_picaso_spectrum(tmp_path, monkeypatch):
    from vulcan_forward import exomolop, paths

    from jwst_tool import datacheck, forward, picaso_engine

    try:
        paths.data_root()
    except RuntimeError:
        pytest.skip("VULCAN_FORWARD_DATA not configured")
    if not exomolop.table_path("H2O").exists():
        pytest.skip("ExoMolOP tables absent")
    if any(i.status != datacheck.OK for i in datacheck.check_picaso()):
        pytest.skip("PICASO environment or data not installed")
    monkeypatch.setattr(picaso_engine, "PICASO_CACHE", tmp_path)
    monkeypatch.setattr(model_cache, "MODEL_CACHE", tmp_path / "model_cache")

    z = np.load(DATA)
    meta = json.loads(bytes(np.asarray(z["meta"])))
    tp = tmp_path / "w39b_picaso_climate.txt"
    with open(tp, "w") as fh:
        fh.write("#(dyne/cm2) (K)\nPressure\tTemp\n")
        p, T = np.asarray(z["p_bar"]), np.asarray(z["T"])
        pd = np.logspace(np.log10(p[0]), np.log10(p[-1]), 600)  # ln-P dense
        fh.writelines(
            f"{a:.6e}\t{b:.4f}\n"
            for a, b in zip(pd[::-1] * 1e6, np.interp(np.log(pd), np.log(p), T)[::-1], strict=True)
        )
    cp = meta["T_profile"]["params"]
    params = dict(
        planet="wasp39b",
        chem_provider="picaso",
        tp_mode="file",
        tp_file="upload",
        tp_file_path=str(tp),
        p_btm_bar=10.0,
        met_x_solar=10.0,
        co_ratio=0.687,
        **{k: cp[k] for k in ("star_teff", "star_logg", "star_feh", "orbit_au", "rp_rjup", "gs_cgs", "p_ref_bar")},
        cloud_on=True,
        log_kappa_cloud=-2.07,
        alpha_cloud=0.0,
        wo_mols=[],
        fisher_params=[],
    )
    out = np.load(forward.run_model(params, log=lambda *a: None), allow_pickle=False)

    lo, hi, pub = (np.asarray(z[k]) for k in ("lo_um", "hi_um", "published_ppm"))
    got = _cell_means(out["wl_um"], np.asarray(out["depth"]) * 1e6, lo, hi)
    keep = np.logical_not((hi > SO2_BAND_UM[0]) & (lo < SO2_BAND_UM[1]))
    r = got[keep] - pub[keep]
    r -= r.mean()
    rms = float(np.sqrt(np.mean(r**2)))
    label = f"rms {rms:.1f} ppm, max {np.abs(r).max():.1f} ppm, std ratio {np.std(got[keep]) / np.std(pub[keep]):.3f}"
    print("\n  " + label)
    assert rms < RMS_MAX_PPM, label
