"""PICASO engine vs a published PICASO spectrum: WASP-39 b, Rustamkulov et al.
2023 (Nature 614, 659), the PICASO 3.0 best fit to the NIRSpec PRISM ERS
spectrum (Zenodo 10.5281/zenodo.7388032, CC-BY-4.0).

Published parameters: 10x solar, C/O 0.68, a grey cloud kappa = 10^-2.07 cm2/g
per gram of atmosphere (their eq. 3, the same deck as the tool's gray cloud),
H2-only Rayleigh, SO2 injected at log10 VMR -5.2. The geometry is the PICASO
WASP-39 b grid's (R = 1.279 RJ at 10 bar, 0.28 MJ, Teff 5326.6 K, a = 0.0486 AU;
attrs of best_fit_PICASO_grey.nc in Zenodo 7101283). The T(P) is this tool's
own PICASO climate at those parameters (T_int 200 K), frozen in the data file
so the test costs one chemistry column and one RT pass (~1 min on daw) instead
of a climate solve; the climate itself is pinned by the e2e fixture.

What this pins: composition imposition, equilibrium chemistry, ExoMolOP
correlated-k RT, the gray cloud and the radius/gravity anchor, together,
against an independent code's published spectrum. At R = 100 over 1.02-5.30 um
with the SO2 band (3.95-4.15 um) excluded and a constant offset removed, the
residual rms is 39.0 ppm (std ratio 0.937; H2O 1.4 um amplitude ratio 0.95,
CO2 4.3 um 0.95 against a 3.70-3.85 um continuum). The published spectrum's
std is 390 ppm, so the 60 ppm bound trips on a ~10% amplitude change or an
equivalent shape change: the same column clear gives 424 ppm, and a climate
run at the tool's default radius anchor (1.279 RJ at 1 mbar, the registry
star and orbit) 120 ppm. The inter-grid spread of
the four published ERS best fits on the same data is 70-122 ppm rms. Peak
memory 5.4 GB (the RT), 53 s on one daw core.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("JWST_TOOL_RUN_SLOW") != "1",
    reason="PICASO vs published spectrum (~1 min): set JWST_TOOL_RUN_SLOW=1")

if importlib.util.find_spec("exojax") is None or \
        importlib.util.find_spec("vulcan3") is None:
    pytest.skip("RT stack not installed", allow_module_level=True)

DATA = Path(__file__).parent / "data" / "w39b_picaso_prism2023_r100.npz"
SO2_BAND_UM = (3.95, 4.15)
RMS_MAX_PPM = 60.0


def _cell_means(wl, y, lo, hi):
    """Cell means of the piecewise-linear model on the data file's cells (the
    operator its published side was binned with; test_e2e_run_model.bin_r100
    builds its own fixed edges, so it cannot take these)."""
    o = np.argsort(wl)
    wl, y = np.asarray(wl, float)[o], np.asarray(y, float)[o]
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(wl))])
    return (np.interp(hi, wl, cum) - np.interp(lo, wl, cum)) / (hi - lo)


def test_picaso_engine_reproduces_the_published_prism_picaso_spectrum(tmp_path,
                                                                      monkeypatch):
    from jwst_tool import datacheck, forward, picaso_engine
    from vulcan_forward import exomolop, paths

    try:
        paths.data_root()
    except RuntimeError:
        pytest.skip("VULCAN_FORWARD_DATA not configured")
    if not exomolop.table_path("H2O").exists():
        pytest.skip("ExoMolOP tables absent")
    if any(i.status != datacheck.OK for i in datacheck.check_picaso()):
        pytest.skip("PICASO environment or data not installed")
    monkeypatch.setattr(picaso_engine, "PICASO_CACHE", tmp_path)
    monkeypatch.setattr(forward, "MODEL_CACHE", tmp_path / "model_cache")

    z = np.load(DATA)
    meta = json.loads(bytes(np.asarray(z["meta"])))
    tp = tmp_path / "w39b_picaso_climate.txt"
    with open(tp, "w") as fh:
        fh.write("#(dyne/cm2) (K)\nPressure\tTemp\n")
        p, T = np.asarray(z["p_bar"]), np.asarray(z["T"])
        pd = np.logspace(np.log10(p[0]), np.log10(p[-1]), 600)   # ln-P dense
        for a, b in zip(pd[::-1] * 1e6, np.interp(np.log(pd), np.log(p), T)[::-1]):
            fh.write(f"{a:.6e}\t{b:.4f}\n")
    cp = meta["T_profile"]["params"]
    params = dict(planet="wasp39b", chem_provider="picaso", tp_mode="file",
                  tp_file="upload", tp_file_path=str(tp), p_btm_bar=10.0,
                  met_x_solar=10.0, co_ratio=0.687,
                  **{k: cp[k] for k in ("star_teff", "star_logg", "star_feh",
                                        "orbit_au", "rp_rjup", "gs_cgs", "p_ref_bar")},
                  cloud_on=True, log_kappa_cloud=-2.07, alpha_cloud=0.0,
                  wo_mols=[], fisher_params=[])
    out = np.load(forward.run_model(params, log=lambda *a: None), allow_pickle=False)

    lo, hi, pub = (np.asarray(z[k]) for k in ("lo_um", "hi_um", "published_ppm"))
    got = _cell_means(out["wl_um"], np.asarray(out["depth"]) * 1e6, lo, hi)
    keep = np.logical_not((hi > SO2_BAND_UM[0]) & (lo < SO2_BAND_UM[1]))
    r = got[keep] - pub[keep]
    r -= r.mean()
    rms = float(np.sqrt(np.mean(r ** 2)))
    label = (f"rms {rms:.1f} ppm, max {np.abs(r).max():.1f} ppm, std ratio "
             f"{np.std(got[keep]) / np.std(pub[keep]):.3f}")
    print("\n  " + label)
    assert rms < RMS_MAX_PPM, label
