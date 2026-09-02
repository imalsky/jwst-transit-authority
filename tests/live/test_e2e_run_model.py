"""Full-chain end-to-end regression: VULCAN chemistry -> ExoJAX RT -> spectrum.

Runs the tool's real `run_model` for the default WASP-39 b case in BOTH
science modes -- a fresh chemistry solve each (the cached spectrum is deleted
first, so this validates the chain, not the cache) -- and asserts the R = 100
binned spectrum against a committed reference fixture generated under the
same `forward._VERSION`. The engine side of the physics is separately pinned
against petitRADTRANS in vulcan-forward's own e2e suite; THIS test pins that
the whole tool still reproduces its own verified state end to end.

Cost: ~2 minutes per mode (chemistry warm-up dominates). Gated like the FD
closure test: JWST_TOOL_RUN_SLOW=1, plus clean skips without the RT stack or
the engine data. JWST_TOOL_E2E_ALLOW_CACHE=1 skips the cache delete for a
quick re-check (then it validates the cache, not the chain -- so the flag is
opt-in and says so).

Tolerances: cross-run JAX nondeterminism on this chain measures well under a
ppm; 50 ppm absolute / 2% relative catches any physics change while never
flaking on solver noise. On a breach: if the change is INTENDED, regenerate
the fixture (tools note inside its meta) and bump forward._VERSION.
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
    reason="full-chain e2e (~5 min): set JWST_TOOL_RUN_SLOW=1")

if importlib.util.find_spec("exojax") is None or \
        importlib.util.find_spec("vulcan_jax") is None:
    pytest.skip("RT stack not installed", allow_module_level=True)

FIXTURE = Path(__file__).parent / "data" / "w39b_v41_reference.npz"

WL_LO, WL_HI, R_BIN = 1.02, 5.26, 100.0


def bin_r100(wl_model, y_model):
    """Cell means of a piecewise-linear model over R=100 log-spaced bins.

    Trapezoid cumulative + linear interpolation at the edges: NOT the exact
    `binning._pl_antideriv` operator the science path requires -- that
    distinction matters when comparing against DATA, but here generator and
    test share this one definition, so the operator cancels exactly.
    Returns (bin centers, cell means) for bins fully inside the model span.
    """
    wl_model = np.asarray(wl_model, float)
    y_model = np.asarray(y_model, float)
    order = np.argsort(wl_model)
    wl_model, y_model = wl_model[order], y_model[order]
    n = int(np.floor(R_BIN * np.log(WL_HI / WL_LO)))
    edges = WL_LO * np.exp(np.arange(n + 1) / R_BIN)
    edges = edges[(edges >= wl_model[0]) & (edges <= wl_model[-1])]
    cum = np.concatenate(
        [[0.0], np.cumsum(0.5 * (y_model[1:] + y_model[:-1])
                          * np.diff(wl_model))])
    integ = np.interp(edges, wl_model, cum)
    means = np.diff(integ) / np.diff(edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return centers, means


def _load_fixture():
    if not FIXTURE.exists():
        pytest.skip(f"reference fixture absent: {FIXTURE}")
    z = np.load(FIXTURE)
    return z, json.loads(bytes(np.asarray(z["meta"])))


def _engine_data_ready():
    from vulcan_forward import exomolop, paths
    try:
        paths.data_root()
    except RuntimeError:
        pytest.skip("VULCAN_FORWARD_DATA not configured")
    if not exomolop.table_path("H2O").exists():
        pytest.skip("ExoMolOP tables absent; "
                    "python -m vulcan_forward.fetch_exomolop --molecules H2O,...")


@pytest.mark.parametrize("mode", ["transmission", "emission"])
def test_full_chain_reproduces_the_verified_reference(mode):
    from jwst_tool import forward

    _engine_data_ready()
    z, meta = _load_fixture()
    ref = meta[mode]
    params = dict(ref["params"])
    cp = forward.canonical_params(params)
    key = forward.params_key(cp)
    assert meta["forward_version"] == forward._VERSION, (
        "fixture was generated under a different forward._VERSION -- "
        "regenerate it (meta['regenerate'])")
    assert key == ref["params_key"], (
        "the canonical parameter set moved since the fixture was generated")

    if os.environ.get("JWST_TOOL_E2E_ALLOW_CACHE") != "1":
        # BOTH caches: cache_path is the SPECTRUM, chem_cache_path the converged
        # CHEMISTRY. Clearing only the first re-runs the RT off a cached column,
        # which is not the full-chain solve this test exists to exercise.
        for cache in (forward.cache_path(params), forward.chem_cache_path(params)):
            cache.unlink(missing_ok=True)

    out = np.load(forward.run_model(params, log=lambda *a: None),
                  allow_pickle=False)
    got_cp = json.loads(str(out["params_json"]))
    assert got_cp["science_mode"] == mode
    assert str(out["science_mode"]) == mode

    wl_c, got = bin_r100(out["wl_um"], np.asarray(out["depth"]) * 1e6)
    want = np.asarray(z[f"{mode}_depth_ppm"])
    assert wl_c.shape == want.shape
    dev = got - want
    label = (f"{mode}: max|d depth| {np.max(np.abs(dev)):.2f} ppm, "
             f"std ratio {np.std(got) / np.std(want):.5f}")
    print("\n  " + label)
    assert np.max(np.abs(dev)) < 50.0, label
    assert abs(np.std(got) / np.std(want) - 1.0) < 0.02, label

    m = (wl_c > 3.0) & (wl_c < 5.26)
    assert abs(np.median(got[m]) - ref["median_ppm_3_5um"]) < 50.0
    assert abs(np.ptp(got[m]) / ref["ptp_ppm_3_5um"] - 1.0) < 0.05
