"""Always-on sanity for the two full-chain e2e reference fixtures (VULCAN and
PICASO; numpy-only).

Runs in the fast suite and CI, so the fixtures behind
tests/live/test_e2e_run_model.py cannot rot silently:

- each stays parseable, finite, and complete;
- its recorded forward._VERSION must equal the LIVE forward._VERSION -- a
  version bump without a fixture regeneration fails HERE, loudly, instead of
  leaving a stale reference that the gated live test would one day chase;
- its recorded cache keys must equal params_key(canonical_params(params))
  recomputed NOW (canonical_params is numpy-only), so any canonical-set
  change surfaces immediately.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jwst_tool import forward

_DATA = Path(__file__).parent.parent / "live" / "data"


# the VULCAN default and the PICASO engine's default (climate T-P)
@pytest.fixture(scope="module", params=["w39b_v59_reference.npz",
                                        "w39b_picaso_reference.npz"])
def fixture(request):
    path = _DATA / request.param
    assert path.exists(), (
        f"missing {path}: the full-chain e2e reference must ship with the "
        "repo (regenerate per its meta['regenerate'])")
    z = np.load(path)
    return z, json.loads(bytes(np.asarray(z["meta"])))


def test_fixture_matches_the_live_version(fixture):
    _, meta = fixture
    assert meta["forward_version"] == forward._VERSION, (
        f"fixture generated under v{meta['forward_version']} but forward is "
        f"at v{forward._VERSION}: regenerate the fixture with the recipe in "
        "its meta['script'] (a version bump is not complete until the "
        "reference moves with it)")


@pytest.mark.parametrize("mode", ["transmission", "emission"])
def test_recorded_keys_recompute_today(fixture, mode):
    _, meta = fixture
    ref = meta[mode]
    cp = forward.canonical_params(dict(ref["params"]))
    assert forward.params_key(cp) == ref["params_key"], (
        f"{mode}: the canonical parameter set moved since the fixture was "
        "generated -- regenerate it")
    assert cp["science_mode"] == mode


@pytest.mark.parametrize("mode", ["transmission", "emission"])
def test_arrays_parse_and_are_physical(fixture, mode):
    z, meta = fixture
    wl = np.asarray(z[f"{mode}_wl_um"])
    d = np.asarray(z[f"{mode}_depth_ppm"])
    assert wl.shape == d.shape and wl.size > 100
    assert np.all(np.isfinite(wl)) and np.all(np.isfinite(d))
    assert np.all(np.diff(wl) > 0)
    assert np.all(d > 0)
    ref = meta[mode]
    m = (wl > 3.0) & (wl < 5.26)
    assert np.median(d[m]) == pytest.approx(ref["median_ppm_3_5um"], rel=1e-9)
    assert np.ptp(d[m]) == pytest.approx(ref["ptp_ppm_3_5um"], rel=1e-9)
