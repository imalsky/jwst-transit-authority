"""Per-mode noise caching (0.27.0): run_modes computes ONLY the cache
misses, batches them into one worker invocation, and stores each mode under
its own single-mode job key -- so a selection change costs exactly the newly
added modes. The worker is stubbed; no pandeia involved."""
from __future__ import annotations

import json

import pytest

from jwst_tool import instruments as ins
from jwst_tool import noise

STAR = {"teff": 5400.0, "log_g": 4.45, "metallicity": 0.0, "ks_mag": 10.0}


@pytest.fixture
def stub(monkeypatch, tmp_path):
    """Route NOISE_CACHE to tmp and count/record every worker batch."""
    monkeypatch.setattr(ins, "NOISE_CACHE", tmp_path)
    # backend_fingerprint probes the backend env; pin it for key stability
    monkeypatch.setattr(noise, "backend_fingerprint",
                        lambda: {"engine_version": "stub"})
    calls = []

    def fake_worker(job, progress=None):
        keys = [m["key"] for m in job["modes"]]
        calls.append(keys)
        out = {k: {"ngroup": 5, "payload_for": k} for k in keys}
        out["__provenance__"] = {"worker_version": job["worker_version"]}
        return out

    monkeypatch.setattr(noise, "_run_worker", fake_worker)
    return calls


def test_only_missing_modes_go_to_the_worker(stub):
    keys3 = ["nirspec_prism", "nirspec_g395h", "miri_lrs"]
    out = noise.run_modes(STAR, keys3)
    assert stub == [keys3]                       # one batch, three modes
    assert all(out[k]["payload_for"] == k for k in keys3)
    assert out["__provenance__"]["worker_version"] == noise.WORKER_VERSION

    # second run: everything cached, worker never invoked
    out2 = noise.run_modes(STAR, keys3)
    assert stub == [keys3]
    assert out2[keys3[0]] == out[keys3[0]]

    # adding one mode costs exactly that mode
    out3 = noise.run_modes(STAR, keys3 + ["niriss_soss"])
    assert stub == [keys3, ["niriss_soss"]]
    assert out3["niriss_soss"]["payload_for"] == "niriss_soss"
    assert out3["nirspec_prism"]["payload_for"] == "nirspec_prism"


def test_missing_modes_reports_the_uncached_subset(stub):
    assert noise.missing_modes(STAR, ["nirspec_prism", "miri_lrs"]) == \
        ["nirspec_prism", "miri_lrs"]
    noise.run_modes(STAR, ["nirspec_prism"])
    assert noise.missing_modes(STAR, ["nirspec_prism", "miri_lrs"]) == \
        ["miri_lrs"]


def test_cache_key_isolates_star_saturation_and_mode(stub):
    """A different star, sat_limit, or mode never hits another's entry."""
    noise.run_modes(STAR, ["nirspec_prism"])
    star2 = dict(STAR, ks_mag=8.0)
    assert noise.missing_modes(star2, ["nirspec_prism"]) == ["nirspec_prism"]
    assert noise.missing_modes(STAR, ["nirspec_prism"],
                               sat_limit=0.7) == ["nirspec_prism"]
    assert noise.missing_modes(STAR, ["nirspec_g395h"]) == ["nirspec_g395h"]


def test_force_recomputes_everything(stub):
    keys = ["nirspec_prism", "miri_lrs"]
    noise.run_modes(STAR, keys)
    noise.run_modes(STAR, keys, force=True)
    assert stub == [keys, keys]


def test_per_mode_files_carry_their_own_provenance(stub, tmp_path):
    noise.run_modes(STAR, ["nirspec_prism"])
    files = list(tmp_path.glob("*.json"))
    payloads = [json.loads(f.read_text()) for f in files
                if "nirspec_prism" in f.read_text()]
    assert payloads and all("__provenance__" in p for p in payloads)
