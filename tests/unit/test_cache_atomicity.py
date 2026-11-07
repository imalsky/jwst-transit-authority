"""Cache concurrency invariants: writes never expose a partial file at the
final path, and a corrupt cache entry is quarantined as a miss instead of
poisoning its key for every later reader.
"""

from __future__ import annotations

import numpy as np
import pytest

from jwst_tool import instruments as ins, model_cache, noise, tp_tables


def test_atomic_write_never_leaves_a_partial_final_file(tmp_path):
    target = tmp_path / "entry.npz"

    def torn(fh):
        fh.write(b"partial")
        raise RuntimeError("killed mid-write")

    with pytest.raises(RuntimeError):
        ins.atomic_write(target, torn)
    assert not target.exists()

    ins.atomic_write(target, lambda fh: np.savez_compressed(fh, x=np.arange(3)))
    with np.load(target) as z:
        assert list(z["x"]) == [0, 1, 2]
    assert [p.name for p in tmp_path.iterdir()] == ["entry.npz"]  # no tmp left


@pytest.mark.parametrize("garbage", [b"", b"not a zip archive"])
def test_corrupt_npz_is_quarantined_and_reads_as_a_miss(tmp_path, garbage):
    p = tmp_path / "abc123.npz"
    p.write_bytes(garbage)
    assert model_cache.load_cached_npz(p) is None
    assert not p.exists()
    assert list(tmp_path.glob("abc123.npz.corrupt-*"))
    assert model_cache.load_cached_npz(p) is None  # miss, stays recoverable


def test_a_staged_upload_reaches_the_disk_and_a_failed_write_leaves_nothing(tmp_path, monkeypatch):
    """The archive's exists() guard would make a torn copy permanent, so staging flushes, or cleans up if it cannot."""

    def disk_full(_fd):
        raise OSError("disk full")

    monkeypatch.setattr(tp_tables, "uploads_dir", lambda: tmp_path)
    monkeypatch.setattr(ins.os, "fsync", disk_full)
    with pytest.raises(OSError, match="disk full"):
        tp_tables.stage_upload(b"table bytes")
    assert not list(tmp_path.iterdir())


def test_corrupt_noise_json_is_quarantined_and_reported_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(ins, "NOISE_CACHE", tmp_path)
    monkeypatch.setattr(noise, "backend_fingerprint", lambda: {"engine_version": "stub"})
    star = {"teff": 5400.0, "log_g": 4.45, "metallicity": 0.0, "ks_mag": 10.0}
    key = noise._mode_key(star, "nirspec_prism", 0.80)
    (tmp_path / f"{key}.json").write_text("{truncated")
    assert noise.missing_modes(star, ["nirspec_prism"]) == ["nirspec_prism"]
    assert not (tmp_path / f"{key}.json").exists()
    assert list(tmp_path.glob(f"{key}.json.corrupt-*"))


def test_a_cold_solve_persists_its_column_under_the_cache_a_run_is_pointed_at(tmp_path, monkeypatch):
    """The cache root has one binding: a run pointed elsewhere creates that directory, and only that one."""
    from types import SimpleNamespace

    from jwst_tool import forward

    target = tmp_path / "fresh" / "model_cache"
    monkeypatch.setattr(ins, "MODEL_CACHE", target)
    monkeypatch.setattr(forward, "ColumnSpectrum", lambda *_: "column")
    monkeypatch.setattr(forward, "check_elements", lambda *_: 0.0)
    cert = ("baseline solve", 10, 0.05, 1e-9, 2, 0.01, "H2O@z0")
    chem = SimpleNamespace(p_bar=np.array([1.0, 0.1]))
    assembly = SimpleNamespace(theta=np.zeros(4), theta_names=["lnZ"], build_chem=lambda **_: chem)
    monkeypatch.setattr(forward, "certified_solve", lambda *_a, **_k: (np.ones((2, 2)), chem, cert, False))
    out = target / "chem-x.npz"
    col = forward._solve_cold(chem, assembly, np.zeros(4), ["H2", "He"], out, None, 0.0, lambda *_: None)
    assert out.is_file(), "the chem column was not written where the run was pointed"
    assert col.solved_here
    assert model_cache.cache_path({}).parent == target


def test_a_cold_solve_that_fails_its_element_gate_writes_no_cache(tmp_path, monkeypatch):
    """The chem cache skips the solve on the next run, so a column that fails the gate must never reach it."""
    from types import SimpleNamespace

    from jwst_tool import forward

    def budget_off(*_a):
        raise RuntimeError("element budget off")

    monkeypatch.setattr(forward, "ColumnSpectrum", lambda *_: "column")
    monkeypatch.setattr(forward, "check_elements", budget_off)
    cert = ("baseline solve", 10, 0.05, 1e-9, 2, 0.01, "H2O@z0")
    chem = SimpleNamespace(p_bar=np.array([1.0, 0.1]))
    monkeypatch.setattr(forward, "certified_solve", lambda *_a, **_k: (np.ones((2, 2)), chem, cert, False))
    assembly = SimpleNamespace(theta=np.zeros(4), theta_names=["lnZ"], build_chem=lambda **_: chem)
    out = tmp_path / "chem-x.npz"
    with pytest.raises(RuntimeError, match="element budget"):
        forward._solve_cold(chem, assembly, np.zeros(4), ["H2", "He"], out, None, 0.0, lambda *_: None)
    assert not out.exists()
    assert not list(tmp_path.iterdir())
