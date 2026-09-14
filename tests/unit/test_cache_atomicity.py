"""Cache concurrency invariants: writes never expose a partial file at the
final path, and a corrupt cache entry is quarantined as a miss instead of
poisoning its key for every later reader."""
from __future__ import annotations

import numpy as np
import pytest

from jwst_tool import forward
from jwst_tool import instruments as ins
from jwst_tool import noise


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
    assert forward._load_cached_npz(p) is None
    assert not p.exists()
    assert list(tmp_path.glob("abc123.npz.corrupt-*"))
    assert forward._load_cached_npz(p) is None      # miss, stays recoverable


def test_corrupt_noise_json_is_quarantined_and_reported_missing(
        tmp_path, monkeypatch):
    monkeypatch.setattr(ins, "NOISE_CACHE", tmp_path)
    monkeypatch.setattr(noise, "backend_fingerprint",
                        lambda: {"engine_version": "stub"})
    star = {"teff": 5400.0, "log_g": 4.45, "metallicity": 0.0, "ks_mag": 10.0}
    key = noise._mode_key(star, "nirspec_prism", 0.80)
    (tmp_path / f"{key}.json").write_text("{truncated")
    assert noise.missing_modes(star, ["nirspec_prism"]) == ["nirspec_prism"]
    assert not (tmp_path / f"{key}.json").exists()
    assert list(tmp_path.glob(f"{key}.json.corrupt-*"))
