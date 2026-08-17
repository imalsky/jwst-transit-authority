"""The v18.1 public-instance run limiter: slot semantics + lifecycle."""
import pytest

from jwst_tool import runlimit


@pytest.fixture(autouse=True)
def _tmp_slots(tmp_path, monkeypatch):
    monkeypatch.setattr(runlimit, "SLOT_DIR", tmp_path / "run_slots")


def test_slots_cap_and_release():
    n = runlimit.MAX_CONCURRENT
    held = [runlimit.acquire(f"a{i}") for i in range(n)]
    assert all(s is not None for s in held)
    assert runlimit.acquire("over") is None       # cap reached
    assert runlimit.busy_count() == n
    held[0].release()
    reused = runlimit.acquire("d")                # freed slot reusable
    assert reused is not None
    for s in held[1:] + [reused]:
        s.release()
    assert runlimit.busy_count() == 0


def test_slot_files_persist_and_carry_metadata():
    import json
    s = runlimit.acquire("tagged")
    p = runlimit.SLOT_DIR / f"slot{s.index}.lock"
    meta = json.loads(p.read_text())
    assert meta["tag"] == "tagged" and meta["pid"] > 0
    s.release()
    # lifecycle contract: the slot FILE persists after release (unlinking a
    # flock'd path is the two-inode double-lock race)
    assert p.is_file()
    s.release()                                    # idempotent


def test_refusal_logged_and_alert_throttled(monkeypatch):
    sent = []
    monkeypatch.setattr(runlimit, "_send_alert", lambda: (
        sent.append("t-test"), True)[1])
    held = [runlimit.acquire(f"h{i}") for i in range(runlimit.MAX_CONCURRENT)]
    try:
        # alerts disabled without the env var, but the refusal still logs
        assert runlimit.acquire("r0") is None
        assert runlimit.refusals_last_24h() == 1 and sent == []
        # enabled: first refusal alerts, the next is inside the throttle
        monkeypatch.setenv("JWST_TOOL_ALERT_NTFY_TOPIC", "t-test")
        assert runlimit.acquire("r1") is None
        assert runlimit.acquire("r2") is None
        assert runlimit.refusals_last_24h() == 3 and sent == ["t-test"]
    finally:
        for s in held:
            s.release()
