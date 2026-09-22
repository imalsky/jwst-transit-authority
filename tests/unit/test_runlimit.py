"""The public-instance run limiter: slot semantics + lifecycle."""
import smtplib
import threading

import pytest

from jwst_tool import instruments as ins
from jwst_tool import runlimit


@pytest.fixture(autouse=True)
def _tmp_slots(tmp_path, monkeypatch):
    monkeypatch.setattr(runlimit, "SLOT_DIR", tmp_path / "run_slots")


def test_slots_cap_and_release():
    n = runlimit.MAX_CONCURRENT
    held = [runlimit.acquire(f"a{i}") for i in range(n)]
    assert all(s is not None for s in held)
    assert runlimit.acquire("over") is None       # cap reached
    held[0].release()
    reused = runlimit.acquire("d")                # freed slot reusable
    assert reused is not None
    for s in held[1:] + [reused]:
        s.release()


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


def test_oldest_start_is_best_effort():
    """The refusal message's elapsed time: the oldest readable t0 among the
    slot files; an empty (mid-rewrite) or malformed file is skipped."""
    import json
    assert runlimit.oldest_start() is None         # no slot file yet
    held = [runlimit.acquire(f"a{i}") for i in range(runlimit.MAX_CONCURRENT)]
    paths = [runlimit.SLOT_DIR / f"slot{s.index}.lock" for s in held]
    t0 = [json.loads(p.read_text())["t0"] for p in paths]
    assert runlimit.oldest_start() == min(t0)
    bad = ["", '{"t0": "soon"}', "[1]", "{}"]
    for i, p in enumerate(paths[:-1]):
        p.write_text(bad[i % len(bad)])
    assert runlimit.oldest_start() == t0[-1]
    paths[-1].write_text("")
    assert runlimit.oldest_start() is None
    for s in held:
        s.release()


def test_refusal_alert_is_opt_in_and_hourly(tmp_path, monkeypatch):
    """One mail per hour when the three secrets are set, none without."""
    bodies, logins = [], []

    class _FakeSMTP:
        def __init__(self, host, port):
            assert (host, port) == ("smtp.gmail.com", 465)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def login(self, user, password):
            logins.append(password)

        def send_message(self, msg):
            bodies.append(msg.get_content())

    def _join():
        for t in threading.enumerate():
            if t.name == "refusal-alert":
                t.join(30)

    monkeypatch.setattr(smtplib, "SMTP_SSL", _FakeSMTP)
    monkeypatch.setattr(ins, "OUTPUT_DIR", tmp_path)
    for k, v in (("ALERT_SMTP_USER", "u"), ("ALERT_SMTP_PASS", "ab cd"),
                 ("ALERT_TO", "m@example.com")):
        monkeypatch.setenv(k, v)
    runlimit.notify_refused("ctx-first")
    runlimit.notify_refused("ctx-second")          # inside the hour
    _join()
    assert len(bodies) == 1 and "ctx-first" in bodies[0]
    assert logins == ["abcd"]                      # Gmail's spaced form works

    monkeypatch.delenv("ALERT_TO")                 # feature off
    (tmp_path / "alert_last_sent").unlink()
    runlimit.notify_refused("ctx-third")
    _join()
    assert len(bodies) == 1
