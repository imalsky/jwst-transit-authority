"""The GUI's run stage without a page: the heavy slot and per-mode isolation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("streamlit")

from jwst_tool import detect, gui_run


class _Slot:
    def __init__(self):
        self.released = 0

    def release(self):
        self.released += 1


def test_the_heavy_slot_is_taken_once_and_released_however_the_run_ends(monkeypatch):
    held = _Slot()
    taken = []

    def acquire(tag):
        taken.append(tag)
        return held

    monkeypatch.setattr(gui_run.runlimit, "acquire", acquire)

    def two_launches(slot, *_a):
        assert gui_run._take_slot(slot, None, None, None)
        assert gui_run._take_slot(slot, None, None, None)
        return {"ok": True}

    monkeypatch.setattr(gui_run, "_run_and_evaluate", two_launches)
    assert gui_run.compute(None, None, None, None) == {"ok": True}
    assert len(taken) == 1 and held.released == 1

    def fails(slot, *_a):
        gui_run._take_slot(slot, None, None, None)
        raise RuntimeError("worker died")

    monkeypatch.setattr(gui_run, "_run_and_evaluate", fails)
    with pytest.raises(RuntimeError, match="worker died"):
        gui_run.compute(None, None, None, None)
    assert held.released == 2


def test_a_fully_cached_run_takes_no_slot(monkeypatch):
    def refuse(_tag):
        raise AssertionError("a cached run must start nothing")

    monkeypatch.setattr(gui_run.runlimit, "acquire", refuse)
    monkeypatch.setattr(gui_run, "_run_and_evaluate", lambda *_a: {"cached": True})
    assert gui_run.compute(None, None, None, None) == {"cached": True}


def test_one_bad_mode_is_reported_and_the_others_still_evaluate(monkeypatch):
    def evaluate(key, *_a, **_k):
        if key == "bad":
            raise ValueError("no overlap")
        if key == "narrow":
            raise detect.ModeUnusable("too narrow")
        return {"mode_key": key}

    monkeypatch.setattr(detect, "evaluate_mode", evaluate)
    keys = ["good", "bad", "narrow", "errored", "empty"]
    usable = {"wl": [1.0]}
    etc = {
        "good": usable,
        "bad": usable,
        "narrow": usable,
        "errored": {"error": "Pandeia refused"},
        "empty": {"wl": [], "reason": "no usable pixels"},
    }
    noise = SimpleNamespace(floors=dict.fromkeys(keys, 0.0), infl=dict.fromkeys(keys, 1.0))
    o = SimpleNamespace(mode_keys=keys, t_base=2.0, r_bin=100, n_transits=1, noise=noise)
    req = SimpleNamespace(goal_meta={"target_mol": None})
    out = gui_run._evaluate_modes({"m": 1}, etc, req, SimpleNamespace(t14=1.0), o)

    assert [r["mode_key"] for r in out["results"]] == ["good"]
    assert dict(out["failed"])["errored"] == "Pandeia refused"
    assert "no overlap" in dict(out["failed"])["bad"]
    assert dict(out["unusable"]) == {"narrow": "too narrow", "empty": "no usable pixels"}


def test_the_gui_asks_for_the_target_alone_on_detect_and_nothing_on_constrain(monkeypatch):
    """The score reads one row and the all-molecule block dominates a cold run, so the GUI asks for the target only;
    the all-molecule batch stays reachable through the API's wo_mols=None default.
    """
    from app_helpers import run_app

    seen = {}
    real = gui_run._validate

    def spy(params, a):
        seen["wo_mols"] = params["wo_mols"]
        return real(params, a)

    monkeypatch.setattr(gui_run, "_validate", spy)
    at = run_app()
    assert not at.exception, at.exception
    target = next(w.value for w in at.selectbox if (w.key or "").startswith("n0_mol_vulcan"))
    assert seen["wo_mols"] == [target]
    at.radio(key="n0_goal").set_value("constrain").run()
    assert not at.exception, at.exception
    assert seen["wo_mols"] == []


def test_a_refusal_is_recorded_once_per_message_and_a_failed_record_is_logged(monkeypatch, tmp_path, caplog):
    """The catch re-runs on every widget change while the settings stay illegal, so one message records once. The
    record is best effort: an unwritable log never disturbs the refusal the user sees, but it does say so.
    """
    mailed = []
    state = {}
    monkeypatch.setattr(gui_run, "st", SimpleNamespace(session_state=state))
    monkeypatch.setattr(gui_run.runlimit, "notify_refused", mailed.append)
    monkeypatch.setattr(gui_run.ins, "OUTPUT_DIR", tmp_path)
    gui_run._record_refusal("nz=5 outside the range")
    gui_run._record_refusal("nz=5 outside the range")
    gui_run._record_refusal("co_ratio=99 outside the range")
    log = (tmp_path / "refusals.log").read_text().splitlines()
    assert len(log) == 2 and "nz=5" in log[0] and "co_ratio=99" in log[1]
    assert len(mailed) == 2

    monkeypatch.setattr(gui_run.ins, "OUTPUT_DIR", tmp_path / "missing")
    gui_run._record_refusal("another refusal")
    assert len(mailed) == 3
    assert "refusals.log" in caplog.text
