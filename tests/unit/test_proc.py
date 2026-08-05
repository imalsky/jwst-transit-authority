"""A GUI worker must not outlive the script run that started it.

Streamlit cancels a run with ScriptControlException, a BaseException; before
`proc.terminating` that left a solver running after `runlimit` released its
slot. The guard is pinned against Exception and BaseException both.
"""
from __future__ import annotations

import subprocess
import sys
import time

import pytest

from jwst_tool import proc as proc_mod


def _sleeper(seconds: int = 60) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-u", "-c", f"import time; time.sleep({seconds})"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _wait_gone(p: subprocess.Popen, timeout: float = 15.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if p.poll() is not None:
            return True
        time.sleep(0.05)
    return False


class _ScriptCancelled(BaseException):
    """Stand-in for streamlit's ScriptControlException (also a BaseException)."""


def test_child_is_killed_when_the_block_is_cancelled():
    p = _sleeper()
    with pytest.raises(_ScriptCancelled):
        with proc_mod.terminating(p):
            raise _ScriptCancelled()
    assert _wait_gone(p), "the worker outlived the cancelled script run"


def test_child_is_killed_on_an_ordinary_exception():
    p = _sleeper()
    with pytest.raises(RuntimeError):
        with proc_mod.terminating(p):
            raise RuntimeError("boom")
    assert _wait_gone(p)


def test_a_child_that_finished_is_left_alone_with_its_returncode():
    p = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with proc_mod.terminating(p):
        p.wait()
    assert p.returncode == 3, "a normal exit must survive the guard intact"


def test_sigterm_refuser_is_killed_within_the_grace_period():
    p = subprocess.Popen(
        [sys.executable, "-u", "-c",
         "import signal, time\n"
         "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
         "print('ready', flush=True)\n"
         "time.sleep(60)\n"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert p.stdout.readline().strip() == b"ready"
    with pytest.raises(_ScriptCancelled):
        with proc_mod.terminating(p, grace_s=1.0):
            raise _ScriptCancelled()
    assert _wait_gone(p), "SIGKILL fallback did not fire"


def test_streamlit_control_exception_is_a_bare_baseexception():
    """The reason `except Exception` was never enough. If streamlit ever
    re-parents these under Exception, this test says so."""
    st_exc = pytest.importorskip(
        "streamlit.runtime.scriptrunner_utils.exceptions")
    assert not issubclass(st_exc.ScriptControlException, Exception)
    assert issubclass(st_exc.ScriptControlException, BaseException)
