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


@pytest.mark.parametrize("exc", [_ScriptCancelled, RuntimeError])
def test_the_child_is_killed_however_the_block_leaves(exc):
    """A cancelled script run (BaseException) and an ordinary exception must
    both take the worker down with them."""
    p = _sleeper()
    with pytest.raises(exc):
        with proc_mod.terminating(p):
            raise exc("boom")
    assert _wait_gone(p), "the worker outlived the failed script run"


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


@pytest.mark.parametrize("last", [
    "RuntimeError: chemistry did NOT converge (stage: detail)",
    "NotCertified: the sensitivity did NOT settle (stage: detail)",
    "jwst_tool.archive.SnapshotError: snapshot missing: detail",
])
def test_exception_sentence_is_the_last_traceback_line_whatever_the_class(last):
    """The GUI shows the worker's own sentence for every exception class the
    traceback can end with; a log with no traceback (a killed worker) falls
    back, although solver lines have the same "<word>: <text>" shape."""
    log = ["integration:  simpson", "Traceback (most recent call last):",
           '  File "forward.py", line 313, in check_converged',
           "    raise NotCertified(", last]
    assert proc_mod.exception_sentence(log, "fallback") == last.split(": ", 1)[1]
    assert proc_mod.exception_sentence(log[:1], "fallback") == "fallback"
