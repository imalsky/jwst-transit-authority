"""Subprocess lifetime guard for the GUI's long-running workers.

Streamlit cancels a script run by raising ScriptControlException inside the
next ``st.*`` call -- a **BaseException**, so it passes through ``except
Exception`` and unwinds out of the run block. Without a guard the child
survives that unwind and keeps a CPU busy after its runlimit slot is
released; ``terminating`` guarantees the child never outlives the script run
that started it.

Lives outside app.py so it is importable and unit-testable without Streamlit.
"""
from __future__ import annotations

import contextlib
import subprocess

TERM_GRACE_S = 10.0


@contextlib.contextmanager
def terminating(proc: subprocess.Popen, grace_s: float = TERM_GRACE_S):
    """Yield ``proc``, guaranteeing it is dead and reaped on the way out.

    A child that exited normally is left alone. One still running when the
    block is left -- for any reason, including a cancelled Streamlit script
    run -- gets SIGTERM, then SIGKILL if it has not gone within ``grace_s``.
    """
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
