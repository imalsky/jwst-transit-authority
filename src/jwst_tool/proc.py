"""Subprocess lifetime guard for the GUI's long-running workers.

The forward model, the Pandeia ETC and the adjoint diagnostics all run as
child processes while the Streamlit script streams their stdout. Streamlit
cancels a script run by raising a ScriptControlException inside the next
``st.*`` call -- and that is a **BaseException**, so it passes through
``except Exception`` untouched and unwinds straight out of the run block.

Without a guard the child survives that unwind: it keeps a CPU busy on
shared hardware after the caller's `runlimit` slot has already been
released, so the limiter believes a slot is free while a solver is still
using it. `terminating` closes that hole -- the child never outlives the
script run that started it.

Lives outside app.py so it is importable and unit-testable without running
Streamlit (same reason as share_config).
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
