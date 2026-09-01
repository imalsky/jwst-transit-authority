"""Subprocess lifetime guard for the GUI's long-running workers.

Streamlit cancels a script run by raising ScriptControlException inside the
next ``st.*`` call -- a **BaseException**, so it passes through ``except
Exception`` and unwinds out of the run block. Without a guard the child
survives that unwind and keeps a CPU busy after its runlimit slot is
released; ``terminating`` guarantees the child never outlives the script run
that started it.

Lives outside app.py so it is importable and unit-testable without Streamlit.
It also owns the prologue the worker entry points share.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from pathlib import Path

TERM_GRACE_S = 10.0


def worker_prologue(output_dir) -> None:
    """Prologue for the GUI's subprocess entry points (forward, adjoint_diag).

    Line-buffers stdout: the GUI pipes the child, which makes Python
    BLOCK-buffer library prints, so progress lines would sit invisible in the
    buffer while the GUI shows nothing. Then moves the process into
    ``<output_dir>/cwd``: vulcan_jax's legacy IO creates RELATIVE output/ and
    plot/ directories in the process CWD (legacy_io.py), junk wherever the app
    was launched from. Library callers of run_model / run_adjoint are
    unaffected -- only the subprocess entry points change directory.
    """
    sys.stdout.reconfigure(line_buffering=True)
    cwd = Path(output_dir) / "cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    os.chdir(cwd)


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
