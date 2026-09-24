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
import re
import subprocess
import sys
from pathlib import Path

TERM_GRACE_S = 10.0
# The line that closes a Python traceback: "<class>: <message>". The class may
# be a builtin (RuntimeError), one of the tool's own printed bare because the
# worker runs as __main__ (NotCertified), or module-qualified
# (jwst_tool.archive.SnapshotError) -- never gate on an Error/Exception suffix.
# Solver log lines match the same shape ("integration:  simpson"), so only the
# lines after the last traceback header are searched.
_EXC_LINE = re.compile(r"^[A-Za-z_][\w.]*: ")


def exception_sentence(lines, default: str) -> str:
    """The user-facing sentence of a failed worker: the message of the
    exception that closes the last traceback in its merged stdout/stderr, else
    ``default`` (a worker killed without a traceback)."""
    starts = [i for i, ln in enumerate(lines)
              if ln.startswith("Traceback (most recent call last):")]
    for ln in lines[starts[-1] + 1:] if starts else ():
        if _EXC_LINE.match(ln):
            return ln.split(": ", 1)[1]
    return default


def worker_prologue(output_dir) -> None:
    """Prologue for the worker entry points (forward.main, adjoint_diag.main).

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
