"""Subprocess lifetime guard for the GUI's long-running workers.

Streamlit cancels a script run by raising ScriptControlException inside the
next ``st.*`` call -- a **BaseException**, so it passes through ``except
Exception`` and unwinds out of the run block. Without a guard the child
survives that unwind and keeps a CPU busy after its runlimit slot is
released; ``terminating`` guarantees the child never outlives the script run
that started it.

Lives outside the GUI so it is importable and unit-testable without Streamlit.
It also owns the prologue the worker entry points share.
"""

from __future__ import annotations

import contextlib
import re
import subprocess
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

TERM_GRACE_S = 10.0
# The line that closes a Python traceback: "<class>: <message>". The class may
# be a builtin (RuntimeError) or module-qualified
# (jwst_tool.certify.NotCertified) -- never gate on an Error/Exception suffix.
# Solver log lines match the same shape ("integration:  simpson"), so only the
# lines after the last traceback header are searched.
_EXC_LINE = re.compile(r"^[A-Za-z_][\w.]*: ")


def exception_sentence(lines: list, default: str) -> str:
    """Return the user-facing sentence of a failed worker.

    The message of the exception that closes the last traceback in its merged stdout/stderr, else ``default`` (a
    worker killed without a traceback).
    """
    starts = [i for i, ln in enumerate(lines) if ln.startswith("Traceback (most recent call last):")]
    for ln in lines[starts[-1] + 1 :] if starts else ():
        if _EXC_LINE.match(ln):
            return ln.split(": ", 1)[1]
    return default


def worker_prologue() -> None:
    """Prologue for the worker entry points (forward.main, adjoint_diag.main).

    Line-buffers stdout: the GUI pipes the child, which makes Python
    BLOCK-buffer library prints, so progress lines would sit invisible in the
    buffer while the GUI shows nothing.
    """
    sys.stdout.reconfigure(line_buffering=True)


@contextlib.contextmanager
def terminating(proc: subprocess.Popen, grace_s: float = TERM_GRACE_S) -> Iterator[Any]:
    """Yield ``proc``, guaranteeing it is dead and reaped on the way out.

    A child that exited normally is left alone, and its stdout pipe is closed either way. One still running when the
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
        if proc.stdout is not None:
            proc.stdout.close()
