"""Cross-process limiter for heavy runs (public-Space protection).

The live Space is public, so any visitor can launch multi-minute
subprocesses. This caps how many run at once per instance with OS-level
advisory locks; when every slot is busy the GUI declines the launch instead
of queueing.

Lifecycle contract (same as the climate cache lock): slot files are flock'd
but NEVER unlinked -- a slot releases when its holder closes the fd or dies,
and unlinking a path another process may still hold flock'd creates two
"exclusive" locks on different inodes. The pid/tag/start-time in a slot file
is observability metadata only.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path

from jwst_tool import instruments as _ins

#: concurrent heavy subprocesses per instance (forward model, Pandeia ETC
#: batch, adjoint diagnostics each hold ONE slot for their full duration).
#: Sized for an 8 vCPU / 32 GB instance: one solve takes ~1.7 cores and
#: ~6.3 GB peak, so four fit with headroom and eight do not.
MAX_CONCURRENT = 4
SLOT_DIR = Path(_ins.OUTPUT_DIR) / "run_slots"


class Slot:
    """A held run slot; ``release()`` exactly once when the run finishes
    (the OS also releases it if the holding process dies)."""

    def __init__(self, fh, index: int):
        self._fh = fh
        self.index = index

    def release(self) -> None:
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except (OSError, ValueError):
            pass
        try:
            self._fh.close()
        except OSError:
            pass


def acquire(tag: str = "run"):
    """A :class:`Slot`, or ``None`` when all slots are busy. Never blocks."""
    SLOT_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(MAX_CONCURRENT):
        fh = open(SLOT_DIR / f"slot{i}.lock", "a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            fh.close()
            import errno as _errno
            if exc.errno not in (_errno.EAGAIN, _errno.EACCES,
                                 _errno.EWOULDBLOCK):
                raise RuntimeError(
                    f"run-slot lock failed on {SLOT_DIR} with {exc!r}: "
                    "this filesystem does not support flock. Point "
                    "JWST_TOOL_OUTPUT_DIR at a filesystem with working "
                    "advisory locks.") from exc
            continue
        fh.truncate(0)
        fh.write(json.dumps({"pid": os.getpid(), "tag": str(tag),
                             "t0": time.time()}))
        fh.flush()
        return Slot(fh, i)
    return None
