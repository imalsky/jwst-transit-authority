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
import logging
import os
import smtplib
import tempfile
import threading
import time
from email.message import EmailMessage
from pathlib import Path

from jwst_tool import instruments as _ins

#: concurrent heavy subprocesses per instance (one Run holds ONE slot,
#: covering its forward model and its Pandeia ETC batch, from its first
#: heavy launch to the end of the run; a fully cached Run takes none).
#: Sized for an 8 vCPU / 32 GB instance: one solve measures 1.75 cores
#: averaged over the run and 6.34 GiB peak, over a 4.6 GiB idle floor, so
#: three fit with memory to spare and four sit at the ceiling.
MAX_CONCURRENT = 3
#: container-local on purpose: a Space replica must own its own slots; on
#: the shared bucket every replica would contend for the same files.
SLOT_DIR = Path(tempfile.gettempdir()) / "jwst_tool_run_slots"


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
                    "this filesystem does not support flock. Set TMPDIR "
                    "to a filesystem with working advisory locks.") from exc
            continue
        fh.truncate(0)
        fh.write(json.dumps({"pid": os.getpid(), "tag": str(tag),
                             "t0": time.time()}))
        fh.flush()
        return Slot(fh, i)
    return None


def oldest_start():
    """Start time (epoch s) of the longest-running held slot, or None.

    Read without the locks, so best effort: a slot file mid-rewrite or
    malformed is skipped, and one read during turnover can be stale. For the
    refusal message only, never for a promised wait.
    """
    t0s = []
    for i in range(MAX_CONCURRENT):
        try:
            meta = json.loads((SLOT_DIR / f"slot{i}.lock").read_text())
            t0s.append(float(meta["t0"]))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return min(t0s) if t0s else None


def notify_refused(context: str) -> None:
    """Mail the maintainer that a visitor was refused (every slot busy, or an
    illegal parameter set), at most once an hour per instance.

    Off unless ALERT_SMTP_USER, ALERT_SMTP_PASS and ALERT_TO are all set, so
    it is inert locally and in the tests. The send runs in a daemon thread --
    the page never waits on SMTP -- and nothing here raises: a failed alert
    must never turn into a second refusal.
    """
    log = logging.getLogger(__name__)
    user = os.environ.get("ALERT_SMTP_USER")
    # Gmail shows an app password in four spaced groups; both forms work
    password = (os.environ.get("ALERT_SMTP_PASS") or "").replace(" ", "")
    to = os.environ.get("ALERT_TO")
    if not (user and password and to):
        return
    stamp = Path(_ins.OUTPUT_DIR) / "alert_last_sent"
    try:
        # stamped BEFORE the thread starts: two refusals a millisecond apart
        # must not both pass the window
        if stamp.exists() and time.time() - stamp.stat().st_mtime < 3600.0:
            return
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.touch()
    except OSError as exc:
        log.warning("refusal alert not sent: %r", exc)
        return

    def _send() -> None:
        try:
            msg = EmailMessage()
            msg["Subject"] = "[jwst-transit-authority] user refused"
            msg["From"], msg["To"] = user, to
            msg.set_content("%s\n%s\n" % (
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), context))
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
                srv.login(user, password)
                srv.send_message(msg)
        except Exception as exc:       # broad: never break a page render
            log.warning("refusal alert failed: %r", exc)

    threading.Thread(target=_send, name="refusal-alert", daemon=True).start()
