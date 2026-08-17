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
import sys
import time
from pathlib import Path

from jwst_tool import instruments as _ins

#: concurrent heavy subprocesses per instance (forward model, Pandeia ETC
#: batch, adjoint diagnostics each hold ONE slot for their full duration).
#: Sized to cpu-upgrade (8 vCPU / 32 GB): one default solve measures ~1.7
#: cores and ~6.3 GB peak, so four fit with headroom and eight do not.
MAX_CONCURRENT = 4
SLOT_DIR = Path(_ins.OUTPUT_DIR) / "run_slots"

#: every declined launch appends one JSON line here (the capacity-demand
#: record); alerts are optional on top and throttled to one per interval.
REFUSAL_LOG_NAME = "refusals.log"
ALERT_STAMP_NAME = "alert.stamp"
ALERT_MIN_INTERVAL_S = 6 * 3600.0


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
    _record_refusal(tag)
    return None


def _record_refusal(tag: str) -> None:
    """Append the declined launch to the refusal log, then maybe alert.

    Never raises: monitoring must not break the decline path."""
    try:
        with open(SLOT_DIR / REFUSAL_LOG_NAME, "a") as fh:
            fh.write(json.dumps({"t": time.time(), "tag": str(tag)}) + "\n")
        _maybe_alert()
    except Exception as exc:
        print(f"jwst_tool.runlimit: refusal record failed: {exc!r}",
              file=sys.stderr)


def refusals_last_24h() -> int:
    p = SLOT_DIR / REFUSAL_LOG_NAME
    if not p.is_file():
        return 0
    cutoff = time.time() - 86400.0
    n = 0
    for line in p.read_text().splitlines():
        try:
            if json.loads(line)["t"] >= cutoff:
                n += 1
        except (ValueError, KeyError, TypeError):
            continue
    return n


def _maybe_alert() -> None:
    """Send at most one capacity alert per ``ALERT_MIN_INTERVAL_S``.

    Channels come from env vars (Space secrets): an ntfy.sh push when
    ``JWST_TOOL_ALERT_NTFY_TOPIC`` is set, and a Gmail message when
    ``JWST_TOOL_ALERT_EMAIL`` + ``JWST_TOOL_ALERT_SMTP_PASS`` (an app
    password) are set. The stamp file is flock'd so concurrent refusals
    cannot double-send, and it is only written on a successful send."""
    env = os.environ
    if not (env.get("JWST_TOOL_ALERT_NTFY_TOPIC")
            or (env.get("JWST_TOOL_ALERT_EMAIL")
                and env.get("JWST_TOOL_ALERT_SMTP_PASS"))):
        return
    fh = open(SLOT_DIR / ALERT_STAMP_NAME, "a+")
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return
        st = os.fstat(fh.fileno())
        if st.st_size and time.time() - st.st_mtime < ALERT_MIN_INTERVAL_S:
            return
        if _send_alert():
            fh.truncate(0)
            fh.write(str(time.time()))
            fh.flush()
    finally:
        fh.close()


_ALERT_TITLE = "jwst-tool Space at capacity"


def _send_alert() -> bool:
    """True if at least one configured channel delivered."""
    body = (f"jwst-tool: all {MAX_CONCURRENT} run slots are busy and a "
            f"visitor's run was declined ({refusals_last_24h()} refusal(s) "
            f"in the last 24 h). Log: run_slots/{REFUSAL_LOG_NAME}")
    ok = False
    topic = os.environ.get("JWST_TOOL_ALERT_NTFY_TOPIC")
    if topic:
        ok |= _post_ntfy(topic, body)
    email = os.environ.get("JWST_TOOL_ALERT_EMAIL")
    smtp_pass = os.environ.get("JWST_TOOL_ALERT_SMTP_PASS")
    if email and smtp_pass:
        ok |= _send_email(email, smtp_pass, body)
    return ok


def _post_ntfy(topic: str, body: str) -> bool:
    import urllib.request

    req = urllib.request.Request(f"https://ntfy.sh/{topic}",
                                 data=body.encode(), method="POST")
    req.add_header("Title", _ALERT_TITLE)
    try:
        urllib.request.urlopen(req, timeout=5).close()
        return True
    except Exception as exc:
        print(f"jwst_tool.runlimit: ntfy alert failed: {exc!r}",
              file=sys.stderr)
        return False


def _send_email(email: str, smtp_pass: str, body: str) -> bool:
    """Self-addressed Gmail via an app password (both ends = ``email``)."""
    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["Subject"] = _ALERT_TITLE
    msg["From"] = email
    msg["To"] = email
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as s:
            s.login(email, smtp_pass)
            s.send_message(msg)
        return True
    except Exception as exc:
        print(f"jwst_tool.runlimit: email alert failed: {exc!r}",
              file=sys.stderr)
        return False


def busy_count() -> int:
    """How many slots are currently held (probe; racy by nature, display
    only)."""
    n = 0
    for i in range(MAX_CONCURRENT):
        p = SLOT_DIR / f"slot{i}.lock"
        if not p.exists():
            continue
        fh = open(p, "a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            n += 1
        finally:
            fh.close()
    return n
