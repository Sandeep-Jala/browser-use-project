"""`os.replace` that survives a reader holding the destination open.

POSIX `rename(2)` over an open file is atomic and always succeeds: the reader keeps its handle on
the old inode and sees that file whole. Two places in this framework are built on exactly that
guarantee and say so in their docstrings — the operator control channel (`control.write_control`,
read every 0.4 s by the running task) and `progress.json` (written by `hybrid`, polled by the UI's
supervisor).

Windows has no such guarantee. `MoveFileEx` fails with `ERROR_SHARING_VIOLATION` /
`ERROR_ACCESS_DENIED` — surfacing as `PermissionError` — whenever a handle to the destination is
open, so the two writes above fail *intermittently*, decided by whether the poll landed inside
the write. A pause that works four times out of five is worse than one that never works, because
nobody believes the bug report.

The retry is the whole fix: the reader's handle is open for microseconds, so losing the race twice
is already unlikely and losing it three times means something else holds the file. On the last
attempt the error is allowed to propagate — a silent failure here would mean an operator's `stop`
vanishing, which is the one outcome worse than an exception.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

_ATTEMPTS = 3
_BACKOFF_S = 0.05


def replace_with_retry(src: Path, dst: Path, *, attempts: int = _ATTEMPTS,
                       backoff_s: float = _BACKOFF_S, sleep=time.sleep) -> None:
    """`os.replace(src, dst)`, retried while a concurrent reader holds `dst` open.

    `PermissionError` only: every other OSError (a missing source, a cross-device move, a bad
    path) is a real fault that retrying cannot mend, and swallowing it for 150 ms would just
    delay the report.
    """
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            sleep(backoff_s)
