"""Scanning `artifacts/` into run rows, and deleting runs.

No HTTP here — the app imports this, tests hit it directly. Imports stay light: this module must
never pull `browser_use` (so never `runner`/`hybrid`).

Recovered from the runs dashboard removed on 2026-07-20
(`stash@{2}^3:automation/dashboard/scan.py`), with one addition.

**The addition: `progress.json` as a second source.** `report.json` is written ONCE, from
`__main__.py` after a clean return — so a run stopped by the operator (exit 130) and a run that
crashed never get one, and the recovered code called both `INCOMPLETE`, which threw away what the
operator already knew. `progress.json` is written incrementally and atomically by
`hybrid._write_progress` all through a run, and carries the task, the plan and the segments, so a
report-less run can still say what it was doing and how far it got.

Its `status` field moves off `running` only on a clean finish (`finished`) or a KeyboardInterrupt
(`interrupted`), so `running` + no report means the process went away mid-run — EXCEPT for the run
that is still going, which looks identical from a directory. That one case cannot be settled by a
scan, so the caller names it: the supervisor knows its own run id and passes `active_run_id`.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from automation.pipeline.report import derive_status

log = logging.getLogger("framework.ui")

# Matches Runner._new_run_dir(): datetime.now().strftime("%Y%m%d_%H%M%S_%f")
RUN_ID_RE = re.compile(r"^\d{8}_\d{6}_\d{6}$")


@dataclass(frozen=True)
class RunSummary:
    """One row of the runs list — the few fields worth showing per run."""

    run_id: str
    path: Path
    started_at: datetime | None
    # "PASS" | "PASS*" | "DONE" | "FAIL"     (from report.json, via derive_status)
    # "RUNNING" | "STOPPED" | "CRASHED"      (from progress.json, when no report was written)
    # "INCOMPLETE"                           (neither file readable)
    status: str
    tone: str              # "success" | "warning" | "error" | "info" | "muted"
    task: str
    duration_seconds: float | None
    n_steps: int | None
    total_tokens: int | None
    total_cost: float | None
    subtasks_replayed: int | None
    subtasks_total: int | None
    size_bytes: int
    has_report_html: bool
    note: str | None       # why a run has no verdict of its own


def load_progress(run_dir: Path) -> dict[str, Any]:
    """`progress.json` for a run, or `{}`.

    Every failure reads as "no progress", deliberately: the file is rewritten atomically
    (temp + `os.replace`) so a torn read should be impossible, but a run that died before its
    first write has no file at all, and an unreadable one must not break the listing.
    """
    try:
        raw = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def scan_runs(artifacts_dir: Path, active_run_id: str | None = None) -> list[RunSummary]:
    """All runs under artifacts_dir, newest first (run ids sort chronologically).

    `active_run_id` names the run currently in flight, which is the one case a directory scan
    cannot tell from a crash.
    """
    if not artifacts_dir.is_dir():
        return []
    runs: list[RunSummary] = []
    for entry in artifacts_dir.iterdir():
        if not entry.is_dir() or not RUN_ID_RE.fullmatch(entry.name):
            continue
        try:
            runs.append(load_run_summary(entry, active=entry.name == active_run_id))
        except FileNotFoundError:
            continue  # deleted between iterdir and load (concurrent delete)
        except Exception:  # noqa: BLE001 - one broken run dir must not take down the page
            log.exception("skipping unreadable run dir %s", entry)
    runs.sort(key=lambda r: r.run_id, reverse=True)
    return runs


def load_run_summary(run_dir: Path, *, active: bool = False) -> RunSummary:
    run_id = run_dir.name
    try:
        started_at = datetime.strptime(run_id, "%Y%m%d_%H%M%S_%f")
    except ValueError:
        started_at = None

    task = ""
    duration = n_steps = tokens = cost = replayed = subtasks_total = None
    note: str | None = None
    try:
        data = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        status, tone = derive_status(data.get("is_successful"), data.get("is_done"),
                                    data.get("assertions_passed"))
        task = str(data.get("task") or "")
        duration = data.get("duration_seconds")
        n_steps = data.get("n_steps")
        usage = data.get("usage") or {}
        tokens = usage.get("total_tokens")
        cost = usage.get("total_cost")
        subtasks = data.get("subtasks")
        if isinstance(subtasks, list):
            subtasks_total = len(subtasks)
            replayed = sum(1 for s in subtasks
                           if isinstance(s, dict) and s.get("mode") == "replay")
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        corrupt = not isinstance(exc, FileNotFoundError)
        status, tone, note, task, replayed, subtasks_total = _status_without_a_report(
            run_dir, active=active, corrupt=corrupt)

    return RunSummary(
        run_id=run_id,
        path=run_dir,
        started_at=started_at,
        status=status,
        tone=tone,
        task=task,
        duration_seconds=duration,
        n_steps=n_steps,
        total_tokens=tokens,
        total_cost=cost,
        subtasks_replayed=replayed,
        subtasks_total=subtasks_total,
        size_bytes=_dir_size_bytes(run_dir),
        has_report_html=(run_dir / "report.html").is_file(),
        note=note,
    )


def _status_without_a_report(
    run_dir: Path, *, active: bool, corrupt: bool,
) -> tuple[str, str, str, str, int | None, int | None]:
    """The verdict for a run that left no usable `report.json`.

    Returns `(status, tone, note, task, segments_done, subtasks_total)`. A corrupt report keeps
    the recovered `INCOMPLETE` reading rather than consulting progress — a half-written report is
    a different fault from never having written one, and conflating them would hide it.
    """
    if corrupt:
        return "INCOMPLETE", "muted", "report.json corrupt", "", None, None

    prog = load_progress(run_dir)
    if not prog:
        return "INCOMPLETE", "muted", "report.json missing", "", None, None

    task = str(prog.get("task") or "")
    segments = prog.get("segments")
    done = len(segments) if isinstance(segments, list) else None
    total = prog.get("subtasks_total")
    total = total if isinstance(total, int) else None
    status_field = str(prog.get("status") or "")

    if status_field == "interrupted":
        return ("STOPPED", "muted", "stopped by the operator before it finished",
                task, done, total)
    if status_field == "finished":
        # progress reached the end but build_report never wrote (or raised). Must not read as a
        # pass just because is_successful happens to be true in progress.
        return ("DONE", "warning", "finished, but report.json was never written",
                task, done, total)
    if active:
        return "RUNNING", "info", "in flight", task, done, total
    # `running` is only ever left behind by a process that went away: the status field is moved
    # by a clean finish or a KeyboardInterrupt, and nothing else.
    return ("CRASHED", "error", "ended without a final status — the run process went away",
            task, done, total)


def delete_run(artifacts_dir: Path, run_id: str) -> None:
    """Remove artifacts_dir/<run_id> permanently. Raises ValueError for anything that
    is not a plain run id (the regex alone forbids '/', '.', and traversal shapes),
    FileNotFoundError if no such run exists."""
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"not a run id: {run_id!r}")
    target = (artifacts_dir / run_id).resolve()
    if target.parent != artifacts_dir.resolve():
        raise ValueError(f"run id escapes artifacts dir: {run_id!r}")
    if not target.is_dir():
        raise FileNotFoundError(f"no such run: {run_id}")
    shutil.rmtree(target)


def _dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        with os.scandir(path) as it:
            for entry in it:
                try:
                    if entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                    elif entry.is_dir(follow_symlinks=False):
                        total += _dir_size_bytes(Path(entry.path))
                except OSError:
                    pass
    except OSError:
        pass
    return total
