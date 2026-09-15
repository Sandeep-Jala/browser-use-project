"""Auto Agent: scanning `artifacts/` into run rows.

The scan layer and its tests are recovered from the runs dashboard removed on 2026-07-20
(`stash@{2}^3:automation/dashboard/scan.py` / `tests/test_dashboard.py`). Kept rather than
rewritten because the details a rewrite loses are the ones that matter: the `FileNotFoundError`
catch for a run deleted between `iterdir` and load, the `target.parent != artifacts_dir.resolve()`
escape check in `delete_run`, and the corrupt-vs-missing note split.

One thing is new, and it is the reason the old dashboard looked broken on this branch:
`report.json` is written ONCE, from `__main__.py` after a clean return — so an interrupted run
(operator `stop`, exit 130) and a crashed one never get a report at all, and the old code
collapsed both into a bare `INCOMPLETE`. 2 of 6 real run dirs on this machine are in that state
right now. `progress.json` is written incrementally and atomically all through a run, so it can
tell those cases apart — and it carries the task and the segments, so a stopped run can still
show what it was doing and how far it got.

The case that needs care: a run that is still going has `progress.json` status `running` and no
`report.json` — which is indistinguishable, from a directory alone, from one that crashed. So the
caller (the run supervisor, which knows its own run id) passes `active_run_id`; only that run is
RUNNING, and anything else in that shape has stopped existing without finishing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from automation.ui.runs import delete_run, load_progress, scan_runs


def make_run(root: Path, run_id: str, *, is_successful=True, is_done=True,
             assertions_passed=True, subtask_modes=("replay", "authored"),
             corrupt=False, no_json=False, no_html=False, progress=None) -> Path:
    """Seed a minimal artifacts/<run_id>/ dir the way a real run leaves it."""
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    if corrupt:
        (run_dir / "report.json").write_text("{not json", encoding="utf-8")
    elif not no_json:
        (run_dir / "report.json").write_text(json.dumps({
            "task": f"task for {run_id}",
            "is_successful": is_successful,
            "is_done": is_done,
            "assertions_passed": assertions_passed,
            "duration_seconds": 58.7,
            "n_steps": 12,
            "usage": {"total_tokens": 12345, "total_cost": 0.0421},
            "subtasks": [{"mode": m} for m in subtask_modes],
        }), encoding="utf-8")
    if not no_html:
        (run_dir / "report.html").write_text(f"<html>report {run_id}</html>", encoding="utf-8")
    if progress is not None:
        (run_dir / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
    return run_dir


def _progress(status="running", *, task="a live task", segments=2, total=4, ok=None):
    return {
        "run_id": "x", "task": task, "task_id": "abc", "status": status,
        "started": "2026-09-10T16:00:00", "updated": "2026-09-10T16:01:00",
        "subtasks_total": total,
        "planned": [{"prompt": f"step {i}", "kind": "action"} for i in range(total)],
        "segments": [{"index": i, "mode": "replay", "ok": True} for i in range(segments)],
        "is_successful": ok,
    }


# ── scanning (recovered) ──────────────────────────────────────────────────────────────────

def test_scan_orders_newest_first_and_skips_non_runs(tmp_path):
    make_run(tmp_path, "20260101_000000_000001")
    make_run(tmp_path, "20260301_000000_000001", is_successful=False, is_done=False)
    (tmp_path / ".DS_Store").write_bytes(b"junk")
    (tmp_path / "notarun").mkdir()
    (tmp_path / "20260201_000000_000001").write_text("a FILE named like a run id")

    runs = scan_runs(tmp_path)
    assert [r.run_id for r in runs] == ["20260301_000000_000001", "20260101_000000_000001"]
    assert [r.status for r in runs] == ["FAIL", "PASS"]
    assert runs[0].tone == "error" and runs[1].tone == "success"


def test_scan_summary_fields(tmp_path):
    make_run(tmp_path, "20260101_000000_000001",
             subtask_modes=("replay", "replay", "replay_failed->authored", "authored"))
    (run,) = scan_runs(tmp_path)
    assert run.task == "task for 20260101_000000_000001"
    assert run.duration_seconds == 58.7 and run.n_steps == 12
    assert run.total_tokens == 12345 and run.total_cost == 0.0421
    assert (run.subtasks_replayed, run.subtasks_total) == (2, 4)  # ->authored counts authored
    assert run.started_at is not None and run.started_at.year == 2026
    assert run.has_report_html and run.size_bytes > 0 and run.note is None


def test_scan_missing_dir(tmp_path):
    assert scan_runs(tmp_path / "nope") == []


def test_a_corrupt_report_is_incomplete_not_a_crash(tmp_path):
    make_run(tmp_path, "20260102_000000_000001", corrupt=True, no_html=True)
    (run,) = scan_runs(tmp_path)
    assert run.status == "INCOMPLETE" and run.note == "report.json corrupt"
    assert not run.has_report_html and run.size_bytes > 0


def test_a_run_with_neither_file_is_incomplete(tmp_path):
    make_run(tmp_path, "20260101_000000_000001", no_json=True)
    (run,) = scan_runs(tmp_path)
    assert (run.status, run.tone) == ("INCOMPLETE", "muted")
    assert run.note == "report.json missing"


# ── delete_run (recovered) ────────────────────────────────────────────────────────────────

def test_delete_run_removes_only_target(tmp_path):
    keep = make_run(tmp_path, "20260101_000000_000001")
    gone = make_run(tmp_path, "20260102_000000_000001")
    delete_run(tmp_path, "20260102_000000_000001")
    assert not gone.exists() and keep.exists()


@pytest.mark.parametrize("bad", [
    "../x", "20260101_000000_000000/../..", "nope", "", ".",
    "/etc/passwd", "20260101_000000_00000",  # one digit short
])
def test_delete_run_rejects_non_run_ids(tmp_path, bad):
    with pytest.raises(ValueError):
        delete_run(tmp_path, bad)


def test_delete_run_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        delete_run(tmp_path, "20260101_000000_000001")


# ── progress.json: telling the report-less runs apart ─────────────────────────────────────

def test_load_progress_reads_a_run_in_flight(tmp_path):
    run_dir = make_run(tmp_path, "20260101_000000_000001", no_json=True, no_html=True,
                       progress=_progress())
    prog = load_progress(run_dir)
    assert prog["status"] == "running" and prog["subtasks_total"] == 4


def test_load_progress_is_empty_when_absent_or_broken(tmp_path):
    run_dir = make_run(tmp_path, "20260101_000000_000001", no_json=True, no_html=True)
    assert load_progress(run_dir) == {}
    (run_dir / "progress.json").write_text("{half-writ")
    assert load_progress(run_dir) == {}


def test_an_operator_stop_reads_as_stopped_not_incomplete(tmp_path):
    """Exit 130 via the control file: progress says `interrupted`, no report was ever written.
    Calling that INCOMPLETE hides the one thing the operator already knows."""
    make_run(tmp_path, "20260101_000000_000001", no_json=True, no_html=True,
             progress=_progress("interrupted", task="the task I stopped", segments=1))
    (run,) = scan_runs(tmp_path)
    assert (run.status, run.tone) == ("STOPPED", "muted")
    assert run.task == "the task I stopped", "progress.json carries the task too"
    assert (run.subtasks_replayed, run.subtasks_total) == (1, 4)
    assert "stopped" in (run.note or "").lower()


def test_a_run_that_died_without_finishing_reads_as_crashed(tmp_path):
    """status still `running` and nobody owns it: the process went away mid-run. The status
    field is only ever moved off `running` by a clean finish or a KeyboardInterrupt, so this
    shape is exactly 'it died'."""
    make_run(tmp_path, "20260101_000000_000001", no_json=True, no_html=True,
             progress=_progress("running"))
    (run,) = scan_runs(tmp_path)
    assert (run.status, run.tone) == ("CRASHED", "error")


def test_the_run_in_flight_is_running_not_crashed(tmp_path):
    """The one case a directory scan cannot call on its own — so the supervisor, which knows
    its own run id, says which one is live."""
    make_run(tmp_path, "20260101_000000_000001", no_json=True, no_html=True,
             progress=_progress("running"))
    (run,) = scan_runs(tmp_path, active_run_id="20260101_000000_000001")
    assert (run.status, run.tone) == ("RUNNING", "info")


def test_a_finished_run_whose_report_never_landed_says_so(tmp_path):
    """progress reached `finished` but build_report never ran (or raised). Rare, and it must
    not masquerade as a pass."""
    make_run(tmp_path, "20260101_000000_000001", no_json=True, no_html=True,
             progress=_progress("finished", ok=True))
    (run,) = scan_runs(tmp_path)
    assert run.status == "DONE" and run.tone == "warning"
    assert "report" in (run.note or "").lower()


def test_report_json_still_wins_when_both_exist(tmp_path):
    """The normal case: a completed run has both files, and the report is the authority."""
    make_run(tmp_path, "20260101_000000_000001", is_successful=True,
             progress=_progress("finished", ok=True))
    (run,) = scan_runs(tmp_path)
    assert run.status == "PASS" and run.note is None
