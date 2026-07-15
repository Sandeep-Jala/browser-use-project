"""Suite runner: execute many TaskSpecs on one login/browser and roll the results up into
artifacts/suites/<suite_id>/suite.json + suite.html.

The loop per task: reset app state -> run the task (replay / adapt / author — whatever
run_task decides) -> evaluate assertions -> build the per-run report -> append a TaskRecord.
A task that raises becomes ERROR and the suite continues (continue_on_failure); if the
browser/CDP connection itself is gone, the remaining tasks are SKIPPED and the suite stops —
without a browser every subsequent task would just burn its timeout to the same ERROR.

The heavy dependencies (running a task, resetting the app, probing the browser) are injected
as callables so the suite logic is testable without Playwright or credentials; __main__ wires
the real ones.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from automation.pipeline import assertions as asr
from automation.pipeline import subtask_store as sstore
from automation.pipeline.report import build_report, build_suite_report
from automation.tasks import TaskSpec

logger = logging.getLogger("framework.suite")

SUITES_DIRNAME = "suites"


@dataclass
class TaskRecord:
    """One row of the suite summary."""
    key: str
    tid: str
    status: str                       # PASS | DONE | FAIL | ERROR | SKIPPED
    assertions_ok: bool | None = None  # None = none evaluated; False turns a PASS into PASS*
    mode: str | None = None           # always "hybrid" (the only execution path)
    duration_seconds: float = 0.0
    healed_steps: list[int] = field(default_factory=list)
    # True when a stale library recording had to be re-authored by the agent mid-run.
    reauthored: bool = False
    assertions: dict[str, Any] = field(default_factory=dict)  # {passed, failed, skipped, failed_names}
    run_id: str | None = None
    report_html: str | None = None    # relative to the suite dir, for linking
    tokens: int | None = None
    cost: float | None = None
    error: str | None = None
    # Hybrid runs only: per-subtask modes (e.g. ["replay", "replay", "authored"]) and how
    # many segments came from the shared library at zero LLM cost. None for whole-task runs.
    subtask_modes: list[str] | None = None
    library_hits: int | None = None

    @property
    def ok(self) -> bool:
        """CI-gating verdict: the flow passed AND no assertion failed. DONE deliberately
        does not count (success unknown != success)."""
        return self.status == "PASS" and self.assertions_ok is not False


def _record_from_result(spec: TaskSpec, result: Any, suite_dir: Path,
                        report_paths: dict[str, Path]) -> TaskRecord:
    if result.is_successful:
        status = "PASS"
    elif result.is_done:
        status = "DONE"
    else:
        status = "FAIL"
    checks = result.assertion_results or []
    replay_log = (result.replay or {}).get("log") or []
    usage = result.usage or {}
    html_path = report_paths.get("html")
    subtasks = getattr(result, "subtasks", None)
    return TaskRecord(
        key=spec.key, tid=sstore.task_id(spec.prompt), status=status,
        assertions_ok=result.assertions_passed,
        mode=result.mode,
        duration_seconds=round(result.duration_seconds, 1),
        healed_steps=sorted({e["step"] for e in replay_log if e.get("healed")}),
        # Re-authoring is now per-SEGMENT: a task counts as re-authored when any subtask's
        # library replay broke and the agent had to take it over.
        reauthored=any(s.get("mode") == "replay_failed->authored" for s in subtasks or []),
        subtask_modes=[s.get("mode") for s in subtasks] if subtasks else None,
        library_hits=(sum(1 for s in subtasks if s.get("mode") == "replay")
                      if subtasks else None),
        assertions={
            "passed": sum(1 for a in checks if a.get("passed") is True),
            "failed": sum(1 for a in checks if a.get("passed") is False),
            "skipped": sum(1 for a in checks if a.get("passed") is None),
            "failed_names": [a["name"] for a in checks if a.get("passed") is False],
        },
        run_id=result.run_id,
        report_html=os.path.relpath(html_path, suite_dir) if html_path else None,
        tokens=usage.get("total_tokens"),
        cost=usage.get("total_cost"),
        error=result.final_result if status != "PASS" else None,
    )


async def run_suite(
    specs: list[TaskSpec],
    run_one: Callable[[TaskSpec], Awaitable[Any]],
    *,
    selector: str = "",
    artifacts_dir: Path | str = "artifacts",
    default_assertions: dict[str, Any] | None = None,
    continue_on_failure: bool = True,
    reset: Callable[[], Awaitable[None]] | None = None,
    browser_alive: Callable[[], Awaitable[bool]] | None = None,
) -> dict[str, Any]:
    """Run `specs` sequentially and write suite.json + suite.html. Returns the suite summary
    dict (its "ok" key is the CI verdict: every task PASS with assertions passing)."""
    defaults = asr.DEFAULT_SPEC if default_assertions is None else default_assertions
    suite_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = Path(artifacts_dir) / SUITES_DIRNAME / suite_id
    suite_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now()
    records: list[TaskRecord] = []
    abort_reason: str | None = None

    for i, spec in enumerate(specs):
        print(f"\n===== [{i + 1}/{len(specs)}] task '{spec.key}' =====")
        if reset is not None:
            await reset()
        try:
            result = await run_one(spec)
            asr.apply(result, asr.merge_spec(defaults, spec.assertions))
            report_paths = build_report(result)
            record = _record_from_result(spec, result, suite_dir, report_paths)
        except Exception as exc:  # noqa: BLE001 - one broken task must not sink the suite
            logger.exception("task %s errored: %s", spec.key, exc)
            record = TaskRecord(spec.key, sstore.task_id(spec.prompt), "ERROR",
                                error=f"{type(exc).__name__}: {exc}")
            if browser_alive is not None and not await browser_alive():
                abort_reason = f"browser gone after task '{spec.key}'"
        records.append(record)
        print(f"----- {spec.key}: {record.status}"
              + (f" ({record.error})" if record.error else ""))

        if abort_reason or (not continue_on_failure and not record.ok):
            remaining = specs[i + 1:]
            reason = abort_reason or "continue-on-failure disabled"
            for skipped in remaining:
                records.append(TaskRecord(skipped.key, sstore.task_id(skipped.prompt),
                                          "SKIPPED", error=reason))
            if remaining:
                print(f"----- skipping {len(remaining)} remaining task(s): {reason}")
            break

    duration = (datetime.now() - started).total_seconds()
    statuses = [r.status for r in records]
    summary = {
        "suite_id": suite_id,
        "selector": selector,
        "started": started.isoformat(timespec="seconds"),
        "duration_seconds": round(duration, 1),
        "ok": bool(records) and all(r.ok for r in records),
        "totals": {
            "tasks": len(records),
            "pass": statuses.count("PASS"),
            "done": statuses.count("DONE"),
            "fail": statuses.count("FAIL"),
            "error": statuses.count("ERROR"),
            "skipped": statuses.count("SKIPPED"),
            "assertion_failures": sum(1 for r in records if r.assertions_ok is False),
            "tokens": sum(r.tokens or 0 for r in records) or None,
            "cost": round(sum(r.cost or 0.0 for r in records), 4) or None,
        },
        "defaults": {"assertions": defaults},
        "tasks": [asdict(r) for r in records],
    }
    (suite_dir / "suite.json").write_text(json.dumps(summary, indent=2, default=str))
    summary["suite_html"] = str(build_suite_report(summary, suite_dir))
    summary["suite_json"] = str(suite_dir / "suite.json")
    return summary
