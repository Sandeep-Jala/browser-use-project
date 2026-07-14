"""Suite-runner tests with fake run_one callables — no browser, no credentials."""
import json
from pathlib import Path

import pytest

from automation.pipeline import assertions as asr
from automation.pipeline.runner import RunResult
from automation.pipeline.suite import run_suite
from automation.tasks import TaskSpec


def _spec(key, assertions=None):
    return TaskSpec(key=key, prompt=f"do the {key} thing", marker="X", assertions=assertions)


def _result(tmp_path, key, *, successful=True, done=True, with_5xx=False, mode="replay",
            healed=None):
    requests = [{"step": 1, "method": "POST", "status": 201, "url": "https://app/api/X",
                 "failed": False, "status_class": "2xx"}]
    if with_5xx:
        requests.append({"step": 2, "method": "GET", "status": 502,
                         "url": "https://app/api/oops", "failed": False, "status_class": "5xx"})
    replay = None
    if healed is not None:
        replay = {"executed": 3, "failed_at": None, "error": None,
                  "log": [{"step": s, "action": "click", "used": "healed:x",
                           "healed": {"tag": "button", "attrs": {}}} for s in healed]}
    return RunResult(
        task=key, run_id=f"run_{key}", artifacts_dir=tmp_path / f"run_{key}",
        expanded_task=None, is_done=done, is_successful=successful,
        has_errors=not successful, final_result=f"{key} finished", urls=[], n_steps=3,
        duration_seconds=2.0, extracted_content=[], model_actions=[], errors=[],
        collector_results={"network": {"summary": {}, "requests": requests},
                           "console": {"summary": {}, "entries": []}},
        usage={"total_tokens": 100, "total_cost": 0.01},
        mode=mode, replay=replay,
    )


async def test_suite_totals_statuses_and_files(tmp_path):
    outcomes = {
        "good": _result(tmp_path, "good", healed=[4]),
        "bad": _result(tmp_path, "bad", successful=False, done=False),
        "done_only": _result(tmp_path, "done_only", successful=False, done=True),
        "unhealthy": _result(tmp_path, "unhealthy", with_5xx=True),
    }

    async def run_one(spec):
        if spec.key == "boom":
            raise RuntimeError("browser exploded")
        return outcomes[spec.key]

    specs = [_spec(k) for k in ("good", "bad", "done_only", "unhealthy", "boom")]
    summary = await run_suite(specs, run_one, selector="all", artifacts_dir=tmp_path)

    assert summary["ok"] is False
    t = summary["totals"]
    assert (t["tasks"], t["pass"], t["fail"], t["done"], t["error"]) == (5, 2, 1, 1, 1)
    assert t["assertion_failures"] == 1
    assert t["tokens"] == 400

    by_key = {r["key"]: r for r in summary["tasks"]}
    assert by_key["good"]["status"] == "PASS" and by_key["good"]["assertions_ok"] is True
    assert by_key["good"]["healed_steps"] == [4]
    assert by_key["unhealthy"]["status"] == "PASS"
    assert by_key["unhealthy"]["assertions_ok"] is False
    assert "no_5xx" in by_key["unhealthy"]["assertions"]["failed_names"]
    assert by_key["boom"]["status"] == "ERROR"
    assert "browser exploded" in by_key["boom"]["error"]

    suite_dir = Path(summary["suite_json"]).parent
    data = json.loads((suite_dir / "suite.json").read_text())
    assert data["totals"] == t
    html = (suite_dir / "suite.html").read_text()
    assert "PASS*" in html            # unhealthy row
    assert "boom" in html and "good" in html
    # Per-run report link is relative to the suite dir.
    assert by_key["good"]["report_html"]
    assert not Path(by_key["good"]["report_html"]).is_absolute()


async def test_all_pass_suite_is_ok(tmp_path):
    async def run_one(spec):
        return _result(tmp_path, spec.key)

    summary = await run_suite([_spec("a"), _spec("b")], run_one, artifacts_dir=tmp_path)
    assert summary["ok"] is True
    assert summary["totals"]["pass"] == 2


async def test_continue_on_failure_off_skips_rest(tmp_path):
    calls = []

    async def run_one(spec):
        calls.append(spec.key)
        return _result(tmp_path, spec.key, successful=spec.key != "bad", done=False)

    specs = [_spec("a"), _spec("bad"), _spec("never")]
    summary = await run_suite(specs, run_one, artifacts_dir=tmp_path,
                              continue_on_failure=False)

    assert calls == ["a", "bad"]
    by_key = {r["key"]: r for r in summary["tasks"]}
    assert by_key["never"]["status"] == "SKIPPED"
    assert summary["ok"] is False


async def test_dead_browser_skips_remaining(tmp_path):
    async def run_one(spec):
        if spec.key == "crash":
            raise ConnectionError("CDP gone")
        return _result(tmp_path, spec.key)

    async def browser_alive():
        return False

    specs = [_spec("a"), _spec("crash"), _spec("x"), _spec("y")]
    summary = await run_suite(specs, run_one, artifacts_dir=tmp_path,
                              browser_alive=browser_alive)

    statuses = [r["status"] for r in summary["tasks"]]
    assert statuses == ["PASS", "ERROR", "SKIPPED", "SKIPPED"]
    assert "browser gone" in summary["tasks"][2]["error"]


async def test_reset_called_before_each_task(tmp_path):
    events = []

    async def reset():
        events.append("reset")

    async def run_one(spec):
        events.append(spec.key)
        return _result(tmp_path, spec.key)

    await run_suite([_spec("a"), _spec("b")], run_one, artifacts_dir=tmp_path, reset=reset)
    assert events == ["reset", "a", "reset", "b"]


async def test_per_task_assertion_override(tmp_path):
    async def run_one(spec):
        return _result(tmp_path, spec.key, with_5xx=True)

    # Task-level override disables the 5xx rule for one task only.
    specs = [_spec("strict"), _spec("lenient", assertions={"no_5xx": False})]
    summary = await run_suite(specs, run_one, artifacts_dir=tmp_path)

    by_key = {r["key"]: r for r in summary["tasks"]}
    assert by_key["strict"]["assertions_ok"] is False
    assert by_key["lenient"]["assertions_ok"] is True
    assert summary["ok"] is False


async def test_hybrid_result_populates_subtask_fields(tmp_path):
    async def run_one(spec):
        result = _result(tmp_path, spec.key, mode="hybrid")
        result.subtasks = [
            {"index": 0, "mode": "replay", "ok": True},
            {"index": 1, "mode": "replay", "ok": True},
            {"index": 2, "mode": "authored", "ok": True},
        ]
        return result

    summary = await run_suite([_spec("hy")], run_one, artifacts_dir=tmp_path)

    record = summary["tasks"][0]
    assert record["subtask_modes"] == ["replay", "replay", "authored"]
    assert record["library_hits"] == 2
    # Whole-task runs keep the fields empty (additive change).
    assert "subtask_modes" in record


async def test_whole_task_result_leaves_subtask_fields_none(tmp_path):
    async def run_one(spec):
        return _result(tmp_path, spec.key)

    summary = await run_suite([_spec("plain")], run_one, artifacts_dir=tmp_path)
    record = summary["tasks"][0]
    assert record["subtask_modes"] is None
    assert record["library_hits"] is None
