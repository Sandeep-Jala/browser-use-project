"""Auto Agent: launching, watching and steering a run.

A run is a SUBPROCESS, never an in-process call, and the reasons are all load-bearing:
`__main__.main()` owns the browser lifecycle inside its own `async_playwright()` context; it calls
`_kill_stale_browser()` unconditionally at startup; `Runner.run_agent_segment` installs a SIGINT
handler and degrades off the main thread; `logging.basicConfig` mutates global logging; and the
operator control channel is a module global plus one fixed file, so two runs in one process would
steal each other's commands.

Every test here drives a FAKE child — `sys.executable -c "<script>"` that prints to both streams,
makes a run dir, writes `progress.json`, and exits with a chosen code. No browser, no LLM, no
spend. The real argv composition and the real exit classification are pure functions, tested
directly, so nothing important is mocked away.

Three behaviours that exist because of measured facts about this codebase:

* **`PYTHONUNBUFFERED=1` is mandatory.** All the `[*] …` progress lines and the whole
  `========== RESULT ==========` block go to stdout, which Python block-buffers to a pipe. Without
  it the UI shows nothing until the process exits. (This is the same trap that hid browser-use's
  pause message during the 2026-09-10 control-channel work.)
* **Exit 130 is not a crash.** It is what an operator `stop` produces during a replay
  (`service_replay_control` raises KeyboardInterrupt). Reporting it as a crash would teach the
  operator to distrust their own stop button.
* **The control buttons stay disabled until the channel is armed.** `control.set_control_path`
  calls `reset_control`, which CLEARS the file — so a pause written before arming is silently
  discarded. The `🎛 operator control:` line that `reset_control` logs is the arming signal.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

from automation.pipeline.control import read_control
from automation.ui.paths import Paths
from automation.ui.supervisor import (
    RunBusy,
    RunRequest,
    RunSupervisor,
    build_argv,
    classify_exit,
)

# ── helpers ───────────────────────────────────────────────────────────────────────────────


def _paths(tmp_path: Path) -> Paths:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    return Paths(
        repo_root=tmp_path, tasks_file=tmp_path / "tasks.yaml",
        prompts_dir=tmp_path / "prompts", uploads_dir=tmp_path / "uploads",
        artifacts_dir=artifacts, library_dir=tmp_path / "library",
        errors_dir=tmp_path / "errors", control_path=artifacts / "control.json",
        ui_state_dir=artifacts / ".auto_agent",
    )


def _child(script: str) -> list[str]:
    return [sys.executable, "-c", script]


#: Prints to both streams, makes a run dir with a progress.json, then exits with EXIT.
FAKE_RUN = """
import json, os, sys, time
run = os.path.join("artifacts", "20260101_000000_000001")
os.makedirs(run, exist_ok=True)
print("[*] stdout line one")
print("OUT-ARMED", file=sys.stderr)
sys.stderr.write("INFO [framework.control] \\u1f39b operator control: write ...\\n")
sys.stderr.flush()
json.dump({"run_id": "r", "task": "t", "status": STATUS, "subtasks_total": 2,
           "planned": [], "segments": SEGMENTS, "is_successful": None},
          open(os.path.join(run, "progress.json"), "w"))
print("[*] stdout line two")
time.sleep(HOLD)
sys.exit(EXIT)
"""


def fake_run(*, exit_code=0, status="finished", segments=0, hold=0.0) -> list[str]:
    script = (FAKE_RUN.replace("EXIT", str(exit_code)).replace("STATUS", repr(status))
              .replace("SEGMENTS", repr([{"index": i, "ok": True} for i in range(segments)]))
              .replace("HOLD", str(hold)))
    return _child(script)


def _wait(sup: RunSupervisor, state: str, timeout: float = 15.0) -> dict:
    """Poll to a state with a deadline. Never `sleep` a fixed guess."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = sup.snapshot()
        if snap["state"] == state:
            return snap
        time.sleep(0.02)
    raise AssertionError(f"never reached {state!r}; last snapshot: {sup.snapshot()}")


async def _start(sup: RunSupervisor, argv: list[str], task: str = "a task") -> dict:
    sup._argv = lambda req: argv      # type: ignore[method-assign]
    return await sup.start(RunRequest(task=task))


# ── argv composition (the real thing, as a pure function) ─────────────────────────────────

def test_argv_runs_the_framework_module_with_the_task():
    argv = build_argv(RunRequest(task="invoice"), python="/py")
    assert argv == ["/py", "-m", "automation", "--task", "invoice"]


def test_argv_carries_a_free_text_prompt_as_one_argument():
    """`resolve_task` accepts free text containing a space; it must arrive unsplit."""
    argv = build_argv(RunRequest(task="go to the bookkeeping module and stop"), python="/py")
    assert argv[-2:] == ["--task", "go to the bookkeeping module and stop"]


def test_argv_includes_every_flag_the_cli_offers():
    argv = build_argv(RunRequest(task="t", fresh=True, marker="Invoices", redecompose=True,
                                 reauthor="0,add invoice", log_all_hosts=True, record=True),
                      python="/py")
    assert "--fresh" in argv and "--redecompose" in argv
    assert "--log-all-hosts" in argv and "--record" in argv
    assert argv[argv.index("--marker") + 1] == "Invoices"
    assert argv[argv.index("--reauthor") + 1] == "0,add invoice"


def test_an_empty_marker_is_not_passed_at_all():
    """Blank must mean "leave the default alone", while the literal 'none' is how the CLI is
    told to DISABLE the gate — two different intentions that a blank string would merge."""
    assert "--marker" not in build_argv(RunRequest(task="t", marker=""), python="/py")
    assert "--marker" not in build_argv(RunRequest(task="t", marker=None), python="/py")
    assert build_argv(RunRequest(task="t", marker="none"), python="/py")[-1] == "none"


# ── exit classification (the table, as a pure function) ───────────────────────────────────

@pytest.mark.parametrize("code,status,verdict", [
    (0, "finished", "Passed"),
    (1, "finished", "Failed"),
    (1, "running", "Did not start"),
    (1, "", "Did not start"),
    (130, "interrupted", "Stopped by you"),
    (130, "running", "Stopped by you"),
    (130, "", "Stopped by you"),
    (2, "running", "Crashed"),
    (-9, "running", "Killed"),
])
def test_exit_classification_covers_the_whole_table(code, status, verdict):
    assert classify_exit(code, status)[0].startswith(verdict)


def test_a_stop_is_never_reported_as_a_crash():
    """Exit 130 is what the operator's own stop produces during a replay."""
    verdict, tone, _ = classify_exit(130, "interrupted")
    assert "crash" not in verdict.lower() and tone == "warning"


def test_a_failure_before_the_first_step_is_distinguished_from_a_failed_run():
    """Exit 1 with no progress file is a SystemExit from main(): an unknown task key, an
    unresolved file reference, a login failure. Telling the operator their prompt "failed" would
    send them to debug the wrong thing."""
    assert classify_exit(1, "finished")[0] == "Failed"
    assert classify_exit(1, "")[0] == "Did not start"


# ── launching ─────────────────────────────────────────────────────────────────────────────

async def test_a_launch_records_the_process_and_reaches_done(tmp_path):
    sup = RunSupervisor(_paths(tmp_path))
    started = await _start(sup, fake_run(exit_code=0))
    assert started["pid"] > 0
    snap = _wait(sup, "done")
    assert snap["exit_code"] == 0 and snap["verdict"].startswith("Passed")


async def test_the_child_runs_with_unbuffered_stdout_and_the_repo_root_as_cwd(tmp_path):
    """Both are invisible until they bite: buffered stdout shows an empty console for minutes,
    and the wrong cwd makes tasks.yaml, library/ and prompts/ all resolve to nothing."""
    paths = _paths(tmp_path)
    sup = RunSupervisor(paths)
    await _start(sup, _child(
        'import os,sys; print(os.environ.get("PYTHONUNBUFFERED","")); print(os.getcwd())'))
    _wait(sup, "done")
    out = "\n".join(l["text"] for l in sup.log_since(0)["lines"])
    assert "1" in out
    assert str(paths.repo_root.resolve()) in out


async def test_a_second_launch_is_refused_while_one_is_running(tmp_path):
    sup = RunSupervisor(_paths(tmp_path))
    await _start(sup, fake_run(hold=5))
    with pytest.raises(RunBusy):
        await sup.start(RunRequest(task="another"))
    sup.kill()


async def test_two_simultaneous_launches_only_ever_start_one(tmp_path):
    """A bare state check would let both through — the window between reading the state and
    spawning is real. Only one of these may win."""
    import asyncio
    sup = RunSupervisor(_paths(tmp_path))
    sup._argv = lambda req: fake_run(hold=3)     # type: ignore[method-assign]
    results = await asyncio.gather(
        sup.start(RunRequest(task="a")), sup.start(RunRequest(task="b")),
        return_exceptions=True)
    started = [r for r in results if not isinstance(r, Exception)]
    refused = [r for r in results if isinstance(r, RunBusy)]
    assert len(started) == 1 and len(refused) == 1
    sup.kill()


async def test_a_finished_run_does_not_block_the_next_one(tmp_path):
    sup = RunSupervisor(_paths(tmp_path))
    await _start(sup, fake_run(exit_code=0))
    _wait(sup, "done")
    await _start(sup, fake_run(exit_code=0))       # must not raise
    _wait(sup, "done")


# ── log capture ───────────────────────────────────────────────────────────────────────────

async def test_both_streams_are_captured_with_monotone_sequence_numbers(tmp_path):
    """stdout carries the progress lines and the RESULT block; stderr carries every
    framework.* and browser_use log line. A UI that reads one of them is blind to half the run."""
    sup = RunSupervisor(_paths(tmp_path))
    await _start(sup, _child(
        'import sys\n'
        'print("from stdout")\n'
        'print("from stderr", file=sys.stderr)\n'))
    _wait(sup, "done")

    lines = sup.log_since(0)["lines"]
    streams = {l["stream"] for l in lines}
    assert streams == {"stdout", "stderr"}
    assert [l["seq"] for l in lines] == sorted(l["seq"] for l in lines)
    assert any("from stdout" in l["text"] for l in lines)
    assert any("from stderr" in l["text"] for l in lines)


async def test_log_since_returns_only_new_lines(tmp_path):
    sup = RunSupervisor(_paths(tmp_path))
    await _start(sup, fake_run())
    _wait(sup, "done")
    first = sup.log_since(0)
    assert first["lines"]
    assert sup.log_since(first["next"])["lines"] == []


async def test_the_console_is_mirrored_to_a_file_before_any_run_dir_exists(tmp_path):
    """The run dir does not exist until after decomposition, and a run that dies during login
    never gets one at all — so the console has to be recoverable from somewhere else."""
    paths = _paths(tmp_path)
    sup = RunSupervisor(paths)
    await _start(sup, fake_run())
    _wait(sup, "done")
    logs = list(paths.ui_state_dir.glob("*.log"))
    assert logs and "stdout line one" in logs[0].read_text()


# ── run-dir discovery ─────────────────────────────────────────────────────────────────────

async def test_the_run_dir_is_discovered_and_its_progress_read(tmp_path):
    """The UI cannot know the run id — the framework allocates it inside Runner._new_run_dir —
    so it watches artifacts/ for a new directory instead of parsing the log for it."""
    sup = RunSupervisor(_paths(tmp_path))
    await _start(sup, fake_run(status="finished", segments=1, hold=0.6))
    # Poll for the PROGRESS, not just the dir: the directory appears first and progress.json lands
    # a moment later, which is exactly the real "Planning the steps" window.
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not sup.snapshot()["progress"]:
        time.sleep(0.02)
    snap = sup.snapshot()
    assert snap["run_id"] == "20260101_000000_000001"
    assert snap["progress"]["subtasks_total"] == 2
    _wait(sup, "done")


async def test_a_run_dir_that_already_existed_is_not_claimed(tmp_path):
    """Otherwise the first launch after any previous run would attach to the old one's dir."""
    paths = _paths(tmp_path)
    (paths.artifacts_dir / "20260101_000000_000001").mkdir()
    sup = RunSupervisor(paths)
    await _start(sup, _child('import time; time.sleep(0.5)'))
    _wait(sup, "done")
    assert sup.snapshot()["run_id"] is None


# ── phases ────────────────────────────────────────────────────────────────────────────────

async def test_the_phase_before_a_run_dir_exists_says_it_is_starting(tmp_path):
    sup = RunSupervisor(_paths(tmp_path))
    await _start(sup, _child('import time; time.sleep(1.5)'))
    assert "start" in sup.snapshot()["phase"].lower()
    sup.kill()


async def test_a_run_dir_without_progress_is_reported_as_planning(tmp_path):
    """The most confusing window in a real run: the dir is created at HybridSession.open but
    progress.json is not written until the decomposition finishes, which on a new prompt is 30s+
    of apparent silence."""
    paths = _paths(tmp_path)
    sup = RunSupervisor(paths)
    await _start(sup, _child(
        'import os, time\n'
        'os.makedirs(os.path.join("artifacts", "20260101_000000_000001"))\n'
        'time.sleep(2)\n'))
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not sup.snapshot()["run_id"]:
        time.sleep(0.02)
    assert "planning" in sup.snapshot()["phase"].lower()
    sup.kill()


# ── the control channel ───────────────────────────────────────────────────────────────────

async def test_control_is_refused_until_the_channel_is_armed(tmp_path):
    """`set_control_path` → `reset_control` CLEARS the file, so a pause written before the run
    arms its channel is discarded without a word. Refusing early is the honest alternative."""
    paths = _paths(tmp_path)
    sup = RunSupervisor(paths)
    await _start(sup, _child('import time; time.sleep(2)'))
    assert sup.snapshot()["control"]["armed"] is False
    with pytest.raises(RuntimeError, match="armed"):
        sup.control("pause")
    sup.kill()


async def test_the_arming_line_on_stderr_enables_control(tmp_path):
    paths = _paths(tmp_path)
    sup = RunSupervisor(paths)
    await _start(sup, _child(
        'import sys, time\n'
        'sys.stderr.write("INFO [framework.control] operator control: write {\\"command\\"} to x\\n")\n'
        'sys.stderr.flush()\n'
        'time.sleep(2)\n'))
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not sup.snapshot()["control"]["armed"]:
        time.sleep(0.02)
    assert sup.snapshot()["control"]["armed"] is True
    sup.control("pause")
    assert read_control(paths.control_path)["command"] == "pause"
    sup.kill()


async def test_a_steer_instruction_reaches_the_control_file(tmp_path):
    paths = _paths(tmp_path)
    sup = RunSupervisor(paths)
    await _start(sup, _child(
        'import sys, time\n'
        'sys.stderr.write("operator control: armed\\n"); sys.stderr.flush()\n'
        'time.sleep(2)\n'))
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not sup.snapshot()["control"]["armed"]:
        time.sleep(0.02)
    sup.control("resume", "close the dialog first")
    raw = read_control(paths.control_path)
    assert raw["command"] == "resume" and raw["instruction"] == "close the dialog first"
    sup.kill()


def test_an_unknown_command_is_refused(tmp_path):
    sup = RunSupervisor(_paths(tmp_path))
    with pytest.raises(ValueError):
        sup.control("paws")


def test_control_is_refused_when_nothing_is_running(tmp_path):
    sup = RunSupervisor(_paths(tmp_path))
    with pytest.raises(RuntimeError):
        sup.control("pause")


async def test_a_stale_command_is_cleared_at_launch(tmp_path):
    """A pause left over from the previous run would hold the new one at its first boundary."""
    paths = _paths(tmp_path)
    paths.control_path.parent.mkdir(parents=True, exist_ok=True)
    paths.control_path.write_text('{"command": "pause"}')
    sup = RunSupervisor(paths)
    await _start(sup, fake_run())
    assert read_control(paths.control_path).get("command") == ""
    _wait(sup, "done")


async def test_a_pause_is_reported_as_requested_then_confirmed(tmp_path):
    """Confirmation has to be INFERRED: `service_control` clears the file the moment it applies
    a pause, so an empty file is both the resting state and the paused state. The supervisor's
    own intent flag plus a PAUSED log line is the best available evidence, and the UI says so."""
    paths = _paths(tmp_path)
    sup = RunSupervisor(paths)
    await _start(sup, _child(
        'import sys, time\n'
        'sys.stderr.write("operator control: armed\\n"); sys.stderr.flush()\n'
        'time.sleep(0.6)\n'
        'sys.stderr.write("INFO [framework.control] Replay PAUSED between steps\\n")\n'
        'sys.stderr.flush()\n'
        'time.sleep(3)\n'))
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not sup.snapshot()["control"]["armed"]:
        time.sleep(0.02)

    sup.control("pause")
    assert sup.snapshot()["control"]["pause_requested"] is True
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not sup.snapshot()["control"]["pause_confirmed"]:
        time.sleep(0.02)
    assert sup.snapshot()["control"]["pause_confirmed"] is True
    assert "paused" in sup.snapshot()["phase"].lower()
    sup.kill()


async def test_a_replay_refusing_an_instruction_is_surfaced(tmp_path):
    """The engine refuses a steer during a replay out loud; the UI must repeat it rather than
    leave the operator believing they redirected the run."""
    sup = RunSupervisor(_paths(tmp_path))
    await _start(sup, _child(
        'import sys, time\n'
        'sys.stderr.write("WARNING [framework.control] a replaying subtask cannot be steered '
        '— IGNORED: \'click Cancel\'\\n")\n'
        'sys.stderr.flush(); time.sleep(0.3)\n'))
    _wait(sup, "done")
    assert any("cannot be steered" in w for w in sup.snapshot()["warnings"])


# ── killing ───────────────────────────────────────────────────────────────────────────────

async def test_kill_ends_a_run(tmp_path):
    sup = RunSupervisor(_paths(tmp_path))
    await _start(sup, _child('import time; time.sleep(30)'))
    sup.kill()
    snap = _wait(sup, "done")
    assert snap["exit_code"] != 0


# ── surviving a server restart ────────────────────────────────────────────────────────────

async def test_a_live_run_is_re_adopted_by_a_fresh_supervisor(tmp_path):
    """Supervisor state is in memory, so without this a server restart would orphan a live run
    and refuse every launch forever."""
    paths = _paths(tmp_path)
    first = RunSupervisor(paths)
    await _start(first, _child('import time; time.sleep(4)'))
    pid = first.snapshot()["pid"]

    second = RunSupervisor(paths)
    second.adopt()
    snap = second.snapshot()
    assert snap["state"] == "running" and snap["pid"] == pid
    assert snap["adopted"] is True
    first.kill()


async def test_adoption_ignores_a_process_that_is_already_gone(tmp_path):
    paths = _paths(tmp_path)
    first = RunSupervisor(paths)
    await _start(first, fake_run())
    _wait(first, "done")

    second = RunSupervisor(paths)
    second.adopt()
    assert second.snapshot()["state"] == "idle"


async def test_the_adoption_record_is_removed_when_a_run_ends(tmp_path):
    paths = _paths(tmp_path)
    sup = RunSupervisor(paths)
    await _start(sup, fake_run())
    _wait(sup, "done")
    assert not (paths.ui_state_dir / "run.json").exists()
