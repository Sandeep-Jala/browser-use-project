"""Launching, watching and steering ONE run.

A run is a subprocess, never an in-process call. `__main__.main()` owns the browser lifecycle
inside its own `async_playwright()` context, calls `_kill_stale_browser()` unconditionally at
startup, installs a SIGINT handler that degrades off the main thread, and mutates global logging —
and the operator control channel is a module global plus one fixed file, so two runs in one
process would steal each other's commands. A subprocess sidesteps all of it, at the price of
having to rediscover state the child owns.

Where the state comes from, and why not from the logs:

* **Progress** — `<run_dir>/progress.json`, written atomically by `hybrid._write_progress` before
  the first subtask and after every completed segment. Parsing the console for progress would be
  guessing at something already serialised.
* **The run id** — by watching `artifacts/` for a new directory. The framework allocates it inside
  `Runner._new_run_dir` and never tells anyone, and a log line naming it would be a format to
  depend on.
* **The verdict** — the exit code crossed with `progress.status` (see `classify_exit`).
* **The console** — both pipes. stdout carries the `[*]` lines and the RESULT block; stderr
  carries every `framework.*` and `browser_use` log line. Reading one of them is being blind to
  half the run.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from automation.pipeline.control import clear_control, read_control, write_control
from automation.ui.paths import Paths
from automation.ui.runs import RUN_ID_RE, load_progress

log = logging.getLogger("framework.ui")

# `reset_control`'s announcement. Its arrival is the only honest signal that the run has reached
# the point where a command will not be discarded — `set_control_path` CLEARS the channel, so
# anything written before this line is silently thrown away.
_ARMED_MARKER = "operator control:"
# `service_replay_control` / the agent path announce a held boundary with this word. Needed because
# a pause cannot be confirmed from the file: it is cleared the instant it is applied.
_PAUSED_MARKER = "paused"
# The engine refuses a steer during a replay out loud. Repeat it rather than let the operator
# believe they redirected the run.
_REFUSAL_MARKER = "cannot be steered"

_LOG_LINE_CAP = 4000
_RUN_DIR_POLL_S = 0.5
_COMMANDS = ("pause", "resume", "stop")


class RunBusy(RuntimeError):
    """A run is already in flight. There is one browser and one control channel."""


class BrowserBusy(RuntimeError):
    """Another process already holds the CDP port — see `start(force=...)`."""


@dataclass
class RunRequest:
    task: str
    fresh: bool = False
    marker: str | None = None
    redecompose: bool = False
    reauthor: str | None = None
    log_all_hosts: bool = False
    record: bool = False


def build_argv(req: RunRequest, *, python: str | None = None) -> list[str]:
    """The exact command line `automation/__main__.py` accepts.

    A blank marker is OMITTED rather than passed: blank means "leave the configured default
    alone", while the literal string `none` is how the CLI is told to DISABLE the ground-truth
    gate. Collapsing those two would silently turn the gate off for every UI run.
    """
    argv = [python or sys.executable, "-m", "automation", "--task", req.task]
    if req.fresh:
        argv.append("--fresh")
    if req.redecompose:
        argv.append("--redecompose")
    if req.log_all_hosts:
        argv.append("--log-all-hosts")
    if req.record:
        argv.append("--record")
    if req.marker:
        argv += ["--marker", req.marker]
    if req.reauthor:
        argv += ["--reauthor", req.reauthor]
    return argv


def classify_exit(exit_code: int, progress_status: str) -> tuple[str, str, str]:
    """`(verdict, tone, reason)` for a finished run.

    The two rows that matter most:

    * **130 is never a crash.** It is what an operator `stop` produces when it lands during a
      replay (`service_replay_control` raises KeyboardInterrupt, which rides hybrid's abort path).
      Calling it a crash would teach the operator to distrust their own stop button.
    * **Exit 1 with no progress file is "did not start", not "failed".** `main()` raises
      `SystemExit(str)` for an unknown task key, an unresolved file reference, a missing CDP
      endpoint or a login failure — all before the first step. Reporting that as a failed run
      sends the operator to debug the prompt instead of the setup.
    """
    status = (progress_status or "").strip()
    if exit_code == 130:
        if status == "interrupted":
            return "Stopped by you", "warning", "the run was stopped before it finished"
        return ("Stopped by you (before the first step boundary)", "warning",
                "stopped so early that no progress was recorded")
    if exit_code < 0:
        return (f"Killed (signal {-exit_code})", "error",
                "the run process was killed from outside")
    if exit_code == 0:
        return "Passed", "success", ""
    if exit_code == 1:
        if status in ("finished", "interrupted"):
            return "Failed", "error", "the run completed but did not reach its end state"
        return ("Did not start", "error",
                "the run exited before its first step — usually an unknown task key, a file "
                "reference that could not be resolved, or a login failure")
    return (f"Crashed (exit {exit_code})", "error",
            "the run process exited with an unexpected status")


@dataclass
class _LogLine:
    seq: int
    stream: str
    ts: str
    text: str


@dataclass
class _State:
    """Everything about the current or most recent run."""

    token: str = ""
    state: str = "idle"              # idle | starting | running | done
    argv: list[str] = field(default_factory=list)
    task_label: str = ""
    pid: int | None = None
    started_at: float = 0.0
    run_id: str | None = None
    exit_code: int | None = None
    verdict: str = ""
    verdict_tone: str = ""
    reason: str = ""
    adopted: bool = False
    armed: bool = False
    pause_requested: bool = False
    pause_confirmed: bool = False
    last_command: str = ""
    last_command_at: str = ""
    warnings: list[str] = field(default_factory=list)


class RunSupervisor:
    """One at a time, deliberately: there is one browser, and the control channel is one file."""

    def __init__(self, paths: Paths, *, python: str | None = None) -> None:
        self.paths = paths
        self._python = python or sys.executable
        self._lock = asyncio.Lock()
        self._mutex = threading.Lock()        # guards _lines / _state from the reader threads
        self._s = _State()
        self._proc: subprocess.Popen[str] | None = None
        self._lines: list[_LogLine] = []
        self._dropped = 0
        self._seq = 0
        self._log_file: Path | None = None
        self._known_run_ids: set[str] = set()

    # -- launching ------------------------------------------------------------------------

    def _argv(self, req: RunRequest) -> list[str]:
        return build_argv(req, python=self._python)

    async def start(self, req: RunRequest, *, force: bool = False) -> dict[str, Any]:
        """Spawn a run. Raises `RunBusy` if one is already in flight, `BrowserBusy` if another
        process holds the CDP port.

        The lock is not decoration: between reading the state and spawning there is a real window,
        and two quick clicks of Run would otherwise both get through.
        """
        async with self._lock:
            if self._s.state in ("starting", "running"):
                raise RunBusy("a run is already in flight — stop it before starting another")
            if not force:
                owner = _foreign_cdp_owner()
                if owner is not None:
                    raise BrowserBusy(
                        f"process {owner} already has a debugging browser open. Starting a run "
                        f"would close it: the framework kills any browser on the CDP port at "
                        f"startup. Stop that run first, or launch with force.")
            return self._spawn(req)

    def _spawn(self, req: RunRequest) -> dict[str, Any]:
        argv = self._argv(req)
        self.paths.ui_state_dir.mkdir(parents=True, exist_ok=True)
        # A pause left in the channel by a previous run would hold this one at its first boundary.
        clear_control(self.paths.control_path)

        token = uuid.uuid4().hex[:12]
        self._known_run_ids = self._existing_run_ids()
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            argv, cwd=str(self.paths.repo_root), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, errors="replace",
            # Its own process group: a Ctrl+C in the server's terminal must not also SIGINT the
            # run, and a hard kill can then signal the whole group.
            start_new_session=True,
        )

        with self._mutex:
            self._s = _State(token=token, state="running", argv=list(argv),
                             task_label=req.task, pid=proc.pid, started_at=time.time())
            self._lines, self._seq, self._dropped = [], 0, 0
        self._proc = proc
        self._log_file = self.paths.ui_state_dir / f"{token}.log"

        (self.paths.ui_state_dir / "run.json").write_text(json.dumps({
            "token": token, "pid": proc.pid, "argv": argv,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }, indent=2), encoding="utf-8")

        for stream_name, pipe in (("stdout", proc.stdout), ("stderr", proc.stderr)):
            threading.Thread(target=self._read_stream, args=(stream_name, pipe),
                             daemon=True, name=f"auto-agent-{stream_name}").start()
        threading.Thread(target=self._watch_run_dir, daemon=True,
                         name="auto-agent-rundir").start()
        threading.Thread(target=self._wait_for_exit, args=(proc, token), daemon=True,
                         name="auto-agent-wait").start()
        log.info("run %s started: %s", token, " ".join(argv))
        return {"run_token": token, "pid": proc.pid}

    # -- background threads ---------------------------------------------------------------

    def _read_stream(self, stream: str, pipe: Any) -> None:
        if pipe is None:
            return
        for raw in iter(pipe.readline, ""):
            text = raw.rstrip("\n")
            with self._mutex:
                self._seq += 1
                self._lines.append(_LogLine(self._seq, stream, _now(), text))
                if len(self._lines) > _LOG_LINE_CAP:
                    self._dropped += len(self._lines) - _LOG_LINE_CAP
                    del self._lines[:-_LOG_LINE_CAP]
                self._note_markers(text)
            if self._log_file is not None:
                try:
                    with self._log_file.open("a", encoding="utf-8") as fh:
                        fh.write(f"{stream}: {text}\n")
                except OSError:
                    pass
        try:
            pipe.close()
        except OSError:
            pass

    def _note_markers(self, text: str) -> None:
        """Read the three things only the console can tell us. Called under the mutex."""
        low = text.lower()
        if not self._s.armed and _ARMED_MARKER in low:
            self._s.armed = True
        if self._s.pause_requested and not self._s.pause_confirmed and _PAUSED_MARKER in low:
            self._s.pause_confirmed = True
        if _REFUSAL_MARKER in low:
            self._s.warnings.append(text.strip())

    def _watch_run_dir(self) -> None:
        """Claim the first NEW run directory. Snapshotting the existing ids first matters: without
        it the next launch would attach to the previous run's directory."""
        while self._proc is not None and self._proc.poll() is None:
            if self._s.run_id is None:
                new = sorted(self._existing_run_ids() - self._known_run_ids)
                if new:
                    with self._mutex:
                        self._s.run_id = new[0]
                    log.info("run %s -> %s", self._s.token, new[0])
            time.sleep(_RUN_DIR_POLL_S)
        # One last look: a short run can finish before the first poll.
        if self._s.run_id is None:
            new = sorted(self._existing_run_ids() - self._known_run_ids)
            if new:
                with self._mutex:
                    self._s.run_id = new[0]

    def _wait_for_exit(self, proc: subprocess.Popen[str], token: str) -> None:
        code = proc.wait()
        # Let the reader threads drain; the verdict reads progress.json, which the child writes
        # before it exits, but the last console lines can still be in flight.
        time.sleep(0.15)
        status = str(self._progress().get("status") or "")
        verdict, tone, reason = classify_exit(code, status)
        with self._mutex:
            if self._s.token != token:
                return                     # superseded by a newer run
            self._s.state = "done"
            self._s.exit_code = code
            self._s.verdict, self._s.verdict_tone, self._s.reason = verdict, tone, reason
        try:
            (self.paths.ui_state_dir / "run.json").unlink(missing_ok=True)
        except OSError:
            pass
        log.info("run %s finished: exit %s -> %s", token, code, verdict)

    # -- reading state --------------------------------------------------------------------

    def _existing_run_ids(self) -> set[str]:
        try:
            return {e.name for e in self.paths.artifacts_dir.iterdir()
                    if e.is_dir() and RUN_ID_RE.fullmatch(e.name)}
        except OSError:
            return set()

    def run_dir(self) -> Path | None:
        return None if not self._s.run_id else self.paths.artifacts_dir / self._s.run_id

    def _progress(self) -> dict[str, Any]:
        d = self.run_dir()
        return load_progress(d) if d is not None else {}

    def snapshot(self) -> dict[str, Any]:
        with self._mutex:
            s = self._s
            progress = self._progress()
            alive = self._proc is not None and self._proc.poll() is None
            return {
                "run_token": s.token,
                "state": s.state,
                "pid": s.pid,
                "phase": _phase(s, progress, alive=alive),
                "task_label": s.task_label,
                "argv": list(s.argv),
                "started_at": s.started_at or None,
                "elapsed_s": round(time.time() - s.started_at, 1) if s.started_at else None,
                "run_id": s.run_id,
                "run_dir": str(self.run_dir()) if s.run_id else None,
                "progress": progress or None,
                "control": {
                    "armed": s.armed,
                    "last_command": s.last_command,
                    "last_command_at": s.last_command_at,
                    "pause_requested": s.pause_requested,
                    "pause_confirmed": s.pause_confirmed,
                },
                "exit_code": s.exit_code,
                "verdict": s.verdict,
                "verdict_tone": s.verdict_tone,
                "reason": s.reason,
                "adopted": s.adopted,
                "warnings": list(s.warnings),
                "log_next_seq": self._seq + 1,
            }

    def log_since(self, seq: int) -> dict[str, Any]:
        with self._mutex:
            lines = [l for l in self._lines if l.seq >= seq]
            return {
                "lines": [{"seq": l.seq, "stream": l.stream, "ts": l.ts, "text": l.text}
                          for l in lines],
                "next": self._seq + 1,
                "dropped": self._dropped,
            }

    # -- steering -------------------------------------------------------------------------

    def control(self, command: str, instruction: str = "") -> dict[str, Any]:
        """Queue an operator command. Refuses before the channel is armed rather than writing
        into a file the run is about to clear."""
        if command not in _COMMANDS:
            raise ValueError(f"unknown command {command!r} — expected one of "
                             f"{', '.join(_COMMANDS)}")
        if self._s.state != "running":
            raise RuntimeError("no run is in flight")
        if not self._s.armed:
            raise RuntimeError(
                "the run has not armed its control channel yet — it does that after login. "
                "A command written now would be cleared when it does.")
        write_control(self.paths.control_path, command, instruction)
        with self._mutex:
            self._s.last_command = command
            self._s.last_command_at = _now()
            if command == "pause":
                self._s.pause_requested, self._s.pause_confirmed = True, False
            else:
                self._s.pause_requested = self._s.pause_confirmed = False
        return {"written": True, "control_path": str(self.paths.control_path),
                "command": command}

    def kill(self) -> dict[str, Any]:
        """Last resort. SIGINT the group first — the framework's own handlers turn that into a
        clean abort that still writes progress.json — then SIGKILL if it will not go."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return {"signalled": False}
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except (OSError, ProcessLookupError):
            proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                proc.kill()
        return {"signalled": True}

    # -- surviving a restart --------------------------------------------------------------

    def adopt(self) -> None:
        """Re-attach to a run started before this server process existed.

        Without it, a server restart during a run leaves an orphan the UI cannot see and a
        `RunBusy` it can never clear. The console of an adopted run is gone — its pipes belonged
        to the old process — and the snapshot says so via `adopted`.
        """
        record = self.paths.ui_state_dir / "run.json"
        try:
            data = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        pid, argv = data.get("pid"), data.get("argv") or []
        if not isinstance(pid, int) or not _process_matches(pid, argv):
            record.unlink(missing_ok=True)
            return
        with self._mutex:
            self._s = _State(
                token=str(data.get("token") or ""), state="running", argv=list(argv),
                task_label=_task_from_argv(argv), pid=pid,
                started_at=time.time(), adopted=True,
                # An adopted run has already armed its channel if it got this far; assume it has,
                # because the announcement was printed to a console we no longer hold.
                armed=True,
                warnings=["re-attached to a run started before the UI restarted — its console "
                          "output was not captured"],
            )
            self._known_run_ids = set()
        # Claim its run dir by picking the newest one, since we have no launch snapshot to diff.
        newest = sorted(self._existing_run_ids())
        if newest:
            with self._mutex:
                self._s.run_id = newest[-1]
        threading.Thread(target=self._watch_adopted, args=(pid,), daemon=True).start()
        log.info("re-adopted run pid %s", pid)

    def _watch_adopted(self, pid: int) -> None:
        while _pid_alive(pid):
            time.sleep(1.0)
        status = str(self._progress().get("status") or "")
        verdict, tone, reason = classify_exit(0 if status == "finished" else 1, status)
        with self._mutex:
            self._s.state = "done"
            self._s.verdict, self._s.verdict_tone = verdict, tone
            self._s.reason = reason + " (exit code unknown: the run was adopted)"
        (self.paths.ui_state_dir / "run.json").unlink(missing_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _task_from_argv(argv: list[str]) -> str:
    try:
        return argv[argv.index("--task") + 1]
    except (ValueError, IndexError):
        return ""


def _phase(s: _State, progress: dict[str, Any], *, alive: bool) -> str:
    """What to tell the operator is happening, derived rather than guessed.

    The window worth naming explicitly: the run dir is created at `HybridSession.open` but
    `progress.json` is not written until the decomposition finishes, which on a new prompt is 30s
    or more of apparent silence. A generic spinner there reads as a hang.
    """
    if s.state == "idle":
        return "Idle"
    if s.state == "done":
        return s.verdict or "Finished"
    if s.pause_confirmed:
        return "Paused"
    if s.pause_requested:
        return "Pause requested — will hold at the next step boundary"
    if s.run_id is None:
        return "Starting — launching the browser and logging in"
    if not progress:
        return "Planning the steps"
    status = str(progress.get("status") or "")
    done = len(progress.get("segments") or [])
    total = progress.get("subtasks_total") or "?"
    if status == "interrupted":
        return "Stopped"
    if status == "finished":
        return "Finishing — writing the report"
    if not alive:
        return "Ended without a final status"
    return f"Running step {done + 1} of {total}"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _process_matches(pid: int, argv: list[str]) -> bool:
    """Is `pid` still the run we recorded?

    A live pid alone is not enough: pids get reused, and adopting an unrelated process would wedge
    the UI on a `RunBusy` that never clears. So compare the live command line against the one that
    was recorded at launch — for a different process to pass, it would have to be running the same
    command, which is near enough the same run.
    """
    if not _pid_alive(pid) or not argv:
        return False
    try:
        import psutil
        cmdline = psutil.Process(pid).cmdline()
    except Exception:  # noqa: BLE001 - psutil absent or the process vanished mid-check
        return False
    return list(cmdline) == list(argv)


def _foreign_cdp_owner() -> int | None:
    """The pid of a debugging browser nobody here owns, or None.

    `__main__.main()` calls `_kill_stale_browser(cdp_port)` unconditionally at startup, killing
    any process whose cmdline mentions that port. So launching from the UI while a run is going in
    a terminal would silently close that run's browser. Better to refuse and say why.
    """
    try:
        import psutil
        from automation.config import Config
        port = Config.from_env().cdp_port
    except Exception:  # noqa: BLE001 - never block a launch on the guard itself failing
        return None
    needle = f"remote-debugging-port={port}"
    try:
        for proc in psutil.process_iter(["pid", "cmdline"]):
            cmdline = proc.info.get("cmdline") or []
            if any(needle in str(part) for part in cmdline):
                return int(proc.info["pid"])
    except Exception:  # noqa: BLE001
        return None
    return None
