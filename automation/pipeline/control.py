"""The operator's control channel: pause, steer and stop a live run from outside it.

One file — `<artifacts_dir>/control.json` — carrying one command at a time. It exists because
Ctrl+C (`Runner._prompt_and_inject`) only reaches a run that is in the foreground of an
attached terminal with the operator sitting at it, which is false for a run driven from a UI,
a background process, or another machine.

TWO consumers, with deliberately different powers, because they are not the same thing:

* An AGENT segment (`service_control`) can be paused, STEERED and stopped. Steering means one
  human instruction injected as an override the model reads on its next step.
* A REPLAYING segment (`service_replay_control`) can only be paused and stopped. There is no
  model in a replay, so there is nothing to steer — and an instruction sent to one is
  REFUSED OUT LOUD rather than dropped, or an operator would believe they had redirected a
  run that carried on regardless.

Both are serviced at a BOUNDARY BETWEEN ACTIONS, never mid-action: the agent's step start,
and the replay's per-verb hook in skills/api.py.

Every command is consumed before it is acted on. Left in the file it would re-fire on every
later step — the same instruction injected ten times over.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from automation.pipeline.atomic import replace_with_retry

logger = logging.getLogger("framework.control")


# ── the file ─────────────────────────────────────────────────────────────────────────────
# Ctrl+C (_prompt_and_inject) only reaches a run that is in the foreground of an attached
# terminal with the operator sitting at it. A control FILE reaches one from anywhere — another
# shell, a background process, a UI — which is what a clickable Pause button needs. Commands
# are serviced at the SAME step boundary the Ctrl+C pause is, so neither can land mid-action.
CONTROL_FILENAME = "control.json"

# How often a held step boundary re-reads the file. A paused run is waiting on a human, so
# the only cost of a slower poll is the operator's patience; the only cost of a faster one is
# a stat per tick. 0.4s is imperceptible either way.
_CONTROL_POLL_S = 0.4

_CONTROL_COMMANDS = ("pause", "resume", "stop")

OVERRIDE_PREFIX = ":warning:  HUMAN OPERATOR OVERRIDE"


def read_control(path: Path) -> dict[str, Any]:
    """The operator's queued command, or `{}` when there is none.

    Every failure mode reads as "no command", deliberately. An external writer is not
    atomic, so a step that reads the file mid-write sees truncated JSON — that must not
    crash the run and must not be guessed at. The operator will still be holding the
    boundary on the next tick.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def write_control(path: Path, command: str, instruction: str = "") -> None:
    """Queue `command` for the next step boundary. Written via os.replace so a step that
    reads while we write sees the OLD file whole, never a half-written one — and retried,
    because on Windows that replace fails outright while the reader holds the file open."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            "command": command,
            "instruction": instruction,
            "written_at": datetime.now().isoformat(timespec="seconds"),
        },
        indent=2,
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    replace_with_retry(tmp, path)


def clear_control(path: Path) -> None:
    """Consume whatever is in the channel, quietly. Used the moment a command is applied, so
    it cannot re-fire on a later step."""
    write_control(path, "")


def reset_control(path: Path) -> None:
    """Clear the channel at the start of a segment AND say where it is — a command left
    behind by a previous run must never steer the next one, and a channel nobody can find is
    not a channel (whoever needs it is by definition not the person who started the run).

    Applying a command uses the quiet `clear_control` instead: live run 20260910_123817
    re-printed these instructions three times, once per command, which is just noise in the
    middle of the log an operator is reading to decide what to do next.
    """
    clear_control(path)
    logger.info('🎛 operator control: write {"command":"pause"} to %s (then "resume" with an '
                'optional "instruction", or "stop")', path)


def _control_command(raw: dict[str, Any]) -> str:
    return str(raw.get("command") or "").strip().lower()


def inject_override(agent: Any, instruction: str) -> bool:
    """Inject one human instruction as an override the agent acts on at its next step.

    The single real primitive is `_message_manager.add_new_task` — what the public
    `Agent.add_new_task` delegates to, used directly because the public method also
    recreates the agent's event bus, which is unsafe mid-run.
    """
    override = f"{OVERRIDE_PREFIX}\n\n{instruction}"
    try:
        mm = getattr(agent, "_message_manager", None)
        if mm is not None and hasattr(mm, "add_new_task"):
            mm.add_new_task(override)
        else:
            agent.add_new_task(override)  # public fallback (recreates the event bus)
        return True
    except Exception as exc:  # noqa: BLE001 - a failed injection must not kill the run
        logger.warning("could not inject human instruction: %s", exc)
        return False


async def service_control(agent: Any, path: Path) -> None:
    """Apply the operator's queued command, if any. Call at a step boundary.

    `pause` HOLDS here until the file says `resume` or `stop` (or the file is deleted —
    `rm control.json` is the bluntest resume and must not wedge the run). `resume` carries
    an optional instruction, which is also how you steer a running agent without pausing it
    first. `stop` asks browser-use to end the run; note the CURRENT step still completes,
    because `on_step_start` fires inside `_execute_step`, after the loop's own stopped check.

    Every command is consumed before it is acted on. A command left in the file would
    re-fire on every later step — the same instruction injected ten times over.
    """
    command = _control_command(read_control(path))
    if not command:
        return

    if command not in _CONTROL_COMMANDS:
        logger.warning("ignoring unknown operator command %r (expected one of %s)",
                       command, ", ".join(_CONTROL_COMMANDS))
        clear_control(path)
        return

    if command == "pause":
        clear_control(path)
        agent.pause()
        command, instruction = await _await_release(path)
        # Clear `state.paused` before the step body runs either way: browser-use's
        # `_check_stop_or_pause` raises InterruptedError on a paused agent (service.py:1024)
        # and it is called mid-step, so a lingering flag would fail the step we just let go.
        agent.resume()
    else:
        instruction = str(read_control(path).get("instruction") or "").strip()
        clear_control(path)

    if instruction:
        inject_override(agent, instruction)
    if command == "stop":
        agent.stop()


async def _await_release(path: Path) -> tuple[str, str]:
    """Hold until the operator releases a pause. Returns the releasing command and any
    instruction it carried, having consumed it."""
    while True:
        await asyncio.sleep(_CONTROL_POLL_S)
        if not path.exists():
            return "resume", ""
        raw = read_control(path)
        command = _control_command(raw)
        if command in ("resume", "stop"):
            clear_control(path)
            return command, str(raw.get("instruction") or "").strip()




# ── the replay consumer ───────────────────────────────────────────────────────────────────
# A replaying subtask runs no agent and no LLM, so it never enters run_agent_segment and never
# saw this channel at all. Measured on run 20260910_125537: subtask 0 replayed in 3.6s and
# logged no announcement, while the only one in that run belonged to the authored subtask. On
# a warm library that is most of a run, so the whole fast path was uninterruptible except by
# Ctrl+C — which aborts rather than holds.
#
# The path is a module global set once per task, the same idiom agent_tools uses for
# set_live_network / set_repeat_budget: per-run state the replay layer needs and cannot be
# handed through a generated skill's signature.
_control_path: Path | None = None


def set_control_path(path: Path | None) -> None:
    """Point the replay layer at the channel (None disables it — the default, so unit tests
    and any non-hybrid caller replay with no channel configured). Clears and announces, so a
    command left by a previous run cannot hold this one."""
    global _control_path
    _control_path = path
    if path is not None:
        reset_control(path)


def control_path() -> Path | None:
    return _control_path


async def service_replay_control(path: Path | None = None) -> None:
    """Apply a queued command between two replayed actions. Pause and stop only.

    `stop` raises KeyboardInterrupt, for two reasons that both matter: every handler between
    a replay verb and the abort is `except Exception`, so a KeyboardInterrupt rides straight
    through to the handler hybrid.py already has for an operator abort (which writes
    progress.json and re-raises); and it can never be read as a FAILED replay, which would
    hand the subtask to the agent — a takeover that is a different feature.
    """
    if path is None:
        path = _control_path
    if path is None:
        return

    raw = read_control(path)
    command = _control_command(raw)
    if not command:
        return

    if command not in _CONTROL_COMMANDS:
        logger.warning("ignoring unknown operator command %r (expected one of %s)",
                       command, ", ".join(_CONTROL_COMMANDS))
        clear_control(path)
        return

    if command == "pause":
        clear_control(path)
        # logger, NOT print: stdout is block-buffered to a pipe, so a `print` here would be
        # invisible in every redirected run — exactly how browser-use's own pause message
        # disappeared on 2026-09-10 and made a held run look like a hung one.
        logger.info('⏸️  Replay PAUSED between steps. Write {"command":"resume"} '
                    '(or {"command":"stop"}) to %s', path)
        command, instruction = await _await_release(path)
        if command != "stop":
            logger.info("▶️  Replay RESUMED.")
    else:
        instruction = str(raw.get("instruction") or "").strip()
        clear_control(path)

    if instruction:
        # Never silently swallowed: the operator has to learn their steer did nothing.
        logger.warning(
            "a replaying subtask cannot be steered — it runs a recorded script with no LLM, "
            "so there is nothing to give an instruction to. IGNORED: %r. To change what this "
            "subtask does, stop the run and re-record it (--reauthor), or edit its entry in "
            "library/.", instruction)

    if command == "stop":
        logger.info("🛑 operator stop during replay — aborting the run")
        raise KeyboardInterrupt("stopped by the operator via the control file")
