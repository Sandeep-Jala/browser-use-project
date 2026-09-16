"""The operator's out-of-band control channel: pause, steer and stop a live segment
WITHOUT a TTY.

Until now the only human-in-the-loop lever was Ctrl+C (`Runner._prompt_and_inject`), which
needs three things at once: the run in the foreground, a terminal attached, and the operator
sitting at that terminal. None of those hold for a run driven from a UI, a background
process, or another machine — which is exactly the gap the Capsitech.AutoAgent comparison
(2026-09-10) exposed. Theirs drove pause/resume over HTTP; the capability was right even
though the mechanism was not (a 0.5s busy-poll and a five-strategy injection, three of whose
strategies either duplicate the one real primitive or write a `str` into a `bool`/
`list[PlanItem]` field).

So: a control FILE, read at the same step boundary the Ctrl+C pause is already serviced at.
Commands are `pause`, `resume` (with an optional instruction — which is also how you steer a
running agent without pausing it first) and `stop`. Pausing and resuming go through
browser-use's own `Agent.pause()` / `Agent.resume()` (service.py:3920) so `state.paused` and
`_external_pause_event` stay truthful for anything else reading them, and the single
injection primitive is `_message_manager.add_new_task` — the one `Agent.add_new_task`
itself delegates to, used directly because the public method recreates the agent's event bus
mid-run.

Two properties this file pins hardest, because both are how a control channel lies:

1. A command is applied EXACTLY ONCE. A command left sitting in the file would re-fire on
   every subsequent step — an instruction injected ten times, or a resume that un-pauses a
   pause the operator issued a moment later.
2. A malformed or half-written file is NOT a command. An external writer is not atomic, so
   a step that reads mid-write must see "no command", never crash the run and never guess.
"""
import asyncio
import inspect
import json
from pathlib import Path

import pytest

from automation.pipeline.runner import (
    Runner,
    read_control,
    reset_control,
    service_control,
    service_replay_control,
    write_control,
)


class FakeMessageManager:
    def __init__(self) -> None:
        self.tasks: list[str] = []

    def add_new_task(self, task: str) -> None:
        self.tasks.append(task)


class FakeAgent:
    """Only the surface `service_control` is allowed to touch."""

    def __init__(self) -> None:
        self._message_manager = FakeMessageManager()
        self.calls: list[str] = []

    def pause(self) -> None:
        self.calls.append("pause")

    def resume(self) -> None:
        self.calls.append("resume")

    def stop(self) -> None:
        self.calls.append("stop")


# ── reading the file ──────────────────────────────────────────────────────────────────────

def test_missing_control_file_is_no_command(tmp_path):
    assert read_control(tmp_path / "control.json") == {}


def test_half_written_control_file_is_no_command(tmp_path):
    """An external writer is not atomic. A step landing mid-write must not raise."""
    path = tmp_path / "control.json"
    path.write_text('{"command": "pa')
    assert read_control(path) == {}


def test_control_file_holding_a_non_object_is_no_command(tmp_path):
    path = tmp_path / "control.json"
    path.write_text('["pause"]')
    assert read_control(path) == {}


def test_write_control_round_trips_through_read_control(tmp_path):
    path = tmp_path / "control.json"
    write_control(path, "resume", instruction="click Save")
    raw = read_control(path)
    assert raw["command"] == "resume"
    assert raw["instruction"] == "click Save"


def test_reset_control_clears_a_stale_command(tmp_path):
    """A command left by a previous run must never steer the next one."""
    path = tmp_path / "control.json"
    write_control(path, "pause")
    reset_control(path)
    assert read_control(path).get("command") == ""


# ── steering ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resume_instruction_is_injected_with_the_override_marker(tmp_path):
    path = tmp_path / "control.json"
    write_control(path, "resume", instruction="the dialog is still open — close it first")
    agent = FakeAgent()

    await service_control(agent, path)

    assert len(agent._message_manager.tasks) == 1
    injected = agent._message_manager.tasks[0]
    assert "HUMAN OPERATOR OVERRIDE" in injected
    assert "the dialog is still open — close it first" in injected


@pytest.mark.asyncio
async def test_an_instruction_is_injected_exactly_once(tmp_path):
    """The command must be consumed. Left in the file it would re-inject every step."""
    path = tmp_path / "control.json"
    write_control(path, "resume", instruction="scroll the panel list to the bottom")
    agent = FakeAgent()

    await service_control(agent, path)
    await service_control(agent, path)
    await service_control(agent, path)

    assert len(agent._message_manager.tasks) == 1


@pytest.mark.asyncio
async def test_a_resume_with_no_instruction_injects_nothing(tmp_path):
    path = tmp_path / "control.json"
    write_control(path, "resume")
    agent = FakeAgent()

    await service_control(agent, path)

    assert agent._message_manager.tasks == []


@pytest.mark.asyncio
async def test_no_command_touches_the_agent_at_all(tmp_path):
    agent = FakeAgent()
    await service_control(agent, tmp_path / "control.json")
    assert agent.calls == []
    assert agent._message_manager.tasks == []


# ── stopping ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stop_stops_the_agent(tmp_path):
    path = tmp_path / "control.json"
    write_control(path, "stop")
    agent = FakeAgent()

    await service_control(agent, path)

    assert "stop" in agent.calls


@pytest.mark.asyncio
async def test_stop_is_applied_exactly_once(tmp_path):
    path = tmp_path / "control.json"
    write_control(path, "stop")
    agent = FakeAgent()

    await service_control(agent, path)
    await service_control(agent, path)

    assert agent.calls.count("stop") == 1


# ── pausing ───────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pause_blocks_until_the_file_says_resume(tmp_path):
    """The whole point of the channel: the step boundary holds while the operator looks at
    the browser, and releases when they say so — no TTY involved."""
    path = tmp_path / "control.json"
    write_control(path, "pause")
    agent = FakeAgent()

    async def release():
        # Let the pause take hold, then release it from "outside the process".
        await asyncio.sleep(0.25)
        write_control(path, "resume", instruction="pick Harris Duncan, not Aaran")

    await asyncio.wait_for(
        asyncio.gather(service_control(agent, path), release()),
        timeout=10,
    )

    assert agent.calls == ["pause", "resume"]
    assert "pick Harris Duncan, not Aaran" in agent._message_manager.tasks[0]


@pytest.mark.asyncio
async def test_a_pause_released_by_stop_does_not_run_on(tmp_path):
    path = tmp_path / "control.json"
    write_control(path, "pause")
    agent = FakeAgent()

    async def release():
        await asyncio.sleep(0.25)
        write_control(path, "stop")

    await asyncio.wait_for(
        asyncio.gather(service_control(agent, path), release()),
        timeout=10,
    )

    assert "stop" in agent.calls


@pytest.mark.asyncio
async def test_deleting_the_control_file_releases_a_pause(tmp_path):
    """`rm control.json` is the operator's bluntest resume. It must not wedge the run."""
    path = tmp_path / "control.json"
    write_control(path, "pause")
    agent = FakeAgent()

    async def release():
        await asyncio.sleep(0.25)
        path.unlink()

    await asyncio.wait_for(
        asyncio.gather(service_control(agent, path), release()),
        timeout=10,
    )

    assert agent.calls == ["pause", "resume"]


@pytest.mark.asyncio
async def test_a_pause_is_not_re_entered_after_it_is_released(tmp_path):
    path = tmp_path / "control.json"
    write_control(path, "pause")
    agent = FakeAgent()

    async def release():
        await asyncio.sleep(0.25)
        write_control(path, "resume")

    await asyncio.wait_for(
        asyncio.gather(service_control(agent, path), release()),
        timeout=10,
    )
    # The next step boundary must sail straight through.
    await service_control(agent, path)

    assert agent.calls == ["pause", "resume"]


@pytest.mark.asyncio
async def test_an_unknown_command_is_ignored_and_cleared(tmp_path):
    """Never block on a typo."""
    path = tmp_path / "control.json"
    (tmp_path / "control.json").write_text(json.dumps({"command": "paws"}))
    agent = FakeAgent()

    await asyncio.wait_for(service_control(agent, path), timeout=5)

    assert agent.calls == []
    assert read_control(path).get("command") == ""


# ── wiring ────────────────────────────────────────────────────────────────────────────────
# The channel is serviced from inside a closure built per segment (`_track_step`), which no
# unit test can reach without a live browser. These pin the wiring by source inspection —
# the same idiom test_element_listing_budget.py uses for the listing budget. A string match
# is a weak assertion, but the failure it catches is the real one: a refactor quietly
# dropping the call and leaving a control file nothing ever reads.

def test_the_step_boundary_services_the_control_channel():
    src = inspect.getsource(Runner.run_agent_segment)
    assert "service_control(" in src


def test_the_segment_clears_a_stale_command_before_it_starts():
    """Otherwise a pause left over from a previous run holds the first step of the next one.
    Quietly: the task start owns the one announcement (see the set_control_path test)."""
    src = inspect.getsource(Runner.run_agent_segment)
    assert "clear_control(" in src


def test_setting_the_channel_path_announces_it_and_clears_it(tmp_path, caplog):
    """The guarantee that moved to the task start when the replay layer joined: an operator
    who did not start the run still learns where the channel is, exactly once, and a command
    left by a previous run cannot hold this one."""
    import logging
    from automation.pipeline import control
    path = tmp_path / "control.json"
    write_control(path, "pause")
    try:
        with caplog.at_level(logging.INFO, logger="framework.control"):
            control.set_control_path(path)
        assert "control.json" in caplog.text
        assert read_control(path).get("command") == ""
        assert control.control_path() == path
    finally:
        control.set_control_path(None)


def test_ctrl_c_and_the_control_file_share_one_injection_primitive():
    """Two injection paths drift. The Capsitech build had five, three of them broken."""
    src = inspect.getsource(Runner._prompt_and_inject)
    assert "inject_override(" in src
    assert "add_new_task" not in src


def test_reset_control_announces_the_path(caplog):
    """A channel nobody can find is not a channel. Every segment says where it is, because
    the operator who needs it is by definition not the one who started the run."""
    import logging
    from pathlib import Path
    with caplog.at_level(logging.INFO, logger="framework.control"):
        reset_control(Path("artifacts") / "control.json")
    assert "control.json" in caplog.text


@pytest.mark.asyncio
async def test_applying_a_command_does_not_re_announce_the_path(tmp_path, caplog):
    """Live run 20260910_123817 logged the 🎛 announcement three extra times — once per
    command applied — because consuming a command cleared the file through the same
    announcing helper the segment start uses. The announcement belongs to the segment
    start only; a paused run re-printing its own instructions is noise."""
    import logging
    path = tmp_path / "control.json"
    write_control(path, "resume", instruction="carry on")
    agent = FakeAgent()
    with caplog.at_level(logging.INFO, logger="framework.control"):
        await service_control(agent, path)
    assert "🎛" not in caplog.text


# ── the replay path ───────────────────────────────────────────────────────────────────────
# A replaying subtask runs no agent and no LLM, so `run_agent_segment` — and with it the
# whole control channel — is never entered. Measured on run 20260910_125537: subtask 0
# replayed in 3.6s and produced no 🎛 line at all, while the only one in the log belonged to
# the authored subtask 1. On a warm library that is most of a run, so the fast path was
# uninterruptible except by Ctrl+C, which aborts rather than holds.
#
# In replay there is nothing to steer — no model to talk to — so the channel offers exactly
# two commands here, and an instruction must be REFUSED out loud rather than silently
# dropped. `stop` raises KeyboardInterrupt: the replay path has no bare or BaseException
# handler anywhere (checked), so it propagates to the abort handler hybrid.py already has,
# and it can never be mistaken for a FAILED replay — which would hand the subtask to the
# agent, a takeover nobody asked for.

@pytest.mark.asyncio
async def test_replay_pause_holds_until_resume(tmp_path):
    path = tmp_path / "control.json"
    write_control(path, "pause")

    async def release():
        await asyncio.sleep(0.25)
        write_control(path, "resume")

    held = asyncio.get_running_loop().time()
    await asyncio.wait_for(
        asyncio.gather(service_replay_control(path), release()), timeout=10)
    assert asyncio.get_running_loop().time() - held >= 0.2


@pytest.mark.asyncio
async def test_replay_stop_raises_keyboard_interrupt(tmp_path):
    """KeyboardInterrupt, so it rides hybrid.py's existing abort path and is never read as a
    failed replay (which would trigger the replay_failed->authored takeover)."""
    path = tmp_path / "control.json"
    write_control(path, "stop")
    with pytest.raises(KeyboardInterrupt):
        await service_replay_control(path)


@pytest.mark.asyncio
async def test_replay_stop_survives_an_except_exception_wrapper(tmp_path):
    """Every handler between a replay verb and the abort is `except Exception`. If stop ever
    became an Exception subclass it would be swallowed and reported as a broken recording."""
    path = tmp_path / "control.json"
    write_control(path, "stop")
    with pytest.raises(KeyboardInterrupt):
        try:
            await service_replay_control(path)
        except Exception:  # noqa: BLE001 - standing in for the replay path's own handlers
            raise AssertionError("stop was swallowed by an `except Exception` handler")


@pytest.mark.asyncio
async def test_replay_refuses_an_instruction_out_loud(tmp_path, caplog):
    """A replay has no LLM to steer. Dropping the instruction silently would let an operator
    believe they had redirected a run that carried on regardless."""
    import logging
    path = tmp_path / "control.json"
    write_control(path, "resume", instruction="click Cancel instead")
    with caplog.at_level(logging.WARNING, logger="framework.control"):
        await asyncio.wait_for(service_replay_control(path), timeout=5)
    assert "click Cancel instead" in caplog.text
    assert "replay" in caplog.text.lower()


@pytest.mark.asyncio
async def test_replay_with_no_command_is_a_no_op(tmp_path):
    await asyncio.wait_for(service_replay_control(tmp_path / "control.json"), timeout=5)


@pytest.mark.asyncio
async def test_the_replay_boundary_does_nothing_until_a_path_is_set(tmp_path):
    """Unit tests and any non-hybrid caller construct a SkillApi with no channel configured."""
    from automation.pipeline import control
    from automation.skills.api import SkillApi
    control.set_control_path(None)
    api = SkillApi(page=None, anchors={})
    await asyncio.wait_for(api.begin_optional(), timeout=5)


@pytest.mark.asyncio
async def test_a_replayed_verb_stops_at_its_boundary(tmp_path):
    """The integration: a stop queued while a recording is replaying aborts it at the next
    verb, with no agent anywhere in the picture."""
    from automation.pipeline import control
    from automation.skills.api import SkillApi
    path = tmp_path / "control.json"
    control.set_control_path(path)
    try:
        write_control(path, "stop")
        api = SkillApi(page=None, anchors={})
        with pytest.raises(KeyboardInterrupt):
            await api.begin_optional()
    finally:
        control.set_control_path(None)


def test_every_replay_verb_passes_through_the_control_boundary():
    """A verb added later must not silently miss the boundary — that is how a channel
    quietly stops covering half the replay surface."""
    import inspect as _inspect
    from automation.skills.api import SkillApi
    missing = [
        name for name, fn in _inspect.getmembers(SkillApi, _inspect.iscoroutinefunction)
        if not name.startswith("_") and not getattr(fn, "__control_boundary__", False)
    ]
    assert missing == [], f"replay verbs with no control boundary: {missing}"


def test_the_hybrid_task_tells_the_replay_layer_where_the_channel_is():
    src = inspect.getsource(__import__("automation.pipeline.hybrid",
                                       fromlist=["x"]))
    assert "set_control_path(" in src


def test_a_runner_without_an_artifacts_dir_disables_the_channel_instead_of_raising():
    """The channel is a convenience; it must never be able to fail a task. 70 hybrid tests
    drive run_hybrid_task with a SimpleNamespace runner that has no config at all, and a
    real one could be misconfigured the same way. Disabled, explicitly — not left pointing
    at whatever path the previous task in this process set."""
    import types
    from automation.pipeline import control, hybrid
    control.set_control_path(Path("artifacts") / "control.json")
    try:
        hybrid._set_task_control_channel(types.SimpleNamespace())
        assert control.control_path() is None
    finally:
        control.set_control_path(None)


@pytest.mark.asyncio
async def test_a_held_replay_says_so_through_the_logger_not_print(tmp_path, caplog):
    """browser-use's own pause message is a `print`, and Python block-buffers stdout to a
    pipe — so when a run's output is redirected to a file (every background run, every UI)
    the confirmation never appears and the operator cannot tell a hold from a hang. That cost
    real time on 2026-09-10. Ours goes through the logger, which is unbuffered on stderr."""
    import logging
    path = tmp_path / "control.json"
    write_control(path, "pause")

    async def release():
        await asyncio.sleep(0.25)
        write_control(path, "resume")

    with caplog.at_level(logging.INFO, logger="framework.control"):
        await asyncio.wait_for(
            asyncio.gather(service_replay_control(path), release()), timeout=10)
    assert "paused" in caplog.text.lower()
    assert "resumed" in caplog.text.lower()
