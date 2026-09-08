"""hybrid engine tests: the subtask loop, gates, and library commit rules.

The browser-facing surface (HybridSession) is faked via an injected seam, while the loop,
the library commit rules (_author_segment), and the gate evaluation run for real against
tmp_path-monkeypatched stores."""
import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from automation.pipeline import hybrid
from automation.pipeline import subtask_store as ss
from automation.pipeline.decompose import Subtask
from automation.pipeline.hybrid import (Gate, Segment, _format_extracts, evaluate_gate,
                                        reauthor_match, run_hybrid_task, segment_gate)
from automation.pipeline.runner import RunResult
from automation.tasks import SubtaskDecl, TaskSpec


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(ss, "LIBRARY_MANIFEST", tmp_path / "library" / "manifest.json")
    monkeypatch.setattr(ss, "DECOMPOSITIONS_DIR", tmp_path / "decompositions")
    (tmp_path / "library").mkdir()
    return tmp_path


PROMPT = ("go to the section. add invoice for customer Suresh Gopi and click save")
SPEC = TaskSpec(
    key="k", prompt=PROMPT, marker="Invoices",
    subtasks=(
        SubtaskDecl(prompt="go to the section."),
        SubtaskDecl(prompt="add invoice for customer {{customer}} and click save",
                    values={"customer": "Suresh Gopi"}, marker="Invoices"),
    ),
)

# A minimal browser-use recording that compiles to one goto step.
FAKE_RECORDING = {"history": [{
    "state": {"url": "http://app/start", "interacted_element": []},
    "model_output": {"action": [{"navigate": {"url": "http://app/section"}}]},
    "result": [],
}]}


def _skeleton_result(ground_truth=None):
    return RunResult(
        task=PROMPT, run_id="test", artifacts_dir=".", is_done=True,
        is_successful=None, has_errors=False, final_result=None, urls=[], n_steps=0,
        duration_seconds=0.0, extracted_content=[], model_actions=[], errors=[],
        ground_truth=ground_truth,
    )


class FakeSession:
    """Scripted HybridSession: `replays` / `agents` queue the Segment each call returns.
    An agent call that receives a record_path writes a compilable fake recording there,
    so the real _author_segment commit path runs end-to-end. `findings_seen` records the
    findings list each agent call received (the carry-over channel under test)."""

    def __init__(self, runner, *, replays=None, agents=None, create_write_seen=True,
                 recording=None, run_dir=None, probes=None):
        self.runner = runner
        self.replays = list(replays or [])
        self.agents = list(agents or [])
        self.probes = list(probes or [])   # scripted probe_condition verdicts, in order
        self.create_write_seen = create_write_seen
        self.recording = recording or FAKE_RECORDING
        # Progress-artifact surface (_write_progress reads these at every segment
        # boundary). Defaults to a throwaway dir so tests that don't inspect
        # progress.json never write into the repo CWD.
        self.run_id = "test"
        self.run_dir = Path(run_dir) if run_dir else Path(tempfile.mkdtemp(prefix="fake-run-"))
        self.started = datetime.now()
        self.collectors = []
        self.replay_calls = 0
        self.replay_branch = []      # `branch` flag each replay was given
        self.agent_calls = 0
        self.findings_seen = []
        self.remaining_seen = []      # still-ahead list each agent call received
        self.next_conditional_seen = []   # the successor-probe handoff each call received
        self.record_paths = []
        self.events = []              # ordered ("open", url)/("close",)/("replay",)/("agent",)
        self.aux_open_error = None    # set to make open_aux_tab raise
        self.window = []              # scripted network window (_refused_write_step reads it)

    def network_watermark(self):
        return 0

    def requests_since(self, _watermark):
        return list(self.window)

    @classmethod
    def make_opener(cls, instance):
        async def opener(runner):
            instance.runner = runner
            return instance
        return SimpleNamespace(open=opener)

    async def current_url(self):
        return "http://app/section"

    async def current_title(self):
        return ""

    async def open_aux_tab(self, url):
        if self.aux_open_error:
            raise RuntimeError(self.aux_open_error)
        self.events.append(("open", url))

    async def close_aux_tab(self):
        self.events.append(("close",))

    def network_watermark(self):
        """Mirrors HybridSession.network_watermark — how far the request log has got."""
        net = next((c for c in self.collectors
                    if getattr(c, "name", "") == "network"), None)
        return len(net.results().get("requests") or []) if net is not None else 0

    async def probe_condition(self, check):
        self.events.append(("probe", check.arg))
        return self.probes.pop(0)

    async def replay_segment(self, sub, sid, context, skill, gate, branch=False):
        self.events.append(("replay",))
        self.replay_calls += 1
        self.replay_branch.append(branch)
        seg = self.replays.pop(0)
        seg.index, seg.sid, seg.context = sub.index, sid, context
        seg.prompt = sub.instantiated_prompt
        # Mirror the real replay_segment, which stamps the node kind onto the segment it
        # builds — without this the stub reports every replay as an "action".
        seg.kind = getattr(sub, "kind", "action")
        return seg

    async def agent_segment(self, sub, sid, context, gate, *, completed, remaining,
                            dirty=False, prior_failure=None, record_path=None,
                            findings=None, next_conditional=None):
        self.events.append(("agent",))
        self.agent_calls += 1
        self.findings_seen.append(list(findings or []))
        self.remaining_seen.append(list(remaining or []))
        self.next_conditional_seen.append(next_conditional)
        self.record_paths.append(record_path)
        seg = self.agents.pop(0)
        seg.index, seg.sid, seg.context = sub.index, sid, context
        seg.prompt = sub.instantiated_prompt
        seg.mode = "authored"
        seg.kind = getattr(sub, "kind", "action")
        if record_path is not None:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps(self.recording))
        return seg

    async def finalize(self, task, parent_marker):
        gt = None
        if parent_marker:
            gt = {"marker": parent_marker, "create_write_seen": self.create_write_seen,
                  "write_step": None}
        return _skeleton_result(ground_truth=gt)


def _seg(ok, *, executed=0, error=None, mode="replay", write_step=None, finding=None,
         extracted=None):
    return Segment(index=0, sid="", prompt="", context="", mode=mode, ok=ok,
                   steps_executed=executed, error=error, write_step=write_step,
                   finding=finding, extracted=extracted)


def _runner():
    return SimpleNamespace(expander_llm=None,
                           config=SimpleNamespace(subtask_max_steps=25))


def _seed_entry(sid, steps=None):
    ss.steps_path(sid).parent.mkdir(parents=True, exist_ok=True)
    ss.steps_path(sid).write_text(json.dumps(steps or [{"action": "wait", "seconds": 1.0}]))
    ss.update_manifest(sid, "prompt", create=True, context="/x")


def _sid_for(sub_prompt, context="/section"):
    return ss.subtask_id(sub_prompt, context)


async def _run(fake, spec=SPEC, **kwargs):
    return await run_hybrid_task(fake.runner or _runner(), PROMPT, spec=spec,
                                 marker="Invoices", **kwargs)


# ------------------------------- loop behavior -------------------------------


async def test_all_segments_authored_then_committed(stores, monkeypatch):
    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored"),
                                          _seg(True, mode="authored")])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await _run(fake)

    assert fake.agent_calls == 2 and fake.replay_calls == 0
    assert result.mode == "hybrid"
    assert result.is_successful is True
    assert [s["mode"] for s in result.subtasks] == ["authored", "authored"]
    # Both segments were committed to the library (real _author_segment ran).
    assert len(ss.load_manifest()) == 2
    for sid, entry in ss.load_manifest().items():
        assert ss.has_script(sid)
        assert entry["context"] and entry["end_context"]
        # The committed body is the tier-1 code skill; the steps file is subsumed by it
        # and deleted (emit_start_goto=False: only the explicit navigate action compiles).
        assert ss.code_path(sid).exists() and ss.anchors_path(sid).exists()
        assert "await api.goto('http://app/section')" in ss.code_path(sid).read_text()
        assert not ss.steps_path(sid).exists()


async def test_library_hit_replays_without_agent(stores, monkeypatch):
    ctx = ss.normalize_context("http://app/section")
    # subtask_id normalizes whitespace/case internally, so seeding with the raw declared
    # prompt lands on the same sid the loop computes.
    for decl in SPEC.subtasks:
        _seed_entry(ss.subtask_id(decl.prompt, ctx))

    fake = FakeSession(_runner(), replays=[_seg(True), _seg(True)])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await _run(fake)

    assert fake.replay_calls == 2 and fake.agent_calls == 0
    assert result.is_successful is True
    assert [s["mode"] for s in result.subtasks] == ["replay", "replay"]
    # uses bumped, consecutive failures reset.
    sid = ss.subtask_id(SPEC.subtasks[0].prompt, ctx)
    assert ss.load_meta(sid)["uses"] == 1


async def test_replay_fail_dirty_recovery_does_not_commit(stores, monkeypatch):
    ctx = ss.normalize_context("http://app/section")
    sid0 = ss.subtask_id(SPEC.subtasks[0].prompt, ctx)
    sid1 = ss.subtask_id(SPEC.subtasks[1].prompt, ctx)
    original = [{"action": "click", "selectors": ["text=Old"]}]
    _seed_entry(sid0, original)
    _seed_entry(sid1)

    # First replay fails MID-SEGMENT (dirty); the agent recovers in place. Second replays.
    fake = FakeSession(_runner(), replays=[_seg(False, executed=2, error="boom"),
                                           _seg(True)],
                       agents=[_seg(True, mode="authored")])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await _run(fake)

    assert result.is_successful is True
    assert result.subtasks[0]["mode"] == "replay_failed->authored"
    # Dirty recovery: fail_count bumped, entry NOT archived (threshold 2), steps unchanged.
    assert ss.load_meta(sid0)["fail_count"] == 1
    assert json.loads(ss.steps_path(sid0).read_text()) == original
    assert not (ss.LIBRARY_DIR / "archive").exists()


async def test_a_failed_dirty_recovery_still_retires_a_failing_entry(stores, monkeypatch):
    """Run 20260907_152222 / library/50410b61b2e42519: an entry that cannot pass and
    cannot retire.

    Its committed end_title pinned the title of the page the segment had already LEFT
    (see _pin_end_title), so the replay failed on the pin — and the takeover agent is
    judged by the SAME gate object, so it failed on the pin too. `_author_segment` returns
    at `if not seg.ok:`, above the dirty-path archive, and the call site skips archiving
    whenever `dirty`. Result: fail_count 4 against a threshold of 2, no `uses` key, and
    five consecutive runs dead at subtask 0.

    A wrong pin from ANY cause must be able to age out. Consecutive failures already reset
    on any success (bump_meta), so a healthy entry is untouched.
    """
    ctx = ss.normalize_context("http://app/section")
    sid0 = ss.subtask_id(SPEC.subtasks[0].prompt, ctx)
    _seed_entry(sid0)
    ss.bump_meta(sid0, fail_count=1)          # one failure already on the record

    # Dirty replay failure -> takeover -> the takeover fails on the same bad gate.
    fake = FakeSession(_runner(), replays=[_seg(False, executed=2, error="bad pin")],
                       agents=[_seg(False, mode="authored", error="bad pin")])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await _run(fake)

    assert result.subtasks[0]["mode"] == "replay_failed->authored"
    assert sid0 not in ss.load_manifest()
    assert list((ss.LIBRARY_DIR / "archive").glob(f"{sid0}.steps.*.json"))


async def test_a_single_failed_dirty_recovery_does_not_retire(stores, monkeypatch):
    """The threshold still binds: one bad run must not delete a recording that worked
    yesterday — the same reasoning as test_replay_fail_at_step_zero_is_clean_reauthor."""
    ctx = ss.normalize_context("http://app/section")
    sid0 = ss.subtask_id(SPEC.subtasks[0].prompt, ctx)
    _seed_entry(sid0)

    fake = FakeSession(_runner(), replays=[_seg(False, executed=2, error="boom")],
                       agents=[_seg(False, mode="authored", error="boom")])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    await _run(fake)

    assert ss.load_meta(sid0)["fail_count"] == 1
    assert sid0 in ss.load_manifest()
    assert not (ss.LIBRARY_DIR / "archive").exists()


async def test_replay_fail_at_step_zero_is_clean_reauthor(stores, monkeypatch):
    """A clean (page-untouched) failure re-authors in place and REPLACES the entry —
    but it does not retire the old one on a single miss. One transient failure (a slow
    render, a cookie banner, a list that had not populated yet) used to delete the
    recording outright, so a recording that worked yesterday was gone today and every
    later run paid for the LLM again."""
    ctx = ss.normalize_context("http://app/section")
    sid0 = ss.subtask_id(SPEC.subtasks[0].prompt, ctx)
    _seed_entry(sid0, [{"action": "click", "selectors": ["text=Gone"]}])

    fake = FakeSession(_runner(), replays=[_seg(False, executed=0, error="no selector")],
                       agents=[_seg(True, mode="authored"), _seg(True, mode="authored")])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await _run(fake)

    assert result.is_successful is True
    # Not archived on the first miss — the fresh authoring simply replaces it.
    assert not list((ss.LIBRARY_DIR / "archive").glob(f"{sid0}.steps.*.json"))
    assert "await api.goto('http://app/section')" in ss.code_path(sid0).read_text()
    assert not ss.steps_path(sid0).exists()


async def test_entry_is_retired_only_after_repeated_clean_failures(stores, monkeypatch):
    """Two CONSECUTIVE clean failures do retire it (bump_meta resets the count on any
    success), so a genuinely dead recording still gets replaced."""
    ctx = ss.normalize_context("http://app/section")
    sid0 = ss.subtask_id(SPEC.subtasks[0].prompt, ctx)
    _seed_entry(sid0, [{"action": "click", "selectors": ["text=Gone"]}])
    ss.bump_meta(sid0, fail_count=1)          # yesterday's miss

    fake = FakeSession(_runner(), replays=[_seg(False, executed=0, error="no selector")],
                       agents=[_seg(True, mode="authored"), _seg(True, mode="authored")])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    await _run(fake)

    assert list((ss.LIBRARY_DIR / "archive").glob(f"{sid0}.steps.*.json"))


# ------------------------------- judge nodes + findings (Phase 0) -------------------------------


async def test_judge_node_never_replays_never_commits(stores, monkeypatch):
    """A DECLARED verification subtask must run live even when a (legacy, hollow) library
    entry exists for it, and nothing of it may be recorded or committed.

    `kind: judge` is the declaration — since 2026-08-28 the wording is not read at all, so
    this slice would be an ordinary recorded action without it."""
    prompt = "go to the section. verify the CC field matches the noted mail"
    spec = TaskSpec(key="j", prompt=prompt, subtasks=(
        SubtaskDecl(prompt="go to the section."),
        SubtaskDecl(prompt="verify the CC field matches the noted mail", kind="judge"),
    ))
    ctx = ss.normalize_context("http://app/section")
    judge_sid = ss.subtask_id(spec.subtasks[1].prompt, ctx)
    hollow = [{"action": "wait", "seconds": 1.0}]
    _seed_entry(judge_sid, hollow)   # a hollow replay would fake-pass; must be ignored

    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored"),
                                          _seg(True, mode="authored")])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await run_hybrid_task(fake.runner or _runner(), prompt, spec=spec)

    assert result.is_successful is True
    assert fake.replay_calls == 0 and fake.agent_calls == 2
    assert [s["kind"] for s in result.subtasks] == ["action", "judge"]
    assert result.subtasks[1]["skip_reason"] == "judge"
    # The judge agent ran WITHOUT a recording path, and the hollow entry was not replaced.
    assert fake.record_paths[1] is None
    assert not ss.recording_path(judge_sid).exists()
    assert json.loads(ss.steps_path(judge_sid).read_text()) == hollow








async def test_repin_after_dead_main_page_realigns_recording_focus():
    """When the pinned main page dies and current_page() re-pins a survivor, the
    browser-use focus event must follow: the RecordingWatchdog streams ONE CDP session
    and silently drops frames from every other, so a silent re-pin used to freeze the
    run video for the rest of the run while the run itself carried on."""
    hs = hybrid.HybridSession.__new__(hybrid.HybridSession)
    dead = SimpleNamespace(url="http://app/x", is_closed=lambda: True)
    alive = SimpleNamespace(url="http://app/y", is_closed=lambda: False)
    hs._aux_page = None
    hs._main_page = dead
    hs.pw_browser = SimpleNamespace(contexts=[SimpleNamespace(pages=[alive])])
    focused = []

    async def fake_focus(page):
        focused.append(page)

    hs._focus_browser_use = fake_focus
    assert hs.current_page() is alive
    await asyncio.sleep(0)              # let the fire-and-forget focus task run
    assert focused == [alive]


_PROBE_COND_LINE = ("If a popup appears at any point after clicking Save & Next, tick "
                    "'Don't show this again', click Process, and continue.")


def _probe_spec():
    from automation.pipeline.checks import Check
    prompt = "go to the section. " + _PROBE_COND_LINE
    spec = TaskSpec(key="p", prompt=prompt, subtasks=(
        SubtaskDecl(prompt="go to the section."),
        SubtaskDecl(prompt=_PROBE_COND_LINE,
                    probe=Check(kind="text_visible", arg="Don't show this again",
                                timeout_s=3.0)),
    ))
    return prompt, spec


async def test_conditional_probe_absent_resolves_without_agent_or_replay(stores,
                                                                         monkeypatch):
    """A FALSE probe resolves a probed conditional as a zero-LLM no-op — even when a
    library entry exists, nothing replays (the branch condition is not raised) and no
    agent runs."""
    prompt, spec = _probe_spec()
    ctx = ss.normalize_context("http://app/section")
    cond_sid = ss.subtask_id(_PROBE_COND_LINE, ctx)
    _seed_entry(cond_sid, [{"action": "click", "selector": "#process"}])

    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored"),
                                          _seg(True, mode="authored")],
                       probes=[False])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await run_hybrid_task(fake.runner or _runner(), prompt, spec=spec)

    assert result.is_successful is True
    assert fake.agent_calls == 1 and fake.replay_calls == 0   # nav slice only
    probe_seg = result.subtasks[1]
    assert probe_seg["ok"] is True
    assert probe_seg["mode"] == "probe"
    assert probe_seg["skip_reason"] == "probe_absent"
    assert probe_seg["steps_executed"] == 0
    assert "probe-skipped" in result.final_result


async def test_conditional_probe_present_replays_entry(stores, monkeypatch):
    """A TRUE probe lets a probed conditional replay its recorded branch like an
    action — the probe carries the presence judgment the agent used to make."""
    prompt, spec = _probe_spec()
    ctx = ss.normalize_context("http://app/section")
    cond_sid = ss.subtask_id(_PROBE_COND_LINE, ctx)
    _seed_entry(cond_sid, [{"action": "click", "selector": "#process"}])

    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored")],
                       replays=[_seg(True, executed=1)], probes=[True])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await run_hybrid_task(fake.runner or _runner(), prompt, spec=spec)

    assert result.is_successful is True
    assert fake.replay_calls == 1 and fake.agent_calls == 1
    assert result.subtasks[1]["mode"] == "replay"


async def test_conditional_probe_present_authors_and_commits(stores, monkeypatch):
    """A TRUE probe with a cold cache authors the branch WITH commit enabled — the
    recording is safe because it only ever replays behind a TRUE probe."""
    prompt, spec = _probe_spec()
    ctx = ss.normalize_context("http://app/section")
    cond_sid = ss.subtask_id(_PROBE_COND_LINE, ctx)

    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored"),
                                          _seg(True, mode="authored")],
                       probes=[True])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await run_hybrid_task(fake.runner or _runner(), prompt, spec=spec)

    assert result.is_successful is True
    assert fake.agent_calls == 2
    assert fake.record_paths[1] is not None      # commit path engaged for the guard
    assert ss.has_script(cond_sid)               # branch recording committed
    assert result.subtasks[1]["skip_reason"] == "no_entry"


async def test_routed_subtask_replays_canonical_entry(stores, monkeypatch):
    """A wording with no direct entry that the router resolves must replay the CANONICAL
    skill — and a routed failure must author under the ORIGINAL sid, never overwriting
    the canonical entry."""
    from automation.pipeline.router import Route

    ctx = ss.normalize_context("http://app/section")
    canon = ss.subtask_id("go to the section.", ctx)
    _seed_entry(canon)
    reworded = TaskSpec(
        key="r", prompt="open the section area. add invoice for customer Suresh Gopi "
                        "and click save",
        marker="Invoices",
        subtasks=(SubtaskDecl(prompt="open the section area."),
                  SubtaskDecl(prompt="add invoice for customer {{customer}} and click "
                                     "save", values={"customer": "Suresh Gopi"},
                              marker="Invoices")),
    )

    async def fake_route(sub, alias_sid, context, llm, **kw):
        if "open the section area" in sub.template_prompt:
            return Route(sid=canon, values={}, via="alias")
        return None

    monkeypatch.setattr(hybrid.router, "route", fake_route)
    fake = FakeSession(_runner(), replays=[_seg(True)],
                       agents=[_seg(True, mode="authored")])
    fake.runner = SimpleNamespace(
        expander_llm=None,
        config=SimpleNamespace(subtask_max_steps=25, semantic_router=True))
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await run_hybrid_task(fake.runner, reworded.prompt, spec=reworded,
                                   marker="Invoices")

    assert result.is_successful is True
    assert fake.replay_calls == 1                       # the routed wording REPLAYED
    assert result.subtasks[0]["sid"] == canon
    assert ss.load_meta(canon)["uses"] == 1


async def test_findings_carry_into_later_agent_segments(stores, monkeypatch):
    """A completed segment's finding (its distilled final result) must reach every later
    agent segment — the note-then-verify data channel."""
    fake = FakeSession(_runner(), agents=[
        _seg(True, mode="authored", finding="CC mail = billing@acme.com"),
        _seg(True, mode="authored"),
    ])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await _run(fake)

    assert result.is_successful is True
    assert fake.findings_seen[0] == []                    # nothing observed yet
    assert fake.findings_seen[1] == ["go to the section.: CC mail = billing@acme.com"]
    assert result.subtasks[0]["finding"] == "CC mail = billing@acme.com"


async def test_reauthor_forces_agent_only_for_named_subtask(stores, monkeypatch):
    """--reauthor: the named subtask goes back through the agent even though its library
    entry replays fine, and its entry is REPLACED by the fresh recording; the other
    subtask still replays untouched."""
    ctx = ss.normalize_context("http://app/section")
    sid0 = ss.subtask_id(SPEC.subtasks[0].prompt, ctx)
    sid1 = ss.subtask_id(SPEC.subtasks[1].prompt, ctx)
    stale = [{"action": "click", "selectors": ["text=Detour"]},
             {"action": "click", "selectors": ["text=Back"]}]
    _seed_entry(sid0, stale)
    _seed_entry(sid1)

    fake = FakeSession(_runner(), replays=[_seg(True)],
                       agents=[_seg(True, mode="authored")])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await _run(fake, reauthor="0")

    assert result.is_successful is True
    assert fake.agent_calls == 1 and fake.replay_calls == 1
    assert result.subtasks[0]["mode"] == "authored"
    assert result.subtasks[1]["mode"] == "replay"
    # The stale detour steps were replaced by the fresh recording's compiled code skill.
    assert "await api.goto('http://app/section')" in ss.code_path(sid0).read_text()
    assert not ss.steps_path(sid0).exists()


async def test_reauthor_failure_keeps_old_entry(stores, monkeypatch):
    """A failed re-author must NOT clobber the working library entry — and its recording
    is kept for diagnosis OFF the canonical path, so the canonical recording can never
    mismatch the committed skill."""
    ctx = ss.normalize_context("http://app/section")
    sid0 = ss.subtask_id(SPEC.subtasks[0].prompt, ctx)
    stale = [{"action": "click", "selectors": ["text=Works"]}]
    _seed_entry(sid0, stale)

    fake = FakeSession(_runner(), agents=[_seg(False, error="lost", mode="authored")])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await _run(fake, reauthor="go to the section")

    assert result.is_successful is False
    assert json.loads(ss.steps_path(sid0).read_text()) == stale
    assert not ss.recording_path(sid0).exists()
    failed = ss.recording_path(sid0).with_suffix(".failed.json")
    assert failed.exists()                       # the failed trace survives for diagnosis


def test_reauthor_match_indexes_and_substrings():
    sub = Subtask(index=3, template_prompt="add estimate, select customer {{customer}}",
                  values={"customer": "Mr Jones"})
    assert reauthor_match("3", sub)
    assert reauthor_match("0, 3", sub)
    assert reauthor_match("Add Estimate", sub)
    assert reauthor_match("mr jones", sub)          # instantiated wording matches too
    assert not reauthor_match("0,1", sub)
    assert not reauthor_match("go to invoices", sub)
    assert not reauthor_match(None, sub)
    assert not reauthor_match("", sub)


async def test_failed_segment_breaks_loop(stores, monkeypatch):
    fake = FakeSession(_runner(), agents=[_seg(False, error="could not", mode="authored"),
                                          _seg(True, mode="authored")])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await _run(fake)

    assert result.is_successful is False
    assert fake.agent_calls == 1          # the second subtask never ran
    assert len(result.subtasks) == 1
    assert result.subtasks[0]["ok"] is False


async def test_parent_marker_gate_still_required(stores, monkeypatch):
    """Every segment ok but no create-write in the whole run's network log -> FAIL."""
    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored"),
                                          _seg(True, mode="authored")],
                       create_write_seen=False)
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await _run(fake)

    assert all(s["ok"] for s in result.subtasks)
    assert result.is_successful is False


# ------------------------------- progress artifact -------------------------------


async def test_progress_json_written_and_finished(stores, monkeypatch):
    """A clean run leaves progress.json with status "finished" and every segment record."""
    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored"),
                                          _seg(True, mode="authored")],
                       run_dir=stores / "artifacts")
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await _run(fake)

    progress = json.loads((stores / "artifacts" / "progress.json").read_text())
    assert progress["status"] == "finished"
    assert progress["is_successful"] is True and result.is_successful is True
    assert progress["subtasks_total"] == 2
    assert [p["prompt"] for p in progress["planned"]] == [
        "go to the section.",
        "add invoice for customer {{customer}} and click save",
    ]
    assert len(progress["segments"]) == 2
    for seg in progress["segments"]:
        assert "gate" in seg and "skip_reason" in seg and seg["ok"] is True


async def test_progress_json_survives_mid_run_crash(stores, monkeypatch):
    """A crash mid-segment leaves progress.json holding every COMPLETED segment.

    Only one queued agent segment for two subtasks: the second agent call pops an empty
    list (IndexError), standing in for a real mid-segment crash. run_hybrid_task's result
    assembly never runs — but the boundary write after segment 0 is already on disk."""
    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored")],
                       run_dir=stores / "artifacts")
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    with pytest.raises(IndexError):
        await _run(fake)

    progress = json.loads((stores / "artifacts" / "progress.json").read_text())
    assert progress["status"] == "running"       # the final "finished" write never ran
    assert progress["subtasks_total"] == 2
    assert len(progress["segments"]) == 1        # segment 0 persisted before the crash
    assert progress["segments"][0]["ok"] is True
    assert progress["is_successful"] is None


async def test_ctrl_c_flushes_interrupted_progress(stores, monkeypatch):
    """A second Ctrl+C aborts via KeyboardInterrupt — four killed runs on 2026-08-05 left
    progress.json stuck at "running" with the in-flight segment unrecorded (video-only
    forensics). The abort path now stubs the in-flight segment, writes
    status="interrupted" (which also flushes collectors), and re-raises."""

    class _Interrupter(FakeSession):
        async def agent_segment(self, sub, *args, **kwargs):
            if self.agent_calls >= 1:
                raise KeyboardInterrupt
            return await super().agent_segment(sub, *args, **kwargs)

    fake = _Interrupter(_runner(), agents=[_seg(True, mode="authored")],
                        run_dir=stores / "artifacts")
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    with pytest.raises(KeyboardInterrupt):
        await _run(fake)

    progress = json.loads((stores / "artifacts" / "progress.json").read_text())
    assert progress["status"] == "interrupted"
    assert progress["is_successful"] is False
    assert len(progress["segments"]) == 2        # completed segment 0 + the stub
    stub = progress["segments"][1]
    assert stub["ok"] is False
    assert "interrupted" in (stub["error"] or "")


def test_interrupted_segment_saves_partial_history(tmp_path):
    """The runner leg of the Ctrl+C flush: the in-flight agent's history lands next to
    the recording path under a distinct .interrupted.json suffix (the promotion path
    must never adopt it), best-effort — a failing save returns None, never raises."""
    from automation.pipeline import runner as runner_mod

    class _FakeAgent:
        def save_history(self, path):
            Path(path).write_text(json.dumps({"history": []}))

    record = tmp_path / "sid.recording.new.json"
    saved = runner_mod._save_interrupted_history(_FakeAgent(), record)
    assert saved is not None and saved.exists()
    assert saved.name.endswith(".interrupted.json")
    assert not record.exists()                    # never the promotable temp name

    class _Boom:
        def save_history(self, path):
            raise RuntimeError("no history yet")

    assert runner_mod._save_interrupted_history(_Boom(), record) is None
    assert runner_mod._save_interrupted_history(_FakeAgent(), None) is None


# --------------------------- fallback-blob degradation ---------------------------
# Runs 20260805_155515/160602/161958: a whole-prompt fallback blob took the download
# gate (its wording contains "download" mid-task) and the loop kind, so the agent hunted
# a Download control from step 1 and once declared the mega-task done on the first
# repeat-until's stop condition. Fallback blobs degrade to the steps gate with a
# whole-task step budget instead.


def test_fallback_blob_gets_steps_gate():
    sub = Subtask(index=0, fallback=True,
                  template_prompt="do the flow then download the export and save")
    assert segment_gate(sub, None, "/x").kind == "steps"
    # The same wording WITHOUT the fallback flag keeps the authoritative download gate.
    normal = Subtask(index=0,
                     template_prompt="do the flow then download the export and save")
    assert segment_gate(normal, None, "/x").kind == "download"


def test_fallback_blob_gets_whole_task_step_budget():
    sub = Subtask(index=0, fallback=True, template_prompt="do everything")
    gate = segment_gate(sub, None, "/x")
    budget = hybrid.segment_step_budget(gate, 25, sub.kind, fallback=True)
    assert budget == 25 + hybrid._FALLBACK_EXTRA_STEPS
    assert hybrid.segment_step_budget(gate, 25, "action") == 25   # non-fallback unchanged


# ------------------------------- aux-tab subtasks -------------------------------


AUX_PROMPT = ("go to the section. search DuckDuckGo for Acting Office and note the title "
              "of the top result")
AUX_SPEC = TaskSpec(
    key="aux", prompt=AUX_PROMPT,
    subtasks=(
        SubtaskDecl(prompt="go to the section."),
        SubtaskDecl(prompt="search DuckDuckGo for {{query}} and note the title of the "
                           "top result", values={"query": "Acting Office"},
                    tab_url="https://duckduckgo.com"),
    ),
)
AUX_CTX = ss.normalize_aux_context("https://duckduckgo.com")


async def _run_aux(fake, **kwargs):
    return await run_hybrid_task(fake.runner or _runner(), AUX_PROMPT, spec=AUX_SPEC,
                                 **kwargs)


async def test_aux_subtask_opens_and_closes_tab_around_segment(stores, monkeypatch):
    """The LOOP owns the helper tab: opened before the segment, closed in a finally — and
    the plain subtask never touches tab machinery."""
    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored"),
                                          _seg(True, mode="authored")])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await _run_aux(fake)

    assert result.is_successful is True
    assert fake.events == [("agent",), ("open", "https://duckduckgo.com"),
                           ("agent",), ("close",)]
    # tab_url reclassifies the note-wording subtask as a cacheable ACTION node (the
    # replayed extract step re-reads the live DOM, so its replay is not hollow).
    assert [s["kind"] for s in result.subtasks] == ["action", "action"]


async def test_aux_tab_closes_even_when_the_segment_fails(stores, monkeypatch):
    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored"),
                                          _seg(False, error="lost", mode="authored")])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await _run_aux(fake)

    assert result.is_successful is False
    assert fake.events[-1] == ("close",)


async def test_aux_sid_keyed_on_tab_url_context_and_manifest_records_it(stores, monkeypatch):
    """Aux identity comes from the DECLARED tab URL (host-qualified), not the main page —
    so the same helper procedure is ONE library entry across every hosting task."""
    # The slice says "note the title of the top result", so its recording carries the
    # capture — without one the producer commit guard refuses it, and rightly: a replay
    # that notes nothing leaves its consumers replaying stale values.
    monkeypatch.setattr(hybrid, "save_steps", _fill_steps_stub([
        {"action": "goto", "url": "https://duckduckgo.com"},
        {"action": "extract", "label": "top_result_title",
         "selectors": ["xpath=/html/body/h3"]},
    ]))
    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored"),
                                          _seg(True, mode="authored")])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await _run_aux(fake)

    aux_sid = ss.subtask_id(AUX_SPEC.subtasks[1].prompt, AUX_CTX)
    assert AUX_CTX == "duckduckgo.com/"
    assert result.subtasks[1]["sid"] == aux_sid
    assert result.subtasks[1]["context"] == AUX_CTX
    entry = ss.load_manifest()[aux_sid]
    assert entry["tab_url"] == "https://duckduckgo.com"
    assert entry["context"] == AUX_CTX


async def test_aux_replay_failure_reensures_tab_for_the_recovering_agent(stores, monkeypatch):
    ctx0 = ss.normalize_context("http://app/section")
    _seed_entry(ss.subtask_id(AUX_SPEC.subtasks[0].prompt, ctx0))
    _seed_entry(ss.subtask_id(AUX_SPEC.subtasks[1].prompt, AUX_CTX))

    fake = FakeSession(_runner(),
                       replays=[_seg(True), _seg(False, executed=2, error="boom")],
                       agents=[_seg(True, mode="authored")])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await _run_aux(fake)

    assert result.is_successful is True
    assert result.subtasks[1]["mode"] == "replay_failed->authored"
    # open -> replay fails -> defensive re-ensure (no-op on a live tab, recreate after a
    # crash) -> recovery agent on the SAME dirty tab -> guaranteed close.
    assert fake.events == [("replay",), ("open", "https://duckduckgo.com"), ("replay",),
                           ("open", "https://duckduckgo.com"), ("agent",), ("close",)]


async def test_aux_open_failure_fails_the_subtask_and_stops(stores, monkeypatch):
    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored")])
    fake.aux_open_error = "net::ERR_NAME_NOT_RESOLVED"
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await _run_aux(fake)

    assert result.is_successful is False
    assert len(result.subtasks) == 2
    assert result.subtasks[1]["ok"] is False
    assert "could not open helper tab" in result.subtasks[1]["error"]


async def test_router_skipped_for_aux_subtasks(stores, monkeypatch):
    routed = []

    async def fake_route(sub, alias_sid, context, llm, **kw):
        routed.append(sub.template_prompt)
        return None

    monkeypatch.setattr(hybrid.router, "route", fake_route)
    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored"),
                                          _seg(True, mode="authored")])
    fake.runner = SimpleNamespace(
        expander_llm=None,
        config=SimpleNamespace(subtask_max_steps=25, semantic_router=True))
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await _run_aux(fake)

    assert result.is_successful is True
    # Only the plain subtask consulted the router; aux entries are direct-hit only (a
    # routed canonical could have been recorded on a different site family).
    assert routed == ["go to the section."]


async def test_replay_finding_carries_into_later_agent_segments(stores, monkeypatch):
    """A REPLAYED segment's finding (its fresh extraction) must reach later agent
    segments exactly like an authored segment's observation does."""
    ctx0 = ss.normalize_context("http://app/section")
    _seed_entry(ss.subtask_id(SPEC.subtasks[0].prompt, ctx0))

    fake = FakeSession(_runner(),
                       replays=[_seg(True, finding="top_result_title = Fresh Value")],
                       agents=[_seg(True, mode="authored")])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await _run(fake)

    assert result.is_successful is True
    assert fake.findings_seen == [["go to the section.: top_result_title = Fresh Value"]]


# ------------------------------- noted-data consumers -------------------------------


NOTED_PROMPT = ("open the generator and note the generated identity. "
                "add employee using the noted generated name and save")
NOTED_SPEC = TaskSpec(
    key="noted", prompt=NOTED_PROMPT,
    subtasks=(
        SubtaskDecl(prompt="open the generator and note the generated identity."),
        SubtaskDecl(prompt="add employee using the noted generated name and save"),
    ),
)




async def test_noted_consumer_replays_when_nothing_was_noted_this_run(stores, monkeypatch):
    """The dynamic gate is wording AND observations: with no upstream findings there is
    nothing for the cached values to be stale against (and nothing the agent could
    substitute either), so the zero-LLM replay stays."""
    prompt = "go to the section. add employee using the noted generated name and save"
    spec = TaskSpec(key="n2", prompt=prompt, subtasks=(
        SubtaskDecl(prompt="go to the section."),
        SubtaskDecl(prompt="add employee using the noted generated name and save"),
    ))
    ctx = ss.normalize_context("http://app/section")
    consumer_sid = ss.subtask_id(spec.subtasks[1].prompt, ctx)
    _seed_entry(ss.subtask_id(spec.subtasks[0].prompt, ctx))
    _seed_entry(consumer_sid)

    fake = FakeSession(_runner(), replays=[_seg(True), _seg(True)])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await run_hybrid_task(fake.runner or _runner(), prompt, spec=spec)

    assert result.is_successful is True
    assert fake.replay_calls == 2 and fake.agent_calls == 0
    assert ss.has_script(consumer_sid)
    assert consumer_sid in ss.load_manifest()


BOUND_PROMPT = ("load the identity page. "
                "add employee using the noted generated name and save")
BOUND_SPEC = TaskSpec(key="b1", prompt=BOUND_PROMPT, subtasks=(
    SubtaskDecl(prompt="load the identity page."),
    SubtaskDecl(prompt="add employee using the noted generated name and save"),
))


def _fill_steps_stub(steps):
    """A hybrid.save_steps stand-in: pretend the recording compiled to `steps`."""
    def stub(rec, steps_path, max_steps=None, emit_start_goto=False, repeat_hint=None,
             loop=False, optional_from=None):
        steps_path.parent.mkdir(parents=True, exist_ok=True)
        steps_path.write_text(json.dumps(steps))
        return steps
    return stub




async def test_pasted_otp_binds_where_six_one_character_fills_could_not(
        stores, monkeypatch):
    """Run 20260825_090938, subtask 6e8c9bb7ee56a6aa: the OTP dialog has six 1-character
    boxes, so the recording typed six 1-character fills and the code the run actually used
    never appeared as a step value. Nothing was flagged (the findings match on whole
    tokens; the length floors skip 1-char values), `runtime_values` came out empty, and the
    commit was refused — 107k tokens and 143s of re-authoring every run.

    Delivered as ONE paste, the same segment binds through the machinery that already
    exists, INCLUDING the line slice: extract_data captured the OTP with the whole employee
    panel stuck to it, so the code is line 0 of a multi-line source."""
    ctx = ss.normalize_context("http://app/section")
    consumer_sid = ss.subtask_id(BOUND_SPEC.subtasks[1].prompt, ctx)
    otp_block = "502956\nSelect Employee\nAarmaan Aman\nGender\nMale"

    # What the OTP segment records once it pastes instead of typing box by box.
    monkeypatch.setattr(hybrid, "save_steps", _fill_steps_stub([
        {"action": "paste", "selectors": ['css=[aria-label="Please enter OTP character 1"]'],
         "value": "502956"},
        {"action": "click", "selectors": ['text="Proceed Securely"']},
    ]))
    fake = FakeSession(_runner(), agents=[
        _seg(True, mode="authored", finding="otp = 502956",
             extracted={"otp": otp_block}),
        _seg(True, mode="authored"),
    ])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await run_hybrid_task(fake.runner or _runner(), BOUND_PROMPT, spec=BOUND_SPEC)

    assert result.is_successful is True
    entry = ss.load_manifest()[consumer_sid]
    assert entry["bindings"] == {
        "bound_1": {"kind": "extract", "label": "otp",
                    "transform": {"line": {"index": 0, "count": 1, "join": " "}}}}
    tmpl = json.loads(ss.template_path(consumer_sid).read_text())
    assert tmpl["steps"][0]["value"] == "{{bound_1}}"

    # Next run: a FRESH code, resolved from this run's own capture — never the authoring
    # run's 502956, which by then the app has long since invalidated.
    fake2 = FakeSession(_runner(), replays=[
        _seg(True, finding="otp = 771403",
             extracted={"otp": "771403\nSelect Employee\nAarmaan Aman"}),
        _seg(True),
    ])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake2))
    result2 = await run_hybrid_task(fake2.runner or _runner(), BOUND_PROMPT, spec=BOUND_SPEC)

    assert result2.is_successful is True
    assert fake2.replay_calls == 2 and fake2.agent_calls == 0
    assert result2.subtasks[1]["mode"] == "replay"








async def test_identity_fork_detected_when_wording_recorded_elsewhere(
        stores, monkeypatch):
    """Same wording committed under a DIFFERENT start context is invisible to the direct
    lookup (sids key wording + context). The miss is no longer silent: the subtask row
    and the summary name the fork instead of a bare cache miss."""
    other_sid = ss.subtask_id(SPEC.subtasks[0].prompt, "/other")
    ss.steps_path(other_sid).parent.mkdir(parents=True, exist_ok=True)
    ss.steps_path(other_sid).write_text(
        json.dumps([{"action": "wait", "seconds": 1.0}]))
    ss.update_manifest(other_sid, SPEC.subtasks[0].prompt, create=True, context="/other",
                       start_url="http://app/other")

    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored"),
                                          _seg(True, mode="authored")])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await _run(fake)

    assert result.subtasks[0]["skip_reason"] == "identity_fork"
    assert result.subtasks[1]["skip_reason"] == "no_entry"
    assert "1 identity_fork" in result.final_result
    assert "1 no_entry" in result.final_result


async def test_fresh_reauthors_with_a_visible_reason(stores, monkeypatch):
    """--fresh bypassing a library hit used to print NOTHING — indistinguishable from a
    cache miss (the live 2026-07-29 'recordings are never used' mystery). It now carries
    a reason end to end: per-subtask row and the summary breakdown."""
    _seed_entry(_sid_for(SPEC.subtasks[0].prompt))
    _seed_entry(_sid_for(SPEC.subtasks[1].prompt))
    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored"),
                                          _seg(True, mode="authored")])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await _run(fake, fresh=True)

    assert fake.replay_calls == 0 and fake.agent_calls == 2
    assert [s["skip_reason"] for s in result.subtasks] == ["fresh", "fresh"]
    assert "(0 replayed, 2 authored: 2 fresh)" in result.final_result


async def test_failed_reauthor_no_longer_destroys_the_canonical_recording(
        stores, monkeypatch):
    """A failed re-authoring records to the temp path and its wreck is set aside as
    .failed.json — the canonical recording stays byte-identical to the committed skill
    (the live 2026-07-29 Dec-26 segment recording loss)."""
    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored"),
                                          _seg(True, mode="authored")])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    assert (await _run(fake)).is_successful is True
    sid0 = _sid_for(SPEC.subtasks[0].prompt)
    good = ss.recording_path(sid0).read_text()

    wreck = {"history": [{
        "state": {"url": "http://app/start", "interacted_element": []},
        "model_output": {"action": [{"navigate": {"url": "http://app/broken"}}]},
        "result": [],
    }]}
    fake2 = FakeSession(_runner(), agents=[_seg(False, mode="authored", error="boom")],
                        recording=wreck)
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake2))
    result2 = await _run(fake2, fresh=True)

    assert result2.is_successful is False
    assert ss.recording_path(sid0).read_text() == good           # canonical intact
    failed = ss.recording_path(sid0).with_suffix(".failed.json")
    assert json.loads(failed.read_text()) == wreck               # wreck set aside
    assert ss.has_script(sid0)                                   # entry still replayable


def test_findings_sourced_values_flags_runtime_data_only():
    """The provenance rule that replaces wording-guessing: values traceable to the RUN'S
    FINDINGS (not the prompt) mark a segment as runtime-consuming; prompt values and
    agent-invented incidentals do not."""
    from automation.pipeline.hybrid import _findings_sourced_values

    prompt = ("Add Employee using the noted generated name, join date 06/04/2026, "
              "NI category A, and save")
    findings = ["...: Employee Alistair Allan has been successfully added, "
                "DOB 21/06/1982"]
    steps = [
        {"action": "fill", "selectors": ["css=#first"], "value": "Alistair"},   # finding
        {"action": "fill", "selectors": ["css=#join"], "value": "06/04/2026"},  # prompt
        {"action": "fill", "selectors": ["css=#county"], "value": "Lanarkshire"},  # invented
        {"action": "click", "selectors": ['text="Mr"',                          # invented pick
                                          'css=[id="react-select-1-option-0"]']},
        {"action": "click", "selectors": ['text="Alistair Allan"',              # finding pick
                                          'css=[id="react-select-2-option-3"]']},
        {"action": "click", "selectors": ['role=link[name="Employees"]']},      # not a pick
    ]
    assert _findings_sourced_values(steps, prompt, findings) == [
        "Alistair", "Alistair Allan"]
    # No findings this run -> nothing can be runtime-sourced.
    assert _findings_sourced_values(steps, prompt, []) == []


def test_findings_sourced_values_catches_clicked_run_created_identifiers():
    """The live 2026-07-24 escape: subtask 6 ('click on the same ref. no. ...') was
    committed with the ref the AUTHORING run created — find_click('PR/.../DR017') plus a
    role=button[name=...] click — because the guard only scanned fills and option picks.
    Clicked-element NAMES sourced from the run's findings are runtime data exactly like
    typed values: the replay clicked the PREVIOUS run's real row."""
    from automation.pipeline.hybrid import _findings_sourced_values

    prompt = ("now click on the same ref. no. of the same Payroll data request, "
              "select employee and click verify")
    findings = [
        "Go to data request, click add request: request created",
        "Then click on status sent on the top sent request, select st: The status of "
        "request PR/01797494/27/DR017 is now set to Submitted with note "
        "“well done” and the changes have been saved.",
    ]
    steps = [
        {"action": "find_click", "text": "PR/01797494/27/DR017"},
        {"action": "click",
         "selectors": ['role=button[name="PR/01797494/27/DR017"]',
                       'text="PR/01797494/27/DR017"', "xpath=/html/body/div[1]/button"]},
        {"action": "click", "selectors": ["xpath=/html/body/div[2]/div"],
         "expect_text": "PR/01797494/27/DR017"},          # name via the landed guard only
        {"action": "click", "selectors": ['role=button[name="Verify"]']},  # prompt word
        {"action": "click", "selectors": ['text="OK"']},   # tiny name: chance collision
    ]
    assert _findings_sourced_values(steps, prompt, findings) == [
        "PR/01797494/27/DR017"]

    # A clicked name that matches only a PRIOR SUBTASK'S WORDING (the "prompt[:80]: "
    # prefix, e.g. a nav menu named like the step that used it) is not runtime data —
    # click names check the finding BODIES only. Typed values still check everything.
    nav = [{"action": "click", "selectors": ['role=menu[name="Data Request"]']}]
    assert _findings_sourced_values(nav, "open the request list", findings) == []
    typed = [{"action": "fill", "selectors": ["css=#x"], "value": "status sent"}]
    assert _findings_sourced_values(typed, "open the request list", findings) == [
        "status sent"]


async def test_commit_guard_blocks_findings_sourced_segments(stores, monkeypatch, capsys):
    """A passed segment whose recording picked a findings-sourced value must NOT be
    committed — the wording-free version of the noted-data consumer rule (the live case:
    an employee-name pick got parameterized to the word 'download')."""
    prompt = "go to the section. select the employee and save the request"
    spec = TaskSpec(key="pv", prompt=prompt, subtasks=(
        SubtaskDecl(prompt="go to the section."),
        SubtaskDecl(prompt="select the employee and save the request"),
    ))
    # Subtask 1's recording clicks a react-select option named from subtask 0's finding.
    pick_recording = {"history": [{
        "state": {"url": "http://app/section", "interacted_element": [
            {"node_name": "div", "ax_name": "Alistair Allan",
             "attributes": {"id": "react-select-9-option-2"},
             "x_path": "//div[@id='react-select-9-option-2']"}]},
        "model_output": {"action": [{"click": {"index": 5}}]},
        "result": [],
    }]}

    fake = FakeSession(_runner(), agents=[
        _seg(True, mode="authored", finding="added employee Alistair Allan"),
        _seg(True, mode="authored"),
    ])
    fake.recording = pick_recording
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await run_hybrid_task(fake.runner or _runner(), prompt, spec=spec)

    assert result.is_successful is True
    ctx = ss.normalize_context("http://app/section")
    consumer_sid = ss.subtask_id(spec.subtasks[1].prompt, ctx)
    # Segment passed but was NOT cached: no steps, no manifest entry — and for the
    # right reason (the provenance guard, not a zero-step compile).
    assert not ss.has_script(consumer_sid)
    assert consumer_sid not in ss.load_manifest()
    out = capsys.readouterr().out
    assert "used runtime data from earlier steps (Alistair Allan)" in out


def test_history_extracts_label_collision_keeps_both_values():
    """An authored aux run that labels two different facts identically must not lose
    the first one — Segment.extracted feeds findings AND the binder's sources."""
    from types import SimpleNamespace

    from automation.pipeline.hybrid import _history_extracts

    def _res(label, value):
        return SimpleNamespace(metadata={"extract": {"label": label, "value": value}})

    history = SimpleNamespace(history=[
        SimpleNamespace(result=[_res("identity_block", "Haiden Christie")]),
        SimpleNamespace(result=[_res("identity_block", "28 Caerfai Bay Road")]),
        SimpleNamespace(result=[_res("identity_block", "Haiden Christie")]),  # retry
        SimpleNamespace(result=[_res("date_of_birth", "April 23, 1963")]),
    ])
    assert _history_extracts(history) == {
        "identity_block": "Haiden Christie",
        "identity_block_2": "28 Caerfai Bay Road",
        "date_of_birth": "April 23, 1963",
    }


def test_format_extracts_collapses_duplicate_blob_values():
    """Several labels resolving to the SAME re-read DOM text (the address-blob case) must
    fold into one entry — repeating the blob per label burns the findings budget and
    presents it as a real per-label split."""
    blob = "75 Monks Way TOMNAVEN AB54 1LP"
    out = _format_extracts({"generated_name": "Kerris McKay", "street_address": blob,
                            "city": blob, "postcode": blob})
    assert out == ("generated_name = Kerris McKay; "
                   f"street_address / city / postcode = {blob}")
    assert _format_extracts({}) == ""


def test_downloads_watermark_windows_segment_downloads():
    """Downloads persist on the SESSION across segments; the watermark pair must yield
    only the files a given segment triggered, as basenames."""
    hs = hybrid.HybridSession(_runner())
    hs.session = SimpleNamespace(downloaded_files=["/tmp/a/earlier.pdf"])
    mark = hs.downloads_watermark()
    assert mark == 1
    hs.session.downloaded_files = ["/tmp/a/earlier.pdf",
                                   "/tmp/a/Forecast report_27.pdf"]
    assert hs.downloads_since(mark) == ["Forecast report_27.pdf"]
    hs.session = None
    assert hs.downloads_watermark() == 0 and hs.downloads_since(0) == []


def test_segment_step_budget_marker_headroom():
    """The save-owning segment gets fix-and-resave headroom; every other gate kind runs
    on the flat configured budget."""
    from automation.pipeline.hybrid import segment_step_budget

    assert segment_step_budget(Gate(kind="marker", marker="Employees"), 25) == 35
    assert segment_step_budget(Gate(kind="steps"), 25) == 25
    assert segment_step_budget(Gate(kind="postcondition",
                                    postcondition={"url_contains": "x"}), 25) == 25


def test_step_budget_has_no_loop_headroom_any_more():
    """The +120 loop budget existed because a 12-iteration slice cost 12 agent steps. One
    repeat_click call costs one, so the headroom went with the `loop` kind (2026-08-28) —
    and the ceiling is a runaway guard again for every segment."""
    assert hybrid.segment_step_budget(Gate(kind="steps"), 25) == 25
    assert hybrid.segment_step_budget(Gate(kind="steps"), 25, "judge") == 25
    # A marker gate still gets its own headroom; that is measured, not wording-derived.
    assert hybrid.segment_step_budget(Gate(kind="marker", marker="/Invoices"), 25) > 25


# ------------------------------- unit: gates -------------------------------


async def test_marker_gate_windows_requests():
    gate = Gate(kind="marker", marker="Invoices")
    hit = [{"method": "POST", "url": "http://api/Invoices/create", "status": 201, "step": 7}]
    ok, detail = await evaluate_gate(gate, steps_ok=False, page=None, requests_window=hit)
    assert ok is True and detail["write_step"] == 7

    miss = [{"method": "GET", "url": "http://api/Invoices", "status": 200},
            {"method": "POST", "url": "http://api/Other", "status": 201}]
    ok, _ = await evaluate_gate(gate, steps_ok=True, page=None, requests_window=miss)
    assert ok is False


async def test_steps_gate_is_passthrough():
    ok, detail = await evaluate_gate(Gate(kind="steps"), steps_ok=True, page=None,
                                     requests_window=[])
    assert ok is True and detail == {"kind": "steps"}
    ok, _ = await evaluate_gate(Gate(kind="steps"), steps_ok=False, page=None,
                                requests_window=[])
    assert ok is False


async def test_steps_gate_demoted_by_refused_only_window():
    """The generic write rule: fired-but-never-accepted business writes fail any
    segment — no declared verify needed (the ad-hoc-prompt case)."""
    window = [{"method": "POST", "url": "http://api/Years/27/FPS", "status": 200,
               "body": '{"status": false, "message": "already submitted"}'}]
    ok, detail = await evaluate_gate(Gate(kind="steps"), steps_ok=True, page=None,
                                     requests_window=window)
    assert ok is False
    assert "already submitted" in detail["write_rollup"][0]
    from automation.pipeline.hybrid import _check_failure_reason
    assert "already submitted" in _check_failure_reason(detail)


async def test_steps_gate_refusal_waived_by_accepted_write():
    window = [{"method": "POST", "url": "http://api/Years/27/FPS", "status": 200},
              {"method": "POST", "url": "http://api/Years/27/FPS", "status": 200,
               "body": '{"status": false, "message": "already submitted"}'}]
    ok, detail = await evaluate_gate(Gate(kind="steps"), steps_ok=True, page=None,
                                     requests_window=window)
    assert ok is True and "write_rollup" not in detail


async def test_write_warning_on_save_cue_with_no_writes():
    ok, detail = await evaluate_gate(Gate(kind="steps"), steps_ok=True, page=None,
                                     requests_window=[],
                                     prompt_text="set Cost to 200 and click Save")
    assert ok is True
    assert "write_warning" in detail
    ok, detail = await evaluate_gate(Gate(kind="steps"), steps_ok=True, page=None,
                                     requests_window=[],
                                     prompt_text="go to Pay Forecast and read the rows")
    assert ok is True and "write_warning" not in detail


async def test_marker_gate_pass_untouched_by_write_rule():
    gate = Gate(kind="marker", marker="Invoices")
    hit = [{"method": "POST", "url": "http://api/Invoices/create", "status": 201}]
    ok, detail = await evaluate_gate(gate, steps_ok=False, page=None,
                                     requests_window=hit)
    assert ok is True and "write_rollup" not in detail


async def test_url_contains_postcondition(monkeypatch):
    monkeypatch.setattr(hybrid, "_SETTLE_DELAY", 0)  # failing case polls the settle window
    gate = Gate(kind="postcondition", postcondition={"url_contains": "invoices"})
    page = SimpleNamespace(url="http://app/x/Invoices/list")
    ok, _ = await evaluate_gate(gate, steps_ok=True, page=page, requests_window=[])
    assert ok is True
    page = SimpleNamespace(url="http://app/x/estimates")
    ok, _ = await evaluate_gate(gate, steps_ok=True, page=page, requests_window=[])
    assert ok is False


async def test_end_context_postcondition(monkeypatch):
    monkeypatch.setattr(hybrid, "_SETTLE_DELAY", 0)
    gate = Gate(kind="postcondition", end_context="/x/inputs/sales")
    page = SimpleNamespace(url="http://app/x/inputs/sales?tab=1")
    ok, detail = await evaluate_gate(gate, steps_ok=True, page=page, requests_window=[])
    assert ok is True and detail["reached"] == "/x/inputs/sales"
    page = SimpleNamespace(url="http://app/x/dashboard")
    ok, detail = await evaluate_gate(gate, steps_ok=True, page=page, requests_window=[])
    assert ok is False and detail["reached"] == "/x/dashboard"


async def test_download_gate_is_authoritative_both_directions():
    """A file in the window passes even when the agent gave up (the download click's
    timeout receipt poisons self-reports — observed live: 4 downloads, then an
    honest-but-wrong failure report); no file fails even a claimed success."""
    gate = Gate(kind="download")
    ok, detail = await evaluate_gate(gate, steps_ok=False, page=None, requests_window=[],
                                     downloads_window=["Forecast report_27.pdf"])
    assert ok is True
    assert detail == {"kind": "download", "files": ["Forecast report_27.pdf"]}
    ok, detail = await evaluate_gate(gate, steps_ok=True, page=None, requests_window=[],
                                     downloads_window=[])
    assert ok is False and detail["files"] == []


def test_segment_gate_resolution():
    sub_marker = Subtask(index=0, template_prompt="save it", marker="Invoices")
    assert segment_gate(sub_marker, None, "/a").kind == "marker"

    sub_post = Subtask(index=0, template_prompt="open flyout",
                       postcondition={"visible": "#flyout"})
    assert segment_gate(sub_post, None, "/a").kind == "postcondition"

    # Download wording -> the download gate; marker and declared postcondition outrank it.
    sub_dl = Subtask(index=0, template_prompt="select download, select PDF")
    assert segment_gate(sub_dl, {"end_context": "/b"}, "/a").kind == "download"
    assert segment_gate(Subtask(index=0, template_prompt="export and save",
                                marker="Reports"), None, "/a").kind == "marker"
    assert segment_gate(Subtask(index=0, template_prompt="download it",
                                postcondition={"visible": "#x"}),
                        None, "/a").kind == "postcondition"

    sub_plain = Subtask(index=0, template_prompt="navigate")
    # Library entry recorded that this segment ENDS somewhere else -> inherited postcondition.
    gate = segment_gate(sub_plain, {"end_context": "/b"}, "/a")
    assert gate.kind == "postcondition" and gate.end_context == "/b"
    # Same end context as start (a fill segment) -> steps floor.
    assert segment_gate(sub_plain, {"end_context": "/a"}, "/a").kind == "steps"
    assert segment_gate(sub_plain, None, "/a").kind == "steps"


# ------------------------------- unit: declared checks on gates -------------------------------


def test_segment_gate_attaches_declared_checks_to_any_kind():
    from automation.pipeline.checks import Check

    verify = (Check(kind="url_contains", arg="payroll"),)
    plain = Subtask(index=0, template_prompt="open payroll", verify=verify)
    gate = segment_gate(plain, None, "/a")
    assert gate.kind == "steps" and gate.checks == verify

    saver = Subtask(index=0, template_prompt="save it", marker="Invoices", verify=verify)
    gate = segment_gate(saver, None, "/a")
    assert gate.kind == "marker" and gate.checks == verify

    bare = Subtask(index=0, template_prompt="open payroll")
    assert segment_gate(bare, None, "/a").checks == ()


async def test_checks_demote_passing_steps_gate(monkeypatch):
    from automation.pipeline import checks as ck
    from automation.pipeline.checks import Check

    monkeypatch.setattr(ck, "_CHECK_POLL_S", 0.01)
    gate = Gate(kind="steps",
                checks=(Check(kind="url_contains", arg="payroll", timeout_s=0.05),))
    page = SimpleNamespace(url="http://app/dashboard")
    ok, detail = await evaluate_gate(gate, steps_ok=True, page=page, requests_window=[])
    assert ok is False
    assert detail["kind"] == "steps"
    [c] = detail["checks"]
    assert c["ok"] is False and c["kind"] == "url_contains"


async def test_checks_never_resurrect_failed_base():
    from automation.pipeline.checks import Check

    gate = Gate(kind="steps", checks=(Check(kind="url_contains", arg="payroll"),))
    page = SimpleNamespace(url="http://app/payroll")
    ok, detail = await evaluate_gate(gate, steps_ok=False, page=page, requests_window=[])
    assert ok is False
    [c] = detail["checks"]
    assert c["ok"] is True  # evaluated once so the report still shows the detail


async def test_checks_demote_authoritative_marker_gate(monkeypatch):
    from automation.pipeline import checks as ck
    from automation.pipeline.checks import Check

    monkeypatch.setattr(ck, "_CHECK_POLL_S", 0.01)
    gate = Gate(kind="marker", marker="Invoices",
                checks=(Check(kind="write_accepted", arg="Payments", timeout_s=0.05),))
    hit = [{"method": "POST", "url": "http://api/Invoices/create", "status": 201,
            "step": 3}]
    ok, detail = await evaluate_gate(gate, steps_ok=True, page=None, requests_window=hit)
    assert ok is False
    assert detail["create_write_seen"] is True  # base verdict still recorded


async def test_checks_single_shot_when_base_failed():
    from automation.pipeline.checks import Check

    calls: list[str] = []

    class Probe:
        url = "http://x"

        async def evaluate(self, expr):
            calls.append(expr)
            return {"count": 0}

    gate = Gate(kind="steps",
                checks=(Check(kind="text_visible", arg="Ghost", timeout_s=5),))
    ok, _ = await evaluate_gate(gate, steps_ok=False, page=Probe(), requests_window=[])
    assert ok is False and len(calls) == 1


def test_describe_expected_end_mentions_checks():
    from automation.pipeline.checks import Check

    gate = Gate(kind="steps", checks=(Check(kind="write_accepted", arg="Employees"),
                                      Check(kind="text_visible", arg="Alistair Allan")))
    text = hybrid._describe_expected_end(gate)
    assert 'an accepted write to "Employees"' in text
    assert '"Alistair Allan"' in text

    postcond = Gate(kind="postcondition", postcondition={"url_contains": "payrun"},
                    checks=(Check(kind="url_contains", arg="payroll"),))
    both = hybrid._describe_expected_end(postcond)
    assert 'the page URL contains "payrun"' in both and '"payroll"' in both

    bare = Gate(kind="steps")
    assert hybrid._describe_expected_end(bare) is None


def test_check_failure_reason_names_first_failure():
    detail = {"kind": "steps", "checks": [
        {"kind": "url_contains", "arg": "payroll", "ok": True,
         "evidence": "http://app/payroll", "error": None},
        {"kind": "write_accepted", "arg": "FPS", "ok": False, "evidence": None,
         "error": 'no accepted create-write matching "FPS" in this segment\'s traffic'},
    ]}
    reason = hybrid._check_failure_reason(detail)
    assert reason.startswith('deterministic check failed: write_accepted "FPS"')
    assert "no accepted" in reason
    assert hybrid._check_failure_reason({"kind": "steps"}) is None


def test_report_renders_check_verdicts():
    from automation.pipeline import report

    html = report._render_subtasks([{
        "index": 0, "prompt": "add employee", "mode": "authored", "ok": False,
        "steps_executed": 5, "duration_seconds": 3.2,
        "gate": {"kind": "steps", "checks": [
            {"kind": "write_accepted", "arg": "Employees", "ok": True,
             "evidence": "POST /api/Employees → 200", "error": None},
            {"kind": "text_visible", "arg": "Alistair", "ok": False,
             "evidence": None, "error": "not found"},
        ]},
    }])
    assert "write_accepted" in html and "text_visible" in html
    assert "POST /api/Employees" in html and "not found" in html


# ------------------------------- unit: receipt roll-up wiring -------------------------------


async def test_rollup_demotes_passing_gate_and_records_reasons():
    ok, detail = await evaluate_gate(
        Gate(kind="steps"), steps_ok=True, page=None, requests_window=[],
        rollup=(False, ["receipts contradict success: X"]))
    assert ok is False
    assert detail["rollup"] == ["receipts contradict success: X"]

    ok, detail = await evaluate_gate(
        Gate(kind="steps"), steps_ok=True, page=None, requests_window=[],
        rollup=(True, []))
    assert ok is True and detail == {"kind": "steps"}


def test_rollup_applies_only_to_selfreport_gates():
    assert hybrid._rollup_applies(Gate(kind="steps")) is True
    assert hybrid._rollup_applies(
        Gate(kind="postcondition", postcondition={"url_contains": "x"})) is True
    assert hybrid._rollup_applies(Gate(kind="marker", marker="m")) is False
    assert hybrid._rollup_applies(Gate(kind="download")) is False


def test_check_failure_reason_covers_rollup():
    detail = {"kind": "steps", "rollup": ["receipts contradict success: Y"]}
    assert hybrid._check_failure_reason(detail) == "receipts contradict success: Y"


async def test_consumer_commits_with_line_and_date_transforms_then_replays_fresh(
        stores, monkeypatch):
    """The reformat/multi-field hole CLOSED (2026-08-12): typed values that are a LINE
    of an extracted block or a date REFORMAT of an extracted date bind with transform
    specs — and the next run replays them against ITS OWN fresh identity."""
    ctx = ss.normalize_context("http://app/section")
    consumer_sid = ss.subtask_id(BOUND_SPEC.subtasks[1].prompt, ctx)
    monkeypatch.setattr(hybrid, "save_steps", _fill_steps_stub([
        {"action": "fill", "selectors": ["css=#name"], "value": "Euan Bruce"},
        {"action": "fill", "selectors": ["css=#dob"], "value": "02/03/1979"},
        {"action": "click", "selectors": ['text="Save"']},
    ]))
    fake = FakeSession(_runner(), agents=[
        _seg(True, mode="authored", finding="identity noted", extracted={
            "identity_block": "Euan Bruce\n70 Telford Street\nBARFORD ST JOHN\nOX15 8PG",
            "dob_block": "Birthday\nMarch 2, 1979"}),
        _seg(True, mode="authored"),
    ])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await run_hybrid_task(fake.runner or _runner(), BOUND_PROMPT,
                                   spec=BOUND_SPEC)

    assert result.is_successful is True
    entry = ss.load_manifest()[consumer_sid]
    assert entry["bindings"]["bound_1"] == {
        "kind": "extract", "label": "identity_block",
        "transform": {"line": {"index": 0, "count": 1, "join": " "}}}
    assert entry["bindings"]["bound_2"] == {
        "kind": "extract", "label": "dob_block", "transform": {"date": "%d/%m/%Y"}}

    # Next run, DIFFERENT identity: both bindings resolve fresh -> zero-LLM replay.
    fake2 = FakeSession(_runner(), replays=[
        _seg(True, extracted={
            "identity_block": "Struan Boyd\n5 Long Acre\nLEEDS\nLS1 4AB",
            "dob_block": "Birthday\nJune 14, 1983"}),
        _seg(True),
    ])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake2))
    result2 = await run_hybrid_task(fake2.runner or _runner(), BOUND_PROMPT,
                                    spec=BOUND_SPEC)
    assert result2.is_successful is True
    assert fake2.replay_calls == 2 and fake2.agent_calls == 0
    assert result2.subtasks[1]["mode"] == "replay"


# ---------------- unanchorable steps, values.json handoff, replay errors ----------------


async def test_unanchorable_step_refuses_the_commit(stores, monkeypatch, capsys):
    """A click the compiler could not tie to a location must NOT be committed: storing
    the tool's text search instead is what made replays hunt tokens and land on the
    wrong element. The segment authors live every run until it can be anchored."""
    monkeypatch.setattr(hybrid, "save_steps", _fill_steps_stub([
        {"action": "click", "selectors": ["xpath=/html/body/button"]},
        {"action": "unanchorable", "why": "find_by_text click on 'Net to gross' "
                                          "captured no anchorable element identity"},
    ]))
    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored"),
                                          _seg(True, mode="authored")])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await run_hybrid_task(fake.runner or _runner(), PROMPT, spec=SPEC,
                                   marker="Invoices")

    assert result.is_successful is True          # the run itself is fine
    assert ss.load_manifest() == {}              # nothing committed
    assert "Net to gross" in capsys.readouterr().out


async def test_run_values_are_written_to_the_run_dir_handoff_file(stores, monkeypatch,
                                                                  tmp_path):
    """The extracted data is a real file the next segment can resolve against, not just
    an in-memory dict (the user's 'temp file' handoff)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fake = FakeSession(_runner(), run_dir=run_dir, agents=[
        _seg(True, mode="authored", extracted={"identity_block": "Ayaan Campbell\n73 Tadcaster Rd"}),
        _seg(True, mode="authored"),
    ])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    await run_hybrid_task(fake.runner or _runner(), PROMPT, spec=SPEC, marker="Invoices")

    values = json.loads((run_dir / hybrid.VALUES_FILE).read_text())
    assert values["identity_block"].startswith("Ayaan Campbell")


async def test_binding_resolver_falls_back_to_the_handoff_file(tmp_path):
    """A resolver built before the producing segment ran (or a resumed process) still
    resolves: values.json is the durable half of the handoff."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / hybrid.VALUES_FILE).write_text(json.dumps(
        {"identity_block": "Struan Boyd\n5 Long Acre\nLEEDS"}))
    resolve = hybrid._binding_resolver({}, SimpleNamespace(run_dir=run_dir))
    assert resolve({"kind": "extract", "label": "identity_block",
                    "transform": {"line": {"index": 1, "count": 1, "join": " "}}}) \
        == "5 Long Acre"
    assert resolve({"kind": "extract", "label": "nope"}) is None


async def test_failed_replay_keeps_its_extracts_and_records_why(stores, monkeypatch,
                                                                tmp_path):
    """A replay that read data off the page before dying must not lose it (seg is
    rebound to the authored segment), and the replay failure must leave an artifact."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sid = _sid_for(SPEC.subtasks[0].prompt)
    _seed_entry(sid)
    fake = FakeSession(_runner(), run_dir=run_dir,
                       replays=[_seg(False, executed=2, error="boom",
                                     extracted={"identity_block": "Aaran Duncan"})],
                       agents=[_seg(True, mode="authored"), _seg(True, mode="authored")])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await run_hybrid_task(fake.runner or _runner(), PROMPT, spec=SPEC,
                                   marker="Invoices")

    assert result.subtasks[0]["mode"] == "replay_failed->authored"
    assert result.subtasks[0]["replay_error"] == "boom"
    values = json.loads((run_dir / hybrid.VALUES_FILE).read_text())
    assert values["identity_block"] == "Aaran Duncan"


# ---------------- producer slices: cacheable, but only with a real capture ----------------
# node_kind now resolves NOTING wording ("note and remember the OTP") to an action rather
# than a judge — its replay is a live extract, not a hollow assertion, exactly as the
# tab_url carve-out has always assumed. That only holds while the recording actually
# CONTAINS the extract, so the commit guard below is what licenses the classification.

_NOTING_SPEC = TaskSpec(
    key="k", prompt=PROMPT, marker="Invoices",
    subtasks=(
        SubtaskDecl(prompt="open the panel and note the reference number shown"),
        SubtaskDecl(prompt="add invoice for customer {{customer}} and click save",
                    values={"customer": "Suresh Gopi"}, marker="Invoices"),
    ),
)




async def test_noting_segment_with_an_extract_commits(stores, monkeypatch):
    """The whole point: a noting slice whose recording carries the capture DOES cache,
    and its replayed extract re-reads the value fresh every run."""
    monkeypatch.setattr(hybrid, "save_steps", _fill_steps_stub([
        {"action": "click", "selectors": ["xpath=/html/body/button"]},
        {"action": "extract", "label": "reference_number",
         "selectors": ["xpath=/html/body/span"]},
    ]))
    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored"),
                                          _seg(True, mode="authored")])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    await run_hybrid_task(fake.runner or _runner(), PROMPT, spec=_NOTING_SPEC,
                          marker="Invoices")

    committed = ss.load_manifest()
    assert committed, "a noting slice WITH an extract step must be cacheable"
    assert any("note the reference number" in e["template_prompt"]
               for e in committed.values())



# --------------------- conditional branch: tolerant replay ---------------------
# A conditional slice ("If a popup appears, tick X and click Process") records the TRUE
# branch on the run where the popup showed. Its recording IS the branch — there is nothing
# else in it — so the FIRST recorded action doubles as the condition's own test: if that
# element is not on the page, the condition is not raised and the whole branch is a no-op.
# Failing AFTER something acted is different: the branch WAS raised and we could not finish
# it, which is a real failure and must stay one.

def _branch_sub(prompt=_PROBE_COND_LINE):
    return Subtask(index=1, template_prompt=prompt, values={})


async def _replay_with(monkeypatch, outcome, *, branch, prompt=_PROBE_COND_LINE):
    hs = hybrid.HybridSession(_runner())
    hs.runner.config.reveal_hidden_controls = False
    hs._main_page = SimpleNamespace()

    async def fake_eval(js):
        return None

    hs._main_page.evaluate = fake_eval
    hs.current_page = lambda: hs._main_page

    async def fake_execute(skill, page, *a, **k):
        return outcome

    async def fake_adopt(sub):
        return None

    monkeypatch.setattr(hybrid.skills, "execute", fake_execute)
    monkeypatch.setattr(hs, "adopt_announced_tab", fake_adopt)
    monkeypatch.setattr(hs, "network_watermark", lambda: 0)
    monkeypatch.setattr(hs, "downloads_watermark", lambda: 0)
    monkeypatch.setattr(hs, "requests_since", lambda m: [])
    monkeypatch.setattr(hs, "downloads_since", lambda m: [])
    return await hs.replay_segment(_branch_sub(prompt), "sid", "/ctx",
                                   SimpleNamespace(body="steps", steps=[], sid="sid"),
                                   Gate(kind="steps"), branch=branch)


async def test_branch_replay_passes_when_nothing_acted(monkeypatch):
    """No popup: the first recorded click resolves nothing, so the branch never applied."""
    seg = await _replay_with(monkeypatch, {
        "executed": 0, "failed_at": 0, "log": [],
        "error": 'no unique candidate matched: css=[id=process] -> no match'},
        branch=True)
    assert seg.ok is True
    assert seg.skip_reason == "branch_absent"
    assert seg.steps_executed == 0
    assert seg.error is None
    assert seg.gate["raised"] is False


async def test_branch_replay_still_fails_once_a_step_acted(monkeypatch):
    """Popup WAS there (the tick landed) but Process could not be clicked — a real failure:
    the branch was raised and left half-done, which must not be waved through."""
    seg = await _replay_with(monkeypatch, {
        "executed": 1, "failed_at": 1,
        "log": [{"step": 0, "action": "click", "used": "css=[id=dontshow]"}],
        "error": 'no unique candidate matched: css=[id=process] -> no match'},
        branch=True)
    assert seg.ok is False
    assert seg.skip_reason is None
    assert "no unique candidate" in seg.error


async def test_non_branch_replay_never_gets_the_concession(monkeypatch):
    """An ordinary action slice failing on its first step is still a failure — the
    concession is only sound where the recording is a conditional branch."""
    seg = await _replay_with(monkeypatch, {
        "executed": 0, "failed_at": 0, "log": [],
        "error": 'no unique candidate matched: css=[id=save] -> no match'},
        branch=False, prompt="Click Save.")
    assert seg.ok is False
    assert seg.skip_reason is None


async def test_unprobed_conditional_replays_with_the_branch_concession(stores, monkeypatch):
    """The wiring: a conditional slice that declares NO probe still caches and replays —
    the user's requirement is that the branch is always saved as a recording — but it
    replays with branch=True, so a run without the popup resolves as "not raised" instead
    of failing. Wording is what identifies it (decompose.is_conditional_guard)."""
    prompt = "go to the section. " + _PROBE_COND_LINE
    spec = TaskSpec(key="p", prompt=prompt, subtasks=(
        SubtaskDecl(prompt="go to the section."),
        SubtaskDecl(prompt=_PROBE_COND_LINE),
    ))
    ctx = ss.normalize_context("http://app/section")
    _seed_entry(ss.subtask_id(_PROBE_COND_LINE, ctx),
                [{"action": "click", "selector": "#process"}])

    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored")],
                       replays=[_seg(True, executed=1)])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await run_hybrid_task(fake.runner or _runner(), prompt, spec=spec)

    assert result.is_successful is True
    assert fake.replay_calls == 1
    assert fake.replay_branch == [True]          # the concession was granted


async def test_ordinary_slice_replays_without_the_branch_concession(stores, monkeypatch):
    """An action slice must never get it: its first step failing is a real failure."""
    prompt = "Click Save."
    spec = TaskSpec(key="p", prompt=prompt,
                    subtasks=(SubtaskDecl(prompt="Click Save."),))
    ctx = ss.normalize_context("http://app/section")
    _seed_entry(ss.subtask_id("Click Save.", ctx),
                [{"action": "click", "selector": "#save"}])

    fake = FakeSession(_runner(), replays=[_seg(True, executed=1)])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    await run_hybrid_task(fake.runner or _runner(), prompt, spec=spec)

    assert fake.replay_calls == 1
    assert fake.replay_branch == [False]


# --------------------- declared allow_write_refusal waiver ---------------------
# A slice whose own wording declares an error branch ("click Submit. if it shows an
# error, click cancel") ends legitimately on a REFUSED write. The window write rule
# cannot know that — it judges traffic and never reads prose, which is exactly what
# makes it hold for undeclared tasks — so the task author declares the exemption.

async def test_declared_waiver_passes_a_refused_only_window():
    window = [{"method": "POST", "url": "http://api/Years/27/FPS", "status": 200,
               "body": '{"status": false, "message": "already submitted"}'}]
    gate = Gate(kind="steps", allow_write_refusal=True)
    ok, detail = await evaluate_gate(gate, steps_ok=True, page=None,
                                     requests_window=window)
    assert ok is True
    # Reported, never swallowed: the run's report still carries the server's refusal.
    assert "already submitted" in detail["write_rollup"][0]


async def test_declared_waiver_is_scoped_to_the_write_rule():
    """The waiver exempts the write rule and nothing else — a segment whose steps
    failed still fails."""
    gate = Gate(kind="steps", allow_write_refusal=True)
    ok, _ = await evaluate_gate(gate, steps_ok=False, page=None, requests_window=[])
    assert ok is False


def test_segment_gate_carries_the_declared_waiver():
    waived = Subtask(index=0, template_prompt="click Submit; if it errors click cancel",
                     allow_write_refusal=True)
    assert segment_gate(waived, None, "/x").allow_write_refusal is True
    plain = Subtask(index=0, template_prompt="click Submit")
    assert segment_gate(plain, None, "/x").allow_write_refusal is False


async def test_waived_refusal_is_never_a_failure_reason():
    """A waived refusal is evidence in the report, never a verdict. When a waived
    segment fails for some other reason (its steps did not complete), the refusal it was
    explicitly told to tolerate must not be handed back as the explanation."""
    window = [{"method": "POST", "url": "http://api/Years/27/FPS", "status": 200,
               "body": '{"status": false, "message": "already submitted"}'}]
    gate = Gate(kind="steps", allow_write_refusal=True)
    ok, detail = await evaluate_gate(gate, steps_ok=False, page=None,
                                     requests_window=window)
    assert ok is False                      # the steps floor failed
    assert "already submitted" in detail["write_rollup"][0]   # still reported
    assert hybrid._check_failure_reason(detail) is None       # but never the reason


def _waived_session(window):
    waived = _seg(True, mode="authored")
    waived.gate = {"kind": "steps",
                   "write_rollup": ['... last refusal: "already submitted"'],
                   "write_refusal_waived": True}
    fake = FakeSession(_runner(), agents=[waived, _seg(True, mode="authored")])
    fake.window = window
    return fake


async def test_waived_refusal_segment_commits_with_the_branch_marked_optional(
        stores, monkeypatch):
    """A waived segment ends on the error branch it declared ("click cancel"). Refusing to
    cache it for that reason made the payroll e2e's bulk-FPS subtask author live at ~400k
    tokens EVERY run. The work is cached; the steps after the refused write carry
    `optional`, so a run whose write IS accepted skips them instead of failing."""
    body = json.dumps({"status": False, "message": "already submitted"})
    # Two agent steps: the submit (step 0, whose write the server refused) and the Cancel
    # that closed the dialog it raised.
    recording = {"history": [
        {"state": {"url": "http://app/start", "interacted_element": []},
         "model_output": {"action": [{"navigate": {"url": "http://app/section"}}]},
         "result": []},
        {"state": {"url": "http://app/section", "interacted_element": [
            {"node_name": "BUTTON", "ax_name": "Cancel", "attributes": {"id": "cancel"},
             "x_path": "html/body/button"}]},
         "model_output": {"action": [{"click": {"index": 7}}]},
         "result": [{"extracted_content": "Clicked button \" Cancel\""}]},
    ]}
    fake = _waived_session([_req(0, method="GET"), _req(0, body=body)])
    fake.recording = recording
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await _run(fake)

    assert fake.agent_calls == 2
    assert [s["ok"] for s in result.subtasks] == [True, True]
    waived_sid = _sid_for(SPEC.subtasks[0].prompt)
    assert waived_sid in ss.load_manifest() and ss.has_script(waived_sid)
    # The code tier is what actually replays a committed entry, so the mark has to be
    # in the generated skill — not only in the compiled step list it came from.
    code = ss.code_path(waived_sid).read_text()
    assert "await api.begin_optional()" in code
    assert code.index("await api.begin_optional()") < code.index("await api.click(")
    # No end_title pinned: with the branch skippable the segment has two possible endings.
    assert ss.load_manifest()[waived_sid].get("end_title") is None


async def test_a_waived_refusal_with_no_identifiable_write_is_still_not_committed(
        stores, monkeypatch):
    """The fallback. With no refused write to point at there is no boundary between the
    work and the branch, and guessing one would cache a Cancel as load-bearing."""
    fake = _waived_session([])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await _run(fake)

    assert [s["ok"] for s in result.subtasks] == [True, True]
    manifest = ss.load_manifest()
    assert len(manifest) == 1
    waived_sid = _sid_for(SPEC.subtasks[0].prompt)
    assert waived_sid not in manifest and not ss.has_script(waived_sid)


# ------------------- where the declared error branch starts -------------------


def _req(step, url="https://api/Payroll/FPS", method="POST", status=200, body=None):
    rec = {"step": step, "method": method, "url": url, "status": status}
    if body is not None:
        rec["body"] = body
    return rec


def test_refused_write_step_is_the_agent_step_the_server_said_no_on():
    body = json.dumps({"status": False,
                       "message": "Bryan Christie's FPS of this period is already submitted"})
    window = [_req(0, method="GET"), _req(9, body=body), _req(10, method="GET")]
    assert hybrid._refused_write_step(window) == 9


def test_refused_write_step_ignores_infrastructure_and_page_load_traffic():
    body = json.dumps({"status": False, "message": "nope"})
    window = [_req(3, url="https://api/auth/webpush", body=body)]
    assert hybrid._refused_write_step(window) is None
    window = [dict(_req(3, body=body), after_page_load=True)]
    assert hybrid._refused_write_step(window) is None


def test_refused_write_step_is_none_when_the_write_was_accepted():
    window = [_req(9, body=json.dumps({"status": True}))]
    assert hybrid._refused_write_step(window) is None


def test_the_last_refusal_wins():
    """The agent retried: the branch begins after the refusal it actually cancelled."""
    body = json.dumps({"status": False, "message": "already submitted"})
    assert hybrid._refused_write_step([_req(4, body=body), _req(9, body=body)]) == 9


def _cond_subs(*probe_args):
    """Subtasks: one plain producer, then one probed guard per arg, then a plain tail."""
    from automation.pipeline.checks import Check
    subs = [Subtask(index=0, template_prompt="click Submit", values={}, marker=None,
                    postcondition=None, tab_url=None)]
    for n, arg in enumerate(probe_args, start=1):
        subs.append(Subtask(index=n, template_prompt=f"If {arg}, click Cancel.",
                            values={}, marker=None, postcondition=None, tab_url=None,
                            probe=Check(kind="text_visible", arg=arg, timeout_s=3.0)))
    subs.append(Subtask(index=len(subs), template_prompt="go to Clients", values={},
                        marker=None, postcondition=None, tab_url=None))
    return subs


def test_describe_next_conditionals_renders_the_successors_probe():
    """The condition crosses the boundary in the SAME words evaluate_gate/probe_condition
    use — via the shared _describe_check — and nothing else does."""
    subs = _cond_subs("already submitted")
    assert hybrid._describe_next_conditionals(subs, 0) == (
        'the text "already submitted" visible on the page', 1)


def test_describe_next_conditionals_walks_consecutive_guards():
    """A guard chain is ONE handoff from the producing step's point of view."""
    desc, n = hybrid._describe_next_conditionals(_cond_subs("first", "second"), 0)
    assert n == 2
    assert desc == ('the text "first" visible on the page or '
                    'the text "second" visible on the page')


def test_describe_next_conditionals_none_without_a_probed_successor():
    """The common case: no probed successor, no block, and the scan stops at the first
    unprobed slice rather than reaching a later one."""
    subs = _cond_subs("already submitted")
    assert hybrid._describe_next_conditionals(subs, 1) == (None, 0)   # guard -> tail
    assert hybrid._describe_next_conditionals(subs, 2) == (None, 0)   # last subtask
    plain = [Subtask(index=i, template_prompt=f"step {i}", values={}, marker=None,
                     postcondition=None, tab_url=None) for i in range(2)]
    assert hybrid._describe_next_conditionals(plain, 0) == (None, 0)


async def test_probed_successors_are_dropped_from_remaining_and_described_instead(
        stores, monkeypatch):
    """A probed successor is DESCRIBED to its producer as an expected outcome, so it must
    not ALSO sit in the still-ahead list — that line handed the agent the successor's
    action words ("click Cancel to close it") under a generic do-not-start rule it stops
    honouring the moment it believes its own step failed (run 20260901_151214)."""
    prompt, spec = _probe_spec()
    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored")], probes=[False])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    await run_hybrid_task(fake.runner or _runner(), prompt, spec=spec)

    assert fake.agent_calls == 1                       # the producer slice only
    assert fake.remaining_seen[0] == []                # the guard was dropped, not listed
    assert _PROBE_COND_LINE not in " ".join(fake.remaining_seen[0])
    assert fake.next_conditional_seen[0] == (
        'the text "Don\'t show this again" visible on the page')
