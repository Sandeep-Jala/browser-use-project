"""hybrid engine tests: the subtask loop, gates, and library commit rules.

The browser-facing surface (HybridSession) is faked via an injected seam, while the loop,
the library commit rules (_author_segment), and the gate evaluation run for real against
tmp_path-monkeypatched stores."""
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
                 recording=None, run_dir=None):
        self.runner = runner
        self.replays = list(replays or [])
        self.agents = list(agents or [])
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
        self.agent_calls = 0
        self.findings_seen = []
        self.record_paths = []
        self.events = []              # ordered ("open", url)/("close",)/("replay",)/("agent",)
        self.aux_open_error = None    # set to make open_aux_tab raise

    @classmethod
    def make_opener(cls, instance):
        async def opener(runner):
            instance.runner = runner
            return instance
        return SimpleNamespace(open=opener)

    async def current_url(self):
        return "http://app/section"

    async def open_aux_tab(self, url):
        if self.aux_open_error:
            raise RuntimeError(self.aux_open_error)
        self.events.append(("open", url))

    async def close_aux_tab(self):
        self.events.append(("close",))

    async def replay_segment(self, sub, sid, context, skill, gate):
        self.events.append(("replay",))
        self.replay_calls += 1
        seg = self.replays.pop(0)
        seg.index, seg.sid, seg.context = sub.index, sid, context
        seg.prompt = sub.instantiated_prompt
        return seg

    async def agent_segment(self, sub, sid, context, gate, *, completed, remaining,
                            dirty=False, prior_failure=None, record_path=None,
                            findings=None):
        self.events.append(("agent",))
        self.agent_calls += 1
        self.findings_seen.append(list(findings or []))
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
                  "write_step": None, "overrode_success": False}
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
    ss.update_manifest(sid, "prompt", context="/x")


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


async def test_replay_fail_at_step_zero_is_clean_reauthor(stores, monkeypatch):
    ctx = ss.normalize_context("http://app/section")
    sid0 = ss.subtask_id(SPEC.subtasks[0].prompt, ctx)
    _seed_entry(sid0, [{"action": "click", "selectors": ["text=Gone"]}])

    fake = FakeSession(_runner(), replays=[_seg(False, executed=0, error="no selector")],
                       agents=[_seg(True, mode="authored"), _seg(True, mode="authored")])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await _run(fake)

    assert result.is_successful is True
    # Failed before touching the page -> old entry archived, clean re-author committed
    # (as a code skill; the fresh steps file is subsumed and deleted).
    archived = list((ss.LIBRARY_DIR / "archive").glob(f"{sid0}.steps.*.json"))
    assert len(archived) == 1
    assert "await api.goto('http://app/section')" in ss.code_path(sid0).read_text()
    assert not ss.steps_path(sid0).exists()


# ------------------------------- judge nodes + findings (Phase 0) -------------------------------


async def test_judge_node_never_replays_never_commits(stores, monkeypatch):
    """A verification subtask must run live even when a (legacy, hollow) library entry
    exists for it, and nothing of it may be recorded or committed."""
    prompt = "go to the section. verify the CC field matches the noted mail"
    spec = TaskSpec(key="j", prompt=prompt, subtasks=(
        SubtaskDecl(prompt="go to the section."),
        SubtaskDecl(prompt="verify the CC field matches the noted mail"),
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


async def test_loop_node_never_replays_never_commits(stores, monkeypatch):
    """A repeat-until subtask must run live even when a stale library entry exists (a
    replayed loop walks a FIXED number of iterations and lands on the wrong row — the
    observed wrong-employee bug), and its recording must never be committed: the
    iteration count is live page state."""
    loop_line = ("process the employees one at a time by clicking Save and Next, and "
                 "after each click check that the next employee has loaded, stopping "
                 "as soon as Owen Millar is the employee shown")
    prompt = "go to the section. " + loop_line
    spec = TaskSpec(key="l", prompt=prompt, subtasks=(
        SubtaskDecl(prompt="go to the section."),
        SubtaskDecl(prompt=loop_line),
    ))
    ctx = ss.normalize_context("http://app/section")
    loop_sid = ss.subtask_id(loop_line, ctx)
    stale = [{"action": "click", "selector": "#save-next"}] * 5   # the old row-jumper
    _seed_entry(loop_sid, stale)

    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored"),
                                          _seg(True, mode="authored")])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await run_hybrid_task(fake.runner or _runner(), prompt, spec=spec)

    assert result.is_successful is True
    assert fake.replay_calls == 0 and fake.agent_calls == 2
    assert [s["kind"] for s in result.subtasks] == ["action", "loop"]
    # The loop agent ran WITHOUT a recording path, and the stale entry was not replaced.
    assert fake.record_paths[1] is None
    assert not ss.recording_path(loop_sid).exists()
    assert json.loads(ss.steps_path(loop_sid).read_text()) == stale


async def test_conditional_guard_never_replays_never_commits(stores, monkeypatch):
    """A leading-"If" branch guard must run live even when a library entry exists (a
    TRUE-branch recording would replay its branch unconditionally on every run), and a
    TRUE-branch run must never commit one."""
    cond_line = ("If you see an error about the minimum wage rate, click Add Payment, "
                 "set Amount to 5000, and click Save and Next")
    prompt = "go to the section. " + cond_line
    spec = TaskSpec(key="c", prompt=prompt, subtasks=(
        SubtaskDecl(prompt="go to the section."),
        SubtaskDecl(prompt=cond_line),
    ))
    ctx = ss.normalize_context("http://app/section")
    cond_sid = ss.subtask_id(cond_line, ctx)
    branch = [{"action": "click", "selector": "#add-payment"}]   # the TRUE branch, baked
    _seed_entry(cond_sid, branch)

    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored"),
                                          _seg(True, mode="authored")])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await run_hybrid_task(fake.runner or _runner(), prompt, spec=spec)

    assert result.is_successful is True
    assert fake.replay_calls == 0 and fake.agent_calls == 2
    # Conditionality is a routing predicate, not a node kind: the segment stays "action".
    assert [s["kind"] for s in result.subtasks] == ["action", "action"]
    assert fake.record_paths[1] is None
    assert not ss.recording_path(cond_sid).exists()
    assert json.loads(ss.steps_path(cond_sid).read_text()) == branch


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


async def test_noted_data_consumer_never_replays_and_retires_entry(stores, monkeypatch):
    """A subtask that USES data noted by an earlier segment must not replay its cached
    recording (it would type the AUTHORING run's stale values — the observed Add Employee
    bug): the stale entry is retired and the agent runs the segment with this run's fresh
    findings. The fresh recording IS taken now (the provenance guard decides commit), but
    with nothing bindable in it the guard refuses and nothing enters the library."""
    ctx = ss.normalize_context("http://app/section")
    consumer_sid = ss.subtask_id(NOTED_SPEC.subtasks[1].prompt, ctx)
    _seed_entry(consumer_sid)   # concrete values baked by a previous authoring run

    fake = FakeSession(_runner(), agents=[
        _seg(True, mode="authored", finding="generated_name = Kerris McKay"),
        _seg(True, mode="authored"),
    ])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await run_hybrid_task(fake.runner or _runner(), NOTED_PROMPT, spec=NOTED_SPEC)

    assert result.is_successful is True
    assert fake.replay_calls == 0 and fake.agent_calls == 2
    # The consumer agent received the producer's fresh observation.
    assert fake.findings_seen[1] == [
        "open the generator and note the generated identity.: "
        "generated_name = Kerris McKay"]
    # Stale entry retired; the guard-refused fresh recording committed nothing.
    assert not ss.has_script(consumer_sid)
    assert consumer_sid not in ss.load_manifest()
    assert list((ss.LIBRARY_DIR / "archive").glob(f"{consumer_sid}.steps.*.json"))
    # The consumer DID record (to the temp path) — commit is the guard's call now.
    assert fake.record_paths[1] is not None
    assert result.subtasks[1]["mode"] == "authored"
    assert result.subtasks[1]["skip_reason"] == "dynamic"


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
    def stub(rec, steps_path, max_steps=None, emit_start_goto=False):
        steps_path.parent.mkdir(parents=True, exist_ok=True)
        steps_path.write_text(json.dumps(steps))
        return steps
    return stub


async def test_consumer_with_structured_source_commits_bindings_then_replays(
        stores, monkeypatch):
    """The 2026-07-24 runtime-bindings reversal, completed: a consumer segment whose
    runtime values ALL bind to a structured source (an extract label this run captured)
    commits WITH bindings — and the next run replays it zero-LLM, resolving the bound
    value from its OWN fresh data instead of the authoring run's literal."""
    ctx = ss.normalize_context("http://app/section")
    consumer_sid = ss.subtask_id(BOUND_SPEC.subtasks[1].prompt, ctx)
    monkeypatch.setattr(hybrid, "save_steps", _fill_steps_stub([
        {"action": "fill", "selectors": ["css=#name"], "value": "Kerris McKay"},
        {"action": "click", "selectors": ['text="Save"']},
    ]))

    fake = FakeSession(_runner(), agents=[
        _seg(True, mode="authored", finding="generated_name = Kerris McKay",
             extracted={"generated_name": "Kerris McKay"}),
        _seg(True, mode="authored"),
    ])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await run_hybrid_task(fake.runner or _runner(), BOUND_PROMPT,
                                   spec=BOUND_SPEC)

    assert result.is_successful is True
    entry = ss.load_manifest()[consumer_sid]
    assert entry["bindings"] == {"bound_1": {"kind": "extract",
                                             "label": "generated_name"}}
    assert entry["params"] == {"bound_1": "Kerris McKay"}
    assert entry["start_url"] == "http://app/section"
    # The committed template carries the token, not the authoring literal (the tier-1
    # transpiler may consume steps.json into a code skill; the template is canonical).
    tmpl = json.loads(ss.template_path(consumer_sid).read_text())
    assert tmpl["steps"][0]["value"] == "{{bound_1}}"
    assert result.subtasks[1]["skip_reason"] == "dynamic"

    # Next run: the producer's fresh extract resolves the binding -> zero-LLM replay.
    fake2 = FakeSession(_runner(), replays=[
        _seg(True, finding="generated_name = Struan Boyd",
             extracted={"generated_name": "Struan Boyd"}),
        _seg(True),
    ])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake2))
    result2 = await run_hybrid_task(fake2.runner or _runner(), BOUND_PROMPT,
                                    spec=BOUND_SPEC)

    assert result2.is_successful is True
    assert fake2.replay_calls == 2 and fake2.agent_calls == 0
    assert result2.subtasks[1]["mode"] == "replay"
    assert result2.subtasks[1]["skip_reason"] is None


async def test_consumer_commit_refused_when_a_typed_value_has_no_provenance(
        stores, monkeypatch):
    """The reformat hole stays closed: a consumer recording carrying a typed value that
    is neither prompt-sourced nor bindable (e.g. a re-formatted date the substring guard
    cannot see) must NOT commit — a baked literal would write the authoring run's data
    into every later run's records (the DR021/DR022 wrong-record class)."""
    ctx = ss.normalize_context("http://app/section")
    consumer_sid = ss.subtask_id(BOUND_SPEC.subtasks[1].prompt, ctx)
    monkeypatch.setattr(hybrid, "save_steps", _fill_steps_stub([
        {"action": "fill", "selectors": ["css=#name"], "value": "Kerris McKay"},
        {"action": "fill", "selectors": ["css=#dob"], "value": "25/10/1971"},
        {"action": "click", "selectors": ['text="Save"']},
    ]))

    fake = FakeSession(_runner(), agents=[
        _seg(True, mode="authored", finding="generated_name = Kerris McKay",
             extracted={"generated_name": "Kerris McKay"}),
        _seg(True, mode="authored"),
    ])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await run_hybrid_task(fake.runner or _runner(), BOUND_PROMPT,
                                   spec=BOUND_SPEC)

    assert result.is_successful is True
    assert consumer_sid not in ss.load_manifest()
    assert not ss.has_script(consumer_sid)


async def test_identity_fork_detected_when_wording_recorded_elsewhere(
        stores, monkeypatch):
    """Same wording committed under a DIFFERENT start context is invisible to the direct
    lookup (sids key wording + context). The miss is no longer silent: the subtask row
    and the summary name the fork instead of a bare cache miss."""
    other_sid = ss.subtask_id(SPEC.subtasks[0].prompt, "/other")
    ss.steps_path(other_sid).parent.mkdir(parents=True, exist_ok=True)
    ss.steps_path(other_sid).write_text(
        json.dumps([{"action": "wait", "seconds": 1.0}]))
    ss.update_manifest(other_sid, SPEC.subtasks[0].prompt, context="/other",
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


def test_segment_step_budget_loop_headroom():
    """A loop segment's one budget must cover EVERY iteration (observed live: 17 Save &
    Next advances to reach the named employee; a successful fully-live pass needed 35
    steps). Judge and action nodes stay flat; marker and loop headroom stack."""
    from automation.pipeline.hybrid import segment_step_budget

    assert segment_step_budget(Gate(kind="steps"), 25, kind="loop") == 60
    assert segment_step_budget(Gate(kind="steps"), 25, kind="judge") == 25
    assert segment_step_budget(Gate(kind="marker", marker="Payroll"), 25,
                               kind="loop") == 70


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
