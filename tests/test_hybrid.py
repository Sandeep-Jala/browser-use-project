"""hybrid engine tests: the subtask loop, gates, and library commit rules.

The browser-facing surface (HybridSession) is faked — the same injected-seam style as
test_suite — while the loop, the library commit rules (_author_segment), and the gate
evaluation run for real against tmp_path-monkeypatched stores."""
import json
from types import SimpleNamespace

import pytest

from automation.pipeline import hybrid
from automation.pipeline import subtask_store as ss
from automation.pipeline.decompose import Subtask
from automation.pipeline.hybrid import (Gate, Segment, evaluate_gate, reauthor_match,
                                        run_hybrid_task, segment_gate)
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
    so the real _author_segment commit path runs end-to-end."""

    def __init__(self, runner, *, replays=None, agents=None, create_write_seen=True):
        self.runner = runner
        self.replays = list(replays or [])
        self.agents = list(agents or [])
        self.create_write_seen = create_write_seen
        self.replay_calls = 0
        self.agent_calls = 0

    @classmethod
    def make_opener(cls, instance):
        async def opener(runner):
            instance.runner = runner
            return instance
        return SimpleNamespace(open=opener)

    async def current_url(self):
        return "http://app/section"

    async def replay_segment(self, sub, sid, context, steps, gate):
        self.replay_calls += 1
        seg = self.replays.pop(0)
        seg.index, seg.sid, seg.context = sub.index, sid, context
        seg.prompt = sub.instantiated_prompt
        return seg

    async def agent_segment(self, sub, sid, context, gate, *, completed, remaining,
                            dirty=False, prior_failure=None, record_path=None):
        self.agent_calls += 1
        seg = self.agents.pop(0)
        seg.index, seg.sid, seg.context = sub.index, sid, context
        seg.prompt = sub.instantiated_prompt
        seg.mode = "authored"
        if record_path is not None:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps(FAKE_RECORDING))
        return seg

    async def finalize(self, task, parent_marker):
        gt = None
        if parent_marker:
            gt = {"marker": parent_marker, "create_write_seen": self.create_write_seen,
                  "write_step": None, "overrode_success": False}
        return _skeleton_result(ground_truth=gt)


def _seg(ok, *, executed=0, error=None, mode="replay", write_step=None):
    return Segment(index=0, sid="", prompt="", context="", mode=mode, ok=ok,
                   steps_executed=executed, error=error, write_step=write_step)


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
        steps = json.loads(ss.steps_path(sid).read_text())
        # emit_start_goto=False: the recording's start URL goto is NOT emitted; only the
        # explicit navigate action compiles.
        assert steps == [{"action": "goto", "url": "http://app/section"}]


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
    # Failed before touching the page -> old entry archived, clean re-author committed.
    archived = list((ss.LIBRARY_DIR / "archive").glob(f"{sid0}.steps.*.json"))
    assert len(archived) == 1
    assert json.loads(ss.steps_path(sid0).read_text()) == \
        [{"action": "goto", "url": "http://app/section"}]


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
    # The stale detour steps were replaced by the fresh recording's compiled steps.
    assert json.loads(ss.steps_path(sid0).read_text()) == \
        [{"action": "goto", "url": "http://app/section"}]


async def test_reauthor_failure_keeps_old_entry(stores, monkeypatch):
    """A failed re-author must NOT clobber the working library entry."""
    ctx = ss.normalize_context("http://app/section")
    sid0 = ss.subtask_id(SPEC.subtasks[0].prompt, ctx)
    stale = [{"action": "click", "selectors": ["text=Works"]}]
    _seed_entry(sid0, stale)

    fake = FakeSession(_runner(), agents=[_seg(False, error="lost", mode="authored")])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await _run(fake, reauthor="go to the section")

    assert result.is_successful is False
    assert json.loads(ss.steps_path(sid0).read_text()) == stale


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


def test_segment_gate_resolution():
    sub_marker = Subtask(index=0, template_prompt="save it", marker="Invoices")
    assert segment_gate(sub_marker, None, "/a").kind == "marker"

    sub_post = Subtask(index=0, template_prompt="open flyout",
                       postcondition={"visible": "#flyout"})
    assert segment_gate(sub_post, None, "/a").kind == "postcondition"

    sub_plain = Subtask(index=0, template_prompt="navigate")
    # Library entry recorded that this segment ENDS somewhere else -> inherited postcondition.
    gate = segment_gate(sub_plain, {"end_context": "/b"}, "/a")
    assert gate.kind == "postcondition" and gate.end_context == "/b"
    # Same end context as start (a fill segment) -> steps floor.
    assert segment_gate(sub_plain, {"end_context": "/a"}, "/a").kind == "steps"
    assert segment_gate(sub_plain, None, "/a").kind == "steps"
