"""Runtime bindings: run-generated values (a fresh data-request ref, a new invoice id)
replay by re-resolving from EACH run's own data — extract labels or create-write response
paths — instead of baking the authoring run's literal (the DR017 stale-ref escape) or
falling back to the agent forever (the pre-binding provenance refusal)."""
import json
from types import SimpleNamespace

import pytest

from automation.pipeline import adapt
from automation.pipeline import hybrid
from automation.pipeline import subtask_store as ss
from automation.pipeline.decompose import consumes_noted_data
from automation.pipeline.hybrid import (_bind_runtime_values, _binding_resolver,
                                        apply_json_path, learn_json_path,
                                        run_hybrid_task)
from automation.skills import base as skills
from automation.tasks import SubtaskDecl, TaskSpec
from tests.test_hybrid import (BOUND_PROMPT, BOUND_SPEC, FakeSession, _fill_steps_stub,
                               _runner, _seg)


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(ss, "LIBRARY_MANIFEST", tmp_path / "library" / "manifest.json")
    monkeypatch.setattr(ss, "DECOMPOSITIONS_DIR", tmp_path / "decompositions")
    (tmp_path / "library").mkdir()
    return tmp_path


class _Net:
    """Network-collector stub carrying pre-captured create-write records."""

    name = "network"

    def __init__(self, requests):
        self._requests = requests

    def results(self):
        return {"requests": self._requests}

    def write(self):
        """The run loop flushes collectors at every segment boundary."""


def _post(url, body_obj):
    return {"method": "POST", "url": url, "body": json.dumps(body_obj)}


# ------------------------------- the pure pieces -------------------------------


def test_learn_and_apply_json_path():
    """One authoring example teaches WHERE the app reports a created record's id; the
    same path then reads each new run's value. Ambiguity refuses (never guess)."""
    body = json.dumps({"result": {"refNo": "PR/X/DR017", "id": "6a63"},
                       "rows": [{"ref": "A"}, {"ref": "B"}]})
    assert learn_json_path(body, "PR/X/DR017") == ["result", "refNo"]
    fresh = json.dumps({"result": {"refNo": "PR/X/DR018", "id": "9f00"}})
    assert apply_json_path(fresh, ["result", "refNo"]) == "PR/X/DR018"

    assert learn_json_path(body, "B") == ["rows", 1, "ref"]        # list indexes
    assert learn_json_path(json.dumps({"n": 42}), "42") == ["n"]   # int leaf, str compare
    assert learn_json_path(json.dumps({"a": "X", "b": "X"}), "X") is None  # ambiguous
    assert learn_json_path("<html>not json</html>", "X") is None
    assert apply_json_path(json.dumps({"other": 1}), ["result", "refNo"]) is None  # drift
    assert apply_json_path(json.dumps({"result": {"refNo": ""}}),
                           ["result", "refNo"]) is None            # empty is a miss


def test_bind_runtime_values_rewrites_every_value_carrier():
    """The flagged literal is rewritten to a {{bound_N}} token wherever the step acts
    with it — find_click text, selector names, expect_text — while unrelated steps stay
    byte-identical. Extract labels are preferred over response paths; a prose-only value
    refuses the whole bind."""
    steps = [
        {"action": "find_click", "text": "PR/X/DR017"},
        {"action": "click", "selectors": ['role=button[name="PR/X/DR017"]',
                                          "xpath=/html/body/button"],
         "expect_text": "PR/X/DR017"},
        {"action": "click", "selectors": ['role=button[name="Verify"]']},
    ]
    bodies = [_post("http://app/api/datarequests/12345", {"refNo": "PR/X/DR017"})]

    bound = _bind_runtime_values(steps, ["PR/X/DR017"], {}, bodies)
    assert bound is not None
    new_steps, params, bindings = bound
    assert new_steps[0]["text"] == "{{bound_1}}"
    assert new_steps[1]["selectors"][0] == 'role=button[name="{{bound_1}}"]'
    assert new_steps[1]["expect_text"] == "{{bound_1}}"
    assert new_steps[2] == steps[2]
    assert params == {"bound_1": "PR/X/DR017"}
    assert bindings["bound_1"] == {
        "kind": "created", "method": "POST",
        "endpoint": ss.normalize_context("http://app/api/datarequests/12345"),
        "path": ["refNo"],
    }

    # An extract this run captured outranks the response path.
    _, _, by_extract = _bind_runtime_values(
        steps, ["PR/X/DR017"], {"ref_no": "PR/X/DR017"}, bodies)
    assert by_extract["bound_1"] == {"kind": "extract", "label": "ref_no"}

    # Prose-only (no structured source anywhere) refuses — the segment stays agent-run.
    assert _bind_runtime_values(steps, ["PR/X/DR017"], {}, []) is None


def test_bind_runtime_values_never_rewrites_inside_larger_literals():
    """Run 20260817_115232: the authoring identity's house number '35' (a real
    standalone field value) substring-matched the unrelated Gross Pay literal '3500',
    compiling fill('3500') into '{{bound_N}}00' — the next run replayed Gross Pay as
    6300 (house number 63 + '00'). A flagged literal must bind only where it stands as
    a whole token (not flanked by alphanumerics); standalone and space/punctuation
    delimited occurrences keep binding."""
    steps = [
        {"action": "fill", "value": "3500"},             # unrelated amount: must stay
        {"action": "fill", "value": "35"},               # the house-number fill: binds
        {"action": "fill", "value": "35 Asfordby Rd"},   # space boundary: binds
    ]
    extracts = {"identity_block": "Ellis McKenzie\n35 Asfordby Rd\nALDENHAM\nWD2 0SZ"}

    bound = _bind_runtime_values(steps, ["35"], extracts, [])
    assert bound is not None
    new_steps, params, _ = bound
    assert new_steps[0]["value"] == "3500"               # NOT '{{bound_1}}00'
    assert new_steps[1]["value"] == "{{bound_1}}"
    assert new_steps[2]["value"] == "{{bound_1}} Asfordby Rd"
    assert params == {"bound_1": "35"}


def test_binding_resolver_first_matching_create_wins():
    hs = SimpleNamespace(collectors=[_Net([
        {"method": "GET", "url": "http://app/api/datarequests/1",
         "body": json.dumps({"refNo": "NOT-A-WRITE"})},
        _post("http://app/api/other/7", {"refNo": "WRONG-ENDPOINT"}),
        _post("http://app/api/datarequests/2", {"refNo": "PR/X/DR018"}),
        _post("http://app/api/datarequests/3", {"refNo": "PR/X/DR019"}),
    ])])
    resolve = _binding_resolver({"ref_no": "EX-1"}, hs)

    assert resolve({"kind": "extract", "label": "ref_no"}) == "EX-1"
    assert resolve({"kind": "extract", "label": "missing"}) is None
    spec = {"kind": "created", "method": "POST",
            "endpoint": ss.normalize_context("http://app/api/datarequests/2"),
            "path": ["refNo"]}
    assert resolve(spec) == "PR/X/DR018"       # chronologically first matching create
    assert resolve({**spec, "endpoint": "/nowhere"}) is None
    assert _binding_resolver({}, SimpleNamespace())({**spec}) is None  # no collectors


def test_body_sourced_values_flag_run_created_identifiers_without_findings():
    """The findings-independent guard leg (the live escape it closes: the producer
    segment replayed silently, findings never named the fresh ref, and the guard
    committed it as a 'clean' literal). A value the run's own create response reports
    is runtime data regardless of findings — unless the TASK's wording names it
    anywhere (prompt data), or it is too short to be an identifier."""
    from automation.pipeline.hybrid import _body_sourced_values

    steps = [
        {"action": "find_click", "text": "PR/X/DR021"},
        {"action": "click", "selectors": ['role=button[name="FOOD LIMITED"]']},
        {"action": "fill", "selectors": ["css=#n"], "value": "22"},
        {"action": "click", "selectors": ['role=button[name="Verify"]']},
    ]
    task_wording = ("go to Payroll module, search and select FOOD LIMITED business "
                    "name \n now click on the same ref. no. and click verify")
    bodies = [_post("http://app/api/datarequests/111", {
        "result": {"autoNumber": 22, "number": "PR/X/DR021",
                   "company": {"name": "FOOD LIMITED"}}})]

    assert _body_sourced_values(steps, task_wording, bodies) == ["PR/X/DR021"]
    # FOOD LIMITED: named by the task's wording. "22": under the length floor.
    # "Verify": not in the body. No bodies at all: nothing to flag.
    assert _body_sourced_values(steps, task_wording, []) == []


async def test_commit_binds_even_when_findings_never_name_the_ref(
        stores, monkeypatch, capsys):
    """The exact run-2 escape, fixed: the producer replays silently (its finding does
    NOT contain the ref), the consumer clicks the on-screen ref, and the commit must
    still flag+bind it via the captured create response — not commit it as clean."""
    prompt = "go to the section. click on the same ref no and verify"
    spec = TaskSpec(key="bind2", prompt=prompt, subtasks=(
        SubtaskDecl(prompt="go to the section."),
        SubtaskDecl(prompt="click on the same ref no and verify",
                    marker="datarequests"),
    ))
    fake = FakeSession(_runner(), agents=[
        _seg(True, mode="authored", finding="panel already closed"),   # no ref anywhere
        _seg(True, mode="authored"),
    ])
    fake.recording = CONSUMER_RECORDING
    fake.collectors = [_Net([_post("http://app/api/datarequests/111",
                                   {"refNo": "PR/X/DR017"})])]
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await run_hybrid_task(fake.runner or _runner(), prompt, spec=spec)

    assert result.is_successful is True
    ctx = ss.normalize_context("http://app/section")
    consumer_sid = ss.subtask_id(spec.subtasks[1].prompt, ctx)
    entry = ss.load_manifest()[consumer_sid]
    assert entry["bindings"]["bound_1"] == {
        "kind": "created", "method": "POST",
        "endpoint": ss.normalize_context("http://app/api/datarequests/111"),
        "path": ["refNo"],
    }
    assert "bound to this run's data" in capsys.readouterr().out


async def test_takeover_of_failed_bound_replay_learns_this_runs_values(
        stores, monkeypatch):
    """When a BOUND entry's replay fails mid-way, the recovering agent must be told
    this run's resolved value (the live failure: the takeover found the STALE record's
    panel open, had no observation naming the fresh ref, and declared the wrong record
    already done)."""
    prompt = "go to the section. open the noted ref and verify it"
    consumer_prompt = "open the noted ref and verify it"
    spec = TaskSpec(key="net2", prompt=prompt, subtasks=(
        SubtaskDecl(prompt="go to the section."),
        SubtaskDecl(prompt=consumer_prompt, marker="datarequests"),
    ))
    ctx = ss.normalize_context("http://app/section")
    sid = ss.subtask_id(consumer_prompt, ctx)
    ss.LIBRARY_DIR.mkdir(exist_ok=True)
    adapt.save_template(ss.template_path(sid), {
        "source_prompt": consumer_prompt,
        "params": {"bound_1": "PR/X/DR017"},
        "bindings": {"bound_1": {"kind": "extract", "label": "ref_no"}},
        "steps": [{"action": "find_click", "text": "{{bound_1}}"}],
    })
    ss.steps_path(sid).write_text(json.dumps([{"action": "find_click",
                                               "text": "{{bound_1}}"}]))
    ss.update_manifest(sid, consumer_prompt, params={"bound_1": "PR/X/DR017"},
                       bindings={"bound_1": {"kind": "extract", "label": "ref_no"}},
                       context=ctx)

    producer = _seg(True, mode="authored", finding="noted the ref")
    producer.extracted = {"ref_no": "PR/X/DR018"}
    fake = FakeSession(_runner(),
                       agents=[producer, _seg(True, mode="authored")],
                       replays=[_seg(False, executed=1, error="row vanished")])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await run_hybrid_task(fake.runner or _runner(), prompt, spec=spec)

    assert result.is_successful is True
    takeover_findings = fake.findings_seen[-1]
    assert any("this run's live value(s) for this step: PR/X/DR018" in f
               for f in takeover_findings)


# ------------------------------- body capture -------------------------------


class _CaptureReq:
    def __init__(self, method):
        self.url = "https://api.app.com/datarequests"
        self.method = method
        self.resource_type = "fetch"
        self.headers: dict = {}


class _CaptureResp:
    def __init__(self, request, status=200, ctype="application/json; charset=utf-8",
                 body='{"refNo": "PR/X/DR018"}'):
        self.request = request
        self.status = status
        self.status_text = "OK"
        self.headers = {"content-type": ctype}
        self._body = body

    async def text(self):
        return self._body


async def test_network_collector_captures_create_write_json_bodies_only(tmp_path):
    """The bindings' data source: a successful create-write's JSON response body is
    captured (capped); reads, non-JSON, and failed writes are not."""
    import asyncio

    from automation.collectors.network import NetworkCollector

    col = NetworkCollector(SimpleNamespace(pages=[]), tmp_path)
    col._active = True
    post = _CaptureReq("POST")
    col._on_response(_CaptureResp(post))
    col._on_response(_CaptureResp(_CaptureReq("GET")))                    # a read
    col._on_response(_CaptureResp(_CaptureReq("POST"), ctype="text/html"))  # not JSON
    col._on_response(_CaptureResp(_CaptureReq("POST"), status=422))       # failed write
    big = _CaptureReq("PUT")
    col._on_response(_CaptureResp(big, body="x" * 40_000))                # capped
    await asyncio.sleep(0)   # let the scheduled capture tasks run

    records = col.results()["requests"]
    assert [("body" in r) for r in records] == [True, False, False, False, True]
    assert records[0]["body"] == '{"refNo": "PR/X/DR018"}'
    assert len(records[4]["body"]) == 16 * 1024


# ------------------------------- loader resolution -------------------------------


def test_load_skill_resolves_bindings_fresh_and_refuses_stale(stores):
    """A bound entry instantiates with THIS run's value (verify_name stamped since the
    find_click text is tokenized); an unresolvable binding — or no resolver at all —
    refuses the load so the stored authoring-run default can never replay."""
    sid = "boundentry000000"
    prompt = "now click on the same ref no and verify"
    template = {
        "source_prompt": prompt,
        "params": {"bound_1": "PR/X/DR017"},
        "bindings": {"bound_1": {"kind": "extract", "label": "ref_no"}},
        "steps": [{"action": "find_click", "text": "{{bound_1}}"}],
    }
    ss.LIBRARY_DIR.mkdir(exist_ok=True)
    adapt.save_template(ss.template_path(sid), template)
    ss.steps_path(sid).write_text(json.dumps(template["steps"]))
    ss.update_manifest(sid, prompt, params=template["params"], context="/section")
    sub = SimpleNamespace(instantiated_prompt=prompt, values={})

    skill = skills.load_skill(sid, sub, run_resolver=lambda spec: "PR/X/DR018")
    assert skill is not None and skill.body == "steps"
    assert skill.steps[0]["text"] == "PR/X/DR018"
    assert skill.steps[0]["verify_name"] is True

    assert skills.load_skill(sid, sub, run_resolver=lambda spec: None) is None
    assert skills.load_skill(sid, sub) is None


# ------------------------------- the full loop -------------------------------


CONSUMER_RECORDING = {"history": [{
    "state": {"url": "http://app/section", "interacted_element": [
        {"node_name": "button", "ax_name": "PR/X/DR017",
         "attributes": {"type": "button"}, "x_path": "html/body/div[1]/button"}]},
    "model_output": {"action": [{"click": {"index": 5}}]},
    "result": [],
}]}


class _CapturingSession(FakeSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skills_seen = {}

    async def replay_segment(self, sub, sid, context, skill, gate):
        self.skills_seen[sid] = skill
        return await super().replay_segment(sub, sid, context, skill, gate)


async def test_commit_binds_created_ref_and_next_run_replays_fresh(
        stores, monkeypatch, capsys):
    """The live crm_data_request escape, end to end: the authoring run clicks the ref it
    just created (flagged as findings-sourced), the binder ties it to the create-write
    response, and the NEXT run replays the segment with ITS OWN fresh ref — no agent."""
    prompt = "go to the section. click on the same ref no and verify"
    # The consumer owns a marker, like the live subtask ("...click verify" gates on the
    # DataRequest write) — which also makes it an action node despite the judge-y verb.
    spec = TaskSpec(key="bind", prompt=prompt, subtasks=(
        SubtaskDecl(prompt="go to the section."),
        SubtaskDecl(prompt="click on the same ref no and verify",
                    marker="datarequests"),
    ))

    # ---- run 1: authoring. The fake create-POST reported DR017; the agent's finding
    # names it; its recording clicks it.
    fake = FakeSession(_runner(), agents=[
        _seg(True, mode="authored", finding="created request PR/X/DR017"),
        _seg(True, mode="authored"),
    ])
    fake.recording = CONSUMER_RECORDING
    fake.collectors = [_Net([_post("http://app/api/datarequests/111",
                                   {"refNo": "PR/X/DR017"})])]
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await run_hybrid_task(fake.runner or _runner(), prompt, spec=spec)
    assert result.is_successful is True

    ctx = ss.normalize_context("http://app/section")
    consumer_sid = ss.subtask_id(spec.subtasks[1].prompt, ctx)
    entry = ss.load_manifest()[consumer_sid]
    assert entry["bindings"]["bound_1"]["kind"] == "created"
    assert entry["bindings"]["bound_1"]["path"] == ["refNo"]
    assert entry["params"] == {"bound_1": "PR/X/DR017"}
    template = adapt.load_template(ss.template_path(consumer_sid))
    assert any("{{bound_1}}" in json.dumps(s) for s in template["steps"])
    assert "bound to this run's data" in capsys.readouterr().out

    # ---- run 2: this run's create-POST reports DR018; the bound entry must replay
    # with DR018 — the stale DR017 default must appear nowhere.
    fake2 = _CapturingSession(_runner(), replays=[_seg(True), _seg(True)])
    fake2.collectors = [_Net([_post("http://app/api/datarequests/222",
                                    {"refNo": "PR/X/DR018"})])]
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake2))
    result2 = await run_hybrid_task(fake2.runner or _runner(), prompt, spec=spec)

    assert result2.is_successful is True
    assert fake2.agent_calls == 0 and fake2.replay_calls == 2
    skill = fake2.skills_seen[consumer_sid]
    payload = (skill.params if skill.body == "code"
               else {"steps": skill.steps})
    assert "PR/X/DR018" in json.dumps(payload)
    assert "PR/X/DR017" not in json.dumps(payload)


async def test_bound_entry_bypasses_noted_data_wording_net(stores, monkeypatch):
    """The consumer-wording net ('the noted ref') forces always-agent for cached entries
    — but a BOUND entry re-resolves its dynamic values from this run's data, so the net
    stands down and the zero-LLM replay proceeds (fed by the producer's extract)."""
    prompt = "go to the section. open the noted ref and verify it"
    consumer_prompt = "open the noted ref and verify it"
    assert consumes_noted_data(consumer_prompt)   # the wording net does match
    spec = TaskSpec(key="net", prompt=prompt, subtasks=(
        SubtaskDecl(prompt="go to the section."),
        SubtaskDecl(prompt=consumer_prompt, marker="datarequests"),
    ))
    ctx = ss.normalize_context("http://app/section")
    sid = ss.subtask_id(consumer_prompt, ctx)
    ss.LIBRARY_DIR.mkdir(exist_ok=True)
    adapt.save_template(ss.template_path(sid), {
        "source_prompt": consumer_prompt,
        "params": {"bound_1": "PR/X/DR017"},
        "bindings": {"bound_1": {"kind": "extract", "label": "ref_no"}},
        "steps": [{"action": "find_click", "text": "{{bound_1}}"}],
    })
    ss.steps_path(sid).write_text(json.dumps([{"action": "find_click",
                                               "text": "{{bound_1}}"}]))
    ss.update_manifest(sid, consumer_prompt, params={"bound_1": "PR/X/DR017"},
                       bindings={"bound_1": {"kind": "extract", "label": "ref_no"}},
                       context=ctx)

    producer = _seg(True, mode="authored", finding="noted the ref")
    producer.extracted = {"ref_no": "PR/X/DR018"}
    fake = _CapturingSession(_runner(), agents=[producer], replays=[_seg(True)])
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await run_hybrid_task(fake.runner or _runner(), prompt, spec=spec)

    assert result.is_successful is True
    assert fake.replay_calls == 1                      # the consumer REPLAYED
    assert sid in ss.load_manifest()                   # and was not retired
    assert fake.skills_seen[sid].steps[0]["text"] == "PR/X/DR018"


# ---------------- transform bindings: line parts + date reformat (2026-08-12) ----------------


def test_parse_fuzzy_date_and_output_formats():
    from automation.pipeline.hybrid import _date_out_format, _parse_fuzzy_date
    d = _parse_fuzzy_date("Birthday\nMarch 2, 1979")
    assert d is not None and (d.day, d.month, d.year) == (2, 3, 1979)
    assert _parse_fuzzy_date("2 March 1979").month == 3
    assert _parse_fuzzy_date("1979-03-02").day == 2
    assert _parse_fuzzy_date("02/03/1979").month == 3      # day-first (UK app)
    assert _parse_fuzzy_date("no date here") is None
    assert _date_out_format("02/03/1979") == "%d/%m/%Y"
    assert _date_out_format("1979-03-02") == "%Y-%m-%d"
    assert _date_out_format("Euan Bruce") is None


def test_bind_runtime_values_line_and_date_transforms():
    """A typed value that is a LINE (or contiguous line run) of an extracted block, or
    a date REFORMAT of an extracted date, binds with a transform spec instead of
    refusing — the multi-field identity case."""
    extracts = {"identity_block": "Euan Bruce\n70 Telford Street\nBARFORD ST JOHN\nOX15 8PG",
                "dob_block": "Birthday\nMarch 2, 1979"}
    steps = [
        {"action": "fill", "selectors": ["css=#street"], "value": "70 Telford Street"},
        {"action": "fill", "selectors": ["css=#city"],
         "value": "BARFORD ST JOHN, OX15 8PG"},
        {"action": "fill", "selectors": ["css=#dob"], "value": "02/03/1979"},
    ]
    bound = _bind_runtime_values(
        steps, ["70 Telford Street", "BARFORD ST JOHN, OX15 8PG", "02/03/1979"],
        extracts, [])
    assert bound is not None
    new_steps, params, bindings = bound
    # longest-first numbering: city+postcode joined run, then street, then the date
    assert bindings["bound_1"] == {
        "kind": "extract", "label": "identity_block",
        "transform": {"line": {"index": 2, "count": 2, "join": ", "}}}
    assert bindings["bound_2"] == {
        "kind": "extract", "label": "identity_block",
        "transform": {"line": {"index": 1, "count": 1, "join": " "}}}
    assert bindings["bound_3"] == {
        "kind": "extract", "label": "dob_block", "transform": {"date": "%d/%m/%Y"}}
    assert new_steps[0]["value"] == "{{bound_2}}"
    assert new_steps[2]["value"] == "{{bound_3}}"


def test_binding_resolver_applies_transforms_to_fresh_values():
    """Replay derives the value from THIS run's fresh source; bounds/parse failures
    refuse (None) so a stale or wrong slice can never be typed."""
    fresh = {"identity_block": "Struan Boyd\n5 Long Acre\nLEEDS\nLS1 4AB",
             "dob_block": "Birthday\nJune 14, 1983"}
    resolve = _binding_resolver(fresh, SimpleNamespace())
    line = lambda i, n, j=" ": {"kind": "extract", "label": "identity_block",
                                "transform": {"line": {"index": i, "count": n, "join": j}}}
    assert resolve(line(0, 1)) == "Struan Boyd"
    assert resolve(line(1, 1)) == "5 Long Acre"
    assert resolve(line(2, 2, ", ")) == "LEEDS, LS1 4AB"
    assert resolve({"kind": "extract", "label": "dob_block",
                    "transform": {"date": "%d/%m/%Y"}}) == "14/06/1983"
    assert resolve(line(9, 1)) is None                       # out of bounds
    assert resolve({"kind": "extract", "label": "identity_block",
                    "transform": {"date": "%d/%m/%Y"}}) is None   # no date in source


async def test_binder_never_binds_a_value_to_the_segments_own_write(stores, monkeypatch,
                                                                    capsys):
    """A typed value may only bind to data that exists BEFORE the segment runs. Binding
    the Add-Employee form's street/first/last to fields of its OWN create-response made
    four bindings that can never resolve at load time (the POST has not happened yet) —
    the segment then re-authored with the LLM on every run, permanently
    (library/b5c40cab30815f2e.template.json, run 20260813_161952)."""
    ctx = ss.normalize_context("http://app/section")
    consumer_sid = ss.subtask_id(BOUND_SPEC.subtasks[1].prompt, ctx)
    monkeypatch.setattr(hybrid, "save_steps", _fill_steps_stub([
        {"action": "fill", "selectors": ["css=#street"], "value": "86 Seaford Road"},
        {"action": "click", "selectors": ['css=[id="save"]']},
    ]))
    # The ONLY structured source naming that street is the segment's own create-write,
    # which fires DURING the segment (it is the Save the segment performs).
    own_write = _post("/payroll/clients/x/employees/",
                      {"result": {"address": {"street": "86 Seaford Road"}}})

    class _WritesDuringSegment(FakeSession):
        async def agent_segment(self, sub, *a, **kw):
            seg = await super().agent_segment(sub, *a, **kw)
            if "add employee" in sub.instantiated_prompt:
                self.collectors[0]._requests.append(own_write)   # ITS OWN Save
            return seg

    fake = _WritesDuringSegment(_runner(), agents=[
        _seg(True, mode="authored", finding="identity noted",
             extracted={"identity_block": "Jude Williamson"}),
        _seg(True, mode="authored"),
    ])
    fake.collectors = [_Net([])]
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await run_hybrid_task(fake.runner or _runner(), BOUND_PROMPT,
                                   spec=BOUND_SPEC)

    assert result.is_successful is True
    entry = ss.load_manifest().get(consumer_sid)
    # No entry at all, or one with no self-referential binding — never a `created`
    # binding pointing at this segment's own POST.
    specs = list((entry or {}).get("bindings", {}).values())
    assert not [s for s in specs if s.get("kind") == "created"]


def test_word_slice_bindings_survive_a_fresh_identity():
    """Forms SPLIT what pages JOIN: the identity block has "Jude Williamson" on one
    line, the form has separate First/Last fields. A word slice binds those, and the
    spec re-derives the NEXT run's names from the same positions."""
    authored = {"name": "Jude Williamson\n86 Seaford Road\nCURRIDGE\nRG18 1BA"}
    first = hybrid._extract_transform_spec("Jude", authored)
    last = hybrid._extract_transform_spec("Williamson", authored)
    street = hybrid._extract_transform_spec("Seaford Road", authored)
    building = hybrid._extract_transform_spec("86", authored)
    assert first["transform"]["words"] == {"start": 0, "count": 1}
    assert last["transform"]["words"] == {"start": 1, "count": 1}
    assert street["transform"] == {"line": {"index": 1, "count": 1, "join": " "},
                                   "words": {"start": 1, "count": 2}}
    # The house number is 2 characters and binds too — short values are fine as long as
    # exactly one word position matches (see the ambiguity rule below).
    assert building["transform"] == {"line": {"index": 1, "count": 1, "join": " "},
                                     "words": {"start": 0, "count": 1}}

    # Next run, different person, same positions.
    fresh = {"name": "Arron McIntosh\n41 Gloucester Road\nCILRHEDYN\nSA35 5WS"}
    resolve = hybrid._binding_resolver(fresh, SimpleNamespace())
    assert resolve(first) == "Arron"
    assert resolve(last) == "McIntosh"
    assert resolve(street) == "Gloucester Road"
    # Out-of-range slices refuse rather than typing something wrong.
    assert resolve({"kind": "extract", "label": "name",
                    "transform": {"line": {"index": 0, "count": 1, "join": " "},
                                  "words": {"start": 5, "count": 1}}}) is None


def test_a_value_the_task_spells_anywhere_is_prompt_data_not_runtime():
    """The gender case (run 20260814_100546): subtask 1 says "note ... the Gender
    (Male)" and subtask 3 says only "enter the noted name, address, and gender". Judging
    "Male" against the CURRENT slice's wording alone made it runtime data — unbindable,
    since no extract holds it — and ONE unbindable value refuses the whole segment's
    commit, so the Add-Employee recording never cached. The whole task's wording is the
    right corpus (the create-write leg already uses it)."""
    from automation.pipeline.hybrid import _findings_sourced_values

    steps = [{"action": "fill", "selectors": ["css=#g"], "value": "Male"},
             {"action": "fill", "selectors": ["css=#n"], "value": "Riley"}]
    findings = ["identity noted: Name: Riley Moore Gender: Male Address: 4 Fox Lane"]
    this_slice = "click Add Employee and fill in the form using the noted details"
    whole_task = this_slice + " \n From the generated identity note the Gender (Male)"

    assert "Male" in _findings_sourced_values(steps, this_slice, findings)
    assert "Male" not in _findings_sourced_values(steps, whole_task, findings)
    # A genuinely run-generated value stays flagged either way.
    assert "Riley" in _findings_sourced_values(steps, whole_task, findings)


def test_short_values_bind_when_the_word_match_is_unambiguous():
    """A house number is 2 characters ("48 Main St"). A blanket length floor left it
    unbindable, and one unbindable value refuses the whole segment — which is what kept
    the Add-Employee recording out of the library in run 20260814_114028. Short values
    bind when exactly ONE word position matches; ambiguity refuses rather than guesses."""
    src = {"generated_identity": "Ty Dickson\n48 Main St\nACHAGLASS\nPA29 2XN"}
    spec = hybrid._extract_transform_spec("48", src)
    assert spec["transform"] == {"line": {"index": 1, "count": 1, "join": " "},
                                 "words": {"start": 0, "count": 1}}
    resolve = hybrid._binding_resolver(
        {"generated_identity": "Ana Rowe\n7 Kirk Way\nDUNDEE\nDD1 4XX"}, SimpleNamespace())
    assert resolve(spec) == "7"

    # Ambiguous: the same token appears twice — refuse, never guess which one.
    assert hybrid._extract_transform_spec("St", {"a": "1 St Mary\nSt Andrews"}) is None


def test_the_binder_matches_its_source_ignoring_case():
    """fakenamegenerator prints the TOWN in UK postal ALL-CAPS ("HOOTON", "BUTT GREEN")
    while the agent often types it title-cased into the form. Case-sensitive matching gave
    that one value no structured source, and ONE unbindable value refuses the whole
    commit — so the Add-Employee recording kept re-authoring at ~300-480k tokens a run
    (measured 2026-08-24 across three live runs; the same refusal cost run 20260819_145019
    its entry). The binding stores the SOURCE, so a replay types the page's own casing."""
    ident = {"identity": "Paul Hughes\n38 Scotswood Road\nHOOTON\nL66 1BW"}

    # A whole line, and a word run inside one, both bind regardless of how it was typed.
    for typed in ("HOOTON", "Hooton", "hooton"):
        spec = hybrid._extract_transform_spec(typed, ident)
        assert spec == {"kind": "extract", "label": "identity",
                        "transform": {"line": {"index": 2, "count": 1, "join": " "}}}, typed
    assert hybrid._extract_transform_spec("scotswood road", ident)["transform"] == {
        "line": {"index": 1, "count": 1, "join": " "}, "words": {"start": 1, "count": 2}}

    # The replay re-derives from the SOURCE, so it types the page's casing, not the
    # authoring run's.
    resolve = hybrid._binding_resolver(
        {"identity": "Cyrus Simpson\n53 Colorado Way\nRHYD-Y-FRO\nSA8 9GQ"},
        SimpleNamespace())
    assert resolve(hybrid._extract_transform_spec("Hooton", ident)) == "RHYD-Y-FRO"

    # Whole-value equality (the fast path in _bind_runtime_values) is case-blind too.
    steps = [{"action": "fill", "selectors": ["css=#t"], "value": "Hooton"}]
    bound = _bind_runtime_values(steps, ["Hooton"], {"town": "HOOTON"}, [])
    assert bound is not None
    rewritten, params, bindings = bound
    assert rewritten[0]["value"] == "{{bound_1}}"
    assert bindings["bound_1"] == {"kind": "extract", "label": "town"}
    assert params["bound_1"] == "Hooton"


def test_drop_unattributable_fills_removes_only_the_invented_values():
    """Unit view of the 2026-08-24 gate relaxation: only the typed steps carrying an
    unattributable value go; clicks, selects the task asked for, and every other fill
    survive untouched."""
    steps = [
        {"action": "fill", "selectors": ["css=#first"], "value": "Paul"},
        {"action": "fill", "selectors": ["css=#town"], "value": "HOOTON"},
        {"action": "fill", "selectors": ["css=#county"], "value": "Merseyside"},
        {"action": "type", "selectors": ["css=#note"], "text": "Merseyside"},
        {"action": "select", "selectors": ["css=#cat"], "value": "A"},
        {"action": "click", "selectors": ['text="Save"']},
    ]
    kept, dropped = hybrid._drop_unattributable_fills(steps, ["Merseyside"])
    assert dropped == ["Merseyside", "Merseyside"]      # the fill AND the type
    assert [s.get("value") or s.get("text") for s in kept] == [
        "Paul", "HOOTON", "A", None]
    # Nothing to drop is a no-op that returns the list unchanged.
    assert hybrid._drop_unattributable_fills(steps, []) == (steps, [])
