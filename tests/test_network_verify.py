"""Network-outcome click receipts: the collector's live write records become the step's
verification signal.

Motivating failure (run 20260805_142523_388784): the May-26 FPS submission POSTed and the
server answered `"isSubmitted": true` — while the agent, reading a stale "No employees FPS
submitted so far" list, concluded the submission failed and re-submitted the whole company
against April (bounced: `"status": false, "message": "… already submitted"`). Both
verdicts sat in the NetworkCollector's memory the entire time. Clicks now report the
write requests they fired — method, status, and a short server-verdict extract from the
captured body — and an in-dialog click that fired NO write says so (the swallowed-save
flag). Condition-based: the receipt waits for the triggered request to settle instead of
the agent burning wait-steps."""
import json
import time
from pathlib import Path
from types import SimpleNamespace

from test_agent_tools import _FakeBrowserSession, _FakeDomNode, _dialog_states

from automation.collectors.network import NetworkCollector
from automation.pipeline import agent_tools


# ------------------------------- collector.writes_since -------------------------------


def _collector(tmp_path):
    return NetworkCollector(SimpleNamespace(), Path(tmp_path))


class _Req:
    """Hashable stand-in for playwright's Request (SimpleNamespace defines __eq__ and
    loses default hashing; the collector keys dicts/sets by the request object)."""

    def __init__(self, method="POST", url="https://api.app/Years/27/FPS", rtype="xhr"):
        self.url, self.method, self.resource_type, self.headers = url, method, rtype, {}


def _request(method="POST", url="https://api.app/Years/27/FPS", rtype="xhr"):
    return _Req(method, url, rtype)


def _respond(collector, req, status=200, headers=None):
    collector._on_response(SimpleNamespace(request=req, status=status, status_text="",
                                           headers=headers or {}))


def test_writes_since_returns_only_writes_started_after_t0(tmp_path):
    c = _collector(tmp_path)
    c._active = True
    early = _request()
    c._on_request(early)
    t0 = time.monotonic()
    c._on_request(_request(method="GET", url="https://api.app/Settings"))  # read: excluded
    late = _request(url="https://api.app/Years/27/FPS")
    c._on_request(late)

    writes = c.writes_since(t0)
    assert [w["record"]["url"] for w in writes] == ["https://api.app/Years/27/FPS"]
    assert writes[0]["settled"] is False

    _respond(c, late, status=200)
    assert c.writes_since(t0)[0]["settled"] is True


def test_writes_since_excludes_out_of_scope_requests(tmp_path):
    c = _collector(tmp_path)
    c._active = True
    t0 = time.monotonic()
    req = _request()
    c._skipped.add(req)
    c._on_request_finished(req)
    assert c.writes_since(t0) == []


# ------------------------------- verdict extraction -------------------------------

_MAY_BODY = json.dumps({"executionTime": 1.19, "result": {
    "periodName": "May-26",
    "employees": [{"id": "x", "name": "Aleksander Millar"}],
    "submitDetail": {"id": "y", "isSubmitted": True, "status": 2}}})

_APRIL_BODY = json.dumps({
    "executionTime": 0.04, "status": False,
    "message": "Aaron Wilson's FPS of this period is already submitted. "
               "Use related reason to correction the FPS.",
    "errors": [{"message": "Aaron Wilson's FPS of this period is already submitted. "
                           "Use related reason to correction the FPS."}]})


def test_verdict_finds_nested_positive_signal():
    negative, text = agent_tools._write_verdict({"body": _MAY_BODY})
    assert negative is False
    assert "isSubmitted: true" in text


def test_verdict_surfaces_server_refusal():
    negative, text = agent_tools._write_verdict({"body": _APRIL_BODY})
    assert negative is True
    assert "already submitted" in text


def test_verdict_absent_without_body():
    assert agent_tools._write_verdict({}) is None
    assert agent_tools._write_verdict({"body": "<html>not json</html>"}) is None


# ------------------------------- click receipt wiring -------------------------------


class _FakeLiveNetwork:
    def __init__(self, writes):
        self._writes = writes

    def writes_since(self, t0):
        return self._writes


def _write_snapshot(status=200, body=None, url="https://api.app/Years/27/FPS",
                    settled=True):
    record = {"url": url, "method": "POST", "status": status, "duration_ms": 97.0,
              "failed": False, "errorText": None,
              "response_headers": {"content-type": "application/json"}}
    if body is not None:
        record["body"] = body
    return {"record": record, "started": 0.0, "settled": settled}


async def _fake_builtin_click(params=None, browser_session=None):
    from browser_use.agent.views import ActionResult

    return ActionResult(extracted_content='Clicked button "Submit FPS"')


def _speed(monkeypatch):
    monkeypatch.setattr(agent_tools, "_WRITE_SNIFF_S", 0.05)
    monkeypatch.setattr(agent_tools, "_WRITE_SETTLE_S", 0.2)
    monkeypatch.setattr(agent_tools, "_BODY_POLL_S", 0.05)


async def test_click_reports_write_with_server_refusal(monkeypatch):
    _speed(monkeypatch)
    _dialog_states(monkeypatch, {"in_dialog": True, "open": 1},
                   {"in_dialog": False, "open": 0})
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK",
                        _FakeLiveNetwork([_write_snapshot(body=_APRIL_BODY)]))
    res = await agent_tools._click_with_dialog_outcome(
        _fake_builtin_click, SimpleNamespace(index=4),
        _FakeBrowserSession({4: _FakeDomNode("Submit FPS")}))

    msg = res.extracted_content
    assert "POST" in msg and "/Years/27/FPS" in msg and "200" in msg
    assert "REFUSED" in msg
    assert "already submitted" in msg
    assert "dialog CLOSED" in msg          # both receipts coexist, network first


async def test_click_reports_clean_write_with_positive_verdict(monkeypatch):
    _speed(monkeypatch)
    _dialog_states(monkeypatch, {"in_dialog": False, "open": 0}, None)
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK",
                        _FakeLiveNetwork([_write_snapshot(
                            body=_MAY_BODY, url="https://api.app/E/PayrollCalculation")]))
    res = await agent_tools._click_with_dialog_outcome(
        _fake_builtin_click, SimpleNamespace(index=4),
        _FakeBrowserSession({4: _FakeDomNode("Save & Next")}))

    msg = res.extracted_content
    assert "PayrollCalculation" in msg and "200" in msg
    assert "isSubmitted: true" in msg


async def test_dialog_click_with_no_write_flags_it(monkeypatch):
    _speed(monkeypatch)
    _dialog_states(monkeypatch, {"in_dialog": True, "open": 1},
                   {"in_dialog": True, "open": 1})
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", _FakeLiveNetwork([]))
    res = await agent_tools._click_with_dialog_outcome(
        _fake_builtin_click, SimpleNamespace(index=4),
        _FakeBrowserSession({4: _FakeDomNode("Save")}))

    msg = res.extracted_content
    assert "no write request was observed after this click" in msg
    assert "nothing reached the server" not in msg   # observation, never a failure verdict
    assert "genuinely absent" in msg                 # the verify-by-state directive
    assert "STILL OPEN" in msg


# ------------------------- write-verdict vs dialog reconciliation -------------------------
# Run 20260807_095537: one receipt printed 'POST /emails → 200' AND 'likely did NOT go
# through' — the agent trusted the scary half and spent 14 steps redoing a send that had
# succeeded. When the fired writes prove the action landed, the still-open advisory must
# say so instead of alleging failure.


def _still_open(monkeypatch):
    _speed(monkeypatch)
    monkeypatch.setattr(agent_tools, "_DIALOG_WRITE_SETTLE_S", 0)
    _dialog_states(monkeypatch, {"in_dialog": True, "open": 1},
                   {"in_dialog": True, "open": 1})


async def test_accepted_write_overrides_still_open_pessimism(monkeypatch):
    _still_open(monkeypatch)
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK",
                        _FakeLiveNetwork([_write_snapshot(body=_MAY_BODY)]))
    res = await agent_tools._click_with_dialog_outcome(
        _fake_builtin_click, SimpleNamespace(index=4),
        _FakeBrowserSession({4: _FakeDomNode("Send")}))

    msg = res.extracted_content
    assert "STILL OPEN" in msg
    assert "SUCCEEDED" in msg and "do NOT redo" in msg
    assert "likely did NOT go through" not in msg


async def test_refused_write_keeps_still_open_pessimism(monkeypatch):
    _still_open(monkeypatch)
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK",
                        _FakeLiveNetwork([_write_snapshot(body=_APRIL_BODY)]))
    res = await agent_tools._click_with_dialog_outcome(
        _fake_builtin_click, SimpleNamespace(index=4),
        _FakeBrowserSession({4: _FakeDomNode("Submit FPS")}))

    msg = res.extracted_content
    assert "REFUSED" in msg                      # the server verdict stays front and center
    assert "likely did NOT go through" in msg    # and the pessimism is warranted


def test_writes_accepted_flag_cases():
    ok = _write_snapshot(body=_MAY_BODY)
    assert agent_tools._writes_accepted([ok]) is True
    assert agent_tools._writes_accepted([_write_snapshot(body=_APRIL_BODY)]) is False
    in_flight = _write_snapshot(status=None, settled=False)
    assert agent_tools._writes_accepted([in_flight]) is False
    assert agent_tools._writes_accepted([ok, in_flight]) is False   # partial ≠ proof
    failed = _write_snapshot()
    failed["record"]["failed"] = True
    assert agent_tools._writes_accepted([failed]) is False
    assert agent_tools._writes_accepted([_write_snapshot(status=500)]) is False
    assert agent_tools._writes_accepted([_write_snapshot()]) is True  # bare 2xx, no body


async def test_plain_click_without_writes_keeps_receipt_unchanged(monkeypatch):
    _speed(monkeypatch)
    _dialog_states(monkeypatch, {"in_dialog": False, "open": 0}, None)
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", _FakeLiveNetwork([]))
    res = await agent_tools._click_with_dialog_outcome(
        _fake_builtin_click, SimpleNamespace(index=4),
        _FakeBrowserSession({4: _FakeDomNode("Employees")}))

    assert res.extracted_content == 'Clicked button "Submit FPS"'


async def test_no_bridge_means_no_change(monkeypatch):
    _dialog_states(monkeypatch, {"in_dialog": False, "open": 0}, None)
    agent_tools.set_live_network(_FakeLiveNetwork([_write_snapshot()]))
    agent_tools.clear_live_network()
    res = await agent_tools._click_with_dialog_outcome(
        _fake_builtin_click, SimpleNamespace(index=4),
        _FakeBrowserSession({4: _FakeDomNode("Save")}))

    assert res.extracted_content == 'Clicked button "Submit FPS"'


# ------------------- structured write_outcome stamping (roll-up source) -------------------
# The gate's receipt roll-up must never regex-parse receipt prose; the click paths stamp
# a small structured verdict instead.


async def _fake_builtin_click_with_meta(**kwargs):
    from browser_use.agent.views import ActionResult

    return ActionResult(extracted_content='Clicked button "Save"',
                        metadata={"hidden_click": True})


async def test_click_stamps_refused_write_outcome(monkeypatch):
    _speed(monkeypatch)
    _dialog_states(monkeypatch, {"in_dialog": True, "open": 1},
                   {"in_dialog": False, "open": 0})
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK",
                        _FakeLiveNetwork([_write_snapshot(body=_APRIL_BODY)]))
    res = await agent_tools._click_with_dialog_outcome(
        _fake_builtin_click, SimpleNamespace(index=4),
        _FakeBrowserSession({4: _FakeDomNode("Submit FPS")}))
    wo = (res.metadata or {}).get("write_outcome")
    assert wo is not None
    assert wo["fired"] is True and wo["accepted"] is False
    assert isinstance(wo["t0"], float)


async def test_click_stamps_accepted_write_outcome_and_merges_metadata(monkeypatch):
    _speed(monkeypatch)
    _dialog_states(monkeypatch, {"in_dialog": False, "open": 0}, None)
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK",
                        _FakeLiveNetwork([_write_snapshot(
                            body=_MAY_BODY, url="https://api.app/E/PayrollCalculation")]))
    res = await agent_tools._click_with_dialog_outcome(
        _fake_builtin_click_with_meta, SimpleNamespace(index=4),
        _FakeBrowserSession({4: _FakeDomNode("Save & Next")}))
    assert res.metadata["hidden_click"] is True          # sibling preserved (merge, not replace)
    assert res.metadata["write_outcome"]["accepted"] is True


async def test_plain_click_without_write_stamps_nothing(monkeypatch):
    _speed(monkeypatch)
    _dialog_states(monkeypatch, {"in_dialog": False, "open": 0}, None)
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", _FakeLiveNetwork([]))
    res = await agent_tools._click_with_dialog_outcome(
        _fake_builtin_click, SimpleNamespace(index=4),
        _FakeBrowserSession({4: _FakeDomNode("Expand row")}))
    assert not (getattr(res, "metadata", None) or {}).get("write_outcome")


def test_with_write_outcome_merges_not_replaces():
    outcome = {"fired": True, "accepted": False, "t0": 1.0}
    merged = agent_tools._with_write_outcome({"interacted_element": {"a": 1}}, outcome)
    assert merged["interacted_element"] == {"a": 1}
    assert merged["write_outcome"] is outcome
    assert agent_tools._with_write_outcome(None, None) is None
    assert agent_tools._with_write_outcome({"x": 1}, None) == {"x": 1}
    assert agent_tools._with_write_outcome(None, outcome) == {"write_outcome": outcome}


# ---------------- fail_and_stop contradiction bounce + toggle doubt suppression ----------------
# Run 20260807_164137 seg2: the Save receipt said "the write above SUCCEEDED: do NOT redo"
# and the agent called fail_and_stop one step later claiming nothing was saved. The bounce
# refuses that ONCE per segment; toggle clicks stop receiving save/submit doubt language
# (step 13's switch click primed the false "form is broken" narrative).


def _accepted_business_write():
    return _write_snapshot(
        body=_MAY_BODY,
        url="https://api.app/Payroll/Clients/6a/Employees/?yearId=27")


async def test_fail_and_stop_bounces_once_on_accepted_segment_write():
    agent_tools.set_live_network(_FakeLiveNetwork([_accepted_business_write()]))
    try:
        first = await agent_tools._fail_and_stop_result("employee not created")
        assert first.error and "REFUSED" in first.error
        assert "/Employees/" in first.error and "200" in first.error
        assert not first.is_done
        second = await agent_tools._fail_and_stop_result("employee not created")
        assert second.is_done is True and second.success is False
    finally:
        agent_tools.clear_live_network()


async def test_fail_and_stop_ignores_infra_writes():
    infra = _write_snapshot(url="https://api.app/auth/webpush")
    agent_tools.set_live_network(_FakeLiveNetwork([infra]))
    try:
        res = await agent_tools._fail_and_stop_result("nothing worked")
        assert res.is_done is True and res.success is False
    finally:
        agent_tools.clear_live_network()


async def test_fail_and_stop_ignores_refused_and_unsettled_writes():
    refused = _write_snapshot(body=_APRIL_BODY)
    inflight = _write_snapshot(status=None, settled=False)
    agent_tools.set_live_network(_FakeLiveNetwork([refused, inflight]))
    try:
        res = await agent_tools._fail_and_stop_result("blocked")
        assert res.is_done is True and res.success is False
    finally:
        agent_tools.clear_live_network()


async def test_fail_and_stop_without_collector_passes_through():
    agent_tools.clear_live_network()
    res = await agent_tools._fail_and_stop_result("cannot proceed")
    assert res.is_done is True and res.success is False


async def test_fail_and_stop_bounce_resets_per_segment():
    agent_tools.set_live_network(_FakeLiveNetwork([_accepted_business_write()]))
    try:
        first = await agent_tools._fail_and_stop_result("claim A")
        assert first.error and "REFUSED" in first.error
        # New segment: the guard re-arms.
        agent_tools.set_live_network(_FakeLiveNetwork([_accepted_business_write()]))
        again = await agent_tools._fail_and_stop_result("claim B")
        assert again.error and "REFUSED" in again.error
    finally:
        agent_tools.clear_live_network()


async def test_toggle_click_in_dialog_gets_no_save_doubt(monkeypatch):
    _speed(monkeypatch)
    _dialog_states(monkeypatch, {"in_dialog": True, "open": 1},
                   {"in_dialog": True, "open": 1})
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", _FakeLiveNetwork([]))
    toggle = _FakeDomNode("Student Loan", attributes={"role": "switch"})
    res = await agent_tools._click_with_dialog_outcome(
        _fake_builtin_click, SimpleNamespace(index=4),
        _FakeBrowserSession({4: toggle}))
    msg = res.extracted_content
    assert "no write request was observed" not in msg
    assert "likely did NOT go through" not in msg
    assert "STILL OPEN" not in msg


async def test_toggle_click_that_fires_write_keeps_receipt(monkeypatch):
    _speed(monkeypatch)
    _dialog_states(monkeypatch, {"in_dialog": True, "open": 1},
                   {"in_dialog": True, "open": 1})
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK",
                        _FakeLiveNetwork([_write_snapshot(body=_MAY_BODY)]))
    toggle = _FakeDomNode("Auto enrol", attributes={"role": "switch"})
    res = await agent_tools._click_with_dialog_outcome(
        _fake_builtin_click, SimpleNamespace(index=4),
        _FakeBrowserSession({4: toggle}))
    msg = res.extracted_content
    assert "POST" in msg and "200" in msg
    assert (res.metadata or {}).get("write_outcome", {}).get("accepted") is True


# ---------------- verify_save_registered: did MY last action write? ----------------
# Run 20260824_165824 seg 4 left Aayan Dickson with two or three £4,000 payments. The tool
# could not have prevented it: no slice of that task declares a marker, so runner.py only
# installs _SAVE_PROBE `if success_marker`, and the tool answered "not available for this
# run; verify via the UI instead" — which is how the agent ended up reading search_page,
# mistaking the row it had just created for a pre-existing one ("from existing entry; did
# not add new payment"), and re-entering. Even when installed the probe returned the
# SEGMENT's first create-write, so save #1 vouched for save #3 forever.


class _WindowedLiveNetwork:
    """writes_since that actually honours t0 — the fake above ignores it, which is fine
    for receipts but would hide the windowing regression this section exists to catch."""

    def __init__(self, writes):
        self._writes = writes

    def writes_since(self, t0):
        return [w for w in self._writes if w["started"] >= t0]


def _at(started, **kw):
    w = _write_snapshot(**kw)
    w["started"] = started
    return w


async def test_accepted_write_since_the_last_action_is_confirmed(monkeypatch):
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", _WindowedLiveNetwork([_at(10.0)]))
    monkeypatch.setattr(agent_tools, "_LAST_ACTION_T0", 5.0)

    msg = await agent_tools._verify_last_action_write()
    assert msg.startswith("CONFIRMED")
    assert "do NOT repeat it" in msg


async def test_a_write_from_before_the_last_action_does_not_confirm(monkeypatch):
    """THE duplicate regression: employee 1's save must not vouch for employee 3's. The old
    probe returned the segment's first create-write and so always said CONFIRMED here."""
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", _WindowedLiveNetwork([_at(1.0)]))
    monkeypatch.setattr(agent_tools, "_LAST_ACTION_T0", 5.0)   # the save came later

    msg = await agent_tools._verify_last_action_write()
    assert msg.startswith("UNCONFIRMED")


async def test_no_write_is_unconfirmed_and_never_orders_a_resave(monkeypatch):
    """A client-staged save fires nothing. Saying "the Save did NOT go through — save
    again" here is a duplicate-add instruction, so the message must refuse to say it."""
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", _WindowedLiveNetwork([]))
    monkeypatch.setattr(agent_tools, "_LAST_ACTION_T0", 5.0)
    monkeypatch.setattr(agent_tools, "_VERIFY_SNIFF_S", 0.05)   # don't wait in tests

    msg = await agent_tools._verify_last_action_write()
    assert msg.startswith("UNCONFIRMED")
    assert "not proof of failure" in msg.lower()
    assert "DUPLICATE" in msg
    assert "save again" not in msg.lower()
    assert "did not go through" not in msg.lower()


async def test_a_refused_write_is_named_as_refused(monkeypatch):
    """2xx whose body is a refusal (the FPS "already submitted" case). Distinct from
    UNCONFIRMED: something DID reach the server, so the fix is to read its verdict."""
    body = json.dumps({"status": False, "message": "FPS already submitted for this period"})
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK",
                        _WindowedLiveNetwork([_at(10.0, body=body)]))
    monkeypatch.setattr(agent_tools, "_LAST_ACTION_T0", 5.0)

    msg = await agent_tools._verify_last_action_write()
    assert msg.startswith("REFUSED")
    # The server's OWN words must reach the agent, or "fix what it names" is unactionable.
    assert "FPS already submitted for this period" in msg
    assert "DID reach the server" in msg


async def test_a_refusal_body_that_lands_during_the_poll_is_not_confirmed(monkeypatch):
    """The verifier must wait for the response BODY, not just for the request to settle.
    `_write_verdict` reads record["body"], so a settled 2xx whose body has not arrived yet
    looks ACCEPTED to `_writes_accepted` — and this app's refusals are 2xx-with-a-body
    ("already submitted"). An earlier hand-rolled copy of the polling here skipped that
    phase and answered CONFIRMED on exactly that shape.

    The fake mutates the record IN PLACE, which is what NetworkCollector.writes_since
    documents ("`record` is the LIVE aggregation dict ... callers re-poll rather than
    copy") — the poll loop re-reads the same dict instead of re-fetching."""
    import asyncio

    monkeypatch.setattr(agent_tools, "_WRITE_SETTLE_S", 0.2)
    monkeypatch.setattr(agent_tools, "_BODY_POLL_S", 0.5)
    now = time.monotonic()
    w = _at(now)                                  # settled 2xx, JSON type, body not in yet
    assert "body" not in w["record"]
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", _WindowedLiveNetwork([w]))
    monkeypatch.setattr(agent_tools, "_LAST_ACTION_T0", now - 0.01)

    body = json.dumps({"status": False, "message": "already submitted"})
    asyncio.get_running_loop().call_later(
        0.15, lambda: w["record"].__setitem__("body", body))

    msg = await agent_tools._verify_last_action_write()
    assert msg.startswith("REFUSED"), msg
    assert "already submitted" in msg


async def test_a_body_that_never_lands_does_not_stall_past_the_deadline(monkeypatch):
    """The body wait is bounded: a 2xx whose JSON body never arrives still answers, on the
    evidence available, rather than hanging the agent's step."""
    monkeypatch.setattr(agent_tools, "_WRITE_SETTLE_S", 0.1)
    monkeypatch.setattr(agent_tools, "_BODY_POLL_S", 0.2)
    now = time.monotonic()
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", _WindowedLiveNetwork([_at(now)]))
    monkeypatch.setattr(agent_tools, "_LAST_ACTION_T0", now - 0.01)

    started = time.monotonic()
    msg = await agent_tools._verify_last_action_write()
    assert msg.startswith("CONFIRMED")            # no refusal evidence to act on
    assert time.monotonic() - started < 2.0       # bounded, not the 6s sniff window


async def test_probe_failure_degrades_to_unconfirmed(monkeypatch):
    class _Boom:
        def writes_since(self, t0):
            raise RuntimeError("collector gone")

    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", _Boom())
    msg = await agent_tools._verify_last_action_write()
    assert msg.startswith("UNCONFIRMED")     # never raises into the run


async def test_tool_answers_from_the_live_collector_without_any_marker(monkeypatch):
    """Cause 1, asserted end to end: a task that declares no marker used to get no save
    probe at all and be told "not available for this run; verify via the UI instead" — which
    is how the agent ended up eyeballing search_page and re-entering a saved payment. The
    live collector is installed unconditionally, so the tool now answers regardless."""
    from test_agent_tools import _registered_action

    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", _WindowedLiveNetwork([_at(10.0)]))
    monkeypatch.setattr(agent_tools, "_LAST_ACTION_T0", 5.0)

    fn, _model = _registered_action("verify_save_registered")
    res = await fn()
    assert "CONFIRMED" in res.extracted_content
    assert "not available for this run" not in res.extracted_content


async def test_stamp_action_windows_the_verifier_to_the_current_action(monkeypatch):
    """The stamp is what makes the window per-action; the click wrappers set it."""
    monkeypatch.setattr(agent_tools, "_LAST_ACTION_T0", 0.0)
    first = agent_tools._stamp_action()
    second = agent_tools._stamp_action()
    assert second >= first
    assert agent_tools._LAST_ACTION_T0 == second


# ------------------------ page-load attribution (`after_page_load`) ------------------------
# A page issues its own boot traffic when it loads — subscriptions, feature probes, push
# registration. None of it is the SEGMENT's work, and one such POST (Addons/MSTeams/
# Subscribe, 200 with a refusing body) failed run 20260901_093026's Pay Forecast segment,
# whose only navigation was the "refresh the page" the subtask asked for. The collector
# marks every request issued since the last document load and NOT yet preceded by an
# interaction; `note_interaction` (one call per acting verb, one per replayed step) is what
# ends the boot window.


class _NavReq(_Req):
    def __init__(self, url="https://app/paye/calculator"):
        super().__init__(method="GET", url=url, rtype="document")

    def is_navigation_request(self):
        return True


def test_requests_after_a_document_load_are_marked_page_load(tmp_path):
    c = _collector(tmp_path)
    c._active = True
    before = _request(url="https://api.app/Save")
    c._on_request(before)
    c._on_request(_NavReq())
    boot = _request(url="https://api.app/Addons/MSTeams/Subscribe")
    c._on_request(boot)

    by_url = {r["url"]: r for r in c.results()["requests"]}
    assert by_url["https://api.app/Save"]["after_page_load"] is False
    assert by_url["https://api.app/Addons/MSTeams/Subscribe"]["after_page_load"] is True


def test_an_interaction_ends_the_page_load_window(tmp_path):
    c = _collector(tmp_path)
    c._active = True
    c._on_request(_NavReq())
    c.note_interaction()
    save = _request(url="https://api.app/Save")
    c._on_request(save)

    record, = [r for r in c.results()["requests"] if r["url"].endswith("/Save")]
    assert record["after_page_load"] is False


def test_an_iframe_document_does_not_open_a_page_load_window(tmp_path):
    """Only the top page's own load is a page load — an embedded document proves nothing
    about what the segment did."""
    c = _collector(tmp_path)
    c._active = True
    page = SimpleNamespace(url="https://app/paye", main_frame="main")
    frame_doc = _NavReq(url="https://app/embed")
    frame_doc.frame = "iframe"
    c._on_request(frame_doc, page)
    after = _request(url="https://api.app/Save")
    c._on_request(after, page)

    record, = [r for r in c.results()["requests"] if r["url"].endswith("/Save")]
    assert record["after_page_load"] is False


# ---------------- the bounce must read the LATEST write, not the first (2026-09-01) -------
# Run 20260901_163709 subtask 8 ("click Next for all the remaining employees, then Submit,
# then close this tab"): the agent advanced 11 employees (11 accepted finalSubmit=false
# writes), clicked Submit, and the server REFUSED it — 200 with "Unable to sent email to
# client." The agent honestly called fail_and_stop. The bounce stepped OVER that refusal,
# found the FIRST of the 11 routine advances, and told the agent the record exists and to
# continue. With the portal tab already closed it re-opened the review panel in the app and
# re-entered all 11 employees through the agent-side screen (agentEntry=true) before the
# run was killed by hand. A refusal that lands AFTER the last accepted write is evidence
# the claim is TRUE — it must never be the thing that refutes it.


def _refused_submit():
    return _write_snapshot(
        body=_APRIL_BODY,
        url="https://api.app/CalculationDataRequest/6a96/Employee/6a91?finalSubmit=true")


async def test_fail_and_stop_does_not_bounce_when_the_latest_write_was_refused():
    agent_tools.set_live_network(_FakeLiveNetwork(
        [_accepted_business_write(), _accepted_business_write(), _refused_submit()]))
    try:
        res = await agent_tools._fail_and_stop_result("Unable to send email to client")
        assert res.is_done is True and res.success is False
        assert res.error == "Unable to send email to client"
    finally:
        agent_tools.clear_live_network()


async def test_fail_and_stop_bounce_cites_the_latest_accepted_write():
    """When the segment's last word IS an acceptance, the bounce still fires — and cites
    the write the agent is actually reacting to (the most recent), not the oldest."""
    early = _write_snapshot(body=_MAY_BODY, url="https://api.app/Years/27/EARLY")
    latest = _write_snapshot(body=_MAY_BODY, url="https://api.app/Years/27/LATEST")
    agent_tools.set_live_network(_FakeLiveNetwork([early, latest]))
    try:
        res = await agent_tools._fail_and_stop_result("nothing saved")
        assert res.error and "REFUSED" in res.error
        assert "LATEST" in res.error and "EARLY" not in res.error
    finally:
        agent_tools.clear_live_network()
