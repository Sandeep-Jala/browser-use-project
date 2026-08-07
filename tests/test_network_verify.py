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
    assert "no write request followed this click" in msg
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
