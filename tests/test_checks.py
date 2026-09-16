"""Deterministic per-subtask checks: schema parsing + evaluation (pipeline/checks.py).

The check layer is the verification path — it must stay LLM-free, fail closed on
anything unevaluable, and mirror the pipeline's existing ground-truth rules
(_first_create_write's 2xx/3xx write scan, _write_verdict's body refusals, the
raw-finder probe contracts).
"""
from __future__ import annotations

import asyncio
import json

import pytest

from automation.pipeline import checks as ck
from automation.pipeline import script_compile as sc


@pytest.fixture(autouse=True)
def fast_poll(monkeypatch):
    """Keep eventually-polling fast in tests (same convention as the gate tests)."""
    monkeypatch.setattr(ck, "_CHECK_POLL_S", 0.01)


class FakePage:
    """Playwright-page stand-in: fixed url, scripted evaluate() results.

    Results are popped in order; the last one repeats once the script runs out
    (so a "never appears" page just keeps answering count=0). An Exception entry
    is raised instead of returned.
    """

    def __init__(self, url="https://app.example.com/Payroll/Clients/12/Employees",
                 results=None):
        self.url = url
        self._results = list(results if results is not None else [{"count": 0}])
        self.eval_calls: list[str] = []

    async def evaluate(self, expr):
        self.eval_calls.append(expr)
        res = self._results.pop(0) if len(self._results) > 1 else self._results[0]
        if isinstance(res, Exception):
            raise res
        return res


def one(kind, arg, **extra):
    return ck.parse_verify([{kind: arg, **extra}])


# ------------------------------- parse_verify -------------------------------


def test_parse_scalar_form_defaults():
    (check,) = ck.parse_verify([{"text_visible": "Alistair Allan"}])
    assert check.kind == "text_visible"
    assert check.arg == "Alistair Allan"
    assert check.timeout_s == ck._CHECK_TIMEOUT_S


def test_parse_timeout_override():
    (check,) = ck.parse_verify([{"write_accepted": "FPS", "timeout_s": 15}])
    assert check.timeout_s == 15.0


def test_parse_all_kinds():
    raw = [{"text_visible": "a"}, {"text_absent": "b"}, {"control_exists": "c"},
           {"url_contains": "d"}, {"write_accepted": "e"}]
    kinds = [c.kind for c in ck.parse_verify(raw)]
    assert kinds == ["text_visible", "text_absent", "control_exists",
                     "url_contains", "write_accepted"]


def test_parse_none_is_empty():
    assert ck.parse_verify(None) == ()


def test_parse_rejects_empty_list():
    with pytest.raises(ValueError, match="at least one"):
        ck.parse_verify([])


def test_parse_rejects_unknown_kind():
    with pytest.raises(ValueError, match="texts_visible"):
        ck.parse_verify([{"texts_visible": "x"}])


def test_parse_rejects_empty_arg():
    with pytest.raises(ValueError, match="text_visible"):
        ck.parse_verify([{"text_visible": "   "}])


@pytest.mark.parametrize("bad", [0, -1, "abc"])
def test_parse_rejects_bad_timeout(bad):
    with pytest.raises(ValueError, match="timeout_s"):
        ck.parse_verify([{"url_contains": "x", "timeout_s": bad}])


def test_parse_rejects_two_kinds_in_one_item():
    with pytest.raises(ValueError, match="exactly one"):
        ck.parse_verify([{"text_visible": "a", "url_contains": "b"}])


def test_parse_rejects_stray_keys():
    with pytest.raises(ValueError, match="foo"):
        ck.parse_verify([{"text_visible": "a", "foo": 1}])


def test_parse_rejects_non_list_and_non_mapping_items():
    with pytest.raises(ValueError):
        ck.parse_verify("text_visible: x")
    with pytest.raises(ValueError):
        ck.parse_verify(["text_visible"])


# ------------------------------- parse_probe -------------------------------


def test_parse_probe_single_mapping_short_default_timeout():
    check = ck.parse_probe({"text_visible": "Don't show this again"})
    assert check.kind == "text_visible"
    assert check.arg == "Don't show this again"
    assert check.timeout_s == ck._PROBE_TIMEOUT_S


def test_parse_probe_declared_timeout_wins():
    check = ck.parse_probe({"text_visible": "x", "timeout_s": 8})
    assert check.timeout_s == 8.0


def test_parse_probe_none_is_none():
    assert ck.parse_probe(None) is None


def test_parse_probe_rejects_list_form():
    with pytest.raises(ValueError, match="single check mapping"):
        ck.parse_probe([{"text_visible": "x"}])


def test_parse_probe_rejects_unknown_kind():
    with pytest.raises(ValueError, match="texts_visible"):
        ck.parse_probe({"texts_visible": "x"})


# ------------------------------- shared tokenizer -------------------------------


def test_query_tokens_lowercases_and_splits():
    assert sc._query_tokens("Alistair  Allan!") == ["alistair", "allan"]
    assert sc._query_tokens("£3,500.00") == ["3", "500", "00"]
    assert sc._query_tokens("---") == []


# ------------------------------- evaluate: page checks -------------------------------


async def test_url_contains_pass_is_case_insensitive():
    page = FakePage(url="https://app.example.com/Payroll/Employees")
    [res] = await ck.evaluate_checks(page, [], one("url_contains", "payroll"),
                                     poll=False)
    assert res["ok"] is True
    assert "Payroll" in res["evidence"]


async def test_url_contains_without_page_fails_closed():
    [res] = await ck.evaluate_checks(None, [], one("url_contains", "payroll"),
                                     poll=False)
    assert res["ok"] is False


async def test_text_visible_pass_records_evidence_and_tokenizes():
    page = FakePage(results=[{"count": 1, "name": "Alistair Allan"}])
    [res] = await ck.evaluate_checks(page, [], one("text_visible", "Alistair Allan"),
                                     poll=False)
    assert res["ok"] is True
    assert "Alistair Allan" in res["evidence"]
    assert json.dumps(["alistair", "allan"]) in page.eval_calls[0]


async def test_text_visible_eventually_passes_on_second_poll():
    page = FakePage(results=[{"count": 0}, {"count": 1, "name": "Bruce Wright"}])
    [res] = await ck.evaluate_checks(page, [], one("text_visible", "Bruce Wright",
                                                   timeout_s=1))
    assert res["ok"] is True
    assert len(page.eval_calls) == 2


async def test_text_visible_times_out_and_fails():
    page = FakePage(results=[{"count": 0}])
    [res] = await ck.evaluate_checks(page, [], one("text_visible", "Ghost",
                                                   timeout_s=0.05))
    assert res["ok"] is False
    assert "not" in (res["evidence"] or "") + (res["error"] or "")


async def test_text_absent_passes_when_gone_and_fails_when_present():
    gone = FakePage(results=[{"count": 0}])
    [res] = await ck.evaluate_checks(gone, [], one("text_absent", "Saving"),
                                     poll=False)
    assert res["ok"] is True
    there = FakePage(results=[{"count": 2, "name": "Saving"}])
    [res] = await ck.evaluate_checks(there, [], one("text_absent", "Saving",
                                                    timeout_s=0.05))
    assert res["ok"] is False
    assert "Saving" in res["evidence"]


async def test_text_absent_eventually_passes_when_it_disappears():
    page = FakePage(results=[{"count": 1, "name": "Saving"}, {"count": 0}])
    [res] = await ck.evaluate_checks(page, [], one("text_absent", "Saving",
                                                   timeout_s=1))
    assert res["ok"] is True


async def test_control_exists_pass_and_fail():
    page = FakePage(results=[{"count": 1, "name": "Save & Next"}])
    [res] = await ck.evaluate_checks(page, [], one("control_exists", "Save & Next"),
                                     poll=False)
    assert res["ok"] is True
    assert "Save & Next" in res["evidence"]
    missing = FakePage(results=[{"count": 0}])
    [res] = await ck.evaluate_checks(missing, [], one("control_exists", "Nope",
                                                      timeout_s=0.05))
    assert res["ok"] is False


async def test_probe_error_result_fails_closed_without_polling():
    page = FakePage(results=[{"error": "boom"}, {"count": 1, "name": "x"}])
    [res] = await ck.evaluate_checks(page, [], one("text_visible", "x", timeout_s=1))
    assert res["ok"] is False
    assert "boom" in res["error"]
    assert len(page.eval_calls) == 1


async def test_probe_exception_fails_closed():
    page = FakePage(results=[RuntimeError("page closed")])
    [res] = await ck.evaluate_checks(page, [], one("text_visible", "x", timeout_s=1))
    assert res["ok"] is False
    assert "page closed" in res["error"]


async def test_poll_false_evaluates_exactly_once():
    page = FakePage(results=[{"count": 0}, {"count": 1, "name": "x"}])
    [res] = await ck.evaluate_checks(page, [], one("text_visible", "x"), poll=False)
    assert res["ok"] is False
    assert len(page.eval_calls) == 1


# ------------------------------- evaluate: write_accepted -------------------------------


def _write(method="POST", url="https://x/api/Payroll/Clients/12/Employees/?yearId=27",
           status=200, **extra):
    return {"method": method, "url": url, "status": status, **extra}


async def test_write_accepted_plain_2xx_passes():
    [res] = await ck.evaluate_checks(None, [_write()], one("write_accepted", "Employees"),
                                     poll=False)
    assert res["ok"] is True
    assert "POST" in res["evidence"] and "200" in res["evidence"]


async def test_write_accepted_fragment_is_case_insensitive():
    [res] = await ck.evaluate_checks(None, [_write()], one("write_accepted", "employees"),
                                     poll=False)
    assert res["ok"] is True


async def test_write_accepted_ignores_reads_and_other_urls():
    window = [_write(method="GET"), _write(url="https://x/api/other")]
    [res] = await ck.evaluate_checks(None, window, one("write_accepted", "Employees",
                                                       timeout_s=0.05))
    assert res["ok"] is False
    assert "no accepted" in res["error"]


async def test_write_accepted_3xx_passes_4xx_fails():
    [res] = await ck.evaluate_checks(None, [_write(status=302)],
                                     one("write_accepted", "Employees"), poll=False)
    assert res["ok"] is True
    [res] = await ck.evaluate_checks(None, [_write(status=404)],
                                     one("write_accepted", "Employees", timeout_s=0.05))
    assert res["ok"] is False


async def test_write_accepted_refused_body_fails():
    body = json.dumps({"status": False, "message": "already submitted"})
    [res] = await ck.evaluate_checks(None, [_write(body=body)],
                                     one("write_accepted", "Employees", timeout_s=0.05))
    assert res["ok"] is False
    assert "already submitted" in res["error"]


async def test_write_accepted_affirming_body_passes():
    body = json.dumps({"result": {"submitDetail": {"isSubmitted": True}}})
    [res] = await ck.evaluate_checks(None, [_write(url="https://x/api/Years/27/FPS",
                                                   body=body)],
                                     one("write_accepted", "FPS"), poll=False)
    assert res["ok"] is True
    assert "isSubmitted" in res["evidence"]


async def test_write_accepted_repolls_live_records_until_settled():
    record = _write(status=None)

    async def settle():
        await asyncio.sleep(0.03)
        record["status"] = 200

    task = asyncio.ensure_future(settle())
    [res] = await ck.evaluate_checks(None, [record],
                                     one("write_accepted", "Employees", timeout_s=1))
    await task
    assert res["ok"] is True


# ------------------------------- evaluate: result shape -------------------------------


async def test_results_preserve_order_and_shape():
    page = FakePage(url="https://x/Payroll", results=[{"count": 1, "name": "Run"}])
    checks = ck.parse_verify([{"url_contains": "payroll"}, {"control_exists": "Run"}])
    results = await ck.evaluate_checks(page, [], checks, poll=False)
    assert [r["kind"] for r in results] == ["url_contains", "control_exists"]
    for r in results:
        assert set(r) == {"kind", "arg", "ok", "evidence", "error"}


# ------------------------------- receipt roll-up -------------------------------
# Two conservative rules only: a trailing error-channel refusal, and a final fired-but-
# not-accepted write with nothing accepted anywhere in the segment's window.


class _R:
    """ActionResult-shaped: is_done / error / metadata are all the roll-up reads."""

    def __init__(self, is_done=False, error=None, metadata=None):
        self.is_done, self.error, self.metadata = is_done, error, metadata


class _Item:
    def __init__(self, *results):
        self.result = list(results)


class _Hist:
    def __init__(self, *items):
        self.history = list(items)


def test_rollup_empty_history_passes():
    assert ck.receipt_rollup(_Hist(), []) == (True, [])


def test_rollup_trailing_error_channel_refusal_fails():
    hist = _Hist(
        _Item(_R(error="input: the only match is the text INSIDE that input",
                 metadata={"no_click": True})),
        _Item(_R(is_done=True)),
    )
    ok, reasons = ck.receipt_rollup(hist, [])
    assert ok is False
    assert "REFUSED" in reasons[0]


def test_rollup_content_channel_no_click_does_not_trip():
    """Candidate listings / static-text probes stamp no_click WITHOUT an error — they
    are answers, not refused actions, and must not fail the segment."""
    hist = _Hist(_Item(_R(metadata={"no_click": True})), _Item(_R(is_done=True)))
    assert ck.receipt_rollup(hist, []) == (True, [])


def test_rollup_refusal_then_recovery_passes():
    hist = _Hist(
        _Item(_R(error="no_fill: dropdown filter", metadata={"no_fill": True})),
        _Item(_R()),  # the recovery action the receipt steered to
        _Item(_R(is_done=True)),
    )
    assert ck.receipt_rollup(hist, [])[0] is True


def test_rollup_refused_final_write_with_nothing_accepted_fails():
    hist = _Hist(
        _Item(_R(metadata={"write_outcome": {"fired": True, "accepted": False,
                                             "t0": 0.0}})),
        _Item(_R(is_done=True)),
    )
    window = [{"method": "POST", "url": "http://api/Years/27/FPS", "status": 200,
               "body": json.dumps({"status": False, "message": "already submitted"})}]
    ok, reasons = ck.receipt_rollup(hist, window)
    assert ok is False
    assert "write" in reasons[0]


def test_rollup_refused_receipt_waived_by_any_accepted_write():
    """Duplicate-save shape: an earlier write in the window was accepted; the refused
    re-submit last is proof of completion, not failure."""
    hist = _Hist(
        _Item(_R(metadata={"write_outcome": {"fired": True, "accepted": False,
                                             "t0": 5.0}})),
        _Item(_R(is_done=True)),
    )
    window = [{"method": "POST", "url": "http://api/Years/27/FPS", "status": 200}]
    assert ck.receipt_rollup(hist, window)[0] is True


def test_rollup_accepted_final_write_passes():
    hist = _Hist(
        _Item(_R(metadata={"write_outcome": {"fired": True, "accepted": True,
                                             "t0": 0.0}})),
        _Item(_R(is_done=True)),
    )
    assert ck.receipt_rollup(hist, [])[0] is True


def test_rollup_refused_final_write_waived_by_declaration():
    """The declared allow_write_refusal waiver reaches the RECEIPT side of the same
    judgment: a slice that declares its own error branch ends on a refused write, and
    both write-acceptance rules must stand down for it — not just the network one."""
    hist = _Hist(
        _Item(_R(metadata={"write_outcome": {"fired": True, "accepted": False,
                                             "t0": 0.0}})),
        _Item(_R(is_done=True)),
    )
    window = [{"method": "POST", "url": "http://api/Years/27/FPS", "status": 200,
               "body": json.dumps({"status": False, "message": "already submitted"})}]
    assert ck.receipt_rollup(hist, window, allow_write_refusal=True) == (True, [])


def test_rollup_waiver_does_not_excuse_a_refused_final_ACTION():
    """The waiver is about WRITES. A tool that refused to click or fill is a different
    failure and still contradicts a success claim."""
    hist = _Hist(
        _Item(_R(error="no_click: the element is detached", metadata={"no_click": True})),
        _Item(_R(is_done=True)),
    )
    ok, reasons = ck.receipt_rollup(hist, [], allow_write_refusal=True)
    assert ok is False and "REFUSED" in reasons[0]


# ------------------------------- window_write_rollup -------------------------------


def test_window_rule_zero_writes_passes():
    assert ck.window_write_rollup([]) == (True, [])
    reads = [_write(method="GET"), _write(method="GET", status=404)]
    assert ck.window_write_rollup(reads) == (True, [])


def test_window_rule_accepted_write_passes():
    assert ck.window_write_rollup([_write()])[0] is True


def test_window_rule_refused_body_only_fails_with_server_text():
    body = json.dumps({"status": False, "message": "already submitted"})
    ok, reasons = ck.window_write_rollup([_write(body=body)])
    assert ok is False
    assert "already submitted" in reasons[0]


def test_window_rule_http_error_only_fails():
    ok, reasons = ck.window_write_rollup([_write(status=500)])
    assert ok is False
    assert "HTTP 500" in reasons[0]


def test_window_rule_network_failure_only_fails():
    rec = _write(status=None, failed=True, errorText="net::ERR_CONNECTION_RESET")
    ok, reasons = ck.window_write_rollup([rec])
    assert ok is False
    assert "ERR_CONNECTION_RESET" in reasons[0]


def test_window_rule_refusal_recovered_by_later_accept_passes():
    body = json.dumps({"status": False, "message": "select a reason"})
    assert ck.window_write_rollup([_write(body=body), _write()])[0] is True


def test_window_rule_accept_waives_later_refused_duplicate():
    body = json.dumps({"status": False, "message": "already submitted"})
    assert ck.window_write_rollup([_write(), _write(body=body)])[0] is True


def test_window_rule_noise_accept_cannot_waive_business_refusal():
    """The waiver hole this rule closes: an accepted infrastructure write must not
    count as proof the segment's business write landed."""
    noise = _write(url="https://x/auth/webpush")
    body = json.dumps({"status": False, "message": "already submitted"})
    ok, reasons = ck.window_write_rollup([noise, _write(body=body)])
    assert ok is False
    assert "already submitted" in reasons[0]


def test_window_rule_noise_only_traffic_passes():
    window = [_write(url="https://x/auth/webpush"),
              _write(url="https://x/oauth/token", status=400),
              _write(url="https://x/client/negotiate?hub=usershub")]
    assert ck.window_write_rollup(window) == (True, [])


def test_window_rule_inflight_write_is_skipped():
    assert ck.window_write_rollup([_write(status=None)]) == (True, [])


def test_business_writes_filters_methods_and_noise():
    window = [_write(), _write(method="GET"),
              _write(url="https://x/hubs/users/negotiate")]
    assert len(ck.business_writes(window)) == 1


def test_save_cue_matches_stems_and_ignores_plain_reads():
    assert ck.save_cue("fill in the form and click Save") is True
    assert ck.save_cue("After saving, continue processing") is True
    assert ck.save_cue("click Submit and wait") is True
    assert ck.save_cue("go to the section and read the employee name") is False
    assert ck.save_cue(None) is False


# ----------------------- page attribution (the boot-traffic rule) -----------------------
# Run 20260901_093026 seg 3: the subtask said "go to Pay Forecast, refresh, pick the
# employee" — it writes nothing. The reload's own boot POST
# (Addons/MSTeams/Subscribe -> 200 {"status": false, "message": "Outlook/Microsoft
# authentication not found for current user."}) was the window's only business write, so
# the rule read the APP's page-load traffic as the SEGMENT's failed save and stopped the
# run. A write the page issued while loading, before the segment touched it, is not the
# segment's work — the collector stamps `after_page_load` and business_writes drops it.


def _boot(**extra):
    return _write(url="https://x/api/Addons/MSTeams/Subscribe", after_page_load=True,
                  **extra)


def test_page_load_write_is_not_the_segments_work():
    body = json.dumps({"status": False,
                       "message": "Outlook/Microsoft authentication not found"})
    assert ck.business_writes([_boot(body=body)]) == []
    assert ck.window_write_rollup([_boot(body=body)]) == (True, [])


def test_page_load_write_does_not_waive_a_real_refusal():
    """The mirror hole: boot traffic must not VOUCH for the segment either. An accepted
    subscribe on load says nothing about the save the segment then made."""
    body = json.dumps({"status": False, "message": "already submitted"})
    ok, reasons = ck.window_write_rollup([_boot(), _write(body=body)])
    assert ok is False
    assert "already submitted" in reasons[0]


def test_write_after_an_interaction_is_still_judged():
    body = json.dumps({"status": False, "message": "already submitted"})
    ok, reasons = ck.window_write_rollup([_write(body=body, after_page_load=False)])
    assert ok is False and "already submitted" in reasons[0]


def test_unstamped_records_are_judged_exactly_as_before():
    """Fail open: recordings and paths that never stamped the flag keep today's verdict —
    an additive gate must not go blind on traffic it cannot attribute."""
    body = json.dumps({"status": False, "message": "already submitted"})
    ok, _ = ck.window_write_rollup([_write(body=body)])
    assert ok is False


async def test_write_accepted_check_ignores_page_load_traffic():
    """Same attribution for the declared check: a `write_accepted` names the save the
    SEGMENT was asked to make, and a write the page fired on load is not it."""
    [res] = await ck.evaluate_checks(
        None, [_write(url="https://x/api/Addons/Subscribe", after_page_load=True)],
        one("write_accepted", "Subscribe"), poll=False)
    assert res["ok"] is False


async def test_page_load_write_cannot_waive_a_contradicting_receipt():
    """receipt_rollup's escape hatch reads the window through the same lens — boot
    traffic must not stand in for the save the agent's own receipt says never landed."""
    hist = _Hist(
        _Item(_R(metadata={"write_outcome": {"fired": True, "accepted": False,
                                             "t0": 0.0}})),
        _Item(_R(is_done=True)),
    )
    ok, reasons = ck.receipt_rollup(hist, [_write(after_page_load=True)])
    assert ok is False and reasons
