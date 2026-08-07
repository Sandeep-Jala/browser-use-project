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
