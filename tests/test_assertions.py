"""Assertion-engine matrix over canned collector results, plus a build_report smoke test."""
from pathlib import Path

from automation.pipeline import assertions as asr


def _req(method="GET", status=200, url="https://app.example/api/x", failed=False,
         error=None, step=1):
    cls = ("failed" if failed else
           "5xx" if status and status >= 500 else
           "4xx" if status and status >= 400 else "2xx")
    return {"step": step, "method": method, "status": None if failed else status,
            "url": url, "failed": failed, "errorText": error, "status_class": cls,
            "duration_ms": 42}


def _con(text="boom", severity="error", is_error=True, is_exception=False, step=1):
    return {"step": step, "severity": severity, "type": severity, "text": text,
            "source": "app.js", "line": 10, "is_error": is_error,
            "is_exception": is_exception}


def _collected(requests=(), entries=()):
    return {"network": {"summary": {}, "requests": list(requests)},
            "console": {"summary": {}, "entries": list(entries), "errors": [], "warnings": []}}


def _by_name(results):
    return {r.name: r for r in results}


# ---------------------------------------------------------------- rules

def test_all_defaults_pass_on_clean_run():
    res = asr.evaluate(_collected([_req()], []), asr.DEFAULT_SPEC)
    assert asr.overall(res) is True
    assert all(r.passed for r in res)


def test_no_5xx_fails_with_evidence():
    res = _by_name(asr.evaluate(
        _collected([_req(status=500, url="https://app/api/Invoices")], []), asr.DEFAULT_SPEC))
    assert res["no_5xx"].passed is False
    assert res["no_5xx"].evidence[0]["url"] == "https://app/api/Invoices"


def test_failed_requests_allowlist():
    reqs = [_req(failed=True, url="https://telemetry.vendor/beacon", error="net::ERR_ABORTED"),
            _req(failed=True, url="https://app/api/save", error="net::ERR_TIMED_OUT")]
    spec = asr.merge_spec(asr.DEFAULT_SPEC,
                          {"no_failed_requests": {"allow_url_patterns": [r"telemetry\.vendor"]}})
    res = _by_name(asr.evaluate(_collected(reqs, []), spec))
    assert res["no_failed_requests"].passed is False
    assert len(res["no_failed_requests"].evidence) == 1
    assert "app/api/save" in res["no_failed_requests"].evidence[0]["url"]


def test_console_errors_allowlist_and_exceptions_counted():
    entries = [_con("ResizeObserver loop limit exceeded"),
               _con("TypeError: x is undefined", is_error=False, is_exception=True)]
    spec = asr.merge_spec(asr.DEFAULT_SPEC,
                          {"no_console_errors": {"allow_patterns": ["ResizeObserver"]}})
    res = _by_name(asr.evaluate(_collected([], entries), spec))
    assert res["no_console_errors"].passed is False   # the exception still counts
    assert len(res["no_console_errors"].evidence) == 1


def test_bad_regex_degrades_to_substring_not_crash():
    spec = asr.merge_spec(asr.DEFAULT_SPEC,
                          {"no_console_errors": {"allow_patterns": ["[unclosed"]}})
    entries = [_con("prefix [unclosed bracket error")]
    res = _by_name(asr.evaluate(_collected([], entries), spec))
    assert res["no_console_errors"].passed is True     # substring matched -> allowlisted
    assert "invalid regex" in res["no_console_errors"].detail


def test_max_http_4xx_threshold():
    reqs = [_req(status=404), _req(status=403)]
    over = _by_name(asr.evaluate(_collected(reqs, []),
                                 asr.merge_spec(asr.DEFAULT_SPEC, {"max_http_4xx": 1})))
    under = _by_name(asr.evaluate(_collected(reqs, []),
                                  asr.merge_spec(asr.DEFAULT_SPEC, {"max_http_4xx": 2})))
    assert over["max_http_4xx"].passed is False
    assert len(over["max_http_4xx"].evidence) == 2
    assert under["max_http_4xx"].passed is True
    assert under["max_http_4xx"].evidence == []


def test_response_ok_generalizes_ground_truth():
    reqs = [_req(method="POST", status=201, url="https://app/api/Invoices/create")]
    spec = asr.merge_spec(asr.DEFAULT_SPEC,
                          {"response_ok": {"method": "POST", "url_contains": "invoices"}})
    res = _by_name(asr.evaluate(_collected(reqs, []), spec))
    assert res["response_ok"].passed is True

    spec = asr.merge_spec(asr.DEFAULT_SPEC,
                          {"response_ok": {"method": "POST", "url_contains": "Payments"}})
    res = _by_name(asr.evaluate(_collected(reqs, []), spec))
    assert res["response_ok"].passed is False


# ------------------------------------------------------- spec + rollup

def test_disabled_rules_are_not_evaluated():
    spec = asr.merge_spec(asr.DEFAULT_SPEC, {"no_5xx": False, "no_console_errors": None})
    names = {r.name for r in asr.evaluate(_collected([_req(status=500)], [_con()]), spec)}
    assert "no_5xx" not in names and "no_console_errors" not in names


def test_missing_collectors_skip_not_fail():
    res = asr.evaluate({}, asr.DEFAULT_SPEC)
    assert all(r.passed is None for r in res)
    assert asr.overall(res) is None


def test_unknown_rule_is_surfaced_as_skipped():
    res = _by_name(asr.evaluate(_collected(), asr.merge_spec(asr.DEFAULT_SPEC, {"no_5xxx": True})))
    assert res["no_5xxx"].passed is None
    assert "unknown" in res["no_5xxx"].detail


def test_evidence_capped():
    reqs = [_req(status=500, url=f"https://app/{i}") for i in range(9)]
    res = _by_name(asr.evaluate(_collected(reqs, []), asr.DEFAULT_SPEC))
    assert len(res["no_5xx"].evidence) == asr._EVIDENCE_CAP


def test_overall_false_wins_over_true():
    good = _req()
    bad = _req(status=500)
    assert asr.overall(asr.evaluate(_collected([good, bad], []), asr.DEFAULT_SPEC)) is False


# ------------------------------------------------------- RunResult + report integration

def _synthetic_result(tmp_path: Path):
    from automation.pipeline.runner import RunResult
    return RunResult(
        task="synthetic", run_id="test_run", artifacts_dir=tmp_path,
        expanded_task=None, is_done=True, is_successful=True, has_errors=False,
        final_result="done", urls=[], n_steps=3, duration_seconds=1.2,
        extracted_content=[], model_actions=[], errors=[],
        collector_results=_collected(
            [_req(status=500, url="https://app/api/broken")],
            [_con("Uncaught TypeError")]),
        ground_truth={"marker": "Invoices", "create_write_seen": True},
        mode="replay",
    )


def test_apply_sets_fields_and_report_renders(tmp_path):
    from automation.pipeline.report import build_report

    result = _synthetic_result(tmp_path)
    asr.apply(result, asr.DEFAULT_SPEC)
    assert result.assertions_passed is False
    assert any(a["passed"] is False for a in result.assertion_results)

    paths = build_report(result)
    html = Path(paths["html"]).read_text()
    assert "Assertions" in html
    assert "no_5xx" in html
    assert "https://app/api/broken" in html
    assert "PASS*" in html  # flow passed, assertions failed -> amber state
    # report.json carries the assertion fields via asdict
    import json
    data = json.loads(Path(paths["json"]).read_text())
    assert data["assertions_passed"] is False
    assert data["assertion_results"]
