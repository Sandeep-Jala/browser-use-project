"""One verdict rule, two consumers.

`report.py` decided a run's headline verdict inline inside `_render_html`, which was fine while
the report was the only thing that had a verdict. The Auto Agent UI now shows the same verdict on
its runs list and its run-detail page, and a dashboard that disagrees with the report it links to
is worse than no dashboard — so the rule moves into a named function both call.

The subtle case, and the reason this is a rule rather than a boolean: `PASS*`. A run whose flow
completed and saved but whose telemetry assertions failed (5xx responses, console errors) is NOT
a pass and NOT a failure — `assertions_passed` is deliberately kept out of `is_successful`
(assertions.py's docstring), so the fourth state has to be carried by the verdict itself.
"""
from types import SimpleNamespace

from automation.pipeline.report import _render_html, derive_status


def _result(**over):
    """A RunResult-shaped stub carrying only what the verdict and the renderer read."""
    base = dict(
        task="t", run_id="20260101_000000_000000", artifacts_dir="artifacts/x",
        is_done=True, is_successful=True, has_errors=False, final_result="",
        urls=[], n_steps=0, duration_seconds=0.0, extracted_content=[], model_actions=[],
        errors=[], collector_results={}, artifacts={}, screenshots=[], steps=[],
        usage=None, ground_truth=None, mode="hybrid", subtasks=[], replay=None,
        assertion_results=[], assertions_passed=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


# ── the four outcomes ─────────────────────────────────────────────────────────────────────

def test_a_successful_run_passes():
    assert derive_status(is_successful=True, is_done=True, assertions_passed=True) == ("PASS", "success")


def test_a_successful_run_with_failed_assertions_is_its_own_state():
    """The flow saved, but the app was unhealthy while it did. Neither green nor red."""
    assert derive_status(is_successful=True, is_done=True,
                         assertions_passed=False) == ("PASS*", "warning")


def test_a_finished_but_unsuccessful_run_is_done_not_failed():
    """The agent ran to completion and simply did not achieve the end state."""
    assert derive_status(is_successful=False, is_done=True, assertions_passed=None) == ("DONE", "warning")


def test_a_run_that_never_finished_fails():
    assert derive_status(is_successful=False, is_done=False, assertions_passed=None) == ("FAIL", "error")


def test_assertions_never_rescue_an_unsuccessful_run():
    """Assertions are demote-only at the task level too: passing telemetry cannot turn a failed
    flow green, the same one-way rule the segment gates follow."""
    assert derive_status(is_successful=False, is_done=True, assertions_passed=True)[0] == "DONE"
    assert derive_status(is_successful=False, is_done=False, assertions_passed=True)[0] == "FAIL"


# ── the renderer still agrees with it ─────────────────────────────────────────────────────

def test_the_rendered_report_shows_the_derived_status_and_its_ring_colour():
    """Pins the wiring: if `_render_html` ever grows its own copy of the rule again, the two
    consumers can drift apart silently — which is exactly what this extraction prevents."""
    for successful, done, asserts_ok, label, colour in [
        (True, True, True, "PASS", "var(--success)"),
        (True, True, False, "PASS*", "var(--warning)"),
        (False, True, None, "DONE", "var(--warning)"),
        (False, False, None, "FAIL", "var(--error)"),
    ]:
        html = _render_html(_result(is_successful=successful, is_done=done,
                                    assertions_passed=asserts_ok))
        assert f"border-color:{colour}'>{label}<" in html, f"{label} ring missing"
