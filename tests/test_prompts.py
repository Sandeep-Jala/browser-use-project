"""scoped_subtask_prompt tests: the per-segment accuracy discipline (ported from the
whole-task expander's per-phase Verify rules) and the gate-derived done-condition."""
from automation.pipeline.hybrid import Gate, _describe_expected_end
from automation.pipeline.prompts import scoped_subtask_prompt


def test_scoped_prompt_always_carries_verification_discipline():
    p = scoped_subtask_prompt("go to Estimates", [], [])
    assert "VERIFY EVERY ACTION" in p
    assert "find_by_text" in p
    assert "independently checks your end state" in p
    # No gate knowledge, no save ownership -> those blocks stay out.
    assert "DONE CONDITION" not in p
    assert "verify_save_registered" not in p


def test_scoped_prompt_expected_end_and_save():
    p = scoped_subtask_prompt(
        "select an item and click save", ["went to estimates"], [],
        expected_end='the page URL path matches "/books/clients/*/dashboard" '
                     '(lowercased; each "*" stands for a record id)',
        owns_save=True)
    assert 'DONE CONDITION: this step is complete ONLY when the page URL path matches' in p
    assert "/books/clients/*/dashboard" in p
    assert "verify_save_registered" in p and "CONFIRMED" in p


def test_scoped_prompt_lists_all_remaining_as_context():
    p = scoped_subtask_prompt("step one", [], ["step two", "step three"])
    assert "context only" in p
    assert "step two" in p and "step three" in p
    assert "do NOT start any of them" in p


def test_describe_expected_end_per_gate_kind():
    assert _describe_expected_end(Gate(kind="marker", marker="Invoices")) is None
    assert _describe_expected_end(Gate(kind="steps")) is None
    d = _describe_expected_end(Gate(kind="postcondition", end_context="/x/inputs/sales"))
    assert '"/x/inputs/sales"' in d
    d = _describe_expected_end(
        Gate(kind="postcondition", postcondition={"url_contains": "estimates"}))
    assert 'URL contains "estimates"' in d
    d = _describe_expected_end(
        Gate(kind="postcondition", postcondition={"visible": "#flyout"}))
    assert '"#flyout" is visible' in d
