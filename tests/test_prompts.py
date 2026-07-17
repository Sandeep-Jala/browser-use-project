"""scoped_subtask_prompt tests: the per-segment accuracy discipline (ported from the
whole-task expander's per-phase Verify rules) and the gate-derived done-condition.
Plus the system-rules invariants that keep the prompt aligned with the agent's actual
capabilities (one action per step) and misclick mitigations."""
from automation.pipeline.hybrid import Gate, _describe_expected_end
from automation.pipeline.prompts import SPEED_OPTIMIZATION_PROMPT, scoped_subtask_prompt


def test_system_rules_match_one_action_per_step():
    # The old "Chain multiple safe actions" advice contradicted max_actions_per_step=1
    # (runner.py): the model planned chains that got truncated, then acted on a false
    # world model. The rule must state the real contract.
    assert "Chain multiple" not in SPEED_OPTIMIZATION_PROMPT
    assert "ONE action per step" in SPEED_OPTIMIZATION_PROMPT


def test_system_rules_carry_discovery_and_settle_blocks():
    p = SPEED_OPTIMIZATION_PROMPT
    # Discovery-before-clicking: an index with no matching label is a guess.
    assert "LOOK BEFORE YOU CLICK" in p
    assert "list_actions" in p and "find_by_text" in p
    # Settle rule: one wait then re-look, never chained waits.
    assert "SETTLE AFTER NAVIGATION" in p
    assert "Never chain repeated waits" in p


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


def test_scoped_prompt_findings_and_observe_blocks():
    p = scoped_subtask_prompt(
        "verify the CC field matches", ["went to settings"], [],
        findings=["note the CC mail: CC = billing@acme.com"], observe=True)
    assert "OBSERVATIONS recorded by the completed steps" in p
    assert "CC = billing@acme.com" in p
    assert "OBSERVATION/VERIFICATION step" in p
    assert "WHAT YOU OBSERVED" in p
    # Honest-reporting demand is part of the observe contract.
    assert "if the check does NOT hold" in p


def test_scoped_prompt_defaults_omit_findings_and_observe():
    p = scoped_subtask_prompt("go to Estimates", [], [])
    assert "OBSERVATIONS recorded" not in p
    assert "OBSERVATION/VERIFICATION" not in p


def test_dirty_prompt_sanitizes_url_looking_failure_text():
    # browser-use navigates to the first URL-looking token in the task text; a raw
    # Playwright error ("Locator.click: ...") made a recovery agent open
    # https://Locator.click as its FIRST action (observed live). The embedded failure
    # must carry no word-adjacent dots and stay one compact line.
    raw = ('Error: Locator.click: Element is not visible\nCall log:\n'
           '  - waiting for locator("[title=\\"View all\\"]").first\n' + "x" * 400)
    p = scoped_subtask_prompt("do the step", [], [], dirty=True, prior_failure=raw)
    assert "Locator.click" not in p
    assert "Locator click" in p
    assert "\nCall log" not in p          # collapsed to one line
    assert "x" * 300 not in p             # capped well below the raw 400



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
