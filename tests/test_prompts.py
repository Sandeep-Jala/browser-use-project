"""scoped_subtask_prompt tests: the per-segment accuracy discipline (ported from the
whole-task expander's per-phase Verify rules) and the gate-derived done-condition.
Plus the system-rules invariants that keep the prompt aligned with the agent's actual
capabilities (one action per step) and misclick mitigations."""
from automation.pipeline.hybrid import Gate, _describe_expected_end
from automation.pipeline.prompts import SPEED_OPTIMIZATION_PROMPT, scoped_subtask_prompt


def test_system_rules_match_max_actions_per_step():
    # The prompt must state the real contract set by max_actions_per_step in runner.py
    # (raised 1→4, 2026-07-30): batching is allowed only across non-re-rendering actions,
    # and any page-changing action must end the step (stale-index guard).
    assert "Chain multiple" not in SPEED_OPTIMIZATION_PROMPT
    assert "up to 4 actions per step" in SPEED_OPTIMIZATION_PROMPT
    assert "stale element indices" in SPEED_OPTIMIZATION_PROMPT


def test_system_rules_carry_discovery_and_settle_blocks():
    p = SPEED_OPTIMIZATION_PROMPT
    # Discovery-before-clicking: an index with no matching label is a guess.
    assert "LOOK BEFORE YOU CLICK" in p
    assert "list_actions" in p and "find_by_text" in p
    # Settle rule: one wait then re-look, never chained waits.
    assert "SETTLE AFTER NAVIGATION" in p
    assert "Never chain repeated waits" in p


def test_system_rules_dummy_values_for_unspecified_required_fields():
    # A required field the task never mentions used to dead-end the run: the validation
    # loop said "fix those fields" but nothing licensed a value, so the agent stalled or
    # called fail_and_stop. Free-input fields now get an invented dummy value; dropdown
    # discipline is unchanged (options come from the list, never from imagination).
    p = SPEED_OPTIMIZATION_PROMPT
    assert "Required fields the task omits" in p
    assert "INVENT" in p and "dummy value" in p
    assert "never replace a task-given value" in p
    assert "NEVER invent credentials" in p
    assert "State every invented value in your done message" in p
    assert "Do NOT type random names into dropdown search fields" in p


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
    # The save step restates the dummy-value license inline: it is the step where an
    # unspecified required field actually blocks the record.
    assert "dummy value" in p
    assert "dummy value" not in scoped_subtask_prompt("go to Estimates", [], [])


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
    # Honest-reporting demand cuts BOTH ways: a check the wording states must be
    # compared and failed honestly when it does not hold...
    assert "stated check does NOT hold" in p
    assert "finish with success=false" in p
    # ...but the criteria come from the step wording ALONE — a button labeled
    # 'Verify' drew judge framing onto a pure-action step, the agent invented a
    # post-click status expectation and false-negatived the whole run.
    assert "ONLY pass/fail criteria" in p
    assert "NEVER invent an expected outcome" in p
    assert "clean receipts IS success" in p
    assert "end state you observed as FACT" in p


def test_scoped_prompt_defaults_omit_findings_and_observe():
    p = scoped_subtask_prompt("go to Estimates", [], [])
    assert "OBSERVATIONS recorded" not in p
    assert "OBSERVATION/VERIFICATION" not in p
    assert "FILE DOWNLOAD" not in p


def test_scoped_prompt_loop_block():
    """The loop node's repeat-until contract: action framing (the judge observation
    block stays OUT), one iteration at a time, no jumping ahead, and a generic
    done-condition — without it, observation framing made the agent declare the
    employees loop done after a single Save & Next."""
    p = scoped_subtask_prompt(
        "process employees one at a time until Owen Millar is shown", [], [], loop=True)
    assert "LOOP step" in p
    assert "ONE iteration at a time" in p
    assert "NEVER jump ahead" in p
    assert "DONE CONDITION" in p and "stop condition holds" in p
    assert "MANY iterations" in p
    assert "final observed state" in p
    assert "OBSERVATION/VERIFICATION" not in p
    # And the block stays out of every non-loop prompt.
    assert "LOOP step" not in scoped_subtask_prompt("go to Estimates", [], [])


def test_scoped_prompt_conditional_block():
    """The branch-guard contract: condition absent -> immediate no-op success. Without
    it, the generic end-state footer made the agent treat a vacuous pass as a failed
    run and hunt for controls matching the branch's action words — observed live: a
    server-suppressed popup's 'click Process' resolved via tooltip text to the
    'Reminder to process the payroll' icon button, opening the email modal in an
    endless open/close loop (and completing two unintended pay runs on the way)."""
    p = scoped_subtask_prompt(
        "If a pop up appears, select 'don't show this again' and click Process",
        [], [], conditional=True)
    assert "CONDITIONAL step" in p
    assert "ONLY IF" in p
    assert "success=true immediately" in p
    assert "condition did not occur" in p
    assert "NEVER click" in p and "MAKE the condition true" in p
    # The tooltip-word trap named generically: no clicking a control just because its
    # name echoes a word from the branch's actions.
    assert "name or tooltip contains a word" in p
    # Both branches stay live: a true condition still performs the stated actions.
    assert "condition DOES hold" in p
    # And the block stays out of every non-conditional prompt.
    assert "CONDITIONAL step" not in scoped_subtask_prompt("go to Estimates", [], [])


def test_scoped_prompt_download_contract():
    """The download segment's verify-then-done rule: click ONCE, a timeout receipt is
    NORMAL, verify_download CONFIRMED = done immediately."""
    p = scoped_subtask_prompt("select download, select PDF", [], [], downloads_file=True)
    assert "FILE DOWNLOAD" in p
    assert "ONCE" in p and "TIMEOUT" in p and "NORMAL" in p
    assert "verify_download" in p
    assert "success=true immediately" in p


def test_scoped_prompt_aux_tab_block():
    p = scoped_subtask_prompt("search DuckDuckGo for X and note the top result", [], [],
                              aux_tab="https://duckduckgo.com")
    assert "SEPARATE HELPER TAB" in p
    assert "https://duckduckgo.com" in p
    # The extraction contract: page-GENERATED facts go through extract_data (prefer one
    # capture of the block that shows them); values the instructions themselves specify
    # are restated in the done message, never hunted on the page.
    assert "extract_data" in p
    assert "ONE" in p and "block" in p
    assert "do NOT extract_data it" in p
    assert "do NOT open or close any tab" in p
    # Default prompts carry none of it.
    assert "SEPARATE HELPER TAB" not in scoped_subtask_prompt("go to Estimates", [], [])


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
