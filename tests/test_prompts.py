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


# ---- the prompt must not promise the agent is already where its step needs it ----
# Run 20260902_120618 subtask 5. The opening line used to assert unconditionally that "the
# page is already in the correct starting state for your step". It was false: the FPS slice
# had left the app on the payroll SUMMARY page. Given that guarantee, "Then go to Payroll &
# RTI and change the period to Jun-26" cannot be read as an instruction to go anywhere, and
# the agent's own reasoning shows it was not — "we need to change the period filter in the
# Payroll & RTI navbar ... this is the final required action". It set the REPORT's period
# filter, the payrun never moved off May-26, and the segment cached a skill with no
# navigation in it. Reinforced by the completed list, which held the byte-identical May-26
# copy of the very line it was executing under "do NOT ... navigate back to these".


def test_scoped_prompt_never_claims_the_page_is_already_the_right_one():
    p = scoped_subtask_prompt("go to Payroll & RTI and change the period", [], [])
    assert "already in the correct starting state" not in p
    # It says what IS true (a session mid-flight) and what the agent must therefore do.
    assert "ALREADY IN PROGRESS" in p
    assert "NOT necessarily where your step needs to be" in p
    assert "make sure you are actually on it" in p
    assert "an action to perform when you are not there" in p


def test_scoped_prompt_still_forbids_restarting_the_whole_flow():
    """The replaced sentence was carrying the anti-restart intent. Dropping the false
    promise must not drop that — it is stated separately, and more precisely."""
    p = scoped_subtask_prompt("go to Estimates", ["opened the client"], [])
    assert "Do NOT navigate to the app root, re-select the business, or restart the flow." in p


def test_the_completed_list_never_excuses_skipping_your_own_job():
    """A repeated workflow states the same step once per cycle, so the completed list will
    hold near-identical wording to the current job. Without the carve-out that reads as
    licence to skip the part that repeats."""
    p = scoped_subtask_prompt(
        "Then go to Payroll & RTI and change the period to Jun-26",
        ["Then go to Payroll & RTI and change the period to May-26"], [])
    assert "do NOT redo, verify, or navigate back to these" in p    # intent kept
    assert "never excuses skipping any part of YOUR OWN job" in p
    assert "even where the wording repeats" in p


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


def test_repeat_capability_is_stated_once_for_every_step():
    """The per-slice LOOP block went with the `loop` kind (2026-08-28). Repetition is now a
    TOOL the agent always has, so the guidance is one general capability statement in the
    system prompt rather than a block injected when wording looked loop-shaped."""
    assert "repeat_click" in SPEED_OPTIMIZATION_PROMPT
    assert "times=0" in SPEED_OPTIMIZATION_PROMPT       # the until-it-stops mode
    # ...and no slice-scoped loop framing survives.
    p = scoped_subtask_prompt("do the thing", [], [])
    assert "LOOP step" not in p


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


def test_scoped_prompt_next_conditional_handoff():
    """The producer half of the probe handshake. A declared `probe:` says the framework
    will look for that outcome before the NEXT slice runs — but the slice that PRODUCES
    it was never told, so it read the outcome as its own failed action. Observed live
    (run 20260901_151214, the FPS submit slice): the server refused the write, the agent
    cancelled the dialog, reopened the form, re-uploaded and re-submitted — 4 submits for
    1, 15 steps, 554k tokens, and the flail cached as 22 replayable actions."""
    p = scoped_subtask_prompt(
        "click Submit", [], [],
        next_conditional='the text "already submitted" visible on the page')
    assert "EXPECTED OUTCOME ALREADY HANDLED BY THE NEXT STEP" in p
    assert "already submitted" in p
    # The outcome is a PASS for this step, reported and quoted (the quote is what hands
    # us the dialog's real text for repairing a stale probe arg).
    assert "success=true" in p and "quoting the message text" in p
    # The two loop shapes measured, both named.
    assert "Do NOT close, cancel, dismiss" in p
    assert "do NOT repeat, re-enter or retry" in p
    # And it must say what it overrides, or it just contradicts the blocks above it.
    assert "overrides the re-click rule" in p
    assert "overrides the done condition" in p
    # The successor's ACTION words never cross the boundary — only its condition.
    assert "Cancel to close it" not in p


def test_handoff_block_is_absent_by_default():
    """Every slice without a probed successor keeps its prompt byte-identical."""
    p = scoped_subtask_prompt("go to Estimates", [], ["step two"])
    assert "EXPECTED OUTCOME ALREADY HANDLED" not in p


def test_handoff_block_follows_the_rules_it_overrides():
    """Position is load-bearing: the block is written as an override of the retry-
    licensing text above it, so it must come AFTER those blocks and sit closest to the
    done instruction — before the still-ahead list and the end-state footer."""
    p = scoped_subtask_prompt(
        "click Submit", [], ["a later step"],
        expected_end='the text "Submitted" visible on the page',
        next_conditional='the text "already submitted" visible on the page')
    assert p.index("EXPECTED OUTCOME ALREADY HANDLED") > p.index("VERIFY EVERY ACTION")
    assert p.index("EXPECTED OUTCOME ALREADY HANDLED") > p.index("DONE CONDITION")
    assert p.index("EXPECTED OUTCOME ALREADY HANDLED") < p.index("Still ahead")
    assert (p.index("EXPECTED OUTCOME ALREADY HANDLED")
            < p.index("independently checks your end state"))


# ---------------- a find miss is not proof of absence (2026-09-01) ----------------
# Run 20260901_174417 subtask 2: the Send Email panel was OPEN and 'Select sender' — the
# combobox's placeholder — was on screen in the state the agent was shown. find_by_text
# correctly found no clickable element (a placeholder is not in selector_map), and both
# the receipt and this prompt told it the element "is not on the page". It concluded the
# dropdown must be an icon button, blind-clicked a nameless one, and re-ran the same
# string through search_page before re-reading the task and getting it right.


def test_system_rules_do_not_claim_a_find_miss_proves_absence():
    p = SPEED_OPTIMIZATION_PROMPT
    assert "the element is not on\n     the page" not in p
    assert "no CLICKABLE" in p
    # And the cross-tool loophole is closed here too.
    assert "through that tool or any other" in p


def test_system_rules_not_found_policy_starts_by_rereading_the_step():
    """Re-reading the step is what actually ended the failure (the agent's own step-3
    thinking quotes the task verbatim), and it is always available — scoped_subtask_prompt
    puts the step's wording in context on every step."""
    p = SPEED_OPTIMIZATION_PROMPT
    assert "re-read your step's own wording" in p
    # The old advice sent a find_by_text miss straight back into find_by_text.
    assert 'FIRST recovery approach: find_by_text' not in p
    # The cross-tool ban stays; only its SCOPE was narrowed on 2026-09-03 (from "a failed
    # search string" to "a DEAD search string, unchanged"), because taken absolutely it also
    # forbade adding near_text — see test_re_issuing_a_search_CHANGED_is_not_a_repeat.
    assert "from tool to tool" in p
    # A miss has three live causes, not just "wrong page". Asserted on the UNWRAPPED text:
    # the original pinned "wrong\n    QUERY", which made the invariant hostage to the line
    # break rather than to the wording (adding the tool name on 2026-09-03 re-wrapped it).
    flat = " ".join(p.split())
    assert "a wrong QUERY" in flat and "not yet open" in flat


def test_system_rules_named_buttons_defer_dropdowns_to_select_dropdown():
    """NAMED ICON BUTTONS routed "click the dropdown next to X" into a text search for a
    control that has no name — which cannot succeed. The carve-out must be read BEFORE
    the find_by_text instruction it excepts."""
    p = SPEED_OPTIMIZATION_PROMPT
    carve = "a dropdown/combobox has no name of its own"
    assert carve in p
    bullet = 'Call find_by_text("<that exact name>", click_first=true)'
    assert p.index(carve) < p.index(bullet)


# ------------------------------- the NOT FOUND policy, narrowed -------------------------------
#
# Run 20260903_102354_153542 subtask 4: the same slice that had authored in 3 steps three
# times took 19 and had to be killed. The mechanism was this section.
#
# The working-tree rewrite deleted the policy's one CONCRETE first move —
#   "FIRST recovery approach: find_by_text("<the element's label>")"
# — replacing it with "re-read your step's own wording" (reflection, not an action), and
# added "do NOT re-issue a failed search string through a different tool". So when
# find_by_text refused 24 ambiguous "Net to gross" pencils, the agent had no named move and
# was forbidden the fallbacks. What was left was "try 3–4 MEANINGFULLY DIFFERENT approaches",
# and it took that literally: it wrote JavaScript into `navigate`, twice, and lost the page.
#
# Narrowed, not reverted. The re-read advice stays first and the ban on flogging a dead
# string stays — both were deliberate, both have a run behind them. What changes: the policy
# names a tool again, "do not repeat" means UNCHANGED, and AMBIGUITY is covered at all.


def _not_found_policy() -> str:
    return (SPEED_OPTIMIZATION_PROMPT
            .split("ELEMENT / OBJECTIVE NOT FOUND POLICY")[1]
            .split("LOOK BEFORE YOU CLICK")[0])


def test_the_not_found_policy_names_a_concrete_first_move():
    """"Re-read your step's wording" is a thought, not an action. The agent that flailed had
    re-read it and still had nowhere to go — the policy has to end in a callable tool."""
    policy = _not_found_policy()

    assert "re-read your step's own wording" in policy      # the deliberate part, kept
    assert "find_by_text(" in policy                        # and something to actually do


def test_re_issuing_a_search_CHANGED_is_not_a_repeat():
    """The ban was aimed at flogging a dead string through tool after tool, and that stays.
    But taken absolutely it also forbade adding a scope — which, since near_text landed, is
    precisely the move the receipts now tell the agent to make."""
    policy = _not_found_policy()

    assert "UNCHANGED" in policy
    assert "near_text" in policy


def test_many_matches_is_covered_as_its_own_case():
    """The whole section is written about NOT FINDING something. The case that actually
    fired was the opposite — 24 hits — so none of its advice applied, and "try a different
    approach" was the only line that seemed to."""
    policy = _not_found_policy()

    assert "MANY matches" in policy


def test_the_stale_index_recovery_bans_only_the_unchanged_retry():
    """Same over-broad clause, second site: "Never re-issue the same string, through that
    tool or any other" sits in the stale-index recovery steps."""
    section = (SPEED_OPTIMIZATION_PROMPT
               .split("Element index N not available")[-1]
               .split("find_elements is for STRUCTURAL")[0])

    assert "unchanged" in section.lower()
