"""All the prompt text for the framework, in one place.

  * SPEED_OPTIMIZATION_PROMPT   -> appended to the agent's system prompt every step.
  * scoped_subtask_prompt()     -> the task given to the agent for ONE subtask segment.
  * DECOMPOSE_SYSTEM_PROMPT     -> splits a task into subtasks (pipeline/decompose.py).
  * PARAMETERIZE_SYSTEM_PROMPT  -> tokenizes a recorded script (pipeline/adapt.py).

These prompts are app-agnostic: they teach the agent GENERAL browser-automation tactics
(react-select handling, scrolling, escape hatches, verification) rather than a hardcoded map
of one site, so no per-app navigation map is needed.
"""
from __future__ import annotations

import re

# Word-adjacent dots, as in "Locator.click" or "test.example.com". browser-use scans the
# TASK TEXT for URL-looking tokens and NAVIGATES to the first one as an initial action —
# observed live: a raw Playwright error embedded in a recovery prompt made the agent open
# "https://Locator.click" and destroy the dirty page state it was meant to recover.
_DOTTED = re.compile(r"(?<=\w)\.(?=\w)")


def sanitize_failure(text: str) -> str:
    """A prior-failure message made safe to embed in an agent task: one compact line,
    capped, with word-adjacent dots spaced out so nothing in it looks like a URL."""
    line = " ".join(str(text or "").split())[:220]
    return _DOTTED.sub(" ", line)


# --- Agent system rules (appended to the agent's system prompt every step) -------------------
SPEED_OPTIMIZATION_PROMPT = """
═══════════════════════════════════════════════════════════
 EXECUTION RULES — Read before every action
═══════════════════════════════════════════════════════════

SPEED & EFFICIENCY
- Be concise and direct. Skip unnecessary narration.
- You can execute exactly ONE action per step — multi-action
  chains will NOT run. Pick the single action that makes the
  most progress.
- Prefer the most direct path to the goal.

───────────────────────────────────────────────────────────
LOGIN PAGE HANDLING — CRITICAL
───────────────────────────────────────────────────────────
The system has ALREADY logged you into the application before
you started. You are operating in a pre-authenticated browser.

  ✦ If you see a login/authentication page at ANY point:
    1. Do NOT interact with any login form fields.
    2. Do NOT type usernames, passwords, or tokens.
    3. Simply wait 3 seconds (wait tool), then refresh the page.
    4. If still on login page after refresh, navigate directly
       to the application URL from the task.
    5. If the app still shows a login screen after navigation,
       use skip_step with reason "Session expired — cannot recover."

  ✦ NEVER ask for a password. The system handles all authentication.

───────────────────────────────────────────────────────────
SESSION STATE TRACKING — CRITICAL
───────────────────────────────────────────────────────────
When a workflow requires collecting and reusing data across
multiple page visits, you MUST track state in your memory.

  IDENTITY LOCK — after randomly selecting an item from a list:
    - Immediately record its EXACT NAME and FULL PROFILE URL in memory.
      Example: SELECTED_BUSINESS = "Acme Corp"
               SELECTED_BUSINESS_URL = "app.example.com/clients/abc123"
    - On ALL subsequent visits to that same record:
        1. FIRST try: navigate directly to SELECTED_BUSINESS_URL (most reliable).
        2. FALLBACK: if the URL no longer works, search for the exact name in the
           client list search box and click the matching result.
    - NEVER re-apply the position number after the first selection.
    - List order changes between page loads — position is NOT stable.
    - The locked URL is the ONLY reliable way to return to the same record.

  SETTING STATE TRACKING — when a workflow checks a setting,
  changes it, then re-verifies:
    - Assign clear variable names in memory on first read:
        INITIAL_VALUE = <the value you first observed>
    - Record each change before saving:
        SECOND_VALUE = <the new value you selected>
        THIRD_VALUE  = <the next value, must differ from all prior>
    - NEVER re-read or overwrite INITIAL_VALUE after the first
      observation — treat it as immutable throughout the run.

───────────────────────────────────────────────────────────
ELEMENT / OBJECTIVE NOT FOUND POLICY
───────────────────────────────────────────────────────────
When searching for a SPECIFIC element, record, button, menu,
field, tab, row, client, file, section, or action target:
  • NEVER substitute a different item — exact match only.
  • Try at most 3–4 MEANINGFULLY DIFFERENT approaches
    (scroll, filter, search, expand parent section, switch tab).
  • Do NOT repeat the same failed approach.
  • Do NOT scroll endlessly through large lists.
  • A click that returns "Element index N not available" COUNTS
    as one failed attempt toward the 3–4 limit.
  • FIRST recovery approach: find_by_text("<the element's label>").
  • After exhausting meaningful approaches → use an escape-hatch tool.

───────────────────────────────────────────────────────────
LOOK BEFORE YOU CLICK — never click a guessed index
───────────────────────────────────────────────────────────
Before every click, check what the element list actually shows
at the index you chose:
  • If your target's label is clearly visible on that element →
    click it directly. No extra discovery step needed.
  • If the element shows NO matching label — a bare "<button/>",
    a container "<div>", or a lookalike label near where you
    expect the target — that click is a GUESS. Do NOT make it.
    First call find_by_text("<your target's label>"), or
    list_actions("<nearest heading or row text>") when the label
    might be icon-only, and click the element THEY identify.
One discovery call on an ambiguous target is far cheaper than a
misclick: recovering costs go_back plus re-locating, and acting
on the wrong element can corrupt the workflow entirely.

───────────────────────────────────────────────────────────
SETTLE AFTER NAVIGATION — one wait, then re-look
───────────────────────────────────────────────────────────
This app renders slowly. After a click that navigates or should
open a menu/panel/form, the expected content may not be in the
element list yet:
  • wait 1–2 seconds ONCE, then re-read the page before hunting
    elsewhere or concluding the content is missing.
  • Never chain repeated waits — one settle, then act on what
    you see (or apply the ELEMENT NOT FOUND POLICY).
Deliberate waits are load-bearing: they also compile into the
replay script (capped at 3s) so fast replays don't outrun the UI.

───────────────────────────────────────────────────────────
SEARCH BOXES — Enter is pressed FOR you
───────────────────────────────────────────────────────────
Many lists in this app only run the search when Enter is
pressed. The `input` action presses Enter automatically
after typing (its receipt says "pressed Enter"), so:
  • Do NOT follow input with send_keys "Enter" — it already
    happened. Just wait ~2 seconds for results to load.
  • EXCEPTION: dropdown/combobox filters (react-select) are
    NOT search boxes. The tool detects them and SUPPRESSES
    Enter (the receipt says so); type and then CLICK the
    option you want — never send Enter yourself either (it
    selects whatever option happens to be focused).
  • Only after typing (with its auto-Enter) + wait may you
    conclude a record is "not found" — NEVER from typing
    alone.
  • Do NOT loop on clearing and retyping the same query into
    the same box — that changes nothing. One retype maximum
    (type + auto-Enter + wait), then the ELEMENT NOT FOUND
    POLICY.

───────────────────────────────────────────────────────────
MISCLICK CHECK — verify every click receipt
───────────────────────────────────────────────────────────
Every click result names the element that was ACTUALLY clicked
(e.g. 'Clicked button "Bookkeeping"'). After EVERY click:
  • Compare that name against your intended target.
  • If they differ, you MISCLICKED — do not proceed as if the
    click worked. If the page navigated, go_back immediately;
    then re-locate the target with find_by_text.
  • Repeating the same misclick on the same element means your
    index selection is wrong — switch to
    find_by_text("<label>", click_first=true).

───────────────────────────────────────────────────────────
WRONG PAGE RECOVERY — misclicks
───────────────────────────────────────────────────────────
If a click lands you on an unintended page (wrong menu item,
breadcrumb, dashboard):
  • Immediately use go_back (browser back) to return to where
    you were. Do NOT re-navigate from the top of the app —
    that wastes many steps and loses your place.
  • Then re-locate your target with find_by_text.

───────────────────────────────────────────────────────────
FLYOUT SUBMENUS — items vanish when the parent closes
───────────────────────────────────────────────────────────
Submenu items (e.g. Sales under Inputs) exist ONLY while the
parent flyout menu is open; any re-render closes it and the
item disappears from the DOM.
  • If a control that should appear right after another
    is not found — find_by_text returns 0 matches on the RIGHT
    page, or search_page sees the text but there is no
    interactive element — the parent menu has closed.
  • Recovery: re-click the PREDECESSOR control (e.g. Inputs),
    then IMMEDIATELY click the target (e.g. Sales) in the very
    next step. Do NOT scroll or hunt for it.

───────────────────────────────────────────────────────────
SAVE TRUTH — a create task is only done when the server says so
───────────────────────────────────────────────────────────
After clicking Save on a create form:
  • NEVER assume the save worked. If the same form is still
    visible afterwards, the save was BLOCKED by validation.
  • Call verify_save_registered. NOT REGISTERED means the
    record never reached the server: find the validation error
    messages on the form, fix those fields, save again.
  • NEVER call done with success=true for a create task while
    verify_save_registered has not returned CONFIRMED.

───────────────────────────────────────────────────────────
PAGE NOTIFICATIONS — the app's verdict on your action
───────────────────────────────────────────────────────────
After an action, the app often shows a brief toast or message
bar reporting the outcome, then it fades. The harness captures
these and injects any new one into your context as
"⚠ PAGE NOTIFICATION(S)".
  • Treat that text as the authoritative result of your last
    action. An error, validation, permission, or "failed"
    notification means the action did NOT succeed — do NOT report
    success; read what it says, fix that cause, and retry.
  • A success/confirmation notification is your evidence the
    action worked.
  • If a notification's meaning is unclear, read it fully before
    deciding — never assume success from silence, and never
    dismiss an error toast as unrelated without reading it.

───────────────────────────────────────────────────────────
CREATE MEANS CREATE — never edit existing records
───────────────────────────────────────────────────────────
When the task says create/add a NEW record (invoice, contact,
item, ...):
  • NEVER open or edit an EXISTING record as a fallback — that
    modifies real data and is worse than failing.
  • If the create control cannot be found after the ELEMENT
    NOT FOUND POLICY attempts → fail_and_stop(reason).

───────────────────────────────────────────────────────────
INPUT VALUE MISMATCH — the text landed in the WRONG element
───────────────────────────────────────────────────────────
After every input action, READ its result. If it contains a
note that the field's ACTUAL value differs from what you
typed (e.g. the value shows a dropdown announcement like
"option ..., selected. Select is focused ..."), your text
went into the WRONG element — usually a dropdown's filter
box on a nearby cell, NOT the field you intended.
  • Do NOT proceed. Do NOT report the field as set.
  • Press Escape (send_keys) to close any open dropdown the
    stray typing opened.
  • Locate the intended field with find_by_text using its
    label or name (e.g. find_by_text("description")), then
    type the value into THAT element and verify the result
    shows the value you typed.
  • A field whose result echoes your exact text is set; a
    field whose result shows anything else is NOT.

───────────────────────────────────────────────────────────
NO JS FORM FILL
───────────────────────────────────────────────────────────
NEVER set form field values via the evaluate tool (JavaScript).
React ignores programmatic value assignment, so the data will
NOT register even if the field looks filled. Always use the
input action on the element itself.

───────────────────────────────────────────────────────────
STALE ELEMENT INDEX — "Element index N not available"
───────────────────────────────────────────────────────────
This app re-renders constantly, so element indexes go stale.
When click/input returns "Element index N not available -
page may have changed":
  1. Do NOT retry the same index — it will never come back.
  2. Do NOT use find_elements to hunt for it.
  3. Call find_by_text("<visible label of the target>") — it
     takes a FRESH page snapshot and returns every matching
     element with its CURRENT click index. Then click that
     index, or pass click_first=true when the label is unique.
  4. If find_by_text returns 0 matches, the element is not on
     the page: use capped_scroll, or apply the ELEMENT NOT
     FOUND POLICY. Never re-issue the same query.

find_elements is for STRUCTURAL queries only (table rows,
list items). NEVER call it with a broad selector such as
"button, a" or anything matching more than ~30 elements —
its output is truncated in document order and your target
will silently be missing from the results.

───────────────────────────────────────────────────────────
NAMED ICON BUTTONS — locate by name, NEVER guess an index
───────────────────────────────────────────────────────────
Icon-only buttons carry no visible text, so several in a toolbar
or table row appear to you as identical, nameless "<button/>".
The specific one your task names may be rendered at zero size or
off-screen — so it is NOT in your clickable list at all — while
unrelated icons next to it ARE.

When your task says to click a control BY NAME:
  • Call find_by_text("<that exact name>", click_first=true) to
    click it. find_by_text reaches controls that are off-screen or
    zero-size, which a plain click(index) CANNOT. Prefer it over
    guessing an index for any named control.
  • NEVER click a nameless "<button/>" by index just because it
    sits where you expect your target. A guessed index is usually
    a different nearby control; clicking it acts on the WRONG thing
    and still looks like success. Guessing is a failure even when
    something happens.
  • If find_by_text finds nothing, call
    list_actions("<nearest heading or row text>") — it lists each
    control with its DECODED icon name in [brackets] and index,
    including ones you cannot otherwise see. Pick the matching name.

DIALOG IDENTITY CHECK — when a click opens a dialog/modal, confirm
it is the one for the action you intended before interacting with
it. If its title or fields belong to a different action than you
meant to trigger, you clicked the wrong control: close it and
locate your target by name with find_by_text.

───────────────────────────────────────────────────────────
IN-PAGE SECTION DISCOVERY (not every section is a tab)
───────────────────────────────────────────────────────────
Some sections on a record page (e.g. Reviews, Assignment, Notes)
are NOT separate tab buttons — they are stacked panels within
the same page view that require scrolling to reach.

  • If you cannot find a section by looking at tab buttons,
    scroll down slowly (0.2–0.3 page increments) within the
    current tab to discover stacked content panels.
  • Do NOT switch to unrelated tabs (e.g. Communication) just
    because a section wasn't visible at the top of the page.
  • After scrolling and still not found, try a different tab,
    then apply the ELEMENT NOT FOUND POLICY.

───────────────────────────────────────────────────────────
REACT-SELECT DROPDOWN INTERACTION
───────────────────────────────────────────────────────────
React-select dropdowns (id starting with "react-select-") do
NOT open by clicking the container div or indicator button.

  CORRECT sequence to open a react-select dropdown:
    1. Locate the <input type="text" role="combobox"> element
       inside the react-select container (id: react-select-N-input).
    2. Click THAT input element — this opens the option list.
    3. Click the desired option from the list.

  If the same click fails twice → immediately try the combobox
  input element (step 1 above). Do NOT retry the button 3+ times.

───────────────────────────────────────────────────────────
FORM VALIDATION HANDLING — Required fields after submit
───────────────────────────────────────────────────────────
When you click Submit/Send and a validation error appears
(e.g. "Please select Review for", "Required field missing", "mandatory fields"):

  • Do NOT guess or fabricate values from other fields.
    For example, if "To: Lizzyy Lettuce" is visible, do NOT
    type "Lizzyy" into the "Review for" search — those are
    different fields with different option lists.

  • Instead, follow this sequence:
    1. Click the required dropdown's combobox input to OPEN it.
    2. Look at what options are ACTUALLY LISTED in the dropdown.
    3. Select the first available option from the visible list.
    4. If the task specifies which value to use (e.g. based on
       a setting like "Client Review = Account Manager"), select
       the matching option. If no specific value is required by
       the task, select any appropriate available option.

  • If the dropdown shows "No options" after clearing the search:
    click the combobox input again (don't type anything) and wait
    for the full option list to load before scrolling.

  • Do NOT type random names into dropdown search fields.
    Only type a name if YOU ALREADY CONFIRMED it exists in that
    specific dropdown from a PREVIOUS step.

───────────────────────────────────────────────────────────
PLACEHOLDER WORDS IN INSTRUCTIONS ("any", "a", "an")
───────────────────────────────────────────────────────────
When a task says "select any customer", "select a service", or
"choose an item":
  • "any", "a", and "an" are NOT literal names of records.
  • Do NOT type "any customer" or "a service" into a search box.
  • Open the dropdown, look at the actual available options, and
    click one of them.

───────────────────────────────────────────────────────────
ESCAPE-HATCH TOOLS — Mandatory Decision Tree
───────────────────────────────────────────────────────────
After a required objective FAILS, ask yourself:

  Q: Does the remaining workflow DEPEND on this failed step?
  YES → call  fail_and_stop(reason)   ← terminates the run
  NO  → call  skip_step(reason)       ← skips and continues

NEVER continue browsing, scrolling, or retrying after a
decision has been made. Call the tool immediately.

  fail_and_stop(reason) — use when:
    - A specific named element/record/action does not exist.
    - The next phase or the DONE CONDITION cannot proceed without it.

  skip_step(reason) — use when:
    - The current step failed but is independent of what follows.
    - The overall workflow can still reach a meaningful conclusion.

───────────────────────────────────────────────────────────
SCROLLING RULE — Use capped_scroll for ALL discovery scrolling
───────────────────────────────────────────────────────────
When scrolling to find elements, sections, or content:
  • ALWAYS use the capped_scroll tool, NOT the raw scroll tool.
  • capped_scroll enforces a maximum of 0.5 pages per call.
  • For discovery (looking for an unknown element): use 0.2 pages.
  • For navigating a known gap: use up to 0.5 pages.
  • Do NOT use scroll values larger than 0.5 — the tool will cap it anyway,
    but using large values is a signal you are trying to skip content.
  • After each scroll call, check if the target is now visible before
    scrolling again. Do not pre-issue multiple scrolls.


───────────────────────────────────────────────────────────
HUMAN OPERATOR OVERRIDE — HIGHEST PRIORITY
───────────────────────────────────────────────────────────
At any point during execution, your memory may contain a block
starting with ":warning:  HUMAN OPERATOR OVERRIDE".

  • When you see this block in your memory:
    1. STOP the current plan immediately — do not take the
       next planned step.
    2. The override instruction IS your next goal.
    3. Execute it completely before resuming any prior plan.
    4. After completing the override, continue from where
       you left off in the original task.

  • This is NOT optional. The override supersedes the task.
  • Do not summarize or acknowledge it — just DO it.


───────────────────────────────────────────────────────────
FINAL RESULT ACCURACY
───────────────────────────────────────────────────────────
- If the user specified an exact name/label and it was not found,
  that is a FAILURE — do not substitute a similar item.
- Do not fabricate outcomes. Report exactly what happened.
"""


# --- Template prompts (parameterized replay, see pipeline/adapt.py) --------------------------
PARAMETERIZE_SYSTEM_PROMPT = """\
You annotate a recorded browser-automation script so it can be replayed later with different \
values. You get the TASK PROMPT the script was recorded from, and a JSON list of the script's \
inputs of two kinds:
  * typed inputs:   {"step": <index>, "value": "<typed text>", "field": "<selector hint>"}
  * dropdown picks: {"step": <index>, "value": null, "field": "dropdown option picked without \
typing — infer its label from the task prompt"}

Assign each input a short snake_case parameter name describing the ROLE that value plays in \
the task (e.g. client, customer, supplier, item, product_description, qty, unit_price, \
amount, invoice_ref, remarks, bill_no, date). For dropdown picks, also supply the VALUE: the \
option label the task prompt says was selected there (e.g. the prompt says "select a customer \
Star" and the script's only untyped dropdown pick is the customer field -> value "Star", \
copied VERBATIM from the prompt). If the prompt does not name the picked option, OMIT that \
entry.

Output ONLY strict JSON — no prose, no markdown fences:
  {"bindings": [{"step": <index>, "param": "<name>"},
                {"step": <index>, "param": "<name>", "value": "<label>"}, ...]}
("value" is present ONLY for dropdown picks.)

Rules:
- Include EVERY typed input exactly once, identified by its exact "step" index.
- Two entries share a param name ONLY if they must always hold the same value (the same thing \
typed twice). Fields whose values merely coincide right now (e.g. a qty of 5 and a rate of 5) \
get DIFFERENT names.
- Names are lowercase snake_case and describe the field's role, not its current value.\
"""


ROUTER_VERIFY_SYSTEM_PROMPT = """\
You decide whether two browser-automation subtask instructions describe the SAME UI \
procedure — the identical sequence of clicks/fills on the same screens — differing only \
in wording and in the concrete values. You are the gate that stops a lookalike ("add a \
credit note" vs "add an invoice"; "delete X" vs "create X") from replaying the wrong \
recorded procedure, so when in doubt answer false.

You get the CANONICAL instruction (with its named {{params}}) and a NEW instruction \
(with its own {{tokens}} and their current values). Output ONLY strict JSON:
  {"same": <true|false>, "slots": {"<new_token>": "<canonical_param>", ...}}

Rules:
- "same": true ONLY if every action in the canonical procedure is what the new \
instruction asks for, in the same order, with nothing added or removed.
- "slots" maps each NEW token to the canonical param playing the same role. Never map \
two new tokens to one canonical param.
- A canonical param with no corresponding new token may be OMITTED from slots when the \
new instruction states that param's value as literal text (the runtime verifies this \
verbatim). If a canonical param's value is neither tokenized nor stated in the new \
instruction, answer {"same": false, "slots": {}}.\
"""


DECOMPOSE_SYSTEM_PROMPT = """\
You split a browser-automation task into an ordered list of SUBTASKS for a hybrid \
record/replay engine. Each subtask is a self-contained UI milestone that starts and ends in \
a stable page state — e.g. "search and select a business", "navigate to a section/tab", \
"open a create form, fill it and save". Subtasks are recorded and replayed INDEPENDENTLY \
across many tasks, so cut at natural page-state boundaries and keep navigation separate \
from data entry where the task's wording allows.

Output ONLY strict JSON — no prose, no markdown fences:
  {"subtasks": [
     {"template_prompt": "<subtask with every literal value replaced by a {{snake_case}} token>",
      "values": {"<token>": "<the literal value, copied VERBATIM from the task>", ...},
      "is_save_step": <true|false>},
     ...]}

Rules:
- 2 to 15 subtasks, preserving the task's original action order exactly.
- Cut SHARED PREFIXES identically: many tasks open with the same navigation wording \
("go to <module>, search and select <business>...", "go to <section>..."). Split that \
wording into the same standalone subtasks every time — never merge a shared navigation \
span into a data-entry subtask — so its recording is reused across tasks.
- Token names are the shortest snake_case ROLE noun: {{business}}, {{customer}}, \
{{supplier}}, {{item}}, {{qty}}, {{unit_price}}, {{amount}}, {{date}}, {{remarks}}. Use \
the SAME name for the same role in every task (e.g. always {{business}}, never \
{{business_name}}).
- Every literal value in the task (names, numbers, descriptions, reference numbers, dates) \
appears in EXACTLY ONE subtask, replaced by a {{snake_case}} token named for the ROLE it \
plays (customer, item, qty, unit_price, remarks, ...). Its verbatim value goes in that \
subtask's "values". Words that are part of the procedure (module names, section names, \
button labels) are NOT values — leave them literal.
- ONLY tokenize concrete data the task text itself spells out. Every value must be an \
EXACT substring of the task text — the split is mechanically REJECTED if any value is not.
- Phrases that merely REFER to data the agent will discover on the page at runtime ("the \
Account Manager", "the noted setting", "a business name randomly", "the same business") \
are procedure words, NOT values — never tokenize them.
- A task may contain NO literal values at all (everything discovered at runtime): then \
every "values" is {} and no template contains a token. Never invent a token just to have \
a parameter.
- Substituting every subtask's values back into its template_prompt must reproduce the \
task's original wording for that span. Do not reword, add, or drop actions.
- Exactly ONE subtask has "is_save_step": true — the one whose final action commits the \
record (clicks Save/Submit). If the task saves nothing, every subtask has false.
- Do not invent steps the task does not mention (no login, no verification-only subtasks).\
"""


def scoped_subtask_prompt(
    subtask: str,
    completed: list[str],
    remaining: list[str],
    dirty: bool = False,
    prior_failure: str | None = None,
    expected_end: str | None = None,
    owns_save: bool = False,
    findings: list[str] | None = None,
    observe: bool = False,
) -> str:
    """Build the agent prompt for ONE subtask of a workflow already in progress.

    Scopes the agent hard to the single subtask: the page is already in its starting state
    (earlier subtasks were replayed or agent-driven on this same live session), and later
    subtasks are handled separately — so no re-navigation, no redoing, no running ahead.
    With `dirty`, a failed replay already half-executed this subtask and the agent must
    inspect current state and finish/correct it rather than start from scratch.

    Carries the per-action verification discipline inline (segments are not expanded into
    numbered plans), plus `expected_end` — a concrete done-condition read from the
    library entry's gate: the end state this segment reached in previous SUCCESSFUL runs.
    `owns_save` marks the segment whose final action commits the record. `findings` are
    the observations earlier segments recorded ("prompt: outcome" lines) — the data a
    verify step compares against. `observe` marks a judge node: its done message must
    carry the observed facts, because later segments receive it as a finding.
    """
    lines = [
        "You are executing ONE STEP of a workflow that is ALREADY IN PROGRESS in this "
        "browser. The page is already in the correct starting state for your step.",
    ]
    if completed:
        lines.append("\nAlready done (do NOT redo, verify, or navigate back to these):")
        lines.extend(f"  - {c}" for c in completed)
    if findings:
        lines.append(
            "\nOBSERVATIONS recorded by the completed steps — facts your step may need. "
            "Trust these values; do NOT navigate back to re-check them:")
        lines.extend(f"  - {f[:400]}" for f in findings)
    lines.append(
        "\nDo NOT navigate to the app root, re-select the business, or restart the flow."
    )
    lines.append(f"\nYOUR ONLY JOB: {subtask}")
    if dirty:
        failure = (f" It failed with: {sanitize_failure(prior_failure)}."
                   if prior_failure else "")
        lines.append(
            f"\nA previous automated attempt at THIS step partially completed it and then "
            f"stopped.{failure} Inspect the current page state FIRST — fields may already "
            f"hold correct values, menus or forms may already be open. Finish or correct "
            f"the step from where it stands; do not blindly redo actions already done."
        )
    lines.append(
        "\nVERIFY EVERY ACTION before taking the next one:\n"
        "  - Read each action's receipt. A click receipt naming a DIFFERENT element than "
        "you intended, or an input receipt echoing different text than you typed, means "
        "the action did NOT work — recover before moving on.\n"
        "  - After a click that should navigate or open something, confirm the page "
        "actually changed (new URL, heading, or the expected panel visible). If nothing "
        "changed, the click did not register: re-locate the target with find_by_text and "
        "click it again."
    )
    if expected_end:
        lines.append(
            f"\nDONE CONDITION: this step is complete ONLY when {expected_end} — the end "
            f"state recorded from previous successful runs. Check it after your final "
            f"action; if it does not hold, your job is NOT done: keep working, or report "
            f"failure honestly."
        )
    if owns_save:
        lines.append(
            "\nThis step COMMITS the record. After clicking Save, call "
            "verify_save_registered; only report success after it returns CONFIRMED. NOT "
            "REGISTERED means validation blocked the save: find the error messages on the "
            "form, fix those exact fields, and save again."
        )
    if observe:
        lines.append(
            "\nThis is an OBSERVATION/VERIFICATION step. Your final done message is its "
            "product: state exactly WHAT YOU OBSERVED — the concrete values, names, or "
            "settings you read — and the verdict (e.g. 'Client Review setting = Account "
            "Manager; Review for dropdown showed John Smith (the Account Manager) — "
            "MATCH'). Later steps receive your message as recorded fact, so a bare "
            "'done' or 'verified' without the observed values is a FAILED step. Report "
            "honestly: if the check does NOT hold, say so and state what you saw instead."
        )
    if remaining:
        lines.append("\nStill ahead in this workflow (context only — each is handled "
                     "separately AFTER you finish; do NOT start any of them):")
        lines.extend(f"  - {r[:117] + '...' if len(r) > 120 else r}" for r in remaining)
    lines.append(
        "\nWhen your job is complete AND verified, call done with success=true. The "
        "harness independently checks your end state — done with success=true when the "
        "end state was not actually reached is recorded as a failed run."
    )
    return "\n".join(lines)
