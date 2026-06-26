"""All the prompt text for the framework, in one place.

Three prompts, each used by a different part of the pipeline:
  * APP_SYSTEM_RULES      -> appended to the agent's system prompt every step (kept lean).
  * APP_MAP               -> the app's navigation structure, given to the expander.
  * EXPANSION_SYSTEM_PROMPT-> the meta-prompt that rewrites a terse task into concrete steps.

`expand_task()` (the only logic here) runs the expander: one LLM call that turns a high-level
task into an explicit step list, using APP_MAP so it references the app's real menus/paths.
"""
from __future__ import annotations

import logging

from browser_use.llm.messages import SystemMessage, UserMessage

logger = logging.getLogger("framework.prompts")


# --- Agent system rules (sent to the agent every step; deliberately short) -------------------
APP_SYSTEM_RULES = """\
You are testing the Acting Office web app (a slow Fluent UI / React app that re-renders often).
Rules:
- Do exactly what the task says: enter only the values it gives, in the fields it names. Never \
invent a value or fill a field the task did not mention.
- Dropdowns are react-select, not native <select>: click the field, type the value to filter, \
then click the matching option or press Enter. Type ONLY the value; press keys like ArrowDown \
and Enter as separate send_keys actions, never as text.
- Search boxes have no submit button: after typing, press Enter. Never click the small x / \
clear icon inside a field.
- For number and price fields, clear the field fully before typing, then check it shows the \
value you intended.
- Add a new row or line only if the task has more than one entry. Set values in the existing \
row's fields; do not use "+" / "Add" buttons to pick a value.
- The app is slow: after navigating or saving, wait and re-read the page before deciding it \
worked. Click Save only once.
- If you cannot find a menu item, scroll or expand sections before concluding it is missing.
- If an action fails or the page is not what you expected, do not repeat it -- try another way.
- Before saving, check the form matches the task (right values, right fields, no extra rows or \
changes) and fix any mismatch first.
- Stay inside this application.\
"""


# --- App navigation map (given to the expander so it uses the app's real menus/paths) --------
APP_MAP = """\
ACTING OFFICE -- APP STRUCTURE (use the real names/paths below; do not invent navigation):

MODULES
- Open modules from the Modules button in the top-right bar. Two main modules:
  - Practice / CRM dashboard (URL /admin).
  - Bookkeeping (the 'Bookkeeping' link, /books).

BOOKKEEPING -> pick a client
- If the task NAMES a specific client, type that name in the search box and click the matching
  result. If the task says ANY client (no name given), just click the first client shown in the
  list -- do NOT type a placeholder. Everything then lives under /books/clients/<clientId>/...

CLIENT LEFT-NAV (inside Bookkeeping): Dashboard, Inputs, Banking, VAT returns, Reports,
Budget manager, Settings.
- Inputs -> two sub-sections: Sales and Purchase.
    - Sales -> Invoices tab -> '+ Invoice' button.   (path .../inputs/sales/invoices)
    - Purchase -> Invoices -> '+ Purchase' / Add.     (path .../inputs/purchase/invoice)
- Banking -> Banks: bank-account cards, '+ Account' to add one.  (path .../banking/banks)
- VAT returns, Reports, Budget manager, Settings: their own sections.

PRACTICE / CRM module (/admin) LEFT-NAV: Dashboard, Emails; Operations (Clients, Tasks,
E-signatures, Deadlines); Sales (Leads, Quotes & Proposals, Letters, Chats); Premium
(Timesheet, Documents, Taxbot, Decision trees); Reports.

UI PATTERNS
- The client left-nav may be a COLLAPSED ICON RAIL -- a menu item (e.g. 'Inputs') may show as
  an icon only; expand/scroll the rail to reveal labels before deciding it is missing.
- 'Add a record' is always a '+<Thing>' button: '+ Invoice', '+ Purchase', '+ Account',
  '+ Service' (adds a line row).
- Customer/Supplier, Item, and Account fields are react-select dropdowns (open, type to filter,
  pick the option).
- Invoice and Purchase forms share one line-item row: Item, Description, Account, Qty, Unit
  price. Finish with the 'Save' button (NOT 'Save & New'). New records get a number
  (INV-xxxx for sales invoices, PUR-xxxx for purchases).\
"""


# --- Expander meta-prompt + logic ------------------------------------------------------------
EXPANSION_SYSTEM_PROMPT = """\
You rewrite a browser-automation task into a precise, step-by-step instruction list for an \
LLM agent that controls a real web browser. The agent is already logged in and on the target \
web application.

Output ONLY the rewritten task: a numbered list of atomic steps, nothing else (no preamble, \
no explanation, no markdown headers).

Write each step as ONE concrete action, and where the action is obvious name the agent's \
actual action verb: click, input_text (type into a field), send_keys (press keys such as \
Tab / ArrowDown / Enter), scroll, wait, go_back, extract, or done. Split compound \
instructions apart.

Rules for the rewrite:
1. Be specific, never vague. Each step names the exact element (visible label, placeholder, \
or id) and the exact value to use. Preserve the labels, ids, numbers, and option text the \
user gave verbatim -- do not invent or change data. BUT if the task says "any", "a", or a \
"random" item without naming a specific one (e.g. "any client", "any supplier"), do NOT invent \
a placeholder value to type -- instead pick the FIRST available one from the list (e.g. "click \
the first client shown"). Only type to search/filter when the task gives a specific name/value.
2. Any CSS-like hint (e.g. `a.ms-Link`, `input#SearchBox129`, an element id/class, or a \
react-select placeholder id) is an element ALREADY ON THE PAGE, NOT a web address. Never tell \
the agent to navigate to it as a URL. Phrase it as "click the element with id/class X" or \
"the field whose placeholder is Y".
3. Never instruct the agent to leave the application, open another site, or use a search \
engine. All actions happen inside the current app.
4. For any dropdown / combobox / react-select field, write SEPARATE steps (never merge the \
value with the keys): (a) click the field to open the list; (b) input_text with ONLY the \
value to filter by (e.g. "Service") -- never include key names in the typed text; (c) click \
the matching option, with a keyboard fallback as its own step: "if the option does not click, \
press ArrowDown then Enter via send_keys (these are keys to press, NOT text to type)". Then \
verify the chosen value is shown.
5. Before typing into a numeric field (quantity, price), clear any existing value first.
6. Make fragile steps error-resilient: phrase them as "if <element> is not visible, scroll to \
it (or wait briefly) and retry" and "if a click does nothing, fall back to keyboard \
navigation with send_keys (Tab to move focus, ArrowDown to choose, Enter to confirm)". \
Prefer keyboard fallbacks whenever a direct click might fail.
7. After a step that navigates to a new page/section/form, add a short VERIFY step for what the \
app actually shows (e.g. "verify the URL now contains the expected path" or "verify the target \
form/page is visible"). Do NOT invent verification the task did not ask for -- in particular, \
NEVER require a "success message" / toast after saving; this app often shows none.
8. The SECOND-TO-LAST step confirms the TASK'S OWN goal was reached, matching what the task \
asked for -- not a fixed template: for a create/save task, the new record appears (in the list \
or as a new reference number) or the form closes; for a navigation task, the target \
page/section/form is shown; for a read/extract task, the requested info is present. Confirm \
ONLY what the task's goal implies, and only via what the app actually shows -- never add \
save/record confirmation to a task that just navigates, and never require a success message \
unless the task asks for one.
9. The LAST step must be an explicit termination instruction: "The task is now COMPLETE. Call \
the done action and stop immediately. Do NOT repeat any earlier step, do NOT redo the task, and \
do NOT start over." The agent tends to loop, so make stopping unambiguous.

Keep the list focused, ordered, and literal. Prefer clear instructions over clever ones.\
"""


async def expand_task(task: str, llm) -> str:
    """Rewrite `task` into explicit step-by-step instructions using `llm` + the app map.

    Returns the expanded task on success, or the original `task` unchanged on any failure.
    """
    try:
        system = (
            EXPANSION_SYSTEM_PROMPT
            + "\n\nUse this map of the target app to turn the task into concrete steps with the "
            "REAL menu/section names and paths (do not invent navigation):\n\n"
            + APP_MAP
        )
        result = await llm.ainvoke(
            [SystemMessage(content=system), UserMessage(content=f"Rewrite this task:\n\n{task}")]
        )
        expanded = (result.completion or "").strip()
        if not expanded:
            logger.warning("expander returned empty output; using original task")
            return task
        logger.info("task expanded (%d -> %d chars)", len(task), len(expanded))
        return expanded
    except Exception as exc:  # noqa: BLE001 - expansion is best-effort
        logger.warning("prompt expansion failed (%s); using original task", exc)
        return task
