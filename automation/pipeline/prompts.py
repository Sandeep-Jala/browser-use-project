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
Verb rules (follow these precisely):
- "select X"   -> choose X by CLICKING the option; for a dropdown you may type to filter first, \
then click. Do NOT type into a plain text/number field you were not told to.
- "set X to Y" -> INPUT_TEXT: type Y into the named field (clear first if prefilled).
- "click X"    -> a direct mouse click (button, link, or menu item). Do NOT type.
Rules:
- Do exactly what the task says: enter only the values it gives, in the fields it names. Never \
invent a value or fill a field the task did not mention.
- Dropdowns are react-select: click the field, type the value to filter, then CLICK the matching \
option once it appears. To press a key (ArrowDown/Enter/Tab) use send_keys, never input_text — \
those keys are actions, not characters. If your filter shows no match, clear it and pick from the \
options listed. If the task gives no specific value (e.g. "select an item from the dropdown"), \
pick the FIRST real option — never type a made-up placeholder.
- In a search box (not a dropdown), press Enter after typing to submit; in a dropdown, choose by \
clicking the option, not Enter. Never click the small x / clear icon inside a field.
- For number and price fields, clear the field fully before typing, then check it shows the \
value you intended. A money/Amount field that shows a "£" prefix (e.g. £0.00, as on Receipts and \
Payments) needs the pound sign: type the amount WITH it (e.g. "£1000") — a plain number may not \
register.
- Add a new row or line only if the task has more than one entry. Set values in the existing \
row's fields; do not use "+" / "Add" buttons to pick a value.
- To open a new-record form, click the exact add button ("+ Invoice", "+ Item", ...) — not a \
table row, a column header (e.g. "Invoice no."), or nearby controls (Import / Scan). Then VERIFY \
a blank form actually opened before filling anything; if it did not, your click missed — click \
the add button again.
- The app is slow: after navigating or saving, re-read the page once it has settled before \
deciding it worked.
- If you cannot find a menu item, scroll or expand sections before concluding it is missing.
- Probe sparingly: use find_elements / search_page at most ONCE for a target, with a broad \
selector, then ACT on the best candidate. Never repeat a probe that returned nothing — change \
strategy instead (scroll, use another anchor like aria-label or id, or click the best match).
- If an action fails or the page is not what you expected, do not repeat it -- try another way.
- Know where you are from the URL. If a navigation target (e.g. "Inputs") is missing, or the URL \
is /books without /clients/<id>, a misclick took you OUT of the client workspace — the target \
does not exist on this page, so do not keep clicking or probing for it. Re-navigate instead: \
module picker → Bookkeeping → search and select the client → continue from there.
- Before saving, check the form matches the task (right values, right fields, no extra rows or \
changes) and fix any mismatch first.
- A greyed-out / disabled Save or Create button means a REQUIRED field is missing or invalid — \
clicking it does nothing and the record is NOT saved. Disabled Save is NEVER a sign of success: \
find the empty/invalid field (e.g. an amount that did not register), fix it, then save.
- Saving can be TWO steps: after you click Save on the form, a confirmation/allocation dialog \
(e.g. "Allocated amount of CRN-xxxx") may pop up with its OWN Save button — click Save in THAT \
dialog too; the record is NOT committed until you do. Use plain Save both times; click "Save & \
New" only if the task explicitly says so. Do not repeatedly click the same Save button.
- Stay inside this application.\
"""


# --- App navigation map (given to the expander so it uses the app's real menus/paths) --------
APP_MAP = """\
ACTING OFFICE -- NAVIGATION & FORM REFERENCE
Use exact labels shown below. Never invent button names, tab names, or field names.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
START STATE & MODULES (read first)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After login you are in the PRACTICE / CRM module (URL /admin), whose left-nav is:
  Dashboard | Emails | Operations (Clients, Tasks, …) | Sales | Premium | Reports.
BOOKKEEPING is a SEPARATE module. For any Bookkeeping task, FIRST open the module picker
(grid icon, aria-label "menus" / id="btn-menus-callout") and click "Bookkeeping", then search
and select the client — only then does the Bookkeeping client left-nav appear:
  Dashboard | Inputs ▾ | Banking | VAT returns | Reports | Budget manager | Settings.
The two modules' left-navs are DIFFERENT but share some labels (BOTH have a "Clients" item), so
do not confuse them: "Inputs" exists ONLY in the Bookkeeping module. When a task says Inputs /
Sales / Purchases, click "Inputs" in the Bookkeeping left-nav — NEVER click "Clients". A CRM task
(e.g. "go to clients section → create invoice") stays in /admin and does NOT switch modules.

URL LANDMARKS (check the URL to know where you are):
  /admin                             = CRM module (start page)
  /books                             = Bookkeeping CLIENT LIST — no client selected yet
  /books/clients/<id>/dashboard      = client workspace — the left-nav with "Inputs" exists HERE
  /books/clients/<id>/inputs/sales   = Inputs > Sales (tab bar: Invoices | Credit notes | ...)
  ...same pattern for other sections (e.g. .../inputs/sales/creditnotes = Credit notes tab).
If the URL is /books WITHOUT /clients/<id>, you have LEFT the client workspace: "Inputs" does
NOT exist there, so do not search for it — re-select the client first, then continue.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUICK NAVIGATION GUIDE (task phrase → exact UI path)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"go to Bookkeeping module"       → click the grid/modules icon in the top-right bar
                                   (button aria-label="menus" or id="btn-menus-callout") to
                                   open the module picker popup, then click "Bookkeeping"
"search and select <client>"     → type name in search box (top-right of client list), press
                                   Enter (send_keys) to run the search, then click the row
"go to inputs section"           → click "Inputs" in left-nav (expands sub-items)
"select sales"                   → click "Sales" under Inputs
"select purchases"               → click "Purchases" under Inputs
"select expense claims"          → click "Expense claims" under Inputs
"select assets"                  → click "Assets" under Inputs
"select journals"                → click "Journals" under Inputs
"select dividends section"       → click "Dividends" under Inputs
"go to banking section"          → click "Banking" in left-nav (NOT under Inputs)
"go to Budget manager"           → click "Budget manager" in left-nav (NOT under Inputs)
"go to Invoices"                 → click "Invoices" tab
"go to Credit Notes"             → click "Credit notes" tab
"go to Estimates"                → click "Estimates" tab
"go to Receipts"                 → click "Receipts" tab
"go to Item"                     → click "Items" tab
"go to Purchase Orders"          → click "Purchase orders" tab
"go to Payments Section"         → click "Payments" tab
"go to Mileages Section"         → click "Mileage claims" tab  ← exact label is "Mileage claims"
"go to Reimbursements Section"   → click "Reimbursements" tab
"go to Refunds Section"          → click "Refunds" tab
"go to disposed"                 → click "Disposed" tab

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BOOKKEEPING CLIENT LEFT-NAV
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dashboard | Inputs ▾ | Banking | VAT returns | Reports | Budget manager | Settings
  Inputs expands to: Sales | Purchases | Expense claims | Assets | Journals | Dividends
  The rail shows icons only — target by anchor, not position: Inputs = aria-label "Inputs"
  (id=inputs); Sales = aria-label "Sales" (id=Sales). A "Clients" item (id=bkClients) sits in
  this rail too — it is NOT Inputs; if a click lands on Clients, re-target aria-label "Inputs".
  NOTE: Banking and Budget manager are direct left-nav items, NOT under Inputs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUTS > SALES  — tab bar: Invoices | Credit notes | Estimates | Receipts | Customers | Items
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INVOICES tab — add button "+ Invoice"
  Customer        react-select  placeholder "Contact name"
  Invoice no.     auto-filled (INV-xxxx) — leave unless task says to change
  P.O. reference  text
  Date / Due date date-pickers
  Discount        type/% selector
  Currency        selector (default £ GBP)
  LINE ITEM ROW (columns left→right):
    Item              react-select  label "Item"       placeholder "Select item"
    Product description  text       label "Product description"
    Account           react-select  label "Account"    placeholder "Select"  ← 1st Select in row
    Qty               numeric       label "Qty"
    Unit price        numeric       label "Unit price"  (clear before typing)
    VAT               react-select  label "VAT"        placeholder "Select"  ← 2nd Select in row
                      options: No VAT, 20% Standard, 5% Standard, etc.
    Net amount        read-only
  "+ Service"  adds another line-item row — do NOT use it to pick a value
  Note         text (bottom)
  Buttons: Save | Save & New | Cancel

CREDIT NOTES tab — add button "+ Credit note"
  The "+ Credit note" button is a small button ABOVE the credit notes list. Do NOT click on any
  table row or column header — those open existing records. After clicking the button, the
  new form opens with all fields empty. The page header shows the business client name as
  read-only context — it is not the credit note customer.
  Customer        react-select  label "Customer"      placeholder "Customer name"
  Credit note no. auto-filled (CRN-xxxx)
  Date            date-picker
  Invoice ref no. react-select  label "Invoice ref no."  placeholder "Select"
                  Options are populated only after a customer is selected (filtered to their
                  invoices). Invoice numbers display as "INV-xxxx". If a short search term
                  returns no results, try the zero-padded number or full "INV-xxxx" format.
  Buttons: Save | Cancel

ESTIMATES tab — add button "+ Estimate"
  Customer        react-select  placeholder "Customer name"
  Estimate no.    auto-filled (EST-xxxx)
  P.O. reference  text
  Date / Expiry date  date-pickers
  Line items (Item, Product description, Qty, Unit price, VAT), Note, Totals
  Buttons: Save | Save & New | Cancel

RECEIPTS tab — add button "+ Receipt"
  Received from   react-select  placeholder "Customer name"  ← field label is "Received from"
  Receipt no.     auto-filled (REC-xxxx)
  Date            date-picker
  Method          react-select  placeholder "Select"
  Amount          numeric (£0.00) — type the value WITH the pound sign (e.g. "£1000"); a plain
                  number may not register and Save will stay disabled
  Auto allocation toggle
  Note            text
  Buttons: Save | Save & New | Cancel

CUSTOMERS tab — add button "+ Customer"  (slide-in panel)
  Name, Contact person, email/phone
  Billing address: Building, Street, City, County, Country, Postcode
  Shipping address (same)
  Payment terms, Bank (react-select), Currency, Discount
  VAT number, EORI number, Project tags, Notes
  Buttons: Save | Cancel

ITEMS tab — add button "+ Item"  (MODAL DIALOG — not a full page)
  Name            text   ← REQUIRED. Fill this FIRST with the given item name and confirm it
                          shows before touching any other field; the modal often opens with focus
                          elsewhere, so the Name field is easy to skip — do NOT skip it.
  Code            text
  Description     two fields side-by-side: "For purchases" | "For sales"
  Unit price      two numeric fields:      "For purchases" | "For sales"
  Account name    two react-selects:       "Select" (purchases) | "Select" (sales)
  VAT rate        two react-selects:       "Select" (purchases) | "Select" (sales)
  Buttons: Create | Cancel   ← button is "Create", not "Save"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUTS > PURCHASES — tab bar: Invoices | Credit notes | Purchase orders | Payments | Suppliers | Items
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INVOICES tab — add button "+ Invoice"  (also: Import | Scan)
  Supplier        react-select  label "Supplier"  placeholder "Contact name"
  Invoice no.     auto-filled — new record gets number PUR-xxxx (not INV-xxxx)
  Line items, VAT, Note, Totals — same column order as Sales invoice
  Buttons: Save | Save & New | Cancel

CREDIT NOTES tab — add button "+ Credit note"
  Supplier        react-select
  Credit note no. auto-filled
  Invoice ref no. react-select  placeholder "Select"
  Buttons: Save | Cancel

PURCHASE ORDERS tab — add button "+ Purchase order"
  Contact name    react-select
  Line items (Item, Description, Qty, Unit price, VAT)
  Buttons: Save | Cancel

PAYMENTS tab — add button "+ Payment"
  Paid to         text / react-select  placeholder "Customer name"
  Payment no.     auto-filled
  Date, Method (react-select), Amount
  Buttons: Save | Cancel

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUTS > EXPENSE CLAIMS — tab bar: Expense claims | Mileage claims | Reimbursements | Refunds | Users
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXPENSE CLAIMS tab — add button "+ Expense"
  Use field labels to locate each field — do NOT rely on the generic placeholder "Select":
  User            react-select  label "User"      — lists director/user names;
                  "select director" / "select a director" = pick first option (role word, not a name)
  Remarks         text          label "Remarks"
  Bill no.        text          label "Bill no."
  Description     text          label "Description"
  Account         react-select  label "Account"   — 1st "Select" dropdown in the form
  Base amount     numeric       label "Base amount" — clear before typing
  VAT             react-select  label "VAT"       — 2nd "Select" dropdown; options: No VAT,
                                5% Standard, 20% Standard, etc.
  Buttons: Save | Cancel

MILEAGE CLAIMS tab — add button "+ Mileage"   ← tab label is "Mileage claims" (tasks say "Mileages Section")
  User            react-select (director dropdown)
  Remarks         text
  Engine type     react-select (Petrol / Diesel / Electric)
  Description     text
  Mileage         numeric
  Rate            react-select (45p / 25p / etc.)
  Buttons: Save | Cancel

REIMBURSEMENTS tab — add button "+ Reimbursement"
  Reimbursed To   react-select (user)
  Account         react-select
  Amount          numeric
  Buttons: Save | Cancel

REFUNDS tab — add button "+ Refund"
  Refund from     react-select
  Account         react-select
  Amount          numeric
  Buttons: Save | Cancel

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUTS > ASSETS — tab bar: Fixed assets | Depreciations/Amortisation | Disposed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIXED ASSETS tab — add button "+ Fixed asset"
  Asset name      text
  Account         react-select
  Purchase price  numeric (clear before typing)
  Supplier        react-select
  Rate            numeric
  Buttons: Save | Cancel

DISPOSED tab — add button "+ Dispose asset"
  Asset           react-select (existing fixed assets)
  Sales proceeds  numeric
  Payment method  react-select
  Customer        react-select
  Buttons: Save | Cancel

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUTS > JOURNALS — add button "+ Journal"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Journal reference  text (e.g. JRN001)
  Account            react-select
  Debit              numeric
  Credit             numeric
  Buttons: Save | Cancel

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUTS > DIVIDENDS — tab bar: Dividends | Shareholders — add button "+ Dividend"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Shareholder / authorised director  react-select
  Type                               react-select
  Dividend per share                 numeric
  Payment date                       date-picker
  Buttons: Save | Cancel

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BANKING (left-nav: "Banking") — add button "+ Account"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Account type    react-select (Savings / Current / etc.)
  Bank            react-select
  Account no.     text
  Sort code       text
  IBAN            text
  Primary account checkbox / toggle
  Buttons: Save | Cancel

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUDGET MANAGER (left-nav: "Budget manager") — add button "+ Budget"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Name       text
  Date       date-picker
  Frequency  react-select (Yearly / Monthly / etc.)
  Duration   react-select / numeric
  Buttons: Save | Cancel

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRACTICE / CRM MODULE (/admin)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Left-nav: Dashboard | Emails | Operations (Clients, Tasks, E-signatures, Deadlines) |
          Sales (Leads, Quotes & Proposals, Letters, Chats) |
          Premium (Timesheet, Documents, Taxbot, Decision trees) | Reports
Clients section: type name in search box → press Enter (send_keys) → click the client NAME
                 link in the result. Clicking the row body only PREVIEWS (URL stays on
                 /admin/clients?... with a &uid= param) — you are NOT on the client page until
                 the URL is /admin/clients/business/<id>. If the row click does not navigate,
                 use the GLOBAL search in the top bar instead: search the name there and click
                 the result (lands on /admin/clients/business/<id>).
                 On Save, a warning dialog may appear — click "Save anyway" to commit.
  CRM invoice form: Service (react-select), Amount, Discount (% + note),
                    Collection method (react-select)
  Buttons: Save | Cancel

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UI RULES (apply everywhere)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Left-nav may be a COLLAPSED ICON RAIL — click the icon to reveal the label before clicking.
- EVERY search box needs Enter to run the search: type the term, then press Enter as a separate
  send_keys step. Typing alone does not search. (Dropdowns are different: they filter as you
  type — click the option, never press Enter there.)
- Every dropdown is react-select: click to open → type to filter → click matching option.
- Auto-numbered fields (Invoice no., Receipt no., etc.) are pre-filled — do not change unless
  the task explicitly asks.
- Saving can be TWO steps: after Save on the form, a confirmation/allocation dialog (e.g.
  "Allocated amount of CRN-xxxx") may appear with its OWN Save button — click Save there too, or
  the record is NOT committed. Use plain "Save" both times unless the task says "Save & New".
- Record numbers after save: INV-xxxx | PUR-xxxx | CRN-xxxx | EST-xxxx | REC-xxxx | FA-xxxx |
  JNL-xxxx | DIV-xxxx.\
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

Task verb conventions (interpret these consistently):
- "select X"  -> CLICK to choose X from a list, menu, or dropdown. Never type into a text field.
- "set X to Y" / "set Y" -> INPUT_TEXT: type Y into the named field. Clear the field first if \
it already has a value.
- "click X"   -> a direct mouse click on a button, link, or menu item.
- "type X"    -> INPUT_TEXT: type X into the currently focused or named field.

Rules for the rewrite:
1. Be specific, never vague. Each step names the exact element (visible label, placeholder, \
or id) and the exact value to use. Preserve ALL specific names, numbers, and values the task \
gives verbatim -- do not skip or ignore them.
   TWO cases for dropdown fields:
   a) The task supplies a specific searchable value for the field — a proper name, reference \
number, account title, or option label. Type it to filter, then click the matching option. \
Examples: "select a customer Acme Ltd" (value = "Acme Ltd"), "select account Sales Revenue", \
"select vat 20% Standard", "select invoice ref INV-0011". NEVER skip the value.
   b) The task names the FIELD (not a specific record): no specific option is named — the phrase \
is just a role/type word (optionally preceded by "a", "an", "any", "the"), whether it ends there \
or is followed only by "from the dropdown/list/menu". Examples: "select director"; "select a \
user"; "select any account"; "select an asset"; "select supplier"; "select an item from the \
dropdown"; "select a service from the list". → Do NOT type and NEVER invent or type a placeholder \
value (e.g. "valid_item_name"): click the field, wait for the options, and pick the FIRST real \
option shown.
2. Any CSS-like hint (e.g. `a.ms-Link`, `input#SearchBox129`, an element id/class, or a \
react-select placeholder id) is an element ALREADY ON THE PAGE, NOT a web address. Never tell \
the agent to navigate to it as a URL. Phrase it as "click the element with id/class X" or \
"the field whose placeholder is Y".
3. Never instruct the agent to leave the application, open another site, or use a search \
engine. All actions happen inside the current app.
4. For any dropdown / combobox / react-select field, write THREE separate steps — never merge \
them:
   (a) click the field to open the list;
   (b) input_text with ONLY the search value (e.g. "Service") — NEVER include key names like \
ArrowDown or Enter in the typed text; the input action types characters only;
   (c) click the matching option that appears in the dropdown list — this is a REQUIRED step, \
always write it; phrase it as "click the option [value] from the dropdown list".
   Then verify the chosen value is now shown in the field.
   If the dropdown's options DEPEND on an earlier field (e.g. "Invoice ref" lists only the \
chosen customer's invoices), add a step to wait for the list to populate after opening it before \
typing. If the search value shows no match, clear it and pick from the options actually listed; \
for reference numbers, also try zero-padded / "INV-xxxx" forms (e.g. 011 -> 0011 -> INV-0011). \
If the task named no specific option (case 1b), skip step (b) and click the first real option.
5. Before typing into a numeric field (quantity, price), clear any existing value first.
6. Make fragile steps error-resilient: phrase them as "if <element> is not visible, scroll to \
it (or wait briefly) and retry" and "if a click does nothing, fall back to keyboard \
navigation with send_keys (Tab to move focus, ArrowDown to choose, Enter to confirm)". \
Prefer keyboard fallbacks whenever a direct click might fail.
7. After EVERY step that navigates (module switch, client select, left-nav click, tab click), \
add a VERIFY step naming the expected URL fragment from the map's URL LANDMARKS (e.g. "verify \
the URL contains /clients/<id>/inputs/sales"), plus a recovery clause: "if the URL does not \
match, you are on the wrong page — do NOT search for the target there; re-navigate (module \
picker → Bookkeeping → search and select the client) and continue". Do NOT invent verification \
the task did not ask for -- in particular, NEVER require a "success message" / toast after \
saving; this app often shows none.
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
