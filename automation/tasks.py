"""Declarative task registry: one TaskSpec per task (prompt + ground-truth marker + tags).

Adding a task is ONE entry in `TASKS`. The `marker` is the network ground-truth URL fragment
(a successful POST/PUT/PATCH to a URL containing it proves the record saved); `None` means
read-only (the gate is disabled). `assertions` holds per-task overrides for the assertion
engine (see pipeline/assertions.py); `tags` group tasks for suite selection (`tag:sales`).

CRITICAL: prompts are identity. task_store.task_id hashes the prompt to find a task's golden
script, so editing a prompt's text (even whitespace is normalized, but words are not) orphans
its recording. tests/test_tasks.py pins every prompt's task id for exactly this reason.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SubtaskDecl:
    """One declared subtask of a task (hybrid subtask engine, see pipeline/hybrid.py).

    `prompt` may carry {{tokens}} whose concrete values live in `values` — the tokenized
    prompt is the subtask's LIBRARY identity, so two tasks that differ only in values share
    one library recording. `marker` marks the save-owning subtask (the parent's create-write
    fires here); `postcondition` is an optional cheap success check for subtasks with no
    write: {"url_contains": "..."} or {"visible": "<selector>"}.
    """
    prompt: str
    values: dict[str, str] | None = None
    marker: str | None = None
    postcondition: dict[str, Any] | None = None


@dataclass(frozen=True)
class TaskSpec:
    key: str
    prompt: str
    marker: str | None = None          # ground-truth URL fragment; None = read-only task
    assertions: dict[str, Any] | None = None  # per-task assertion overrides; None = defaults
    tags: tuple[str, ...] = ()
    # Explicit subtask decomposition for the hybrid engine; None = LLM decomposition
    # (computed once per prompt and cached under decompositions/<tid>.json).
    subtasks: tuple[SubtaskDecl, ...] | None = None


# Terse, high-level task prompts (the app-aware expander turns these into concrete steps).
# Markers are best guesses based on the app's REST conventions — verify against real network
# captures if a task fails the ground-truth gate unexpectedly.
TASKS: dict[str, TaskSpec] = {t.key: t for t in (
    # ------------------------------------------------------------------ Sales
    TaskSpec(
        key="invoice",
        prompt="""go to Bookkeeping module, search and select 290 CREW LIMITED business name. go to inputs section,select sales,go to Invoices,add invoice,select a customer Suresh Gopi,select an item bike, set product description 'buying a new bike', set Qty 5, Unit price 500 and click on save""",
        marker="Invoices",
        tags=("sales",),
        # Reference explicit decomposition for the hybrid subtask engine. The tokenized
        # prompts are shared library identities: every task starting with the same
        # business-selection / navigation subtasks reuses the SAME recordings.
        subtasks=(
            SubtaskDecl(
                prompt="go to Bookkeeping module, search and select {{business}} business name",
                values={"business": "290 CREW LIMITED"},
            ),
            SubtaskDecl(prompt="go to inputs section, select sales, go to Invoices"),
            SubtaskDecl(
                prompt="add invoice: select a customer {{customer}}, select an item "
                       "{{item}}, set product description '{{product_description}}', "
                       "set Qty {{qty}}, Unit price {{unit_price}} and click on save",
                values={"customer": "Suresh Gopi", "item": "bike",
                        "product_description": "buying a new bike",
                        "qty": "5", "unit_price": "500"},
                marker="Invoices",
            ),
        ),
    ),
    TaskSpec(
        key="credit_notes",
        prompt="""go to Bookkeeping module, search and select 290 CREW LIMITED business name. Go to inputs section,select sales,go to Credit Notes,add credit note,select any customer,select any invoice ref from the dropdown, and click on save.""",
        marker="Refunds",  # sales credit notes are committed via the /Refunds endpoint
        tags=("sales",),
    ),
    TaskSpec(
        key="estimates",
        prompt="""go to Bookkeeping module, search and select 290 CREW LIMITED business name.go to inputs section,select sales,go to Estimates,add estimate,select customer Mr Jones, select an item from the dropdown and click on save.""",
        marker="Invoices",  # estimates are persisted via the /Invoices endpoint
        tags=("sales",),
    ),
    # NOTE: run 20260707_143005 saved a receipt yet only POST /Payments fired — if receipts
    # show up in the app despite FAIL verdicts here, the marker is wrong: change to "Payments".
    TaskSpec(
        key="receipt",
        prompt="""go to Bookkeeping module, search and select 290 CREW LIMITED business name.go to inputs section,select sales,go to Receipts,add receipt,enter Suresh Gopi in receipts from filed,enter amount 1000,save receipt.""",
        marker="Receipts",
        tags=("sales", "flaky-ui"),  # UI issue while saving
    ),
    TaskSpec(
        key="item",
        prompt="""go to Bookkeeping module, search and select 290 CREW LIMITED business name.go to inputs section,select sales,go to Item,add item,enter name Office Expences, set purchases description to 'buying a new item', set sales description to 'selling a new item',enter unit price purchase 100,enter unit price sell 150,create item.""",
        marker="Items",
        tags=("sales",),
    ),
    # -------------------------------------------------------------- Purchases
    TaskSpec(
        key="purchase",
        prompt="""go to Bookkeeping module, search and select 290 CREW LIMITED business name.go to inputs section,select purchases,add invoice,select customer Le Marche,select an item car from dropdown, set Qty 5, Unit price 500,set vat to No VAT and click on save.""",
        marker="Purchase",
        tags=("purchases",),
    ),
    TaskSpec(
        key="purchase_credit_notes",
        prompt="""go to Bookkeeping module, search and select 290 CREW LIMITED business name.go to inputs section,select purchases,go to Credit Notes,add credit note,select a supplier Lina,set invoice ref PUR-0071 and click on save.""",
        marker="Purchase",
        tags=("purchases",),
    ),
    TaskSpec(
        key="purchase_po",
        prompt="""go to Bookkeeping module, search and select 290 CREW LIMITED business name.go to inputs section,select purchases,go to Purchase Orders,add purchase order,select contact name John,select an item furniture and click on save.""",
        marker="PurchaseOrders",
        tags=("purchases",),
    ),
    TaskSpec(
        key="purchase_payment",
        prompt="""go to Bookkeeping module, search and select 290 CREW LIMITED business name.go to inputs section,select purchases,go to Payments Section,add payment,enter Gabriel Dobson in Paid to field,enter amount 500,save payment.""",
        marker="Payment",
        tags=("purchases", "flaky-ui"),  # UI issue while saving
    ),
    # --------------------------------------------------------- Expense Claims
    TaskSpec(
        key="reimbursements",
        prompt="""go to Bookkeeping module, search and select 290 CREW LIMITED business name.go to inputs section,select expense claims,go to Reimbursements Section,click add reimbursement,select an user name in 'Reimbursed To' field, select an account, enter amount 200,and click on save.""",
        marker="Reimbursements",
        tags=("expense_claims",),
    ),
    # Verified from run 20260708_161610: the commit is POST .../MileageClaims ("Mileages"
    # is NOT a substring of it and falsely failed a saved record).
    TaskSpec(
        key="mileage",
        prompt="""go to Bookkeeping module, search and select 290 CREW LIMITED business name.go to inputs section,select expense claims,go to Mileages Section,click add mileage,select a director from user dropdown,enter 'Mileages Business Trip' in Remarks field,select engine type Petrol,enter description mileage London to Manchester,enter mileage 200,select rate 45p,and click on save.""",
        marker="MileageClaims",
        tags=("expense_claims",),
    ),
    TaskSpec(
        key="expense_claims",
        prompt="""go to Bookkeeping module, search and select 290 CREW LIMITED business name.go to inputs section,select expense claims,click add expense button,select director,enter Office Equipment in remarks field,enter bill no EXP123,enter description Buying New Desks,select account Eu services - 3/4,enter base amount 1500,select vat 5% standard,and click on save.""",
        marker="ExpenseClaims",
        tags=("expense_claims",),
    ),
    TaskSpec(
        key="refund",
        prompt="""go to Bookkeeping module, search and select 290 CREW LIMITED business name.go to inputs section,select expense claims,go to Refunds Section,click add refund,select a value in refund from field,select an account,Enter amount 1000 and click on save.""",
        marker="Refunds",
        tags=("expense_claims", "flaky-ui"),  # UI issue while saving
    ),
    # ------------------------- Journals / Assets / Banking / Budget / Dividends
    TaskSpec(
        key="journals",
        prompt="""go to Bookkeeping module, search and select 290 CREW LIMITED business name.go to inputs section,select journals,click add journal button,enter JRN001 in journal reference field,select an account,enter value in debit 1000,and click on save.""",
        marker="Journals",
        tags=("journals", "flaky-ui"),  # UI issue while saving
    ),
    TaskSpec(
        key="fixed_asset",
        prompt="""go to Bookkeeping module, search and select 290 CREW LIMITED business name.go to inputs section,select assets,click add Fixed assets,enter asset name MacBook Pro,select an account,set purchase price 100,select supplier AO, enter rate 1200 and click on save.""",
        marker="Assets",
        tags=("assets",),
    ),
    TaskSpec(
        key="disposed_asset",
        prompt="""go to Bookkeeping module, search and select 290 CREW LIMITED business name.go to inputs section,select assets,go to disposed,add dispose asset,select an asset,enter sales proceeds 800,select payment method Customer,select customer Suresh Raina and click on save.""",
        marker="Assets",
        tags=("assets",),
    ),
    TaskSpec(
        key="banking",
        prompt="""go to Bookkeeping module, search and select 290 CREW LIMITED business name.go to banking section,click add account,account type savings, select bank CAF, enter account no 126525678, enter sort code 77-26-89,enter IBAN RB003GSD, Make it as Primary account.and click on save.""",
        marker="Banking",  # commits via POST /Banking/ — known issue with IBAN field
        tags=("banking",),
    ),
    TaskSpec(
        key="budget_manager",
        prompt="""go to Bookkeeping module, search and select 290 CREW LIMITED business name. Go to Budget manager and click add budget.Enter name Q4 Marketing,date 5th Dec,2027.select Frequency yearly,duration 1 year. and click on save.""",
        marker="Budget",
        tags=("budget",),
    ),
    TaskSpec(
        key="dividend",
        prompt="""go to Bookkeeping module, search and select 290 CREW LIMITED business name.go to inputs section,select dividends section,click dividends,select an authorised director from dropdown,select a type,enter dividend per share 10,enter payment date 10/10/2026,save asset,save.""",
        marker="Dividends",
        tags=("dividends",),
    ),
    # ------------------ Full end-to-end sample with new contact and item creation
    TaskSpec(
        key="invoice_full_creation",
        prompt="""go to Bookkeeping module.search for MAK NOTTINGHAM LTD business name and select it.go to inputs section,select sales,go to Invoices,add invoice,create a new Contact name 'DamBro' in customer field. set address jodhpur, rajasthan, 342015. set supplier address barmer, rajasthan, india, 325486. Supplier Set Due Date 01/01/2027. Set invoice no INV123425. set p.o. reference abdf435, create an item 'IT Services',set product description 'service provider', set qty 10 and unit price 5000, set vat to No Vat, set discount 10% and click on save.""",
        marker="Invoices",
        tags=("sales", "e2e"),
    ),
    # -------------------------------------------------------------------- CRM
    TaskSpec(
        key="crm_create_invoice",
        prompt="""Go to clients section.Search for Zachary Spencer and select it. Go to "create invoice", select a service. set amount to 5000 and set discount to 10%. create a discount note.Select a collection method and click on save.""",
        marker="Invoices",
        tags=("crm",),
    ),
)}


# Verbs that mean the task WRITES something (create/modify/remove a record). A free-text task
# containing none of these is read-only — it will legitimately produce no create-write, so it
# must not get a marker (the ground-truth gate would force-fail an honest success otherwise).
_WRITE_VERBS = re.compile(
    r"\b(add|create|save|submit|enter|set|make|new|record|update|edit|modify|change|delete|"
    r"remove|upload|import|approve|pay|dispose|generate)\b", re.IGNORECASE)


def _infer_marker(prompt: str) -> str | None:
    """Best-guess success marker for a free-text task, or None for a read-only task.

    None disables the network ground-truth gate: a task that only reads/verifies (check a
    balance, confirm a record exists) fires no create-write, so success falls back to the
    agent's self-report + the judge. Write tasks map to a marker by record type (ordered most
    specific first — e.g. invoice tasks mention items, so "invoice" is checked before "item").
    """
    p = prompt.lower()
    if not _WRITE_VERBS.search(p):
        return None
    checks = [
        ("purchase order", "PurchaseOrders"),
        ("credit note", "Purchase" if "purchase" in p else "Refunds"),
        ("reimbursement", "Reimbursements"),
        ("mileage", "MileageClaims"),
        ("refund", "Refunds"),
        ("expense", "ExpenseClaims"),
        ("journal", "Journals"),
        ("asset", "Assets"),
        ("bank", "Banking"),
        ("budget", "Budget"),
        ("dividend", "Dividends"),
        ("receipt", "Receipts"),
        ("payment", "Payment"),
        ("estimate", "Invoices"),
        ("invoice", "Invoices"),
        ("item", "Items"),
        ("purchase", "Purchase"),
    ]
    for keyword, marker in checks:
        if keyword in p:
            return marker
    return "Invoices"


def resolve_task(raw: str) -> TaskSpec:
    """Resolve --task input to a TaskSpec: a known key, or a free-text prompt (must contain a
    space so a typo'd key is not silently run as a one-word prompt). Raises ValueError for an
    unknown key so the CLI can print the known keys."""
    key = raw.strip().lower()
    if key in TASKS:
        return TASKS[key]
    if " " in raw:
        prompt = raw.strip()
        return TaskSpec(key="adhoc", prompt=prompt, marker=_infer_marker(prompt))
    raise ValueError(
        f"Unknown TASK key {raw!r}. Known keys: {', '.join(sorted(TASKS))}.\n"
        'Or pass a full prompt: --task "go to Bookkeeping module, ..."'
    )


def select_tasks(selector: str) -> list[TaskSpec]:
    """Expand a suite selector into TaskSpecs: 'all', 'tag:<tag>', or a comma-list of keys.
    Raises ValueError for an unknown key/tag so the CLI can fail before logging in."""
    sel = selector.strip().lower()
    if sel == "all":
        return list(TASKS.values())
    if sel.startswith("tag:"):
        tag = sel[len("tag:"):].strip()
        specs = [t for t in TASKS.values() if tag in t.tags]
        if not specs:
            known = sorted({tag for t in TASKS.values() for tag in t.tags})
            raise ValueError(f"No tasks tagged {tag!r}. Known tags: {', '.join(known)}.")
        return specs
    specs = []
    for part in sel.split(","):
        key = part.strip()
        if not key:
            continue
        if key not in TASKS:
            raise ValueError(f"Unknown task key {key!r}. Known keys: {', '.join(sorted(TASKS))}.")
        specs.append(TASKS[key])
    if not specs:
        raise ValueError("Empty suite selector.")
    return specs
