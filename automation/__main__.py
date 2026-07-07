"""Entry point: authenticate, then run each task through the Runner.

login.py launches Chromium (CDP open) and logs in; the Runner attaches browser-use to that
same browser per task. In AUTO mode it replays the task's recorded selector script (fast, no
LLM) if one exists, otherwise it runs the agent and records one. Telemetry (network/console)
and an HTML report are produced per run.

Env knobs: TASK=<key from ALL_TASKS>, AUTO=1 (replay-or-author), FRESH=1 (force re-author).
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess

from playwright.async_api import async_playwright

from automation.browser.login import login
from automation.collectors.console import ConsoleCollector
from automation.collectors.network import NetworkCollector
from automation.config import Config
from automation.llm import build_expander_llm
from automation.pipeline import task_store as ts
from automation.pipeline.prompts import APP_SYSTEM_RULES
from automation.pipeline.report import build_report
from automation.pipeline.runner import Runner
from automation.pipeline.script_compile import save_steps

log = logging.getLogger("framework.main")

# Terse, high-level task prompts (the app-aware expander turns these into concrete steps).
INVOICE_TASK = (
    """go to Bookkeeping module, search and select 290 CREW LIMITED business name. go to inputs section, select sales, go to Invoices, add invoice, select a customer Suresh Gopi, select an item bike, set product description 'buying a new bike', set Qty 5, Unit price 500 and click on save"""
)
 
CREDIT_NOTES_TASK = (
    """go to Bookkeeping module, search and select 290 CREW LIMITED business name. go to inputs section, select sales, go to Credit Notes, add credit note, select a customer Suresh Gopi, select a invoice ref 011, and click on save"""
)
 
ESTIMATES_TASK = (
    """go to Bookkeeping module, search and select 290 CREW LIMITED business name. go to inputs section, select sales, go to Estimates, add estimate, select customer Mr Jones, select an item from the dropdown and click on save"""
)
 
# UI issue while saving
RECEIPT_TASK = (
    """go to Bookkeeping module, search and select 290 CREW LIMITED business name. go to inputs section, select sales, go to Receipts, add receipt, enter Suresh Gopi in receipts from field, enter amount 1000, save receipt"""
)
 
ITEM_TASK = (
    """go to Bookkeeping module, search and select 290 CREW LIMITED business name. go to inputs section, select sales, go to Item, add item, enter name Office Expences, set purchases description to 'buying a new item', set sales description to 'selling a new item', enter unit price purchase 100, enter unit price sell 150, create item"""
)
 
# ---------------------------------------------------------------------------
# Purchases
# ---------------------------------------------------------------------------
 
PURCHASE_TASK = (
    """go to Bookkeeping module, search and select 290 CREW LIMITED business name. go to inputs section, select purchases, add invoice, select customer Le Marche, select an item car from dropdown, set Qty 5, Unit price 500, set vat to No VAT and click on save"""
)
 
PURCHASE_CREDIT_NOTES_TASK = (
    """go to Bookkeeping module, search and select 290 CREW LIMITED business name. go to inputs section, select purchases, go to Credit Notes, add credit note, select a supplier Lina, set invoice ref PUR-0071 and click on save"""
)
 
PURCHASE_PO_TASK = (
    """go to Bookkeeping module, search and select 290 CREW LIMITED business name. go to inputs section, select purchases, go to Purchase Orders, add purchase order, select contact name John, select an item furniture and click on save"""
)
 
# UI issue while saving
PURCHASE_PAYMENT_TASK = (
    """go to Bookkeeping module, search and select 290 CREW LIMITED business name. go to inputs section, select purchases, go to Payments Section, add payment, enter Gabriel Dobson in Paid to field, enter amount 500, save payment"""
)
 
# ---------------------------------------------------------------------------
# Expense Claims
# ---------------------------------------------------------------------------
 
REIMBURSEMENTS_TASK = (
    """go to Bookkeeping module, search and select 290 CREW LIMITED business name. go to inputs section, select expense claims, go to Reimbursements Section, click add reimbursement, select an user name in 'Reimbursed To' field, select an account, enter amount 200, and click on save"""
)
 
MILEAGE_TASK = (
    """go to Bookkeeping module, search and select 290 CREW LIMITED business name. go to inputs section, select expense claims, go to Mileages Section, click add mileage, select a director from user dropdown, enter 'Mileages Business Trip' in Remarks field, select engine type Petrol, enter description mileage London to Manchester, enter mileage 200, select rate 45p, and click on save"""
)
 
EXPENSE_CLAIMS_TASK = (
    """go to Bookkeeping module, search and select 290 CREW LIMITED business name. go to inputs section, select expense claims, click add expense button, select director, enter Office Equipment in remarks field, enter bill no EXP123, enter description Buying New Desks, select account Eu services - 3/4, enter base amount 1500, select vat 5% standard, and click on save"""
)
 
# UI issue while saving
REFUND_TASK = (
    """go to Bookkeeping module, search and select 290 CREW LIMITED business name. go to inputs section, select expense claims, go to Refunds Section, click add refund, select a value in refund from field, select an account, enter amount 1000 and click on save"""
)
 
# ---------------------------------------------------------------------------
# Journals / Assets / Banking / Budget / Dividends
# ---------------------------------------------------------------------------
 
# UI issue while saving
JOURNALS_TASK = (
    """go to Bookkeeping module, search and select 290 CREW LIMITED business name. go to inputs section, select journals, click add journal button, enter JRN001 in journal reference field, select an account, enter value in debit 1000, and click on save"""
)
 
FIXED_ASSET_TASK = (
    """go to Bookkeeping module, search and select 290 CREW LIMITED business name. go to inputs section, select assets, click add Fixed assets, enter asset name MacBook Pro, select an account, set purchase price 100, select supplier AO, enter rate 1200 and click on save"""
)
 
DISPOSED_ASSET_TASK = (
    """go to Bookkeeping module, search and select 290 CREW LIMITED business name. go to inputs section, select assets, go to disposed, add dispose asset, select an asset, enter sales proceeds 800, select payment method Customer, select customer Suresh Raina and click on save"""
)
 
# Issue with IBAN
BANKING_TASK = (
    """go to Bookkeeping module, search and select 290 CREW LIMITED business name. go to banking section, click add account, account type savings, select bank CAF, enter account no 126525678, enter sort code 77-26-89, enter IBAN RB003GSD, make it as Primary account, and click on save"""
)
 
BUDGET_MANAGER_TASK = (
    """go to Bookkeeping module, search and select 290 CREW LIMITED business name. go to Budget manager and click add budget, enter name Q4 Marketing, date 5th Dec, 2027, select Frequency yearly, duration 1 year, and click on save"""
)
 
DIVIDEND_TASK = (
    """go to Bookkeeping module, search and select 290 CREW LIMITED business name. go to inputs section, select dividends section, click dividends, select an authorised director from dropdown, select a type, enter dividend per share 10, enter payment date 10/10/2026, save asset, save"""
)
 
# ---------------------------------------------------------------------------
# Full end-to-end sample with creation of new contact and item
# ---------------------------------------------------------------------------
 
INVOICE_FULL_CREATION_TASK = (
    """go to Bookkeeping module, search for 290 CREW LIMITED business name and select it. go to inputs section, select sales, go to Invoices, add invoice, create a new Contact name 'dpscvs' in customer field, set address jodhpur, rajasthan, 342015, set supplier address barmer, rajasthan, india, 325486, set Supplier Due Date 01/01/2027, set invoice no INV12345, set p.o. reference abc123, create an item 'IT Services', set product description 'service provider', set qty 10 and unit price 5000, set vat to No Vat, set discount 10% and click on save"""
)
 
# ---------------------------------------------------------------------------
# CRM
# ---------------------------------------------------------------------------
 
CRM_CREATE_INVOICE_TASK = (
    """go to clients section, search for Zachary Spencer and select it. go to create invoice, select a service, set amount to 5000 and set discount to 10%, create a discount note, select a collection method and click on save"""
)
 
# Convenience mapping for parametrised test runners
ALL_TASKS = {
    "invoice": INVOICE_TASK,
    "credit_notes": CREDIT_NOTES_TASK,
    "estimates": ESTIMATES_TASK,
    "receipt": RECEIPT_TASK,
    "item": ITEM_TASK,
    "purchase": PURCHASE_TASK,
    "purchase_credit_notes": PURCHASE_CREDIT_NOTES_TASK,
    "purchase_po": PURCHASE_PO_TASK,
    "purchase_payment": PURCHASE_PAYMENT_TASK,
    "reimbursements": REIMBURSEMENTS_TASK,
    "mileage": MILEAGE_TASK,
    "expense_claims": EXPENSE_CLAIMS_TASK,
    "refund": REFUND_TASK,
    "journals": JOURNALS_TASK,
    "fixed_asset": FIXED_ASSET_TASK,
    "disposed_asset": DISPOSED_ASSET_TASK,
    "banking": BANKING_TASK,
    "budget_manager": BUDGET_MANAGER_TASK,
    "dividend": DIVIDEND_TASK,
    "invoice_full_creation": INVOICE_FULL_CREATION_TASK,
    "crm_create_invoice": CRM_CREATE_INVOICE_TASK,
}

_TASK_KEY = os.getenv("TASK", "invoice").strip().lower()
TASK = ALL_TASKS.get(_TASK_KEY, INVOICE_TASK)

# Per-task network ground-truth marker: a successful create-write to a URL containing this
# substring is what proves the record was actually saved. These URL fragments are best guesses
# based on the app's REST conventions — verify against real network captures if a task fails
# the ground-truth gate unexpectedly.
SUCCESS_MARKERS = {
    "invoice":               "Invoices",
    "credit_notes":          "Refunds",   # sales credit notes are committed via the /Refunds endpoint
    "estimates":             "Invoices",   # estimates are persisted via the /Invoices endpoint
    "receipt":               "Receipts",
    "item":                  "Items",
    "purchase":              "Purchase",
    "purchase_credit_notes": "Purchase",
    "purchase_po":           "PurchaseOrders",
    "purchase_payment":      "Payment",
    "reimbursements":        "Reimbursements",
    "mileage":               "Mileages",
    "expense_claims":        "ExpenseClaims",
    "refund":                "Refunds",
    "journals":              "Journals",
    "fixed_asset":           "Assets",
    "disposed_asset":        "Assets",
    "banking":               "Banking",   # commits via POST /Banking/
    "budget_manager":        "Budget",
    "dividend":              "Dividends",
    "invoice_full_creation": "Invoices",
    "crm_create_invoice":    "Invoices",
}
SUCCESS_MARKER = SUCCESS_MARKERS.get(_TASK_KEY, "Invoices")


def _kill_stale_browser(port: int) -> None:
    """Best-effort: kill any leftover Chromium still holding the CDP debug port from a
    previously-killed run, so this run attaches to a fresh browser."""
    try:
        subprocess.run(["pkill", "-f", f"remote-debugging-port={port}"],
                       check=False, capture_output=True, timeout=5)
    except Exception:  # noqa: BLE001 - guard is best-effort
        pass


async def run_task(runner: Runner, task: str, auto: bool, fresh: bool, marker: str):
    """Replay the task's compiled script if one exists (AUTO), else author + validate + commit."""
    if not auto:
        return await runner.run(task, max_steps=30, success_marker=marker)

    tid = ts.task_id(task)
    script_path = ts.steps_path(tid)
    if script_path.exists() and not fresh:
        print(f"[*] task {tid}: script found -> fast run (no LLM)")
        return await runner.run_script(script_path, success_marker=marker)

    print(f"[*] task {tid}: authoring with the agent (recording it)")
    result = await runner.run(
        task, max_steps=30, record_path=ts.recording_path(tid), success_marker=marker
    )
    if not result.is_successful:
        print(f"[*] task {tid}: run did not succeed (no create-write) -> NOT recording a script")
        return result

    # Author run succeeded (ground-truth). Compile to a temp path and replay-verify it before
    # committing as the golden script. This ensures we never save a fragile script — we only
    # commit a script that has JUST proven it can replay perfectly end-to-end.
    tmp_path = script_path.with_suffix(".tmp.json")
    try:
        n = len(save_steps(ts.recording_path(tid), tmp_path))
        print(f"[*] task {tid}: compiled {n} steps — validating replay...")
        val = await runner.run_script(tmp_path, success_marker=marker)
        if val.is_successful:
            os.replace(tmp_path, script_path)
            ts.update_manifest(tid, task, steps=n)
            print(f"[*] task {tid}: validation PASSED -> committed {n}-step golden script")
        else:
            # Validation failed: leave any existing golden script untouched.
            log.warning("validation FAILED for %s: %s", tid, val.final_result)
            print(f"[*] task {tid}: validation FAILED ({val.final_result})")
            print(f"[*] task {tid}: NOT overwriting golden script; raw trace kept for debugging")
            tmp_path.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001 - validation error must not crash the session
        log.exception("validation error for %s: %s", tid, exc)
        print(f"[*] task {tid}: validation error: {exc} -> NOT overwriting golden script")
        tmp_path.unlink(missing_ok=True)
    return result


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = Config.from_env()
    config.ensure_dirs()
    _kill_stale_browser(config.cdp_port)

    auto = os.getenv("AUTO", "").strip().lower() in {"1", "true", "yes", "on"}
    fresh = os.getenv("FRESH", "").strip().lower() in {"1", "true", "yes", "on"}

    async with async_playwright() as playwright:
        browser, _page, cdp_url = await login(playwright, config)
        print(f"[*] Login complete (CDP {cdp_url}) | model: {config.active_model} "
              f"vision={config.use_vision}")

        expander_llm = build_expander_llm(config) if config.expand_prompt else None
        try:
            runner = Runner(
                cdp_url, config, playwright,
                collector_factories=[NetworkCollector, ConsoleCollector],
                expander_llm=expander_llm,
                expand_prompt=config.expand_prompt,
                judge_llm=expander_llm,  # feeds browser-use's built-in end-of-run judge
                extend_system_message=APP_SYSTEM_RULES,
            )

            result = await run_task(runner, TASK, auto, fresh, SUCCESS_MARKER)
            paths = build_report(result)
            result.artifacts["report_html"] = paths["html"]

            print("\n========== RESULT ==========")
            print(result.summary())
            gt = result.ground_truth or {}
            if gt:
                print(f"   ground truth: create-write to '{gt.get('marker')}' seen in network: "
                      f"{gt.get('create_write_seen')}"
                      + ("  (self-reported success OVERRIDDEN → FAIL)"
                         if gt.get("overrode_success") else ""))
            j = result.judgement or {}
            if j:
                verdict = {True: "PASS", False: "FAIL"}.get(j.get("verdict"), "N/A")
                print(f"   judge: {verdict}{(' — ' + j['failure_reason']) if j.get('failure_reason') else ''}")
            if result.usage:
                print(f"   tokens: {result.usage.get('total_tokens')}  "
                      f"cost=${result.usage.get('total_cost', 0):.4f}")
            print(f"   report: {paths['html']}\n")
        finally:
            await browser.close()


def cli() -> None:
    """Synchronous console-script entry point (see [project.scripts] in pyproject.toml)."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()
