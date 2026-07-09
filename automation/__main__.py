"""Entry point: authenticate, then run each task through the Runner.

login.py launches Chromium (CDP open) and logs in; the Runner attaches browser-use to that
same browser per task. Telemetry (network/console) and an HTML report are produced per run.

Replay-or-author is the default: a recorded task replays its selector script (fast, no LLM);
a new one is authored by the agent and recorded. Pass --no-auto to force a plain agent run
that ignores recordings, or --fresh to re-author a known task.

Task selection: --task <key from ALL_TASKS> (or a full free-text prompt), also settable via
the TASK env var.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess

import psutil

from playwright.async_api import async_playwright

from automation.browser.login import login
from automation.pipeline import adapt
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
    """go to Bookkeeping module, search and select 290 CREW LIMITED business name. go to inputs section, select sales, go to Invoices, add invoice, select a customer Suresh Gopi, select an item bike, set product description 'buying a new cycle', set Qty 3, Unit price 500 and click on save"""
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



# Per-task network ground-truth marker: a successful create-write to a URL containing this
# substring is what proves the record was actually saved. These URL fragments are best guesses
# based on the app's REST conventions — verify against real network captures if a task fails
# the ground-truth gate unexpectedly.
SUCCESS_MARKERS = {
    "invoice":               "Invoices",
    "credit_notes":          "Refunds",   # sales credit notes are committed via the /Refunds endpoint
    "estimates":             "Invoices",   # estimates are persisted via the /Invoices endpoint
    # NOTE: run 20260707_143005 saved a receipt yet only POST /Payments fired — if receipts
    # show up in the app despite FAIL verdicts here, the marker is wrong: change to "Payments".
    "receipt":               "Receipts",
    "item":                  "Items",
    "purchase":              "Purchase",
    "purchase_credit_notes": "Purchase",
    "purchase_po":           "PurchaseOrders",
    "purchase_payment":      "Payment",
    "reimbursements":        "Reimbursements",
    # Verified from run 20260708_161610: the commit is POST .../MileageClaims ("Mileages"
    # is NOT a substring of it and falsely failed a saved record).
    "mileage":               "MileageClaims",
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
def _infer_marker(prompt: str) -> str:
    """Best-guess success marker for a free-text task, from its record type (ordered most
    specific first — e.g. invoice tasks mention items, so "invoice" is checked before "item")."""
    p = prompt.lower()
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





def _kill_stale_browser(port: int) -> None:
    """Best-effort: kill any leftover Chromium still holding the CDP debug port from a
    previously-killed run, so this run attaches to a fresh browser."""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline') or []
                if any(f"remote-debugging-port={port}" in arg for arg in cmdline):
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except Exception:  # noqa: BLE001 - guard is best-effort
        pass


async def _save_template(tid: str, task: str, steps: list, llm) -> None:
    """Best-effort: parameterize a just-committed golden script into <tid>.template.json
    (values bound to named params, e.g. customer/qty/unit_price) so future prompts that only
    change values can reuse it via _try_adaptation. Never blocks the run on failure."""
    try:
        template = await adapt.parameterize(task, steps, llm)
        if not template:
            return
        adapt.save_template(ts.template_path(tid), template)
        ts.update_manifest(tid, task, params=template["params"])
        print(f"[*] task {tid}: template saved — params: "
              f"{json.dumps(template['params'])}")
    except Exception as exc:  # noqa: BLE001 - a template is a bonus, not a requirement
        log.warning("could not save template for %s: %s", tid, exc)


async def _try_adaptation(runner: Runner, task: str, tid: str, script_path, marker: str):
    """Template tier: match `task` against recorded templates, fill each template parameter
    with the value read from the new prompt, and replay the instantiated script. Returns the
    RunResult if the replay passed the ground-truth gate (script + inherited template
    committed under `tid`), else None so the caller falls back to agent authoring."""
    if runner.expander_llm is None:
        return None
    candidates = [
        {"id": t, "prompt": entry["prompt"], "params": entry["params"]}
        for t, entry in ts.load_manifest().items()
        if t != tid and entry.get("prompt") and entry.get("params")
        and ts.template_path(t).exists()
    ]
    if not candidates:
        return None

    print(f"[*] task {tid}: no exact script — matching against "
          f"{len(candidates)} recorded template(s)...")
    match = adapt.match_template(task, candidates)
    if match is None:
        print(f"[*] task {tid}: no template match -> authoring")
        return None

    tmp_path = script_path.with_suffix(".tmp.json")
    try:
        template = adapt.load_template(ts.template_path(match.source_tid))
        new_steps = adapt.instantiate(template, match.values)
        if new_steps is None:
            print(f"[*] task {tid}: match left template params unresolved -> authoring")
            return None
        changed = {k: v for k, v in match.values.items()
                   if template["params"].get(k) != v}
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(new_steps, indent=2))
        print(f"[*] task {tid}: instantiated template {match.source_tid} with "
              f"{json.dumps(changed) if changed else 'unchanged values'} -> validating replay...")
        result = await runner.run_script(tmp_path, success_marker=marker)
        if result.is_successful:
            os.replace(tmp_path, script_path)
            # The new task inherits the template: same tokenized steps, its own defaults.
            new_params = {**template["params"],
                          **{k: v for k, v in match.values.items() if k in template["params"]}}
            adapt.save_template(ts.template_path(tid), {
                "source_prompt": task, "params": new_params, "steps": template["steps"],
            })
            ts.update_manifest(tid, task, steps=len(new_steps),
                               adapted_from=match.source_tid, params=new_params)
            print(f"[*] task {tid}: adapted replay PASSED -> committed "
                  f"{len(new_steps)}-step golden script + template")
            return result
        print(f"[*] task {tid}: adapted replay FAILED ({result.final_result}) "
              f"-> falling back to authoring")
        tmp_path.unlink(missing_ok=True)
        return None
    except Exception as exc:  # noqa: BLE001 - adaptation must never block the authoring path
        log.exception("adaptation error for %s: %s", tid, exc)
        print(f"[*] task {tid}: adaptation error: {exc} -> falling back to authoring")
        tmp_path.unlink(missing_ok=True)
        return None


async def run_task(runner: Runner, task: str, auto: bool, fresh: bool, marker: str):
    """Replay the task's compiled script if one exists (AUTO), else author + validate + commit."""
    if not auto:
        return await runner.run(task, max_steps=60, success_marker=marker)

    tid = ts.task_id(task)
    script_path = ts.steps_path(tid)
    if script_path.exists() and not fresh:
        print(f"[*] task {tid}: script found -> fast run (no LLM)")
        return await runner.run_script(script_path, success_marker=marker)

    # No exact script: before paying for a full agent authoring run, try adapting a recorded
    # task that is the same procedure with different values (one cheap LLM call + a replay).
    # FRESH skips this tier too — it means "re-author, period".
    if not fresh:
        adapted = await _try_adaptation(runner, task, tid, script_path, marker)
        if adapted is not None:
            return adapted

    print(f"[*] task {tid}: authoring with the agent (recording it)")
    # 60 steps: with max_actions_per_step=1 every fill/click is its own step, so a full create
    # task legitimately needs ~35-40 steps; 60 leaves room to recover from a few missteps.
    result = await runner.run(
        task, max_steps=60, record_path=ts.recording_path(tid), success_marker=marker
    )
    gt = result.ground_truth or {}
    if not result.is_successful and not gt.get("create_write_seen"):
        print(f"[*] task {tid}: run did not succeed (no create-write) -> NOT recording a script")
        return result

    # Rescue path: the agent reported failure, but the network says the record WAS saved
    # (e.g. it flailed on a follow-up form after an unnoticed successful save). Compile the
    # recording anyway, truncated at the step the create-write fired on, so the post-save
    # flailing never reaches the script. Replay validation below still decides the commit.
    truncate_at = None
    if not result.is_successful:
        truncate_at = gt.get("write_step")
        print(f"[*] task {tid}: agent reported failure but the create-write DID fire "
              f"(step {truncate_at}) -> compiling anyway, truncated at that step")

    # Author run succeeded (ground-truth). Compile to a temp path and replay-verify it before
    # committing as the golden script. This ensures we never save a fragile script — we only
    # commit a script that has JUST proven it can replay perfectly end-to-end.
    tmp_path = script_path.with_suffix(".tmp.json")
    try:
        steps = save_steps(ts.recording_path(tid), tmp_path, max_steps=truncate_at)
        n = len(steps)
        print(f"[*] task {tid}: compiled {n} steps — validating replay...")
        val = await runner.run_script(tmp_path, success_marker=marker)
        if val.is_successful:
            os.replace(tmp_path, script_path)
            ts.update_manifest(tid, task, steps=n)
            print(f"[*] task {tid}: validation PASSED -> committed {n}-step golden script")
            await _save_template(tid, task, steps, runner.expander_llm)
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


async def main(task_raw: str, auto: bool, fresh: bool, success_marker: str | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = Config.from_env()
    config.ensure_dirs()
    _kill_stale_browser(config.cdp_port)

    task_key = task_raw.strip().lower()
    if task_key in ALL_TASKS:
        task = ALL_TASKS[task_key]
    elif " " in task_raw:
        task = task_raw.strip()
    else:
        raise SystemExit(
            f"Unknown TASK key {task_raw!r}. Known keys: {', '.join(sorted(ALL_TASKS))}.\n"
            'Or pass a full prompt: --task "go to Bookkeeping module, ..."'
        )
        
    if not success_marker:
        success_marker = SUCCESS_MARKERS.get(task_key, "Invoices") if task_key in ALL_TASKS else _infer_marker(task)

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

            result = await run_task(runner, task, auto, fresh, success_marker)
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
    import argparse
    parser = argparse.ArgumentParser(description="Run the automation framework tasks.")
    parser.add_argument("--task", default=os.getenv("TASK", "invoice"), help="Task key or free-form prompt.")
    # Replay-first by default: reuse a recorded script when one exists. Use --no-auto to
    # force a plain agent run that ignores recordings.
    parser.add_argument("--auto", action=argparse.BooleanOptionalAction, default=True, help="Replay a recorded script when one exists (default: on; use --no-auto to force a fresh agent run).")
    parser.add_argument("--fresh", action="store_true", help="Force re-authoring, ignoring existing scripts.")
    parser.add_argument("--marker", default=os.getenv("SUCCESS_MARKER", "").strip() or None, help="Success marker URL fragment.")
    args = parser.parse_args()
    
    asyncio.run(main(args.task, args.auto, args.fresh, args.marker))


if __name__ == "__main__":
    cli()
