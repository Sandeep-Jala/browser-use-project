"""Entry point: authenticate, then run each task through the Runner.

login.py launches Chromium (CDP open) and logs in; the Runner attaches browser-use to that
same browser per task. In AUTO mode it replays the task's recorded selector script (fast, no
LLM) if one exists, otherwise it runs the agent and records one. Telemetry (network/console)
and an HTML report are produced per run.

Env knobs: TASK=invoice|purchase, AUTO=1 (replay-or-author), FRESH=1 (force re-author).
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

# Terse, high-level task prompts (the app-aware expander turns these into concrete steps).
INVOICE_TASK = (
    "go to bookkeeping module, click on any client, select inputs, click on Sales, "
    "add some details about a sales, and save it"
)
PURCHASE_TASK = (
    "go to bookkeeping module, click on any client, select inputs, click on purchases, "
    "click add purchases."
)
TASK = {"purchase": PURCHASE_TASK, "invoice": INVOICE_TASK}.get(
    os.getenv("TASK", "invoice").strip().lower(), INVOICE_TASK
)


def _kill_stale_browser(port: int) -> None:
    """Best-effort: kill any leftover Chromium still holding the CDP debug port from a
    previously-killed run, so this run attaches to a fresh browser."""
    try:
        subprocess.run(["pkill", "-f", f"remote-debugging-port={port}"],
                       check=False, capture_output=True, timeout=5)
    except Exception:  # noqa: BLE001 - guard is best-effort
        pass


async def run_task(runner: Runner, task: str, auto: bool, fresh: bool):
    """Replay the task's compiled script if one exists (AUTO), else run/record the agent."""
    if not auto:
        return await runner.run(task, max_steps=30)

    tid = ts.task_id(task)
    script_path = ts.steps_path(tid)
    if script_path.exists() and not fresh:
        print(f"[*] task {tid}: script found -> fast run (no LLM)")
        return await runner.run_script(script_path)

    print(f"[*] task {tid}: authoring with the agent (recording it)")
    result = await runner.run(task, max_steps=30, record_path=ts.recording_path(tid))
    n = len(save_steps(ts.recording_path(tid), script_path))
    ts.update_manifest(tid, task, steps=n)
    print(f"[*] task {tid}: recorded + compiled {n} steps")
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

            result = await run_task(runner, TASK, auto, fresh)
            paths = build_report(result)
            result.artifacts["report_html"] = paths["html"]

            print("\n========== RESULT ==========")
            print(result.summary())
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
