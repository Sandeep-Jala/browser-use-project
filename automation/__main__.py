"""Entry point: authenticate, then run each task through the hybrid subtask engine.

login.py launches Chromium (CDP open) and logs in; the Runner attaches browser-use to that
same browser per task. Telemetry (network/console) and an HTML report are produced per run.

Every task runs subtask-by-subtask (pipeline/hybrid.py): each subtask the shared library
already knows replays from its recording with no LLM, and only the gaps are authored by the
agent — then committed to the library so the next run replays them too. --fresh re-authors
every subtask; --reauthor re-authors only the ones you name.

Task selection: --task <key from tasks.yaml> (or a full free-text prompt), also settable via
the TASK env var. Tasks are defined declaratively in tasks.yaml at the repo root.
"""
from __future__ import annotations

import asyncio
import logging
import os

import psutil

from playwright.async_api import async_playwright

from automation.browser.login import login
from automation.pipeline import assertions as asserts
from automation.collectors.console import ConsoleCollector
from automation.collectors.network import NetworkCollector
from automation.config import Config
from automation.llm import build_expander_llm
from automation.pipeline.agent_tools import build_tools
from automation.pipeline.hybrid import run_hybrid_task
from automation.pipeline.prompts import SPEED_OPTIMIZATION_PROMPT
from automation.pipeline.report import build_report
from automation.pipeline.runner import Runner
from automation.tasks import resolve_task

log = logging.getLogger("framework.main")


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


async def run_task(runner: Runner, task: str, fresh: bool, marker: str | None, spec=None,
                   redecompose: bool = False, reauthor: str | None = None):
    """Execute `task` through the hybrid subtask engine — the only execution path.

    Every run goes subtask-by-subtask (pipeline/hybrid.py): a subtask the library knows
    replays with no LLM, a miss is authored by the agent and committed to the library. A
    failed replay hands the same live page to the agent for in-place recovery.

    `fresh` re-authors (and re-records) EVERY subtask, ignoring the library. `reauthor` names
    subtasks — indexes and/or prompt substrings — to re-author while the rest still replay.
    """
    return await run_hybrid_task(runner, task, spec=spec, marker=marker, fresh=fresh,
                                 redecompose=redecompose, reauthor=reauthor)


async def main(task_raw: str, fresh: bool, success_marker: str | None = None,
               redecompose: bool = False, reauthor: str | None = None) -> bool:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = Config.from_env()
    config.ensure_dirs()
    _kill_stale_browser(config.cdp_port)

    try:
        spec = resolve_task(task_raw)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    task = spec.prompt

    # Resolve the ground-truth marker. --marker none/off disables the gate explicitly;
    # --marker <fragment> enables it for a free-text prompt; otherwise the task's
    # tasks.yaml marker applies. A task WITHOUT a marker (new/verification/read-only
    # flows — free-text prompts never get one) runs with the gate disabled: it may
    # legitimately fire no create-write, and the gate would force-fail an honest success.
    if success_marker and success_marker.strip().lower() in ("none", "off"):
        success_marker = None
    elif not success_marker:
        success_marker = spec.marker
    if success_marker is None:
        print("[*] no ground-truth marker (or --marker none): network gate disabled; "
              "success comes from the segment gates + judge")

    async with async_playwright() as playwright:
        browser, _page, cdp_url = await login(playwright, config)
        print(f"[*] Login complete (CDP {cdp_url}) | model: {config.active_model} "
              f"vision={config.use_vision}")

        # Always built: the subtask decomposer, the end-of-run judge, and adapt.parameterize
        # all need this LLM (it is not optional the way the old prompt-expander was).
        expander_llm = build_expander_llm(config)
        try:
            runner = Runner(
                cdp_url, config, playwright,
                collector_factories=[NetworkCollector, ConsoleCollector],
                expander_llm=expander_llm,
                judge_llm=expander_llm,  # feeds browser-use's built-in end-of-run judge
                extend_system_message=SPEED_OPTIMIZATION_PROMPT,
                tools=build_tools(),  # custom actions the prompts call (escape hatches, UI scans)
            )

            result = await run_task(runner, task, fresh, success_marker, spec=spec,
                                    redecompose=redecompose, reauthor=reauthor)
            checks = asserts.apply(result, asserts.merge_spec(asserts.DEFAULT_SPEC,
                                                              spec.assertions))
            paths = build_report(result)
            result.artifacts["report_html"] = paths["html"]

            print("\n========== RESULT ==========")
            print(result.summary())
            if result.mode:
                print(f"   mode: {result.mode}")
            if result.subtasks:
                print("   subtasks:")
                for s in result.subtasks:
                    status = "ok" if s.get("ok") else "FAIL"
                    print(f"     {s['index']:>2}. {status:<5} {s.get('mode') or '—':<26} "
                          f"{s.get('steps_executed', 0):>3} steps "
                          f"{s.get('duration_seconds', 0):>6}s  "
                          f"{(s.get('prompt') or '')[:60]}")
            if checks:
                failed = [c for c in checks if c.passed is False]
                print(f"   assertions: {'FAIL' if failed else 'pass'} "
                      f"({sum(1 for c in checks if c.passed is True)} passed, "
                      f"{len(failed)} failed, "
                      f"{sum(1 for c in checks if c.passed is None)} skipped)")
                for c in failed:
                    print(f"     ✗ {c.name}: {c.detail}")
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
            # Combined verdict for CI: the flow completed AND the telemetry was healthy.
            return bool(result.is_successful) and result.assertions_passed is not False
        finally:
            await browser.close()


def cli() -> None:
    """Synchronous console-script entry point (see [project.scripts] in pyproject.toml)."""
    import argparse
    parser = argparse.ArgumentParser(description="Run the automation framework tasks.")
    parser.add_argument("--task", default=os.getenv("TASK", "invoice"),
                        help="Task key from tasks.yaml or a free-form prompt.")
    parser.add_argument("--fresh", action="store_true",
                        help="Re-author AND re-record EVERY subtask, ignoring the library. "
                             "The passing ones are committed back, so the next run replays "
                             "them.")
    parser.add_argument("--marker", default=os.getenv("SUCCESS_MARKER", "").strip() or None,
                        help="Success marker URL fragment; pass 'none' to disable the "
                             "network ground-truth gate. Tasks without a marker in "
                             "tasks.yaml (and free-text prompts) already run with the "
                             "gate disabled.")
    parser.add_argument("--redecompose", action="store_true",
                        help="Regenerate the task's cached subtask decomposition "
                             "(the cache is otherwise immutable per prompt).")
    parser.add_argument("--reauthor", default=None,
                        help="Re-record specific subtasks with the LLM even though their "
                             "library replay works — a comma-list of subtask indexes and/or "
                             "prompt substrings (e.g. --reauthor 0 or --reauthor 'add "
                             "estimate'). Other subtasks still replay; the entry is replaced "
                             "only if the new recording passes its gate.")
    args = parser.parse_args()

    ok = asyncio.run(main(args.task, args.fresh, args.marker,
                          redecompose=args.redecompose, reauthor=args.reauthor))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    cli()
