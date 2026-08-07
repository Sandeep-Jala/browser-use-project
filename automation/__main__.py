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
from functools import partial

import psutil

from playwright.async_api import async_playwright

from automation.browser.login import login
from automation.browser.session import launch_session
from automation.pipeline import assertions as asserts
from automation.pipeline import files as pfiles
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
               redecompose: bool = False, reauthor: str | None = None,
               log_all_hosts: bool = False, record: bool = False) -> bool:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = Config.from_env()
    if record:  # the flag turns recording ON; RECORD_VIDEO=true in .env is the standing way
        config.record_video = True
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
              "success comes from the segment gates")

    # Files the prompt names ("upload X.csv ...") must exist NON-EMPTY in
    # automation/uploads/ BEFORE login: the resolved absolute paths become the agent's
    # upload_file allowlist. A bad path is worse than a failed run — observed live: a
    # nonexistent relative path "uploaded" fine, then Save crashed the tab
    # (RESULT_CODE_KILLED_BAD_MESSAGE) when the page tried to read it.
    upload_files, file_problems = pfiles.resolve_prompt_files(task)
    if file_problems:
        raise SystemExit("[!] file reference(s) could not be resolved:\n    "
                         + "\n    ".join(file_problems))
    for f in upload_files:
        print(f"[*] file for upload: {f} ({os.path.getsize(f):,} bytes)")

    # Assertions judge the app under test only: scope defaults to the login URL's
    # host, so helper-tab sites and third-party trackers can't fail a healthy run.
    # A task's own `assertions.scope` (hosts list, or None = everything) overrides.
    # Resolved BEFORE the Runner because the LOG CAPTURE uses the same scope: the
    # collectors record only events produced by in-scope pages ("the app's logs"),
    # and one shared derivation guarantees a captured-out entry can never be one an
    # assertion rule needed. --log-all-hosts records everything (debugging a helper
    # site itself); the assertions stay scoped either way.
    defaults = dict(asserts.DEFAULT_SPEC)
    scope_hosts = asserts.scope_hosts_for(config.login_url)
    if scope_hosts:
        defaults["scope"] = {"hosts": scope_hosts}
    assert_spec = asserts.merge_spec(defaults, spec.assertions)
    log_scope = None if log_all_hosts else (assert_spec.get("scope") or {}).get("hosts")

    async with async_playwright() as playwright:
        # INVERTED handoff: browser-use launches and OWNS the browser (the session then
        # classifies LOCAL — validated uploads, native download handling); login connects
        # to it over CDP to authenticate. See browser/session.py.
        session = launch_session(config, config.artifacts_dir / ".downloads_staging")
        await session.start()
        cdp_url = session.cdp_url
        if not cdp_url:
            raise SystemExit("[!] the agent browser exposed no CDP endpoint")
        await login(playwright, config, cdp_url)
        print(f"[*] Login complete (CDP {cdp_url}) | model: {config.active_model} "
              f"vision={config.use_vision}")

        # Always built: the subtask decomposer, the end-of-run judge, and adapt.parameterize
        # all need this LLM (it is not optional the way the old prompt-expander was).
        expander_llm = build_expander_llm(config)
        try:
            runner = Runner(
                cdp_url, config, playwright,
                session=session,
                collector_factories=[
                    partial(NetworkCollector, scope_hosts=log_scope),
                    partial(ConsoleCollector, scope_hosts=log_scope),
                ],
                expander_llm=expander_llm,
                extend_system_message=SPEED_OPTIMIZATION_PROMPT,
                tools=build_tools(),  # custom actions the prompts call (escape hatches, UI scans)
                available_files=upload_files,
            )

            result = await run_task(runner, task, fresh, success_marker, spec=spec,
                                    redecompose=redecompose, reauthor=reauthor)
            checks = asserts.apply(result, assert_spec)
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
            downloads = [d for s in (result.subtasks or [])
                         for d in (s.get("downloads") or [])]
            if downloads:
                print(f"   downloads ({len(downloads)}) in "
                      f"{result.artifacts_dir}/downloads/:")
                for d in downloads:
                    print(f"     ⬇ {d}")
            gt = result.ground_truth or {}
            if gt:
                print(f"   ground truth: create-write to '{gt.get('marker')}' seen in network: "
                      f"{gt.get('create_write_seen')}"
                      + ("  (self-reported success OVERRIDDEN → FAIL)"
                         if gt.get("overrode_success") else ""))
            if result.usage:
                print(f"   tokens: {result.usage.get('total_tokens')}  "
                      f"cost=${result.usage.get('total_cost', 0):.4f}")
            print(f"   report: {paths['html']}\n")
            # Combined verdict for CI: the flow completed AND the telemetry was healthy.
            return bool(result.is_successful) and result.assertions_passed is not False
        finally:
            # The session owns the browser now (keep_alive only spans segments, not the
            # process): kill closes Chromium itself.
            try:
                await session.kill()
            except Exception as exc:  # noqa: BLE001 - teardown must never mask the verdict
                log.debug("session kill at exit: %s", exc)


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
    parser.add_argument("--log-all-hosts", action="store_true",
                        help="Record console/network telemetry from EVERY page. By default "
                             "the logs are scoped to the app under test (same scope as the "
                             "assertions), so helper-tab sites and their ad stacks stay out "
                             "of the artifacts; use this when debugging a helper site "
                             "itself.")
    parser.add_argument("--record", action="store_true",
                        help="Save an .mp4 of the run to artifacts/<run_id>/run.mp4. The "
                             "capture is a TIME-LAPSE of the page viewport: the browser "
                             "emits a frame only when the page changes, so the waits "
                             "between LLM steps collapse and a long run becomes a short "
                             "clip. Set RECORD_VIDEO_SIZE=1280x800 to cut the cost.")
    args = parser.parse_args()

    ok = asyncio.run(main(args.task, args.fresh, args.marker,
                          redecompose=args.redecompose, reauthor=args.reauthor,
                          log_all_hosts=args.log_all_hosts, record=args.record))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    cli()
