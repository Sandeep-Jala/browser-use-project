"""Entry point: authenticate, then run each task through the Runner.

login.py launches Chromium (CDP open) and logs in; the Runner attaches browser-use to that
same browser per task. Telemetry (network/console) and an HTML report are produced per run.

Replay-or-author is the default: a recorded task replays its selector script (fast, no LLM);
a new one is authored by the agent and recorded. Pass --no-auto to force a plain agent run
that ignores recordings, or --fresh to re-author a known task.

Task selection: --task <key from automation.tasks.TASKS> (or a full free-text prompt), also
settable via the TASK env var.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime

import psutil

from playwright.async_api import async_playwright

from automation.browser.login import login
from automation.pipeline import adapt
from automation.pipeline import assertions as asserts
from automation.collectors.console import ConsoleCollector
from automation.collectors.network import NetworkCollector
from automation.config import Config
from automation.llm import build_expander_llm
from automation.pipeline import task_store as ts
from automation.pipeline.agent_tools import build_tools
from automation.pipeline.prompts import SPEED_OPTIMIZATION_PROMPT
from automation.pipeline.report import build_report
from automation.pipeline.runner import Runner
from automation.pipeline.script_compile import promote_healed, save_steps
from automation.pipeline.suite import run_suite
from automation.tasks import resolve_task, select_tasks

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
            # Persist heals into the just-committed script. The TEMPLATE is deliberately not
            # rewritten: templates tokenize values, not selectors, and _try_adaptation
            # replay-validates every instantiation anyway.
            _promote_heals(tid, task, script_path, result)
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


async def _reset_app_state(runner: Runner) -> None:
    """Best-effort reset between runs on the same browser: navigate the driven page back to
    the app origin (an SPA reload clears stuck modals/flyouts a failed replay can leave
    behind) and close extra tabs so run_script's "first non-blank page" pick stays
    deterministic. Never raises — a reset failure just means the next run starts dirtier."""
    from urllib.parse import urlsplit
    try:
        pw_browser = await runner.playwright.chromium.connect_over_cdp(runner.cdp_url)
        try:
            pages = [p for ctx in pw_browser.contexts for p in ctx.pages]
            real_pages = [p for p in pages if p.url != "about:blank"]
            keep = real_pages[0] if real_pages else (pages[0] if pages else None)
            if keep is None:
                return
            for p in pages:
                if p is not keep:
                    await p.close()
            parts = urlsplit(runner.config.login_url)
            await keep.goto(f"{parts.scheme}://{parts.netloc}",
                            wait_until="domcontentloaded", timeout=15000)
            await keep.wait_for_timeout(1000)
        finally:
            await pw_browser.close()  # detach CDP; the login-owned browser stays alive
    except Exception as exc:  # noqa: BLE001 - reset is best-effort by contract
        log.warning("app-state reset failed: %s", exc)


def _promote_heals(tid: str, task: str, script_path, result) -> None:
    """Persist any fingerprint healings a SUCCESSFUL replay used into the golden script, so
    the next replay resolves directly instead of re-healing (or eventually failing). Only
    ever called on a passed run — a failed replay must never rewrite its script."""
    log_entries = (result.replay or {}).get("log") or []
    if not any(e.get("healed") for e in log_entries):
        return
    try:
        promoted = promote_healed(script_path, log_entries)
        if promoted:
            ts.update_manifest(tid, task,
                               healed=datetime.now().isoformat(timespec="seconds"),
                               healed_steps=promoted)
            print(f"[*] task {tid}: promoted healed selectors into steps {promoted}")
    except Exception as exc:  # noqa: BLE001 - promotion is a bonus; the run already passed
        log.warning("heal promotion failed for %s: %s", tid, exc)


async def run_task(runner: Runner, task: str, auto: bool, fresh: bool, marker: str,
                   fallback: bool = True, spec=None, use_subtasks: bool = False,
                   redecompose: bool = False):
    """Replay the task's compiled script if one exists (AUTO), else author + validate + commit.

    With `fallback` (default), a broken golden-script replay archives the script and
    re-authors the task with the agent instead of failing the run — the app changed, so the
    recording is stale by definition. `--no-fallback` keeps the failure as the result.

    With `use_subtasks`, the two expensive paths — authoring a new task and repairing a
    broken replay — go through the hybrid subtask engine (pipeline/hybrid.py) instead of a
    whole-task agent run: recorded subtasks replay from the shared library, the LLM fills
    only the gaps, and a passing run stitches back into a whole-task golden script. The
    whole-task replay fast path is unchanged either way.
    """
    if not auto:
        result = await runner.run(task, max_steps=90, success_marker=marker)
        result.mode = "agent"
        return result

    tid = ts.task_id(task)
    script_path = ts.steps_path(tid)
    if script_path.exists() and not fresh:
        print(f"[*] task {tid}: script found -> fast run (no LLM)")
        result = await runner.run_script(script_path, success_marker=marker)
        result.mode = "replay"
        if result.is_successful:
            _promote_heals(tid, task, script_path, result)
            return result
        if not fallback:
            return result
        # Broken replay: keep its report as evidence, retire the stale script, and re-author
        # from scratch. From scratch (not resume-from-failed-step): the page is mid-flow with
        # a half-filled form, and only a whole run can pass the ground-truth gate honestly.
        build_report(result)
        print(f"[*] task {tid}: replay FAILED ({result.final_result}) -> archiving script "
              f"and re-authoring with the "
              f"{'hybrid subtask engine' if use_subtasks else 'agent'}")
        archived = ts.archive_script(tid)
        for path in archived:
            print(f"[*] task {tid}: archived {path}")
        ts.update_manifest(
            tid, task,
            reauthored=datetime.now().isoformat(timespec="seconds"),
            reauthor_count=ts.load_manifest().get(tid, {}).get("reauthor_count", 0) + 1,
        )
        await _reset_app_state(runner)
        if use_subtasks:
            from automation.pipeline.hybrid import run_hybrid_task

            result = await run_hybrid_task(runner, task, spec=spec, marker=marker,
                                           redecompose=redecompose)
            result.mode = "replay_failed->hybrid"
        else:
            result = await _author_and_commit(runner, task, tid, marker)
            result.mode = "replay_failed->authored"
        return result

    # No exact script: before paying for a full agent authoring run, try adapting a recorded
    # task that is the same procedure with different values (one cheap LLM call + a replay).
    # FRESH skips this tier too — it means "re-author, period". (No fallback wrapper here:
    # a failed adaptation already falls through to authoring inside _try_adaptation's caller.)
    if not fresh:
        adapted = await _try_adaptation(runner, task, tid, script_path, marker)
        if adapted is not None:
            adapted.mode = "adapted"
            return adapted

    if use_subtasks:
        from automation.pipeline.hybrid import run_hybrid_task

        return await run_hybrid_task(runner, task, spec=spec, marker=marker,
                                     fresh=fresh, redecompose=redecompose)
    return await _author_and_commit(runner, task, tid, marker)


async def _author_and_commit(runner: Runner, task: str, tid: str, marker: str):
    """Author the task with the agent (recording it), then compile + commit the golden
    script and its template if the run passed the ground-truth gate."""
    print(f"[*] task {tid}: authoring with the agent (recording it)")
    # 90 steps: with max_actions_per_step=1 every fill/click is its own step, so a full create
    # task legitimately needs ~35-40 steps; 90 leaves room to recover from missteps (run
    # 20260710_094628 hit the old 60-step ceiling mid-form). The ground-truth gate still
    # decides success, so a longer leash can't fake a pass.
    result = await runner.run(
        task, max_steps=90, record_path=ts.recording_path(tid), success_marker=marker
    )
    result.mode = "authored"
    gt = result.ground_truth or {}
    if not result.is_successful and not gt.get("create_write_seen"):
        print(f"[*] task {tid}: run did not succeed (no create-write) -> NOT recording a script")
        return result

    # Rescue path: the agent reported failure, but the network says the record WAS saved
    # (e.g. it flailed on a follow-up form after an unnoticed successful save). Compile the
    # recording anyway, truncated at the step the create-write fired on, so the post-save
    # flailing never reaches the script.
    truncate_at = None
    if not result.is_successful:
        truncate_at = gt.get("write_step")
        print(f"[*] task {tid}: agent reported failure but the create-write DID fire "
              f"(step {truncate_at}) -> compiling anyway, truncated at that step")

    # Author run passed the network ground-truth gate: compile and commit the golden script
    # directly. (No proof-replay: the workflow is author once with the LLM, then replay with
    # different values/names via the template tier — that first value-swapped replay is the
    # real test, and a broken script simply fails there and can be re-authored with --fresh.)
    try:
        steps = save_steps(ts.recording_path(tid), script_path, max_steps=truncate_at)
        n = len(steps)
        ts.update_manifest(tid, task, steps=n)
        print(f"[*] task {tid}: compiled {n} steps -> committed golden script")
        await _save_template(tid, task, steps, runner.expander_llm)
    except Exception as exc:  # noqa: BLE001 - compile failure must not crash the session
        log.exception("compile error for %s: %s", tid, exc)
        print(f"[*] task {tid}: compile error: {exc} -> golden script NOT saved")
    return result


async def main(task_raw: str, auto: bool, fresh: bool, success_marker: str | None = None,
               fallback: bool = True, use_subtasks: bool | None = None,
               redecompose: bool = False) -> bool:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = Config.from_env()
    config.ensure_dirs()
    _kill_stale_browser(config.cdp_port)
    if use_subtasks is None:
        use_subtasks = config.use_subtasks

    try:
        spec = resolve_task(task_raw)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    task = spec.prompt

    # Resolve the ground-truth marker. --marker none/off disables the gate explicitly;
    # otherwise the task's registry marker applies (resolve_task already inferred one for
    # free-text prompts — including None for READ-ONLY tasks with no write verbs, which
    # produce no create-write and must not be force-failed by the gate).
    if success_marker and success_marker.strip().lower() in ("none", "off"):
        success_marker = None
    elif not success_marker:
        success_marker = spec.marker
    if success_marker is None:
        print("[*] read-only task (or --marker none): network ground-truth gate disabled; "
              "success comes from the agent + judge")

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
                extend_system_message=SPEED_OPTIMIZATION_PROMPT,
                tools=build_tools(),  # custom actions the prompts call (escape hatches, UI scans)
            )

            result = await run_task(runner, task, auto, fresh, success_marker,
                                    fallback=fallback, spec=spec,
                                    use_subtasks=use_subtasks, redecompose=redecompose)
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


async def main_suite(selector: str, auto: bool, fresh: bool, fallback: bool,
                     continue_on_failure: bool, use_subtasks: bool | None = None,
                     redecompose: bool = False) -> bool:
    """Run a set of tasks on ONE login/browser and write a suite-level report.
    Returns the CI verdict: every task PASS with assertions passing."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = Config.from_env()
    config.ensure_dirs()
    _kill_stale_browser(config.cdp_port)
    if use_subtasks is None:
        use_subtasks = config.use_subtasks

    try:
        specs = select_tasks(selector)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if fresh and selector.strip().lower() == "all":
        raise SystemExit("--fresh with --suite all would re-author EVERY task (a long, "
                         "token-heavy run). Name the tasks explicitly: "
                         "--suite invoice,purchase --fresh")

    async with async_playwright() as playwright:
        browser, _page, cdp_url = await login(playwright, config)
        print(f"[*] Login complete (CDP {cdp_url}) | model: {config.active_model} "
              f"| suite: {len(specs)} task(s)")
        expander_llm = build_expander_llm(config) if config.expand_prompt else None
        try:
            runner = Runner(
                cdp_url, config, playwright,
                collector_factories=[NetworkCollector, ConsoleCollector],
                expander_llm=expander_llm,
                expand_prompt=config.expand_prompt,
                judge_llm=expander_llm,
                extend_system_message=SPEED_OPTIMIZATION_PROMPT,
                tools=build_tools(),
            )

            async def run_one(spec):
                return await run_task(runner, spec.prompt, auto, fresh, spec.marker,
                                      fallback=fallback, spec=spec,
                                      use_subtasks=use_subtasks, redecompose=redecompose)

            async def reset():
                await _reset_app_state(runner)

            async def browser_alive() -> bool:
                try:
                    probe = await runner.playwright.chromium.connect_over_cdp(runner.cdp_url)
                    await probe.close()
                    return True
                except Exception:  # noqa: BLE001 - any failure means the browser is gone
                    return False

            summary = await run_suite(
                specs, run_one, selector=selector, artifacts_dir=config.artifacts_dir,
                continue_on_failure=continue_on_failure, reset=reset,
                browser_alive=browser_alive,
            )
        finally:
            await browser.close()

    totals = summary["totals"]
    print("\n========== SUITE RESULT ==========")
    for t in summary["tasks"]:
        status = t["status"]
        if status == "PASS" and t.get("assertions_ok") is False:
            status = "PASS*"
        line = f"  {t['key']:<24} {status:<8} {t.get('mode') or '—':<24} {t['duration_seconds']}s"
        if t.get("error"):
            line += f"  {str(t['error'])[:80]}"
        print(line)
    print(f"  {'-' * 60}")
    print(f"  {totals['pass']} pass / {totals['fail']} fail / {totals['error']} error / "
          f"{totals['done']} done / {totals['skipped']} skipped"
          + (f" / {totals['assertion_failures']} with failed assertions"
             if totals.get("assertion_failures") else ""))
    print(f"  suite report: {summary['suite_html']}\n")
    return bool(summary["ok"])


def cli() -> None:
    """Synchronous console-script entry point (see [project.scripts] in pyproject.toml)."""
    import argparse
    parser = argparse.ArgumentParser(description="Run the automation framework tasks.")
    parser.add_argument("--task", default=os.getenv("TASK", "invoice"), help="Task key or free-form prompt.")
    parser.add_argument("--suite", default=os.getenv("SUITE", "").strip() or None,
                        help="Run a task set instead of --task: 'all', 'tag:<tag>', or a "
                             "comma-list of keys. Writes a suite report and exits non-zero "
                             "unless every task passes.")
    parser.add_argument("--continue-on-failure", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Suite mode: keep running after a failed task (default: on).")
    # Replay-first by default: reuse a recorded script when one exists. Use --no-auto to
    # force a plain agent run that ignores recordings.
    parser.add_argument("--auto", action=argparse.BooleanOptionalAction, default=True, help="Replay a recorded script when one exists (default: on; use --no-auto to force a fresh agent run).")
    parser.add_argument("--fresh", action="store_true", help="Force re-authoring, ignoring existing scripts.")
    parser.add_argument("--fallback", action=argparse.BooleanOptionalAction, default=True,
                        help="On a broken golden-script replay, archive the script and "
                             "re-author with the agent (default: on).")
    parser.add_argument("--marker", default=os.getenv("SUCCESS_MARKER", "").strip() or None,
                        help="Success marker URL fragment; pass 'none' to disable the "
                             "network ground-truth gate (read-only tasks are auto-detected).")
    parser.add_argument("--subtasks", action=argparse.BooleanOptionalAction, default=None,
                        help="Author/repair through the hybrid subtask engine: replay "
                             "recorded subtasks from the shared library, LLM only for the "
                             "gaps (default: SUBTASKS env var, else on).")
    parser.add_argument("--redecompose", action="store_true",
                        help="Regenerate the task's cached subtask decomposition "
                             "(the cache is otherwise immutable per prompt).")
    args = parser.parse_args()

    if args.suite:
        ok = asyncio.run(main_suite(args.suite, args.auto, args.fresh, args.fallback,
                                    args.continue_on_failure, use_subtasks=args.subtasks,
                                    redecompose=args.redecompose))
    else:
        ok = asyncio.run(main(args.task, args.auto, args.fresh, args.marker,
                              fallback=args.fallback, use_subtasks=args.subtasks,
                              redecompose=args.redecompose))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    cli()
