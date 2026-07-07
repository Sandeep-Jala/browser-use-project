"""Agent run wrapper.

`Runner` turns a task string into a structured `RunResult`, pulling everything off
`agent.history`. It owns the per-task lifecycle:

  * a fresh browser-use `BrowserSession` is attached to the already-authenticated browser
    (over CDP) for each task and cleanly stopped afterwards, so runs are isolated; and
  * a Playwright connection is opened to that same browser (also over CDP) so telemetry
    collectors can listen to console/network events with Playwright's native API. This
    connection is closed after the run without touching the underlying browser (login.py
    owns it).
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from browser_use import Agent
from browser_use.agent.service import Agent as AgentType
from playwright.async_api import BrowserContext, Playwright

from automation.collectors.base import Collector
from automation.config import Config
from automation.llm import build_llm
from automation.browser.session import attach_session

logger = logging.getLogger("framework.runner")

# A collector factory builds a fresh collector bound to one run's Playwright context +
# artifacts dir.
CollectorFactory = Callable[[BrowserContext, Path], Collector]


def _create_write_seen(requests: list[dict[str, Any]], marker: str) -> bool:
    """True if the network log contains a successful create-write for this task.

    A non-idempotent write (POST/PUT/PATCH) to a URL containing `marker` that returned 2xx/3xx
    is the ground truth that the record was actually saved — the same check the replay path uses.
    It is what distinguishes a real success from an agent that *claims* success but never fired
    the create request.
    """
    return any(
        r.get("method") in ("POST", "PUT", "PATCH")
        and marker.lower() in str(r.get("url", "")).lower()
        and 200 <= (r.get("status") or 0) < 400
        for r in requests
    )


@dataclass
class RunResult:
    """Structured outcome of a single task run, distilled from agent.history."""

    task: str
    run_id: str
    artifacts_dir: Path
    # The expanded/rewritten task actually given to the agent (None if expansion was off).
    expanded_task: str | None
    is_done: bool
    is_successful: bool | None
    has_errors: bool
    final_result: str | None
    urls: list[str | None]
    n_steps: int
    duration_seconds: float
    extracted_content: list[str]
    model_actions: list[dict[str, Any]]
    errors: list[str | None]
    # Per-collector results keyed by collector name (e.g. "network", "console").
    collector_results: dict[str, Any] = field(default_factory=dict)
    # Output files keyed by collector name.
    artifacts: dict[str, Path] = field(default_factory=dict)
    # Per-step screenshots as base64 strings (None where a step had no screenshot).
    screenshots: list[str | None] = field(default_factory=list)
    # Per-step progress timeline from the agent's own history: {n, evaluation, next_goal, url}.
    steps: list[dict[str, Any]] = field(default_factory=list)
    # browser-use's built-in end-of-run judge verdict: {verdict, reasoning,
    # failure_reason, reached_captcha}, or None if it didn't run.
    judgement: dict[str, Any] | None = None
    # LLM token usage + cost for the run (browser-use UsageSummary as a dict), or None.
    usage: dict[str, Any] | None = None
    # Network ground-truth check: {"marker": str, "create_write_seen": bool}, or None if no
    # marker was configured for this task. When present and create_write_seen is False, the
    # run's self-reported success was overridden to False (nothing was actually saved).
    ground_truth: dict[str, Any] | None = None

    def summary(self) -> str:
        status = "✅" if self.is_successful else ("⚠️" if self.is_done else "❌")
        return (
            f"{status} steps={self.n_steps} done={self.is_done} success={self.is_successful} "
            f"errors={self.has_errors} {self.duration_seconds:.1f}s\n"
            f"   final: {self.final_result}"
        )


class Runner:
    """Runs tasks on the configured LLM against an authenticated CDP browser."""

    def __init__(
        self,
        cdp_url: str,
        config: Config,
        playwright: Playwright,
        llm: Any | None = None,
        collector_factories: list[CollectorFactory] | None = None,
        expander_llm: Any | None = None,
        expand_prompt: bool = False,
        judge_llm: Any | None = None,
        extend_system_message: str | None = None,
    ) -> None:
        self.cdp_url = cdp_url
        self.config = config
        self.playwright = playwright
        # Appended to the agent's system prompt (sent every step, never trimmed).
        self.extend_system_message = extend_system_message
        # Build the LLM once and reuse it across tasks.
        self.llm = llm if llm is not None else build_llm(config)
        self.collector_factories: list[CollectorFactory] = list(collector_factories or [])
        # Prompt expansion: rewrite the task before running (only if both enabled + a model).
        self.expander_llm = expander_llm
        self.expand_prompt = expand_prompt and expander_llm is not None
        # Optional post-run QA judge (a strong LLM scoring the run); None disables verdicts.
        self.judge_llm = judge_llm

    def _new_run_dir(self) -> tuple[str, Path]:
        """Allocate a unique run id + artifacts directory."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = self.config.artifacts_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_id, run_dir

    async def run(
        self, task: str, max_steps: int = 25, record_path: Path | None = None,
        success_marker: str | None = None,
    ) -> RunResult:
        """Run a single task and return a structured result.

        A fresh session is attached for this task and stopped afterwards; the underlying
        authenticated browser (owned by login.py) stays alive throughout. Enabled
        collectors listen to the browser's pages via a Playwright connection for the
        duration of the run, each writing into this run's artifacts directory.

        If `record_path` is given, the agent's action history is saved there so the run can
        later be replayed deterministically without the LLM (see `replay`).

        If `success_marker` is given, success is cross-checked against the network log: the agent
        (and the LLM judge) can *claim* success without the record ever being saved, so a run that
        reports success but never fired a create-write to `success_marker` is downgraded to
        failure. This is the same ground-truth check the replay path uses.
        """
        run_id, run_dir = self._new_run_dir()
        logger.info("▶ run %s: %s", run_id, task)

        # Optionally rewrite the task into explicit step-by-step instructions before running.
        expanded_task: str | None = None
        agent_task = task
        if self.expand_prompt:
            from automation.pipeline.prompts import expand_task

            expanded_task = await expand_task(task, self.expander_llm)
            agent_task = expanded_task
            try:
                (run_dir / "expanded_task.txt").write_text(expanded_task, encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                logger.debug("could not save expanded_task.txt: %s", exc)

        session = await attach_session(self.cdp_url, self.config)
        # Connect CDP now (idempotent with agent.run) so the browser is live before we
        # attach Playwright telemetry listeners.
        await session.start()

        # Open a Playwright connection to the same browser for telemetry. Listen across all
        # existing contexts so we catch whichever one the agent drives.
        pw_browser = await self.playwright.chromium.connect_over_cdp(self.cdp_url)
        collectors: list[Collector] = []
        for context in pw_browser.contexts:
            for factory in self.collector_factories:
                collector = factory(context, run_dir)
                try:
                    await collector.start()
                    collectors.append(collector)
                except Exception as exc:  # noqa: BLE001 - a collector must not block the run
                    logger.warning("collector %s failed to start: %s", collector.name, exc)

        agent = Agent(
            task=agent_task,
            llm=self.llm,
            browser_session=session,
            use_vision=self.config.use_vision,
            vision_detail_level=self.config.vision_detail_level,
            max_history_items=self.config.max_history_items,
            extend_system_message=self.extend_system_message,
            # browser-use's built-in end-of-run judge (use_judge defaults True); run it on our
            # model. We read its verdict below instead of running a second judge of our own.
            judge_llm=self.judge_llm,
        )

        # Stamp each captured telemetry event with the agent step it fired on: bump every
        # collector's step counter at the start of each step.
        step_state = {"n": 0}

        async def _track_step(_agent: "AgentType") -> None:
            step_state["n"] += 1
            for collector in collectors:
                collector.current_step = step_state["n"]

        collector_results: dict[str, Any] = {}
        artifacts: dict[str, Path] = {}
        try:
            history = await agent.run(max_steps=max_steps, on_step_start=_track_step)
        finally:
            # Stop collectors, gather their results, then tear down both connections.
            for collector in collectors:
                try:
                    await collector.stop()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("collector %s stop error: %s", collector.name, exc)

            for collector in collectors:
                collector_results[collector.name] = collector.results()
                path = collector.write()
                if path is not None:
                    artifacts[collector.name] = path

            try:
                # connect_over_cdp: closes our Playwright connection only, not the browser.
                await pw_browser.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("playwright connection close error: %s", exc)

            try:
                await session.stop()
            except Exception as exc:  # noqa: BLE001
                logger.debug("session.stop() during cleanup: %s", exc)

        # Save the action trace for later deterministic replay (no LLM) if requested.
        if record_path is not None:
            try:
                record_path.parent.mkdir(parents=True, exist_ok=True)
                agent.save_history(str(record_path))
                logger.info("recorded action trace -> %s", record_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not save recording: %s", exc)

        # Per-step screenshots browser-use already captured for its own history (it grabs one
        # every step regardless of use_vision), aligned to steps. Cheaper + better-aligned
        # than taking our own.
        screenshots: list[str | None] = []
        try:
            screenshots = history.screenshots(return_none_if_not_screenshot=True)
        except Exception as exc:  # noqa: BLE001
            logger.debug("could not read screenshots from history: %s", exc)

        # Per-step progress timeline from the agent's own reasoning (no extra LLM call): each
        # step's goal + its evaluation of the prior step, so the report shows what the agent
        # was doing at every step and exactly where it ended up.
        steps: list[dict[str, Any]] = []
        try:
            thoughts = history.model_thoughts()
            urls = history.urls()
            for i, brain in enumerate(thoughts):
                steps.append(
                    {
                        "n": i + 1,
                        "evaluation": getattr(brain, "evaluation_previous_goal", None),
                        "next_goal": getattr(brain, "next_goal", None),
                        "url": urls[i] if i < len(urls) else None,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("could not build step timeline from history: %s", exc)

        # LLM token usage + cost for the run (browser-use computes this into history.usage).
        usage: dict[str, Any] | None = None
        try:
            raw_usage = history.usage
            if raw_usage is not None:
                usage = raw_usage.model_dump() if hasattr(raw_usage, "model_dump") else dict(raw_usage)
        except Exception as exc:  # noqa: BLE001
            logger.debug("could not read token usage from history: %s", exc)

        # browser-use's built-in judge runs at the end of the agent loop (use_judge) and
        # attaches its verdict to the last done action result. Read it instead of running a
        # second judge of our own.
        judgement = self._extract_judgement(history)

        result = RunResult(
            task=task,
            run_id=run_id,
            artifacts_dir=run_dir,
            expanded_task=expanded_task,
            is_done=history.is_done(),
            is_successful=history.is_successful(),
            has_errors=history.has_errors(),
            final_result=history.final_result(),
            urls=history.urls(),
            n_steps=history.number_of_steps(),
            duration_seconds=history.total_duration_seconds(),
            extracted_content=history.extracted_content(),
            model_actions=history.model_actions(),
            errors=history.errors(),
            collector_results=collector_results,
            artifacts=artifacts,
            screenshots=screenshots,
            steps=steps,
            judgement=judgement,
            usage=usage,
        )

        # Ground-truth gate: the agent + LLM judge can report success without the record ever
        # being saved. If a marker is configured and no matching create-write hit the network,
        # override the self-reported success to failure so the report reflects reality.
        if success_marker:
            requests = collector_results.get("network", {}).get("requests", []) or []
            create_write_seen = _create_write_seen(requests, success_marker)
            overridden = bool(not create_write_seen and result.is_successful)
            result.ground_truth = {
                "marker": success_marker,
                "create_write_seen": create_write_seen,
                "overrode_success": overridden,
            }
            if overridden:
                logger.info(
                    "◀ ground-truth override %s: reported success but no create-write to '%s' "
                    "seen in network → marking failed", run_id, success_marker,
                )
                result.is_successful = False
                result.has_errors = True

        logger.info("◀ done %s: %s", run_id, result.summary())
        return result

    async def run_script(self, steps_path: Path, success_marker: str = "Invoices") -> RunResult:
        """Run a compiled selector script (no LLM, no browser-use Agent) over Playwright.

        Fast deterministic replay: executes the compiled steps back-to-back with Playwright
        auto-wait. Success is judged from the network log (a write to `success_marker`
        returned 2xx) — the same ground-truth check as `replay`.
        """
        import json as _json

        from automation.pipeline.script_compile import run_steps

        run_id, run_dir = self._new_run_dir()
        logger.info("▶ SCRIPT %s from %s", run_id, steps_path)
        steps = _json.loads(Path(steps_path).read_text())

        pw_browser = await self.playwright.chromium.connect_over_cdp(self.cdp_url)
        page = next((ctx.pages[0] for ctx in pw_browser.contexts if ctx.pages), None)
        collectors: list[Collector] = []
        for context in pw_browser.contexts:
            for factory in self.collector_factories:
                collector = factory(context, run_dir)
                try:
                    await collector.start()
                    collectors.append(collector)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("collector %s failed to start: %s", collector.name, exc)

        started = datetime.now()
        outcome: dict[str, Any] = {"executed": 0, "failed_at": None, "error": None}
        collector_results: dict[str, Any] = {}
        artifacts: dict[str, Path] = {}
        try:
            if page is None:
                raise RuntimeError("no open page to run the script against")
            outcome = await run_steps(page, steps)
        finally:
            for collector in collectors:
                try:
                    await collector.stop()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("collector %s stop error: %s", collector.name, exc)
            for collector in collectors:
                collector_results[collector.name] = collector.results()
                path = collector.write()
                if path is not None:
                    artifacts[collector.name] = path
            try:
                await pw_browser.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("playwright close error: %s", exc)

        requests = collector_results.get("network", {}).get("requests", []) or []
        saved = _create_write_seen(requests, success_marker)
        duration = (datetime.now() - started).total_seconds()
        final = f"Script ran {outcome['executed']}/{len(steps)} steps in {duration:.1f}s"
        if outcome["failed_at"] is not None:
            final += f" — FAILED at step {outcome['failed_at']}: {outcome['error']}"
        final += f"; create-write seen in network: {saved}"
        result = RunResult(
            task=f"(script) {Path(steps_path).name}",
            run_id=run_id, artifacts_dir=run_dir, expanded_task=None,
            is_done=True, is_successful=saved, has_errors=not saved, final_result=final,
            urls=[], n_steps=outcome["executed"], duration_seconds=duration,
            extracted_content=[], model_actions=[],
            errors=[outcome["error"]] if outcome["error"] else [],
            collector_results=collector_results, artifacts=artifacts,
            screenshots=[], steps=[], judgement=None, usage=None,
            ground_truth={"marker": success_marker, "create_write_seen": saved},
        )
        logger.info("◀ SCRIPT done %s: success=%s executed=%s/%s %.1fs",
                    run_id, saved, outcome["executed"], len(steps), duration)
        return result

    @staticmethod
    def _extract_judgement(history: Any) -> dict[str, Any] | None:
        """Pull browser-use's built-in judge verdict off the last done action result."""
        try:
            for item in reversed(history.history):
                for res in reversed(getattr(item, "result", None) or []):
                    j = getattr(res, "judgement", None)
                    if j is not None:
                        return {
                            "verdict": getattr(j, "verdict", None),
                            "reasoning": getattr(j, "reasoning", None),
                            "failure_reason": getattr(j, "failure_reason", None),
                            "reached_captcha": getattr(j, "reached_captcha", None),
                        }
        except Exception as exc:  # noqa: BLE001 - reading the judge verdict must not crash the run
            logger.debug("could not read built-in judge verdict: %s", exc)
        return None
