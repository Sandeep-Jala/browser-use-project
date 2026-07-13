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
import re
import signal
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from browser_use import Agent
from browser_use.agent.service import Agent as AgentType
from browser_use.llm.messages import UserMessage
from playwright.async_api import BrowserContext, Playwright

from automation.collectors.base import Collector
from automation.config import Config
from automation.llm import build_llm
from automation.browser.session import attach_session
from automation.pipeline import agent_tools

logger = logging.getLogger("framework.runner")

# A collector factory builds a fresh collector bound to one run's Playwright context +
# artifacts dir.
CollectorFactory = Callable[[BrowserContext, Path], Collector]


def _first_create_write(requests: list[dict[str, Any]], marker: str) -> dict[str, Any] | None:
    """The first successful create-write for this task in the network log, or None.

    A non-idempotent write (POST/PUT/PATCH) to a URL containing `marker` that returned 2xx/3xx
    is the ground truth that the record was actually saved — the same check the replay path uses.
    It is what distinguishes a real success from an agent that *claims* success but never fired
    the create request. The record carries the agent `step` it fired on, which lets the caller
    truncate a recording at the save.
    """
    for r in requests:
        if (
            r.get("method") in ("POST", "PUT", "PATCH")
            and marker.lower() in str(r.get("url", "")).lower()
            and 200 <= (r.get("status") or 0) < 400
        ):
            return r
    return None


def _create_write_seen(requests: list[dict[str, Any]], marker: str) -> bool:
    return _first_create_write(requests, marker) is not None


# React-select's filter input: typing there is dropdown filtering, NOT a list search — Enter
# would select whatever option is focused, so the search-Enter nudge must never fire on it.
_RS_FILTER_ID = re.compile(r"^react-select-\d+-input$")
# Attributes whose value containing "search" marks a list/table search box.
_SEARCHY_ATTRS = ("placeholder", "aria-label", "id", "name")


def _search_typed_text(last_action: dict[str, Any] | None) -> str | None:
    """The text the agent just typed into a list/table SEARCH box, or None.

    `last_action` is one entry of history.model_actions(): {"<name>": params,
    "interacted_element": element}. A search box is an input whose role/type is search or
    whose placeholder/aria-label/id/name mentions "search"; dropdown (react-select/combobox)
    filters are excluded — Enter there selects the focused option instead of searching.
    """
    if not isinstance(last_action, dict):
        return None
    params = last_action.get("input")
    if not isinstance(params, dict):
        return None
    text = str(params.get("text") or "").strip()
    if not text:
        return None
    el = last_action.get("interacted_element")
    attrs = getattr(el, "attributes", None)
    if attrs is None and isinstance(el, dict):
        attrs = el.get("attributes")
    attrs = attrs or {}
    if attrs.get("role") == "combobox" or _RS_FILTER_ID.match(attrs.get("id") or ""):
        return None
    if attrs.get("role") == "searchbox" or attrs.get("type") == "search":
        return text
    if any("search" in str(attrs.get(k) or "").lower() for k in _SEARCHY_ATTRS):
        return text
    return None


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
        tools: Any | None = None,
    ) -> None:
        self.cdp_url = cdp_url
        self.config = config
        self.playwright = playwright
        # Appended to the agent's system prompt (sent every step, never trimmed).
        self.extend_system_message = extend_system_message
        # Custom action registry (browser-use Tools) exposing our escape-hatch + UI-scan
        # actions to the agent; None falls back to browser-use's built-ins only.
        self.tools = tools
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
                logger.exception("could not save expanded_task.txt: %s", exc)

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
            # ONE action per LLM step (default 5). This app re-renders after every input AND
            # after every click, so any action queued behind another acts on a stale page.
            # 2 was tried with a text rule forbidding a click as the second action, but the
            # model still queued input+click (run 20260708_153223 step 12: typed "bike" into
            # the Item react-select and clicked option-1 in the same step — the click index
            # came from the pre-typing DOM, so it hit a stale option and the item never truly
            # registered: Account/VAT were not auto-filled). Structural enforcement is the
            # only thing that reliably prevents it; every click now sees a fresh DOM.
            max_actions_per_step=1,
            # plan_update echo in every step's output; redundant with the expanded task that is
            # resent each step. ENABLE_PLANNING=false in .env turns it off (default on).
            enable_planning=self.config.enable_planning,
            extend_system_message=self.extend_system_message,
            # Custom actions the prompts rely on (skip_step, fail_and_stop, capped_scroll,
            # detect_layout_issues, run_accessibility_scan) plus all built-ins. None → built-ins.
            tools=self.tools,
            # browser-use's built-in end-of-run judge (use_judge defaults True); run it on our
            # model. We read its verdict below instead of running a second judge of our own.
            judge_llm=self.judge_llm,
            # We own SIGINT ourselves (see _prompt_and_inject) to offer a human-in-the-loop
            # override prompt on Ctrl+C, so disable browser-use's own signal handler.
            enable_signal_handler=False,
        )

        # Stamp each captured telemetry event with the agent step it fired on, and service any
        # queued human-in-the-loop pause — both at the START of each step, a safe boundary where
        # no action is in flight.
        step_state = {"n": 0}

        def _nudge_if_unintended_navigation(_agent: "AgentType") -> None:
            """Misclick detector: the model often doesn't notice a click navigated the page and
            keeps hunting on the wrong screen. If the last action changed the URL back to a page
            ALREADY visited this run (the misclick signature — forward navigation to a new page
            is normal progress), inject a go_back nudge into the next step's context."""
            try:
                urls = [u for u in _agent.history.urls() if u]
                if len(urls) < 3 or urls[-1] == urls[-2] or urls[-1] not in urls[:-2]:
                    return
                receipt = ""
                last = getattr(_agent.state, "last_result", None) or []
                if last and getattr(last[-1], "extracted_content", None):
                    content = last[-1].extracted_content
                    # A deliberate go_back is the RECOVERY, not a misclick — don't re-nudge it.
                    if "Navigated back" in content:
                        return
                    receipt = f" ({content[:80]})"
                notice = (
                    f"⚠ NAVIGATION CHECK: your last action{receipt} changed the page from "
                    f"{urls[-2]} to {urls[-1]}, which you had ALREADY visited earlier in this "
                    f"run. If this navigation was not your goal, you misclicked: call go_back "
                    f"NOW to return to {urls[-2]}, then re-locate your target with "
                    f"find_by_text. Do NOT re-navigate from the top."
                )
                mm = getattr(_agent, "_message_manager", None)
                if mm is not None and hasattr(mm, "_add_context_message"):
                    mm._add_context_message(UserMessage(content=notice))
                elif mm is not None and hasattr(mm, "add_new_task"):
                    mm.add_new_task(notice)
                else:
                    return
                logger.info("⚠ unintended-navigation nudge injected (back to %s)", urls[-1])
            except Exception as exc:  # noqa: BLE001 - a nudge must never break a step
                logger.debug("navigation nudge skipped: %s", exc)

        def _nudge_if_search_typed(_agent: "AgentType") -> None:
            """Search-Enter enforcement: many lists in this app only run the search when Enter
            is pressed, and the model reliably forgets. If the LAST action typed into a search
            box (not a dropdown filter — see _search_typed_text), inject a reminder that the
            NEXT action must be send_keys Enter. Fires once per typing action by construction:
            at the following step the last action is no longer that input."""
            try:
                actions = _agent.history.model_actions()
                if not actions:
                    return
                typed = _search_typed_text(actions[-1])
                if typed is None:
                    return
                notice = (
                    f"⚠ SEARCH CHECK: you typed '{typed[:60]}' into a search box. This app "
                    "often only runs the search when Enter is pressed. Your NEXT action MUST "
                    "be send_keys with 'Enter' (focus is already in the search field), then "
                    "wait ~2 seconds before reading the result list."
                )
                mm = getattr(_agent, "_message_manager", None)
                if mm is not None and hasattr(mm, "_add_context_message"):
                    mm._add_context_message(UserMessage(content=notice))
                elif mm is not None and hasattr(mm, "add_new_task"):
                    mm.add_new_task(notice)
                else:
                    return
                logger.info("⚠ search-Enter nudge injected (typed %r)", typed[:40])
            except Exception as exc:  # noqa: BLE001 - a nudge must never break a step
                logger.debug("search nudge skipped: %s", exc)

        # Set by our SIGINT handler when the operator hits Ctrl+C during a step; consumed at the
        # next step boundary so we never interrupt an action mid-flight.
        pause_state = {"requested": False}

        def _on_sigint(_signum: int, _frame: Any) -> None:
            if pause_state["requested"]:
                # A pause is already queued and they hit Ctrl+C again → abort the whole run.
                raise KeyboardInterrupt
            pause_state["requested"] = True
            print("\n⏸️  Pause queued — you'll be prompted for an instruction at the next step "
                  "boundary (Ctrl+C again to abort).", flush=True)

        async def _track_step(_agent: "AgentType") -> None:
            step_state["n"] += 1
            for collector in collectors:
                collector.current_step = step_state["n"]
            _nudge_if_unintended_navigation(_agent)
            _nudge_if_search_typed(_agent)
            if pause_state["requested"]:
                pause_state["requested"] = False
                self._prompt_and_inject(_agent)

        collector_results: dict[str, Any] = {}
        artifacts: dict[str, Path] = {}
        # Own SIGINT for the duration of the run so Ctrl+C opens the override prompt instead of
        # killing the process (browser-use's own handler is off via enable_signal_handler=False).
        prev_sigint: Any = None
        hitl_active = False
        try:
            prev_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, _on_sigint)
            hitl_active = True
        except (ValueError, OSError):
            logger.debug("could not install SIGINT handler (not main thread?); HITL disabled")

        # Wire the agent's verify_save_registered tool to the SAME ground truth the end-of-run
        # gate uses (_first_create_write over the live network log), so the agent can check
        # mid-run whether its Save actually reached the server and fix validation errors.
        network_collector = next((c for c in collectors if c.name == "network"), None)
        if success_marker and network_collector is not None:
            agent_tools.set_save_probe(
                lambda: _first_create_write(
                    network_collector.results().get("requests", []) or [], success_marker
                )
            )

        try:
            history = await agent.run(max_steps=max_steps, on_step_start=_track_step)
        finally:
            agent_tools.clear_save_probe()
            if hitl_active:
                try:
                    signal.signal(signal.SIGINT, prev_sigint)
                except (ValueError, OSError):
                    pass
            # Stop collectors, gather their results, then tear down both connections.
            for collector in collectors:
                try:
                    await collector.stop()
                except Exception as exc:  # noqa: BLE001
                    logger.exception("collector %s stop error: %s", collector.name, exc)

            for collector in collectors:
                collector_results[collector.name] = collector.results()
                path = collector.write()
                if path is not None:
                    artifacts[collector.name] = path

            try:
                # connect_over_cdp: closes our Playwright connection only, not the browser.
                await pw_browser.close()
            except Exception as exc:  # noqa: BLE001
                logger.exception("playwright connection close error: %s", exc)

            try:
                await session.stop()
            except Exception as exc:  # noqa: BLE001
                logger.exception("session.stop() during cleanup: %s", exc)

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
            logger.exception("could not read screenshots from history: %s", exc)

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
            logger.exception("could not build step timeline from history: %s", exc)

        # LLM token usage + cost for the run (browser-use computes this into history.usage).
        usage: dict[str, Any] | None = None
        try:
            raw_usage = history.usage
            if raw_usage is not None:
                usage = raw_usage.model_dump() if hasattr(raw_usage, "model_dump") else dict(raw_usage)
        except Exception as exc:  # noqa: BLE001
            logger.exception("could not read token usage from history: %s", exc)

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
            write = _first_create_write(requests, success_marker)
            create_write_seen = write is not None
            overridden = bool(not create_write_seen and result.is_successful)
            result.ground_truth = {
                "marker": success_marker,
                "create_write_seen": create_write_seen,
                # Agent step the commit fired on — lets run_task truncate a rescued
                # recording at the save, cutting any post-save flailing.
                "write_step": write.get("step") if write else None,
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

    def _prompt_and_inject(self, agent: Any) -> None:
        """Human-in-the-loop: at a between-steps pause (Ctrl+C), read ONE instruction from the
        terminal and inject it into the agent's context as a follow-up request, prefixed with the
        override marker the system prompt documents. Empty input resumes; Ctrl+C / EOF at the
        prompt stops the run.

        Injection goes through the message manager's add_new_task — the same primitive
        Agent.add_new_task delegates to — rather than Agent.add_new_task itself, because the
        latter recreates the agent's event bus, which is unsafe to do mid-run.
        """
        # Restore default SIGINT for the duration of the prompt so Ctrl+C raises KeyboardInterrupt
        # (lets the operator abort from the prompt), then put our handler back afterwards.
        prev = None
        try:
            prev = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, signal.default_int_handler)
        except (ValueError, OSError):
            prev = None

        print("\n➡️  Press [Enter] to resume, or type an instruction to steer the agent "
              "(Ctrl+C to abort).", flush=True)
        try:
            user_input = input("Instruction (or Enter to resume): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n🛑 Aborting run.", flush=True)
            try:
                agent.stop()
            except Exception as exc:  # noqa: BLE001
                logger.debug("agent.stop() during abort failed: %s", exc)
            user_input = ""
        finally:
            if prev is not None:
                try:
                    signal.signal(signal.SIGINT, prev)
                except (ValueError, OSError):
                    pass

        if not user_input:
            print("▶️  Resuming.", flush=True)
            return

        override = f":warning:  HUMAN OPERATOR OVERRIDE\n\n{user_input}"
        try:
            mm = getattr(agent, "_message_manager", None)
            if mm is not None and hasattr(mm, "add_new_task"):
                mm.add_new_task(override)
            else:
                agent.add_new_task(override)  # public fallback (recreates the event bus)
            print("✅ Instruction injected — the agent will act on it on the next step.", flush=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not inject human instruction: %s", exc)
            print(f"❌ Could not inject instruction: {exc}", flush=True)

    async def run_script(self, steps_path: Path, success_marker: str | None = "Invoices") -> RunResult:
        """Run a compiled selector script (no LLM, no browser-use Agent) over Playwright.

        Fast deterministic replay: executes the compiled steps back-to-back with Playwright
        auto-wait. Success is judged from the network log (a write to `success_marker`
        returned 2xx) — the same ground-truth check as `replay`. With `success_marker=None`
        (read-only tasks) success is simply a clean run: every step executed without failure.
        """
        import json as _json

        from automation.pipeline.script_compile import run_steps

        run_id, run_dir = self._new_run_dir()
        logger.info("▶ SCRIPT %s from %s", run_id, steps_path)
        steps = _json.loads(Path(steps_path).read_text())

        pw_browser = await self.playwright.chromium.connect_over_cdp(self.cdp_url)
        
        pages = [p for ctx in pw_browser.contexts for p in ctx.pages]
        page = None
        if pages:
            real_pages = [p for p in pages if p.url != "about:blank"]
            page = real_pages[0] if real_pages else pages[0]
            if len(pages) > 1:
                logger.warning("Multiple pages found (%d). Selected page URL: %s", len(pages), page.url)
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
            # The final Save's POST can still be IN FLIGHT when the last step returns —
            # observed: POST /Invoices issued but response not yet received (status None) at
            # teardown, failing the gate on a run that actually saved. Poll the live network
            # log briefly so in-flight writes can complete before we stop listening.
            if success_marker and outcome["failed_at"] is None:
                net = next((c for c in collectors if c.name == "network"), None)
                for _ in range(16):  # up to ~8s
                    if net is not None and _create_write_seen(
                            net.results().get("requests", []) or [], success_marker):
                        break
                    await page.wait_for_timeout(500)
        finally:
            for collector in collectors:
                try:
                    await collector.stop()
                except Exception as exc:  # noqa: BLE001
                    logger.exception("collector %s stop error: %s", collector.name, exc)
            for collector in collectors:
                collector_results[collector.name] = collector.results()
                path = collector.write()
                if path is not None:
                    artifacts[collector.name] = path
            try:
                await pw_browser.close()
            except Exception as exc:  # noqa: BLE001
                logger.exception("playwright close error: %s", exc)

        duration = (datetime.now() - started).total_seconds()
        clean = outcome["failed_at"] is None
        final = f"Script ran {outcome['executed']}/{len(steps)} steps in {duration:.1f}s"
        if outcome["failed_at"] is not None:
            final += f" — FAILED at step {outcome['failed_at']}: {outcome['error']}"
        if success_marker:
            requests = collector_results.get("network", {}).get("requests", []) or []
            saved = _create_write_seen(requests, success_marker)
            ok = saved
            ground_truth = {"marker": success_marker, "create_write_seen": saved}
            final += f"; create-write seen in network: {saved}"
        else:
            # Read-only task: no write is expected — a clean end-to-end run IS the success.
            ok = clean
            ground_truth = None
        result = RunResult(
            task=f"(script) {Path(steps_path).name}",
            run_id=run_id, artifacts_dir=run_dir, expanded_task=None,
            is_done=True, is_successful=ok, has_errors=not ok, final_result=final,
            urls=[], n_steps=outcome["executed"], duration_seconds=duration,
            extracted_content=[], model_actions=[],
            errors=[outcome["error"]] if outcome["error"] else [],
            collector_results=collector_results, artifacts=artifacts,
            screenshots=[], steps=[], judgement=None, usage=None,
            ground_truth=ground_truth,
        )
        logger.info("◀ SCRIPT done %s: success=%s executed=%s/%s %.1fs",
                    run_id, ok, outcome["executed"], len(steps), duration)
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
            logger.exception("could not read built-in judge verdict: %s", exc)
        return None
