"""Agent run wrapper.

`Runner` holds the per-run configuration (LLM, collectors, custom tools) and runs ONE
agent segment at a time via `run_agent_segment`, returning the raw history/usage for the
caller to distil into a `RunResult`.

It owns NO browser lifecycle: the hybrid engine (pipeline/hybrid.py) opens the
`BrowserSession`, the Playwright CDP connection, and the collectors ONCE for a whole task and
tears them down at the end — which is what lets it interleave agent segments with script
replays on the same live page.
"""
from __future__ import annotations

import json
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
from automation.pipeline import agent_tools
# React-select's filter input regex, shared with the compiler/tools: typing there is
# dropdown filtering, NOT a list search — the search-Enter nudge must never fire on it.
from automation.pipeline.script_compile import _RS_FILTER_ID

logger = logging.getLogger("framework.runner")

# A collector factory builds a fresh collector bound to one run's Playwright context +
# artifacts dir.
CollectorFactory = Callable[[BrowserContext, Path], Collector]


# FileSystem-supported formats that survive a text round-trip; binary formats (xlsx,
# images, real pdf/docx bytes) cannot be seeded into the workspace and ride the
# available_file_paths allowlist instead.
_WORKSPACE_TEXT_EXTS = {"csv", "txt", "json", "jsonl", "md", "xml", "html"}


def _workspace_files_note(files: list[str]) -> str:
    """The task-suffix telling the agent its upload files — the cloud-workspaces UX:
    reference by NAME (the string the prompt itself spells), with the resolved absolute
    paths as the accepted fallback for formats the workspace cannot hold."""
    names = "; ".join(Path(f).name for f in files)
    return ("\n\nWORKSPACE FILES available to the upload_file action — pass just the "
            f"file NAME: {names}. (Absolute paths are also accepted: "
            + "; ".join(files) + ")")


async def _seed_workspace_files(agent: Any, files: list[str]) -> list[str]:
    """Copy the run's upload files into the agent's FileSystem — the open-source analog
    of cloud `workspaces.upload` (docs.browser-use.com/cloud/agent/workspaces). The agent
    then references files by BASENAME and browser-use's upload_file resolves + validates
    them natively (local sessions only — the inverted handoff is what makes this path
    live; see browser/session.py). Non-text formats are skipped and stay covered by the
    available_file_paths allowlist. Returns the seeded basenames."""
    fs = getattr(agent, "file_system", None)
    if fs is None or not files:
        return []
    seeded: list[str] = []
    for path in files:
        p = Path(path)
        if p.suffix.lstrip(".").lower() not in _WORKSPACE_TEXT_EXTS:
            continue
        try:
            await fs.write_file(p.name, p.read_text())
            seeded.append(p.name)
        except Exception as exc:  # noqa: BLE001 - the allowlist still covers this file
            logger.debug("workspace seed failed for %s: %s", p.name, exc)
    if seeded:
        logger.info("🗂 workspace seeded: %s", ", ".join(seeded))
    return seeded


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


# Read-only discovery actions: none of these change the page, so a long unbroken run of
# them means the agent is hunting in circles instead of acting.
_DISCOVERY_ACTIONS = {"list_actions", "search_page", "find_elements",
                      "scroll", "find_by_text_lookup"}
# Consecutive discovery actions before the loop nudge fires (and re-fires every multiple).
_DISCOVERY_LOOP_AT = 4


def _action_name(entry: dict[str, Any]) -> str:
    """The action's name from a history.model_actions() entry ({name: params,
    "interacted_element": ...})."""
    return next((k for k in entry if k != "interacted_element"), "")


# browser-use serialises the clickable-element listing to at most this many characters
# (its own default is 40000 — browser_use/agent/views.py:92) and CUTS MID-ELEMENT when the
# page is denser, then prints "[End of page]" after the cut. Measured on the FPS bulk-upload
# panel (run 20260904_160415, confirmed against the live DOM): 1055 interactive controls,
# 956 printed in 40016 chars = 41.9 chars each, so the full listing is ~44.2k and the panel's
# submit button — 94.2% of the way through the page's interactive order, behind 108 grid
# rows — was in the dropped 99. The agent clicked the nameless panel close button that
# happened to be listed under a loose "FPS" text label and lost five ticked employees.
# Worst page anywhere in the recorded corpus is ~51.6k, so this clears it ~1.5x.
# The cap only bites when exceeded and the state message lives in ONE REPLACED SLOT
# (message_manager/service.py:559), never a growing history: the cost is a few thousand
# characters on the minority of dense steps and zero on the rest. Keep it FINITE — this
# app's grid grows by an employee every run (see _PANEL_SCROLL_ROUNDS) — and re-measure
# with tests/test_element_listing_budget.py if a page ever passes ~1800 controls.
CLICKABLE_LISTING_BUDGET = 80_000

_LISTING_TRUNCATED_RE = re.compile(r"Interactive elements \(truncated to \d+ characters\)")
_LISTING_INTERACTIVE_RE = re.compile(r"(\d+) interactive")
_LISTING_INDEX_RE = re.compile(r"\[(\d+)\]")


def truncated_listing_notice(state_message: str | None) -> str | None:
    """The notice for a control listing that was cut short, or None when it was complete.

    browser-use already prints "(truncated to N characters)" in the header and it did not
    help: prompts.py:276 appends "[End of page]" AFTER the cut, so the last thing the model
    reads is a severed attribute followed by a marker saying it has seen everything. Worse,
    the cut leaves bare TEXT labels stranded next to whatever element was listed near them,
    which is a direct invitation to the wrong inference — run 20260904_160415 read
    "[2238]<button />" printed under a loose "FPS" label as the FPS button and clicked the
    panel's close X instead.

    So the notice does the two things the header does not: it says HOW MANY controls are
    missing, and it routes to find_by_text, which searches the page rather than this list.
    """
    text = state_message or ""
    if not _LISTING_TRUNCATED_RE.search(text):
        return None
    total_match = _LISTING_INTERACTIVE_RE.search(text)
    total = int(total_match.group(1)) if total_match else 0
    printed = len(set(_LISTING_INDEX_RE.findall(text)))
    missing = max(total - printed, 0)
    count = (f"{missing} of the page's {total} interactive controls are NOT in it"
             if missing else "some of this page's interactive controls are NOT in it")
    return (
        f"\u26a0 THE CONTROL LIST BELOW IS INCOMPLETE — it was cut at a character limit, so "
        f"{count}. The missing ones are the LAST in page order: the bottom of a long list, "
        f"and a panel's or dialog's footer buttons. The cut lands mid-element and the "
        f"\"[End of page]\" marker printed after it is NOT true.\n"
        f"A control that is not listed has NO index, so you cannot click it by index. And "
        f"the cut strands loose TEXT labels beside whatever element happened to be listed "
        f"near them: an index printed under or beside a label is NOT necessarily that "
        f"label's control. A listed control carries its own name on its own line.\n"
        f"So if the control your step names is not in the list WITH ITS OWN NAME, do not "
        f"pick a nearby index and do not infer one from an adjacent label. Reach it by name "
        f"with find_by_text('<the text on it>', click_first=true), which searches the whole "
        f"page instead of this list."
    )


def dropped_done_notice(action_names: list[str], finished: bool) -> str | None:
    """The notice for a `done` browser-use silently discarded, or None when nothing was.

    browser-use runs a batched step in order and BREAKS at a `done` that is not the first
    action — "Done action is allowed only as a single action" (agent/service.py) — logging
    it at DEBUG, so it never reaches the run log and the agent is handed another step with
    no reason given.

    Run 20260902_105732 subtask 3 is what this is for. The agent batched
    [repeat_click(times=5), done]; the repeat landed all five clicks, the `done` vanished,
    and on the unexplained extra turn the agent read the correct end state (the 6th, unsaved
    employee on screen) as a failure and clicked Save & Next five more times by hand. Ten
    employees were paid for a five-employee slice. Telling it what actually happened is what
    lets it re-issue `done` instead of inventing a recovery.
    """
    if finished or "done" not in action_names[1:]:
        return None
    ran = [n for n in action_names[:action_names.index("done", 1)]]
    did = ", ".join(ran) if ran else "your earlier actions"
    return (
        "⚠ YOUR `done` WAS DROPPED — not because anything failed. This framework's agent "
        "loop only accepts `done` as the ONLY action in a step, so the `done` you batched "
        f"behind {did} was discarded before it ran. That is the ONLY reason you are being "
        "asked for another step.\n"
        f"Everything before it DID run and its receipts stand: {did}. Do NOT repeat any of "
        "it, do NOT treat the extra turn as evidence that it failed, and do NOT invent a "
        "recovery for work that already succeeded. If the step is finished, call `done` "
        "now as the ONLY action in this step."
    )


def discovery_loop_notice(actions: list[dict[str, Any]]) -> str | None:
    """A loop-breaking notice when the trailing actions are ALL read-only discovery, else
    None. Observed live: after misreading a find_by_text click receipt, the agent spent 18
    consecutive list/search/scroll steps hunting a control that had already been clicked —
    this fires every _DISCOVERY_LOOP_AT-th consecutive discovery step to force a decision.

    A find_by_text WITHOUT click_first counts as discovery; with click_first it acts on
    the page and resets the streak (model_actions carries the raw params, so the caller
    maps it to "find_by_text_lookup" before counting — see _nudge_if_discovery_loop)."""
    streak = 0
    for entry in reversed(actions):
        if _action_name(entry) in _DISCOVERY_ACTIONS:
            streak += 1
        else:
            break
    if streak < _DISCOVERY_LOOP_AT or streak % _DISCOVERY_LOOP_AT:
        return None
    return (
        f"⚠ DISCOVERY LOOP: your last {streak} actions were ALL read-only discovery "
        "(scroll / list_actions / search_page / find_elements) — no clicks, no input. "
        "More listing will not reveal anything new. Decide NOW:\n"
        "  1. If an earlier find_by_text receipt said it ALREADY CLICKED your target, the "
        "click happened — check for its EFFECT (did a panel/section open?) instead of "
        "re-finding the control.\n"
        "  2. Otherwise call find_by_text('<target label>', click_first=true) ONCE.\n"
        "  3. If that fails, apply the ELEMENT NOT FOUND POLICY (skip_step or "
        "fail_and_stop). Do NOT run another discovery action."
    )


def _save_interrupted_history(agent: Any, record_path: Path | None) -> Path | None:
    """Best-effort save of an interrupted segment's partial agent history.

    Ctrl+C mid-segment used to lose the step log entirely (the recording is written only
    at segment end) — four killed runs on 2026-08-05 left video-only forensics. The
    partial history lands under a distinct `.interrupted.json` suffix so the on-success
    promotion path can never adopt it. Returns the path written, or None (no record path
    was requested, or the agent has nothing saveable yet) — never raises."""
    if record_path is None:
        return None
    try:
        target = Path(record_path).with_suffix("")
        target = target.with_name(target.name + ".interrupted.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        agent.save_history(str(target))
        return target if target.exists() else None
    except Exception as exc:  # noqa: BLE001 - evidence write must never mask the abort
        logger.debug("could not save interrupted history: %s", exc)
        return None


def restore_result_metadata(history: Any, record_path: Path) -> bool:
    """Re-inject ActionResult.metadata into a saved recording (True if anything landed).

    browser-use's save_history serializes results WITHOUT their `metadata` field — which is
    where find_by_text records the element it clicked (agent_tools). Without this pass,
    every find_by_text click in a saved recording compiles to NOTHING and the committed
    script silently loses its clicks (observed live: the Reviews "View all" segment
    committed zero steps). The in-memory history still carries the metadata, so after
    save_history we copy it into the JSON where compile_recording expects it.
    """
    try:
        data = json.loads(record_path.read_text())
        items = data.get("history", [])
        changed = False
        for i, item in enumerate(getattr(history, "history", None) or []):
            if i >= len(items):
                break
            saved = items[i].get("result") or []
            for j, res in enumerate(getattr(item, "result", None) or []):
                md = getattr(res, "metadata", None)
                if md and j < len(saved) and isinstance(saved[j], dict) \
                        and not saved[j].get("metadata"):
                    saved[j]["metadata"] = md
                    changed = True
        if changed:
            record_path.write_text(json.dumps(data, indent=2))
        return changed
    except Exception as exc:  # noqa: BLE001 - enrichment must never lose the recording
        logger.warning("could not restore result metadata into %s: %s", record_path, exc)
        return False


def _inject_context(agent: Any, notice: str) -> bool:
    """Inject a one-off context message into the agent's NEXT step, via the browser-use
    message-manager seam. Returns True if it landed. Shared by every step-boundary nudge."""
    mm = getattr(agent, "_message_manager", None)
    if mm is not None and hasattr(mm, "_add_context_message"):
        mm._add_context_message(UserMessage(content=notice))
        return True
    if mm is not None and hasattr(mm, "add_new_task"):
        mm.add_new_task(notice)
        return True
    return False


@dataclass
class RunResult:
    """Structured outcome of a single task run, distilled from agent.history."""

    task: str
    run_id: str
    artifacts_dir: Path
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
    # LLM token usage + cost for the run (browser-use UsageSummary as a dict), or None.
    usage: dict[str, Any] | None = None
    # Network ground-truth check: {"marker": str, "create_write_seen": bool}, or None if no
    # marker was configured for this task. When present and create_write_seen is False, the
    # run's self-reported success was overridden to False (nothing was actually saved).
    ground_truth: dict[str, Any] | None = None
    # How the result was produced; always "hybrid" from run_task (the only execution path).
    mode: str | None = None
    # One entry per subtask segment — {index, sid, prompt, context, mode, ok, gate,
    # steps_executed, duration_seconds, healed_steps, tokens, error}. Each segment's own
    # mode is "replay" | "authored" | "replay_failed->authored".
    subtasks: list[dict[str, Any]] | None = None
    # Replay segments only: the raw outcome {executed, failed_at, error, log}. The log says
    # which selector located each step — including "healed" winner identities that
    # promote_healed persists into the golden script after a successful run.
    replay: dict[str, Any] | None = None
    # Post-run telemetry assertions (see pipeline/assertions.py). assertions_passed gates a
    # SEPARATE verdict from is_successful: overall PASS = is_successful and
    # assertions_passed is not False. None = no assertions were evaluated.
    assertion_results: list[dict[str, Any]] = field(default_factory=list)
    assertions_passed: bool | None = None

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
        extend_system_message: str | None = None,
        tools: Any | None = None,
        available_files: list[str] | None = None,
        session: Any | None = None,
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
        # Used by the subtask decomposer (decompose.py) and adapt.parameterize.
        self.expander_llm = expander_llm
        # Resolved absolute paths of the task's upload files (validated at startup by
        # pipeline/files.py against automation/uploads/). Passed to every Agent as
        # browser-use's upload_file allowlist. (Under the inverted handoff the session is
        # LOCAL and browser-use validates too; the startup validation stays as the
        # first, loudest line.)
        self.available_files: list[str] = list(available_files or [])
        # The browser-OWNING BrowserSession (launched in __main__, already started and
        # logged in). The hybrid engine reuses it for every segment instead of attaching
        # a fresh session — the browser and its authenticated state are process-scoped.
        self.session = session

    def _new_run_dir(self) -> tuple[str, Path]:
        """Allocate a unique run id + artifacts directory."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = self.config.artifacts_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_id, run_dir

    async def _start_collectors(self, pw_browser: Any, run_dir: Path) -> list[Collector]:
        """Start one collector per (context, factory) across the browser's contexts.
        A collector that fails to start is skipped — telemetry must not block a run."""
        collectors: list[Collector] = []
        for context in pw_browser.contexts:
            for factory in self.collector_factories:
                collector = factory(context, run_dir)
                try:
                    await collector.start()
                    collectors.append(collector)
                except Exception as exc:  # noqa: BLE001 - a collector must not block the run
                    logger.warning("collector %s failed to start: %s", collector.name, exc)
        return collectors

    async def run_agent_segment(
        self, agent_task: str, session: Any, collectors: list[Collector], *,
        max_steps: int = 25, record_path: Path | None = None,
        success_marker: str | None = None, request_offset: int = 0,
        repeat_budget: int | None = None,
    ) -> dict[str, Any]:
        """Run ONE agent execution against an already-started session with already-running
        collectors. Owns NO lifecycle: the caller starts/stops the session, the Playwright
        connection, and the collectors — which is what lets the hybrid engine interleave
        agent segments with script replays on the same live page.

        `request_offset` windows the save probe to requests captured from that index on, so
        a mid-task segment doesn't credit a create-write an earlier segment fired.

        Returns {"history", "usage"}.
        """
        if self.available_files:
            agent_task += _workspace_files_note(self.available_files)
        agent = Agent(
            task=agent_task,
            llm=self.llm,
            browser_session=session,
            available_file_paths=self.available_files or None,
            use_vision=self.config.use_vision,
            vision_detail_level=self.config.vision_detail_level,
            max_history_items=self.config.max_history_items,
            # The clickable-element listing budget. See CLICKABLE_LISTING_BUDGET: the
            # 40000 default silently dropped the FPS panel's submit button and the agent
            # clicked a nameless neighbour instead.
            max_clickable_elements_length=CLICKABLE_LISTING_BUDGET,
            # o4-mini needs an explicit per-call budget: browser-use's model-name
            # heuristic hands it the default 75s (only o3/claude/deepseek get 90),
            # which medium-effort thinks can exceed. step_timeout is ONE asyncio
            # timeout around the LLM call PLUS tool execution, so it must clear
            # llm_timeout with action headroom.
            llm_timeout=180,
            step_timeout=300,
            # Raised 1→4 (2026-07-30, experiment). CAUTION: 1 was a deliberate fix — this app
            # re-renders after every input AND click, so an action queued behind another acts
            # on a stale page. 2 was tried with a text rule forbidding a click as the second
            # action, but the model still queued input+click (run 20260708_153223 step 12:
            # typed "bike" into the Item react-select and clicked option-1 in the same step —
            # the click index came from the pre-typing DOM, so it hit a stale option and the
            # item never truly registered: Account/VAT were not auto-filled). If stale-index
            # misclicks reappear, drop back to 1 (structural enforcement was the only thing
            # that reliably prevented them).
            max_actions_per_step=4,
            # plan_update echo in every step's output; redundant with the expanded task that is
            # resent each step. ENABLE_PLANNING=false in .env turns it off (default on).
            # Worth an A/B off: scoped_subtask_prompt already restates the job every step, so
            # the echo is redundant token weight a small model can drift on. Not flipped yet —
            # measure before changing the default.
            enable_planning=self.config.enable_planning,
            # Pinned explicitly (0.13.3 default is already True) so a library upgrade or
            # env drift can't silently drop the per-step thinking field.
            use_thinking=True,
            # Recent browser events (navigations, dialogs, downloads) in the step context:
            # the agent sees WHAT just happened instead of inferring it from a changed DOM.
            include_recent_events=True,
            # Few-shot tool-call examples in the system context — steadier action-schema
            # output from small/low-effort models.
            include_tool_call_examples=True,
            extend_system_message=self.extend_system_message,
            # Custom actions the prompts rely on (skip_step, fail_and_stop, scroll_panels,
            # detect_layout_issues, run_accessibility_scan) plus all built-ins. None → built-ins.
            tools=self.tools,
            # browser-use's end-of-run judge (use_judge defaults True) is OFF — the
            # 2026-07-30 decision: the hybrid engine's segment gates are the verdict, and
            # the judge never overrides the agent's self-reported success.
            # Found regressed to True on 2026-08-07: every authored segment ended with a
            # full-trace judgement on the medium-effort expander deployment (240s timeout,
            # 5 retries) — minutes of dead air between subtasks, verdict thrown away.
            use_judge=False,
            # We own SIGINT ourselves (see _prompt_and_inject) to offer a human-in-the-loop
            # override prompt on Ctrl+C, so disable browser-use's own signal handler.
            enable_signal_handler=False,
            # Recovery headroom for consecutive step failures — the 0.13.3 default, pinned
            # explicitly so a library upgrade can't silently move it.
            max_failures=5,
            # Never let browser-use auto-navigate to a URL it thinks it sees in the task
            # text. Every segment starts on an already-correct live page, and the extractor
            # misreads prose as domains (observed live: the decomposer's "Bookkeeping
            # module.search for..." wording became a navigate to https://module.search,
            # ERR_NAME_NOT_RESOLVED, failing the whole run).
            directly_open_url=False,
        )
        await _seed_workspace_files(agent, self.available_files)

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
                if not _inject_context(_agent, notice):
                    return
                logger.info("⚠ unintended-navigation nudge injected (back to %s)", urls[-1])
            except Exception as exc:  # noqa: BLE001 - a nudge must never break a step
                logger.debug("navigation nudge skipped: %s", exc)

        def _nudge_if_discovery_loop(_agent: "AgentType") -> None:
            """Flail detector: inject a loop-breaking redirect when the agent has done
            nothing but read-only discovery for several steps (see discovery_loop_notice)."""
            try:
                actions = []
                for entry in _agent.history.model_actions() or []:
                    name = _action_name(entry)
                    if name == "find_by_text" and not (entry.get(name) or {}).get("click_first"):
                        actions.append({"find_by_text_lookup": {}})  # lookup = discovery
                    else:
                        actions.append(entry)
                notice = discovery_loop_notice(actions)
                if notice and _inject_context(_agent, notice):
                    logger.info("⚠ discovery-loop nudge injected")
            except Exception as exc:  # noqa: BLE001 - a nudge must never break a step
                logger.debug("discovery-loop nudge skipped: %s", exc)

        def _nudge_if_listing_truncated(_agent: "AgentType") -> None:
            """Warn while the page is still dense enough to cut the listing short.

            Read on_step_start, so this reflects the state message the agent was LAST
            shown — which is the right one: a panel that overflowed the budget on the
            previous step is still open on this one, and that is exactly when the agent is
            picking indexes inside it. With CLICKABLE_LISTING_BUDGET raised this fires
            rarely; it is the net under the budget, not a substitute for it."""
            try:
                mm = getattr(_agent, "_message_manager", None)
                history = getattr(getattr(mm, "state", None), "history", None)
                message = getattr(history, "state_message", None)
                notice = truncated_listing_notice(getattr(message, "content", None))
            except Exception as exc:  # noqa: BLE001 - never break a step
                logger.debug("listing-truncation check skipped: %s", exc)
                return
            if notice and _inject_context(_agent, notice):
                logger.info("\u26a0 truncated control listing surfaced to agent")

        def _nudge_if_done_dropped(_agent: "AgentType") -> None:
            """Tell the agent when its batched `done` was silently discarded, so it
            re-issues it instead of inventing a recovery (see dropped_done_notice)."""
            try:
                items = getattr(_agent.history, "history", None) or []
                if not items:
                    return
                last = items[-1]
                out = getattr(last, "model_output", None)
                if out is None:
                    return
                names = []
                for act in getattr(out, "action", None) or []:
                    data = act.model_dump(exclude_unset=True)
                    names.append(next(iter(data), "") if data else "")
                finished = any(getattr(r, "is_done", False)
                               for r in (getattr(last, "result", None) or []))
                notice = dropped_done_notice(names, finished)
                if notice and _inject_context(_agent, notice):
                    logger.info("⚠ dropped-done nudge injected (batch was %s)",
                                " + ".join(n for n in names if n))
            except Exception as exc:  # noqa: BLE001 - a nudge must never break a step
                logger.debug("dropped-done nudge skipped: %s", exc)

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
                    f"⚠ SEARCH CHECK: you typed '{typed[:60]}' into a search box. Enter was "
                    "already pressed for you (the input tool does it automatically) — do NOT "
                    "send Enter again. Wait ~2 seconds for the filtered results to load "
                    "before reading the list."
                )
                if not _inject_context(_agent, notice):
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

        async def _surface_notifications(_agent: "AgentType") -> None:
            """Toasts/message bars fade before the agent's next look, so capture them as they
            appear (agent_tools installs the observer on first read) and inject any new ones
            into this step's context — the app's own verdict on the last action."""
            try:
                notices = await agent_tools.read_new_notifications(session)
            except Exception as exc:  # noqa: BLE001 - never break a step
                logger.debug("notification surfacing skipped: %s", exc)
                return
            if not notices:
                return
            joined = " | ".join(n[:200] for n in notices[:5])
            notice = (
                f"⚠ PAGE NOTIFICATION(S) since your last action: {joined}\n"
                "This is the app reporting the OUTCOME of what you just did. If it confirms "
                "success, treat that as your evidence. If it states an error, do NOT report "
                "success and do NOT react blindly: first reason about WHY, given the steps "
                "already completed this run, then follow the PAGE NOTIFICATIONS protocol — "
                "validation: fix the named fields and save again; ALREADY done/exists (said "
                "about the outcome you were told to produce): an earlier step or run already "
                "produced it — verify the state and skip_step, never force it again; "
                "permission/blocked: fail_and_stop quoting it; transient: retry once."
            )
            if _inject_context(_agent, notice):
                logger.info("⚠ page-notification surfaced: %s", joined[:80])

        # Downloads persist on the SESSION across segments; baseline at this segment's
        # start so only downloads the agent itself triggered get surfaced to it.
        def _session_downloads() -> list[str]:
            try:
                return list(getattr(session, "downloaded_files", None) or [])
            except Exception:  # noqa: BLE001 - download accounting is best-effort
                return []

        seen_downloads = {"n": len(_session_downloads())}

        def _surface_downloads(_agent: "AgentType") -> None:
            """A click that starts a download often returns a TIMEOUT receipt (the click
            watchdog waits for a page consequence that never comes) — observed live: the
            agent read the timeout as failure, re-clicked, and downloaded the file twice.
            Surfacing the download as the step's authoritative outcome corrects the lie."""
            files = _session_downloads()
            new = files[seen_downloads["n"]:]
            if not new:
                return
            seen_downloads["n"] = len(files)
            names = ", ".join(Path(p).name for p in new)
            notice = (
                f"⚠ DOWNLOAD COMPLETED since your last action: {names}. The click that "
                f"triggered it SUCCEEDED even if its receipt showed a timeout or error — "
                f"the download WAS the click's outcome. Do NOT click the control again; "
                f"that would download a duplicate. Confirm with verify_download if needed "
                f"and move on."
            )
            if _inject_context(_agent, notice):
                logger.info("📥 download surfaced to agent: %s", names[:100])

        async def _track_step(_agent: "AgentType") -> None:
            step_state["n"] += 1
            for collector in collectors:
                collector.current_step = step_state["n"]
            _nudge_if_unintended_navigation(_agent)
            _nudge_if_done_dropped(_agent)
            _nudge_if_search_typed(_agent)
            _nudge_if_discovery_loop(_agent)
            _nudge_if_listing_truncated(_agent)
            # Re-assert the reveal stylesheet on the current document each step (idempotent,
            # one cheap CDP eval; on_step_start, so it lands BEFORE this step's snapshot):
            # covers tabs created outside login.py's init-scripted context.
            if self.config.reveal_hidden_controls:
                await agent_tools.ensure_reveal_css(session)
            # Same per-step heal for the callout scroll pin, but ungated: a popup dismissed
            # by a scroll loses the value being typed into it (see ensure_callout_scroll_pin).
            await agent_tools.ensure_callout_scroll_pin(session)
            await _surface_notifications(_agent)
            _surface_downloads(_agent)
            if pause_state["requested"]:
                pause_state["requested"] = False
                self._prompt_and_inject(_agent)

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

        # Live network bridge: verify_save_registered and the click receipts both read the
        # write requests fired (with the server's body verdict) from here — the ground truth
        # that was sitting unshown in the collector while the agent re-submitted an
        # already-accepted FPS. It replaced a marker-gated first-create-write probe, which
        # no task without a declared marker ever got and which saturated after one save.
        network_collector = next((c for c in collectors if c.name == "network"), None)
        if network_collector is not None:
            agent_tools.set_live_network(network_collector)
        # The segment's declared repeat count ("exactly 5 more clicks"), which caps
        # repeat_click's total for one control. set_live_network above clears the ledger, so
        # this must follow it.
        agent_tools.set_repeat_budget(repeat_budget)

        try:
            history = await agent.run(max_steps=max_steps, on_step_start=_track_step)
        except KeyboardInterrupt:
            saved = _save_interrupted_history(agent, record_path)
            if saved is not None:
                logger.info("⏸ interrupted — partial segment history saved to %s", saved)
            raise
        finally:
            agent_tools.clear_live_network()
            agent_tools.set_repeat_budget(None)
            if hitl_active:
                try:
                    signal.signal(signal.SIGINT, prev_sigint)
                except (ValueError, OSError):
                    pass

        # Save the action trace for later deterministic replay (no LLM) if requested.
        if record_path is not None:
            try:
                record_path.parent.mkdir(parents=True, exist_ok=True)
                agent.save_history(str(record_path))
                # save_history drops ActionResult.metadata — the ONLY channel carrying
                # find_by_text's and click's element record, extract_data/copy_text's
                # captured value, and paste_text's delivered value. Put it back, or those
                # steps compile to nothing at all.
                restore_result_metadata(history, record_path)
                logger.info("recorded action trace -> %s", record_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not save recording: %s", exc)

        # LLM token usage + cost for the run (browser-use computes this into history.usage).
        usage: dict[str, Any] | None = None
        try:
            raw_usage = history.usage
            if raw_usage is not None:
                usage = raw_usage.model_dump() if hasattr(raw_usage, "model_dump") else dict(raw_usage)
        except Exception as exc:  # noqa: BLE001
            logger.exception("could not read token usage from history: %s", exc)

        return {
            "history": history,
            "usage": usage,
        }

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

