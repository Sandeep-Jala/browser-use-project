"""Hybrid subtask engine: replay recorded subtasks, LLM only for the gaps.

Executes one task as an ordered list of subtasks (see decompose.py) on ONE live browser
session: each subtask is replayed from the shared library (subtask_store) when a recording
exists — zero LLM — and agent-authored (and committed to the library) when it doesn't. A
failed replay hands the SAME live page to the agent for in-place recovery, so a broken
segment never forces a whole-task re-author.

Seamless handoff is the core mechanic: `HybridSession` opens one browser-use BrowserSession,
one Playwright CDP connection, and the telemetry collectors for the WHOLE task; nothing is
torn down between subtasks, so an agent segment starts from the exact DOM state the previous
replay left (and vice versa). Segment steps carry no leading goto — the context-keyed library
lookup guarantees the page is already in the segment's start state.

This is the ONLY execution path: there is no whole-task replay tier above it. A task gets
faster as its subtasks land in the library — and because the library is keyed on (tokenized
prompt, page context) rather than the parent task, a subtask recorded by one task replays
inside every other task that shares that wording.

Composition nodes are TYPED (decompose.node_kind): "action" nodes replay from the library;
"judge" nodes are cognitive — their success is a judgment (verify/compare/observe) that
cannot survive compilation into a selector script, so they always run as agent segments and
are never committed (a replayed judge would walk the clicks with nobody looking and report
a hollow pass). Observations flow FORWARD: each completed segment's finding (its distilled
final result) is handed to every later agent segment, so a note-then-verify task can compare
against what was actually observed instead of guessing.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from playwright.async_api import Page

from automation import skills
from automation.pipeline import adapt
from automation.pipeline import router
from automation.pipeline import subtask_store as sstore
from automation.pipeline.decompose import Subtask, get_decomposition
from automation.pipeline.prompts import scoped_subtask_prompt
from automation.pipeline.runner import RunResult, Runner, _first_create_write
from automation.pipeline.script_compile import promote_healed, save_steps
from automation.browser.session import attach_session

logger = logging.getLogger("framework.hybrid")

# A library entry that fails this many CONSECUTIVE replays is stale by definition: archive
# it so the next run authors a clean replacement (see subtask_store.archive_if_failing).
_ARCHIVE_AFTER_FAILURES = 2


# ------------------------------- gates -------------------------------


@dataclass
class Gate:
    """How one segment's success is judged (see plan: marker > postcondition > steps)."""

    kind: str                       # "marker" | "postcondition" | "steps"
    marker: str | None = None       # kind == "marker": create-write URL fragment
    postcondition: dict[str, Any] | None = None   # {"url_contains": ...} | {"visible": ...}
    end_context: str | None = None  # recorded end context (normalized URL) to compare


def segment_gate(sub: Subtask, entry: dict[str, Any] | None, context: str) -> Gate:
    """Resolve the gate for a subtask: its marker (the save-owning segment) wins; else a
    declared postcondition; else the library entry's recorded end_context when it differs
    from the start context (a navigation segment must actually land somewhere); else the
    steps floor (clean execution / agent self-report)."""
    if sub.marker:
        return Gate(kind="marker", marker=sub.marker)
    if sub.postcondition:
        return Gate(kind="postcondition", postcondition=sub.postcondition)
    end_context = (entry or {}).get("end_context")
    if end_context and end_context != context:
        return Gate(kind="postcondition", end_context=end_context)
    return Gate(kind="steps")


def _describe_expected_end(gate: Gate) -> str | None:
    """The gate's pass condition in words the agent can act on, or None when the gate has
    no page-state condition (marker gates verify via verify_save_registered instead, and
    steps gates have nothing to check). This is what turns the library's recorded outcome
    into the authoring agent's explicit done-condition."""
    if gate.kind != "postcondition":
        return None
    # Branch order mirrors evaluate_gate exactly, so the condition described to the agent
    # is always the one the gate will enforce.
    if gate.postcondition and gate.postcondition.get("url_contains"):
        return f'the page URL contains "{gate.postcondition["url_contains"]}"'
    if gate.postcondition and gate.postcondition.get("visible"):
        return f'the element matching "{gate.postcondition["visible"]}" is visible'
    if gate.end_context:
        return (f'the page URL path matches "{gate.end_context}" '
                f'(lowercased; each "*" stands for a record id)')
    return None


# Postcondition settle window: an SPA can still be re-rendering/navigating when a
# segment's last step returns, so every page-state check gets a few short re-polls before
# a miss counts as a failure. (Gate tests shrink the delay to keep the suite fast.)
_SETTLE_TRIES = 6
_SETTLE_DELAY = 0.5


async def _settled(check: Any, steps_ok: bool) -> bool:
    """True as soon as the page-state `check` passes, re-polling briefly while it doesn't.
    A failed-steps segment gets its single honest evaluation (so the gate detail is still
    recorded) but no settle window — the gate cannot pass anyway."""
    for i in range(_SETTLE_TRIES):
        if await check():
            return True
        if not steps_ok or i == _SETTLE_TRIES - 1:
            return False
        await asyncio.sleep(_SETTLE_DELAY)
    return False


async def evaluate_gate(
    gate: Gate, *, steps_ok: bool, page: Page | None,
    requests_window: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    """Evaluate a segment gate. Returns (ok, detail).

    kind == "marker" is authoritative: the segment passed iff its network window carries a
    successful create-write to the marker — exactly the whole-task ground-truth rule, scoped
    to this segment's traffic. The other kinds additionally require steps_ok.
    """
    if gate.kind == "marker":
        write = _first_create_write(requests_window, gate.marker or "")
        return write is not None, {
            "kind": "marker", "marker": gate.marker,
            "create_write_seen": write is not None,
            "write_step": write.get("step") if write else None,
        }
    if gate.kind == "postcondition":
        detail: dict[str, Any] = {"kind": "postcondition"}
        ok = steps_ok
        check = None
        try:
            if gate.postcondition and gate.postcondition.get("url_contains"):
                frag = str(gate.postcondition["url_contains"]).lower()
                detail["url_contains"] = frag

                async def check() -> bool:
                    return page is not None and frag in page.url.lower()
            elif gate.postcondition and gate.postcondition.get("visible"):
                sel = str(gate.postcondition["visible"])
                detail["visible"] = sel

                async def check() -> bool:
                    return page is not None and \
                        await page.locator(sel).first.is_visible()
            elif gate.end_context:
                detail["end_context"] = gate.end_context

                async def check() -> bool:
                    detail["reached"] = sstore.normalize_context(
                        page.url if page is not None else "")
                    return detail["reached"] == gate.end_context
            if check is not None:
                # No short-circuit on a failed-steps segment: _settled still evaluates
                # once so the detail (e.g. "reached") lands in the report.
                ok = await _settled(check, steps_ok) and ok
        except Exception as exc:  # noqa: BLE001 - an unreadable page fails the check honestly
            logger.warning("postcondition check errored: %s", exc)
            detail["error"] = str(exc)
            ok = False
        return ok, detail
    return steps_ok, {"kind": "steps"}


# ------------------------------- segment result -------------------------------


@dataclass
class Segment:
    """Outcome of one subtask segment."""

    index: int
    sid: str
    prompt: str                     # the instantiated (concrete) subtask prompt
    context: str
    mode: str                       # "replay" | "authored" | "replay_failed->authored"
    kind: str = "action"            # composition node kind: "action" | "judge"
    ok: bool = False
    gate: dict[str, Any] = field(default_factory=dict)
    steps_executed: int = 0
    duration_seconds: float = 0.0
    healed_steps: list[int] = field(default_factory=list)
    tokens: int = 0
    cost: float = 0.0
    error: str | None = None
    replay: dict[str, Any] | None = None
    write_step: int | None = None   # agent-relative step the create-write fired on (rescue)
    # Distilled observation from an agent segment (its final result) — carried forward into
    # every later agent segment's context so note-then-verify flows can actually compare.
    finding: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "sid": self.sid, "prompt": self.prompt,
            "context": self.context, "mode": self.mode, "kind": self.kind, "ok": self.ok,
            "gate": self.gate, "steps_executed": self.steps_executed,
            "duration_seconds": round(self.duration_seconds, 1),
            "healed_steps": self.healed_steps, "tokens": self.tokens, "error": self.error,
            "finding": self.finding,
        }


# ------------------------------- the shared session -------------------------------


class HybridSession:
    """One task's live execution surface: a single BrowserSession (agent driver), a single
    Playwright CDP connection (replay driver + telemetry), and the collectors — all opened
    once and torn down only in finalize(). Both drivers point at the same Chromium, and the
    one-real-tab invariant makes "the single non-blank page" the shared state."""

    def __init__(self, runner: Runner) -> None:
        self.runner = runner
        self.run_id: str = ""
        self.run_dir: Path = Path(".")
        self.session: Any = None
        self.pw_browser: Any = None
        self.collectors: list[Any] = []
        self.started = datetime.now()

    @classmethod
    async def open(cls, runner: Runner) -> "HybridSession":
        hs = cls(runner)
        hs.run_id, hs.run_dir = runner._new_run_dir()
        hs.session = await attach_session(runner.cdp_url, runner.config)
        await hs.session.start()
        hs.pw_browser = await runner.playwright.chromium.connect_over_cdp(runner.cdp_url)
        hs.collectors = await runner._start_collectors(hs.pw_browser, hs.run_dir)
        hs.started = datetime.now()
        return hs

    def current_page(self) -> Page | None:
        """The single live tab, re-queried each call (agent segments can mutate the page
        list). Prefers the first non-blank page, so the page pick stays deterministic."""
        pages = [p for ctx in self.pw_browser.contexts for p in ctx.pages]
        if not pages:
            return None
        real = [p for p in pages if p.url != "about:blank"]
        return real[0] if real else pages[0]

    async def current_url(self) -> str:
        page = self.current_page()
        return page.url if page is not None else ""

    async def close_extra_tabs(self) -> None:
        """Re-enforce the one-real-tab invariant after an agent segment (a misclick can open
        a new tab, and the next segment's page pick must stay deterministic)."""
        try:
            pages = [p for ctx in self.pw_browser.contexts for p in ctx.pages]
            keep = self.current_page()
            for p in pages:
                if p is not keep:
                    await p.close()
        except Exception as exc:  # noqa: BLE001 - best-effort hygiene
            logger.debug("close_extra_tabs: %s", exc)

    def network_watermark(self) -> int:
        """Length of the cumulative request log now — segment gates window from here."""
        net = next((c for c in self.collectors if c.name == "network"), None)
        if net is None:
            return 0
        return len(net.results().get("requests", []) or [])

    def requests_since(self, watermark: int) -> list[dict[str, Any]]:
        net = next((c for c in self.collectors if c.name == "network"), None)
        if net is None:
            return []
        return (net.results().get("requests", []) or [])[watermark:]

    def all_requests(self) -> list[dict[str, Any]]:
        return self.requests_since(0)

    async def wait_for_inflight_write(self, marker: str) -> None:
        """The final Save's POST can still be in flight when the last step returns; poll the
        live log briefly so it can land before the gate reads it."""
        for _ in range(16):  # up to ~8s
            if _first_create_write(self.all_requests(), marker) is not None:
                return
            await asyncio.sleep(0.5)

    async def finalize(self, task: str, parent_marker: str | None) -> RunResult:
        """Stop collectors + session ONCE, gather telemetry, and distill the whole-task
        RunResult skeleton (the caller fills subtasks/success/mode)."""
        collector_results: dict[str, Any] = {}
        artifacts: dict[str, Path] = {}
        if parent_marker:
            try:
                await self.wait_for_inflight_write(parent_marker)
            except Exception as exc:  # noqa: BLE001
                logger.debug("in-flight write poll failed: %s", exc)
        for collector in self.collectors:
            try:
                await collector.stop()
            except Exception as exc:  # noqa: BLE001
                logger.exception("collector %s stop error: %s", collector.name, exc)
        for collector in self.collectors:
            collector_results[collector.name] = collector.results()
            path = collector.write()
            if path is not None:
                artifacts[collector.name] = path
        try:
            await self.pw_browser.close()
        except Exception as exc:  # noqa: BLE001
            logger.exception("playwright connection close error: %s", exc)
        try:
            await self.session.stop()
        except Exception as exc:  # noqa: BLE001
            logger.exception("session.stop() during cleanup: %s", exc)

        duration = (datetime.now() - self.started).total_seconds()
        ground_truth: dict[str, Any] | None = None
        if parent_marker:
            requests = collector_results.get("network", {}).get("requests", []) or []
            write = _first_create_write(requests, parent_marker)
            ground_truth = {
                "marker": parent_marker,
                "create_write_seen": write is not None,
                "write_step": write.get("step") if write else None,
                "overrode_success": False,
            }
        return RunResult(
            task=task, run_id=self.run_id, artifacts_dir=self.run_dir,
            is_done=True, is_successful=None, has_errors=False, final_result=None,
            urls=[], n_steps=0, duration_seconds=duration, extracted_content=[],
            model_actions=[], errors=[], collector_results=collector_results,
            artifacts=artifacts, screenshots=[], steps=[], judgement=None, usage=None,
            ground_truth=ground_truth,
        )

    # ------------------------------- segment execution -------------------------------

    async def replay_segment(
        self, sub: Subtask, sid: str, context: str, skill: skills.Skill, gate: Gate,
    ) -> Segment:
        """Replay a library skill on the live page. No LLM."""
        started = datetime.now()
        watermark = self.network_watermark()
        page = self.current_page()
        seg = Segment(index=sub.index, sid=sid, prompt=sub.instantiated_prompt,
                      context=context, mode="replay")
        if page is None:
            seg.error = "no open page to replay against"
            return seg
        outcome = await skills.execute(skill, page)
        if gate.kind == "marker" and gate.marker:
            await self.wait_for_inflight_write(gate.marker)
        seg.replay = outcome
        seg.steps_executed = outcome.get("executed", 0)
        seg.error = outcome.get("error")
        seg.healed_steps = [e["step"] for e in (outcome.get("log") or []) if e.get("healed")]
        steps_ok = outcome.get("failed_at") is None
        seg.ok, seg.gate = await evaluate_gate(
            gate, steps_ok=steps_ok, page=self.current_page(),
            requests_window=self.requests_since(watermark),
        )
        if steps_ok and not seg.ok:
            seg.error = seg.error or f"segment gate failed: {seg.gate}"
        seg.duration_seconds = (datetime.now() - started).total_seconds()
        return seg

    async def agent_segment(
        self, sub: Subtask, sid: str, context: str, gate: Gate, *,
        completed: list[str], remaining: list[str],
        dirty: bool = False, prior_failure: str | None = None,
        record_path: Path | None = None, findings: list[str] | None = None,
    ) -> Segment:
        """Run the LLM agent for ONE subtask on the shared live session. `findings` are the
        observations earlier segments recorded (each a "prompt: outcome" line) — the data
        channel that lets a verify step compare against what a note step actually saw."""
        started = datetime.now()
        watermark = self.network_watermark()
        kind = getattr(sub, "kind", "action")
        prompt = scoped_subtask_prompt(
            sub.instantiated_prompt, completed, remaining, dirty, prior_failure,
            expected_end=_describe_expected_end(gate),
            owns_save=gate.kind == "marker",
            findings=findings, observe=kind == "judge")
        seg = Segment(index=sub.index, sid=sid, prompt=sub.instantiated_prompt,
                      context=context, mode="authored", kind=kind)
        try:
            out = await self.runner.run_agent_segment(
                prompt, self.session, self.collectors,
                max_steps=self.runner.config.subtask_max_steps,
                record_path=record_path, success_marker=gate.marker,
                request_offset=watermark,
            )
        except Exception as exc:  # noqa: BLE001 - a crashed segment is a failed segment
            logger.exception("agent segment %s crashed: %s", sid, exc)
            seg.error = str(exc)
            seg.duration_seconds = (datetime.now() - started).total_seconds()
            return seg
        finally:
            await self.close_extra_tabs()
        history = out["history"]
        seg.steps_executed = history.number_of_steps()
        usage = out.get("usage") or {}
        seg.tokens = int(usage.get("total_tokens") or 0)
        seg.cost = float(usage.get("total_cost") or 0.0)
        if gate.kind == "marker" and gate.marker:
            await self.wait_for_inflight_write(gate.marker)
        steps_ok = bool(history.is_successful())
        seg.ok, seg.gate = await evaluate_gate(
            gate, steps_ok=steps_ok, page=self.current_page(),
            requests_window=self.requests_since(watermark),
        )
        seg.write_step = seg.gate.get("write_step")
        if not seg.ok:
            if steps_ok:
                # The agent believed it succeeded but the gate disagreed: surface the gate
                # verdict, not the agent's happy final text.
                seg.error = (f"gate failed: {seg.gate} "
                             f"(agent claimed success: {history.final_result()!r})")
            else:
                seg.error = history.final_result() or f"segment gate failed: {seg.gate}"
        else:
            # The segment's distilled observation, carried into later segments' context.
            final = " ".join(str(history.final_result() or "").split())
            seg.finding = final[:300] or None
        seg.duration_seconds = (datetime.now() - started).total_seconds()
        return seg


# ------------------------------- commit / heal helpers -------------------------------


async def _save_segment_template(sid: str, prompt: str, steps: list[dict[str, Any]],
                                 llm: Any) -> dict[str, str] | None:
    """Parameterize a just-committed segment into its library template (best-effort)."""
    try:
        template = await adapt.parameterize(prompt, steps, llm)
        if not template:
            return None
        adapt.save_template(sstore.template_path(sid), template)
        return template["params"]
    except Exception as exc:  # noqa: BLE001 - a template is a bonus, not a requirement
        logger.warning("could not save template for segment %s: %s", sid, exc)
        return None


def _promote_segment_heals(sid: str, seg: Segment, *, from_template: bool) -> None:
    """Persist a PASSED replay's healings into the library entry. Skipped for
    template-instantiated replays (v1): the healed selectors would carry concrete values
    into the tokenized steps/anchors. Concrete entries promote as-is — into the ANCHOR
    bundle when the replay ran the code tier (its log entries carry the anchor handle),
    else into the steps file."""
    if from_template or not seg.healed_steps:
        return
    try:
        log = (seg.replay or {}).get("log") or []
        if any("handle" in e for e in log):
            promoted: list[Any] = skills.promote_healed_anchors(
                sstore.anchors_path(sid), log)
        else:
            promoted = promote_healed(sstore.steps_path(sid), log)
        if promoted:
            sstore.update_manifest(sid, "", healed_steps=promoted,
                                   healed=datetime.now().isoformat(timespec="seconds"))
            logger.info("segment %s: promoted healed selectors into %s", sid, promoted)
    except Exception as exc:  # noqa: BLE001 - promotion is a bonus; the run already passed
        logger.warning("heal promotion failed for segment %s: %s", sid, exc)


async def _author_segment(
    hs: HybridSession, sub: Subtask, sid: str, context: str, gate: Gate, *,
    completed: list[str], remaining: list[str],
    dirty: bool = False, prior_failure: str | None = None,
    findings: list[str] | None = None, commit: bool = True,
) -> Segment:
    """Agent-author one subtask and commit it to the library when honest.

    Commit only from a CLEAN start: a dirty-recovery run's recording begins mid-broken-state
    and would not replay from the entry's declared context (the per-subtask analogue of
    "only a whole run can pass the gate honestly"). A clean author that passed its gate is
    compiled (no leading goto, truncated at the save when the gate was a marker),
    parameterized, and registered in the manifest with its start/end contexts.

    `commit=False` (judge nodes) runs the agent without recording or committing anything:
    a cognitive segment's success is a judgment, and a compiled replay of it would be a
    hollow pass — so nothing of it may ever enter the library.
    """
    seg = await hs.agent_segment(
        sub, sid, context, gate, completed=completed, remaining=remaining,
        dirty=dirty, prior_failure=prior_failure, findings=findings,
        record_path=sstore.recording_path(sid) if commit and not dirty else None,
    )
    if not seg.ok or not commit:
        return seg
    if dirty:
        # Recovered in place, but the recording is not committable. A stale entry that keeps
        # failing gets retired so the NEXT run authors a clean replacement.
        sstore.archive_if_failing(sid, threshold=_ARCHIVE_AFTER_FAILURES)
        return seg

    # Rescue truncation: if the agent flailed after the save landed, cut the segment's
    # script at the write step so post-save junk never enters the library.
    truncate_at = None
    if gate.kind == "marker" and seg.write_step:
        truncate_at = seg.write_step
    try:
        steps = save_steps(sstore.recording_path(sid), sstore.steps_path(sid),
                           max_steps=truncate_at, emit_start_goto=False)
        if not steps:
            # A zero-step script would replay as a hollow no-op pass. Leave NO entry (the
            # recording stays for diagnosis); the next run authors this segment again.
            sstore.steps_path(sid).unlink(missing_ok=True)
            logger.warning("segment %s: recording compiled to ZERO steps; not committing "
                           "(agent likely acted only through tools the compiler drops)", sid)
            return seg
        end_context = sstore.normalize_context(await hs.current_url())
        params = await _save_segment_template(
            sid, sub.instantiated_prompt, steps, hs.runner.expander_llm)
        sstore.update_manifest(
            sid, sub.template_prompt,
            params=params or dict(sub.values or {}),
            context=context, end_context=end_context,
            marker_write=gate.kind == "marker", steps=len(steps), provenance="clean",
        )
        # Tier-1 upgrade (best-effort): transpile the committed steps into a code skill
        # (<sid>.skill.py + anchors). On any failure the steps stay authoritative and any
        # stale code artifacts are removed (codegen.compile_code_skill owns that).
        skills.compile_code_skill(sid)
        logger.info("segment %s: committed %d steps to the library", sid, len(steps))
    except Exception as exc:  # noqa: BLE001 - compile failure must not fail a passed segment
        logger.exception("segment %s compile/commit error: %s", sid, exc)
    return seg


# ------------------------------- the hybrid loop -------------------------------


def reauthor_match(reauthor: str | None, sub: Subtask) -> bool:
    """True when --reauthor names this subtask. `reauthor` is a comma-list mixing subtask
    indexes ("3") and case-insensitive prompt substrings ("add estimate"); either the
    template or the instantiated wording can match."""
    if not reauthor:
        return False
    for token in (t.strip().lower() for t in reauthor.split(",")):
        if not token:
            continue
        if token.isdigit():
            if int(token) == sub.index:
                return True
        elif token in sub.template_prompt.lower() \
                or token in sub.instantiated_prompt.lower():
            return True
    return False


async def run_hybrid_task(
    runner: Runner, task: str, spec: Any = None, marker: str | None = None, *,
    fresh: bool = False, redecompose: bool = False, reauthor: str | None = None,
) -> RunResult:
    """Execute `task` subtask-by-subtask: replay what the library knows, author the rest.

    Per subtask: resolve its library id from (template prompt, live page context); replay a
    hit (healing included); on replay failure hand the same live page to the agent for
    in-place recovery; on a miss author it with a scoped agent run and commit the segment to
    the library. A failed segment breaks the loop — later subtasks depend on its state.

    Task verdict = every segment ok AND (when the parent has a marker) the create-write seen
    anywhere in the whole run's network log — the parent gate's semantics are unchanged.
    """
    tid = sstore.task_id(task)
    subtasks = await get_decomposition(task, llm=runner.expander_llm, spec=spec,
                                       marker=marker, redecompose=redecompose)
    logger.info("▶ HYBRID %s: %d subtask(s)", tid, len(subtasks))
    hs = await HybridSession.open(runner)
    segments: list[Segment] = []
    completed: list[str] = []
    findings: list[str] = []        # "prompt: observation" lines, fed to later segments
    try:
        for i, sub in enumerate(subtasks):
            context = sstore.normalize_context(await hs.current_url())
            sid = sstore.subtask_id(sub.template_prompt, context)
            force_author = reauthor_match(reauthor, sub)
            is_judge = getattr(sub, "kind", "action") == "judge"

            # Semantic routing: a wording with NO direct entry may still be a known
            # procedure (alias table -> local embeddings -> one LLM verify). A routed sid
            # replays the canonical skill with values re-keyed to its params; authoring
            # after a routed failure goes under the ORIGINAL sid so the canonical entry
            # is never overwritten by a different wording's recording.
            author_sid, route_values = sid, None
            if (not fresh and not force_author and not is_judge
                    and not sstore.has_script(sid)
                    and getattr(runner.config, "semantic_router", False)):
                routed = await router.route(
                    sub, sid, context, runner.expander_llm,
                    model_name=getattr(runner.config, "embedding_model",
                                       router._DEFAULT_MODEL))
                if routed is not None:
                    print(f"[*] subtask {i}: wording routed to library entry "
                          f"[{routed.sid}] via {routed.via}")
                    sid, route_values = routed.sid, routed.values

            entry = sstore.load_manifest().get(sid)
            gate = segment_gate(sub, entry, context)
            remaining = [s.instantiated_prompt for s in subtasks[i + 1:]]
            seg: Segment | None = None

            # Judge nodes never touch the library in EITHER direction: a replayed judge
            # would click through with nobody looking (hollow pass), and its recording
            # must never be committed for the same reason.
            if not fresh and not force_author and not is_judge and sstore.has_script(sid):
                load_sub = sub if route_values is None else SimpleNamespace(
                    instantiated_prompt=sub.instantiated_prompt, values=route_values)
                skill = skills.load_skill(sid, load_sub)
                if skill is not None:
                    print(f"[*] subtask {i} [{sid}]: library hit -> replay "
                          f"({len(skill)} steps, no LLM)")
                    seg = await hs.replay_segment(sub, sid, context, skill, gate)
                    if seg.ok:
                        from_template = bool((entry or {}).get("params"))
                        _promote_segment_heals(sid, seg, from_template=from_template)
                        sstore.bump_meta(sid, uses=1)
                    else:
                        sstore.bump_meta(sid, fail_count=1)
                        dirty = seg.steps_executed > 0
                        print(f"[*] subtask {i} [{sid}]: replay FAILED "
                              f"({seg.error}) -> agent takes over in place"
                              f"{' (dirty state)' if dirty else ''}")
                        if not dirty and route_values is None:
                            # Failed before touching the page: the recovery run starts from
                            # the entry's declared context, so it IS a clean re-author.
                            # (A ROUTED failure never archives the canonical entry — the
                            # wording mapping may be at fault, not the recording.)
                            sstore.archive_entry(sid)
                        prior = seg.error
                        seg = await _author_segment(
                            hs, sub, author_sid, context, gate, completed=completed,
                            remaining=remaining, dirty=dirty, prior_failure=prior,
                            findings=findings)
                        seg.mode = "replay_failed->authored"
                else:
                    print(f"[*] subtask {i} [{sid}]: library hit but values did not "
                          f"resolve -> authoring")

            if seg is None:
                if force_author and sstore.has_script(sid):
                    # The existing entry stays as the gate's end_context reference and is
                    # only overwritten if the fresh authoring passes its gate.
                    print(f"[*] subtask {i} [{sid}]: --reauthor -> authoring with the "
                          f"agent (entry replaced only on success)")
                elif is_judge:
                    print(f"[*] subtask {i} [{sid}]: judge node (verification) -> agent "
                          f"runs it live, never cached")
                elif not sstore.has_script(sid):
                    print(f"[*] subtask {i} [{sid}]: no library entry -> authoring "
                          f"with the agent")
                seg = await _author_segment(hs, sub, author_sid, context, gate,
                                            completed=completed, remaining=remaining,
                                            findings=findings, commit=not is_judge)

            segments.append(seg)
            if not seg.ok:
                print(f"[*] subtask {i} [{sid}]: FAILED ({seg.error}) -> stopping task "
                      f"(later subtasks depend on this state)")
                break
            completed.append(sub.instantiated_prompt)
            if seg.finding:
                findings.append(f"{sub.instantiated_prompt[:80]}: {seg.finding}")
    finally:
        result = await hs.finalize(task, marker)

    result.subtasks = [s.as_dict() for s in segments]
    result.mode = "hybrid"
    all_ok = bool(segments) and all(s.ok for s in segments) \
        and len(segments) == len(subtasks)
    if marker:
        gt = result.ground_truth or {}
        result.is_successful = all_ok and bool(gt.get("create_write_seen"))
    else:
        result.is_successful = all_ok
    result.has_errors = not result.is_successful
    result.n_steps = sum(s.steps_executed for s in segments)
    total_tokens = sum(s.tokens for s in segments)
    total_cost = sum(s.cost for s in segments)
    if total_tokens:
        result.usage = {"total_tokens": total_tokens, "total_cost": total_cost}
    replayed = sum(1 for s in segments if s.mode == "replay")
    authored = len(segments) - replayed
    result.final_result = (
        f"Hybrid run: {len(segments)}/{len(subtasks)} subtasks "
        f"({replayed} replayed, {authored} authored)"
        + ("" if all_ok else f" — FAILED at subtask {len(segments) - 1}")
    )

    logger.info("◀ HYBRID done %s: success=%s %s", tid, result.is_successful,
                result.final_result)
    return result
