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
a hollow pass). "loop" nodes repeat an action until a stated stop condition holds ("Save &
Next ... until X is shown"): they get ACTION framing plus an extended step budget — judge's
observation framing made the agent declare a loop done after one iteration — but are just
as uncacheable, because the iteration count is live page state. A leading-"If" conditional
guard (decompose.is_conditional_guard) also always runs live and uncommitted: a recording
could only capture ONE branch. Observations flow FORWARD: each completed segment's finding
(its distilled final result) is handed to every later agent segment, so a note-then-verify
task can compare against what was actually observed instead of guessing. A segment that CONSUMES such
observations ("add employee using the noted generated name") is the third routing rule:
once this run has findings, it always runs as an agent segment — a cached replay could only
type the authoring run's stale values — and is never committed (see the dynamic-input gate
in run_hybrid_task).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

from playwright.async_api import Page

from automation import skills
from automation.browser.recording import start_run_recording, stop_run_recording
from automation.pipeline import adapt
from automation.pipeline import router
from automation.pipeline import subtask_store as sstore
from automation.pipeline.checks import (business_writes, evaluate_checks,
                                        receipt_rollup, save_cue,
                                        window_write_rollup)
from automation.pipeline.decompose import (Subtask, consumes_noted_data, downloads_file,
                                           get_decomposition, is_conditional_guard)
from automation.pipeline.prompts import scoped_subtask_prompt
from automation.pipeline.runner import RunResult, Runner, _first_create_write
from automation.pipeline.script_compile import (REVEAL_CSS_JS, _atomic_write, _esc,
                                                _names_value, merge_extract,
                                                promote_healed, save_steps)

logger = logging.getLogger("framework.hybrid")

# A library entry that fails this many CONSECUTIVE replays is stale by definition: archive
# it so the next run authors a clean replacement (see subtask_store.archive_if_failing).
_ARCHIVE_AFTER_FAILURES = 2

# Ad/tracker hosts blocked in HELPER TABS ONLY (open_aux_tab): foreign sites the aux
# machinery visits are ad-saturated (fakenamegenerator's ad iframes pushed every DOM
# snapshot to 15-30s+ and timed out the watchdogs), and an aux tab exists to read one
# fact, never to render ads. The app tab is untouched. Keep this list to PURE ad/tracking
# domains — NEVER add CMP/consent hosts (cookielaw.org, consensu.org, consentmanager.net):
# recordings legitimately click the consent banner, which must keep appearing.
_AUX_BLOCKED_HOSTS = frozenset({
    "adnxs.com",
    "adsafeprotected.com",
    "adservice.google.com",
    "amazon-adsystem.com",
    "casalemedia.com",
    "criteo.com",
    "doubleclick.net",
    "google-analytics.com",
    "googleadservices.com",
    "googlesyndication.com",
    "googletagservices.com",
    "openx.net",
    "outbrain.com",
    "pubmatic.com",
    "rubiconproject.com",
    "taboola.com",
})


def _is_blocked_ad_host(url: str) -> bool:
    """True when `url`'s host is (or is a subdomain of) a blocked ad/tracker domain."""
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return False
    return any(host == h or host.endswith("." + h) for h in _AUX_BLOCKED_HOSTS)


async def _abort_ad_requests(route) -> None:
    """page.route handler for aux tabs: abort ad/tracker requests, pass everything else.
    A handler that neither aborts nor continues would hang its request, so every path
    falls back to continue_."""
    try:
        if _is_blocked_ad_host(route.request.url):
            logger.debug("aux tab blocked ad request: %s", route.request.url[:120])
            await route.abort()
            return
    except Exception:  # noqa: BLE001 - blocking is best-effort sugar
        pass
    try:
        await route.continue_()
    except Exception as exc:  # noqa: BLE001 - request likely gone (tab closing)
        logger.debug("aux tab route continue_ failed: %s", exc)


# ------------------------------- gates -------------------------------


@dataclass
class Gate:
    """How one segment's success is judged (marker > postcondition > download > steps).
    `checks` are the subtask's declared deterministic checks (pipeline/checks.py): they
    ride along on WHATEVER base kind resolves and evaluate on top of it, demote-only."""

    kind: str                       # "marker" | "postcondition" | "download" | "steps"
    marker: str | None = None       # kind == "marker": create-write URL fragment
    postcondition: dict[str, Any] | None = None   # {"url_contains": ...} | {"visible": ...}
    end_context: str | None = None  # recorded end context (normalized URL) to compare
    checks: tuple = ()              # declared verify checks (tuple of checks.Check)


def segment_gate(sub: Subtask, entry: dict[str, Any] | None, context: str) -> Gate:
    """Resolve the gate for a subtask (see _base_gate for the kind precedence), then
    attach the subtask's declared verify checks — they apply to every kind."""
    gate = _base_gate(sub, entry, context)
    gate.checks = tuple(getattr(sub, "verify", None) or ())
    return gate


def _base_gate(sub: Subtask, entry: dict[str, Any] | None, context: str) -> Gate:
    """Resolve the base gate for a subtask: its marker (the save-owning segment) wins;
    else a declared postcondition; else the DOWNLOAD gate when the wording says the
    segment downloads/exports a file (its truth is "a file arrived", not the page state —
    and not the agent's self-report, which a download click's inevitable timeout receipt
    poisons); else the library entry's recorded end_context when it differs from the
    start context (a navigation segment must actually land somewhere); else the steps
    floor (clean execution / agent self-report)."""
    if sub.marker:
        return Gate(kind="marker", marker=sub.marker)
    if sub.postcondition:
        return Gate(kind="postcondition", postcondition=sub.postcondition)
    if getattr(sub, "fallback", False):
        # A whole-prompt fallback blob mentions "download" mid-task, but its deliverable
        # is the WHOLE task — the download gate would also inject the FILE DOWNLOAD
        # prompt block at the top, which sent the blob runs hunting a Download control
        # from step 1. The blob's arbiter is the run-level ground truth; degrade to steps.
        return Gate(kind="steps")
    if downloads_file(sub.template_prompt):
        return Gate(kind="download")
    end_context = (entry or {}).get("end_context")
    if end_context and end_context != context:
        return Gate(kind="postcondition", end_context=end_context)
    return Gate(kind="steps")


def _describe_check(check: Any) -> str:
    """One declared check as agent-actionable words. Wording order and content mirror
    what evaluate_gate will enforce, same convention as the postcondition branch."""
    return {
        "text_visible": f'the text "{check.arg}" visible on the page',
        "text_absent": f'the text "{check.arg}" no longer visible',
        "control_exists": f'a control named "{check.arg}" present',
        "url_contains": f'the page URL containing "{check.arg}"',
        "write_accepted": f'an accepted write to "{check.arg}"',
    }.get(check.kind, f'{check.kind} "{check.arg}"')


def _describe_expected_end(gate: Gate) -> str | None:
    """The gate's pass condition in words the agent can act on, or None when the gate has
    no page-state condition (marker gates verify via verify_save_registered instead, and
    bare steps gates have nothing to check). This is what turns the library's recorded
    outcome — and any declared verify checks — into the authoring agent's explicit
    done-condition."""
    base = None
    if gate.kind == "postcondition":
        # Branch order mirrors evaluate_gate exactly, so the condition described to the
        # agent is always the one the gate will enforce.
        if gate.postcondition and gate.postcondition.get("url_contains"):
            base = f'the page URL contains "{gate.postcondition["url_contains"]}"'
        elif gate.postcondition and gate.postcondition.get("visible"):
            base = f'the element matching "{gate.postcondition["visible"]}" is visible'
        elif gate.end_context:
            base = (f'the page URL path matches "{gate.end_context}" '
                    f'(lowercased; each "*" stands for a record id)')
    if not gate.checks:
        return base
    described = "; ".join(_describe_check(c) for c in gate.checks)
    line = f"mechanical verification will additionally require: {described}"
    return f"{base}; and {line}" if base else line


# Extra agent steps for the SAVE-OWNING segment. Its job is a loop — save, verify against
# the network, read the form's validation errors, fix exactly those fields, save again —
# and a flat budget starves that loop: observed live, the agent diagnosed the one blocking
# field at step 23/25 and had no room left to fix it. The form's own validation is the
# GENERAL mechanism for learning what a form requires (no per-task prompt tuning); this
# headroom is what lets the agent act on it. A clean save never uses the extra steps.
_MARKER_EXTRA_STEPS = 10


# Extra agent steps for a LOOP segment. Its job is inherently many iterations of the same
# action (observed live: 17 Save & Next advances to reach the named employee, and a
# successful fully-live pass needed 35 steps), and the flat budget starves it the same way
# it starved the save-verify loop. Sized so the observed worst pass fits with headroom for
# mid-loop dialogs and error branches; a short loop never uses the extra steps.
_LOOP_EXTRA_STEPS = 35


# Extra agent steps for the whole-prompt FALLBACK blob: one segment must cover the entire
# task (a mega-task blob at the flat 25-step base would die a third of the way in).
_FALLBACK_EXTRA_STEPS = 75


def segment_step_budget(gate: Gate, base: int, kind: str = "action",
                        fallback: bool = False) -> int:
    """Max agent steps for one segment: the configured base, plus fix-and-resave headroom
    when the segment owns the marker (its save must be verified and possibly repaired),
    plus repeat-until headroom when the node is a loop (one budget must cover every
    iteration), plus whole-task headroom for a fallback blob (the segment IS the task)."""
    return (int(base)
            + (_MARKER_EXTRA_STEPS if gate.kind == "marker" else 0)
            + (_LOOP_EXTRA_STEPS if kind == "loop" else 0)
            + (_FALLBACK_EXTRA_STEPS if fallback else 0))


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


def _rollup_applies(gate: Gate) -> bool:
    """Receipt roll-up guards only the gate kinds that TRUST self-report. Marker and
    download verdicts are already deterministic in both directions — rolling receipts
    into them would false-fail the proven-save-then-refused-duplicate pattern (an
    "already submitted" bounce after a marker-proven save is completion, not failure)."""
    return gate.kind in ("steps", "postcondition")


async def evaluate_gate(
    gate: Gate, *, steps_ok: bool, page: Page | None,
    requests_window: list[dict[str, Any]],
    downloads_window: list[str] | None = None,
    rollup: tuple[bool, list[str]] | None = None,
    prompt_text: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate a segment gate: the base kind (see _evaluate_base_gate), then the
    receipt roll-up verdict (computed by the agent call site, None on replay), then the
    window write rule (window_write_rollup — every segment, agent or replay, declared
    or ad-hoc: fired-but-never-accepted business writes fail the segment), then any
    declared verify checks. Roll-up, write rule, and checks are demote-only — they can
    fail a segment the base gate passed but never resurrect a failed one; checks on a
    failed base still get their single honest evaluation so the detail lands in the
    report (the same convention the settle window applies to a failed-steps
    postcondition). `detail["checks"]`/`detail["rollup"]`/`detail["write_rollup"]`
    appear only when declared/failing — a bare gate's detail is byte-identical to
    before. `prompt_text` (the subtask wording) feeds only the report-only
    `detail["write_warning"]`: a save-cue with zero observed business writes flags a
    possible silent-save blind spot without ever affecting `ok`."""
    ok, detail = await _evaluate_base_gate(
        gate, steps_ok=steps_ok, page=page, requests_window=requests_window,
        downloads_window=downloads_window)
    if rollup is not None and not rollup[0]:
        detail["rollup"] = rollup[1]
        ok = False
    wr_ok, wr_reasons = window_write_rollup(requests_window)
    if not wr_ok:
        detail["write_rollup"] = wr_reasons
        ok = False
    if prompt_text and save_cue(prompt_text) and not business_writes(requests_window):
        detail["write_warning"] = (
            "wording implies a save but no write request was observed — this app may "
            "save without network traffic; only a declared verify: check can see such "
            "a save")
    if gate.checks:
        results = await evaluate_checks(page, requests_window, gate.checks, poll=ok)
        detail["checks"] = results
        ok = ok and all(r["ok"] for r in results)
    return ok, detail


def _check_failure_reason(detail: dict[str, Any]) -> str | None:
    """The first failing declared check as a one-line human verdict, or None. This is
    what a segment's error should lead with — the deterministic reason, not the agent's
    happy final text."""
    for r in detail.get("checks") or []:
        if not r.get("ok"):
            why = r.get("error") or "not satisfied"
            if r.get("evidence"):
                why = f"{why} [{r['evidence']}]"
            return f'deterministic check failed: {r["kind"]} "{r["arg"]}" — {why}'
    wrollup = detail.get("write_rollup") or []
    if wrollup:
        return str(wrollup[0])
    rollup = detail.get("rollup") or []
    if rollup:
        return str(rollup[0])
    return None


async def _evaluate_base_gate(
    gate: Gate, *, steps_ok: bool, page: Page | None,
    requests_window: list[dict[str, Any]],
    downloads_window: list[str] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate a gate's BASE kind. Returns (ok, detail).

    kind == "marker" is authoritative: the segment passed iff its network window carries a
    successful create-write to the marker — exactly the whole-task ground-truth rule, scoped
    to this segment's traffic. kind == "download" is authoritative the same way: the
    segment passed iff a file arrived in its window, in BOTH directions — an agent that
    gave up after the file landed still passes (observed live: 4 successful downloads then
    an honest-but-wrong failure report), and a claimed success with no file fails. The
    other kinds additionally require steps_ok.
    """
    if gate.kind == "marker":
        write = _first_create_write(requests_window, gate.marker or "")
        return write is not None, {
            "kind": "marker", "marker": gate.marker,
            "create_write_seen": write is not None,
            "write_step": write.get("step") if write else None,
        }
    if gate.kind == "download":
        files = list(downloads_window or [])
        return bool(files), {"kind": "download", "files": files}
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
                    # Aux-tab contexts are host-qualified (they start with a hostname,
                    # main contexts with "/") — pick the matching normalizer.
                    normalize = (sstore.normalize_context
                                 if gate.end_context.startswith("/")
                                 else sstore.normalize_aux_context)
                    detail["reached"] = normalize(page.url if page is not None else "")
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
    kind: str = "action"            # composition node kind: "action" | "judge" | "loop"
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
    # Values captured by extract_data / replayed extract steps ({label: text}) — the fresh
    # data an aux-tab segment exists to fetch; also folded into `finding`.
    extracted: dict[str, str] | None = None
    # Files downloaded during this segment's window (basenames; the files live in the
    # run's artifacts downloads/ folder).
    downloads: list[str] = field(default_factory=list)
    # Why this segment did NOT replay (None on replays): "fresh" | "reauthor" | "judge" |
    # "loop" | "conditional" | "dynamic" | "no_entry" | "identity_fork" |
    # "values_unresolved". The answer to "why didn't it use the recording?" without
    # archaeology — surfaced in the report row and the run summary.
    skip_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "sid": self.sid, "prompt": self.prompt,
            "context": self.context, "mode": self.mode, "kind": self.kind, "ok": self.ok,
            "gate": self.gate, "steps_executed": self.steps_executed,
            "duration_seconds": round(self.duration_seconds, 1),
            "healed_steps": self.healed_steps, "tokens": self.tokens, "error": self.error,
            "finding": self.finding, "extracted": self.extracted,
            "downloads": self.downloads, "skip_reason": self.skip_reason,
        }


def _write_progress(hs: "HybridSession", *, task: str, tid: str, subtasks: list[Any],
                    segments: list[Segment], status: str,
                    is_successful: bool | None = None) -> None:
    """Persist the run's machine-readable state so far to <run_dir>/progress.json.

    Called at every segment boundary: report.json exists only after a CLEAN finish (it is
    assembled after finalize and written by __main__), so a crashed or killed run used to
    leave no per-segment record at all — no gates, no modes, no skip_reason breakdown to do
    forensics on. progress.json is that evidence, updated as the run advances; the final
    write flips status to "finished". The collectors flush at the same boundary so
    network.json/console.json survive a hard kill as of the last completed segment.
    Persistence is best-effort by rule: an evidence write must never break the run.
    """
    try:
        payload = {
            "run_id": hs.run_id,
            "task": task,
            "task_id": tid,
            "status": status,
            "started": hs.started.isoformat(timespec="seconds"),
            "updated": datetime.now().isoformat(timespec="seconds"),
            "subtasks_total": len(subtasks),
            "planned": [{"prompt": s.template_prompt, "kind": getattr(s, "kind", "action")}
                        for s in subtasks],
            "segments": [s.as_dict() for s in segments],
            "is_successful": is_successful,
        }
        _atomic_write(hs.run_dir / "progress.json",
                      json.dumps(payload, indent=2, default=str))
    except Exception as exc:  # noqa: BLE001 - evidence write must never break the run
        logger.exception("progress.json write failed: %s", exc)
    for collector in getattr(hs, "collectors", None) or []:
        try:
            collector.write()
        except Exception as exc:  # noqa: BLE001 - same rule as above
            logger.exception("collector %s mid-run flush failed: %s",
                             getattr(collector, "name", "?"), exc)


# ------------------------------- the shared session -------------------------------


class HybridSession:
    """One task's live execution surface: a single BrowserSession (agent driver), a single
    Playwright CDP connection (replay driver + telemetry), and the collectors — all opened
    once and torn down only in finalize(). Both drivers point at the same Chromium. The
    shared page state is explicit: the MAIN page is pinned at open() and never navigated
    away by aux work; an optional AUX page (helper tab, `tab_url` subtasks) exists only
    while its subtask runs and is what current_page() returns while it is alive."""

    def __init__(self, runner: Runner) -> None:
        self.runner = runner
        self.run_id: str = ""
        self.run_dir: Path = Path(".")
        self.session: Any = None
        self.pw_browser: Any = None
        self.collectors: list[Any] = []
        self.started = datetime.now()
        self.video_path: Path | None = None
        self._main_page: Page | None = None
        self._aux_page: Page | None = None

    @classmethod
    async def open(cls, runner: Runner) -> "HybridSession":
        hs = cls(runner)
        hs.run_id, hs.run_dir = runner._new_run_dir()
        # INVERTED handoff: the browser-OWNING session was launched, started, and logged
        # in by main() — reuse it (starting a second session would launch a second,
        # unauthenticated browser).
        hs.session = runner.session
        if hs.session is None:
            raise RuntimeError("Runner has no browser session; main() must launch and "
                               "log in before running tasks")
        hs.pw_browser = await runner.playwright.chromium.connect_over_cdp(runner.cdp_url)
        if runner.config.reveal_hidden_controls:
            # Durable reveal-CSS: an init script lives only as long as the Playwright
            # connection that installed it, and login's connection closes after
            # authenticating — THIS connection spans the whole run, so install here.
            # (Per-step and per-segment re-asserts remain the backstop.)
            for ctx in hs.pw_browser.contexts:
                try:
                    await ctx.add_init_script(REVEAL_CSS_JS)
                except Exception as exc:  # noqa: BLE001 - re-asserts cover a miss
                    logger.debug("reveal init script install failed: %s", exc)
        # Route downloads to the run's artifacts with their REAL suggested filenames
        # (behavior "allow"). Two layers fight us here, both observed live: browser-use
        # classifies a CDP-attached session as REMOTE and skips its own
        # setDownloadBehavior setup, and Playwright's launch-time behavior (self-deleting
        # temp dir, UUID names) is set PER BROWSER CONTEXT — a context-scoped setting
        # beats a browser-wide default, so the override must be asserted for EVERY
        # existing context as well as the default. eventsEnabled keeps browser-use's
        # download TRACKING alive (that part does work for remote sessions).
        try:
            cdp = await hs.pw_browser.new_browser_cdp_session()
            ctx_ids = (await cdp.send("Target.getBrowserContexts")).get(
                "browserContextIds") or []
            for ctx_id in [None, *ctx_ids]:
                params: dict[str, Any] = {
                    "behavior": "allow",
                    "downloadPath": str((hs.run_dir / "downloads").resolve()),
                    "eventsEnabled": True,
                }
                if ctx_id:
                    params["browserContextId"] = ctx_id
                await cdp.send("Browser.setDownloadBehavior", params)
        except Exception as exc:  # noqa: BLE001 - downloads degrade, the run must not die
            logger.warning("could not route downloads to the run dir: %s", exc)
        hs.collectors = await runner._start_collectors(hs.pw_browser, hs.run_dir)
        hs._main_page = hs._pick_main_page()
        if runner.config.record_video:
            # AFTER the main page is pinned: the recorder detects its frame size ONCE, from
            # the live viewport, and keeps it for the whole run — so let a real page be up
            # first. Failure here returns None and the run continues without a video.
            hs.video_path = await start_run_recording(
                hs.session, hs.run_dir / "run.mp4", size=runner.config.record_video_size)
        hs.started = datetime.now()
        return hs

    def _pick_main_page(self) -> Page | None:
        """First non-blank page across contexts — pins the main page at open() and re-pins
        if the pinned page ever dies."""
        pages = [p for ctx in self.pw_browser.contexts for p in ctx.pages]
        if not pages:
            return None
        real = [p for p in pages if p.url != "about:blank"]
        return real[0] if real else pages[0]

    def current_page(self) -> Page | None:
        """The page segments run against: the live aux (helper) tab while one is open, else
        the pinned main page. Explicit handles, not a scan — with two live tabs a scan
        could not tell which one the segment means."""
        for page in (self._aux_page, self._main_page):
            if page is not None and not page.is_closed():
                return page
        self._main_page = self._pick_main_page()
        return self._main_page

    async def current_url(self) -> str:
        page = self.current_page()
        return page.url if page is not None else ""

    async def close_extra_tabs(self) -> None:
        """Re-enforce the tab invariant after an agent segment (a misclick can open a new
        tab, and the next segment's page pick must stay deterministic): every page except
        the pinned main page and the live aux tab is closed."""
        try:
            keep = {p for p in (self._main_page, self._aux_page)
                    if p is not None and not p.is_closed()}
            if not keep:
                keep = {self.current_page()}
            pages = [p for ctx in self.pw_browser.contexts for p in ctx.pages]
            for p in pages:
                if p not in keep:
                    await p.close()
        except Exception as exc:  # noqa: BLE001 - best-effort hygiene
            logger.debug("close_extra_tabs: %s", exc)

    async def _page_target_id(self, page: Page) -> str | None:
        """CDP target id of a Playwright page — the identity browser-use tracks tabs by."""
        try:
            cdp = await page.context.new_cdp_session(page)
            try:
                info = await cdp.send("Target.getTargetInfo")
                return (info.get("targetInfo") or {}).get("targetId")
            finally:
                await cdp.detach()
        except Exception as exc:  # noqa: BLE001 - focus alignment is best-effort
            logger.debug("could not read page target id: %s", exc)
            return None

    async def _focus_browser_use(self, page: Page | None) -> None:
        """Point browser-use's agent focus at `page` (best-effort, never raises).

        The two drivers track tabs independently: Playwright opening a page does not move
        browser-use's agent_focus_target_id, and an agent segment started on the wrong
        focus would act (and screenshot) the wrong tab. Dispatching SwitchTabEvent with
        the page's CDP target id aligns them. browser-use's SessionManager can lag a
        freshly created target by ~50ms, hence the short retries; a None target_id means
        "most recently opened" — an acceptable last resort, since the aux tab is always
        the newest page and after close_aux_tab only the main page remains."""
        if page is None or self.session is None:
            return
        try:
            from browser_use.browser.events import SwitchTabEvent
        except Exception as exc:  # noqa: BLE001 - focus alignment is best-effort
            logger.debug("SwitchTabEvent unavailable: %s", exc)
            return
        for attempt in (1, 2, 3):
            try:
                tid = await self._page_target_id(page)
                await self.session.event_bus.dispatch(SwitchTabEvent(target_id=tid))
                return
            except Exception as exc:  # noqa: BLE001
                logger.debug("browser-use focus attempt %d failed: %s", attempt, exc)
                await asyncio.sleep(0.2)

    async def open_aux_tab(self, url: str) -> Page:
        """Ensure THE helper tab for the current subtask is open, fronted, and focused.

        Idempotent within a subtask: a live aux tab is only re-fronted — NO re-goto, so
        after a failed replay the recovering agent inspects the dirty state, not a fresh
        page. The tab is opened BY the main page (window.open), so headful Chrome docks
        it as a sibling tab in the main page's own window and it lives in the same
        context — inheriting the reveal-CSS init script and the telemetry collectors'
        context-level page hook."""
        if self._aux_page is not None and not self._aux_page.is_closed():
            await self._aux_page.bring_to_front()
            await self._focus_browser_use(self._aux_page)
            return self._aux_page
        self._aux_page = None
        main = self.current_page()
        if main is None:
            raise RuntimeError("no open page; cannot host a helper tab")
        try:
            # A page-opened popup lands as a TAB in the main page's window, where a
            # CDP-created page (context.new_page -> Target.createTarget) becomes a
            # separate WINDOW in headful Chrome. Playwright launches Chromium with
            # --disable-popup-blocking, so the gesture-less window.open never blocks.
            async with main.context.expect_page(timeout=5_000) as new_page_info:
                await main.evaluate("window.open('about:blank', '_blank')")
            page = await new_page_info.value
        except Exception as exc:  # noqa: BLE001 - placement is sugar; never fail the subtask on it
            logger.debug("window.open helper-tab path failed (%s); falling back to "
                         "context.new_page (may open as a separate window)", exc)
            page = await main.context.new_page()
        try:
            # Registered BEFORE the goto so the initial ad barrage never loads. Dies with
            # the page in close_aux_tab; the app tab never gets a route.
            await page.route("**/*", _abort_ad_requests)
            logger.info("aux tab: ad/tracker request blocking active (%d hosts; app tab "
                        "untouched)", len(_AUX_BLOCKED_HOSTS))
        except Exception as exc:  # noqa: BLE001 - blocking is best-effort sugar
            logger.debug("aux tab ad blocking unavailable: %s", exc)
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        except Exception:
            try:
                await page.close()
            except Exception as exc:  # noqa: BLE001 - the goto error is the one to raise
                logger.debug("aux page close after failed goto: %s", exc)
            raise
        self._aux_page = page
        await page.bring_to_front()
        await self._focus_browser_use(page)
        return page

    async def close_aux_tab(self) -> None:
        """Close the helper tab (if any) and hand focus back to the pinned main page."""
        page, self._aux_page = self._aux_page, None
        if page is not None and not page.is_closed():
            try:
                await page.close()
            except Exception as exc:  # noqa: BLE001 - best-effort hygiene
                logger.debug("close_aux_tab: %s", exc)
        main = self.current_page()
        if main is not None:
            try:
                await main.bring_to_front()
            except Exception as exc:  # noqa: BLE001 - best-effort hygiene
                logger.debug("close_aux_tab bring_to_front: %s", exc)
            await self._focus_browser_use(main)

    def downloads_watermark(self) -> int:
        """Count of files the session has downloaded so far — segments window from here."""
        try:
            return len(getattr(self.session, "downloaded_files", None) or [])
        except Exception:  # noqa: BLE001 - download accounting is best-effort
            return 0

    def downloads_since(self, watermark: int) -> list[str]:
        """Basenames of files downloaded after `watermark`. Safety net: a tracked file
        still sitting OUTSIDE the run's downloads dir (a browser temp dir that dies at
        teardown) is copied out while it exists — a downloaded artifact must never be
        lost again, even if some future browser layer re-routes the save location."""
        try:
            files = list(getattr(self.session, "downloaded_files", None) or [])
        except Exception:  # noqa: BLE001
            return []
        dl_dir = self.run_dir / "downloads"
        out: list[str] = []
        for p in files[watermark:]:
            src = Path(p)
            out.append(src.name)
            try:
                if src.is_file() and dl_dir.resolve() not in src.resolve().parents:
                    dl_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dl_dir / src.name)
                    logger.info("⬇ copied download out of temp: %s", src.name)
            except OSError as exc:
                logger.debug("download copy failed for %s: %s", src, exc)
        return out

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
        # Before the browser goes: stopping the screencast and flushing the encoder both
        # need the live CDP connection. Unconditional — with no recording running,
        # browser-use returns None.
        if self.video_path is not None:
            saved = await stop_run_recording(self.session)
            if saved is not None:
                artifacts["video"] = saved
        try:
            await self.pw_browser.close()
        except Exception as exc:  # noqa: BLE001
            logger.exception("playwright connection close error: %s", exc)
        # The BrowserSession is PROCESS-owned (launched and killed by main), not
        # run-owned: stopping it here would tear down the watchdogs/event bus of the
        # browser the next segment — and main's teardown — still expect to control.

        duration = (datetime.now() - self.started).total_seconds()
        ground_truth: dict[str, Any] | None = None
        if parent_marker:
            requests = collector_results.get("network", {}).get("requests", []) or []
            write = _first_create_write(requests, parent_marker)
            ground_truth = {
                "marker": parent_marker,
                "create_write_seen": write is not None,
                "write_step": write.get("step") if write else None,
            }
        return RunResult(
            task=task, run_id=self.run_id, artifacts_dir=self.run_dir,
            is_done=True, is_successful=None, has_errors=False, final_result=None,
            urls=[], n_steps=0, duration_seconds=duration, extracted_content=[],
            model_actions=[], errors=[], collector_results=collector_results,
            artifacts=artifacts, screenshots=[], steps=[], usage=None,
            ground_truth=ground_truth,
        )

    # ------------------------------- segment execution -------------------------------

    async def replay_segment(
        self, sub: Subtask, sid: str, context: str, skill: skills.Skill, gate: Gate,
    ) -> Segment:
        """Replay a library skill on the live page. No LLM."""
        started = datetime.now()
        watermark = self.network_watermark()
        dl_mark = self.downloads_watermark()
        page = self.current_page()
        seg = Segment(index=sub.index, sid=sid, prompt=sub.instantiated_prompt,
                      context=context, mode="replay")
        if page is None:
            seg.error = "no open page to replay against"
            return seg
        if self.runner.config.reveal_hidden_controls:
            # Replay-only runs never execute an agent step, so re-assert the reveal
            # stylesheet here (idempotent; normally a no-op — login.py's init script already
            # covered this document). Keeps 0-size targets resolvable by _resolve's
            # visibility gate even after a rare hard navigation mid-replay.
            try:
                await page.evaluate(REVEAL_CSS_JS)
            except Exception as exc:  # noqa: BLE001 - best-effort; replay proceeds anyway
                logger.debug("reveal css injection skipped: %s", exc)
        outcome = await skills.execute(skill, page)
        if gate.kind == "marker" and gate.marker:
            await self.wait_for_inflight_write(gate.marker)
        seg.replay = outcome
        seg.steps_executed = outcome.get("executed", 0)
        seg.error = outcome.get("error")
        seg.healed_steps = [e["step"] for e in (outcome.get("log") or []) if e.get("healed")]
        # Replayed extract steps re-read the live DOM, so a zero-LLM replay still yields a
        # FRESH observation — folded into `finding` so the loop carries it forward exactly
        # like an agent segment's.
        seg.extracted = outcome.get("extracted") or None
        if seg.extracted:
            # Cap sized for block captures: one extract on the identity card carries
            # several facts, and truncating it would silently drop the later ones.
            seg.finding = _format_extracts(seg.extracted)[:800] or None
        steps_ok = outcome.get("failed_at") is None
        seg.downloads = self.downloads_since(dl_mark)
        seg.ok, seg.gate = await evaluate_gate(
            gate, steps_ok=steps_ok, page=self.current_page(),
            requests_window=self.requests_since(watermark),
            downloads_window=seg.downloads,
            prompt_text=sub.instantiated_prompt,
        )
        if steps_ok and not seg.ok:
            seg.error = (seg.error or _check_failure_reason(seg.gate)
                         or f"segment gate failed: {seg.gate}")
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
        dl_mark = self.downloads_watermark()
        kind = getattr(sub, "kind", "action")
        prompt = scoped_subtask_prompt(
            sub.instantiated_prompt, completed, remaining, dirty, prior_failure,
            expected_end=_describe_expected_end(gate),
            owns_save=gate.kind == "marker",
            downloads_file=gate.kind == "download",
            findings=findings, observe=kind == "judge", loop=kind == "loop",
            conditional=is_conditional_guard(sub.template_prompt),
            aux_tab=getattr(sub, "tab_url", None))
        seg = Segment(index=sub.index, sid=sid, prompt=sub.instantiated_prompt,
                      context=context, mode="authored", kind=kind)
        try:
            out = await self.runner.run_agent_segment(
                prompt, self.session, self.collectors,
                max_steps=segment_step_budget(gate, self.runner.config.subtask_max_steps,
                                              kind,
                                              fallback=getattr(sub, "fallback", False)),
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
        seg.extracted = _history_extracts(history) or None
        seg.downloads = self.downloads_since(dl_mark)
        usage = out.get("usage") or {}
        seg.tokens = int(usage.get("total_tokens") or 0)
        seg.cost = float(usage.get("total_cost") or 0.0)
        if gate.kind == "marker" and gate.marker:
            await self.wait_for_inflight_write(gate.marker)
        steps_ok = bool(history.is_successful())
        rollup = (receipt_rollup(history, self.requests_since(watermark))
                  if _rollup_applies(gate) else None)
        seg.ok, seg.gate = await evaluate_gate(
            gate, steps_ok=steps_ok, page=self.current_page(),
            requests_window=self.requests_since(watermark),
            downloads_window=seg.downloads,
            rollup=rollup,
            prompt_text=sub.instantiated_prompt,
        )
        seg.write_step = seg.gate.get("write_step")
        if not seg.ok:
            if steps_ok:
                # The agent believed it succeeded but the gate disagreed: surface the gate
                # verdict — leading with the deterministic check reason when one failed —
                # not the agent's happy final text.
                core = _check_failure_reason(seg.gate) or f"gate failed: {seg.gate}"
                seg.error = (f"{core} "
                             f"(agent claimed success: {history.final_result()!r})")
            else:
                seg.error = (history.final_result() or _check_failure_reason(seg.gate)
                             or f"segment gate failed: {seg.gate}")
        else:
            # The segment's distilled observation, carried into later segments' context.
            final = " ".join(str(history.final_result() or "").split())
            if not final and seg.extracted:
                final = _format_extracts(seg.extracted)
            seg.finding = final[:800] or None
        seg.duration_seconds = (datetime.now() - started).total_seconds()
        return seg


def _format_extracts(extracted: dict[str, str]) -> str:
    """Extracts as '; '-joined 'label = value' pairs, with labels that resolved to the
    SAME text collapsed into one entry ('street / city / postcode = 75 Monks Way …'). A
    replayed extract re-reads whole DOM nodes, so several labels can land on one address
    blob — repeating it per label would burn the findings budget and dress the blob up
    as a real per-label split, misleading the agent that consumes the observation."""
    by_value: dict[str, list[str]] = {}
    for label, value in (extracted or {}).items():
        by_value.setdefault(value, []).append(label)
    return "; ".join(f"{' / '.join(labels)} = {value}"
                     for value, labels in by_value.items())


def _history_extracts(history: Any) -> dict[str, str]:
    """The extract_data captures in an agent history ({label: value}) — how an AUTHORED
    segment's fresh observations reach Segment.extracted before the recording is even
    compiled. Label collisions keep BOTH values (label_2, ... — merge_extract): 'later
    calls win' silently dropped the generated NAME when the address extract reused its
    label (observed live), starving every later consumer of the fact."""
    out: dict[str, str] = {}
    for item in getattr(history, "history", None) or []:
        for res in getattr(item, "result", None) or []:
            md = getattr(res, "metadata", None)
            ext = md.get("extract") if isinstance(md, dict) else None
            if isinstance(ext, dict) and ext.get("label") is not None:
                merge_extract(out, str(ext["label"]), str(ext.get("value") or ""))
    return out


# ------------------------------- commit / heal helpers -------------------------------


# Quoted name=/text= values inside compiled selectors — the names a click acts on.
_NAME_IN_SEL = re.compile(r'(?:name|text)="([^"]+)"')


def _findings_sourced_values(steps: list[dict[str, Any]], prompt: str,
                             findings: list[str] | None) -> list[str]:
    """Values this segment acted with that came from the RUN'S FINDINGS rather than its
    prompt — the provenance check that decides cacheability, independent of wording.

    A typed value or an acted-on element's NAME that is absent from the prompt but
    present in an earlier segment's findings is runtime data by construction: a cached
    replay would re-use THIS run's value forever (observed live three times: a generated
    employee identity replayed verbatim; an employee-name pick parameterized to the word
    "download" — the only prompt token the binder could reach; and a data-request ref
    created by the authoring run baked into a find_click, so every replay clicked the
    PREVIOUS run's real row and only the marker gate stopped a wrong-record commit).
    Prompt-sourced values cache fine; agent-invented incidentals (a title pick, a dummy
    county) also cache fine — only findings provenance marks the segment dynamic.

    Typed values (fill/select/type) are checked against the findings verbatim — prompt
    prefixes included, since re-typing another segment's wording is just as stale.
    CLICK-TARGET names (find_click text, name=/text= selector values) are checked against
    the finding BODIES only (entries are formatted "prompt[:80]: body" by the segment
    loop): a prior subtask's wording naming a menu ("Go to data request") must not
    un-cache an ordinary navigation click onto that menu.
    """
    finding_text = " \n ".join(findings or [])
    if not finding_text:
        return []
    finding_bodies = " \n ".join(f.split(": ", 1)[-1] for f in findings or [])
    out: list[str] = []
    for value, kind in _step_value_candidates(steps):
        value = value.strip()
        # Tiny click names ("OK", "1") collide with finding tokens by chance; a runtime
        # identifier is never that short. Typed values keep the historical no-floor rule.
        if kind == "click" and len(value) < 3:
            continue
        corpus = finding_text if kind == "typed" else finding_bodies
        if (value and value not in out
                and not _names_value(prompt, value)
                and _names_value(corpus, value)):
            out.append(value)
    return out


def _step_value_candidates(steps: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """(value, kind) pairs a segment acted with — kind "typed" (fill/select/type values)
    or "click" (find_click texts, click-target names). The one extraction both
    provenance checks (findings-sourced and body-sourced) scan."""
    out: list[tuple[str, str]] = []
    for step in steps:
        action = step.get("action")
        if action in ("fill", "select"):
            out.append((str(step.get("value") or ""), "typed"))
        elif action == "type":
            out.append((str(step.get("text") or ""), "typed"))
        elif action == "find_click":
            out.append((str(step.get("text") or ""), "click"))
        elif action == "click":
            names = [m.group(1) for s in step.get("selectors") or []
                     for m in _NAME_IN_SEL.finditer(s)]
            if step.get("expect_text"):
                names.append(str(step["expect_text"]))
            out.extend((name, "click") for name in names)
    return out


def _unattributed_typed_values(steps: list[dict[str, Any]], prompt: str,
                               flagged: list[str]) -> list[str]:
    """Typed values with NO provenance at all — absent from the prompt and not flagged
    for binding. On a CONSUMER segment (wording uses noted data) these are refusal-grade:
    a RE-FORMATTED runtime value ("October 25, 1971" typed as 25/10/1971) escapes the
    substring guard entirely, and a baked literal would write the authoring run's data
    into every later run's records (the DR021/DR022 wrong-record class). Only a commit
    that accounts for every typed value is honest. Length floor 3 skips micro-picks
    ("A", "Mr") that carry no identity."""
    out: list[str] = []
    for value, kind in _step_value_candidates(steps):
        value = value.strip()
        if kind != "typed" or len(value) < 3:
            continue
        if value in flagged or value in out or _names_value(prompt, value):
            continue
        out.append(value)
    return out


def _body_sourced_values(steps: list[dict[str, Any]], task_wording: str,
                         bodies: list[dict[str, Any]]) -> list[str]:
    """Step values that THIS RUN's create-write responses report (as a unique JSON leaf)
    and that no subtask's wording names — run-generated identifiers by construction.

    This closes the guard's findings blind spot (observed live 2026-07-24 evening: once
    the producer segment became a silent replay, the run's findings never named the
    fresh ref, the findings-relative guard saw nothing, and a stale ref was committed
    as a 'clean' literal again). The exclusion corpus is the WHOLE task's wording —
    values the task spells anywhere ("FOOD LIMITED", "payroll review") are prompt data
    even when they also echo in a response body. Length floor 4: short numerics ("22")
    collide with incidental response fields (autoNumber counters)."""
    if not bodies:
        return []
    out: list[str] = []
    for value, _kind in _step_value_candidates(steps):
        value = value.strip()
        if (len(value) >= 4 and value not in out
                and not _names_value(task_wording, value)
                and any(learn_json_path(r.get("body") or "", value) for r in bodies)):
            out.append(value)
    return out


# ------------------------------- runtime bindings -------------------------------
#
# The tier that makes RUN-GENERATED values replayable (user-approved 2026-07-24,
# superseding the always-LLM-for-consumers rule for values with a STRUCTURED source).
# A binding is an ordinary template param whose value comes from the run instead of the
# prompt: either an extract label this run captured, or a JSON path into a create-write
# response this run fired. Resolution happens at skill-load time; an unresolvable
# binding refuses the replay and the agent authors — never a stale literal.


def _json_leaf_paths(node: Any, want: str, path: tuple = ()) -> list[list[Any]]:
    """Every path in a parsed JSON tree whose scalar leaf prints as `want`."""
    out: list[list[Any]] = []
    if isinstance(node, dict):
        for k, v in node.items():
            out.extend(_json_leaf_paths(v, want, path + (k,)))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out.extend(_json_leaf_paths(v, want, path + (i,)))
    elif isinstance(node, (str, int, float)) and str(node).strip() == want:
        out.append(list(path))
    return out


def learn_json_path(body: str, value: str) -> list[Any] | None:
    """The UNIQUE path to `value` in a JSON body, or None. One authoring example is
    enough to learn where the app reports a created record's identifier; a non-unique
    match refuses (same philosophy as _resolve: never guess among candidates)."""
    try:
        tree = json.loads(body)
    except Exception:  # noqa: BLE001 - not JSON: nothing to learn
        return None
    paths = _json_leaf_paths(tree, value.strip())
    return paths[0] if len(paths) == 1 else None


def apply_json_path(body: str, path: list[Any]) -> str | None:
    """Resolve a learned path against THIS run's response body ('' / miss -> None)."""
    try:
        node: Any = json.loads(body)
        for key in path:
            node = node[key]
    except Exception:  # noqa: BLE001 - schema drift: the binding just doesn't resolve
        return None
    if isinstance(node, (str, int, float)):
        text = str(node).strip()
        return text or None
    return None


def _captured_write_bodies(hs: Any) -> list[dict[str, Any]]:
    """This run's create-write records that carry a captured response body, in firing
    order (the collector's list is chronological, so 'first create wins' falls out)."""
    net = next((c for c in getattr(hs, "collectors", []) or []
                if getattr(c, "name", "") == "network"), None)
    if net is None:
        return []
    try:
        requests = net.results().get("requests") or []
    except Exception:  # noqa: BLE001 - collector already stopped
        return []
    return [r for r in requests
            if r.get("body") and r.get("method") in ("POST", "PUT", "PATCH")]


def _binding_resolver(run_values: dict[str, str], hs: Any):
    """resolve(spec) -> concrete value from THIS run, or None (refuse the replay)."""
    def resolve(spec: dict[str, Any]) -> str | None:
        try:
            if spec.get("kind") == "extract":
                value = (run_values or {}).get(str(spec.get("label")))
                return str(value).strip() or None if value else None
            if spec.get("kind") == "created":
                for rec in _captured_write_bodies(hs):
                    if rec.get("method") != spec.get("method"):
                        continue
                    if sstore.normalize_context(str(rec.get("url") or "")) \
                            != spec.get("endpoint"):
                        continue
                    value = apply_json_path(rec.get("body") or "",
                                            spec.get("path") or [])
                    if value:
                        return value
        except Exception:  # noqa: BLE001 - resolution must never crash the loop
            return None
        return None
    return resolve


def _bind_runtime_values(
    steps: list[dict[str, Any]], values: list[str],
    extracts: dict[str, str], bodies: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, dict[str, Any]]] | None:
    """Rewrite the guard-flagged runtime literals into {{bound_N}} tokens, each bound to
    a structured source this run produced. Returns (steps, params, bindings), or None
    when ANY value has no structured source (prose-only observations stay agent-run).

    params carry the authoring run's literals as defaults — self-documenting, and the
    load path refuses to instantiate a bound entry without resolving them fresh, so a
    stale default can never replay. Longest value first so one flagged value embedded
    in another cannot corrupt the rewrite."""
    params: dict[str, str] = {}
    bindings: dict[str, dict[str, Any]] = {}
    replacements: list[tuple[str, str]] = []
    for n, value in enumerate(sorted(values, key=len, reverse=True), start=1):
        name = f"bound_{n}"
        label = next((lb for lb, v in (extracts or {}).items()
                      if str(v).strip() == value), None)
        spec: dict[str, Any] | None = None
        if label is not None:
            spec = {"kind": "extract", "label": label}
        else:
            for rec in bodies:
                path = learn_json_path(rec.get("body") or "", value)
                if path is not None:
                    spec = {"kind": "created", "method": rec.get("method"),
                            "endpoint": sstore.normalize_context(
                                str(rec.get("url") or "")),
                            "path": path}
                    break
        if spec is None:
            return None
        params[name] = value
        bindings[name] = spec
        replacements.append((value, "{{%s}}" % name))
    rewritten: list[dict[str, Any]] = []
    for step in steps:
        new_step = dict(step)
        for key in ("text", "value", "expect_text"):
            if isinstance(new_step.get(key), str):
                for literal, token in replacements:
                    new_step[key] = new_step[key].replace(literal, token)
        if new_step.get("selectors"):
            sels = []
            for sel in new_step["selectors"]:
                for literal, token in replacements:
                    sel = sel.replace(_esc(literal), token).replace(literal, token)
                sels.append(sel)
            new_step["selectors"] = sels
        rewritten.append(new_step)
    return rewritten, params, bindings


def _recording_had_page_actions(path: Any) -> bool:
    """Did the authoring history contain ANY action beyond `done`? Distinguishes the two
    zero-step compiles: an agent that verified a condition already held and finished
    (nothing to commit — expected for 'if X, do Y' guard subtasks) vs. one that acted
    only through tools the compiler drops (a coverage gap worth a warning). Unreadable
    recordings keep the conservative (coverage-gap) reading."""
    try:
        history = json.loads(Path(path).read_text()).get("history") or []
    except Exception:  # noqa: BLE001 - diagnosis only; never fail the commit path
        return True
    for item in history:
        for action in (item.get("model_output") or {}).get("action") or []:
            if isinstance(action, dict) and any(k != "done" for k in action):
                return True
    return False


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
    run_values: dict[str, str] | None = None,
    start_url: str | None = None, dynamic: bool = False,
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
    segment_started = datetime.now().timestamp()
    # Authoring records to a TEMP path and promotes only on success: the canonical
    # recording.json always corresponds to the last COMMITTED skill, and a failed
    # re-authoring can no longer destroy it (observed live 2026-07-29: a failed --fresh
    # re-author overwrote the Dec-26 segment's good recording, then set the wreck aside
    # as .failed.json).
    rec_tmp = (sstore.recording_path(sid).with_suffix(".new.json")
               if commit and not dirty else None)
    seg = await hs.agent_segment(
        sub, sid, context, gate, completed=completed, remaining=remaining,
        dirty=dirty, prior_failure=prior_failure, findings=findings,
        record_path=rec_tmp,
    )
    if not seg.ok:
        # Keep a FAILED authoring's trace for diagnosis, but OFF the canonical path. The
        # mtime guard makes sure we only move a trace THIS segment wrote — not a stale
        # temp file orphaned by a crashed earlier run.
        try:
            if rec_tmp is not None and rec_tmp.exists() \
                    and rec_tmp.stat().st_mtime >= segment_started - 1:
                failed = sstore.recording_path(sid).with_suffix(".failed.json")
                rec_tmp.replace(failed)
                logger.info("segment %s: failed authoring trace kept at %s",
                            sid, failed.name)
        except OSError as exc:
            logger.debug("could not set aside failed recording for %s: %s", sid, exc)
        return seg
    if not commit:
        return seg
    if dirty:
        # Recovered in place, but the recording is not committable. A stale entry that keeps
        # failing gets retired so the NEXT run authors a clean replacement.
        sstore.archive_if_failing(sid, threshold=_ARCHIVE_AFTER_FAILURES)
        return seg
    # Promote the fresh recording to canonical before compiling from it. A segment that
    # recorded nothing (or only an orphaned stale temp exists) has nothing to commit —
    # never re-compile a previous run's trace under a fresh pass.
    try:
        if rec_tmp is None or not rec_tmp.exists() \
                or rec_tmp.stat().st_mtime < segment_started - 1:
            return seg
        rec_tmp.replace(sstore.recording_path(sid))
    except OSError as exc:
        logger.warning("segment %s: could not promote fresh recording: %s", sid, exc)
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
            if _recording_had_page_actions(sstore.recording_path(sid)):
                logger.warning("segment %s: recording compiled to ZERO steps; not "
                               "committing (agent acted only through tools the compiler "
                               "drops)", sid)
            else:
                logger.info("segment %s: agent finished without any page action (a "
                            "conditional guard already satisfied this run); nothing to "
                            "commit — future runs re-check it live", sid)
            return seg
        # Provenance commit guard — the general, wording-free memory rule: a segment
        # that acted with values sourced from the run's FINDINGS consumed runtime data,
        # and a cached replay would re-use this run's values forever. Since 2026-07-24
        # the guard BINDS before it refuses: a flagged value with a structured source
        # (an extract this run captured, or a unique leaf in a create-write response)
        # is rewritten to a {{bound_N}} param resolved fresh from EACH run's own data
        # at load time. Only prose-only values still refuse the commit — those segments
        # keep authoring fresh every run.
        runtime_values = _findings_sourced_values(
            steps, sub.instantiated_prompt, findings)
        # Findings-independent leg: values the run's own create-writes reported are
        # runtime data even when no finding names them (the findings channel goes
        # quiet once the producer segments replay).
        bodies = _captured_write_bodies(hs)
        task_wording = " \n ".join([sub.instantiated_prompt, *completed, *remaining])
        for value in _body_sourced_values(steps, task_wording, bodies):
            if value not in runtime_values:
                runtime_values.append(value)
        if dynamic:
            # The wording DECLARES consumption of noted data, so this commit is held to
            # a stricter bar than the substring guard alone: every typed value must be
            # prompt-sourced or flagged for binding, and there must be something to bind.
            # An unattributable value may be runtime data the guard cannot see (a
            # re-formatted date), and a consumer recording with no bindable value at all
            # keeps the pre-bindings behavior: author fresh every run.
            loose = _unattributed_typed_values(steps, sub.instantiated_prompt,
                                               runtime_values)
            if loose or not runtime_values:
                sstore.steps_path(sid).unlink(missing_ok=True)
                what = (f"unattributable typed value(s) "
                        f"{', '.join(v[:32] for v in loose[:3])}" if loose
                        else "no bindable runtime value in the recording")
                print(f"[*] segment [{sid}]: consumes noted data with {what} -> not "
                      f"cached; future runs author it with their own fresh values")
                return seg
        bound = None
        if runtime_values:
            sources = {**(run_values or {}), **(seg.extracted or {})}
            bound = _bind_runtime_values(steps, runtime_values, sources, bodies)
            if bound is None:
                sstore.steps_path(sid).unlink(missing_ok=True)
                print(f"[*] segment [{sid}]: used runtime data from earlier steps "
                      f"({', '.join(v[:40] for v in runtime_values[:3])}) with no "
                      f"structured source to bind -> never cached; future runs author "
                      f"it with their own fresh values")
                return seg
            steps, bind_params, bindings = bound
            _atomic_write(sstore.steps_path(sid), json.dumps(steps, indent=2))
            summary = "; ".join(f"{name}<-{spec['kind']}" for name, spec in bindings.items())
            print(f"[*] segment [{sid}]: runtime value(s) bound to this run's data -> "
                  f"replayable ({summary})")
        # An aux segment's pages are foreign origins, so its end context is host-qualified
        # like its start context (current_url still reads the aux page here — the loop
        # closes the helper tab only after the segment commits).
        normalize = (sstore.normalize_aux_context if getattr(sub, "tab_url", None)
                     else sstore.normalize_context)
        end_context = normalize(await hs.current_url())
        if bound is not None:
            # A bound entry's template is written deterministically (params = the
            # authoring literals, bindings = their runtime sources); the LLM
            # parameterizer is skipped — it knows nothing about binding tokens and
            # could mangle them.
            adapt.save_template(sstore.template_path(sid), {
                "source_prompt": sub.instantiated_prompt,
                "params": bind_params, "steps": steps, "bindings": bindings,
            })
            params = bind_params
        else:
            params = await _save_segment_template(
                sid, sub.instantiated_prompt, steps, hs.runner.expander_llm)
        manifest_fields: dict[str, Any] = dict(
            params=params or dict(sub.values or {}),
            context=context, end_context=end_context,
            marker_write=gate.kind == "marker", steps=len(steps), provenance="clean",
        )
        if start_url:
            # The raw page the authoring run started from — informational (identity-fork
            # log lines point here); never navigated to automatically.
            manifest_fields["start_url"] = start_url
        if bound is not None:
            manifest_fields["bindings"] = bindings
        if getattr(sub, "tab_url", None):
            # Informational: the declared subtask is what triggers the helper tab at
            # replay time; the manifest field keeps the entry self-describing.
            manifest_fields["tab_url"] = sub.tab_url
        sstore.update_manifest(sid, sub.template_prompt, **manifest_fields)
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
    run_values: dict[str, str] = {}  # structured {label: value} extracts, run-wide
    resolve_binding = _binding_resolver(run_values, hs)
    # First evidence write BEFORE any segment runs: a run that dies in subtask 0 still
    # leaves the decomposition plan on disk.
    _write_progress(hs, task=task, tid=tid, subtasks=subtasks, segments=segments,
                    status="running")
    try:
        for i, sub in enumerate(subtasks):
            # An aux-tab subtask is keyed on its DECLARED tab URL (host-qualified — see
            # normalize_aux_context), not on wherever the main page happens to be: the
            # framework's goto(tab_url) IS the replay precondition, and main-page keying
            # would split the identical helper procedure into one entry per hosting task.
            aux_url = getattr(sub, "tab_url", None)
            raw_start_url = aux_url or await hs.current_url()
            context = (sstore.normalize_aux_context(aux_url) if aux_url
                       else sstore.normalize_context(raw_start_url))
            sid = sstore.subtask_id(sub.template_prompt, context)
            force_author = reauthor_match(reauthor, sub)
            is_judge = getattr(sub, "kind", "action") == "judge"
            is_loop = getattr(sub, "kind", "action") == "loop"
            # A leading-"If" branch guard: whether its actions run at all depends on live
            # page state, so a recording of the TRUE branch must never replay (and a TRUE
            # branch run must never commit one). See decompose.is_conditional_guard.
            is_conditional = is_conditional_guard(sub.template_prompt)
            # Dynamic-input gate: a subtask whose wording USES data noted by an earlier
            # segment ("the noted generated name") must not replay a plain recording once
            # this run HAS such observations — runtime values are never parameterizable
            # (adapt lifts only values the prompt spells out), so a cached script would
            # type the AUTHORING run's concrete ones: stale by construction. It runs with
            # the agent, which receives the fresh findings. Its recording IS committed
            # when the provenance guard can bind EVERY runtime value to a structured
            # source this run produced ({{bound_N}}, resolved fresh at load time);
            # unattributable or prose-only values refuse the commit and the segment keeps
            # authoring each run. Without findings there is nothing fresh to be stale
            # against (and nothing the agent could substitute either), so the zero-LLM
            # replay stays.
            is_consumer = bool(findings) and consumes_noted_data(sub.template_prompt)
            is_dynamic = is_consumer

            # Semantic routing: a wording with NO direct entry may still be a known
            # procedure (alias table -> local embeddings -> one LLM verify). A routed sid
            # replays the canonical skill with values re-keyed to its params; authoring
            # after a routed failure goes under the ORIGINAL sid so the canonical entry
            # is never overwritten by a different wording's recording. Aux subtasks are
            # direct-hit only (v1): a routed canonical entry could have been recorded on
            # a different site family.
            author_sid, route_values = sid, None
            if (not fresh and not force_author and not is_judge and not is_loop
                    and not is_conditional and not is_dynamic
                    and not aux_url and not sstore.has_script(sid)
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
            skip_reason: str | None = None   # why this subtask did not replay

            if is_dynamic and (entry or {}).get("bindings"):
                # The entry was committed WITH runtime bindings: its dynamic values
                # re-resolve from THIS run's own data at load time, so the consumer-
                # wording net stands down and the zero-LLM replay proceeds.
                is_dynamic = False

            if is_dynamic and sstore.has_script(sid):
                # The entry predates this rule (or slipped in on a findings-free run):
                # retire it AFTER the gate above captured its recorded end_context.
                print(f"[*] subtask {i} [{sid}]: consumes data noted this run -> retiring "
                      f"the cached recording (it would type the authoring run's stale "
                      f"values)")
                sstore.archive_entry(sid)

            if aux_url:
                # The helper tab is the LOOP's responsibility, not the segments': opening
                # it here lets a failed replay hand the SAME live (dirty) tab to the
                # recovering agent, and the finally below guarantees the main page is all
                # that survives the subtask.
                try:
                    await hs.open_aux_tab(aux_url)
                    print(f"[*] subtask {i} [{sid}]: helper tab opened at {aux_url}")
                except Exception as exc:  # noqa: BLE001 - unreachable site fails the subtask
                    seg = Segment(index=sub.index, sid=sid,
                                  prompt=sub.instantiated_prompt, context=context,
                                  mode="authored", kind=getattr(sub, "kind", "action"),
                                  error=f"could not open helper tab {aux_url}: {exc}")
                    segments.append(seg)
                    _write_progress(hs, task=task, tid=tid, subtasks=subtasks,
                                    segments=segments, status="running")
                    print(f"[*] subtask {i} [{sid}]: FAILED ({seg.error}) -> stopping "
                          f"task (later subtasks depend on this state)")
                    break

            try:
                # Judge, loop, and conditional-guard nodes never touch the library in
                # EITHER direction: a replayed judge would click through with nobody
                # looking (hollow pass), a replayed loop would walk a fixed number of
                # iterations and land anywhere, a replayed conditional would take its
                # branch unconditionally — and their recordings must never be committed
                # for the same reasons.
                if not fresh and not force_author and not is_judge and not is_loop \
                        and not is_conditional and not is_dynamic \
                        and sstore.has_script(sid):
                    load_sub = sub if route_values is None else SimpleNamespace(
                        instantiated_prompt=sub.instantiated_prompt, values=route_values)
                    skill = skills.load_skill(sid, load_sub,
                                              run_resolver=resolve_binding)
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
                            # A bound entry's takeover must know THIS run's resolved
                            # values (observed live: a dirty recovery found the stale
                            # record's panel open, had no observation naming the fresh
                            # ref, and declared the WRONG record already done).
                            takeover_findings = findings
                            resolved = [v for v in (
                                resolve_binding(spec) for spec in
                                ((entry or {}).get("bindings") or {}).values()) if v]
                            if resolved:
                                takeover_findings = findings + [
                                    f"{sub.instantiated_prompt[:80]}: this run's live "
                                    f"value(s) for this step: {', '.join(resolved)}"]
                            if aux_url:
                                # The failure may have crashed/closed the helper tab;
                                # re-ensure it (a live tab is re-fronted untouched — the
                                # dirty state is exactly what the recovery agent needs).
                                try:
                                    await hs.open_aux_tab(aux_url)
                                except Exception as exc:  # noqa: BLE001
                                    logger.warning("could not re-open helper tab for "
                                                   "recovery: %s", exc)
                            seg = await _author_segment(
                                hs, sub, author_sid, context, gate, completed=completed,
                                remaining=remaining, dirty=dirty, prior_failure=prior,
                                findings=takeover_findings, run_values=run_values,
                                start_url=raw_start_url, dynamic=is_consumer)
                            seg.mode = "replay_failed->authored"
                    else:
                        skip_reason = "values_unresolved"
                        print(f"[*] subtask {i} [{sid}]: library hit but values did not "
                              f"resolve -> authoring")

                if seg is None:
                    if force_author and sstore.has_script(sid):
                        # The existing entry stays as the gate's end_context reference and is
                        # only overwritten if the fresh authoring passes its gate.
                        skip_reason = "reauthor"
                        print(f"[*] subtask {i} [{sid}]: --reauthor -> authoring with the "
                              f"agent (entry replaced only on success)")
                    elif getattr(sub, "fallback", False):
                        skip_reason = "fallback"
                        print(f"[*] subtask {i} [{sid}]: whole-prompt fallback node -> "
                              f"agent runs the ENTIRE task as one segment (decomposition "
                              f"unavailable), whole-task step budget, never cached")
                    elif is_judge:
                        skip_reason = "judge"
                        print(f"[*] subtask {i} [{sid}]: judge node (verification) -> agent "
                              f"runs it live, never cached")
                    elif is_loop:
                        skip_reason = "loop"
                        print(f"[*] subtask {i} [{sid}]: loop node (repeat-until) -> agent "
                              f"runs it live with an extended step budget, never cached")
                    elif is_conditional:
                        skip_reason = "conditional"
                        print(f"[*] subtask {i} [{sid}]: conditional branch guard -> agent "
                              f"runs it live, never cached")
                    elif is_dynamic:
                        skip_reason = "dynamic"
                        print(f"[*] subtask {i} [{sid}]: uses data noted by an earlier "
                              f"step -> agent runs it with this run's fresh values "
                              f"(cached only if the provenance guard can bind them)")
                    elif fresh and sstore.has_script(sid):
                        # Without this print a --fresh run with a library hit is
                        # indistinguishable from a cache miss (observed live 2026-07-29:
                        # "recordings are never used" was a --fresh run).
                        skip_reason = "fresh"
                        print(f"[*] subtask {i} [{sid}]: --fresh -> ignoring the library "
                              f"entry, re-authoring (entry replaced on success)")
                    elif not sstore.has_script(sid):
                        fork = sstore.find_same_template_entry(sub.template_prompt, sid)
                        if fork is not None:
                            osid, oentry = fork
                            skip_reason = "identity_fork"
                            print(f"[*] subtask {i} [{sid}]: identity fork — same wording "
                                  f"recorded from {oentry.get('context')} as [{osid}] "
                                  f"(start {oentry.get('start_url') or 'unknown'}), not "
                                  f"reusable from {context} -> authoring fresh")
                        else:
                            skip_reason = "no_entry"
                            print(f"[*] subtask {i} [{sid}]: no library entry -> authoring "
                                  f"with the agent")
                    seg = await _author_segment(hs, sub, author_sid, context, gate,
                                                completed=completed, remaining=remaining,
                                                findings=findings, run_values=run_values,
                                                start_url=raw_start_url,
                                                dynamic=is_consumer,
                                                # A fallback blob never commits: a whole-
                                                # task recording replayed blind is the
                                                # pre-hybrid behavior this mode degrades
                                                # FROM, not a library asset.
                                                commit=not is_judge and not is_loop
                                                and not is_conditional
                                                and not getattr(sub, "fallback", False))
                    seg.skip_reason = skip_reason
            finally:
                if aux_url:
                    # Subtask-scoped lifetime: whatever happened above, the helper tab is
                    # gone and the pinned main page is the state the next subtask starts
                    # from. close_aux_tab never raises.
                    await hs.close_aux_tab()

            segments.append(seg)
            _write_progress(hs, task=task, tid=tid, subtasks=subtasks, segments=segments,
                            status="running")
            if not seg.ok:
                print(f"[*] subtask {i} [{sid}]: FAILED ({seg.error}) -> stopping task "
                      f"(later subtasks depend on this state)")
                break
            completed.append(sub.instantiated_prompt)
            if seg.extracted:
                # Structured twin of the prose findings: the run-value store bindings
                # resolve against (later collisions win, matching _history_extracts).
                run_values.update({str(k): str(v) for k, v in seg.extracted.items()})
            if seg.finding:
                findings.append(f"{sub.instantiated_prompt[:80]}: {seg.finding}")
    except KeyboardInterrupt:
        # Second Ctrl+C aborts mid-segment. Four killed runs on 2026-08-05 left
        # progress.json stuck at "running" with the in-flight segment unrecorded —
        # video-only forensics. Stub the segment, flush (which also writes the
        # collectors), and propagate the abort.
        try:
            segments.append(Segment(
                index=sub.index, sid=sid, prompt=sub.instantiated_prompt,
                context=context, mode="authored",
                kind=getattr(sub, "kind", "action"), ok=False,
                error="interrupted by user (Ctrl+C)"))
        except Exception:  # noqa: BLE001 - interrupted before the loop bound its vars
            pass
        _write_progress(hs, task=task, tid=tid, subtasks=subtasks, segments=segments,
                        status="interrupted", is_successful=False)
        raise
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
    # Break the authored count down by WHY each segment did not replay — a bare
    # "0 replayed" hides whether the cache was cold, bypassed (--fresh), or the
    # subtasks are live-by-kind.
    reasons = Counter(s.skip_reason for s in segments if s.skip_reason)
    breakdown = ", ".join(f"{n} {r}" for r, n in sorted(reasons.items()))
    result.final_result = (
        f"Hybrid run: {len(segments)}/{len(subtasks)} subtasks "
        f"({replayed} replayed, {authored} authored"
        + (f": {breakdown}" if breakdown else "") + ")"
        + ("" if all_ok else f" — FAILED at subtask {len(segments) - 1}")
    )
    # Final evidence write: status flips to "finished" with the verdict. report.json (the
    # richer artifact, written by __main__ AFTER assertions) stays authoritative;
    # progress.json is kept so a crash between here and report writing still leaves proof.
    _write_progress(hs, task=task, tid=tid, subtasks=subtasks, segments=segments,
                    status="finished", is_successful=result.is_successful)

    logger.info("◀ HYBRID done %s: success=%s %s", tid, result.is_successful,
                result.final_result)
    return result
