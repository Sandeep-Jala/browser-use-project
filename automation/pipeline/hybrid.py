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
guard (a slice that declares a `probe:`) also always runs live and uncommitted: a recording
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
from automation.pipeline.decompose import (Subtask, announces_new_tab,
                                           downloads_file, get_decomposition,
                                           is_conditional_guard)
from automation.pipeline.prompts import scoped_subtask_prompt
from automation.pipeline.runner import RunResult, Runner, _first_create_write
from automation.pipeline.script_compile import (CALLOUT_SCROLL_PIN_JS, REVEAL_CSS_JS,
                                                _atomic_write, _esc,
                                                repeat_hint_from_wording,
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
    end_title: str | None = None    # recorded end document title, demote-only (see
                                    # _recording_end_title); rides on any base kind
    checks: tuple = ()              # declared verify checks (tuple of checks.Check)
    # Declared exemption from the window write rule — see Subtask.allow_write_refusal.
    allow_write_refusal: bool = False


# The normalized forms of "no location": normalize_context("about:blank") == "blank" and
# normalize_context("") == "/". Never valid as a postcondition — see _base_gate's heal.
_DEGENERATE_CONTEXTS = ("blank", "/", "")


def _is_degenerate_url(url: Any) -> bool:
    """True when `url` names no location at all — a blank tab or an unreadable/closed page.

    One predicate, two callers: _pick_main_page has always skipped these when choosing the
    page to pin, and the commit site must skip them when recording an end context.
    normalize_context turns "about:blank" into "blank" and "" into "/", and that second one
    is indistinguishable from a legitimate app root — which is why this is judged on the
    RAW url, before normalization flattens the difference.
    """
    return str(url or "").strip() in ("", "about:blank")


def segment_gate(sub: Subtask, entry: dict[str, Any] | None, context: str) -> Gate:
    """Resolve the gate for a subtask (see _base_gate for the kind precedence), then
    attach the subtask's declared verify checks — they apply to every kind — and its
    declared write-rule waiver."""
    gate = _base_gate(sub, entry, context)
    gate.checks = tuple(getattr(sub, "verify", None) or ())
    gate.allow_write_refusal = bool(getattr(sub, "allow_write_refusal", False))
    return gate


def _base_gate(sub: Subtask, entry: dict[str, Any] | None, context: str) -> Gate:
    gate = _base_gate_kind(sub, entry, context)
    # Rides on WHATEVER kind resolved — the title is evidence about the end state, not a
    # kind of its own, and evaluate_gate applies it demote-only (same shape as `checks`).
    gate.end_title = (entry or {}).get("end_title") or None
    return gate


def _base_gate_kind(sub: Subtask, entry: dict[str, Any] | None, context: str) -> Gate:
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
    if end_context in _DEGENERATE_CONTEXTS:
        # Heal on read. An entry committed before the close-its-own-page guard can carry
        # the normalized form of "no location" ("blank" from about:blank, "/" from a closed
        # page) as its postcondition, which is unsatisfiable except by coincidence — see
        # library/7320039db9ba7e26 and run 20260826_130102. The guard at the commit site
        # stops NEW ones; this stops the existing ones gating.
        #
        # Note for _evaluate_base_gate below: "blank" is also why the normalizer it picks
        # from the string's shape went wrong. "blank" has no leading "/", so the AUX
        # normalizer ran and `reached` came back host-qualified, unable to match a
        # main-style context — a second, independent reason that gate could never pass.
        end_context = None
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


def _describe_next_conditionals(subtasks: list[Subtask],
                                i: int) -> tuple[str | None, int]:
    """The probe condition(s) of the run of conditional slices that immediately FOLLOW
    subtask `i`, in agent-actionable words, plus how many slices that run covers.

    A declared `probe:` says deterministically that the framework will look for this
    outcome BEFORE that slice runs. The step that PRODUCES the outcome is its
    predecessor, and nothing told the predecessor so: it read the outcome as its own
    failed action and did the successor's work itself (run 20260901_151214, the FPS
    submit slice — the server refused the write, the agent cancelled the dialog,
    reopened the form, re-uploaded and re-submitted; 4 submits for 1, 15 steps, 554k
    tokens, and the flail was committed as 22 replayable actions).

    Only the CONDITION crosses the boundary, never the successor's action words —
    telling the predecessor "click Cancel" is the very thing this stops. Consecutive
    guards are ONE handoff from the producing step's point of view, so the scan walks
    the whole run; the count is what lets the caller drop those slices from the
    still-ahead list, where their action words would otherwise be handed over verbatim.
    """
    run = []
    for sub in subtasks[i + 1:]:
        probe = getattr(sub, "probe", None)
        if probe is None:
            break
        run.append(probe)
    if not run:
        return None, 0
    return " or ".join(_describe_check(p) for p in run), len(run)


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
#
# Raised 35 -> 120 (2026-08-24) for loops whose ITERATION is itself multi-step. The old
# size assumed one click per pass; the portal data-request loop opens three dialogs per
# employee and run 20260824_165824 measured 10 agent steps for a single employee (one
# misclick recovery included) against a 12-employee list — 120 steps of work under a
# 60-step ceiling. This is a CEILING, not a target: a loop that converges early still
# stops early, and the only cost of the headroom is how far a genuinely runaway loop gets
# before the wall (~21k tokens/step in that run).


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
    or ad-hoc: fired-but-never-accepted business writes fail the segment, unless the
    subtask declared allow_write_refusal, which records the refusal without failing),
    then any declared verify checks. Roll-up, write rule, and checks are demote-only — they can
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
    if gate.end_title and page is not None:
        # Demote-only, like the roll-up and the declared checks below. A segment whose
        # recording CHANGED the document title must change it the same way on replay —
        # the only readable difference between an accepted and a refused in-page
        # submission (see _recording_end_title). Fails OPEN on an unreadable title: an
        # additive gate must never become a new false-fail source.
        reached: str | None = None

        async def _title_ok() -> bool:
            nonlocal reached
            try:
                reached = " ".join(str(await page.title() or "").split())
            except Exception as exc:  # noqa: BLE001 - unreadable title proves nothing
                logger.debug("end_title read failed: %s", exc)
                reached = None
                return True
            return reached == gate.end_title
        matched = await _settled(_title_ok, steps_ok)
        if reached is not None:
            # Recorded on the PASS too — same convention as detail["checks"], which appears
            # whenever checks are declared. A gate carrying a pin is not a bare gate, and
            # without this there is no artifact evidence the check ever ran: a silently
            # inert gate would look exactly like a passing one.
            detail["end_title"] = {"expected": gate.end_title, "reached": reached,
                                   "ok": matched}
            ok = ok and matched
    wr_ok, wr_reasons = window_write_rollup(requests_window)
    if not wr_ok:
        # Recorded either way — the report keeps the server's refusal verbatim. The
        # declared waiver only stops it FAILING the segment (see Gate.allow_write_refusal):
        # a slice that declares its own error branch ends legitimately on a refusal.
        detail["write_rollup"] = wr_reasons
        if gate.allow_write_refusal:
            # Evidence, not a verdict: _check_failure_reason skips a waived rollup so a
            # refusal the slice was told to tolerate can never be handed back as the
            # explanation for some OTHER failure, and _author_segment reads this flag to
            # refuse the commit (the recording ends on the error branch).
            detail["write_refusal_waived"] = True
        else:
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
    title = detail.get("end_title") or {}
    if title and not title.get("ok"):
        return (f'the page never reached its recorded end state: expected the title '
                f'"{title.get("expected")}", found "{title.get("reached")}" — the '
                f"segment's actions did not take effect")
    wrollup = detail.get("write_rollup") or []
    if wrollup and not detail.get("write_refusal_waived"):
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
                                    # | "probe" (conditional resolved by a FALSE probe)
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
    # The replay failure that preceded an in-place takeover (mode
    # "replay_failed->authored"). Without this the reason a library entry stopped
    # working left no artifact at all — only a stdout line nobody kept.
    replay_error: str | None = None
    # Why this segment did NOT replay (None on replays): "fresh" | "reauthor" | "judge" |
    # "loop" | "conditional" | "probe_absent" | "dynamic" | "fallback" | "no_entry" |
    # "identity_fork" | "values_unresolved". The answer to "why didn't it use the
    # recording?" without archaeology — surfaced in the report row and the run summary.
    skip_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "sid": self.sid, "prompt": self.prompt,
            "context": self.context, "mode": self.mode, "kind": self.kind, "ok": self.ok,
            "gate": self.gate, "steps_executed": self.steps_executed,
            "duration_seconds": round(self.duration_seconds, 1),
            "healed_steps": self.healed_steps, "tokens": self.tokens, "error": self.error,
            "replay_error": self.replay_error,
            "finding": self.finding, "extracted": self.extracted,
            "downloads": self.downloads, "skip_reason": self.skip_reason,
        }


VALUES_FILE = "values.json"


def _write_run_values(hs: "HybridSession", run_values: dict[str, str]) -> None:
    """Persist the run's extracted values to <run_dir>/values.json — the handoff file
    between the segment that READS data off a page and the later segments that TYPE
    it. In-memory `run_values` is still the primary; this file makes the handoff
    inspectable when a binding misses, and survives the process (a resolver miss falls
    back to it). Best-effort by rule: an evidence write must never break the run."""
    try:
        _atomic_write(hs.run_dir / VALUES_FILE,
                      json.dumps({str(k): str(v) for k, v in (run_values or {}).items()},
                                 indent=2))
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not write %s: %s", VALUES_FILE, exc)


def _read_run_values(hs: Any) -> dict[str, str]:
    """values.json as a dict ({} when absent/unreadable)."""
    try:
        path = Path(getattr(hs, "run_dir", "") or ".") / VALUES_FILE
        if path.exists():
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
    except Exception as exc:  # noqa: BLE001 - the in-memory store is the primary
        logger.debug("could not read %s: %s", VALUES_FILE, exc)
    return {}


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


def _page_url(page: Any) -> str:
    """A page's URL for a log line, never raising on a page that just went away."""
    try:
        return str(page.url)
    except Exception:  # noqa: BLE001 - a closed/detached page still deserves a name
        return "<unreadable>"


class HybridSession:
    """One task's live execution surface: a single BrowserSession (agent driver), a single
    Playwright CDP connection (replay driver + telemetry), and the collectors — all opened
    once and torn down only in finalize(). Both drivers point at the same Chromium. The
    shared page state is explicit: the MAIN page is pinned at open() and never navigated
    away by aux work; an optional AUX page is what current_page() returns while it is
    alive. The AUX slot holds either a helper tab the framework opened for a `tab_url`
    subtask (closed at that subtask's end) or a tab the app opened because a subtask said
    it would (adopt_announced_tab) — the latter outlives its opener on purpose, since the
    subtasks that follow are the ones that need it, and it goes away when the task closes
    it or the run ends."""

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
        # The callout scroll pin, on the same long-lived connection but NOT gated on
        # reveal_hidden_controls: it is correctness (a popup that vanishes loses the value
        # being typed), not cosmetics. Installing at document start matters — the pin's
        # scroll listener must be registered BEFORE Fluent registers the one it dismisses
        # on, and same-phase listeners fire in registration order.
        for ctx in hs.pw_browser.contexts:
            try:
                await ctx.add_init_script(CALLOUT_SCROLL_PIN_JS)
            except Exception as exc:  # noqa: BLE001 - the per-step re-assert covers a miss
                logger.debug("callout scroll pin install failed: %s", exc)
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
        real = [p for p in pages if not _is_degenerate_url(p.url)]
        return real[0] if real else pages[0]

    def current_page(self) -> Page | None:
        """The page segments run against: the live aux (helper) tab while one is open, else
        the pinned main page. Explicit handles, not a scan — with two live tabs a scan
        could not tell which one the segment means."""
        moved = False
        if self._aux_page is not None and self._aux_page.is_closed():
            # The helper tab is GONE — a segment that ended "…and then close this tab"
            # (replayed by script_compile._close_and_return, or done live by the agent).
            # Release the pin HERE rather than at each call site: a closed page that still
            # owns the session is how the OTP portal tab kept the next subtask inside it
            # (run 20260902_091047_561480).
            self._aux_page, moved = None, True
        if self._aux_page is not None and not self._aux_page.is_closed():
            return self._aux_page
        if self._main_page is None or self._main_page.is_closed():
            self._main_page = self._pick_main_page()
            moved = True
        if moved and self._main_page is not None:
            # The page the run acts on CHANGED (the aux tab closed, or the pinned page
            # died and a survivor takes its place). browser-use's RecordingWatchdog
            # streams frames from ONE CDP session and silently drops every other
            # session's (recording_watchdog.on_screencastFrame), so without a focus event
            # here the run video freezes for the rest of the run while the run itself
            # carries on — and a takeover agent would act on the tab it still thinks is
            # current. Fire-and-forget from this sync path; keep a ref so the loop cannot
            # GC the task mid-flight.
            try:
                self._bg_focus_task = asyncio.get_running_loop().create_task(
                    self._focus_browser_use(self._main_page))
            except RuntimeError:
                pass  # no running loop (bare construction in unit tests)
        return self._main_page

    async def current_url(self) -> str:
        """The current page's URL, or "" when there is no readable page. Guarded here so
        callers do not each re-implement the closed-page check."""
        page = self.current_page()
        if page is None:
            return ""
        try:
            return "" if page.is_closed() else page.url
        except Exception as exc:  # noqa: BLE001 - an unreadable page is not a crash
            logger.debug("current_url read failed: %s", exc)
            return ""

    async def current_title(self) -> str:
        """The current page's document title, or "" when there is no readable page. The
        live twin of current_url() — and the ONLY trustworthy source for an end title,
        since a recorded one lags the SPA's async title update (see _pin_end_title)."""
        page = self.current_page()
        if page is None:
            return ""
        try:
            return "" if page.is_closed() else (await page.title() or "")
        except Exception as exc:  # noqa: BLE001 - an unreadable title is not a crash
            logger.debug("current_title read failed: %s", exc)
            return ""

    async def probe_condition(self, check: Any) -> bool:
        """One declared probe check against the live page — the deterministic stand-in
        for a conditional guard's presence judgment (see run_hybrid_task). Fails
        closed (absent) on an unevaluable page: the branch is then skipped, and if
        the condition WAS raised the next segment's failure hands recovery to the
        agent as usual."""
        results = await evaluate_checks(self.current_page(), [], (check,))
        return bool(results and results[0].get("ok"))

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

    async def _focused_page(self, candidates: list[Page]) -> Page | None:
        """The candidate browser-use's AGENT FOCUS points at, or None when unidentifiable.

        This is the page the segment finished on, stated by the driver that did the work
        rather than inferred by counting tabs. browser-use auto-focuses any newly opened
        tab, so focus at OPEN time proves nothing — but focus at segment END is the result
        of every switch and click the agent made, which is exactly the fact the consumers
        want."""
        try:
            target = self.session.get_focused_target() if self.session is not None else None
            target_id = getattr(target, "target_id", None)
            if not target_id:
                return None
            for page in candidates:
                if await self._page_target_id(page) == target_id:
                    return page
        except Exception as exc:  # noqa: BLE001 - falls back to the count rule below
            logger.debug("could not resolve the agent's focused page: %s", exc)
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

    async def adopt_announced_tab(self, sub: Subtask) -> None:
        """Register a tab the SUBTASK SAID would open as the aux page, so the tab
        invariant spares it and the next subtask runs inside it.

        Run 20260824_163152 died here: subtask 3 clicked the app's external-link button,
        did the whole OTP handshake in the tab it opened, and reported "Proceed Securely
        submitted successfully in the new tab" — then close_extra_tabs swept that tab as a
        misclick popup on the way out of the segment, and subtask 4 (the edits that have
        to happen INSIDE the portal) had nowhere to run. The tab carries a per-request
        signed URL, so the wording cannot name it and `tab_url` is never set.

        The wording gate is what separates this from a genuine misclick: browser-use
        focuses ANY newly opened tab, so a tab appearing proves nothing, while "the task
        said this would happen" does.

        WHICH tab, though, is answered by the agent's own focus at segment end (see
        _focused_page) — not by counting. Counting failed the case it was written for: in
        run 20260825_105115 a replay's force-retry opened the SAME portal tab twice, so
        two extras existed, `len(extras) != 1` adopted nothing, the sweep took both, and
        the postcondition gate then measured the app tab and failed a segment whose work
        had actually succeeded. Two tabs to the same destination are not ambiguous, and
        the agent had explicitly switched into the one it finished in. The count rule
        survives only as the fallback for when focus cannot be resolved.
        """
        try:
            if self._aux_page is not None and not self._aux_page.is_closed():
                return  # a declared helper tab owns the slot; never displace it
            if not announces_new_tab(sub.template_prompt):
                return
            main = self._main_page
            if main is None or main.is_closed():
                # No live pin to measure "extra" against, and current_page() is about to
                # re-pin a survivor as MAIN — adopting that same page as AUX would give
                # one tab both roles.
                return
            extras = [p for ctx in self.pw_browser.contexts for p in ctx.pages
                      if p is not main and not p.is_closed()]
            if not extras:
                return
            chosen = await self._focused_page(extras)
            if chosen is None and len(extras) != 1:
                if extras:
                    # NAME them: "left 2 extra tabs open" alone cost a network-log
                    # forensics pass to explain (run 20260825_105115, where a replay's
                    # force-retry had opened the SAME tab twice), and the gate that fails
                    # afterwards reports only the page it settled on.
                    logger.info("subtask announced a new tab, left %d extra tabs open, and "
                                "the agent's focus resolved to none of them — adopting none "
                                "(a misclick is indistinguishable here): %s", len(extras),
                                "; ".join(_page_url(pg) for pg in extras))
                return
            self._aux_page = chosen or extras[0]
            # Same reason open_aux_tab focuses: the two drivers track tabs independently,
            # and browser-use's RecordingWatchdog streams frames from ONE CDP session — an
            # unaligned focus acts on, screenshots, and records the wrong tab.
            await self._focus_browser_use(self._aux_page)
            logger.info("adopted the tab this subtask announced, by %s (%s) — it survives "
                        "into the next subtask",
                        "the agent's focus" if chosen is not None else "elimination",
                        _page_url(self._aux_page))
        except Exception as exc:  # noqa: BLE001 - adoption is best-effort hygiene
            logger.debug("adopt_announced_tab: %s", exc)

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
        branch: bool = False,
    ) -> Segment:
        """Replay a library skill on the live page. No LLM.

        `branch` marks a CONDITIONAL slice, whose entire recording is the TRUE branch of an
        "If X, do Y" guard. Such a recording is made on the run where the popup appeared and
        replayed on runs where it may not; the first recorded action therefore doubles as the
        condition's own test. If NOTHING acted before the failure, the branch was never
        raised and the segment is a no-op that passes — the alternative (what happened
        before) is that every popup-free run fails the segment and pays for an agent
        takeover. Once any step HAS acted the concession is off: the branch was raised and
        left half-done, which is a real failure and still reported as one.
        """
        started = datetime.now()
        watermark = self.network_watermark()
        dl_mark = self.downloads_watermark()
        page = self.current_page()
        seg = Segment(index=sub.index, sid=sid, prompt=sub.instantiated_prompt,
                      context=context, mode="replay",
                      # Report the node kind a REPLAY ran under too. It defaulted to
                      # "action" here, which was invisible while only actions could
                      # replay; now that loops cache, progress.json would have called
                      # every replayed loop an action.
                      kind=getattr(sub, "kind", "action"))
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
        # Same re-assert for the callout scroll pin, ungated. Replay needs it MORE than the
        # live path, not less: a replay-only run executes no agent step, so the per-step
        # heal (agent_tools.ensure_callout_scroll_pin) never fires, and a compiled skill
        # clicks a popup opener and fills its field back to back with no LLM pause in
        # between — exactly the window in which a scroll dismissal goes unnoticed. Run
        # 20260824_121419 seg 4 died there: the Net-amount input resolved to "1 match(es),
        # none visible".
        try:
            await page.evaluate(CALLOUT_SCROLL_PIN_JS)
        except Exception as exc:  # noqa: BLE001 - best-effort; replay proceeds anyway
            logger.debug("callout scroll pin injection skipped: %s", exc)
        outcome = await skills.execute(skill, page)
        # Replay never sweeps tabs, so an announced tab already survives here — but
        # unadopted it is nobody's page, and current_page() would hand the gate below (and
        # the next subtask) the APP tab instead. Today no announced-tab subtask can reach
        # this path (the OTP slice consumes noted data, and its prose-only code refuses the
        # commit), which is exactly why the trap is worth closing before it can open.
        await self.adopt_announced_tab(sub)
        if branch and outcome.get("error") and not any(
                e.get("used") for e in (outcome.get("log") or [])):
            # Nothing resolved an element, so the guard's own first control was absent:
            # the condition is not raised. `used` is the tier-agnostic witness — both
            # run_steps and SkillApi._record stamp the winning selector there, while the
            # element-free verbs (wait/press/goto/type) leave it "".
            seg.ok = True
            seg.skip_reason = "branch_absent"
            seg.steps_executed = 0
            seg.gate = {"kind": "branch", "ok": True, "raised": False,
                        "check": str(outcome.get("error"))[:200]}
            seg.duration_seconds = (datetime.now() - started).total_seconds()
            logger.info("branch guard %s: first action found nothing -> condition not "
                        "raised; moving on (no agent, no failure)", sid)
            return seg
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
        next_conditional: str | None = None,
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
            findings=findings, observe=kind == "judge",
            conditional=getattr(sub, "probe", None) is not None,
            aux_tab=getattr(sub, "tab_url", None),
            next_conditional=next_conditional)
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
                # The slice's own declared cadence ("exactly 5 more clicks"), the same number
                # save_steps pins the compiled count to. Handing it to the agent's
                # repeat_click as a BUDGET is what stops a redundant second call from doing
                # the whole pass again (run 20260901_122209 subtask 15: 16 clicks for 5).
                repeat_budget=repeat_hint_from_wording(sub.instantiated_prompt),
            )
        except Exception as exc:  # noqa: BLE001 - a crashed segment is a failed segment
            logger.exception("agent segment %s crashed: %s", sid, exc)
            seg.error = str(exc)
            seg.duration_seconds = (datetime.now() - started).total_seconds()
            return seg
        finally:
            # Order matters, and it is what makes every downstream measurement correct.
            # Adoption resolves the page the agent FINISHED on and installs it as the aux
            # page; the sweep then spares it, and current_page() hands that same page to
            # all four page-dependent consumers — the end_context / url_contains / visible
            # postconditions, the declared verify: checks, and the end_context committed to
            # the library. Run 20260825_105115 seg 3 is what this ordering is for: the OTP
            # handshake succeeded in the portal tab, adoption declined it (two extras, both
            # the same tab opened twice by a replay's force-retry), the sweep took it, and
            # the gate measured the app tab instead. Adoption asking the agent's focus is
            # the fix; a fallback measurement bolted onto the gate was not, because the
            # committed end_context read further out would still have been wrong.
            await self.adopt_announced_tab(sub)
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
        rollup = (receipt_rollup(history, self.requests_since(watermark),
                                 allow_write_refusal=gate.allow_write_refusal)
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
# A click's target name as it appears INSIDE a selector. `:has-text("...")` is in here for
# the row-gate case: a checkbox with no name of its own is anchored by the row's contents
# (`[role="row"]:has-text("Russell Boyle") …`), and that name is run DATA — the employee this
# run generated. Without this branch the binder never saw it, so the entry committed with the
# authoring run's employee baked in and three position-keyed fallbacks behind it, and a
# replay ticked whichever employee happened to sit in that row.
_NAME_IN_SEL = re.compile(r'(?:name|text)="([^"]+)"|:has-text\("([^"]+)"\)')


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
        if action in ("fill", "select", "paste"):
            # `paste` is a fill with a different delivery mechanism (agent_tools.paste_text
            # for a value the page splits across several inputs), so its value is judged
            # for provenance exactly like a typed one — that is what makes a pasted OTP
            # bindable where six 1-character fills were not.
            out.append((str(step.get("value") or ""), "typed"))
        elif action == "type":
            out.append((str(step.get("text") or ""), "typed"))
        elif action == "find_click":
            out.append((str(step.get("text") or ""), "click"))
        elif action == "click":
            # BOTH groups: _NAME_IN_SEL has two alternatives (`name=`/`text="…"` and
            # `:has-text("…")`), so group(1) is None whenever the second one matched.
            # Reading group(1) alone put a None in this list, `value.strip()` below raised,
            # and the commit block's `except Exception` swallowed it — run 20260828_131821's
            # subtasks 6-9 passed their gates and silently saved nothing. str(... or "") is
            # the same coercion every other branch here already applies.
            names = [str(m.group(1) or m.group(2) or "")
                     for s in step.get("selectors") or []
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
        if _NOTED_TOKEN.search(value):
            # Already self-bound (_bind_self_noted): this step reads the live value an
            # earlier step of the same recording captured, so it has the best provenance
            # there is — it never types anything the run did not just observe.
            continue
        if value in flagged or value in out or _names_value(prompt, value):
            continue
        out.append(value)
    return out


def _drop_unattributable_fills(steps: list[dict[str, Any]], loose: list[str]
                               ) -> tuple[list[dict[str, Any]], list[str]]:
    """Remove the typed steps carrying `loose` values, returning (steps, dropped labels).

    An unattributable value is one NEITHER the task wording asked for NOR any observed
    page data supplied — the agent invented it. The Add Employee form's optional County
    field is the standing case: fakenamegenerator's address card has no county, so the
    agent derives one from the postcode ("Merseyside" for L66, run 20260824_101412).

    Baking that literal and trusting archive-on-failure does NOT self-correct: a wrong
    county still saves, the create-write fires, the gate passes, and every later run
    writes the authoring run's county forever. Dropping the step instead fails LOUDLY or
    not at all — an optional field simply stays empty (the correct record), and a field
    that turns out to be required makes the save fail, which archive_if_failing retires
    after two consecutive misses so the next run authors a clean replacement."""
    if not loose:
        return steps, []
    unwanted = {v.strip() for v in loose}
    kept, dropped = [], []
    for step in steps:
        key = "text" if step.get("action") == "type" else "value"
        value = str(step.get(key) or "").strip()
        if step.get("action") in ("fill", "select", "type", "paste") and value in unwanted:
            dropped.append(value)
            continue
        kept.append(step)
    return kept, dropped


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


def _captured_write_bodies(hs: Any, before: int | None = None) -> list[dict[str, Any]]:
    """This run's create-write records that carry a captured response body, in firing
    order (the collector's list is chronological, so 'first create wins' falls out).

    `before` (a network watermark) keeps only the writes that fired BEFORE that point —
    what the BINDER must use: a value bound to the segment's OWN create-response can
    never resolve at load time, because the segment types that value before making the
    request that would return it. Four such bindings on the Add-Employee segment made
    it re-author with the LLM on every run (run 20260813_161952)."""
    net = next((c for c in getattr(hs, "collectors", []) or []
                if getattr(c, "name", "") == "network"), None)
    if net is None:
        return []
    try:
        requests = net.results().get("requests") or []
    except Exception:  # noqa: BLE001 - collector already stopped
        return []
    if before is not None:
        requests = requests[:before]
    return [r for r in requests
            if r.get("body") and r.get("method") in ("POST", "PUT", "PATCH")]


# Binding transforms (2026-08-12): a typed value that is not extract-EQUAL may still
# be derived from one — a LINE (or contiguous line run) of a multi-line block, or a
# strict date REFORMAT. The vocabulary is closed on purpose: no fuzzy matching, and a
# transform that fails on a fresh source resolves to None (the load refuses and the
# segment authors live once) — never a stale or wrong slice.
_DATE_IN_FORMATS = ("%B %d, %Y", "%d %B %Y", "%Y-%m-%d", "%d/%m/%Y")
_DATE_OUT_FORMATS = ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d.%m.%Y")
_DATE_CANDIDATE_RE = re.compile(
    r"[A-Z][a-z]+ \d{1,2}, \d{4}"      # March 2, 1979
    r"|\d{1,2} [A-Z][a-z]+ \d{4}"      # 2 March 1979
    r"|\d{4}-\d{2}-\d{2}"              # 1979-03-02
    r"|\d{1,2}/\d{1,2}/\d{4}")         # 02/03/1979 (day-first — UK app)


def _parse_fuzzy_date(text: Any) -> datetime | None:
    """First parseable date inside `text` under the strict input formats, else None."""
    for cand in _DATE_CANDIDATE_RE.findall(str(text or "")):
        for fmt in _DATE_IN_FORMATS:
            try:
                return datetime.strptime(cand, fmt)
            except ValueError:
                continue
    return None


def _date_out_format(value: Any) -> str | None:
    """The output format `value` is written in, when the WHOLE value is a date."""
    v = str(value or "").strip()
    for fmt in _DATE_OUT_FORMATS:
        try:
            datetime.strptime(v, fmt)
            return fmt
        except ValueError:
            continue
    return None


def _source_lines(src: Any) -> list[str]:
    return [" ".join(ln.split()) for ln in str(src or "").splitlines() if ln.strip()]


def _extract_transform_spec(value: str, extracts: dict[str, str] | None
                            ) -> dict[str, Any] | None:
    """A binding spec deriving `value` from one of `extracts` via the closed transform
    vocabulary (line slice, word slice within a line, date reformat), or None. Exact
    equality is the caller's faster first check.

    The word slice exists because forms SPLIT what pages JOIN: the identity block holds
    "Jude Williamson" and "86 Seaford Road" on single lines, while the Add Employee form
    has separate First/Last and Building/Street fields. Without it those values have no
    structured source at all and the segment can never be committed.

    Matching ignores CASE. fakenamegenerator prints the town in UK postal ALL-CAPS
    ("HOOTON", "BUTT GREEN") and the agent often types it title-cased; treating those as
    different facts left that one value with no source, and ONE unbindable value refuses
    the whole commit — the Add-Employee segment then authored live at ~300-480k tokens a
    run (run 20260819_145019; re-measured across three runs 2026-08-24). The binding
    stores the SOURCE, so a replay types the page's own casing — the same value, spelled
    the way the page spells it."""
    v = str(value).strip()
    vf = v.casefold()
    if v:
        for label, src in (extracts or {}).items():
            lines = _source_lines(src)
            if len(lines) < 2:
                continue
            for i in range(len(lines)):
                for n in range(1, len(lines) - i + 1):
                    for join in (" ", ", "):
                        if join.join(lines[i:i + n]).casefold() == vf:
                            return {"kind": "extract", "label": label,
                                    "transform": {"line": {"index": i, "count": n,
                                                           "join": join}}}
            # No whole-line match: a contiguous WORD run inside one line. Short values
            # (a 2-character house number) bind here, so there is no length floor —
            # instead the match must be UNIQUE in the source. A value that could be read
            # from several positions refuses rather than guessing one (the same rule
            # learn_json_path applies to ambiguous response paths).
            hits: list[dict[str, Any]] = []
            for i, line in enumerate(lines):
                words = line.split()
                for w0 in range(len(words)):
                    for wn in range(1, len(words) - w0 + 1):
                        if " ".join(words[w0:w0 + wn]).casefold() == vf:
                            hits.append({"line": {"index": i, "count": 1, "join": " "},
                                         "words": {"start": w0, "count": wn}})
            if len(hits) == 1:
                return {"kind": "extract", "label": label, "transform": hits[0]}
    fmt = _date_out_format(v)
    if fmt:
        for label, src in (extracts or {}).items():
            d = _parse_fuzzy_date(src)
            if d is not None and d.strftime(fmt) == v:
                return {"kind": "extract", "label": label,
                        "transform": {"date": fmt}}
    return None


def _apply_extract_transform(src: str, transform: dict[str, Any] | None) -> str | None:
    """Derive the bound value from a FRESH source value; None on any failure."""
    if not transform:
        return str(src).strip() or None
    line = transform.get("line")
    if line:
        lines = _source_lines(src)
        i, n = int(line.get("index", 0)), int(line.get("count", 1))
        if not (i >= 0 and n >= 1 and i + n <= len(lines)):
            return None
        out = str(line.get("join", " ")).join(lines[i:i + n]).strip()
        words = transform.get("words")
        if words:
            parts = out.split()
            w0, wn = int(words.get("start", 0)), int(words.get("count", 1))
            if not (w0 >= 0 and wn >= 1 and w0 + wn <= len(parts)):
                return None
            out = " ".join(parts[w0:w0 + wn])
        return out or None
    fmt = transform.get("date")
    if fmt:
        d = _parse_fuzzy_date(src)
        return d.strftime(str(fmt)) if d is not None else None
    return None


def _binding_resolver(run_values: dict[str, str], hs: Any):
    """resolve(spec) -> concrete value from THIS run, or None (refuse the replay)."""
    def resolve(spec: dict[str, Any]) -> str | None:
        try:
            if spec.get("kind") == "extract":
                label = str(spec.get("label"))
                value = (run_values or {}).get(label)
                if not value:
                    # Second look at the run's handoff file: it holds every value this
                    # run has extracted so far, so a resolver created before the
                    # producing segment ran (or a resumed process) still resolves.
                    value = _read_run_values(hs).get(label)
                if not value:
                    return None
                return _apply_extract_transform(str(value), spec.get("transform"))
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
                      if str(v).strip().casefold() == str(value).strip().casefold()), None)
        spec: dict[str, Any] | None = None
        if label is not None:
            spec = {"kind": "extract", "label": label}
        if spec is None:
            # Not extract-equal: a line slice or date reformat of one still binds.
            spec = _extract_transform_spec(value, extracts)
        if spec is None:
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

    return tokenize_steps(steps, replacements), params, bindings


def _sub_whole(text: str, literal: str, token: str) -> str:
    # Boundary-safe: an occurrence flanked by alphanumerics is PART of some other
    # value, not this literal — the house number '35' must never rewrite the
    # unrelated Gross Pay '3500' into '{{bound_N}}00' (run 20260817_115232 replayed
    # Gross Pay as 6300 exactly that way). Lambda replacement keeps the token
    # immune to re.sub escape interpretation.
    return re.sub(r"(?<![A-Za-z0-9])" + re.escape(literal) + r"(?![A-Za-z0-9])",
                  lambda _m: token, text)


# The SELF-scoped twin of a runtime binding (2026-08-26). _bind_runtime_values binds only
# to values EARLIER segments produced, because a binding is resolved when the entry LOADS
# and a segment's own extracts happen after that. A slice that both notes a value and uses
# it — "click Get OTP, copy the 6 digit number ... paste it into the code box" — therefore
# got no binding at all and codegen baked the authoring run's literal: run 20260826_095932
# captured a fresh 776106 and pasted the dead 587923, and the segment still passed because
# the URL is the same either side of the OTP wall.
#
# {{noted:<label>}} is the token that resolves at STEP-EXECUTION time instead: whichever
# tier runs the step reads the label out of the live extract ledger the copy step just
# filled (SkillApi.noted / script_compile.resolve_noted). It is not a param and not a
# binding — there is nothing for the load path to resolve, so a self-noted entry replays
# at zero tokens instead of authoring fresh every run.
_NOTED_TOKEN = re.compile(r"\{\{noted:([A-Za-z0-9_]+)\}\}")


def _bind_self_noted(steps: list[dict[str, Any]], extracted: dict[str, str],
                     ) -> tuple[list[dict[str, Any]], list[str]]:
    """Rewrite each typed value that equals a value THIS recording extracted at an EARLIER
    step into {{noted:<label>}}. Returns (steps, labels bound).

    VALUES only, never selectors: an element's identity must not become a runtime value.
    ORDER matters — a label whose producing step comes after the consumer could not have
    filled the ledger yet, so that value stays a literal and the ordinary provenance guards
    judge it. Matching is case-insensitive (the app echoes 'BUTT GREEN' for 'Butt Green')
    and boundary-safe via _sub_whole (the house-number lesson, run 20260817_115232)."""
    # label -> index of the first step that produces it.
    produced_at: dict[str, int] = {}
    for i, step in enumerate(steps):
        label = str(step.get("label") or "")
        if step.get("action") in ("extract", "copy") and label:
            produced_at.setdefault(label, i)
    if not produced_at:
        return steps, []
    out: list[dict[str, Any]] = []
    bound: list[str] = []
    for i, step in enumerate(steps):
        key = {"fill": "value", "paste": "value", "select": "value",
               "type": "text"}.get(str(step.get("action")))
        raw = str(step.get(key) or "") if key else ""
        if not raw.strip():
            out.append(step)
            continue
        new = raw
        for label, at in produced_at.items():
            src = str((extracted or {}).get(label) or "").strip()
            if not src or at >= i:
                continue
            rewritten = _sub_whole_ci(new, src, "{{noted:%s}}" % label)
            if rewritten != new:
                new = rewritten
                if label not in bound:
                    bound.append(label)
        if new == raw:
            out.append(step)
        else:
            out.append({**step, key: new})
    return out, bound


def _sub_whole_ci(text: str, literal: str, token: str) -> str:
    """_sub_whole, case-insensitively — the app renders a noted value in its own casing."""
    return re.sub(r"(?<![A-Za-z0-9])" + re.escape(literal) + r"(?![A-Za-z0-9])",
                  lambda _m: token, text, flags=re.IGNORECASE)


def tokenize_steps(steps: list[dict[str, Any]],
                   replacements: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Rewrite each (literal -> {{token}}) pair through a step list's values and
    selectors. Split out of _bind_runtime_values so a recompiled recording can be
    re-tokenized against an entry's EXISTING params — the same rewrite, never a
    second implementation of it."""
    rewritten: list[dict[str, Any]] = []
    for step in steps:
        new_step = dict(step)
        for key in ("text", "value", "expect_text"):
            if isinstance(new_step.get(key), str):
                for literal, token in replacements:
                    new_step[key] = _sub_whole(new_step[key], literal, token)
        if new_step.get("selectors"):
            sels = []
            for sel in new_step["selectors"]:
                for literal, token in replacements:
                    sel = _sub_whole(_sub_whole(sel, _esc(literal), token),
                                     literal, token)
                sels.append(sel)
            new_step["selectors"] = sels
        rewritten.append(new_step)
    return rewritten


async def _settled_title(read: Any) -> str:
    """The document title once it stops changing — the only safe input for a pin.

    Reading LIVE is not enough, which is the correction run 20260903_091650_671199 forced.
    This SPA updates document.title a whole navigation behind, and the lag is in the APP,
    not in browser-use's capture — so a live read taken at the instant the segment ends is
    exactly as stale as a recorded one. library/77a0af5f643402f2 ends on the Employees page
    and its commit read "Payroll & RTI - Acting Office - Live Test", the title of the page
    BEFORE it; that became the entry's expected end state. Every run after it false-failed:
    all seven steps ran, the url gate matched "/paye/clients/*/", and only the pin
    disagreed, because evaluate_gate reads the title through _settled() — a polling window
    — and so sees the caught-up "Employees - …". Commit read early, gate read late; they
    could never agree. Four consecutive runs died there.

    The fix is symmetry: give the commit the same settle budget the gate will use, by
    polling until two consecutive reads agree. A title that never settles returns its last
    read rather than hanging — _pin_end_title's own guards still apply to whatever comes
    back, and the pin is demote-only regardless. An unreadable/empty title returns at once:
    it pins nothing anyway, so polling it would be pure delay on every commit.
    """
    last = " ".join(str(await read() or "").split())
    if not last:
        return last
    for _ in range(_SETTLE_TRIES - 1):
        await asyncio.sleep(_SETTLE_DELAY)
        now = " ".join(str(await read() or "").split())
        if now == last:
            return now
        last = now
    return last


def _pin_end_title(start_title: Any, end_title: Any, record_path: Any) -> str | None:
    """The document title to pin as this segment's end state, or None to pin nothing.

    Why a title at all: a segment whose only effect is IN-PAGE has nothing else to prove it
    worked. The OTP slice pasted a dead code, clicked Proceed Securely and PASSED — the
    paste landed, the button was there, and the URL is identical either side of the wall
    (run 20260826_095932). Three other channels were measured dead first: the response body
    is never captured on acceptance (the page navigates and tears down before it can be
    read — the ONLY body ever captured came from the rejected run), console/401 is identical
    in both, and the recording's own url and tab count are byte-identical across the
    deciding step. The title is not: the wall sets no <title> so it shows the raw URL, while
    the accepted view shows "Acting Office - Live Test".

    BOTH titles must be read LIVE — never from the recording. browser-use captures an item's
    state before that item's actions, and this SPA updates document.title asynchronously, so
    a recorded title can be a whole page stale: subtask 0's final `done` item records url
    ".../rti/payrun" next to title "Dashboard - …". Pinning that false-failed a correct
    segment on the first live run after it shipped.

    Live is NECESSARY BUT NOT SUFFICIENT, as this docstring assumed until 2026-09-03: the
    lag is the app's, not the capture's, so an immediate live read at segment end carries
    the same stale value. The `end_title` handed in must come from _settled_title — see
    there for the run that proved it.

    Pinned conservatively — this is an ADDITIVE gate, and a false-fail here is worse than
    the blind spot it closes:
      - only when the title CHANGED across the segment (an unchanged title proves nothing);
      - only when it has no digits, so a record ref or id can never be baked in. NOTE the
        digit rule does NOT make the learn-on-replay path safe, as this docstring claimed
        until 2026-08-28: the OTP portal's ACCEPTED view is titled "Employee Approval
        Request - Acting Office - Live Test" — clean text, no digits — and run
        20260828_124929 learned exactly that from a leftover portal tab, turning a wrong
        page into the expected end state of an unrelated subtask. What stops that class now
        is registration: a learner may refine an entry the commit path registered, never
        invent one (subtask_store.update_manifest's `create` flag). A leftover tab can still
        mislead the learner for an entry that IS registered — the durable fix is to read the
        title from the page the segment ACTED on. NOTE the network write gate's page
        attribution is a different lens and does NOT supply this one: it asks WHEN (a write
        issued during a page load, before the segment touched the page — checks.business_writes),
        not WHICH PAGE. An `acted_urls` lens was designed for that gate on 2026-08-21 and
        never reached a commit on any branch;
      - never for a segment that closed its own page — its surviving title is the same coin
        flip as its surviving url (see _recording_closed_its_page).
    """
    start = " ".join(str(start_title or "").split())
    end = " ".join(str(end_title or "").split())
    if not start or not end or start == end:
        return None
    if any(ch.isdigit() for ch in end):
        return None
    if record_path is not None and _recording_closed_its_page(record_path):
        return None
    return end


def _recording_closed_its_page(path: Any) -> bool:
    """Did the authoring history end with FEWER tabs than it had — i.e. did the segment
    close the page it was working in?

    The mirror of script_compile._stamp_opens_tab, which reads the same `state.tabs`
    growth to spot a tab OPENING. Used at the commit site: a segment whose last
    instruction closes its own page ("… click submit, and then close this tab") has no end
    location, so whatever current_url() reports afterwards is an accident of which target
    survived. Deciding that from the RECORDING rather than from the surviving URL is the
    whole point — run 20260826_124837 survived on about:blank and run 130102 on the app's
    own datarequests tab, and both are accidents of the same segment. A guard that only
    rejected degenerate-LOOKING urls would have committed 130102's ordinary
    '/paye/clients/*/datarequests' and failed every blank-survivor run after it.

    Fails OPEN (unreadable -> False): guessing "closed" would silently drop a legitimate
    postcondition gate, which is the more dangerous mistake.
    """
    try:
        history = json.loads(Path(path).read_text()).get("history") or []
    except Exception:  # noqa: BLE001 - never fail the commit path over a diagnosis
        return False
    counts = [len((item.get("state") or {}).get("tabs") or []) for item in history]
    counts = [c for c in counts if c]          # items that recorded no tab list say nothing
    return len(counts) >= 2 and counts[-1] < max(counts)


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


def _refused_write_step(requests_window: list[dict[str, Any]]) -> int | None:
    """The agent step whose business write the SERVER refused, or None.

    Marks the boundary between a slice's work and the error branch it declares ("if it
    shows an error, click cancel"): everything the agent did after this step was closing
    that dialog. Reuses the write rule's own lenses — business_writes drops infrastructure
    and page-load traffic, _write_verdict reads the 2xx bodies that refuse — so it can
    never disagree with the gate that set write_refusal_waived in the first place.

    The LAST refusal wins: an agent that retried was cancelling the final one.
    """
    from automation.pipeline.agent_tools import _write_verdict  # lazy (browser stack)
    latest: int | None = None
    for r in business_writes(requests_window):
        step = r.get("step")
        if not isinstance(step, int):
            continue
        status = r.get("status")
        if r.get("failed") or (isinstance(status, int) and status >= 400):
            latest = step
            continue
        if not (isinstance(status, int) and 200 <= status < 400):
            continue
        verdict = _write_verdict(r)
        if verdict is not None and verdict[0]:
            latest = step
    return latest


async def _author_segment(
    hs: HybridSession, sub: Subtask, sid: str, context: str, gate: Gate, *,
    completed: list[str], remaining: list[str],
    dirty: bool = False, prior_failure: str | None = None,
    findings: list[str] | None = None, commit: bool = True,
    run_values: dict[str, str] | None = None,
    start_url: str | None = None, start_title: str | None = None,
    dynamic: bool = False, next_conditional: str | None = None,
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
    # Where the network log stood BEFORE this segment acted: the binder may only bind
    # typed values to data that already existed (see _captured_write_bodies).
    writes_before = hs.network_watermark() if hasattr(hs, "network_watermark") else None
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
        record_path=rec_tmp, next_conditional=next_conditional,
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
    optional_from: int | None = None
    if seg.gate.get("write_refusal_waived"):
        # Passed only because the slice declares its own error branch (see
        # Gate.allow_write_refusal): this trace ends on that branch — the refusal dialog
        # and the Cancel that closes it. Refusing to cache the whole segment for that
        # reason cost the payroll e2e ~400k tokens a run (the bulk-FPS subtask, whose FPS
        # is always already submitted) AND re-keyed the subtask after it, since sids hash
        # the start context and a live authoring ends wherever it ends. So cache the WORK
        # and mark the branch: everything after the refused write compiles `optional`, and
        # a run whose write IS accepted — no dialog to cancel — skips it instead of
        # failing (script_compile.compile_recording's `optional_from`).
        optional_from = _refused_write_step(
            hs.requests_since(writes_before)
            if writes_before is not None and hasattr(hs, "requests_since") else [])
        if optional_from is None:
            # No refused write we can point at, so we cannot say where the work stops and
            # the branch starts. Keep the old behaviour rather than guess a boundary.
            print(f"[*] segment [{sid}]: not cached — its write was refused and the slice "
                  f"declares that as an acceptable ending, so this trace records the error "
                  f"branch, not the work")
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
                           max_steps=truncate_at, emit_start_goto=False,
                           # "exactly N clicks" wording pins a lone repeat cluster's
                           # count — the recorded count can be short one collapsed retry.
                           repeat_hint=repeat_hint_from_wording(sub.instantiated_prompt),
                           optional_from=optional_from)
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
        unanchorable = [s for s in steps if s.get("action") == "unanchorable"]
        if unanchorable:
            # A click the compiler could not tie to an exact location. Committing it
            # would mean recording a TEXT SEARCH, and a replay that hunts for elements
            # lands on the wrong one (find_click('Net to gross') → 36 matches). Refuse
            # the whole segment: it authors live every run until the click can be
            # anchored, which is honest and never wrong.
            sstore.steps_path(sid).unlink(missing_ok=True)
            print(f"[*] segment [{sid}]: not cached — "
                  f"{unanchorable[0].get('why', 'a step has no anchorable element')}; "
                  f"a recording must locate elements, never search for them")
            return seg
        # Producer commit guard — the safety the node_kind producer rule depends on. A
        # noting slice is cacheable only because its compiled EXTRACT step re-reads the
        # page every replay; a recording that noted the value in prose alone carries no
        # such step, so its replay would report nothing, the run's findings would go
        # quiet, and every downstream consumer would fall through to replaying THIS run's
        # stale values (the consumer gate keys on `bool(findings)`). Refuse instead: the
        # slice authors live each run, exactly as it does today.
        # Provenance commit guard — the general, wording-free memory rule: a segment
        # that acted with values sourced from the run's FINDINGS consumed runtime data,
        # and a cached replay would re-use this run's values forever. Since 2026-07-24
        # the guard BINDS before it refuses: a flagged value with a structured source
        # (an extract this run captured, or a unique leaf in a create-write response)
        # is rewritten to a {{bound_N}} param resolved fresh from EACH run's own data
        # at load time. Only prose-only values still refuse the commit — those segments
        # keep authoring fresh every run.
        # Judge values against the WHOLE task's wording, not just this slice's: a value
        # another slice spells out ("note ... the Gender (Male)") is prompt data even
        # when this slice only says "enter the noted gender". Judging it slice-locally
        # made it runtime data with no structured source, and ONE unbindable value
        # refuses the entire commit — which is why the Add-Employee recording never
        # cached (run 20260814_100546). The create-write leg below already uses this
        # corpus; the findings leg now agrees with it.
        task_wording = " \n ".join([sub.instantiated_prompt, *completed, *remaining])
        # SELF-scoped leg, FIRST: a value this recording extracted at an earlier step and
        # then typed becomes {{noted:label}}, resolved when the step runs. It must precede
        # every guard below — once the literal is a token it is neither an unattributable
        # typed value nor something the load-time binder should try to resolve, and the
        # merged OTP slice can finally cache instead of re-authoring (or, worse, caching
        # the authoring run's dead code as it did through run 20260826_095932).
        steps, self_noted = _bind_self_noted(steps, seg.extracted or {})
        if self_noted:
            _atomic_write(sstore.steps_path(sid), json.dumps(steps, indent=2))
            print(f"[*] segment [{sid}]: value(s) this segment noted itself "
                  f"({', '.join(self_noted)}) bound to the live extract -> replays with "
                  f"each run's own fresh value")
        runtime_values = _findings_sourced_values(steps, task_wording, findings)
        # Findings-independent leg: values the run's own create-writes reported are
        # runtime data even when no finding names them (the findings channel goes
        # quiet once the producer segments replay).
        bodies = _captured_write_bodies(hs, before=writes_before)
        for value in _body_sourced_values(steps, task_wording, bodies):
            if value not in runtime_values:
                runtime_values.append(value)
        # Structured-source leg: a typed value that equals an extract an EARLIER
        # segment produced — or derives from one via the closed transforms (a line of
        # a block, a date reformat) — is runtime data even when no finding prose names
        # it. Flagging it here both feeds the binder and stops the consumer gate below
        # from refusing the exact values bindings can now carry. Deliberately PRIOR
        # sources only (run_values): a segment's own extracts happen after its load,
        # so a self-referential binding could never resolve at replay time.
        prior_sources = dict(run_values or {})
        for value, kind in _step_value_candidates(steps):
            v = value.strip()
            if (kind == "typed" and len(v) >= 3 and v not in runtime_values
                    and not _names_value(sub.instantiated_prompt, v)):
                if any(str(s).strip().casefold() == v.casefold()
                       for s in prior_sources.values()) \
                        or _extract_transform_spec(v, prior_sources) is not None:
                    runtime_values.append(v)
        sources = {**(run_values or {}), **(seg.extracted or {})}
        if dynamic:
            # The wording DECLARES consumption of noted data, so this commit is held to
            # a stricter bar than the substring guard alone: every typed value must be
            # prompt-sourced or flagged for binding, and there must be something to bind.
            # An unattributable value may be runtime data the guard cannot see (a
            # re-formatted date), and a consumer recording with no bindable value at all
            # keeps the pre-bindings behavior: author fresh every run.
            loose = _unattributed_typed_values(steps, task_wording, runtime_values)
            if loose:
                # User-approved 2026-08-24: DROP the invented value rather than refuse the
                # whole commit over it. Refusing cost the Add Employee segment its entry on
                # every run where the agent filled the optional County field (~300-480k
                # tokens re-authoring); baking the literal instead would pass forever while
                # writing the authoring run's county into every later employee. See
                # _drop_unattributable_fills for why dropping is the self-correcting one.
                steps, dropped = _drop_unattributable_fills(steps, loose)
                if dropped:
                    print(f"[*] segment [{sid}]: dropped unattributable typed value(s) "
                          f"{', '.join(v[:32] for v in dropped[:3])} from the recording "
                          f"(neither the task nor the page supplied them) -> replays with "
                          f"those fields left as the form defaults them")
            if not runtime_values and not self_noted:
                # self_noted counts: those values ARE bound, just at step-execution time
                # rather than at load time. Refusing over them sent the merged OTP slice
                # back to the agent on every run that happened to carry an earlier
                # finding (~240k tokens, 405s — run 20260826_091931).
                sstore.steps_path(sid).unlink(missing_ok=True)
                print(f"[*] segment [{sid}]: consumes noted data with no bindable runtime "
                      f"value in the recording -> not cached; future runs author it with "
                      f"their own fresh values")
                return seg
        bound = None
        if runtime_values:
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
        # like its start context. A DECLARED helper tab is still live here (the loop closes
        # it only after the segment commits), but an ANNOUNCED tab that adopt_announced_tab
        # declined has already been swept by agent_segment, so current_url() would read the
        # app tab that survived. Committing THAT as end_context is worse than a failed
        # segment: _gate_for turns a stored end_context into the next run's postcondition
        # gate, so one such commit bakes a wrong gate into every future run of this subtask.
        # Prefer the URL captured before the sweep, for the same reason the gate does.
        normalize = (sstore.normalize_aux_context if getattr(sub, "tab_url", None)
                     else sstore.normalize_context)
        raw_end_url = await hs.current_url()
        # A segment that closed the page it worked in has NO end location, so there is
        # nothing honest to pin. Two limbs, and the first is the load-bearing one:
        #
        #   1. the RECORDING says the tab count shrank. This is a fact about what the
        #      segment did, not an observation of what happened to survive — and the
        #      survivor is a coin flip: subtask 8 of run 20260826_124837 ended on
        #      about:blank, run 130102 on the app's own datarequests tab. Judging by the
        #      url's shape alone would have committed 130102's perfectly ordinary
        #      '/paye/clients/*/datarequests' and then failed every blank-survivor run —
        #      the same bug with the values swapped.
        #   2. the surviving url names no location anyway (blank / unreadable / closed),
        #      which also covers a page that died for an unrelated reason.
        #
        # Either way commit end_context=None explicitly rather than omitting the key:
        # update_manifest merges, so None is what CLEARS a value an earlier poisoned
        # commit left behind. _base_gate then falls through to Gate(kind="steps") — which
        # is exactly what run 123706 did, and it passed.
        if _is_degenerate_url(raw_end_url) \
                or _recording_closed_its_page(sstore.recording_path(sid)):
            end_context = None
            print(f"[*] segment [{sid}]: ends by closing its own page -> no end context "
                  f"pinned (judged on clean execution instead)")
        else:
            end_context = normalize(raw_end_url)
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
            # An optional tail means the recorded ending is one of TWO endings (the
            # error branch ran this time; a later run may skip it and stop earlier), so
            # there is no single title to pin — the same reasoning as a segment that
            # closed its own page.
            end_title=(None if any(st.get("optional") for st in steps)
                       else _pin_end_title(start_title,
                                           await _settled_title(hs.current_title),
                                           sstore.recording_path(sid))),
            marker_write=gate.kind == "marker", steps=len(steps), provenance="clean",
        )
        if self_noted:
            # Self-describing, and the flag the stale-recording net reads: an entry that
            # re-reads its own noted values types nothing from the authoring run.
            manifest_fields["self_noted"] = self_noted
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
        # create=True: this is the ONE call that registers an entry, and the only one
        # holding a complete one (context, start_url, steps, params, bindings).
        sstore.update_manifest(sid, sub.template_prompt, create=True, **manifest_fields)
        # Tier-1 upgrade (best-effort): transpile the committed steps into a code skill
        # (<sid>.skill.py + anchors). On any failure the steps stay authoritative and any
        # stale code artifacts are removed (codegen.compile_code_skill owns that).
        skills.compile_code_skill(sid)
        logger.info("segment %s: committed %d steps to the library", sid, len(steps))
    except Exception as exc:  # noqa: BLE001 - compile failure must not fail a passed segment
        # PRINTED as well as logged. Swallowing is right — a compile bug must not fail a
        # segment that genuinely passed — but a silent swallow makes "saved nothing" look
        # exactly like "saved fine" in the run output, and the deliberate refusals above all
        # print. Run 20260828_131821 lost four commits to a swallowed AttributeError and the
        # only clue was a line in the log file.
        logger.exception("segment %s compile/commit error: %s", sid, exc)
        print(f"[*] segment [{sid}]: NOT cached — the commit hit an unexpected error "
              f"({type(exc).__name__}: {exc}). The segment itself passed; this is a "
              f"framework bug, not a task failure.")
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
            raw_start_title = await hs.current_title()
            context = (sstore.normalize_aux_context(aux_url) if aux_url
                       else sstore.normalize_context(raw_start_url))
            sid = sstore.subtask_id(sub.template_prompt, context)
            force_author = reauthor_match(reauthor, sub)
            is_judge = getattr(sub, "kind", "action") == "judge"
            # A leading-"If" branch guard: whether its actions run at all depends on live
            # page state. Such a slice DOES record and replay like any other — what makes
            # that safe is `branch=True` below, which lets a replay whose first action
            # finds nothing resolve as "condition not raised" instead of failing the
            # segment into an agent takeover (see replay_segment). A declared `probe:` is
            # the cheaper form of the same judgment: it answers the question before the
            # page is touched at all. Wording is the fallback for a slice that declares
            # neither, so an unprobed guard is protected too.
            is_branch = (getattr(sub, "probe", None) is not None
                         or is_conditional_guard(sub.template_prompt))
            # Conditional too is declaration-driven: a `probe:` in tasks.yaml, never a
            # leading "If" in the prose.
            is_conditional = getattr(sub, "probe", None) is not None
            # ... unless the slice declares a `probe:` — a deterministic check that
            # replaces the agent's live presence judgment. A FALSE probe resolves the
            # segment as a zero-LLM no-op; a TRUE probe lets the branch replay/commit
            # like an action, because recorded TRUE-branch steps then only ever run
            # behind a TRUE probe. Ignored on non-conditional slices.
            probe = getattr(sub, "probe", None) if is_conditional else None
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
            # No consumer gate: a slice saying "the noted employee's name" is an ordinary
            # recorded action now (user decision 2026-08-28 — wording must not decide
            # recordability). What still protects a cached consumer from replaying the
            # AUTHORING run's values is the provenance commit guard below, which judges the
            # values a recording actually typed against extracts the run actually produced —
            # a measured fact, not a reading of the prose. A value it cannot bind still
            # refuses the commit, so that slice keeps authoring live.

            # Semantic routing: a wording with NO direct entry may still be a known
            # procedure (alias table -> local embeddings -> one LLM verify). A routed sid
            # replays the canonical skill with values re-keyed to its params; authoring
            # after a routed failure goes under the ORIGINAL sid so the canonical entry
            # is never overwritten by a different wording's recording. Aux subtasks are
            # direct-hit only (v1): a routed canonical entry could have been recorded on
            # a different site family.
            author_sid, route_values = sid, None
            if (not fresh and not force_author and not is_judge
                    and not is_conditional
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
            next_conditional, n_cond = _describe_next_conditionals(subtasks, i)
            # The probed successors are DESCRIBED to this slice as an expected outcome,
            # so they must not ALSO appear in the still-ahead list: that line hands the
            # agent the successor's action words verbatim ("click Cancel to close it")
            # under a generic do-not-start rule it stops honouring the moment it believes
            # its own step failed (run 20260901_151214). With no probed successor the
            # slice is subtasks[i + 1:], byte-identical to before.
            remaining = [s.instantiated_prompt for s in subtasks[i + 1 + n_cond:]]
            seg: Segment | None = None
            skip_reason: str | None = None   # why this subtask did not replay



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
                if probe is not None:
                    if await hs.probe_condition(probe):
                        print(f"[*] subtask {i} [{sid}]: probe {probe.kind} "
                              f"\"{probe.arg}\" PRESENT -> condition raised; running "
                              f"the branch")
                    else:
                        skip_reason = "probe_absent"
                        print(f"[*] subtask {i} [{sid}]: probe {probe.kind} "
                              f"\"{probe.arg}\" absent -> condition not raised; "
                              f"moving on (no agent, no replay)")
                        seg = Segment(
                            index=sub.index, sid=sid, prompt=sub.instantiated_prompt,
                            context=context, mode="probe",
                            kind=getattr(sub, "kind", "action"), ok=True,
                            gate={"kind": "probe", "ok": True, "present": False,
                                  "check": f'{probe.kind} "{probe.arg}"'},
                            skip_reason="probe_absent")

                # Judge and conditional-guard nodes never touch the library in EITHER
                # direction: a replayed judge would click through with nobody looking
                # (hollow pass) and a replayed conditional would take its branch
                # unconditionally — and their recordings must never be committed for the
                # same reasons. Exception: a conditional WITH a declared probe is
                # action-like on the TRUE path — the probe above already decided the
                # branch is raised, so its recording replays/commits safely.
                #
                # LOOPS DO CACHE (user decision 2026-08-25). What the library stores is
                # the iteration COUNT the authoring agent stopped at, and the user's
                # position is that a recording replays in the same setting it was made
                # in, so that count holds. Overshoot is already honest — repeat_click's
                # readiness poll raises "the page likely stopped advancing" when the
                # control runs out early. An UNDER-run (a longer list than the recording
                # saw) stops quietly at the cached count; no guard was built for it, by
                # decision.
                if seg is None and not fresh and not force_author and not is_judge \
                        and (not is_conditional or probe is not None) \
                        and sstore.has_script(sid):
                    load_sub = sub if route_values is None else SimpleNamespace(
                        instantiated_prompt=sub.instantiated_prompt, values=route_values)
                    skill = skills.load_skill(sid, load_sub,
                                              run_resolver=resolve_binding)
                    if skill is not None:
                        print(f"[*] subtask {i} [{sid}]: library hit -> replay "
                              f"({len(skill)} steps, no LLM)")
                        seg = await hs.replay_segment(sub, sid, context, skill,
                                                      gate, branch=is_branch)
                        if seg.ok:
                            from_template = bool((entry or {}).get("params"))
                            _promote_segment_heals(sid, seg, from_template=from_template)
                            if not (entry or {}).get("end_title"):
                                # Learn the end-title pin the same way heals are promoted.
                                # An entry committed before this gate existed replays
                                # forever and would never otherwise gain one — including
                                # library/07044b6a0dbf7988, the OTP slice this was built
                                # for. A replay that PASSED demonstrably reached the
                                # intended end state, so the live title is trustworthy
                                # (and better evidence than a recorded one, which lags —
                                # see _pin_end_title). _pin_end_title's digit rule is what
                                # makes it safe: a false pass leaves the page on the OTP
                                # wall, whose title is a raw URL full of digits, so the
                                # wrong title can never be learned.
                                learned = _pin_end_title(
                                    raw_start_title,
                                    await _settled_title(hs.current_title),
                                    sstore.recording_path(sid))
                                if learned:
                                    sstore.update_manifest(sid, sub.template_prompt,
                                                           end_title=learned)
                                    print(f"[*] subtask {i} [{sid}]: learned end title "
                                          f"{learned!r} from this passing replay")
                            sstore.bump_meta(sid, uses=1)
                        else:
                            sstore.bump_meta(sid, fail_count=1)
                            dirty = seg.steps_executed > 0
                            print(f"[*] subtask {i} [{sid}]: replay FAILED "
                                  f"({seg.error}) -> agent takes over in place"
                                  f"{' (dirty state)' if dirty else ''}")
                            if not dirty and route_values is None:
                                # Failed before touching the page: the recovery run starts
                                # from the entry's declared context, so it IS a clean
                                # re-author. Retire the old entry only after REPEATED
                                # consecutive failures (bump_meta above counts them and
                                # resets on any success) — deleting it on the first miss
                                # meant one slow render or un-populated list destroyed a
                                # working recording for good, which is why recordings kept
                                # "not working the next day". A successful re-author
                                # overwrites the entry anyway.
                                # (A ROUTED failure never archives the canonical entry —
                                # the wording mapping may be at fault, not the recording.)
                                sstore.archive_if_failing(
                                    sid, threshold=_ARCHIVE_AFTER_FAILURES)
                            prior = seg.error
                            # A failed replay can still have READ data off the page
                            # before it died (extract steps run before the step that
                            # fails). `seg` is rebound to the authored segment below,
                            # so harvest those values now or they are lost — and they
                            # are exactly what a later segment's bindings resolve
                            # against.
                            if seg.extracted:
                                run_values.update({str(k): str(v)
                                                   for k, v in seg.extracted.items()})
                                _write_run_values(hs, run_values)
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
                                start_url=raw_start_url,
                                start_title=raw_start_title, dynamic=False,
                                next_conditional=next_conditional)
                            seg.mode = "replay_failed->authored"
                            # Why the replay failed used to die with the rebound
                            # segment — stdout only, no artifact. Keep it on the record.
                            seg.replay_error = prior
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
                    elif is_conditional and probe is None:
                        skip_reason = "conditional"
                        print(f"[*] subtask {i} [{sid}]: conditional branch guard -> agent "
                              f"runs it live, never cached (declare a probe: to make "
                              f"it replayable)")
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
                                start_title=raw_start_title,
                                                dynamic=False,
                                                next_conditional=next_conditional,
                                                # A fallback blob never commits: a whole-
                                                # task recording replayed blind is the
                                                # pre-hybrid behavior this mode degrades
                                                # FROM, not a library asset.
                                                commit=not is_judge
                                                and (not is_conditional
                                                     or probe is not None)
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
                _write_run_values(hs, run_values)
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
    probe_skipped = sum(1 for s in segments if s.mode == "probe")
    authored = len(segments) - replayed - probe_skipped
    # Break the authored count down by WHY each segment did not replay — a bare
    # "0 replayed" hides whether the cache was cold, bypassed (--fresh), or the
    # subtasks are live-by-kind. Probe-resolved no-ops are neither replayed nor
    # authored and get their own count.
    reasons = Counter(s.skip_reason for s in segments
                      if s.skip_reason and s.mode != "probe")
    breakdown = ", ".join(f"{n} {r}" for r, n in sorted(reasons.items()))
    result.final_result = (
        f"Hybrid run: {len(segments)}/{len(subtasks)} subtasks "
        f"({replayed} replayed, "
        + (f"{probe_skipped} probe-skipped, " if probe_skipped else "")
        + f"{authored} authored"
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
