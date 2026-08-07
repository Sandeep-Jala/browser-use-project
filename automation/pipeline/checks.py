"""Deterministic per-subtask checks (the verification path — LLM-free by design).

A `verify:` block on a declared subtask lists checks that must ALL hold at segment
end (implicit AND). Evaluation is Predicate-style "retry verification, not actions":
each check re-polls briefly (up to its `timeout_s`, default 10 s) so an async SPA or
an in-flight write can settle, then fails closed — an unevaluable probe is a failed
check, never a pass. Checks can only DEMOTE a segment the base gate passed; they
never resurrect a failed one (enforced by the caller, evaluate_gate).

Page checks run the same shadow-DOM-aware raw finders the authoring tools and replay
use (RAW_FIND_JS / RAW_TEXT_FIND_JS); write checks mirror _first_create_write's
2xx/3xx create-write scan plus _write_verdict's body reading, over the SEGMENT's own
network window — whose records are live dicts, so re-polling sees late settles.

Top-level imports stay stdlib-only so `tasks.py → parse_verify` never drags
browser_use/playwright into task loading; probe constants and network helpers are
imported lazily inside the evaluators.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

_CHECK_KINDS = ("text_visible", "text_absent", "control_exists", "url_contains",
                "write_accepted")
_CHECK_TIMEOUT_S = 10.0
_CHECK_POLL_S = 0.5


@dataclass(frozen=True)
class Check:
    kind: str
    arg: str
    timeout_s: float = _CHECK_TIMEOUT_S


def parse_verify(raw: Any, *, where: str = "verify") -> tuple[Check, ...]:
    """tasks.yaml `verify:` list → validated Check tuple. Bad declarations fail loud
    at load time (same contract as _spec_from_entry) — a broken check must never
    surface as a silent always-pass or a mid-run type error."""
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            f"{where}: must be a list of checks "
            f"(e.g. \"- text_visible: 'Alistair Allan'\"), got {type(raw).__name__}")
    if not raw:
        raise ValueError(f"{where}: declare at least one check or drop the key")
    checks: list[Check] = []
    for i, item in enumerate(raw):
        label = f"{where}[{i}]"
        if not isinstance(item, dict):
            raise ValueError(
                f"{label}: each check must be a mapping like "
                f"\"- text_visible: '…'\", got {item!r}")
        kinds = [k for k in item if k in _CHECK_KINDS]
        if len(kinds) != 1:
            raise ValueError(
                f"{label}: exactly one check kind per item (one of "
                f"{', '.join(_CHECK_KINDS)}); got keys {sorted(item)}")
        kind = kinds[0]
        stray = [k for k in item if k not in (kind, "timeout_s")]
        if stray:
            raise ValueError(
                f"{label}: unknown key(s) {stray} — only the check kind and "
                f"timeout_s are allowed")
        arg = item[kind]
        if not isinstance(arg, str) or not arg.strip():
            raise ValueError(f"{label}: {kind} needs a non-empty string value")
        timeout = item.get("timeout_s", _CHECK_TIMEOUT_S)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) \
                or timeout <= 0:
            raise ValueError(
                f"{label}: timeout_s must be a positive number, got {timeout!r}")
        checks.append(Check(kind=kind, arg=arg.strip(), timeout_s=float(timeout)))
    return tuple(checks)


async def evaluate_checks(page: Any, requests_window: list[dict[str, Any]],
                          checks: tuple[Check, ...], *,
                          poll: bool = True) -> list[dict[str, Any]]:
    """Evaluate every check, in declared order, against the live page and the
    segment's network window. `poll=False` gives each check its single honest
    evaluation (the failed-base-gate convention — outcome can't change, but the
    detail still lands in the report)."""
    return [await _evaluate_one(page, requests_window, check, poll=poll)
            for check in checks]


async def _evaluate_one(page: Any, requests_window: list[dict[str, Any]],
                        check: Check, *, poll: bool) -> dict[str, Any]:
    deadline = time.monotonic() + check.timeout_s
    while True:
        try:
            ok, evidence, error, final = await _probe(page, requests_window, check)
        except Exception as exc:  # noqa: BLE001 - an unevaluable probe fails closed
            ok, evidence, error, final = False, None, str(exc) or repr(exc), True
        if ok or final or not poll or time.monotonic() >= deadline:
            if not ok and error is None:
                error = (f'{check.kind} "{check.arg}" not satisfied within '
                         f"{check.timeout_s:g}s")
            return {"kind": check.kind, "arg": check.arg, "ok": ok,
                    "evidence": evidence, "error": error}
        await asyncio.sleep(_CHECK_POLL_S)


async def _probe(page: Any, requests_window: list[dict[str, Any]],
                 check: Check) -> tuple[bool, str | None, str | None, bool]:
    """One evaluation of one check: (ok, evidence, error, final). `final=True` stops
    the eventually-poll (hard failures: no page handle, probe error) — a miss that
    time might fix keeps final=False."""
    if check.kind == "url_contains":
        if page is None:
            return False, None, "no page handle available", True
        url = str(page.url)
        return check.arg.lower() in url.lower(), url, None, False
    if check.kind == "write_accepted":
        return _probe_write(requests_window, check.arg)
    # The three raw-finder page probes share one result contract.
    if page is None:
        return False, None, "no page handle available", True
    from automation.pipeline.script_compile import (  # lazy: keeps import light
        RAW_FIND_JS, RAW_TEXT_FIND_JS, _query_tokens,
    )
    tokens = _query_tokens(check.arg)
    if not tokens:
        return False, None, f"no searchable tokens in {check.arg!r}", True
    if check.kind == "control_exists":
        expr = RAW_FIND_JS % (json.dumps(tokens), "false")
    else:
        expr = RAW_TEXT_FIND_JS % json.dumps(tokens)
    raw = await page.evaluate(expr)
    if raw and raw.get("error"):
        return False, None, str(raw["error"]), True
    count = int((raw or {}).get("count") or 0)
    name = " ".join(str((raw or {}).get("name") or "").split())
    if check.kind == "text_absent":
        if count == 0:
            return True, f'"{check.arg}" is absent', None, False
        return False, f'still visible: "{name or check.arg}"', None, False
    if count >= 1:
        noun = "control" if check.kind == "control_exists" else "text"
        return True, f'{noun} "{name or check.arg}" visible', None, False
    return False, None, None, False


def _probe_write(requests_window: list[dict[str, Any]],
                 frag: str) -> tuple[bool, str | None, str | None, bool]:
    """Mirror of _first_create_write (POST/PUT/PATCH + URL fragment + 2xx/3xx),
    plus _write_verdict body reading: a 2xx whose captured body refuses is NOT
    accepted. Window records are live dicts — the caller's poll loop re-reads them,
    which is how an in-flight write gets its chance to settle."""
    from automation.pipeline.agent_tools import _format_write, _write_verdict  # lazy
    refusal: str | None = None
    for r in requests_window:
        if r.get("method") not in ("POST", "PUT", "PATCH"):
            continue
        if frag.lower() not in str(r.get("url", "")).lower():
            continue
        if not 200 <= (r.get("status") or 0) < 400:
            continue
        verdict = _write_verdict(r)
        if verdict is not None and verdict[0]:
            refusal = verdict[1]
            continue
        return True, _format_write({"record": r}), None, False
    if refusal is not None:
        return False, None, f'the server REFUSED the write: "{refusal}"', False
    return (False, None,
            f'no accepted create-write matching "{frag}" in this segment\'s traffic',
            False)
