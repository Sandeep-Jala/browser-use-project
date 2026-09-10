"""Declarative post-run assertions over the collected telemetry.

Pure functions over RunResult-shaped data — no I/O, no Playwright. A spec is a small dict of
rules (DEFAULT_SPEC globally, overridable per task via TaskSpec.assertions); `evaluate`
turns it into AssertionResults with the offending requests/console entries as evidence, and
`apply` stamps them onto a RunResult.

Assertions gate a SEPARATE `assertions_passed` verdict, never `is_successful`:
`is_successful` is load-bearing control flow (template commits, heal promotion, and
record-or-not all key off it) and answers "did the workflow complete and save"; assertions
answer "was the app healthy while it did". A console error must not stop a perfectly good
script from being committed. The combined verdict for reports/exit codes is
`is_successful and assertions_passed is not False`.

Rule semantics (a rule set to None or False is disabled and not evaluated):
  no_5xx: True                — no response with a 5xx status
  no_failed_requests: {"allow_url_patterns": [...]}   — no network-level failures (aborted/
                                DNS/timeout), except URLs matching an allowlist regex
  no_console_errors: {"allow_patterns": [...]}        — no console errors / uncaught page
                                errors, except messages matching an allowlist regex
  max_http_4xx: int           — at most N 4xx responses (0 = none)
  response_ok: {"method": "POST", "url_contains": "...", "status_range": [200, 399]}
                              — at least one matching request succeeded (generalizes the
                                ground-truth create-write gate to extra endpoints; opt-in
                                per task so it never duplicates the marker check)
  scope: {"hosts": [...]}     — NOT a rule: restricts what every rule above sees. Network
                                records count only when the REQUEST url's host is one of
                                the hosts (or a subdomain of one); console entries count
                                only when the PAGE that produced them is in scope. The
                                assertions judge the app under test, so helper-tab sites
                                (fakenamegenerator + its ad stack: 312 failed requests,
                                277 "console errors" in one live run) and third-party
                                trackers must not fail a healthy run. The runner injects
                                the app's own host by default (scope_hosts_for); a task
                                may override or disable it (scope: None = everything).

A malformed allowlist regex falls back to substring matching (surfaced in the assertion
detail) — a spec typo must never crash a run.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlparse

DEFAULT_SPEC: dict[str, Any] = {
    "no_5xx": True,
    "no_failed_requests": {"allow_url_patterns": []},
    "no_console_errors": {"allow_patterns": []},
    "max_http_4xx": None,
    "response_ok": None,
    "scope": None,
}


def scope_hosts_for(url: str) -> list[str]:
    """The default assertion scope for a run against `url`: its registrable host.

    For a 3+-label host (test.actingoffice.com) the parent domain is used, so sibling
    subdomains (api., www.) stay in scope while everything else — helper-tab sites and
    third-party trackers — drops out. A bare or unparsable url returns [] (= unscoped)."""
    host = (urlparse(url or "").hostname or "").lower().strip(".")
    if not host:
        return []
    labels = host.split(".")
    if len(labels) >= 3:
        return [".".join(labels[1:])]
    return [host]


def _host_in_scope(url: str, hosts: list[str]) -> bool:
    host = (urlparse(url or "").hostname or "").lower()
    return any(host == h or host.endswith("." + h) for h in hosts)

_EVIDENCE_CAP = 5
# Request fields worth showing as evidence (headers etc. stay in the full network log).
_REQ_FIELDS = ("step", "method", "status", "url", "page_url", "errorText", "duration_ms")
_CONSOLE_FIELDS = ("step", "severity", "type", "text", "source", "page_url", "line")


@dataclass
class AssertionResult:
    name: str
    passed: bool | None          # None = skipped (collector missing at runtime)
    detail: str
    evidence: list[dict[str, Any]] = field(default_factory=list)


def merge_spec(defaults: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """Shallow merge: a per-task value replaces the default for that rule (None/False
    disables it). Unknown rule names are kept — evaluate() reports them as skipped so a
    typo'd rule name is visible instead of silently ignored."""
    spec = dict(defaults)
    spec.update(override or {})
    return spec


def _matchers(patterns: Any) -> tuple[list[Any], list[str]]:
    """Compile allowlist patterns into predicate functions; a bad regex degrades to substring
    matching and is reported in the assertion detail."""
    preds, notes = [], []
    for pat in patterns or []:
        pat = str(pat)
        try:
            rx = re.compile(pat)
            preds.append(rx.search)
        except re.error:
            notes.append(f"invalid regex {pat!r}: matched as substring")
            preds.append(lambda text, _p=pat: _p in text)
    return preds, notes


def _allowed(text: str, preds: list[Any]) -> bool:
    return any(p(text or "") for p in preds)


def _trim(record: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {k: record.get(k) for k in fields if record.get(k) is not None}


def evaluate(collector_results: dict[str, Any], spec: dict[str, Any]) -> list[AssertionResult]:
    network = collector_results.get("network") or {}
    console = collector_results.get("console") or {}
    requests = network.get("requests") or []
    entries = console.get("entries") or []
    results: list[AssertionResult] = []

    # Scope filter (see module docstring): every rule judges only in-scope telemetry.
    # Network scopes on the REQUEST url (5xx from the app's servers matter; a blocked ad
    # does not); console scopes on the PAGE that raised the entry (an app bundle may be
    # served from a CDN, so the entry's script `source` would lie). Entries recorded
    # before page_url existed count as in-scope rather than silently vanishing.
    scope = spec.get("scope") if isinstance(spec.get("scope"), dict) else None
    hosts = [str(h).lower().lstrip(".") for h in (scope or {}).get("hosts") or [] if h]
    scope_note = ""
    if hosts:
        requests = [r for r in requests if _host_in_scope(str(r.get("url") or ""), hosts)]
        entries = [e for e in entries
                   if e.get("page_url") is None
                   or _host_in_scope(str(e.get("page_url") or ""), hosts)]
        scope_note = f" [scope: {', '.join(hosts)}]"

    for name, rule in spec.items():
        if name == "scope" or rule is None or rule is False:
            continue

        if name == "no_5xx":
            if not network:
                results.append(AssertionResult(name, None, "network collector missing"))
                continue
            bad = [r for r in requests if r.get("status_class") == "5xx"]
            results.append(AssertionResult(
                name, not bad,
                f"{len(bad)} response(s) with status >= 500" if bad else "no 5xx responses",
                [_trim(r, _REQ_FIELDS) for r in bad[:_EVIDENCE_CAP]]))

        elif name == "no_failed_requests":
            if not network:
                results.append(AssertionResult(name, None, "network collector missing"))
                continue
            preds, notes = _matchers((rule if isinstance(rule, dict) else {}).get("allow_url_patterns"))
            bad = [r for r in requests
                   if r.get("failed") and not _allowed(str(r.get("url", "")), preds)]
            detail = (f"{len(bad)} failed request(s)" if bad else "no failed requests")
            if notes:
                detail += " (" + "; ".join(notes) + ")"
            results.append(AssertionResult(
                name, not bad, detail, [_trim(r, _REQ_FIELDS) for r in bad[:_EVIDENCE_CAP]]))

        elif name == "no_console_errors":
            if not console:
                results.append(AssertionResult(name, None, "console collector missing"))
                continue
            preds, notes = _matchers((rule if isinstance(rule, dict) else {}).get("allow_patterns"))
            bad = [e for e in entries
                   if (e.get("is_error") or e.get("is_exception"))
                   and not _allowed(str(e.get("text", "")), preds)]
            detail = (f"{len(bad)} console error(s)/exception(s)" if bad else "no console errors")
            if notes:
                detail += " (" + "; ".join(notes) + ")"
            results.append(AssertionResult(
                name, not bad, detail, [_trim(e, _CONSOLE_FIELDS) for e in bad[:_EVIDENCE_CAP]]))

        elif name == "max_http_4xx":
            if not network:
                results.append(AssertionResult(name, None, "network collector missing"))
                continue
            bad = [r for r in requests if r.get("status_class") == "4xx"]
            limit = int(rule)
            results.append(AssertionResult(
                name, len(bad) <= limit,
                f"{len(bad)} 4xx response(s) (limit {limit})",
                [_trim(r, _REQ_FIELDS) for r in bad[:_EVIDENCE_CAP]] if len(bad) > limit else []))

        elif name == "response_ok":
            if not network:
                results.append(AssertionResult(name, None, "network collector missing"))
                continue
            want = rule if isinstance(rule, dict) else {}
            method = (want.get("method") or "").upper() or None
            fragment = str(want.get("url_contains") or "").lower()
            lo, hi = (list(want.get("status_range") or (200, 399)) + [399])[:2]
            hits = [r for r in requests
                    if (method is None or r.get("method") == method)
                    and fragment in str(r.get("url", "")).lower()
                    and lo <= (r.get("status") or 0) <= hi]
            label = f"{method or 'any'} *{fragment}* -> {lo}-{hi}"
            results.append(AssertionResult(
                name, bool(hits),
                f"matched {len(hits)} request(s) for {label}" if hits
                else f"no request matched {label}",
                [_trim(r, _REQ_FIELDS) for r in hits[:1]]))

        else:
            results.append(AssertionResult(name, None, "unknown assertion rule (typo in spec?)"))

    if scope_note:
        # Stamp what every evaluated verdict was judged against, so a report reader can
        # see out-of-scope noise was excluded by design, not lost.
        for r in results:
            if r.passed is not None:
                r.detail += scope_note

    return results


def overall(results: list[AssertionResult]) -> bool | None:
    """Roll AssertionResults up into one verdict: False if anything failed, None if nothing
    was actually evaluated (all skipped/disabled), else True."""
    if any(r.passed is False for r in results):
        return False
    if any(r.passed is True for r in results):
        return True
    return None


def apply(result: Any, spec: dict[str, Any]) -> list[AssertionResult]:
    """Evaluate `spec` against a RunResult's collector output and stamp
    `assertion_results` / `assertions_passed` onto it."""
    results = evaluate(result.collector_results or {}, spec)
    result.assertion_results = [asdict(r) for r in results]
    result.assertions_passed = overall(results)
    return results
