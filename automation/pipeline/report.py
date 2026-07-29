"""Reporting — compile a RunResult into a polished HTML report + JSON.

`build_report(result)` writes two files into the run's artifacts directory:
  * `report.json` — the full machine-readable result (telemetry summaries + details).
  * `report.html` — a dark "Agent QA Report" dashboard: run overview, KPI cards, task
    details, the agent's final findings, the steps it took, and full Browser Console and
    Network log tables built from the collectors. Uses Google Fonts + Material Icons.
"""
from __future__ import annotations

import html
import json
import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from automation.pipeline.runner import RunResult

logger = logging.getLogger("framework.report")


def build_report(result: "RunResult") -> dict[str, Path]:
    """Write report.json + report.html into result.artifacts_dir. Returns their paths."""
    out_dir = Path(result.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(_result_to_dict(result), indent=2, default=str))

    html_path = out_dir / "report.html"
    html_path.write_text(_render_html(result), encoding="utf-8")

    return {"json": json_path, "html": html_path}


def _result_to_dict(result: "RunResult") -> dict[str, Any]:
    data = asdict(result)
    data["artifacts_dir"] = str(result.artifacts_dir)
    data["artifacts"] = {k: str(v) for k, v in result.artifacts.items()}
    # Screenshots are large base64 blobs — keep them out of the JSON (they live in the HTML),
    # recording only how many steps had one.
    data["screenshots"] = sum(1 for s in result.screenshots if s)
    return data


# --------------------------- HTML rendering ---------------------------

_CSS = (Path(__file__).parent.parent / "templates" / "report_styles.css").read_text(encoding="utf-8")


def _esc(s: Any) -> str:
    return html.escape(str(s)) if s is not None else ""


def _severity_badge(severity: str) -> str:
    cls = {"error": "b-error", "warning": "b-warn", "debug": "b-info"}.get(severity, "b-muted")
    return f'<span class="badge {cls}">{_esc(severity.upper())}</span>'


def _status_badge(req: dict[str, Any]) -> str:
    cls_map = {"2xx": "b-ok", "3xx": "b-ok", "4xx": "b-warn", "5xx": "b-error", "failed": "b-error"}
    sc = req.get("status_class", "pending")
    label = "FAILED" if req.get("failed") else (str(req.get("status")) if req.get("status") else sc)
    return f'<span class="badge {cls_map.get(sc, "b-muted")}">{_esc(label)}</span>'


def _kpi(icon: str, val: Any, label: str, sub: str, color: str, cls: str = "") -> str:
    return (
        f'<div class="kpi-card {cls}">'
        f'<span class="material-icons kpi-icon" style="color:{color}">{icon}</span>'
        f'<span class="kpi-val" style="color:{color}">{_esc(val)}</span>'
        f'<span class="kpi-label">{_esc(label)}</span>'
        f'<span class="kpi-sub">{_esc(sub)}</span></div>'
    )


def _headers_block(title: str, headers: dict[str, Any]) -> str:
    if not headers:
        return ""
    dumped = _esc(json.dumps(headers, indent=2))
    return (
        f'<details class="hdr"><summary>{_esc(title)}</summary>'
        f"<pre>{dumped}</pre></details>"
    )


def _render_html(result: "RunResult") -> str:
    net = result.collector_results.get("network", {})
    con = result.collector_results.get("console", {})
    net_sum = net.get("summary", {})
    con_sum = con.get("summary", {})

    net_problems = net_sum.get("http_4xx", 0) + net_sum.get("http_5xx", 0) + net_sum.get("failed", 0)
    con_errors = con_sum.get("errors", 0)

    if result.is_successful and getattr(result, "assertions_passed", None) is False:
        # The flow completed and saved, but telemetry assertions failed (5xx, console
        # errors, ...): a distinct state so a "green" run with an unhealthy app stands out.
        status, ring = "PASS*", "var(--warning)"
    elif result.is_successful:
        status, ring = "PASS", "var(--success)"
    elif result.is_done:
        status, ring = "DONE", "var(--warning)"
    else:
        status, ring = "FAIL", "var(--error)"

    p: list[str] = []
    p.append("<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>")
    p.append("<title>AI Agent Test Report</title>")
    p.append(
        "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap' rel='stylesheet'>"
    )
    p.append("<link href='https://fonts.googleapis.com/icon?family=Material+Icons' rel='stylesheet'>")
    p.append(f"<style>{_CSS}</style></head><body>")

    # ---- Sidebar ----
    p.append("<div class='sidebar'><div class='sidebar-header'>Run</div><div class='run-overview'>")
    p.append("<div class='run-overview-title'>RUN OVERVIEW</div>")
    p.append("<div style='display:flex; justify-content:center;'>")
    p.append(f"<div class='donut-chart-box' style='border-color:{ring}'>{status}</div></div>")
    p.append(
        f"<div class='run-stat-row'><span><span class='dot' style='background:var(--error)'></span> "
        f"Console errors</span><span class='run-stat-val'>{con_errors}</span></div>"
    )
    p.append(
        f"<div class='run-stat-row'><span><span class='dot' style='background:var(--warning)'></span> "
        f"Network problems</span><span class='run-stat-val'>{net_problems}</span></div>"
    )
    assertion_results = getattr(result, "assertion_results", None) or []
    if assertion_results:
        failed_asserts = sum(1 for a in assertion_results if a.get("passed") is False)
        p.append(
            f"<div class='run-stat-row'><span><span class='dot' style='background:"
            f"{'var(--error)' if failed_asserts else 'var(--success)'}'></span> "
            f"Assertions failed</span><span class='run-stat-val'>{failed_asserts}</span></div>"
        )
    p.append(
        "<div class='run-stat-row' style='margin-top:20px; border-top:1px solid var(--border); "
        f"padding-top:20px;'><span>Total steps</span><span class='run-stat-val'>{result.n_steps}</span></div>"
    )
    p.append("</div></div>")

    # ---- Main ----
    p.append("<div class='main-area'><div class='main-header'>")
    p.append(
        "<div class='main-title'>Agent QA Report <span class='badge-app'>browser-use</span></div>"
    )
    p.append("<div style='display:flex; align-items:center; gap:20px;'>")
    p.append(
        f"<div class='header-meta'><span>run {_esc(result.run_id)}</span>"
        f"<span>{result.duration_seconds:.1f}s</span></div>"
    )
    p.append("<button class='dl-btn' onclick='downloadReport()'>"
             "<span class='material-icons' style='font-size:18px;'>download</span> Download</button>")
    p.append("</div></div><div class='content-wrapper'>")

    # ---- KPI grid ----
    p.append("<div class='kpi-grid'>")
    p.append(_kpi("list", result.n_steps, "Steps", "Agent actions", "var(--text-main)"))
    p.append(_kpi(
        "error", con_errors, "Console Errors",
        f"{con_sum.get('warnings', 0)} warnings",
        "var(--error)" if con_errors else "var(--success)",
        "failed" if con_errors else "passed",
    ))
    p.append(_kpi(
        "lan", net_sum.get("total", 0), "Network Requests",
        f"{net_sum.get('fetch_xhr', 0)} fetch/xhr", "var(--text-main)",
    ))
    p.append(_kpi(
        "report", net_problems, "Network Problems", "4xx / 5xx / failed",
        "var(--error)" if net_problems else "var(--success)",
        "failed" if net_problems else "passed",
    ))
    p.append(_kpi("timer", f"{result.duration_seconds:.0f}s", "Total Duration", "Wall clock time",
                  "var(--text-main)"))
    usage = result.usage or {}
    total_tokens = usage.get("total_tokens")
    token_val = f"{total_tokens:,}" if isinstance(total_tokens, int) else "—"
    token_sub = (
        f"${usage.get('total_cost', 0):.4f} · {usage.get('entry_count', 0)} calls"
        if usage else "not reported"
    )
    p.append(_kpi("toll", token_val, "LLM Tokens", token_sub, "var(--primary)"))
    p.append("</div>")

    # ---- Task details ----
    p.append("<div class='details-section'><div class='details-header'>"
             "<span class='material-icons' style='font-size:1rem'>description</span>"
             " TEST ENVIRONMENT &amp; TASK DETAILS</div><div class='details-grid'>")
    p.append(f"<div class='detail-label'>Run ID</div><div class='detail-val'>{_esc(result.run_id)}</div>")
    p.append(f"<div class='detail-label'>Status</div><div class='detail-val'>{status} "
             f"(done={result.is_done}, success={result.is_successful})</div>")
    final_url = result.urls[-1] if result.urls else None
    p.append(f"<div class='detail-label'>Final URL</div><div class='detail-val'>{_esc(final_url)}</div>")
    if usage:
        tokens_detail = (
            f"{usage.get('total_tokens', 0):,} total — "
            f"{usage.get('total_prompt_tokens', 0):,} prompt "
            f"({usage.get('total_prompt_cached_tokens', 0):,} cached), "
            f"{usage.get('total_completion_tokens', 0):,} completion · "
            f"${usage.get('total_cost', 0):.4f}"
        )
        p.append(f"<div class='detail-label'>Tokens</div><div class='detail-val'>{_esc(tokens_detail)}</div>")
    p.append(
        "<div class='detail-label'>Task</div>"
        f"<div class='detail-val' style='line-height:1.4; white-space:pre-wrap; color:var(--text-muted)'>"
        f"{_esc(result.task)}</div>"
    )
    p.append("</div></div>")

    # ---- Agent findings ----
    if result.final_result:
        findings = _esc(result.final_result).replace("\n", "<br>")
        p.append("<div class='section'><div class='section-title'>"
                 "<span class='material-icons' style='color:var(--primary)'>fact_check</span>"
                 " Agent Final Findings</div>"
                 f"<div class='result-box'>{findings}</div></div>")

    # ---- Hybrid subtask segments (pipeline/hybrid.py) ----
    p.append(_render_subtasks(getattr(result, "subtasks", None)))

    # ---- Ground-truth network check (did the record actually get saved?) ----
    p.append(_render_ground_truth(result.ground_truth))

    # ---- Telemetry assertions (pipeline/assertions.py) ----
    p.append(_render_assertions(assertion_results))

    # ---- Judge verdict (browser-use's built-in judge) ----
    p.append(_render_judgement(result.judgement, result.is_successful))

    # ---- UI & Accessibility scans (detect_layout_issues / run_accessibility_scan) ----
    p.append(_render_ui_scans(result.extracted_content))

    # ---- Agent steps (per-step progress timeline) ----
    p.append(_render_steps(result.steps, result.model_actions, result.n_steps))

    # ---- Run video (--record) ----
    p.append(_render_video(result))

    # ---- Screenshots gallery ----
    p.append(_render_screenshots(result.screenshots))

    # ---- Console logs ----
    p.append(_render_console(con))

    # ---- Network logs ----
    p.append(_render_network(net))

    p.append("</div></div>")  # content-wrapper, main-area

    # ---- Screenshot lightbox modal ----
    p.append(
        "<div id='shot-modal' class='modal' onclick='closeShot()'>"
        "<span class='close'>&times;</span><span class='cap' id='shot-cap'></span>"
        "<img id='shot-img' src='' alt='screenshot'></div>"
    )

    p.append(
        "<script>function downloadReport(){"
        "const c=document.documentElement.outerHTML;"
        "const b=new Blob([c],{type:'text/html'});const u=URL.createObjectURL(b);"
        "const a=document.createElement('a');a.href=u;"
        "a.download='QA_Report_'+new Date().toISOString().replace(/[:.]/g,'-')+'.html';"
        "document.body.appendChild(a);a.click();document.body.removeChild(a);URL.revokeObjectURL(u);}"
        "function openShot(el){const img=el.querySelector('img');"
        "document.getElementById('shot-img').src=img.src;"
        "document.getElementById('shot-cap').textContent=el.dataset.cap||'';"
        "document.getElementById('shot-modal').style.display='flex';}"
        "function closeShot(){document.getElementById('shot-modal').style.display='none';}"
        "document.addEventListener('keydown',function(e){if(e.key==='Escape')closeShot();});"
        "</script>"
    )
    p.append("</body></html>")
    return "".join(p)


def _render_ground_truth(ground_truth: dict[str, Any] | None) -> str:
    """Render the network ground-truth check: was the record actually saved?

    This is the objective counterweight to the agent's self-report and the LLM judge (both of
    which can be fooled). When a create-write was expected but never hit the network, it flags
    that the reported success was overridden to failure.
    """
    if not ground_truth:
        return ""
    marker = ground_truth.get("marker")
    seen = ground_truth.get("create_write_seen")
    overrode = ground_truth.get("overrode_success")
    if seen:
        icon, color, label, badge = "check", "var(--success)", "SAVED", "background:var(--success);color:#fff"
    else:
        icon, color, label, badge = "close", "var(--error)", "NO WRITE", "background:var(--error);color:#fff"

    p: list[str] = []
    p.append("<div class='section'><div class='section-title'>"
             "<span class='material-icons' style='color:var(--primary)'>lan</span>"
             " Ground Truth (network)</div><div class='obs-list'><div class='obs-row'>")
    p.append(
        "<div class='obs-head'>"
        f"<span class='material-icons' style='color:{color}'>{icon}</span>"
        f"<span class='obs-title'>Create-write to <code>{_esc(marker)}</code> seen in network</span>"
        f"<span class='badge' style='{badge}'>{label}</span></div>"
    )
    if overrode:
        p.append("<div class='obs-reason'><em>Note:</em> the run reported success but no matching "
                 "create-write was sent — nothing was actually saved. <em>Reported success was "
                 "overridden to FAIL.</em></div>")
    elif not seen:
        p.append("<div class='obs-reason'>No matching create-write was sent during this run.</div>")
    p.append("</div></div></div>")
    return "".join(p)


def _render_subtasks(subtasks: list[dict[str, Any]] | None) -> str:
    """Render a hybrid run's per-segment outcomes: one row per subtask with its mode
    (library replay vs agent-authored), gate verdict, and cost."""
    if not subtasks:
        return ""
    replayed = sum(1 for s in subtasks if s.get("mode") == "replay")
    p: list[str] = []
    p.append("<div class='section'><div class='section-title'>"
             "<span class='material-icons' style='color:var(--primary)'>account_tree</span>"
             f" Subtasks ({replayed} replayed / {len(subtasks) - replayed} authored)"
             "</div><div class='obs-list'>")
    for s in subtasks:
        ok = s.get("ok")
        if ok:
            icon, color, label, badge = ("check", "var(--success)", "OK",
                                         "background:var(--success);color:#fff")
        else:
            icon, color, label, badge = ("close", "var(--error)", "FAIL",
                                         "background:var(--error);color:#fff")
        gate = s.get("gate") or {}
        detail = (f"{s.get('mode')} · {s.get('steps_executed', 0)} steps · "
                  f"{s.get('duration_seconds', 0)}s · gate: {gate.get('kind', '—')}")
        if s.get("tokens"):
            detail += f" · {s['tokens']:,} tokens"
        if s.get("healed_steps"):
            detail += f" · healed {s['healed_steps']}"
        p.append("<div class='obs-row'><div class='obs-head'>"
                 f"<span class='material-icons' style='color:{color}'>{icon}</span>"
                 f"<span class='obs-title'>{s.get('index')}. "
                 f"{_esc((s.get('prompt') or '')[:120])}<br>"
                 f"<span style='color:var(--text-muted)'>{_esc(detail)}</span></span>"
                 f"<span class='badge' style='{badge}'>{label}</span></div>")
        if s.get("error"):
            p.append(f"<div class='obs-reason'><code>{_esc(str(s['error'])[:300])}</code></div>")
        p.append("</div>")
    p.append("</div></div>")
    return "".join(p)


def _render_assertions(assertion_results: list[dict[str, Any]]) -> str:
    """Render the telemetry assertions: one row per rule with a PASS/FAIL/SKIPPED badge and
    the offending requests/console entries as evidence. Assertions never change
    is_successful (the flow verdict) — a failure here means "the flow passed but the app
    wasn't healthy while it did" (the PASS* state)."""
    if not assertion_results:
        return ""
    p: list[str] = []
    p.append("<div class='section'><div class='section-title'>"
             "<span class='material-icons' style='color:var(--primary)'>rule</span>"
             " Assertions</div><div class='obs-list'>")
    for a in assertion_results:
        passed = a.get("passed")
        if passed is True:
            icon, color, label, badge = "check", "var(--success)", "PASS", "background:var(--success);color:#fff"
        elif passed is False:
            icon, color, label, badge = "close", "var(--error)", "FAIL", "background:var(--error);color:#fff"
        else:
            icon, color, label, badge = "help", "var(--text-muted)", "SKIPPED", "background:#1e293b;color:var(--text-muted)"
        p.append("<div class='obs-row'><div class='obs-head'>"
                 f"<span class='material-icons' style='color:{color}'>{icon}</span>"
                 f"<span class='obs-title'><code>{_esc(a.get('name'))}</code> — "
                 f"{_esc(a.get('detail'))}</span>"
                 f"<span class='badge' style='{badge}'>{label}</span></div>")
        for ev in a.get("evidence") or []:
            if "url" in ev:  # a network request record
                line = (f"step {ev.get('step', '—')}: {ev.get('method', '')} "
                        f"{ev.get('status') or ev.get('errorText') or '?'} {ev.get('url', '')}")
            else:            # a console entry
                src = f" ({ev.get('source')}:{ev.get('line')})" if ev.get("source") else ""
                line = f"step {ev.get('step', '—')}: [{ev.get('severity', '?')}] {ev.get('text', '')}{src}"
            p.append(f"<div class='obs-reason'><code>{_esc(line[:300])}</code></div>")
        p.append("</div>")
    p.append("</div></div>")
    return "".join(p)


def _render_judgement(judgement: dict[str, Any] | None, agent_success: Any) -> str:
    """Render browser-use's built-in judge verdict, flagging agent-vs-judge disagreement."""
    if not judgement:
        return ""
    verdict = judgement.get("verdict")
    if verdict is True:
        icon, color, label, badge = "check", "var(--success)", "PASS", "background:var(--success);color:#fff"
    elif verdict is False:
        icon, color, label, badge = "close", "var(--error)", "FAIL", "background:var(--error);color:#fff"
    else:
        icon, color, label, badge = "help", "var(--text-muted)", "N/A", "background:#1e293b;color:var(--text-muted)"

    p: list[str] = []
    p.append("<div class='section'><div class='section-title'>"
             "<span class='material-icons' style='color:var(--primary)'>gavel</span>"
             " Judge Verdict</div><div class='obs-list'><div class='obs-row'>")
    p.append(
        "<div class='obs-head'>"
        f"<span class='material-icons' style='color:{color}'>{icon}</span>"
        "<span class='obs-title'>Independent judge (browser-use)</span>"
        f"<span class='badge' style='{badge}'>{label}</span></div>"
    )
    # Surface disagreement between the agent's self-report and the judge.
    if verdict is False and agent_success is True:
        p.append("<div class='obs-reason'><em>Note:</em> the agent reported success but the "
                 "judge disagreed.</div>")
    if judgement.get("failure_reason"):
        p.append(f"<div class='obs-reason'><em>Failure reason:</em> "
                 f"{_esc(judgement.get('failure_reason'))}</div>")
    if judgement.get("reasoning"):
        p.append(f"<div class='obs-reason'>{_esc(judgement.get('reasoning'))}</div>")
    if judgement.get("reached_captcha"):
        p.append("<div class='obs-reason'><em>⚠️ Captcha encountered during the run.</em></div>")
    p.append("</div></div></div>")
    return "".join(p)


# The UI-scan tools (automation/pipeline/agent_tools.py) write their results into the agent's
# extracted_content as text with these stable prefixes and line formats. We parse the LAST
# occurrence of each (the final page state) back into a table for the report.
_A11Y_PREFIX = "Accessibility scan (axe-core)"
_LAYOUT_PREFIX = "Layout scan"
# "- button-name (critical): 40 node(s) — Buttons must have discernible text"
_A11Y_LINE = re.compile(
    r"^-\s*(?P<id>\S+)\s*\((?P<impact>[^)]*)\):\s*(?P<nodes>\d+)\s*node\(s\)\s*[—-]\s*(?P<help>.*)$"
)
# "- zero-size-control: 1+ clickable element(s) rendered at ~0 size"
_LAYOUT_LINE = re.compile(r"^-\s*(?P<type>[\w-]+):\s*(?P<detail>.*)$")
_A11Y_BADGE = {"critical": "b-error", "serious": "b-error", "moderate": "b-warn", "minor": "b-info"}


def _last_startswith(entries: list[str] | None, prefix: str) -> str | None:
    return next(
        (e for e in reversed(entries or []) if isinstance(e, str) and e.startswith(prefix)),
        None,
    )


def _render_ui_scans(extracted_content: list[str] | None) -> str:
    """Render the layout + accessibility scan results (skipped entirely on script replays,
    which have no agent and therefore no scan output)."""
    layout = _last_startswith(extracted_content, _LAYOUT_PREFIX)
    a11y = _last_startswith(extracted_content, _A11Y_PREFIX)
    if not layout and not a11y:
        return ""

    p: list[str] = []
    p.append("<div class='section'><div class='section-title'>"
             "<span class='material-icons' style='color:var(--primary)'>accessibility_new</span>"
             " UI &amp; Accessibility</div>")
    if layout:
        p.append(_render_layout_block(layout))
    if a11y:
        p.append(_render_a11y_block(a11y))
    p.append("</div>")
    return "".join(p)


def _render_layout_block(text: str) -> str:
    lines = text.splitlines()
    issues = [m.groupdict() for m in map(_LAYOUT_LINE.match, lines[1:]) if m]
    p: list[str] = []
    p.append("<div class='obs-list' style='margin-bottom:14px;'>")
    p.append("<div class='obs-row'><div class='obs-head'>"
             "<span class='material-icons' style='color:%s'>%s</span>"
             "<span class='obs-title'>Layout scan</span>"
             "<span class='badge' style='background:%s;color:#fff'>%s</span></div>" % (
                 ("var(--success)", "check", "var(--success)", "CLEAN") if not issues
                 else ("var(--warning)", "warning", "var(--warning)", f"{len(issues)} ISSUE"
                       + ("S" if len(issues) != 1 else ""))))
    for it in issues:
        p.append(f"<div class='obs-reason'><code>{_esc(it['type'])}</code> — {_esc(it['detail'])}</div>")
    p.append("</div></div>")
    return "".join(p)


def _render_a11y_block(text: str) -> str:
    lines = text.splitlines()
    rows = [m.groupdict() for m in map(_A11Y_LINE.match, lines[1:]) if m]
    # A non-matching tail line like "  ...and N more rule(s)" is surfaced as a note.
    notes = [ln.strip() for ln in lines[1:]
             if ln.strip() and not _A11Y_LINE.match(ln) and ln.strip().startswith("...")]
    clean = not rows and "no violations" in lines[0].lower()

    p: list[str] = []
    p.append("<div class='obs-list'>")
    p.append("<div class='obs-row'><div class='obs-head'>"
             "<span class='material-icons' style='color:%s'>%s</span>"
             "<span class='obs-title'>Accessibility scan (axe-core)</span>"
             "<span class='badge' style='background:%s;color:#fff'>%s</span></div></div>" % (
                 ("var(--success)", "check", "var(--success)", "CLEAN") if clean
                 else ("var(--error)", "close", "var(--error)",
                       f"{len(rows)} RULE" + ("S" if len(rows) != 1 else ""))))
    if rows:
        p.append("<div class='log-scroll'><table class='log'><thead><tr>"
                 "<th style='width:220px'>Rule</th><th style='width:110px'>Severity</th>"
                 "<th style='width:80px'>Nodes</th><th>Description</th></tr></thead><tbody>")
        for r in rows:
            impact = (r.get("impact") or "").lower()
            badge = _A11Y_BADGE.get(impact, "b-muted")
            p.append(
                f"<tr><td class='mono nowrap' style='color:var(--text-main)'>{_esc(r['id'])}</td>"
                f"<td class='nowrap'><span class='badge {badge}'>{_esc(impact.upper() or 'N/A')}</span></td>"
                f"<td class='mono nowrap'>{_esc(r['nodes'])}</td>"
                f"<td style='color:#cbd5e1'>{_esc(r['help'])}</td></tr>"
            )
        p.append("</tbody></table></div>")
    elif not clean:
        # Parsing failed (format drift) — show the raw text rather than dropping the data.
        p.append(f"<div class='result-box muted'><pre style='white-space:pre-wrap;margin:0'>"
                 f"{_esc(text)}</pre></div>")
    for note in notes:
        p.append(f"<div class='obs-reason muted'>{_esc(note)}</div>")
    p.append("</div>")
    return "".join(p)


def _render_steps(
    steps: list[dict[str, Any]], model_actions: list[dict[str, Any]] | None, n_steps: int
) -> str:
    """Per-step progress timeline (the agent's own goal/eval per step). Falls back to the raw
    action list if the timeline is unavailable."""
    reached = len(steps) or n_steps
    p: list[str] = []
    p.append("<div class='section'><div class='section-title'>"
             "<span class='material-icons'>list</span> Agent Steps "
             f"<span class='muted' style='font-weight:400; font-size:0.85rem'>"
             f"(reached step {reached} of {n_steps})</span></div>")
    if steps:
        last = len(steps)
        p.append("<div class='steps-table'>")
        for s in steps:
            n = s.get("n")
            border = "border-left:3px solid var(--primary);" if n == last else ""
            p.append(
                f"<div class='step-row' style='align-items:start; {border}'>"
                f"<div class='step-num'>#{_esc(n)}</div><div class='step-desc'>"
                f"{_esc(s.get('next_goal') or '(no goal recorded)')}"
            )
            if s.get("evaluation"):
                p.append(f"<div class='muted' style='font-size:0.78rem; margin-top:3px;'>"
                         f"<em>eval:</em> {_esc(s.get('evaluation'))}</div>")
            if s.get("url"):
                p.append(f"<div class='req-url' style='margin-top:3px;'>{_esc(s.get('url'))}</div>")
            p.append("</div></div>")
        p.append("</div>")
    elif model_actions:
        p.append("<div class='steps-table'>")
        for i, a in enumerate(model_actions, 1):
            p.append(f"<div class='step-row'><div class='step-num'>#{i}</div>"
                     f"<div class='step-desc'>{_esc(_action_label(a))}</div></div>")
        p.append("</div>")
    else:
        p.append("<div class='result-box muted'>No steps recorded.</div>")
    p.append("</div>")
    return "".join(p)


def _render_video(result: "RunResult") -> str:
    """Player for the run's .mp4 (--record), or "" when the run wasn't recorded.

    Referenced RELATIVELY: report.html and run.mp4 are written to the same run directory,
    so the page stays portable if the folder is copied or zipped."""
    video = (result.artifacts or {}).get("video")
    if not video or not Path(video).exists():
        return ""
    return (
        "<div class='section'><div class='section-title'>"
        "<span class='material-icons' style='color:var(--primary)'>movie</span>"
        " Run Recording <span class='muted' style='font-weight:400; font-size:0.85rem'>"
        "(time-lapse — the browser emits frames only when the page changes, so waits "
        "between steps are skipped)</span></div>"
        f"<video src='{_esc(Path(video).name)}' controls preload='metadata' "
        "style='width:100%; max-width:1100px; border-radius:8px; background:#000'>"
        "</video></div>"
    )


def _render_screenshots(screenshots: list[str | None]) -> str:
    shots = [(i + 1, s) for i, s in enumerate(screenshots) if s]
    if not shots:
        return ""
    p: list[str] = []
    p.append("<div class='section'><div class='section-title'>"
             "<span class='material-icons' style='color:var(--primary)'>photo_camera</span>"
             f" Screenshots <span class='muted' style='font-weight:400; font-size:0.85rem'>"
             f"({len(shots)} captured)</span></div><div class='gallery'>")
    for step, b64 in shots:
        src = b64 if str(b64).startswith("data:") else f"data:image/png;base64,{b64}"
        cap = f"Step {step}"
        p.append(
            f"<div class='shot-card' onclick='openShot(this)' data-cap='{cap}'>"
            f"<img src='{src}' alt='{cap}' loading='lazy'>"
            f"<div class='cap'>{cap}</div></div>"
        )
    p.append("</div></div>")
    return "".join(p)


def _render_console(con: dict[str, Any]) -> str:
    s = con.get("summary", {})
    entries = con.get("entries", []) or []
    # Show problems first (errors/warnings), then the rest, capped for readability.
    entries = sorted(entries, key=lambda e: 0 if e.get("is_error") else (1 if e.get("is_warning") else 2))

    p: list[str] = []
    p.append("<div class='section'><div class='section-title'>"
             "<span class='material-icons' style='color:var(--warning)'>terminal</span>"
             " Browser Console Logs</div>")
    p.append("<div class='log-box' style='border-left:4px solid var(--warning)'><details>")
    p.append("<summary class='log-summary'><div><div class='log-summary-title'>Console Summary</div>"
             f"<div class='log-summary-sub'>{s.get('errors', 0)} errors, {s.get('warnings', 0)} warnings, "
             f"{s.get('debug', 0)} debug, {s.get('exceptions', 0)} JS exceptions, "
             f"{s.get('total', 0)} total console events</div></div>")
    p.append("<div class='log-badges'>"
             f"<span class='badge b-error'>Errors: {s.get('errors', 0)}</span>"
             f"<span class='badge b-warn'>Warnings: {s.get('warnings', 0)}</span>"
             f"<span class='badge b-info'>Debug: {s.get('debug', 0)}</span>"
             f"<span class='badge b-error'>Exceptions: {s.get('exceptions', 0)}</span></div></summary>")

    if entries:
        p.append("<div class='log-scroll'><table class='log'><thead><tr>"
                 "<th style='width:90px'>Step</th><th style='width:110px'>Severity</th>"
                 "<th style='width:150px'>Type</th><th>Message</th>"
                 "<th style='width:280px'>Source</th></tr></thead><tbody>")
        for e in entries[:400]:
            p.append(
                f"<tr><td class='muted mono nowrap'>Step {_esc(e.get('step'))}</td>"
                f"<td class='nowrap'>{_severity_badge(e.get('severity', 'info'))}</td>"
                f"<td class='nowrap'>{_esc(e.get('type'))}</td>"
                f"<td class='mono' style='color:#cbd5e1; white-space:pre-wrap; word-break:break-word;'>"
                f"{_esc(e.get('text'))}</td>"
                f"<td class='muted mono' style='font-size:0.72rem; word-break:break-word;'>"
                f"{_esc(e.get('source'))}</td></tr>"
            )
        p.append("</tbody></table></div>")
    else:
        p.append("<div style='padding:20px;' class='muted'>No console events captured.</div>")
    p.append("</details></div></div>")
    return "".join(p)


def _render_network(net: dict[str, Any]) -> str:
    s = net.get("summary", {})
    requests = net.get("requests", []) or []
    # Problems first (failed / 4xx / 5xx), then the rest.
    requests = sorted(requests, key=lambda r: 0 if r.get("is_error") else 1)

    p: list[str] = []
    p.append("<div class='section'><div class='section-title'>"
             "<span class='material-icons' style='color:var(--primary)'>lan</span>"
             " Network Logs</div>")
    p.append("<div class='log-box' style='border-left:4px solid var(--primary)'><details>")
    p.append("<summary class='log-summary'><div><div class='log-summary-title'>Network Summary</div>"
             f"<div class='log-summary-sub'>{s.get('total', 0)} network events, "
             f"{s.get('fetch_xhr', 0)} Fetch/XHR, {s.get('failed', 0)} failed requests</div></div>")
    p.append("<div class='log-badges'>"
             f"<span class='badge b-ok'>2xx/3xx: {s.get('http_2xx_3xx', 0)}</span>"
             f"<span class='badge b-warn'>4xx: {s.get('http_4xx', 0)}</span>"
             f"<span class='badge b-error'>5xx: {s.get('http_5xx', 0)}</span>"
             f"<span class='badge b-error'>Failed: {s.get('failed', 0)}</span>"
             f"<span class='badge b-warn'>Slow: {s.get('slow', 0)}</span></div></summary>")

    if requests:
        p.append("<div class='log-scroll'><table class='log'><thead><tr>"
                 "<th style='width:90px'>Step</th><th style='width:100px'>Status</th>"
                 "<th style='width:90px'>Method</th><th>Request</th>"
                 "<th style='width:110px'>Type</th></tr></thead><tbody>")
        for r in requests[:400]:
            dur = r.get("duration_ms")
            dur_txt = f"{dur:.0f}ms" if isinstance(dur, (int, float)) else "—"
            status = "FAILED" if r.get("failed") else r.get("status")
            title = f"HTTP {_esc(status)} {_esc(r.get('method'))} {_esc(r.get('url'))} ({dur_txt})"
            meta = (f"Method: {_esc(r.get('method'))} | Status: {_esc(status)} | "
                    f"Type: {_esc(r.get('resourceType'))} | Duration: {dur_txt}")
            if r.get("errorText"):
                meta += f" | Error: {_esc(r.get('errorText'))}"
            p.append(
                f"<tr><td class='muted mono nowrap'>Step {_esc(r.get('step'))}</td>"
                f"<td class='nowrap'>{_status_badge(r)}</td>"
                f"<td class='mono nowrap' style='color:var(--text-main)'>{_esc(r.get('method'))}</td>"
                f"<td class='mono' style='color:#cbd5e1; word-break:break-word;'>"
                f"<div class='req-title'>{title}</div>"
                f"<div class='req-meta'>{meta}</div>"
                f"<div class='req-url'>{_esc(r.get('url'))}</div>"
                f"{_headers_block('Request Headers', r.get('request_headers'))}"
                f"{_headers_block('Response Headers', r.get('response_headers'))}"
                f"</td>"
                f"<td class='muted mono nowrap' style='font-size:0.75rem'>{_esc(r.get('resourceType'))}</td></tr>"
            )
        p.append("</tbody></table></div>")
    else:
        p.append("<div style='padding:20px;' class='muted'>No network events captured.</div>")
    p.append("</details></div></div>")
    return "".join(p)


def _action_label(action: dict[str, Any]) -> str:
    """Turn a model action dict into a short readable label."""
    if not isinstance(action, dict):
        return str(action)
    # actions look like {"<action_name>": {..params..}, "interacted_element": ...}
    for key, val in action.items():
        if key == "interacted_element":
            continue
        params = ""
        if isinstance(val, dict):
            params = ", ".join(f"{k}={v}" for k, v in val.items() if k != "interacted_element")
        return f"{key}: {params}" if params else str(key)
    return str(action)
