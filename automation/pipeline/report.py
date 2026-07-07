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

_CSS = """
:root {
    --bg-dark: #12141d; --bg-card: #1c1f2e; --bg-card-hover: #232738; --border: #2a2d3e;
    --primary: #3b82f6; --primary-hover: #2563eb;
    --success: #10b981; --error: #ef4444; --warning: #f59e0b;
    --text-main: #f8fafc; --text-muted: #94a3b8;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Inter', sans-serif; background: var(--bg-dark); color: var(--text-main);
       display: flex; height: 100vh; overflow: hidden; }
.sidebar { width: 240px; background: var(--bg-card); border-right: 1px solid var(--border);
           display: flex; flex-direction: column; flex-shrink: 0; }
.sidebar-header { padding: 20px; border-bottom: 1px solid var(--border); font-weight: 700;
                  font-size: 1.1rem; color: var(--text-muted); text-transform: uppercase;
                  letter-spacing: 0.05em; }
.run-overview { padding: 20px; }
.run-overview-title { font-size: 0.75rem; color: var(--text-muted); font-weight: 600;
                      letter-spacing: 0.05em; margin-bottom: 15px; }
.donut-chart-box { width: 80px; height: 80px; border-radius: 50%; border: 6px solid var(--success);
                   display: flex; align-items: center; justify-content: center; font-size: 1.1rem;
                   font-weight: 800; margin-bottom: 20px; }
.run-stat-row { display: flex; align-items: center; justify-content: space-between;
                font-size: 0.85rem; margin-bottom: 10px; color: var(--text-muted); }
.run-stat-row .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 8px; }
.run-stat-row span { display: flex; align-items: center; }
.run-stat-val { color: var(--text-main); font-weight: 600; }
.main-area { flex-grow: 1; display: flex; flex-direction: column; overflow-y: auto; overflow-x: hidden; }
.main-header { display: flex; justify-content: space-between; align-items: center; padding: 24px 32px;
               border-bottom: 1px solid var(--border); background: var(--bg-dark); position: sticky;
               top: 0; z-index: 10; }
.main-title { font-size: 1.5rem; font-weight: 700; display: flex; align-items: center; gap: 12px; }
.badge-app { font-size: 0.65rem; background: rgba(59, 130, 246, 0.15); color: var(--primary);
             padding: 4px 8px; border-radius: 20px; border: 1px solid rgba(59, 130, 246, 0.3);
             letter-spacing: 0.05em; font-weight: 700; }
.header-meta { font-size: 0.85rem; color: var(--text-muted); display: flex; align-items: center; gap: 15px; }
.content-wrapper { padding: 32px; max-width: 1400px; margin: 0 auto; width: 100%; }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px; margin-bottom: 32px; }
.kpi-card { background: var(--bg-card); border: 1px solid var(--border); padding: 20px;
            border-radius: 12px; display: flex; flex-direction: column; justify-content: center;
            position: relative; overflow: hidden; height: 120px; transition: background 0.2s; }
.kpi-card:hover { background: var(--bg-card-hover); }
.kpi-icon { font-size: 1.5rem; margin-bottom: 12px; opacity: 0.8; }
.kpi-val { font-size: 2rem; font-weight: 800; line-height: 1; margin-bottom: 6px; }
.kpi-label { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;
             font-weight: 600; letter-spacing: 0.05em; }
.kpi-sub { font-size: 0.7rem; color: var(--text-muted); margin-top: 4px; }
.kpi-card.failed { border-top: 3px solid var(--error); }
.kpi-card.passed { border-top: 3px solid var(--success); }
.details-section { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px;
                   margin-bottom: 32px; overflow: hidden; }
.details-header { padding: 16px 20px; font-size: 0.85rem; font-weight: 700; color: var(--text-muted);
                  display: flex; align-items: center; gap: 10px; border-bottom: 1px solid var(--border);
                  background: rgba(255, 255, 255, 0.02); }
.details-grid { display: grid; grid-template-columns: 150px 1fr; padding: 20px; font-size: 0.85rem;
                line-height: 1.6; row-gap: 12px; }
.detail-label { color: var(--text-muted); font-weight: 500; }
.detail-val { color: var(--text-main); font-family: monospace; }
.section { margin-bottom: 32px; }
.section-title { font-size: 1.1rem; font-weight: 700; margin-bottom: 16px; display: flex;
                 align-items: center; gap: 10px; color: var(--text-main); }
.result-box { background: var(--bg-card); border: 1px solid var(--border);
              border-left: 4px solid var(--primary); border-radius: 8px; padding: 24px;
              font-size: 0.95rem; line-height: 1.7; color: #cbd5e1; }
.steps-table { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.step-row { display: grid; grid-template-columns: 60px 1fr; align-items: center; padding: 14px 16px;
            border-bottom: 1px solid var(--border); }
.step-row:last-child { border-bottom: none; }
.step-num { font-family: monospace; color: var(--text-muted); font-size: 0.8rem; }
.step-desc { font-size: 0.88rem; color: var(--text-main); }
.log-box { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.log-summary { cursor: pointer; list-style: none; padding: 20px 24px; display: flex;
               align-items: center; justify-content: space-between; gap: 16px;
               background: rgba(255,255,255,0.02); border-bottom: 1px solid var(--border); }
.log-summary::-webkit-details-marker { display: none; }
.log-summary-title { font-weight: 700; color: var(--text-main); }
.log-summary-sub { font-size: 0.82rem; color: var(--text-muted); margin-top: 6px; }
.log-badges { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; font-size: 0.72rem; }
.log-scroll { max-height: 560px; overflow: auto; background: rgba(0,0,0,0.16); }
table.log { width: 100%; min-width: 1000px; border-collapse: collapse; font-size: 0.78rem; }
table.log thead tr { background: rgba(15,23,42,0.8); color: var(--text-muted); text-transform: uppercase;
                     letter-spacing: 0.04em; font-size: 0.68rem; }
table.log th { text-align: left; padding: 10px 12px; }
table.log td { padding: 10px 12px; vertical-align: top; border-top: 1px solid var(--border); }
.mono { font-family: monospace; }
.nowrap { white-space: nowrap; }
.muted { color: var(--text-muted); }
.badge { padding: 4px 10px; border-radius: 4px; font-size: 0.72rem; font-weight: 600;
         border: 1px solid transparent; display: inline-block; }
.b-error { background: rgba(239,68,68,0.15); color: var(--error); border-color: rgba(239,68,68,0.3); }
.b-warn { background: rgba(245,158,11,0.15); color: var(--warning); border-color: rgba(245,158,11,0.3); }
.b-ok { background: rgba(16,185,129,0.12); color: var(--success); border-color: rgba(16,185,129,0.2); }
.b-info { background: rgba(59,130,246,0.15); color: var(--primary); border-color: rgba(59,130,246,0.3); }
.b-muted { background: #1e293b; color: var(--text-muted); }
.req-title { color: var(--text-main); margin-bottom: 4px; }
.req-meta { color: var(--text-muted); font-size: 0.72rem; margin-bottom: 6px; }
.req-url { color: #93c5fd; font-size: 0.72rem; word-break: break-all; overflow-wrap: anywhere; }
details.hdr { margin-top: 8px; }
details.hdr summary { cursor: pointer; color: var(--primary); }
details.hdr pre { white-space: pre-wrap; word-break: break-word; padding: 10px; margin-top: 8px;
                  background: rgba(0,0,0,0.25); border-radius: 6px; color: #cbd5e1; font-size: 0.72rem; }
.dl-btn { background: var(--primary); color: white; border: none; padding: 8px 16px; border-radius: 8px;
          cursor: pointer; font-weight: 600; display: flex; align-items: center; gap: 6px; }
.dl-btn:hover { background: var(--primary-hover); }
.obs-list { display: flex; flex-direction: column; gap: 8px; }
.obs-row { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px;
           padding: 14px 16px; display: flex; flex-direction: column; gap: 6px; }
.obs-head { display: flex; align-items: center; gap: 14px; }
.obs-head .material-icons { font-size: 1.4rem; }
.obs-title { flex-grow: 1; font-size: 1rem; }
.obs-reason { font-size: 0.84rem; color: var(--text-muted); padding-left: 38px; }
.obs-reason em { color: var(--text-main); font-style: normal; font-weight: 600; }
.shot-btn { background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.3);
            color: var(--primary); border-radius: 6px; padding: 5px 10px; cursor: pointer;
            font-size: 0.72rem; display: inline-flex; align-items: center; gap: 4px; white-space: nowrap; }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 14px; }
.shot-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px;
             overflow: hidden; cursor: pointer; transition: border-color 0.15s; }
.shot-card:hover { border-color: var(--primary); }
.shot-card img { width: 100%; height: 130px; object-fit: cover; object-position: top; display: block;
                 background: #000; }
.shot-card .cap { padding: 8px 10px; font-size: 0.75rem; color: var(--text-muted); }
.modal { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.88); z-index: 9999;
         align-items: center; justify-content: center; backdrop-filter: blur(6px); }
.modal img { max-width: 92vw; max-height: 86vh; border-radius: 8px; border: 1px solid var(--border); }
.modal .close { position: absolute; top: 20px; right: 28px; color: #fff; font-size: 2rem; cursor: pointer; }
.modal .cap { position: absolute; top: 24px; left: 28px; color: var(--text-muted); font-size: 0.9rem; }
"""


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

    if result.is_successful:
        status, ring, ring_color = "PASS", "var(--success)", "var(--success)"
    elif result.is_done:
        status, ring, ring_color = "DONE", "var(--warning)", "var(--warning)"
    else:
        status, ring, ring_color = "FAIL", "var(--error)", "var(--error)"

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

    # ---- Ground-truth network check (did the record actually get saved?) ----
    p.append(_render_ground_truth(result.ground_truth))

    # ---- Judge verdict (browser-use's built-in judge) ----
    p.append(_render_judgement(result.judgement, result.is_successful))

    # ---- Agent steps (per-step progress timeline) ----
    p.append(_render_steps(result.steps, result.model_actions, result.n_steps))

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
