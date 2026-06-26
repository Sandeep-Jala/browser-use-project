# Automation

A browser-use agent framework that logs into a web app, runs natural-language tasks
against it, captures **network + console telemetry**, and writes a self-contained HTML
report per run.

## Pipeline

```
browser/login.py   launch Chromium (CDP open) + log in   -> (browser, page, cdp_url)
browser/session.py attach a browser-use BrowserSession over CDP (no second login)
pipeline/runner.py run the task on the configured LLM; collectors listen via Playwright
pipeline/report.py compile the RunResult into report.html + report.json
```

## Layout

```
automation/
  __main__.py          # entry point: login -> Runner -> report (python -m automation)
  config.py            # Config.from_env(): all settings in one place
  llm.py               # build the browser-use chat model (OpenRouter dev / Groq final)
  browser/
    login.py           # Playwright login, hands off via CDP
    error_capture.py   # failure screenshots
    session.py         # attach browser-use to the authenticated browser over CDP
  pipeline/
    runner.py          # Runner + RunResult; owns per-run session + telemetry lifecycle
    prompt_expansion.py# rewrite a task into explicit step-by-step instructions
    report.py          # RunResult -> HTML + JSON
  collectors/
    base.py            # Collector ABC (attaches to a Playwright BrowserContext)
    network.py         # request/response/requestfailed via Playwright
    console.py         # console messages + uncaught page errors via Playwright
```

Telemetry is captured with Playwright: the runner opens a second CDP connection
(`connect_over_cdp`) to the same browser the agent drives and attaches native
`page.on("console" | "pageerror" | "response" | "requestfailed")` listeners.

## Setup

```bash
uv sync
uv run playwright install chromium
```

Create a `.env` (see `config.py` for all keys):

```
LOGIN_URL=...
LOGIN_EMAIL=...
LOGIN_PASSWORD=...
LLM_PROVIDER=openrouter        # or "groq"
OPEN_ROUTER_KEY=...            # for openrouter + the prompt expander
GROQ_API_KEY=...              # for groq
HEADFUL=true                  # visible browser
```

## Run

```bash
uv run python -m automation     # or: uv run automation
```

Each run writes `network.json`, `console.json`, `report.html`, and `report.json` into
`artifacts/<run_id>/`.
