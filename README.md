# Automation

A browser-use agent framework that logs into a web app, runs natural-language tasks
against it, captures **network + console telemetry**, and writes a self-contained HTML
report per run. A successful run can be **recorded and compiled into a fast, no-LLM
selector script**, so recurring tasks replay in seconds at zero token cost.

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
  llm.py               # build the browser-use chat model (Azure OpenAI default / Groq)
  browser/
    login.py           # Playwright login, hands off via CDP
    error_capture.py   # failure screenshots
    session.py         # attach browser-use to the authenticated browser over CDP
  pipeline/
    runner.py          # Runner + RunResult; owns per-run session + telemetry lifecycle
    prompts.py         # all prompt text (agent rules, app map, expander) + expand_task
    report.py          # RunResult -> HTML + JSON
    script_compile.py  # compile a recorded run into a fast, no-LLM selector script
    task_store.py      # task identity (prompt hash) + recording/script registry
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

Copy the sample env file and fill in your values (`.env.example` documents every key):

```bash
cp .env.example .env
```

Minimum required: `LOGIN_URL`, `LOGIN_EMAIL`, `LOGIN_PASSWORD`, and `AZURE_OPENAI_KEY`.
`.env` is gitignored; never commit it.

## Run

```bash
# Run the agent on the default task, then write a report
uv run python -m automation

# AUTO (recommended): replay the task's recorded script if one exists (fast, no LLM),
# else run the agent once and record + compile a script for next time
AUTO=1 uv run python -m automation

# Pick the task; force re-authoring an already-recorded task
TASK=purchase AUTO=1 uv run python -m automation
FRESH=1 TASK=purchase AUTO=1 uv run python -m automation
```

| Env var | Effect |
|---------|--------|
| `TASK` | which task to run: `invoice` (default) or `purchase` |
| `AUTO=1` | replay the recorded script if present, else author + record one |
| `FRESH=1` | force re-authoring even if a script already exists |

### Outputs

- `artifacts/<run_id>/` — `network.json`, `console.json`, `report.html`, `report.json`
- `recordings/<task_id>.steps.json` — the compiled fast-replay script (created on first AUTO author)
- `recordings/manifest.json` — `task_id -> prompt` map

### How record / replay works

The first AUTO run of a task has the agent (LLM) perform it and saves the trace, then compiles
it into a stable-selector script. Subsequent AUTO runs of the **same prompt** detect that script
by a hash of the prompt and replay it over Playwright — no LLM, in seconds, at zero token cost.
Editing the prompt text changes its hash, so it will re-author.
