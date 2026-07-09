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
# Replay-first by default: replay the task's recorded script if one exists (fast, no LLM),
# else run the agent once and record + compile a script for next time
uv run python -m automation

# Pick the task
TASK=purchase uv run python -m automation

# Force re-authoring an already-recorded task
TASK=purchase uv run python -m automation --fresh

# Force a plain agent run that ignores recordings (no replay, nothing recorded)
uv run python -m automation --no-auto
```

| Flag / Env var | Effect |
|---------|--------|
| `TASK` / `--task` | which task to run: `invoice` (default) or `purchase` |
| `--no-auto` | force a plain agent run, ignoring recordings (replay is on by default) |
| `--fresh` | force re-authoring even if a script already exists |

### Outputs

- `artifacts/<run_id>/` — `network.json`, `console.json`, `report.html`, `report.json`
- `recordings/<task_id>.steps.json` — the compiled fast-replay script (created the first time a task is authored)
- `recordings/<task_id>.template.json` — the script with its values lifted into a `{{param}}`
  dictionary (`customer`, `qty`, `unit_price`, ...), created alongside each golden script
- `recordings/manifest.json` — `task_id -> prompt` map (+ each task's `params` dictionary)

### How record / replay works

The first run of a task has the agent (LLM) perform it and saves the trace, then compiles
it into a stable-selector script. Subsequent runs of the **same prompt** detect that script
by a hash of the prompt and replay it over Playwright — no LLM, in seconds, at zero token cost.
(This is the default; `--no-auto` opts out of it for a one-off plain agent run.)

Editing only the prompt's **values** (customer, qty, unit price, ...) does not re-author either:
each committed script is parameterized into a template whose values live in a named dictionary
(`recordings/<task_id>.template.json`). A changed prompt is matched against recorded templates
(one small LLM call), the new values are read into the dictionary, the tokens are swapped in,
and the instantiated script is replay-validated before being committed as a golden script of
its own. Structural edits (different tab, extra fields) fall back to full agent re-authoring.
