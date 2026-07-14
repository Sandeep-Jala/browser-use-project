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
  tasks.py             # declarative task registry: TaskSpec(prompt, marker, assertions, tags)
  config.py            # Config.from_env(): all settings in one place
  llm.py               # build the browser-use chat model (Azure OpenAI default / Groq)
  browser/
    login.py           # Playwright login, hands off via CDP
    error_capture.py   # failure screenshots
    session.py         # attach browser-use to the authenticated browser over CDP
  pipeline/
    runner.py          # Runner + RunResult; owns per-run session + telemetry lifecycle
    prompts.py         # all prompt text (agent rules, app map, expander) + expand_task
    report.py          # RunResult -> HTML + JSON (+ suite-level report)
    script_compile.py  # compile a recorded run into a fast, no-LLM selector script
    assertions.py      # declarative telemetry assertions (no_5xx, no_console_errors, ...)
    suite.py           # run many tasks on one login -> suite.json + suite.html
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

Run the offline test suite (no app credentials needed) with `uv run pytest`.

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

# Run a whole suite on one login: every task, or a tag, or an explicit list
uv run python -m automation --suite all
uv run python -m automation --suite tag:sales
uv run python -m automation --suite invoice,purchase,item
```

| Flag / Env var | Effect |
|---------|--------|
| `TASK` / `--task` | which task to run: any key from `automation/tasks.py` (default `invoice`), or a full free-text prompt |
| `SUITE` / `--suite` | run a task set instead: `all`, `tag:<tag>`, or `k1,k2,...`; writes `artifacts/suites/<id>/suite.html` + `suite.json` and exits non-zero unless every task passes |
| `--no-auto` | force a plain agent run, ignoring recordings (replay is on by default) |
| `--fresh` | force re-authoring even if a script already exists (suite mode: only with an explicit task list) |
| `--no-fallback` | keep a broken replay as the result instead of archiving the script and re-authoring with the agent (fallback is on by default) |
| `--no-continue-on-failure` | suite mode: stop at the first task that doesn't pass |

The exit code is CI-ready: `0` only when the flow completed (ground-truth gate) **and** the
telemetry assertions passed.

### Outputs

- `artifacts/<run_id>/` — `network.json`, `console.json`, `report.html`, `report.json`
  (+ `replay_log.json` for script replays: which selector located each step)
- `artifacts/suites/<suite_id>/` — `suite.html` + `suite.json` (suite mode): per-task
  status/mode/duration/assertions with links to each run's report
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

### Self-healing replays

Replay failure is never a dead end:

- **Selector drift** — when every recorded selector for a step fails, the replay engine scores
  the live DOM against the step's recorded fingerprint and acts on a confident, unambiguous
  match. After a replay that **passed** its ground-truth gate, any healed steps are persisted:
  durable selectors synthesized from the healed element are prepended to the step's candidate
  list in the golden script (the old anchors stay as fallbacks), so the next replay resolves
  directly instead of re-healing.
- **Structural drift** — when a replay still fails (the app genuinely changed), the stale
  script + template are archived to `recordings/archive/` and the task is automatically
  re-authored by the agent from scratch, committing a fresh golden script (`--no-fallback`
  opts out). The failed replay's report is kept as evidence.

### Assertions

Every run is checked against declarative telemetry assertions (`pipeline/assertions.py`):
no 5xx responses, no failed requests, no console errors/exceptions by default; per-task
overrides live on the task's `TaskSpec.assertions` (e.g. allowlist a known noisy error, add a
`response_ok` check for an extra endpoint, or disable a rule). Assertions never change the
flow verdict (`is_successful` — that's the ground-truth gate's job); they gate a separate
`assertions_passed` verdict. A run whose flow passed but whose telemetry didn't shows as
**PASS\*** (amber) in the report, and the combined verdict drives the exit code.

### Adding a task

One entry in `automation/tasks.py`: a `TaskSpec(key=..., prompt=..., marker=..., tags=(...))`.
The `marker` is the URL fragment of the create-write that proves the record saved (the
ground-truth gate). **Never edit an existing prompt's wording casually** — the prompt's hash
is the task's identity, so a reworded prompt orphans its golden script
(`tests/test_tasks.py` pins every hash to catch this).
