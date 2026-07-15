# Automation

A browser-use agent framework that logs into a web app, runs natural-language tasks
against it, captures **network + console telemetry**, and writes a self-contained HTML
report per run. Every task runs **subtask-by-subtask**: each step a task has performed
before replays from a shared library as a no-LLM selector script, and the LLM is spent only
on the gaps — so recurring work costs seconds and near-zero tokens.

## Pipeline

```
browser/login.py     launch Chromium (CDP open) + log in   -> (browser, page, cdp_url)
browser/session.py   attach a browser-use BrowserSession over CDP (no second login)
pipeline/decompose.py split the task into subtasks (cached / derived / one LLM call)
pipeline/hybrid.py   per subtask: replay it from library/ (no LLM), else author + record it
pipeline/runner.py   run ONE agent segment on the shared session; collectors via Playwright
pipeline/report.py   compile the RunResult into report.html + report.json
```

## Layout

```
automation/
  __main__.py          # entry point: login -> hybrid engine -> report (python -m automation)
  tasks.py             # declarative task registry: TaskSpec(prompt, marker, assertions, tags)
  config.py            # Config.from_env(): all settings in one place
  llm.py               # build the browser-use chat model (Azure OpenAI default / Groq)
  browser/
    login.py           # Playwright login, hands off via CDP
    error_capture.py   # failure screenshots
    session.py         # attach browser-use to the authenticated browser over CDP
  pipeline/
    runner.py          # Runner + RunResult; runs ONE agent segment (owns no lifecycle)
    prompts.py         # all prompt text: agent rules, per-subtask prompt, decompose
    report.py          # RunResult -> HTML + JSON (+ suite-level report)
    script_compile.py  # compile a recording into a fast, no-LLM selector script; replay it
    assertions.py      # declarative telemetry assertions (no_5xx, no_console_errors, ...)
    suite.py           # run many tasks on one login -> suite.json + suite.html
    subtask_store.py   # task + subtask identity, and the SHARED subtask library (library/)
    decompose.py       # split a task into subtasks (spec-declared / cached / derived / LLM)
    hybrid.py          # THE execution engine: replay recorded subtasks, LLM only for the gaps
  collectors/
    base.py            # Collector ABC (attaches to a Playwright BrowserContext)
    network.py         # request/response/requestfailed via Playwright
    console.py         # console messages + uncaught page errors via Playwright
```

Telemetry is captured with Playwright: the engine opens a second CDP connection
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
# Replay-first by default: every subtask the library already knows replays with no LLM;
# the rest are authored by the agent, recorded, and committed for next time
uv run python -m automation

# Pick the task
TASK=purchase uv run python -m automation

# Re-author AND re-record EVERY subtask from scratch
TASK=purchase uv run python -m automation --fresh

# Re-record only the subtasks you name; the rest still replay
TASK=purchase uv run python -m automation --reauthor 2
TASK=purchase uv run python -m automation --reauthor 'add invoice'

# Run a whole suite on one login: every task, or a tag, or an explicit list
uv run python -m automation --suite all
uv run python -m automation --suite tag:sales
uv run python -m automation --suite invoice,purchase,item
```

| Flag / Env var | Effect |
|---------|--------|
| `TASK` / `--task` | which task to run: any key from `automation/tasks.py` (default `invoice`), or a full free-text prompt |
| `SUITE` / `--suite` | run a task set instead: `all`, `tag:<tag>`, or `k1,k2,...`; writes `artifacts/suites/<id>/suite.html` + `suite.json` and exits non-zero unless every task passes |
| `--fresh` | re-author **and re-record every subtask**, ignoring the library; each one that passes its gate is committed back (suite mode: only with an explicit task list) |
| `--reauthor` | re-record only the subtasks you name — comma-list of indexes and/or prompt substrings, e.g. `--reauthor 0` or `--reauthor 'add estimate'`. Others still replay, and an entry is replaced only if the new recording passes its gate. Use it when a recording works but wanders |
| `--redecompose` | regenerate the task's cached subtask **split** (the cache is otherwise immutable per prompt) |
| `--marker` | success-marker URL fragment; `none` disables the network ground-truth gate |
| `--no-continue-on-failure` | suite mode: stop at the first task that doesn't pass |

The exit code is CI-ready: `0` only when the flow completed (ground-truth gate) **and** the
telemetry assertions passed.

### Outputs

- `artifacts/<run_id>/` — `network.json`, `console.json`, `report.html`, `report.json`
- `artifacts/suites/<suite_id>/` — `suite.html` + `suite.json` (suite mode): per-task
  status/mode/duration/assertions with links to each run's report
- `library/` — **the** recording store: per subtask `<sid>.steps.json` (the compiled no-LLM
  script), `.template.json` (values lifted into a `{{param}}` dictionary), `.recording.json`
  (raw authoring trace), `.meta.json` (uses/failures), plus `manifest.json` and `archive/`
- `decompositions/<task_id>.json` — cached subtask split per task prompt

### How record / replay works

Every task is **decomposed** into subtasks (`pipeline/decompose.py`): a cached split if the
prompt has been seen, a derived match (same task shape, different values — no LLM), or one
LLM call, cached forever after. `--redecompose` regenerates it.

Each subtask is keyed into a **global shared library** (`library/`) by its *parameterized*
prompt plus the normalized URL context it starts from. That key is the whole trick: the
"go to Bookkeeping, search and select {{business}}" prefix that a dozen tasks share is ONE
library entry — authored once by whichever task ran first, replayed by all the others.
Values are template-swapped at replay, so a different customer costs nothing.

Then, per subtask, in order:

- **Library hit** → the compiled selector script replays over Playwright. No LLM, seconds,
  zero tokens, self-healing included.
- **Miss** → the agent authors just that step, with a prompt scoped to it alone plus the
  end-state its gate expects. On a passing gate the trace is compiled and committed to the
  library, so it replays from then on.

All segments share ONE live browser session: replay and agent segments interleave on the
same page with no teardown, so each picks up exactly where the previous left off. A task
therefore gets cheaper every time it runs, and a *new* task built from familiar steps can
be almost free on its first run.

Per-subtask success gates: the save-owning subtask must fire the parent's create-write
(network ground truth, windowed to that segment); navigation subtasks must reach the URL
context their recording ended on; the parent-level marker gate still applies on top, so a
run where every segment "passed" but nothing was saved still fails.

### Self-healing replays

Replay failure is never a dead end:

- **Selector drift** — when every recorded selector for a step fails, the replay engine scores
  the live DOM against the step's recorded fingerprint and acts on a confident, unambiguous
  match. After a replay that **passed** its ground-truth gate, any healed steps are persisted:
  durable selectors synthesized from the healed element are prepended to the step's candidate
  list in the library entry (the old anchors stay as fallbacks), so the next replay resolves
  directly instead of re-healing.
- **Structural drift** — when a replay still fails, what happens depends on how far it got.
  Failing on the *first* step means the page was never touched, so the stale entry is
  archived to `library/archive/` and the agent re-authors it **cleanly**, committing a fresh
  recording in its place. Failing *mid-segment* leaves the page half-mutated, so the agent
  takes over that live page and finishes the step in place — but commits nothing, since a
  recording that starts mid-flow could never replay from the top; that entry is retired
  after two consecutive failures so the next run re-authors it clean.

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
is the task's identity, so a reworded prompt orphans its cached subtask split and forces a
fresh decomposition, which may cut the task differently and miss the library entries the old
split reused (`tests/test_tasks.py` pins every hash to catch this).

Word shared steps **identically** across tasks. Subtask identity is the wording, so phrasing
a common prefix the same way in a new task means it replays that prefix from day one instead
of paying to author it again.
