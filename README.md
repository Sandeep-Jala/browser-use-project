# Automation

A browser-use agent framework that logs into a web app, runs natural-language tasks from
`tasks.yaml` against it, captures network + console telemetry, and writes an HTML report
per run. Steps a task has run before replay from a shared library with no LLM.

## Pipeline

```
browser/login.py     launch Chromium (CDP open) + log in   -> (browser, page, cdp_url)
browser/session.py   attach a browser-use BrowserSession over CDP (no second login)
pipeline/files.py    resolve filenames the prompt names -> the upload allowlist (pre-login)
pipeline/decompose.py split the task into typed subtasks (declared / cached / derived / LLM)
pipeline/router.py   map a differently-worded subtask to an existing skill (alias ->
                     local embeddings -> one LLM verify)
pipeline/hybrid.py   per subtask: replay its skill (no LLM), else author + commit it
pipeline/agent_tools.py  the agent's action registry: honest, replayable custom tools
skills/              the executable unit: generated skill.py over the api.* runtime,
                     anchors for self-healing, deterministic transpiler + AST lint
pipeline/adapt.py    lift a committed script's typed values into {{param}} templates
pipeline/checks.py   deterministic per-subtask verify:/probe: checks (no LLM)
pipeline/runner.py   run ONE agent segment on the shared session; collectors via Playwright
pipeline/report.py   compile the RunResult into report.html + report.json
```

## Layout

```
tasks.yaml             # THE task registry: prompt (or declared subtasks) + marker/tags
automation/
  __main__.py          # entry point: login -> hybrid engine -> report (python -m automation)
  tasks.py             # loads tasks.yaml into TaskSpecs (load_tasks / resolve_task)
  config.py            # Config.from_env(): all settings in one place
  llm.py               # build the browser-use chat model (Azure OpenAI default / Groq)
  uploads/             # files a task prompt names by BASENAME (checked before login)
  browser/
    login.py           # Playwright login, hands off via CDP
    error_capture.py   # failure screenshots
    session.py         # attach browser-use to the authenticated browser over CDP
    recording.py       # per-run mp4 via browser-use's CDP screencast recorder (--record)
  skills/
    base.py            # Skill loader/executor: code tier preferred, steps tier fallback
    api.py             # SkillApi: the ONLY surface generated code touches (heals anchors)
    codegen.py         # deterministic transpiler steps->code + the AST whitelist lint
  pipeline/
    runner.py          # Runner + RunResult; runs ONE agent segment (owns no lifecycle)
    agent_tools.py     # the custom Tools registry the prompts rely on
    prompts.py         # all prompt text: agent rules, per-subtask prompt, decompose, router
    report.py          # RunResult -> HTML + JSON
    script_compile.py  # compile a recording into selector steps; replay + self-heal them
    adapt.py           # parameterize a committed script; match + instantiate templates
    checks.py          # deterministic verify:/probe: checks, LLM-free and fail-closed
    assertions.py      # declarative telemetry assertions (no_5xx, no_console_errors, ...)
    files.py           # resolve prompt-named upload files to absolute, existing paths
    subtask_store.py   # task + subtask identity, and the SHARED skill library (library/)
    decompose.py       # split a task into typed subtasks (action|judge)
    router.py          # semantic router: alias table -> local embeddings -> LLM verify
    hybrid.py          # THE execution engine: replay known skills, LLM only for the gaps
  collectors/
    base.py            # Collector ABC (attaches to a Playwright BrowserContext)
    network.py         # request/response/requestfailed via Playwright
    console.py         # console messages + uncaught page errors via Playwright
```

## Setup

```bash
uv sync
uv run playwright install chromium
```

Run the offline test suite (no app credentials needed) with `uv run pytest` — 1079 tests
across 49 files, none of which touch the live app.

Copy the sample env file and fill in your values (`.env.example` documents the common
keys):

```bash
cp .env.example .env
```

Minimum required: `LOGIN_URL`, `LOGIN_EMAIL`, `LOGIN_PASSWORD`, and `AZURE_OPENAI_KEY`.
`.env` is gitignored; never commit it.

Everything else has a working default (`automation/config.py` is the one place they live):

| Key | Default | Effect |
|---|---|---|
| `LLM_PROVIDER` | `azure` | `azure` or `groq` |
| `AZURE_OPENAI_MODEL` | `gpt-4.1-mini` | the model the browser agent itself runs on |
| `EXPANDER_MODEL` | `gpt-4.1-mini` | the non-agent LLM: decomposition, router verify, parameterization (name is historical — see `config.py`) |
| `SUBTASK_MAX_STEPS` | `25` | agent step budget per subtask segment |
| `USE_VISION` / `VISION_DETAIL_LEVEL` | `true` / `high` | screenshots into the agent's context |
| `SEMANTIC_ROUTER` | `true` | embeddings tier of the router; `false` = alias table only |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | local fastembed model, no network |
| `REVEAL_HIDDEN_CONTROLS` | `true` | un-hide CSS-hidden controls before snapshotting |
| `ENABLE_PLANNING` | `true` | browser-use's planner |
| `MAX_HISTORY_ITEMS` | `20` | agent history window |
| `HEADFUL` | `false` | `true` shows the browser |
| `CDP_PORT` | `9222` | port login.py opens and session.py attaches to |
| `RECORD_VIDEO` / `RECORD_VIDEO_SIZE` | `false` / native | same as `--record`; size cuts the cost |
| `ARTIFACTS_DIR` / `ERRORS_DIR` | `artifacts` / `errors` | output roots |

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
```

| Flag / Env var | Effect |
|---------|--------|
| `TASK` / `--task` | which task to run: any key from `tasks.yaml` (default `invoice`), or a full free-text prompt |
| `--fresh` | re-author **and re-record every subtask**, ignoring the library; each one that passes its gate is committed back |
| `--reauthor` | re-record only the subtasks you name — comma-list of indexes and/or prompt substrings, e.g. `--reauthor 0` or `--reauthor 'add estimate'`. Others still replay, and an entry is replaced only if the new recording passes its gate. Use it when a recording works but wanders |
| `--redecompose` | regenerate the task's cached subtask **split** (the cache is otherwise immutable per prompt) |
| `--marker` | success-marker URL fragment; `none` disables the network ground-truth gate |
| `--log-all-hosts` | record telemetry from EVERY page. By default logs are scoped to the app under test (the same scope as the assertions), so helper-tab sites and their ad stacks stay out of the artifacts. Use it when debugging a helper site itself |
| `--record` | save `artifacts/<run_id>/run.mp4`. The capture is a **time-lapse**: the browser emits a frame only when the page changes, so the waits between LLM steps collapse and a long run becomes a short clip. `RECORD_VIDEO_SIZE=1280x800` cuts the cost |

The exit code is CI-ready: `0` only when the flow completed (ground-truth gate) **and** the
telemetry assertions passed.

### Outputs

- `artifacts/<run_id>/` — `network.json`, `console.json`, `report.html`, `report.json`,
  plus `downloads/` (anything the run downloaded) and `run.mp4` under `--record`
- `library/` — **the** skill store, per subtask: `<sid>.skill.py` (the generated no-LLM
  executable) + `.anchors.json` (its element identities — where self-healing writes),
  `.template.json` (values lifted into a `{{param}}` dictionary), `.recording.json` (raw
  authoring trace — the re-compile source), `.meta.json` (uses/failures, kept out of the
  manifest so replays don't serialize on one rewritten file). `.steps.json` exists only for
  entries the transpiler can't express. A trace that failed or was interrupted parks under
  `.recording.failed.json` / `.recording.new.interrupted.json` instead of overwriting the
  good one. Shared: `manifest.json`, `aliases.json` + `embeddings.json` (semantic router),
  `archive/`
- `decompositions/<task_id>.json` — cached subtask split per task prompt
