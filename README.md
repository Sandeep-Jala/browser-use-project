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
pipeline/decompose.py split the task into typed subtasks (action|judge; cached / derived / LLM)
pipeline/router.py   map a differently-worded subtask to an existing skill (alias ->
                     local embeddings -> one LLM verify)
pipeline/hybrid.py   per subtask: replay its skill (no LLM), else author + commit it
skills/              the executable unit: generated skill.py over the api.* runtime,
                     anchors for self-healing, deterministic transpiler + AST lint
pipeline/runner.py   run ONE agent segment on the shared session; collectors via Playwright
pipeline/report.py   compile the RunResult into report.html + report.json
```

## Layout

```
tasks.yaml             # THE task registry: prompt + optional marker/tags per task
automation/
  __main__.py          # entry point: login -> hybrid engine -> report (python -m automation)
  tasks.py             # loads tasks.yaml into TaskSpecs (resolve_task / select_tasks)
  config.py            # Config.from_env(): all settings in one place
  llm.py               # build the browser-use chat model (Azure OpenAI default / Groq)
  browser/
    login.py           # Playwright login, hands off via CDP
    error_capture.py   # failure screenshots
    session.py         # attach browser-use to the authenticated browser over CDP
  skills/
    base.py            # Skill loader/executor: code tier preferred, steps tier fallback
    api.py             # SkillApi: the ONLY surface generated code touches (heals anchors)
    codegen.py         # deterministic transpiler steps->code + the AST whitelist lint
  pipeline/
    runner.py          # Runner + RunResult; runs ONE agent segment (owns no lifecycle)
    prompts.py         # all prompt text: agent rules, per-subtask prompt, decompose, router
    report.py          # RunResult -> HTML + JSON
    script_compile.py  # compile a recording into selector steps; replay + self-heal them
    assertions.py      # declarative telemetry assertions (no_5xx, no_console_errors, ...)
    subtask_store.py   # task + subtask identity, and the SHARED skill library (library/)
    decompose.py       # split a task into typed subtasks (action|judge)
    router.py          # semantic router: alias table -> local embeddings -> LLM verify
    hybrid.py          # THE execution engine: replay known skills, LLM only for the gaps
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
```

| Flag / Env var | Effect |
|---------|--------|
| `TASK` / `--task` | which task to run: any key from `tasks.yaml` (default `invoice`), or a full free-text prompt |
| `--fresh` | re-author **and re-record every subtask**, ignoring the library; each one that passes its gate is committed back |
| `--reauthor` | re-record only the subtasks you name — comma-list of indexes and/or prompt substrings, e.g. `--reauthor 0` or `--reauthor 'add estimate'`. Others still replay, and an entry is replaced only if the new recording passes its gate. Use it when a recording works but wanders |
| `--redecompose` | regenerate the task's cached subtask **split** (the cache is otherwise immutable per prompt) |
| `--marker` | success-marker URL fragment; `none` disables the network ground-truth gate |

The exit code is CI-ready: `0` only when the flow completed (ground-truth gate) **and** the
telemetry assertions passed.

### Outputs

- `artifacts/<run_id>/` — `network.json`, `console.json`, `report.html`, `report.json`
- `library/` — **the** skill store, per subtask: `<sid>.skill.py` (the generated no-LLM
  executable) + `.anchors.json` (its element identities — where self-healing writes),
  `.template.json` (values lifted into a `{{param}}` dictionary), `.recording.json` (raw
  authoring trace — the re-compile source), `.meta.json` (uses/failures). `.steps.json`
  exists only for entries the transpiler can't express. Shared: `manifest.json`,
  `aliases.json` + `embeddings.json` (semantic router), `archive/`
- `decompositions/<task_id>.json` — cached subtask split per task prompt

### How record / replay works

Every task is **decomposed** into typed subtasks (`pipeline/decompose.py`): a cached split
if the prompt has been seen, a derived match (same task shape, different values — no LLM),
or one LLM call, cached forever after. `--redecompose` regenerates it. Nodes are
`action` (replayable) or `judge` (verification wording — its success is a judgment that
cannot survive compilation, so a judge node **always runs live with the LLM and is never
cached**; the navigation around it still replays). Each passed segment's observation (its
final result) flows into every later segment's prompt, so note-then-verify tasks compare
against what was actually seen.

Each action subtask is keyed into a **global shared library** (`library/`) by its
*parameterized* prompt plus the normalized URL context it starts from. That key is the
whole trick: the "go to Bookkeeping, search and select {{business}}" prefix that a dozen
tasks share is ONE library entry — authored once by whichever task ran first, replayed by
all the others. Values are template-swapped at replay, so a different customer costs
nothing. A wording with NO entry goes through the **semantic router**
(`pipeline/router.py`): alias table (free) → local embeddings (fastembed, same-context
candidates, floor + margin gates) → one LLM verify that maps the slots — then the wording
is aliased and routes for free forever. Embeddings never decide alone.

Then, per subtask, in order:

- **Library hit** → the entry's **generated code skill** (`skill.py`, produced by a
  deterministic transpiler — no LLM writes code) executes over the `api.*` runtime: every
  verb resolves through the entry's anchor bundle with ranked selectors + fingerprint
  healing, so DOM drift heals in the anchors, never in the code. Entries the transpiler
  can't express replay their compiled steps instead. No LLM, seconds, zero tokens.
- **Miss** → the agent authors just that step, with a prompt scoped to it alone plus the
  end-state its gate expects. On a passing gate the trace is compiled, transpiled, and
  committed to the library, so it replays from then on.

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

One entry in `tasks.yaml`: a key, a `prompt` written the way a user would say it (no login
steps — the framework logs in itself), and optionally a `marker` — the URL fragment of the
create-write that proves the record saved (the ground-truth gate). **Omit the marker** for
verification / read-only tasks: the gate is then disabled instead of force-failing an
honest run. **Never edit an existing prompt's wording casually** — the prompt's hash is the
task's identity, so a reworded prompt orphans its cached subtask split and forces a fresh
decomposition, which may cut the task differently and miss the library entries the old
split reused (`tests/test_tasks.py` pins every hash to catch this).

Word shared steps **identically** across tasks. Subtask identity is the wording, so phrasing
a common prefix the same way in a new task means it replays that prefix from day one instead
of paying to author it again.
