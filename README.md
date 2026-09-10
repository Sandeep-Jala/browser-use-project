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
  uploads/             # files a task prompt names by BASENAME (see Uploads below)
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
    agent_tools.py     # the custom Tools registry the prompts rely on (see below)
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

Telemetry is captured with Playwright: the engine opens a second CDP connection
(`connect_over_cdp`) to the same browser the agent drives and attaches native
`page.on("console" | "pageerror" | "response" | "requestfailed")` listeners.

## Setup

```bash
uv sync
uv run playwright install chromium
```

Run the offline test suite (no app credentials needed) with `uv run pytest` — 1082 tests
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
| `EXPANDER_MODEL` | `gpt-4.1-mini` | the **non-agent** LLM — see below |
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

### A note on `EXPANDER_MODEL`

The name is stale and worth explaining, because it does not describe what the setting does.
There is no prompt expander: it was `EXPAND_SYSTEM_PROMPT` + `expand_task()` inside
`prompts.py`, which turned a terse task into one numbered prose plan for a single agent run,
and it was deleted when the whole-task execution path went away. `decompose.py` is a
different, older component that outlived it.

What the setting selects today is the **non-agent LLM**, with three live consumers:

| Consumer | Called when |
|---|---|
| the decomposer's LLM tier | **tier 4 only** — no declared `subtasks:`, no cached split, no derived match |
| the semantic router's verify tier | `SEMANTIC_ROUTER=true`, the subtask has no script yet, and it is not conditional or aux-tab |
| `adapt.parameterize` | on every segment commit |

A fourth consumer is named in the surrounding comments but never runs: browser-use's
end-of-run judge is off (`use_judge=False`), a deliberate choice — the hybrid engine's
segment gates are the verdict.

Note the shape of this: on a mature run of a task with a declared `subtasks:` block, the
decomposition the name gestures at is exactly the part that does not happen — tier 1 wins
outright and calls no LLM, and a cached split short-circuits it again. What actually flows
through this model on such a run is the router's verify tier and template parameterization.

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

### How record / replay works

Every task is **decomposed** into typed subtasks (`pipeline/decompose.py`), in four tiers —
the first that applies wins, and only the last costs tokens:

1. a `subtasks:` block declared on the task in `tasks.yaml` — built directly, **no LLM**
2. an exact cache hit on the prompt's hash (`decompositions/<task_id>.json`)
3. a **derived match**: a cached split whose parent prompt is this one with only values
   swapped, re-instantiated with the new values — same task, different customer, and it
   reuses the SAME library entries at zero token cost
4. one LLM call, validated against hallucination (a rejected attempt is retried once with
   the reason fed back), then cached forever after. Failing that, the task degenerates to a
   single whole-prompt subtask

`--redecompose` regenerates the cache, which is otherwise immutable per prompt hash.

Nodes are `action` (replayable) or `judge` (always live, never cached). **Kind is declared,
never inferred.** Until 2026-08-28 four regex nets read kind out of the wording, and each
could silently stop a segment recording: "tick the checkbox and click Verify" is a click
sequence, but the *button* is named Verify, so it was held out of the library; a slice
mentioning "the noted employee's name" re-authored at full cost every run. Now a
verification is `kind: judge` in `tasks.yaml` and everything else is an action that records.
(`kind: loop` went at the same time — it existed so the compiler would read adjacent
same-target clicks as iterations rather than slow-app retries, a guess the `repeat_click`
tool makes unnecessary by stating its own count.) Each passed segment's observation flows
into every later segment's prompt, so note-then-verify tasks compare against what was seen.

Each action subtask is keyed into a **global shared library** (`library/`) by its
*parameterized* prompt plus the normalized URL context it starts from. That key is the
whole trick: the "go to Bookkeeping, search and select {{business}}" prefix that a dozen
tasks share is ONE library entry — authored once by whichever task ran first, replayed by
all the others. Values are template-swapped at replay (`pipeline/adapt.py` binds each typed
value to a named parameter at commit time, so two fields that happen to share a value are
separate parameters and can never cross-contaminate), so a different customer costs nothing.
A wording with NO entry goes through the **semantic router** (`pipeline/router.py`): alias
table (free) → local embeddings (fastembed, same-context candidates, floor + margin gates)
→ one LLM verify that maps the slots — then the wording is aliased and routes for free
forever. Embeddings never decide alone.

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

## The agent tool layer

`pipeline/agent_tools.py` builds the browser-use `Tools` registry the agent actually gets:
**36 actions** — browser-use's 24 built-ins minus `evaluate`, plus 13 new ones. Eight of
the built-ins (`click`, `input`, `navigate`, `close`, `scroll`, `send_keys`, `find_text`,
`select_dropdown`) are re-registered under the same name: each delegates to the original and
adds behaviour, so recorded history and the compiler's branches are untouched.

`evaluate` is **removed**, not overridden: JS form-fills are invisible to the recorder, so a
run that leaned on them compiled to a script missing those fields.

| Group | Tools | For |
|---|---|---|
| Finding | `find_by_text`, `list_actions`, `find_text`, `scroll_panels`, `scroll` | locating a control in a fresh snapshot, including the nameless icon buttons browser-use renders blank and rows inside virtualized panels |
| Acting | `click`, `input`, `select_dropdown`, `repeat_click`, `send_keys`, `copy_text`, `paste_text`, `navigate`, `close` | the interaction verbs, each recorded in a shape `script_compile.py` can replay |
| Reading | `extract_data` | capture on-page data by a **replayable** locator — the value returns to the agent now *and* is re-read fresh on every replay |
| Verifying | `verify_save_registered`, `verify_download`, `detect_layout_issues`, `run_accessibility_scan` | ground truth from outside the agent's own account of events |
| Escape hatches | `skip_step`, `fail_and_stop` | abandon one objective, or end the run as a failure |

Two design rules run through all of them, and most of the file is their enforcement:

**A tool reports what happened, including nothing.** A miss is returned as a miss. When
`find_by_text` cannot find its anchor it says so rather than answering with the page's first
30 controls under a plausible-looking heading — that exact silent substitution once
convinced an agent an unopened grid was in front of it. Refusals ride the *error* channel so
a batched follow-up action cannot fire on an unchanged page, and receipts never contradict
the task's own wording.

**A tool only does what a replay can redo.** `input` presses Enter (this app's search boxes
apply on Enter) and records that it did, so the compiler mirrors it with a `press` step;
`find_by_text(click_first=True)` records the element it clicked, because a custom action has
no `index` and browser-use would otherwise drop the click from the compiled script entirely.

## Declared subtasks

The decomposer is usually right. When it isn't, `tasks.yaml` takes a `subtasks:` list and
the split becomes the edit surface — the task's `prompt` is then *derived* as the join of
its slices, so there is one copy of the wording, not two.

| Field | Meaning |
|---|---|
| `prompt` | the slice text; may carry `{{tokens}}` |
| `values` | concrete values for those tokens — the **tokenized** prompt is the library identity, so two tasks differing only in values share one recording |
| `kind` | `action` (default) or `judge` |
| `marker` | marks the save-owning slice: the parent's create-write must fire here |
| `postcondition` | cheap success check for a slice with no write: `{url_contains}` or `{visible}` |
| `tab_url` | run this slice in a helper tab at that absolute URL — opened before, closed after; the app page is never navigated |
| `verify` | deterministic end-of-segment checks (below) |
| `probe` | one check standing in for a leading-"If" conditional; FALSE = a zero-LLM no-op |

**Wording is identity.** `subtask_store.task_id` hashes the prompt to key its cached
decomposition, so rewording a prompt orphans that cache and the library reuse that came with
it. Word shared steps **identically** across tasks: subtask identity is the wording, so
phrasing a common prefix the same way in a new task means it replays that prefix from day
one instead of paying to author it again.

## Verification (`checks.py`)

A `verify:` block lists checks that must **all** hold at segment end. They are deliberately
LLM-free: `text_visible`, `text_absent`, `control_exists`, `url_contains`, `write_accepted`.

Each check re-polls briefly (default 10 s) so an async SPA or an in-flight write can settle,
then **fails closed** — a probe that cannot be evaluated is a failed check, never a pass.
Checks can only **demote** a segment the base gate already passed; they never resurrect a
failed one. Page checks run the same shadow-DOM-aware finders the authoring tools and replay
use, and write checks read the segment's own network window, so a late settle is still seen.

`probe:` is the conditional form — one check, a short poll (absent is a routine outcome, not
a failure), gating a leading-"If" slice. FALSE means the slice is skipped with no LLM spend
at all; TRUE means it replays or authors like any action.

## Uploads and downloads

Task files live in `automation/uploads/` and the prompt names them by **basename**
("upload the file `New_Employees_List.csv` using the Select or drop file control"). Before
login, every filename the prompt mentions must resolve to a non-empty file there; the
resolved **absolute** paths become the agent's upload allowlist and the paths a replayed
upload step re-checks.

That absolute-and-existing requirement is not a nicety. A native OS file dialog can't be
driven from the DOM, so the file is attached programmatically over CDP — and a relative
nonexistent path "uploads" perfectly well, right up until clicking Save makes the page read
the ungranted file and Chrome kills the renderer.

Downloads are re-pointed per run into `artifacts/<run_id>/downloads/`, and `verify_download`
checks the files that actually landed rather than the agent's account of what it clicked.

## Self-healing replays

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
  recording that starts mid-flow could never replay from the top; that entry is retired on
  its **first** failure (`_ARCHIVE_AFTER_FAILURES = 1`) so the next run re-authors it clean.
  The threshold was 2 until 2026-09-09: waiting for a second failure cost a whole extra run
  re-fighting a stale recording, and protected less than it appeared to — a silent
  wrong-row replay counts as a pass and never trips the counter at all. Consecutive
  failures reset on any success, so a healthy entry is never touched.

## Assertions

Every run is checked against declarative telemetry assertions (`pipeline/assertions.py`):
no 5xx responses, no failed requests, no console errors/exceptions by default; per-task
overrides live on the task's `TaskSpec.assertions` (e.g. allowlist a known noisy error, add a
`response_ok` check for an extra endpoint, or disable a rule). Assertions never change the
flow verdict (`is_successful` — that's the ground-truth gate's job); they gate a separate
`assertions_passed` verdict. A run whose flow passed but whose telemetry didn't shows as
**PASS\*** (amber) in the report, and the combined verdict drives the exit code.

## Adding a task

One entry in `tasks.yaml`: a key, a `prompt` written the way a user would say it (no login
steps — the framework logs in itself), and optionally a `marker` — the URL fragment of the
create-write that proves the record saved (the ground-truth gate). **Omit the marker** for
verification / read-only tasks: the gate is then disabled instead of force-failing an
honest run. Add `tags` if you want the task grouped, and `subtasks` only when the decomposer
keeps splitting it wrongly.

Then re-read **Wording is identity** above before editing any prompt that already runs.
