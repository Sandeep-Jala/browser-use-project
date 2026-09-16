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

## Quick start (no terminal needed)

For handing this to someone who is not going to run five commands.

1. **macOS** — double-click `start.command`.  **Windows** — double-click `start.bat`.
2. The first run installs everything it needs, which takes a few minutes; later runs start in
   seconds. When it is ready it opens `http://127.0.0.1:8765` in a browser.
3. The first run also creates a `.env`, opens it in a text editor, and stops so four values can
   be filled in: `LOGIN_URL`, `LOGIN_EMAIL`, `LOGIN_PASSWORD`, `AZURE_OPENAI_KEY`. Save it and
   double-click again.
4. To stop it: press Ctrl+C in the black window, or just close the window.

Every failure prints one plain sentence saying what to do. On Linux, or from a terminal,
`python3 launch.py` does exactly the same thing.

Both files are thin shims over `launch.py`, which holds all the logic and is deliberately
stdlib-only — it has to run *before* the dependencies it installs exist.

<details><summary>macOS says "Apple could not verify start.command is free of malware"</summary>

Gatekeeper blocks scripts that arrived by download, AirDrop or chat, and that dialog has no
"Open" button. Either Control-click (right-click) `start.command` → **Open** → **Open**, or go to
**System Settings → Privacy & Security**, find *"start.command was blocked"*, click **Open
Anyway**, then double-click again. `git clone` avoids this entirely — files written by git are
never quarantined.

If double-clicking opens the file in TextEdit instead, its executable bit was lost in transit.
In Terminal, once: `chmod +x "<this folder>/start.command"`.
</details>

<details><summary>It says uv is not installed</summary>

macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `brew install uv`)
Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

Then double-click the start file again. The launcher deliberately does not install uv for you:
the installer edits shell profiles, which is a persistent change to the machine.
</details>

**Windows status:** the UI runs, and the process-control and text-encoding fixes needed for runs
are in (`ui/supervisor.py`'s process group and liveness probe, `encoding="utf-8"` on every text
read). It has not yet been exercised end to end on a Windows machine — do that before relying on
it. `.claude/launch.json` is macOS-only and dev-only; `start.bat` is the supported entry point.

## Setup

```bash
uv sync
uv run playwright install chromium
```

Run the offline test suite (no app credentials needed) with `uv run pytest` — 1333 tests
across 61 files, none of which touch the live app.

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

## Auto Agent (the web UI)

```bash
uv run auto-agent
```

That is the developer path. `start.command` / `start.bat` do the whole setup first and then start
this same server via `python -m automation.ui` — see [Quick start](#quick-start-no-terminal-needed).

Opens `http://127.0.0.1:8765` — bound to loopback only, with no auth, because it starts runs that
drive a logged-in browser and it deletes run directories. A `Host` header that is not this machine
is refused outright: binding to loopback does not stop a page on the internet resolving its own
hostname to 127.0.0.1, and checking `Host` does.

A [Gradio](https://gradio.app) app mounted on a small FastAPI app — mounted rather than
`launch()`ed so that Host check survives, and so run reports are served from one guarded route
instead of exposing every file under `artifacts/`. There is no code path that can mint a public
share link.

Four tabs: **Prompts · Files · Run · History**.

- **Prompts** — create, edit, delete and run prompts, written as plain English. Each step gets a
  type: *Action* (recorded once, replayed free), *Check* (`kind: judge` — the agent judges it live
  every run, never cached), or *Conditional* (a `probe:` — skipped at zero AI cost when the page
  does not show what you named). End-of-step `verify:` checks are rows of a dropdown and a text
  box. No YAML or JSON to type — there is a read-only *Raw YAML* view for reading what a prompt
  became, but nothing to hand-author. Validation runs the framework's real loaders as you type, so
  the editor cannot save a prompt the CLI would reject.
- Prompts live in `prompts/<key>.yaml`, one file each, in the same schema as `tasks.yaml`. The
  entries in `tasks.yaml` itself are **read-only** here — its comments record why each step and
  marker is worded as it is, and no YAML writer preserves them — but they can be run, or copied
  into an editable prompt.
- **Files** — upload what a prompt names, with the exact snippet to paste (quoted when the name
  has a space, because the bare-name scanner cannot see one otherwise), which prompts use each
  file, and a warning on a 0-byte iCloud placeholder.
- **Run** — start a run, watch its steps live, and **Pause / Resume / Stop** it. The controls stay
  disabled until the run arms its control channel after login, because a command sent before that
  is discarded. Sending an instruction with Resume steers an agent step; on a replaying step it is
  refused out loud and the refusal is shown.
- **History** — every run, with `PASS` / `PASS*` / `DONE` / `FAIL`, plus `STOPPED` and `CRASHED`
  for runs that never wrote a report, and the full HTML report inline.

### Steering a live run

Two levers, both serviced at a **step boundary** — never mid-action.

**Ctrl+C**, if you are sitting at the terminal: the first press queues a pause and prompts
for one instruction (Enter alone resumes); a second press aborts, saving the partial
recording and `progress.json`.

**`artifacts/control.json`**, if you are not — another shell, a background run, a UI. Each
segment clears the file and logs its path on startup. Write one command to it:

```bash
echo '{"command":"pause"}'  > artifacts/control.json   # holds at the next step boundary
echo '{"command":"resume","instruction":"close the FPS panel first"}' > artifacts/control.json
echo '{"command":"stop"}'   > artifacts/control.json   # no further action is taken
rm artifacts/control.json                              # the blunt resume; also works
```

`stop` takes no further action on the page: the step already in flight finishes, and
browser-use enters the next step and aborts it before any action runs (it logs that as
`⏹️ Agent stopping` → `The agent was interrupted mid-step` → `🛑 Agent stopped`). The
interrupted segment fails, so its trace parks as `.recording.failed.json` and cannot
overwrite a working library entry — but the task summary attributes the failure to the
segment's last receipt, not to you, so check the log for `Agent stopping` to tell an
operator stop apart from a real failure.

A command is consumed as it is applied, so it never re-fires on later steps. `resume` with
an `instruction` also works on a *running* agent — that is how you steer one without pausing
it first. The instruction is injected as a human override the agent reads on its next step.
A malformed or half-written file reads as "no command", so a hand-edit in progress can
never derail the run.

A **replaying** subtask (from `library/`, no agent, no LLM) accepts `pause` and `stop` too —
serviced between two replayed actions — but **not** steering: there is no model to give an
instruction to, so an `instruction` sent to a replay is refused out loud in the log and
ignored. `stop` during a replay aborts the run the same way Ctrl+C does, writing
`progress.json` with `status: interrupted` (and, like Ctrl+C, ending on a `KeyboardInterrupt`
traceback).

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
