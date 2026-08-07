# Deterministic per-subtask verification (in-house check layer) + dead-weight cleanup

Branch: `feature/hybrid-pipeline` (HEAD `0b9f203`) · Repo: `/Users/sandeepjala/Desktop/Automation`
Design reviewed against code by a second pass (2026-08-07); all findings folded in below.

## Context

browser-use segments frequently claim success without doing the work ("misses"). Root cause, confirmed in code: the default gate kind `steps` (`automation/pipeline/hybrid.py:300`) is `return steps_ok`, where `steps_ok = bool(history.is_successful())` — the model's own `done(success=true)` flag. The latest run's gate mix was 10× steps, 1 postcondition, 1 download, 1 interrupted. Only `marker`/`download` gates check reality; tasks.yaml declares zero postconditions. Deterministic evidence is already captured per action (click receipts with server-body verdicts, fill read-backs, per-segment network/download windows, shadow-DOM-aware probes) but `evaluate_gate` never reads it. A self-reported pass also promotes recordings into `library/` and feeds unverified "findings" downstream.

Inspiration: Predicate Systems' verification-first model (declarative checks, "retry verification not actions", required checks override agent claims). **User decisions (2026-08-07):** build **in-house** (their SDK needs a cloud API key + browser extension — rejected); on failed check **fail honestly and stop** (no repair loop — the verify/repair pipeline was rolled back 2026-08-03 and stays out); check sources are **declared `verify:` blocks in tasks.yaml + receipt roll-up** (no decomposer/LLM-emitted checks; **zero LLM calls in the verification path**); cleanup buckets approved: judge remnants, discarded-result plumbing, housekeeping (**layout/a11y subsystem stays**).

Binding rulings honored from the rolled-back pipeline: fail-closed (unevaluable check ⇒ fail); checks **demote only** (never resurrect a failed base gate; marker/download stay authoritative in both directions); `use_judge` stays False; machinery only — no task-specific prompt tuning.

## Design

### 1. New module `automation/pipeline/checks.py`

**Check schema** — a `verify:` list on a subtask; implicit AND; each item a single-key mapping (+ optional `timeout_s`):

```yaml
subtasks:
  - prompt: >-
      Return to the Payroll tab and click Add Employee, then fill in the form … Save.
    verify:
      - write_accepted: "Employees"
      - text_visible: "Employee"
        timeout_s: 15
```

Phase-1 vocabulary (YAGNI — no combinators; `any_of` noted as future work only):

| kind | passes when | evaluated with |
|---|---|---|
| `text_visible: str` | text (all tokens) visible on the page | `RAW_TEXT_FIND_JS` via `page.evaluate(RAW_TEXT_FIND_JS % json.dumps(tokens))` — the exact call shape of replay's `_extract_value` fallback (`script_compile.py:1927`), shadow-aware |
| `text_absent: str` | that probe finds nothing (eventually-absent, e.g. dialog gone) | same probe, inverted |
| `control_exists: str` | a control named X exists & visible | `RAW_FIND_JS % (json.dumps(tokens), "false")` (`script_compile.py:1923` call shape) |
| `url_contains: str` | substring of `page.url` (case-insensitive) | mirrors existing postcondition branch (`hybrid.py:267-272`) |
| `write_accepted: str` | a `POST/PUT/PATCH` whose URL contains the fragment, in **this segment's window**, status 2xx/3xx ("settled" = `status is not None` — window records have no `settled` wrapper key), and — when a JSON body was captured — `_write_verdict` non-negative (`agent_tools.py:1080-1134`; a 200 whose body says "already submitted" is a REFUSAL) | mirror `_first_create_write` exactly (`runner.py:82-98`: method + URL-substring + 2xx/3xx, no resourceType filter), then apply `_write_verdict` to `record["body"]` when present |

Notes:
- **Tokenizer**: there is no shared helper today — `[t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]` is an inline idiom at `agent_tools.py:1690`, `script_compile.py:1840`, `script_compile.py:1921`. Extract a tiny `_query_tokens()` in `script_compile.py` and reuse it from checks.py (optionally repoint the three inline sites).
- **Eventually semantics**: each check polls (0.5 s interval) until pass or `timeout_s` (default `_CHECK_TIMEOUT_S = 10`), generalizing `_settled` (`hybrid.py:220-235`). Network window records are **live dicts** (`results()` returns the same objects handlers mutate, `network.py:239-240`; `requests_since` slices that list, `hybrid.py:652-656`), so `write_accepted` re-polls status/body on the same records — no in-flight false-fails.
- **Noise writes**: every segment window contains `/auth/webpush`, `/client/negotiate`, `/oauth/token` etc. — fragments must stay specific; windowing + specificity suffices (verified in runs 120553/125406).
- Probe exception ⇒ check fails with `error` recorded (fail-closed, mirroring `hybrid.py:295-298`).
- **Import discipline**: checks.py top-level imports stdlib-only; import `_write_verdict` / `_first_create_write` lazily inside evaluators so `tasks.py → checks.parse_verify` never pulls `browser_use` into `load_tasks()`.

API: `parse_verify(raw) -> tuple[Check, ...]` (loud `ValueError` on unknown kind/empty arg/bad timeout — same fail-loud style as `_spec_from_entry`), `evaluate_checks(page, requests_window, checks, *, poll=True) -> list[dict]` returning per-check `{kind, arg, ok, evidence, error}`, and `receipt_rollup(...)` (§4).

### 2. Declared checks: tasks.yaml → gate (exact threading sites)

- `SubtaskDecl` (`automation/tasks.py:36-56`) gains `verify: tuple[dict, ...] | None`; parsed/validated in `_spec_from_entry` (`tasks.py:86-133`).
- **Token substitution at LOAD time** in `_spec_from_entry` via the existing `_instantiated`/`_DECL_TOKEN` mechanism (`tasks.py:76-83`) — NOT downstream: `sstore.TOKEN_RE` is lowercase-only (`{{[a-z][a-z0-9_]*}}`, `subtask_store.py:54`) vs `_DECL_TOKEN`'s `{{\w+}}`, so late substitution would silently skip uppercase token names.
- **Identity invariant**: `_spec_from_entry` joins slice wording only (`tasks.py:107-122`); `task_id` hashes prompt only (`subtask_store.py:114-122`), `subtask_id` hashes template+context only (`subtask_store.py:135-138`). Adding `verify:` cannot move `f3bd8b35ccb044a9`. Pinned by test.
- Threading (all four tiers construct `Subtask` through one function):
  - `Subtask` dataclass gains `verify` field, default `None` (`decompose.py:218-238`);
  - tier-1 spec-declared raw dict adds `"verify": d.verify` (`decompose.py:541-546` — it enumerates fields explicitly);
  - `_build_subtasks` ctor call adds `verify=d.get("verify")` (`decompose.py:284-294`) — tier-2/3/4 dicts lack the key ⇒ `None`;
  - `_as_cache` (`decompose.py:308-319`) enumerates fields explicitly and therefore **drops `verify` from saved caches** — desired: cache untouched, tier-3's `{**sub, …}` (`decompose.py:456`) can never resurrect stale checks.
- `whole_prompt_fallback` (`decompose.py:417-418`) builds its blob without `verify` ⇒ a fallback segment never carries declared checks (harmless; spec-declared tasks can't fall back — invalid specs already error loudly at `decompose.py:550-552`).

### 3. Gate integration (the seam)

- `Gate` (`hybrid.py:128-135`) gains `checks: tuple = ()`. `segment_gate` (`hybrid.py:138-161`) attaches `sub.verify` to **whatever** base kind it resolves. Both `segment_gate` (5 early returns) and `evaluate_gate` (4 exit branches) need restructuring to a single exit — or attach checks post-hoc around the existing returns.
- `evaluate_gate` (`hybrid.py:237-300`): compute base `(ok, detail)` exactly as today; then if `gate.checks`: when base passed, evaluate with polling and `ok = ok and all(c["ok"])`; when base failed, evaluate once without polling purely for report detail (mirrors the existing failed-steps single-evaluation convention). `detail["checks"]` is added **only when `gate.checks` is non-empty** — `test_steps_gate_is_passthrough` pins `detail == {"kind": "steps"}` exactly (`tests/test_hybrid.py:1162-1164`). Segment error names the first failing check + evidence, e.g. `deterministic check failed: write_accepted "Employees" — no accepted create-write in this segment's traffic`.
- Call sites already have everything in scope: `page=self.current_page()` (`hybrid.py:766, 825`), requests/downloads windows (`hybrid.py:767, 826`).
- **Window-slice gap**: `requests_window` is sliced before the call; a write that *starts* after the slice is invisible to `write_accepted`. Narrow in practice (the click wrapper already waits up to `_WRITE_SETTLE_S=8 s` inside the action, `agent_tools.py:1198-1201`); accept and list under risks — or pass a window-provider closure if it bites.
- **Replay segments**: a replay that passes steps but fails a declared check gets `seg.ok=False`, which triggers the **existing** in-place agent takeover (`hybrid.py:1536-1575`) — this is pre-existing hybrid recovery machinery, distinct from the rolled-back verify/repair loop, and the takeover's authored segment then faces the same checks. Chosen behavior: **keep the takeover**. Side effect to watch (risk #4): `bump_meta(fail_count=1)` + `_ARCHIVE_AFTER_FAILURES=2` (`hybrid.py:71, 1238-1241`) means a *wrong* declared check archives a healthy library entry after two runs.
- **Free consequences** (state in tests): a failed-check segment cannot commit its recording (`hybrid.py:1221-1234`), cannot promote heals (`hybrid.py:1532-1534`), and stops the run via the existing `break` (`hybrid.py:1653-1656`).
- `_describe_expected_end` (`hybrid.py:164-179`): generically append the declared checks in gate-enforced order (e.g. `mechanical verification will require: an accepted write to "Employees"; the text "Alistair Allan" visible`). Keep the mirror-the-gate branch-order convention. Generic machinery — only tasks with `verify:` see it.

### 4. Receipt roll-up (steps/postcondition agent segments only)

Replay `steps_ok` is already honest (`run_steps` raises). **Scope**: roll-up applies only to agent segments whose gate kind consults `steps_ok` (`steps`, `postcondition`); marker/download verdicts are already deterministic in both directions (`hybrid.py:252-261`) and are exempt — otherwise a marker-proven save followed by a redundant refused re-submit ("already submitted", the documented FPS pattern) would false-fail.

**Structured stamping first** (no prose parsing). Today `_network_outcome` (`agent_tools.py:1184-1215`) returns `(text, fired, accepted)` but `_click_outcome_suffix` (`1294-1344`) returns only the string — the booleans never reach the wrappers. Change: `_click_outcome_suffix` returns `(suffix, write_outcome | None)` where `write_outcome = {"fired": bool, "accepted": bool, "t0": monotonic}`; both call sites stamp it into `ActionResult.metadata`:
- `_click_with_dialog_outcome` (`agent_tools.py:1364-1370`): `res.model_copy(update={"metadata": …})` **replaces** the field — must merge `{**(res.metadata or {}), "write_outcome": …}`;
- find_by_text click branch (`agent_tools.py:1888-1893`): `meta = {"interacted_element": captured} if captured else None` — merge, don't overwrite (the compiler reads `metadata["interacted_element"]` and only `.get()`s known keys — `script_compile.py:538, 664` — so a sibling key is safe).
- One-line comment at the stamping site: if browser-use's coordinate-clicking re-registration ever drops the click wrapper (documented hazard, `agent_tools.py:1383-1385`), stamps vanish and roll-up goes silently inert — fail-safe in the right direction.

**Phase-1 rules (conservative, exactly two):**
1. **Trailing refusal**: the segment's last non-`done` action ended on the **error channel** with a refusal-family metadata stamp (`no_click`: `agent_tools.py:1829, 1835, 1867, 1884`; `no_fill`: `551, 569, 580, 588, 599, 1430`), yet the agent reported success ⇒ fail. Content-channel `no_click` stamps (candidate listing `1920-1921`, static-text probe `1777-1778`, not-clickable listing `1754-1755`) must NOT trip this — hence error-channel + stamp required together.
2. **Refused/failed final write, nothing accepted**: the last `write_outcome`-stamped receipt has `fired and not accepted` **and no accepted write exists anywhere in the segment's window** (an "already exists" refusal after an earlier accepted write is proof of completion — the app's own notification protocol treats it that way, `runner.py:554-559`) ⇒ re-poll live records at gate time (`_writes_accepted`-style scan, `agent_tools.py:1162-1181`); still nothing accepted ⇒ fail.

History accessors (the `_history_extracts` pattern, `hybrid.py:868-873`): `history.history` → `item.result: list[ActionResult]` → `res.is_done` / `res.error` / `res.metadata`. Metadata survives in-memory — the gate site reads `out["history"]` directly (`hybrid.py:814`); only `save_history` drops it (why `restore_result_metadata` exists, `runner.py:203-232`).

`receipt_rollup(history, requests_window) -> (ok, reasons)` in checks.py; computed at the agent gate call site and passed as `evaluate_gate(..., rollup=...)`; demote-only; `detail["rollup"] = reasons`. Deliberately NOT rules: dialog-outcome heuristics, unrepaired-fill tracking, `skip_step` accounting — future work only if a real run motivates them.

### 5. Surfacing

- `progress.json`: gate detail now carries `checks`/`rollup` (serialization is already `Segment.as_dict` → gate dict, `hybrid.py:341-350`).
- `report.html`: per-subtask row (`report.py:318-354`) renders per-check ✓/✗ + evidence; failure reason flows through `seg.error`.
- Console: segment-failure print already surfaces `seg.error` — now deterministic and specific.

### 6. Pilot checks in tasks.yaml (small, provably-true only)

Target `payroll_food_limited_e2e_rti` (id `f3bd8b35ccb044a9` — verify blocks don't touch wording, id must not drift). Start with **four**:
- Add-Employee slice (`tasks.yaml:760-767`): `write_accepted: "Employees"` (create POST proven in run 120553, body captured).
- Both RTI loop slices (`tasks.yaml:805-815`, `823-830`): `text_visible: "Alistair Allan"` / `text_visible: "Bruce Wright"` — their declared stop conditions, literal on the page on success. These are steps-gated loop segments, the exact class that lied in the latest run (died at index 12).
- FPS submit slice (`tasks.yaml:831-835`): `write_accepted: "FPS"` (proven `POST …/Years/27/FPS`).

Later candidates (proven writes with captured bodies in run 125406): `write_accepted: "DataRequest"` (slice 7), `write_accepted: "ManualStatus"` (slice 9). **Avoid** `write_accepted: "emails"` — `/emails/saveDraft` fires on panel open and substring-matches.

**Authoring rules (document as a tasks.yaml comment):** never declare `write_accepted` where the app stages client-side (modal Save fires NO request; Pay-Forecast cell edits fire no write even on success) — only writes proven in a passing run's `network.json`. For `url_contains`, use a literal stable fragment of the raw URL (e.g. `employees`) — never a normalized context from `progress.json` (those are lowercased with volatile segments replaced by `*`, `subtask_store.py:57-83`, and will never substring-match a real URL). Dynamic values (noted fakenamegenerator identity) cannot be declared statically — don't try.

## Cleanup (separate commits, suite green after each)

**(a) Judge remnants** (`use_judge=False` at `runner.py:432` makes these dead; keep that flag + its regression note, trimming only the part of the `runner.py:427-431` comment that references `judge_llm` internals):
`_extract_judgement` (`runner.py:773-789`) + `"judgement"` in the segment return dict (`runner.py:717`, docstring `384`); `judge_llm` ctor param/attr (`runner.py:318, 337-338`) + `Agent(judge_llm=…)` kwarg (`runner.py:433`) + wiring at `__main__.py:154`; dead judge print (`__main__.py:199-202`); `_render_judgement` (`report.py:394-428` — **stop at 428**: lines 431-442 are `_A11Y_PREFIX`/`_LAYOUT_*` constants of the kept layout/a11y renderer) + call (`report.py:233`); `RunResult.judgement` field (`runner.py:273-275`) + `judgement=None` (`hybrid.py:720`); stale "end-of-run judge" comment (`hybrid.py:151-155`); stale judge-phrase comments at `adapt.py:10` ("one LLM call" — `match_template` is deterministic), `tasks.yaml:11`, `__main__.py:100`, `runner.py:3` docstring.

**(b) Discarded-result plumbing** (finalize builds `RunResult` with empty lists, `hybrid.py:715-721`; hybrid consumes only `out["history"]`/`out["usage"]`, `hybrid.py:814-820`):
per-segment screenshot/step extraction (`runner.py:671-696` — runs every segment, always thrown away; `history.screenshots()` isn't free); `_render_steps` + `_action_label` (`report.py:531-571`, `711-723`); `_render_screenshots` (`report.py:592-612`); `final_url` from always-empty `urls` (`report.py:197`); the dead `overrode_success` ground-truth override note — `hybrid.py:713` (hardcoded `False`), the read at `report.py:292` + `if overrode:` branch `report.py:308-311` inside `_render_ground_truth`, print at `__main__.py:196-198`. **Do NOT touch the `PASS*` status logic at `report.py:99-104`** — it is live, driven by `assertions_passed`. Keep the `RunResult` fields themselves (report.json schema stability) — delete only dead computation and renderers. **Keep `_render_ui_scans` + layout/a11y tools.**

**(c) Housekeeping:**
`git worktree remove .claude/worktrees/compassionate-moser-5569e9` (stale, detached at 1cb8451); delete 9 stale `.pyc` of removed modules (ui, contract, intent, knowledge, page_js, planner, suite, task_store, verify) under `automation/**/__pycache__/`; delete orphan `library/c39225403028a237.recording.json` (280,786 B, no manifest entry, no siblings); delete `DIALOG_ANCESTOR_JS` (`script_compile.py:1129-1152` incl. its comment; next live symbol `DIALOG_STAMP_JS` starts at 1163 — don't over-delete) + its only usage (one test in `tests/test_dialog_refind_js.py:117`); dedupe `_RS_FILTER_ID` (`runner.py:103` → import from `script_compile.py:152`; runner→script_compile is cycle-free); drop unused `match_template(llm=…)` param (`adapt.py:283-288`; sole caller `skills/base.py:99` doesn't pass it); fix stale docstring `subtask_store.py:17` (`.steps.json` no longer primary); drop stale `.gitignore:32` `recordings/` entry; drop commented dup `llm.py:45`.

## Implementation stages (TDD per stage; run tests via `.venv/bin/python -m pytest` — never anaconda pytest)

1. **`checks.py` core**: schema + `parse_verify` + `evaluate_checks` + polling + fail-closed; `_query_tokens()` extraction in script_compile. New `tests/test_checks.py`.
2. **Declaration path**: `SubtaskDecl.verify` → `_spec_from_entry` validation + load-time token substitution → the three decompose threading sites (§2). Tests in `test_tasks.py`/`test_decompose.py` incl. **id-stability** (same task_id with/without `verify:`; FROZEN_TIDS untouched) and cache-drop round-trip.
3. **Gate wiring**: `Gate.checks`, `segment_gate` attach, `evaluate_gate` merge (demote-only both directions: base-pass+check-fail ⇒ fail; base-fail+check-pass ⇒ fail with detail; marker-pass+check-fail ⇒ fail), `detail["checks"]` only when non-empty (existing passthrough test stays green), `_describe_expected_end` line, progress/report surfacing. Tests in `test_hybrid.py`, `test_prompts.py`; report render smoke.
4. **Receipt roll-up**: `_click_outcome_suffix` signature change + `write_outcome` merge-stamping in both click paths (`test_agent_tools.py` — including metadata-merge tests: `interacted_element` sibling preserved), `receipt_rollup` rules (trailing error-channel refusal fails; content-channel no_click does NOT trip; refused-final-write with empty window fails; refused-after-accepted passes; refusal-then-recovery passes; empty history passes; marker/download exempt), call-site wiring + `rollup=` kwarg.
5. **Cleanup commits** (a) → (b) → (c), each with the full suite green (**baseline 441 tests**; one dialog-JS test case deliberately removed).
6. **Pilot + live verification**: add the four pilot `verify:` blocks + the authoring-rules comment; then
   - unit-level: full suite green;
   - live negative test on a CHEAP task: temporarily add an unsatisfiable check (e.g. `text_visible: "zz-not-on-page"`), run, confirm the segment fails with the deterministic reason, `progress.json` gate detail shows the failing check, no recording committed; revert;
   - live positive: run `payroll_food_limited_e2e_rti` (~25-40 min; machine must stay awake) and confirm previously-passing segments still pass, report shows per-check ✓, `write_accepted` verdicts appear, and the two loop checks bite (or pass honestly). Watch the false-fail vectors below.

## Risks (top false-fail vectors → mitigations)

1. **`write_accepted` where no write exists** (client-staged saves) → authoring rules above; pilot limited to proven writes.
2. **In-flight final write at gate time** → live-record re-poll up to `timeout_s`; roll-up rule 2 re-polls before verdicting. Residual: a write that *starts* after the pre-call window slice is invisible (narrow — the click wrapper already waits 8 s in-action); if observed, pass a window-provider closure.
3. **`text_visible` on lazy/virtualized content** → 10 s eventually-poll, shadow-aware probe; pilot texts are the loop stop-condition names the app renders in stable chrome.
4. **A wrong declared check attrits the library**: replay check-fail → agent takeover (kept, pre-existing recovery) but also `fail_count` bump; two consecutive check-fails archive the entry (`_ARCHIVE_AFTER_FAILURES=2`). Mitigation: pilot checks are obviously-true; check failures print verbatim evidence so a wrong check is diagnosed in one read.
5. **Roll-up over-reach** → exactly two conservative rules, steps/postcondition gates only, last-action/last-write only, reasons quoted in `seg.error`; rules trivially prunable.

## Out of scope (explicitly)

Repair/retry loops (rolled back 08-03, stays out); decomposer/LLM-emitted checks; Predicate SDK; `any_of` combinators; loop-`until` structured enforcement; dialog-outcome roll-up rules; layout/a11y removal; artifacts/ retention policy.

## Post-approval note

First execution step: copy this design into `docs/superpowers/specs/2026-08-07-deterministic-subtask-verification-design.md` and commit (plan mode forbade repo writes during planning).
