"""Registry (tasks.yaml) loading, identity, and selection tests."""

import pytest

import automation.tasks as tasks_mod
from automation.pipeline import subtask_store as ss
from automation.tasks import SubtaskDecl, TaskSpec, load_tasks, resolve_task


# ------------------------------- the real registry -------------------------------


# payroll_food_limited_e2e_rti STARTED as a byte-exact join of payroll_food_limited_e2e +
# payroll_rti_process, and was pinned to stay one. That guard was dropped 2026-07-27: the
# merged prompt has since been tuned on its own (pay-forecast "clear the ... now set it",
# explicit Cost/Employee contribution fields, no repeated navigation), so it is the refined
# copy and the halves are the standalone ones. They now drift on purpose — when a shared
# step is reworded, update each copy that needs it deliberately.


def test_no_login_steps_or_credentials_in_prompts():
    # The framework logs in itself; a prompt must never carry the login step, credentials,
    # or the raw books URL (tasks say "go to Bookkeeping module" instead).
    for key, spec in load_tasks().items():
        low = spec.prompt.lower()
        for forbidden in ("login using", "log in to http", "password", "aoadmin",
                          "welcome1@", "@capsitech.com", "demo.admin@",
                          "actingoffice.com/books"):
            assert forbidden not in low, f"{key}: prompt still carries {forbidden!r}"


def test_markers_only_where_known():
    tasks = load_tasks()
    # Bookkeeping/CRM write tasks keep their verified markers...
    assert tasks["invoice"].marker == "Invoices"
    assert tasks["credit_notes"].marker == "Refunds"
    assert tasks["mileage"].marker == "MileageClaims"
    # ...and every NPS verification task runs with the gate disabled (marker absent) —
    # a guessed marker would force-fail honest successes.
    nps = [t for t in tasks.values() if "nps" in t.tags]
    assert len(nps) == 35
    assert all(t.marker is None for t in nps)


def test_specs_are_frozen_taskspecs():
    tasks = load_tasks()
    assert all(isinstance(t, TaskSpec) and t.key == k for k, t in tasks.items())
    with pytest.raises(Exception):
        tasks["invoice"].marker = "X"  # type: ignore[misc]


# ------------------------------- resolution / selection -------------------------------


def test_resolve_known_key_case_insensitive():
    assert resolve_task("  INVOICE ").key == "invoice"


def test_resolve_free_text_gets_no_marker():
    # No marker inference: an ad-hoc task may fire no create-write at all; the gate is
    # enabled for ad-hoc prompts only explicitly, via --marker.
    spec = resolve_task("add a new invoice for ACME with qty 3")
    assert spec.key == "adhoc"
    assert spec.marker is None


def test_resolve_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown TASK key"):
        resolve_task("not_a_task")


# ------------------------------- loader edge cases -------------------------------


def test_missing_registry_is_empty_and_adhoc_still_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks_mod, "TASKS_FILE", tmp_path / "tasks.yaml")
    assert load_tasks() == {}
    spec = resolve_task("do a thing on the current page")
    assert spec.key == "adhoc" and spec.marker is None
    with pytest.raises(ValueError, match="Unknown TASK key"):
        resolve_task("invoice")


def test_entry_without_prompt_fails_loud(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("broken:\n  tags: [x]\n")
    with pytest.raises(ValueError, match="non-empty 'prompt'"):
        load_tasks(p)


def test_non_mapping_registry_fails_loud(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_tasks(p)


def test_subtask_tab_url_parses_and_rejects_relative(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        "aux_task:\n"
        "  subtasks:\n"
        "    - prompt: 'search DuckDuckGo for the thing'\n"
        "      tab_url: https://duckduckgo.com\n"
    )
    spec = load_tasks(p)["aux_task"]
    assert spec.subtasks[0].tab_url == "https://duckduckgo.com"

    p.write_text(
        "aux_task:\n"
        "  subtasks:\n"
        "    - prompt: 'search DuckDuckGo for the thing'\n"
        "      tab_url: duckduckgo.com\n"
    )
    with pytest.raises(ValueError, match="tab_url must be an absolute"):
        load_tasks(p)


def test_entry_fields_parse(tmp_path):
    # A prompt kept alongside subtasks must AGREE with their join (2026-08-05: the
    # slices are the source of truth; the prompt here pins block-scalar folding).
    p = tmp_path / "t.yaml"
    p.write_text(
        "My_Task:\n"
        "  prompt: >-\n"
        "    open settings\n"
        "    save it\n"
        "  marker: Things\n"
        "  tags: [a, b]\n"
        "  subtasks:\n"
        "    - prompt: 'open {{page}}'\n"
        "      values: {page: settings}\n"
        "    - prompt: 'save it'\n"
        "      marker: Things\n"
    )
    tasks = load_tasks(p)
    spec = tasks["my_task"]  # keys are lowercased
    assert spec.prompt == "open settings save it"  # block-scalar folding collapsed
    assert spec.marker == "Things" and spec.tags == ("a", "b")
    assert spec.subtasks == (
        SubtaskDecl(prompt="open {{page}}", values={"page": "settings"}),
        SubtaskDecl(prompt="save it", marker="Things"),
    )


async def test_e2e_rti_declared_subtasks_survive_validation(tmp_path, monkeypatch):
    """The declared split of the e2e_rti mega-task must pass Tier 1 validation and never
    degrade to the whole-prompt fallback (the LLM decomposer rejected its own splits 3x
    per run on 2026-08-05 and every run degenerated to one blob). The declared prompts
    are VERBATIM slices whose concatenation reproduces the task prompt exactly — the
    identity contract that keeps shared RTI-half sids aligned with payroll_rti_process."""
    from automation.pipeline import decompose

    monkeypatch.setattr(ss, "LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(ss, "LIBRARY_MANIFEST", tmp_path / "library" / "manifest.json")
    monkeypatch.setattr(ss, "DECOMPOSITIONS_DIR", tmp_path / "decompositions")
    spec = load_tasks()["payroll_food_limited_e2e_rti"]

    joined = " ".join(" ".join(d.prompt.split()) for d in spec.subtasks)
    assert joined == spec.prompt          # verbatim partition, no reword drift

    subs = await decompose.get_decomposition(spec.prompt, llm=None, spec=spec)
    # 2026-09-09: TWENTY-THREE slices (was 22 — the Add Employee save grew a probe-gated
    # 'NI category applied by default … continue with the same?' slice behind it; every
    # index after slice 2 shifted by +1).
    # 2026-09-08: TWENTY-TWO slices (was 20 — the Send Email slice's inline 'click Send
    # again' and 'still Drafted' recoveries became probe-gated slices; every index after
    # slice 7 shifted by +2).
    # 2026-09-01: TWENTY slices. The RTI half was re-commented on 2026-08-24 (leaving six)
    # and uncommented for good two days later in 24dd5f3, restoring the Data Request +
    # Send Email + Sent-badge + Verify slices along with the payrun pass; 85c3709 then
    # split the FPS slice's trailing "if it shows an error, click cancel" into a probe-
    # gated slice of its own, taking 19 to 20. The e2e half is unchanged and still the
    # user's chosen shape: the fakenamegenerator aux producer, then two consumers.
    assert len(subs) == 23                # Tier 1 won; no fallback blob
    assert not any(getattr(s, "fallback", False) for s in subs)
    # Every slice is a recordable action — no judge, so every segment can cache. subs[12]
    # ("tick the Select Employee checkbox ... and click Verify") is the one to watch: its
    # kind: action is commented out in tasks.yaml, and it classifies action anyway only
    # because "click Verify" is an imperative on a control name. If this list ever grows
    # a "judge", that slice reworded into verification wording and stopped caching.
    assert [s.kind for s in subs] == ["action"] * 23
    assert subs[1].tab_url and "fakenamegenerator" in subs[1].tab_url   # aux producer
    # The producer-override keeps the noting slice OUT of the consumer net; the two
    # slices that USE the noted identity stay consumers (bindings-only replay).
    assert not decompose.consumes_noted_data(subs[1].template_prompt)
    assert decompose.consumes_noted_data(subs[2].template_prompt)
    assert decompose.consumes_noted_data(subs[4].template_prompt)
    assert "download" in subs[6].template_prompt.lower()              # download subtask
    # The RTI half, back from the commented block: the probe-gated popup slice and the
    # probe-gated Submit-error slice are what make the pass deterministic, and the FPS
    # slice is the one that needs FOOD LIMITED reset.
    assert subs[14].probe and subs[20].probe
    assert subs[19].allow_write_refusal                # FPS submit may honestly refuse


# --------------------- payroll_detailed_review_fps, split into parts ---------------------
# 2026-08-26 (commit 24dd5f3): the four parts became TWO and the whole-task key was deleted,
# so the parts are now the only vehicle and there is no second copy of any slice to keep in
# sync — the drift-in-two-places arrangement noted here before is gone with it.
# part1 = everything up to and including Verify all (old parts 1-3, minus their duplicated
# openers). part2 = the old part4, renamed, with its single Jun-26 payrun pass expanded to
# ten monthly passes.
FPS_PART_SLICE_COUNTS = {
    "payroll_detailed_review_fps_part1": 10,  # opener + request + email + bonus + OTP
                                              # portal + payment/expense/deduction + Next
                                              # + Verify all
    # opener + Apr-26 period + the Save-&-Next popup conditional (3), then Apr-26's
    # Save-&-Next-x5 + FPS + sync conditional + Payroll-&-RTI quartet (4), then that
    # quartet behind a period change for each of May-26 .. Mar-27 (11 * 5 = 55).
    "payroll_detailed_review_fps_part2": 62,
}
# The opener stops at the business NAME on purpose: which business a part runs against is
# ordinary task data and changes freely (part2 moved FOOD LIMITED -> CREAMOS LTD.), while
# the property worth guarding is that each part re-states the selection at all — without it
# the part cannot run on its own.
FPS_PART_OPENER = "Go to the Payroll module, search for and select the business name "
# Node kinds the parts resolve to — all ACTION, which is the property that matters: judge
# and (until it was removed on 2026-08-28) loop nodes are never cached, so any slice that
# flips off "action" starts re-authoring live every run. Two slices are only action because
# they were made to be. part1's OTP slice is an action despite "copy the 6 digit number"
# (producer wording is not verification), and its last slice carries an explicit
# `kind: action` because the button named "Verify all" is otherwise enough to classify it
# judge. All wording-driven otherwise; see decompose.node_kind.
FPS_PART_SLICE_KINDS = {
    "payroll_detailed_review_fps_part1": ["action"] * 10,
    "payroll_detailed_review_fps_part2": ["action"] * 62,
}


def test_detailed_review_fps_parts_are_independently_runnable():
    """Each part re-states the business selection so it runs on its own, and its prompt is
    derived from its own slices (one edit surface, no parallel copy).

    This test used to additionally assert the parts rebuild the full task's slice list
    byte-for-byte. That premise is dead as of 2026-08-25 and deliberately so: part2 splits
    the portal body into payment / expense / deduction slices where the full task has one,
    and its final slice now says step through the remaining employees doing NOTHING else,
    while the full task's still says "repeat all of the above for them". The two copies have
    diverged in INTENT, which is the accepted cost of keeping both (see the tasks.yaml
    comment above the parts). Restoring the rebuild assertion would mean reverting a
    deliberate wording choice, so guard what is still true instead."""
    tasks = load_tasks()
    for key in FPS_PART_SLICE_COUNTS:
        spec = tasks[key]
        slices = [" ".join(d.prompt.split()) for d in spec.subtasks]
        assert len(slices) == FPS_PART_SLICE_COUNTS[key], key
        assert slices[0].startswith(FPS_PART_OPENER), key   # or it cannot run alone
        assert spec.tags == ("payroll",), key
        assert spec.prompt == " ".join(slices), key


@pytest.mark.parametrize("key", list(FPS_PART_SLICE_COUNTS))
async def test_detailed_review_fps_parts_survive_validation(key, tmp_path, monkeypatch):
    """Every part must pass Tier 1 on its own and never degrade to the whole-prompt
    fallback blob — the guarantee that makes the parts separately runnable."""
    from automation.pipeline import decompose

    monkeypatch.setattr(ss, "LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(ss, "LIBRARY_MANIFEST", tmp_path / "library" / "manifest.json")
    monkeypatch.setattr(ss, "DECOMPOSITIONS_DIR", tmp_path / "decompositions")
    spec = load_tasks()[key]

    subs = await decompose.get_decomposition(spec.prompt, llm=None, spec=spec)
    assert len(subs) == FPS_PART_SLICE_COUNTS[key]
    assert not any(getattr(s, "fallback", False) for s in subs)
    assert [s.kind for s in subs] == FPS_PART_SLICE_KINDS[key]


def test_prompt_derived_from_declared_subtasks(tmp_path):
    """A task with `subtasks:` needs no `prompt:` — the spec's prompt is the single-space
    join of the instantiated slices (tokens substituted). One edit surface: the user
    rewords a slice, never a parallel prompt copy."""
    p = tmp_path / "t.yaml"
    p.write_text(
        "derived:\n"
        "  subtasks:\n"
        "    - prompt: 'open the {{page}} page.'\n"
        "      values: {page: settings}\n"
        "    - prompt: 'click save'\n"
    )
    spec = load_tasks(p)["derived"]
    assert spec.prompt == "open the settings page. click save"
    assert spec.subtasks and len(spec.subtasks) == 2


def test_prompt_and_subtasks_disagreement_fails_loud(tmp_path):
    """Keeping BOTH a prompt and subtasks is allowed only while they agree — a mismatch
    is silent drift waiting to poison identities, so the registry load refuses it."""
    p = tmp_path / "t.yaml"
    p.write_text(
        "drifted:\n"
        "  prompt: open the settings page. click SUBMIT\n"
        "  subtasks:\n"
        "    - prompt: 'open the settings page.'\n"
        "    - prompt: 'click save'\n"
    )
    with pytest.raises(ValueError, match="disagree"):
        load_tasks(p)


# ------------------------------- declared verify checks -------------------------------


def test_subtask_verify_parsed_and_substituted():
    """A verify: block parses into Check tuples at load time, with {{tokens}}
    substituted from the slice's own values (load-time, because the downstream
    token grammar is lowercase-only)."""
    from automation.pipeline.checks import Check

    entry = {"subtasks": [
        {"prompt": "Add {{name}} then save.",
         "values": {"name": "Alistair Allan"},
         "verify": [{"text_visible": "{{name}}"},
                    {"write_accepted": "Employees", "timeout_s": 5}]},
    ]}
    spec = tasks_mod._spec_from_entry("t", entry)
    assert spec.subtasks[0].verify == (
        Check(kind="text_visible", arg="Alistair Allan"),
        Check(kind="write_accepted", arg="Employees", timeout_s=5.0),
    )


def test_subtask_without_verify_is_none():
    entry = {"subtasks": [{"prompt": "Open payroll."}]}
    spec = tasks_mod._spec_from_entry("t", entry)
    assert spec.subtasks[0].verify is None


def test_subtask_verify_invalid_fails_loud():
    entry = {"subtasks": [{"prompt": "Open payroll.",
                           "verify": [{"nope": "x"}]}]}
    with pytest.raises(ValueError, match="nope"):
        tasks_mod._spec_from_entry("t", entry)


def test_subtask_verify_unresolved_token_fails_loud():
    """A verify arg has no downstream closure validation, so a leftover {{token}}
    must fail at load rather than probe for the literal braces at runtime."""
    entry = {"subtasks": [{"prompt": "Open payroll.",
                           "verify": [{"text_visible": "{{missing}}"}]}]}
    with pytest.raises(ValueError, match="missing"):
        tasks_mod._spec_from_entry("t", entry)


def test_subtask_verify_does_not_change_identity():
    """Ids derive from slice WORDING only — adding verify: must never drift a
    task_id (that would orphan decomposition caches and FROZEN_TIDS)."""
    base = {"subtasks": [{"prompt": "Open payroll."}]}
    with_verify = {"subtasks": [{"prompt": "Open payroll.",
                                 "verify": [{"url_contains": "payroll"}]}]}
    a = tasks_mod._spec_from_entry("t", base)
    b = tasks_mod._spec_from_entry("t", with_verify)
    assert a.prompt == b.prompt
    assert ss.task_id(a.prompt) == ss.task_id(b.prompt)


# ------------------------------- declared conditional probe -------------------------------


def test_subtask_probe_parsed_and_substituted():
    """A probe: mapping parses into ONE Check at load time (short default poll — an
    absent popup is a routine outcome, not a failure to wait out), with {{tokens}}
    substituted from the slice's values like verify args."""
    from automation.pipeline.checks import _PROBE_TIMEOUT_S, Check

    entry = {"subtasks": [
        {"prompt": "If a {{thing}} popup appears, dismiss it.",
         "values": {"thing": "Process"},
         "probe": {"text_visible": "Don't show this {{thing}} again"}},
    ]}
    spec = tasks_mod._spec_from_entry("t", entry)
    assert spec.subtasks[0].probe == Check(
        kind="text_visible", arg="Don't show this Process again",
        timeout_s=_PROBE_TIMEOUT_S)


def test_subtask_without_probe_is_none():
    entry = {"subtasks": [{"prompt": "Open payroll."}]}
    assert tasks_mod._spec_from_entry("t", entry).subtasks[0].probe is None


def test_subtask_probe_list_form_fails_loud():
    entry = {"subtasks": [{"prompt": "If a popup appears, dismiss it.",
                           "probe": [{"text_visible": "x"}]}]}
    with pytest.raises(ValueError, match="single check mapping"):
        tasks_mod._spec_from_entry("t", entry)


def test_subtask_probe_unresolved_token_fails_loud():
    entry = {"subtasks": [{"prompt": "If a popup appears, dismiss it.",
                           "probe": {"text_visible": "{{missing}}"}}]}
    with pytest.raises(ValueError, match="missing"):
        tasks_mod._spec_from_entry("t", entry)


def test_subtask_probe_does_not_change_identity():
    base = {"subtasks": [{"prompt": "If a popup appears, dismiss it."}]}
    with_probe = {"subtasks": [{"prompt": "If a popup appears, dismiss it.",
                                "probe": {"text_visible": "popup"}}]}
    a = tasks_mod._spec_from_entry("t", base)
    b = tasks_mod._spec_from_entry("t", with_probe)
    assert ss.task_id(a.prompt) == ss.task_id(b.prompt)


def test_subtask_allow_write_refusal_parses_and_rejects_non_bool(tmp_path):
    """A slice that declares its own error branch waives the window write rule; the
    declaration is author-written data, so a typo must be loud (same contract as kind)."""
    p = tmp_path / "t.yaml"
    p.write_text(
        "fps_task:\n"
        "  subtasks:\n"
        "    - prompt: 'click Submit; if it shows an error, click cancel'\n"
        "      allow_write_refusal: true\n"
        "    - prompt: 'then go to the next page'\n"
    )
    subs = load_tasks(p)["fps_task"].subtasks
    assert subs[0].allow_write_refusal is True
    assert subs[1].allow_write_refusal is False

    p.write_text(
        "fps_task:\n"
        "  subtasks:\n"
        "    - prompt: 'click Submit'\n"
        "      allow_write_refusal: yes-please\n"
    )
    with pytest.raises(ValueError, match="allow_write_refusal must be true or false"):
        load_tasks(p)
