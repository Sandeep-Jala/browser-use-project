"""The editor's data model and the `prompts/` store.

`model.py` is the ONLY place the UI's wire shape becomes YAML, so it is the only place a prompt's
identity can be corrupted. Identity here is not a metaphor: `subtask_id` is a sha256 of the
normalized prompt text and `task_id` a sha256 of the whole joined prompt, so a round-trip that
changes one character of wording silently orphans a `library/` recording and forces the LLM to
re-author that step at full token cost. Hence the strongest test in this file
(`test_real_tasks_yaml_entries_survive_a_round_trip_with_their_identity`): take real entries out of
`tasks.yaml`, push them through the editor's model and back, and demand the resulting `TaskSpec`
be *identical*, `task_id` included.

Two traps this file pins, both verified against the source rather than assumed:

1. **Unknown keys must survive.** `_spec_from_entry` reads only the keys it knows and silently
   ignores the rest — there is no strict-key check. So a naive `to_entry()` that emits only known
   fields would quietly delete `assertions:` and `postcondition:` from an imported entry.
2. **Token names must be lowercase.** `tasks._DECL_TOKEN` is `\\{\\{(\\w+)\\}\\}` and substitutes
   `{{Business}}` happily, but `subtask_store.TOKEN_RE` is `\\{\\{([a-z][a-z0-9_]*)\\}\\}` and will
   NOT erase it in `normalize_template`. So an uppercase token survives normalization literally,
   baking its own name into the step's identity and forking it from the otherwise-identical
   lowercase version. The editor has to refuse it, because nothing downstream will.
"""
from __future__ import annotations

import pytest
import yaml

from automation import tasks as tasks_mod
from automation.pipeline.subtask_store import task_id
from automation.tasks import _spec_from_entry, load_tasks
from automation.ui import store
from automation.ui.model import PromptModel, StepModel, from_entry, to_entry


@pytest.fixture()
def prompts_dir(tmp_path, monkeypatch):
    d = tmp_path / "prompts"
    d.mkdir()
    # One setattr: store._dir() resolves through tasks.PROMPTS_DIR rather than keeping a copy.
    monkeypatch.setattr(tasks_mod, "PROMPTS_DIR", d)
    return d


def _raw_entry(key: str) -> dict:
    """The VERBATIM tasks.yaml mapping for a key (not the parsed TaskSpec)."""
    data = yaml.safe_load(tasks_mod.TASKS_FILE.read_text(encoding="utf-8"))
    for k, entry in data.items():
        if str(k).strip().lower() == key:
            return entry
    raise KeyError(key)


# ── the identity round trip ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("key", [
    "payroll_food_limited_e2e_rti",      # probes + allow_write_refusal, 19 slices
    "payroll_client_2",                  # values / {{tokens}}
    "payroll_detailed_review_fps_part1",  # declared kind
    "employee_addition",                 # plain slices, no declarations
])
def test_real_tasks_yaml_entries_survive_a_round_trip_with_their_identity(key):
    """Proof the editor cannot corrupt a prompt. If this fails, importing a curated task into the
    UI and saving it would re-hash its steps and throw away their recordings."""
    original = _raw_entry(key)
    before = _spec_from_entry(key, original)

    after = _spec_from_entry(key, to_entry(from_entry(key, original)))

    assert after == before, "the round-tripped spec differs from the original"
    assert task_id(after.prompt) == task_id(before.prompt), "the task identity moved"


def test_a_verify_block_survives_a_round_trip():
    """Constructed rather than taken from tasks.yaml: `verify:` is supported by the loader but
    used by no entry in the file today, so the UI is its first real consumer."""
    entry = {"subtasks": [
        {"prompt": "Add the employee and save.",
         "verify": [{"text_visible": "Employees"},
                    {"write_accepted": "Employees", "timeout_s": 5}]},
    ]}
    before = _spec_from_entry("k", entry)
    after = _spec_from_entry("k", to_entry(from_entry("k", entry)))
    assert after == before
    assert len(after.subtasks[0].verify) == 2
    assert after.subtasks[0].verify[1].timeout_s == 5


def test_unknown_keys_survive_a_round_trip():
    """`_spec_from_entry` ignores unknown keys silently, so only the model can protect them."""
    entry = {
        "assertions": {"no_5xx": False},
        "tags": ["x"],
        "subtasks": [{"prompt": "do a thing", "postcondition": {"url_contains": "/books"}}],
    }
    out = to_entry(from_entry("k", entry))
    assert out["assertions"] == {"no_5xx": False}
    assert out["subtasks"][0]["postcondition"] == {"url_contains": "/books"}


# ── what the editor must never emit ───────────────────────────────────────────────────────

def test_step_mode_never_writes_prompt_alongside_subtasks():
    """Writing both is what makes `_spec_from_entry` raise its word-index divergence error. The
    editor makes that unreachable by construction rather than trying to keep them in sync."""
    out = to_entry(PromptModel(key="k", steps=[StepModel(prompt="do a thing")]))
    assert "subtasks" in out and "prompt" not in out


def test_one_instruction_mode_never_writes_subtasks():
    out = to_entry(PromptModel(key="k", prompt="do the whole thing in one go"))
    assert out["prompt"] == "do the whole thing in one go"
    assert "subtasks" not in out


def test_empty_declarations_are_omitted_not_written_as_nulls():
    """A `kind: null` or `probe: {}` would either fail the loader or change nothing while making
    the file unreadable. Absent means absent."""
    out = to_entry(PromptModel(key="k", steps=[StepModel(prompt="do a thing")]))
    step = out["subtasks"][0]
    assert set(step) == {"prompt"}, f"unexpected keys written: {sorted(set(step) - {'prompt'})}"
    assert "marker" not in out and "tags" not in out


def test_the_three_step_types_map_onto_the_two_schema_axes():
    """`kind` and `probe` are independent in the engine (hybrid computes is_judge and
    is_conditional separately); the UI's single three-way picker must collapse onto them exactly."""
    action = to_entry(PromptModel(key="k", steps=[
        StepModel(prompt="click save", step_type="action")]))["subtasks"][0]
    assert "kind" not in action and "probe" not in action      # action is the default

    check = to_entry(PromptModel(key="k", steps=[
        StepModel(prompt="confirm the total is right", step_type="check")]))["subtasks"][0]
    assert check["kind"] == "judge" and "probe" not in check

    cond = to_entry(PromptModel(key="k", steps=[
        StepModel(prompt="If a warning appears, dismiss it.", step_type="conditional",
                  probe_kind="text_visible", probe_text="warning")]))["subtasks"][0]
    assert cond["probe"] == {"text_visible": "warning"} and "kind" not in cond


def test_a_conditional_probe_timeout_is_only_written_when_it_differs():
    """The default comes from checks._PROBE_TIMEOUT_S; writing it explicitly would freeze today's
    value into every file and drift the moment the engine's default changes."""
    from automation.pipeline.checks import _PROBE_TIMEOUT_S
    default = to_entry(PromptModel(key="k", steps=[
        StepModel(prompt="If x, do y.", step_type="conditional", probe_kind="text_visible",
                  probe_text="x", probe_timeout_s=_PROBE_TIMEOUT_S)]))["subtasks"][0]
    assert default["probe"] == {"text_visible": "x"}

    custom = to_entry(PromptModel(key="k", steps=[
        StepModel(prompt="If x, do y.", step_type="conditional", probe_kind="text_visible",
                  probe_text="x", probe_timeout_s=9)]))["subtasks"][0]
    assert custom["probe"] == {"text_visible": "x", "timeout_s": 9}


# ── validation the engine cannot do for us ────────────────────────────────────────────────

def test_an_uppercase_token_is_refused_with_its_reason():
    """It would substitute correctly and then silently fork the step's identity."""
    model = PromptModel(key="k", steps=[
        StepModel(prompt="open {{Business}} and save", values={"Business": "CREAMOS LTD"})])
    errors = store.validate_model(model)
    assert any("lowercase" in e.message.lower() for e in errors), errors


def test_a_lowercase_token_is_accepted():
    model = PromptModel(key="k", steps=[
        StepModel(prompt="open {{business}} and save", values={"business": "CREAMOS LTD"})])
    assert store.validate_model(model) == []


def test_a_step_with_no_text_is_refused():
    model = PromptModel(key="k", steps=[StepModel(prompt="   ")])
    assert any(e.step == 0 for e in store.validate_model(model))


def test_a_prompt_with_no_steps_and_no_text_is_refused():
    assert store.validate_model(PromptModel(key="k")) != []


def test_loader_errors_are_attributed_to_the_step_that_raised():
    """The loader's messages embed `subtask {i}`; the editor shows them on that step's card."""
    model = PromptModel(key="k", steps=[
        StepModel(prompt="fine"),
        StepModel(prompt="bad helper", tab_url="not-a-url"),
    ])
    errors = store.validate_model(model)
    assert any(e.step == 1 and "tab_url" in e.message for e in errors), errors


# ── the store ─────────────────────────────────────────────────────────────────────────────

def test_write_then_read_round_trips_through_the_registry(prompts_dir):
    model = PromptModel(key="my_prompt", marker="Invoices", tags=("ui",), steps=[
        StepModel(prompt="Go to the Bookkeeping module."),
        StepModel(prompt="If a warning appears, dismiss it.", step_type="conditional",
                  probe_kind="text_visible", probe_text="warning"),
    ])
    store.write_prompt(model)

    assert (prompts_dir / "my_prompt.yaml").is_file()
    spec = load_tasks()["my_prompt"]
    assert spec.marker == "Invoices"
    assert spec.subtasks[1].probe.kind == "text_visible"
    assert store.read_prompt("my_prompt").steps[1].probe_text == "warning"


def test_a_write_leaves_no_temp_file_behind(prompts_dir):
    store.write_prompt(PromptModel(key="p", steps=[StepModel(prompt="do a thing")]))
    assert [f.name for f in prompts_dir.iterdir()] == ["p.yaml"]


def test_delete_trashes_rather_than_unlinks(prompts_dir):
    store.write_prompt(PromptModel(key="p", steps=[StepModel(prompt="do a thing")]))
    trashed = store.delete_prompt("p")
    assert not (prompts_dir / "p.yaml").exists()
    assert trashed.is_file() and trashed.parent.name == ".trash"
    assert "p" not in load_tasks(), "a trashed prompt must leave the registry"


def test_listing_reports_a_broken_file_instead_of_hiding_it(prompts_dir):
    store.write_prompt(PromptModel(key="good", steps=[StepModel(prompt="do a thing")]))
    (prompts_dir / "broken.yaml").write_text("broken:\n  tags: [x]\n")

    rows, skipped = store.list_prompts()
    assert [r.key for r in rows] == ["good"]
    assert len(skipped) == 1 and skipped[0].key == "broken"
    assert "non-empty 'prompt'" in skipped[0].error


def test_slugify_produces_a_usable_key():
    assert store.slugify("Payroll — New Employee!") == "payroll_new_employee"
    assert store.slugify("  multiple   spaces  ") == "multiple_spaces"
    assert store.slugify("123 start") == "123_start"


def test_slugify_refuses_to_invent_an_empty_key():
    with pytest.raises(ValueError):
        store.slugify("!!!")


def test_importing_a_curated_task_copies_it_verbatim(prompts_dir):
    """Importing must not reword anything — that is the whole point of the copy affordance."""
    before = _spec_from_entry("payroll_client_2", _raw_entry("payroll_client_2"))
    store.import_task("payroll_client_2", "my_copy")

    after = load_tasks()["my_copy"]
    assert after.prompt == before.prompt, "the imported copy's wording moved"
    assert task_id(after.prompt) == task_id(before.prompt)
    assert len(after.subtasks) == len(before.subtasks)
