"""The step editor: its form dict, its mutators, and the identity that must survive them.

`@gr.render` needs a plain structure it can diff, and every step and check needs a STABLE id so a
reorder moves a component's key with it rather than by position. `PromptModel` has neither, so the
form dict is a THIRD representation sitting between the widgets and `to_entry`:

    widgets ⇄ form dict ⇄ PromptModel ⇄ tasks.yaml entry

`test_ui_store.py` already pins the right-hand crossing. This file pins the new left-hand one, and
for the same reason: `subtask_id` is a sha256 of the normalized step text and `task_id` a sha256 of
the whole joined prompt, so a round-trip that alters one character silently orphans the `library/`
recording that step would otherwise have replayed for free — at full re-authoring token cost, with
no error anywhere. Before this UI existed the browser owned that boundary and nothing tested it.

The mutators are pure functions on purpose. The render calls them after harvesting the live widget
values, so they never need to know a thing about Gradio, and neither does this file.
"""
from __future__ import annotations

import pytest
import yaml

from automation import tasks as tasks_mod
from automation.pipeline.subtask_store import subtask_id, task_id
from automation.tasks import _spec_from_entry
from automation.ui import ops
from automation.ui.model import PromptModel, StepModel, from_entry, to_entry


# ── the form-dict round trip ──────────────────────────────────────────────────────────────

def _round_trip(model: PromptModel) -> PromptModel:
    return ops.model_from_form(ops.model_to_form(model))


def test_a_plain_model_survives_the_form_dict():
    model = PromptModel(key="k", marker="Invoices", tags=("sales", "ui"),
                        steps=[StepModel(prompt="go to Bookkeeping"),
                               StepModel(prompt="add an invoice")])
    back = _round_trip(model)
    assert back.key == model.key
    assert back.marker == model.marker
    assert back.tags == model.tags
    assert [s.prompt for s in back.steps] == [s.prompt for s in model.steps]


def test_every_step_field_survives_the_form_dict():
    """Anything the form drops is a declaration the user made and the run will not honour."""
    from automation.ui.model import CheckModel
    model = PromptModel(key="k", steps=[StepModel(
        prompt="open the panel",
        step_type="conditional", probe_kind="text_visible", probe_text="warning",
        probe_timeout_s=3.0,
        verify=[CheckModel(kind="url_contains", text="/Invoices", timeout_s=9.0)],
        marker="Invoices", tab_url="https://example.test/helper",
        allow_write_refusal=True, values={"name": "Bob"},
        declared_kind=None, extra={"postcondition": {"x": 1}})])
    s = _round_trip(model).steps[0]
    assert s.step_type == "conditional"
    assert (s.probe_kind, s.probe_text, s.probe_timeout_s) == ("text_visible", "warning", 3.0)
    assert (s.verify[0].kind, s.verify[0].text, s.verify[0].timeout_s) == \
        ("url_contains", "/Invoices", 9.0)
    assert s.marker == "Invoices"
    assert s.tab_url == "https://example.test/helper"
    assert s.allow_write_refusal is True
    assert s.values == {"name": "Bob"}
    assert s.extra == {"postcondition": {"x": 1}}, "unknown keys must round-trip untouched"


def test_one_instruction_mode_survives():
    model = PromptModel(key="k", prompt="do the whole thing in one go")
    back = _round_trip(model)
    assert back.prompt == model.prompt
    assert back.steps == []


def test_every_step_and_check_gets_a_unique_id():
    """Stable ids are what let a reorder carry a component's key with it rather than leaving the
    value behind on whatever now occupies that position."""
    from automation.ui.model import CheckModel
    form = ops.model_to_form(PromptModel(key="k", steps=[
        StepModel(prompt="a", verify=[CheckModel(text="x"), CheckModel(text="y")]),
        StepModel(prompt="b", verify=[CheckModel(text="z")])]))
    uids = [s["uid"] for s in form["steps"]]
    uids += [c["uid"] for s in form["steps"] for c in s["verify"]]
    assert len(uids) == len(set(uids)) == 5


@pytest.mark.parametrize("key", ["invoice", "payroll_client", "banking"])
def test_real_tasks_yaml_entries_survive_the_form_dict_with_their_identity(key):
    """The strongest test here, and the mirror of test_ui_store.py's own.

    Take a real entry out of tasks.yaml, push it all the way through the editor's new boundary —
    entry → model → FORM DICT → model → entry — and demand the resulting TaskSpec keep its
    IDENTITY. Those hashes are what decide whether a run replays a recording or pays an LLM to
    author the step again.

    Identity, not byte-equality, is the right assertion, and the difference is load-bearing. The
    form dict strips textarea whitespace at the edge (`ops.model_from_form`), and `tasks.yaml` has
    at least one slice whose text ends in trailing spaces — `payroll_client`'s CREAMOS LTD step.
    That step's raw `prompt` therefore does change. Its identity does not: `normalize_template`
    collapses whitespace before `subtask_id` hashes it, so both ids come out the same (verified
    directly — `subtask_id(t, "") == subtask_id(t + "   ", "")`). Asserting byte-equality here
    would fail on a difference that cannot orphan a recording, and would pressure someone into
    removing the strip that stops a browser's trailing newline reaching a step.
    """
    data = yaml.safe_load(tasks_mod.TASKS_FILE.read_text(encoding="utf-8"))
    entry = next((e for k, e in data.items() if str(k).strip().lower() == key), None)
    if entry is None:
        pytest.skip(f"{key} is not in tasks.yaml")

    before = _spec_from_entry(key, entry)
    after = _spec_from_entry(key, to_entry(_round_trip(from_entry(key, entry))))

    assert after.prompt == before.prompt
    assert task_id(after.prompt) == task_id(before.prompt)
    assert after.marker == before.marker
    assert after.tags == before.tags

    assert len(after.subtasks or ()) == len(before.subtasks or ())
    for got, want in zip(after.subtasks or (), before.subtasks or ()):
        assert subtask_id(got.prompt, "") == subtask_id(want.prompt, ""), got.prompt
        assert got.prompt.strip() == want.prompt.strip()
        # Every declaration the author made, preserved exactly — these have no normalization
        # to hide behind, so a drop here is a behaviour change.
        for field in ("values", "marker", "postcondition", "kind", "tab_url", "verify", "probe",
                      "allow_write_refusal"):
            assert getattr(got, field) == getattr(want, field), f"{field} on {want.prompt[:40]!r}"


# ── the mutators ──────────────────────────────────────────────────────────────────────────

def _form(*prompts: str) -> dict:
    return ops.model_to_form(PromptModel(key="k", steps=[StepModel(prompt=p) for p in prompts]))


def _prompts(form: dict) -> list[str]:
    return [s["prompt"] for s in form["steps"]]


def test_add_step_appends_a_blank_one():
    form = ops.add_step(_form("a"))
    assert _prompts(form) == ["a", ""]


def test_remove_step():
    assert _prompts(ops.remove_step(_form("a", "b", "c"), 1)) == ["a", "c"]


def test_removing_a_step_that_is_not_there_changes_nothing():
    """A stale index arrives whenever a click lands against a list that has already moved."""
    assert _prompts(ops.remove_step(_form("a", "b"), 7)) == ["a", "b"]


def test_move_step():
    assert _prompts(ops.move_step(_form("a", "b", "c"), 1, -1)) == ["b", "a", "c"]
    assert _prompts(ops.move_step(_form("a", "b", "c"), 1, +1)) == ["a", "c", "b"]


def test_moving_off_either_end_is_a_no_op():
    assert _prompts(ops.move_step(_form("a", "b"), 0, -1)) == ["a", "b"]
    assert _prompts(ops.move_step(_form("a", "b"), 1, +1)) == ["a", "b"]


def test_a_moved_step_keeps_its_uid():
    """If a reorder minted new ids, every keyed component would be rebuilt and the browser would
    lose the caret mid-edit."""
    form = _form("a", "b")
    before = [s["uid"] for s in form["steps"]]
    after = [s["uid"] for s in ops.move_step(form, 0, +1)["steps"]]
    assert after == [before[1], before[0]]


def test_add_and_remove_check():
    form = ops.add_check(_form("a"), 0)
    assert len(form["steps"][0]["verify"]) == 1
    form = ops.add_check(form, 0)
    assert len(form["steps"][0]["verify"]) == 2
    form = ops.remove_check(form, 0, 0)
    assert len(form["steps"][0]["verify"]) == 1


def test_check_mutators_ignore_an_index_that_is_gone():
    form = _form("a")
    assert ops.remove_check(form, 0, 3)["steps"][0]["verify"] == []
    assert ops.add_check(form, 9)["steps"][0]["verify"] == []


def test_a_blank_step_is_a_valid_action_step():
    """The blank the Add button inserts must be something `model_from_form` can read without a
    KeyError, or the first keystroke after adding a step would raise."""
    model = ops.model_from_form({"key": "k", "steps": [ops.blank_step()]})
    assert model.steps[0].step_type == "action"
    assert model.steps[0].prompt == ""


# ── the editor's widgets must be explicitly interactive ───────────────────────────────────

def test_every_editor_widget_declares_interactive():
    """Guards a bug that no behavioural test can see.

    Gradio infers editability from `interactive=None` by asking whether a component is an input to
    some event at config time. For components built inside `@gr.render`, that inference resolves to
    DISABLED once the block re-renders: the first render looks right, and then loading a prompt
    turns the whole editor read-only — radios, textareas, everything.

    It is invisible from the server side. The wiring is correct, the handlers fire, and a
    programmatic write (Selenium/Playwright `.value =`, Gradio's own API) succeeds because setting
    a value ignores the `disabled` attribute. Only a human clicking finds it, which is exactly how
    it was found — after a browser check that used a programmatic write and therefore passed.

    So this is a source-level assertion rather than a behavioural one: every widget the editor
    binds into `comps` (i.e. every widget a user must be able to change) has to say
    `interactive=True` out loud.
    """
    import ast
    import pathlib

    src = pathlib.Path("automation/ui/gradio_app.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    editor = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "editor"), None)
    assert editor is not None, "the @gr.render editor function has been renamed — update this test"

    offenders = []
    for node in ast.walk(editor):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        # only the widgets bound into comps[...] — those are the user-editable ones
        target = node.targets[0]
        if not (isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name) and target.value.id == "comps"):
            continue
        call = node.value
        name = getattr(call.func, "attr", "?")
        if not any(kw.arg == "interactive" for kw in call.keywords):
            offenders.append(f"gr.{name} at line {node.lineno}")

    assert not offenders, (
        "these editor widgets rely on Gradio's interactive inference and will render DISABLED "
        "after a re-render — pass interactive=True explicitly:\n  " + "\n  ".join(offenders))
