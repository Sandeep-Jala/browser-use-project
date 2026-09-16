"""`prompts/` — the UI-owned half of the task registry.

`tasks.yaml` is a DOCUMENT, not a database: its ~1576 lines carry hand-written `#` comments
recording why each marker and each slice is worded the way it is (several of them record a marker
corrected after watching real traffic). Any YAML writer round-tripping that file destroys all of
it, and `ruamel` round-trip mode would be a new dependency just to preserve comments the UI has no
business touching. So the Auto Agent UI never writes `tasks.yaml`. It owns `prompts/`, one file per
prompt, in the EXACT same schema — which means `_spec_from_entry` stays the only validator, and
Tier-1 decomposition, the check parser and subtask identity all work on a UI prompt unchanged.

Two policies this file pins, both about blast radius:

1. **The merge only happens for the default registry.** `load_tasks(path)` with an explicit path is
   how the tests and any other caller ask for "exactly this file" — merging a real directory into
   that would make every explicit-path test depend on the working tree.
2. **A bad prompt file is skipped, not fatal.** `load_tasks()` feeds the CLI and the whole test
   suite; a hard raise would let one file in a UI-managed directory break `automation --task
   invoice` and collect-time for 1100 tests. A collision with a curated `tasks.yaml` key therefore
   lets tasks.yaml win and logs loudly, so the failure stays local and legible (`resolve_task`
   raises "Unknown TASK key" for that one key) rather than global. Two PROMPT files colliding is
   still a hard error: both are UI-owned and the UI can fix it.
"""
import logging

import pytest

from automation import tasks as tasks_mod
from automation.tasks import load_tasks


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    """An isolated registry: a tasks.yaml and a prompts/ dir, neither of them the real ones."""
    tasks_file = tmp_path / "tasks.yaml"
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    monkeypatch.setattr(tasks_mod, "TASKS_FILE", tasks_file)
    monkeypatch.setattr(tasks_mod, "PROMPTS_DIR", prompts)
    return tasks_file, prompts


# ── the merge ─────────────────────────────────────────────────────────────────────────────

def test_a_prompt_file_joins_the_registry(registry):
    tasks_file, prompts = registry
    tasks_file.write_text("curated:\n  prompt: do the curated thing\n")
    (prompts / "mine.yaml").write_text("mine:\n  prompt: do my thing\n")

    reg = load_tasks()

    assert set(reg) == {"curated", "mine"}
    assert reg["mine"].prompt == "do my thing"


def test_a_prompt_file_keeps_the_full_schema(registry):
    """The whole point of reusing the schema: a UI prompt is a first-class task, probes and all."""
    _, prompts = registry
    (prompts / "mine.yaml").write_text(
        "mine:\n"
        "  marker: Invoices\n"
        "  tags: [ui]\n"
        "  subtasks:\n"
        "    - prompt: open the thing\n"
        "    - prompt: If a warning appears, dismiss it.\n"
        "      probe: {text_visible: 'warning'}\n"
        "      kind: action\n"
    )
    spec = load_tasks()["mine"]
    assert spec.marker == "Invoices" and spec.tags == ("ui",)
    assert len(spec.subtasks) == 2
    assert spec.subtasks[1].probe.kind == "text_visible"
    assert spec.subtasks[1].probe.arg == "warning"


def test_a_missing_prompts_dir_is_a_no_op(registry, monkeypatch):
    tasks_file, prompts = registry
    tasks_file.write_text("curated:\n  prompt: do the curated thing\n")
    monkeypatch.setattr(tasks_mod, "PROMPTS_DIR", prompts.parent / "nope")
    assert set(load_tasks()) == {"curated"}


def test_prompt_files_alone_are_a_registry(registry):
    """tasks.yaml missing is an empty registry, not an error — that must still hold with prompts."""
    _, prompts = registry
    (prompts / "mine.yaml").write_text("mine:\n  prompt: do my thing\n")
    assert set(load_tasks()) == {"mine"}


def test_keys_are_lowercased_like_tasks_yaml(registry):
    _, prompts = registry
    (prompts / "MyPrompt.yaml").write_text("MyPrompt:\n  prompt: do my thing\n")
    assert "myprompt" in load_tasks()


# ── only for the default registry ─────────────────────────────────────────────────────────

def test_an_explicit_path_does_not_merge_prompts(registry):
    """`load_tasks(p)` means "exactly this file". The 10 explicit-path callers in the suite must
    not start depending on whatever is in the working tree's prompts/."""
    tasks_file, prompts = registry
    tasks_file.write_text("curated:\n  prompt: do the curated thing\n")
    (prompts / "mine.yaml").write_text("mine:\n  prompt: do my thing\n")

    assert set(load_tasks(tasks_file)) == {"curated"}


# ── the file stem is the key ───────────────────────────────────────────────────────────────

def test_a_stem_that_disagrees_with_its_key_is_rejected(registry, caplog):
    """`prompts/<key>.yaml` addressing is what lets the UI edit and delete by key. A file whose
    stem and key disagree would be unreachable, so it is skipped and named."""
    _, prompts = registry
    (prompts / "mine.yaml").write_text("something_else:\n  prompt: do my thing\n")
    with caplog.at_level(logging.ERROR, logger="framework.tasks"):
        assert load_tasks() == {}
    assert "mine.yaml" in caplog.text and "something_else" in caplog.text


def test_a_prompt_file_holding_more_than_one_entry_is_rejected(registry, caplog):
    _, prompts = registry
    (prompts / "mine.yaml").write_text("mine:\n  prompt: a b\nother:\n  prompt: c d\n")
    with caplog.at_level(logging.ERROR, logger="framework.tasks"):
        assert load_tasks() == {}
    assert "mine.yaml" in caplog.text


# ── fail soft: one bad file must never break the CLI ──────────────────────────────────────

def test_a_malformed_prompt_file_is_skipped_not_fatal(registry, caplog):
    tasks_file, prompts = registry
    tasks_file.write_text("curated:\n  prompt: do the curated thing\n")
    (prompts / "broken.yaml").write_text("broken:\n  tags: [x]\n")   # no prompt, no subtasks
    (prompts / "fine.yaml").write_text("fine:\n  prompt: do my thing\n")

    with caplog.at_level(logging.ERROR, logger="framework.tasks"):
        reg = load_tasks()

    assert set(reg) == {"curated", "fine"}, "a broken sibling must not take the registry down"
    assert "broken.yaml" in caplog.text
    assert "non-empty 'prompt'" in caplog.text, "the loader's own message must survive"


def test_unreadable_yaml_in_a_prompt_file_is_skipped_not_fatal(registry, caplog):
    tasks_file, prompts = registry
    tasks_file.write_text("curated:\n  prompt: do the curated thing\n")
    (prompts / "bad.yaml").write_text("mine: [unclosed\n")

    with caplog.at_level(logging.ERROR, logger="framework.tasks"):
        assert set(load_tasks()) == {"curated"}
    assert "bad.yaml" in caplog.text


def test_a_collision_with_a_curated_key_lets_tasks_yaml_win(registry, caplog):
    """A UI file must never be able to shadow a curated task — that would silently redirect
    `automation --task invoice` to whatever someone saved in the browser."""
    tasks_file, prompts = registry
    tasks_file.write_text("invoice:\n  prompt: the curated invoice flow\n")
    (prompts / "invoice.yaml").write_text("invoice:\n  prompt: my impostor\n")

    with caplog.at_level(logging.ERROR, logger="framework.tasks"):
        reg = load_tasks()

    assert reg["invoice"].prompt == "the curated invoice flow"
    assert "invoice.yaml" in caplog.text and "tasks.yaml" in caplog.text


def test_two_prompt_files_colliding_is_a_hard_error(registry):
    """Both are UI-owned, so this is unambiguous and the UI can fix it. Unlike the cases above
    there is no correct winner to fall back to.

    The collision is reached via a stem that differs only by surrounding whitespace, because a
    case variant would not reach it: the stem comparison lowercases, and on macOS `MINE.yaml`
    and `mine.yaml` are the SAME FILE, so the second write just overwrites the first. A
    hand-placed `mine .yaml` is a real way to end up here — the UI would then save its edits to
    `mine.yaml` and both files would claim the key.
    """
    _, prompts = registry
    (prompts / "mine.yaml").write_text("mine:\n  prompt: first\n")
    (prompts / "mine .yaml").write_text("mine:\n  prompt: second\n")

    with pytest.raises(ValueError, match="two prompt files claim the task key 'mine'"):
        load_tasks()


# ── housekeeping files are not prompts ────────────────────────────────────────────────────

def test_dotfiles_and_trash_are_ignored(registry):
    """`delete` trashes into prompts/.trash/ rather than unlinking, and an editor may leave
    dotfiles. Neither is a prompt."""
    _, prompts = registry
    (prompts / ".hidden.yaml").write_text("hidden:\n  prompt: do not load me\n")
    (prompts / ".trash").mkdir()
    (prompts / ".trash" / "old.yaml").write_text("old:\n  prompt: do not load me either\n")
    (prompts / "notes.txt").write_text("not yaml at all")
    (prompts / "mine.yaml").write_text("mine:\n  prompt: do my thing\n")

    assert set(load_tasks()) == {"mine"}


# ── error messages must name the real file ────────────────────────────────────────────────

def test_a_prompt_files_errors_name_that_file_not_tasks_yaml(registry, caplog):
    """Every loader message is hardcoded "tasks.yaml entry …". Pointing a user at the wrong file
    while they are editing in a browser is its own bug."""
    _, prompts = registry
    (prompts / "mine.yaml").write_text(
        "mine:\n"
        "  subtasks:\n"
        "    - prompt: do a thing\n"
        "      kind: sideways\n"
    )
    with caplog.at_level(logging.ERROR, logger="framework.tasks"):
        load_tasks()
    assert "prompts/mine.yaml" in caplog.text or "mine.yaml" in caplog.text
    assert "tasks.yaml entry" not in caplog.text, "the message still blames tasks.yaml"
    assert "kind must be 'action' or 'judge'" in caplog.text
