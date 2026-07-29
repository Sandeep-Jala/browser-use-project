"""Transpiler + lint: committed steps/templates -> tier-1 code skills."""
import json

import pytest

from automation.pipeline import subtask_store as ss
from automation.skills.codegen import compile_code_skill, lint_code, transpile


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(ss, "LIBRARY_MANIFEST", tmp_path / "library" / "manifest.json")
    (tmp_path / "library").mkdir()
    return tmp_path


# A trimmed version of a real committed template (business search-and-select).
TEMPLATE_STEPS = [
    {"action": "click",
     "selectors": ['role=link[name="Clients"]', 'css=[id="clients"]'],
     "fingerprint": {"tag": "a", "text": "Clients", "attrs": {"id": "clients"}}},
    {"action": "fill",
     "selectors": ['css=[placeholder="Search"]'],
     "value": "{{business_name}}", "clear": True,
     "fingerprint": {"tag": "input", "text": "Search", "attrs": {"placeholder": "Search"}}},
    {"action": "press", "keys": "Enter"},
    {"action": "wait", "seconds": 2.0},
    {"action": "click",
     "selectors": ['role=link[name="{{business_name}}"]', 'text="{{business_name}}"']},
]


def test_transpile_parameterized_template():
    code, anchors = transpile("sid1", TEMPLATE_STEPS,
                              source_prompt="select a business Acme Ltd",
                              params={"business_name": "Acme Ltd"})
    assert "async def run(api, *, business_name='Acme Ltd'):" in code
    assert "await api.click('clients')" in code
    assert "await api.fill('search', business_name)" in code
    assert "await api.press('Enter')" in code
    assert "await api.wait(2.0)" in code
    assert "await api.click('business-name-target')" in code
    # Element identities live in the anchors, tokens preserved for load-time substitution.
    assert anchors["clients"]["selectors"] == ['role=link[name="Clients"]',
                                               'css=[id="clients"]']
    assert anchors["business-name-target"]["selectors"][0] == \
        'role=link[name="{{business_name}}"]'
    # The emitted artifact must pass its own safety gate.
    assert lint_code(code) == []


def test_upload_step_transpiles_and_parameterizes():
    from automation.skills.codegen import lint_code, transpile

    steps = [{"action": "upload", "selectors": ['css=[id="zone"]'],
              "fingerprint": {"tag": "div"}, "hidden_ok": True, "value": "{{file}}"}]
    code, anchors = transpile("s", steps, params={"file": "list.csv"})
    assert "await api.upload(" in code and ", file)" in code
    assert lint_code(code) == []
    (_handle, anchor), = anchors.items()
    assert anchor["hidden_ok"] is True

    # Unparameterized value stays a literal.
    code, _ = transpile("s", [{**steps[0], "value": "raw.csv"}])
    assert "'raw.csv')" in code


def test_select_pair_collapses_to_select_option():
    steps = [
        {"action": "type", "text": "{{customer}}"},
        {"action": "click", "selectors": ['role=option[name="{{customer}}"]',
                                          'css=[id$="-option-0"]']},
    ]
    code, anchors = transpile("sid2", steps, params={"customer": "Suresh Gopi"})
    assert "await api.select_option(customer)" in code
    assert anchors == {}          # a by-label pick needs no anchor
    assert lint_code(code) == []


def test_transpile_concrete_steps_zero_args():
    steps = [{"action": "goto", "url": "http://app/section"},
             {"action": "click", "selectors": ["text=Save"],
              "fingerprint": {"text": "Save", "attrs": {}}}]
    code, anchors = transpile("sid3", steps)
    assert "async def run(api):" in code
    assert "await api.goto('http://app/section')" in code
    assert "await api.click('save')" in code
    assert set(anchors) == {"save"}
    assert lint_code(code) == []


def test_embedded_token_becomes_fstring():
    steps = [{"action": "fill", "selectors": ["css=[name=ref]"],
              "value": "INV-{{n}}-x", "fingerprint": {"attrs": {"name": "ref"}}}]
    code, _ = transpile("sid4", steps, params={"n": "42"})
    assert "await api.fill('ref', f'INV-{n}-x')" in code
    assert lint_code(code) == []


def test_unknown_action_raises():
    with pytest.raises(ValueError, match="cannot transpile"):
        transpile("sid5", [{"action": "hover", "selectors": ["x"]}])
    with pytest.raises(ValueError, match="nothing to transpile"):
        transpile("sid6", [])


# ------------------------------- lint -------------------------------


def test_lint_rejects_escapes():
    assert lint_code("import os\n")                       # not even a run()
    assert any("Import" in p for p in lint_code(
        'async def run(api):\n    import os\n'))
    assert any("api.<verb>" in p for p in lint_code(
        'async def run(api):\n    print("hi")\n'))
    assert any("attribute access" in p for p in lint_code(
        'async def run(api):\n    x = api\n    await x.page.goto("evil")\n'))
    assert any("exactly one" in p for p in lint_code(
        'async def run(api):\n    await api.wait(1.0)\nx = 1\n'))
    assert any("positional argument" in p for p in lint_code(
        'async def run(api, extra):\n    await api.wait(1.0)\n'))


def test_lint_allows_hand_added_control_flow():
    code = (
        'async def run(api, *, qty="5"):\n'
        "    await api.click('add')\n"
        "    if qty != '0':\n"
        "        await api.fill('qty', qty)\n"
        "    await api.press('Enter')\n"
    )
    assert lint_code(code) == []


# ------------------------------- commit-time upgrade -------------------------------


def test_compile_code_skill_end_to_end(stores):
    sid = "abc"
    ss.steps_path(sid).write_text(json.dumps(
        [{"action": "goto", "url": "http://app/x"}]))
    assert compile_code_skill(sid) is not None
    assert ss.code_path(sid).exists() and ss.anchors_path(sid).exists()
    assert "await api.goto('http://app/x')" in ss.code_path(sid).read_text()
    # The code body subsumes the steps body: steps.json is deleted on success...
    assert not ss.steps_path(sid).exists()
    # ...and the entry still counts as present.
    assert ss.has_script(sid)


def test_compile_failure_removes_stale_artifacts(stores):
    sid = "abc"
    ss.steps_path(sid).write_text(json.dumps([{"action": "goto", "url": "http://app/x"}]))
    assert compile_code_skill(sid) is not None
    # The entry is re-committed with a step the transpiler can't express: the old code
    # must NOT survive to shadow the fresh steps.
    ss.steps_path(sid).write_text(json.dumps([{"action": "warp", "to": "mars"}]))
    assert compile_code_skill(sid) is None
    assert not ss.code_path(sid).exists() and not ss.anchors_path(sid).exists()
