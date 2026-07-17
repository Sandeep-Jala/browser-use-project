"""Compile coverage fixes surfaced by live NPS runs: scrolls must compile, a
metadata-less find_by_text click must not vanish, save_history's metadata drop must be
repaired, and a zero-step script must never be committed."""
import json
from types import SimpleNamespace

import pytest

from automation.pipeline import subtask_store as ss
from automation.pipeline.runner import restore_result_metadata
from automation.pipeline.script_compile import compile_recording
from automation.skills.codegen import lint_code, transpile


def _item(action, url="http://app/x", result=None):
    return {"state": {"url": url, "interacted_element": []},
            "model_output": {"action": [action]},
            "result": result if result is not None else []}


def _write(tmp_path, history):
    p = tmp_path / "rec.json"
    p.write_text(json.dumps({"history": history}))
    return p


def test_scrolls_compile(tmp_path):
    history = [
        _item({"capped_scroll": {"down": True, "pages": 0.2}}),
        _item({"scroll": {"down": False, "num_pages": 2.0}}),   # built-in shape, capped
    ]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert steps == [
        {"action": "scroll", "down": True, "pages": 0.2},
        {"action": "scroll", "down": False, "pages": 1.0},
    ]


def test_find_by_text_click_without_metadata_compiles_to_semantic_find_click(tmp_path):
    # No recorded element -> replay the INTENT with the tool's own algorithm, not a lossy
    # selector translation.
    history = [_item({"find_by_text": {"text": "View all", "click_first": True}},
                     result=[{"extracted_content": "clicked"}])]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)
    assert steps == [{"action": "find_click", "text": "View all"}]
    # A non-clicking find_by_text still compiles to nothing.
    history = [_item({"find_by_text": {"text": "View all"}})]
    assert compile_recording(_write(tmp_path, history), emit_start_goto=False) == []


def test_restore_result_metadata_reinjects_dropped_field(tmp_path):
    rec = tmp_path / "rec.json"
    rec.write_text(json.dumps({"history": [
        {"state": {"url": "u"}, "model_output": {"action": [{"find_by_text": {}}]},
         "result": [{"extracted_content": "clicked"}]},
    ]}))
    element = {"node_name": "button", "ax_name": "View all", "attributes": {}}
    in_memory = SimpleNamespace(history=[
        SimpleNamespace(result=[
            SimpleNamespace(metadata={"interacted_element": element}),
        ]),
    ])
    assert restore_result_metadata(in_memory, rec) is True
    saved = json.loads(rec.read_text())
    assert saved["history"][0]["result"][0]["metadata"]["interacted_element"] == element
    # Idempotent: a second pass changes nothing.
    assert restore_result_metadata(in_memory, rec) is False


def test_transpiler_maps_scroll():
    steps = [{"action": "scroll", "down": True, "pages": 0.2},
             {"action": "scroll", "down": False, "pages": 0.5}]
    code, anchors = transpile("s", steps)
    assert "await api.scroll(0.2)" in code
    assert "await api.scroll(0.5, down=False)" in code
    assert anchors == {} and lint_code(code) == []


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(ss, "LIBRARY_MANIFEST", tmp_path / "library" / "manifest.json")
    monkeypatch.setattr(ss, "DECOMPOSITIONS_DIR", tmp_path / "decompositions")
    (tmp_path / "library").mkdir()
    return tmp_path


async def test_zero_step_recording_is_never_committed(stores, monkeypatch):
    """An agent that acted only through non-compiling tools must leave NO library entry —
    an empty script would replay as a hollow no-op pass."""
    from automation.pipeline import hybrid
    from tests.test_hybrid import FakeSession, _runner, _seg, _run

    empty_recording = {"history": [
        {"state": {"url": "http://app/section", "interacted_element": []},
         "model_output": {"action": [{"search_page": {"pattern": "Reviews"}}]},
         "result": []},
    ]}
    fake = FakeSession(_runner(), agents=[_seg(True, mode="authored"),
                                          _seg(True, mode="authored")])
    fake.recording = empty_recording
    monkeypatch.setattr(hybrid, "HybridSession", FakeSession.make_opener(fake))
    result = await _run(fake)

    assert result.is_successful is True          # the segments themselves passed
    assert ss.load_manifest() == {}              # but nothing hollow was committed
    for sid_file in (ss.LIBRARY_DIR).glob("*.steps.json"):
        raise AssertionError(f"unexpected committed steps: {sid_file}")


def test_find_by_text_hidden_clicks_replay_the_tool_normal_ones_keep_selectors(tmp_path):
    """A click made through the tool's hidden-control path (hover-revealed/0-size —
    observed live: the Reviews 'View all' icon) compiles to a SEMANTIC find_click step:
    selector+pointer replay is structurally unstable for such controls. A normal
    find_by_text click (snapshot path, full element identity) keeps the proven selector
    replay, with the hidden recovery as a safety net."""
    hidden = [_item({"find_by_text": {"text": "View all", "click_first": True}},
                    result=[{"metadata": {"interacted_element": {
                        "node_name": "button", "ax_name": "View all",
                        "attributes": {"title": "View all", "role": "button"},
                        "hidden_click": True}}}])]
    steps = compile_recording(_write(tmp_path, hidden), emit_start_goto=False)
    assert steps == [{"action": "find_click", "text": "View all"}]

    normal = [_item({"find_by_text": {"text": "btnInvoice", "click_first": True}},
                    result=[{"metadata": {"interacted_element": {
                        "node_name": "button", "ax_name": "btnInvoice",
                        "attributes": {"aria-label": "btnInvoice"},
                        "x_path": "//button[1]"}}}])]
    steps = compile_recording(_write(tmp_path, normal), emit_start_goto=False)
    assert steps[0]["action"] == "click" and steps[0]["hidden_ok"] is True
    assert any("btnInvoice" in s for s in steps[0]["selectors"])


def test_find_click_transpiles_and_parameterizes():
    from automation.skills.codegen import lint_code, transpile

    steps = [{"action": "scroll", "down": True, "pages": 0.5},
             {"action": "find_click", "text": "{{label}}"},
             {"action": "find_click", "text": "View all"}]
    code, anchors = transpile("s", steps, params={"label": "Reviews"})
    assert "await api.find_click(label)" in code
    assert "await api.find_click('View all')" in code
    assert anchors == {} and lint_code(code) == []


def test_hidden_ok_flows_through_anchors_and_api():
    from automation.skills.api import SkillApi
    from automation.skills.codegen import _anchor

    step = {"action": "click", "selectors": ['css=[title="View all"]'],
            "fingerprint": {"tag": "button"}, "hidden_ok": True}
    anchor = _anchor(step)
    assert anchor["hidden_ok"] is True

    api = SkillApi(page=None, anchors={"view-all": anchor})
    rebuilt = api._step_for("view-all")
    assert rebuilt["hidden_ok"] is True
    assert rebuilt["selectors"] == ['css=[title="View all"]']

    # A normal anchor carries no such permission.
    assert "hidden_ok" not in _anchor({"action": "click", "selectors": ["x"]})


def test_discovery_loop_notice_fires_on_pure_discovery_streaks():
    from automation.pipeline.runner import discovery_loop_notice

    def acts(*names):
        return [{n: {}, "interacted_element": None} for n in names]

    # A streak of 4 read-only discovery actions -> nudge; re-fires at 8, silent between.
    assert discovery_loop_notice(acts("click", "list_actions", "search_page",
                                      "capped_scroll", "find_elements")) is not None
    assert discovery_loop_notice(acts("list_actions", "search_page", "capped_scroll")) is None
    assert discovery_loop_notice(
        acts("list_actions", "search_page", "capped_scroll", "find_elements",
             "list_actions")) is None                       # streak 5: between multiples
    assert discovery_loop_notice(
        acts(*(["list_actions"] * 8))) is not None          # streak 8: re-fires
    # Any real interaction resets the streak.
    assert discovery_loop_notice(acts("list_actions", "search_page", "click",
                                      "list_actions", "search_page")) is None
    notice = discovery_loop_notice(acts(*(["search_page"] * 4)))
    assert "ALREADY CLICKED" in notice and "find_by_text" in notice
