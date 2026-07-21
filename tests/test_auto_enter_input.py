"""The auto-Enter `input` replacement: dropdown-filter detection, registry override, and
compile_recording mirroring the Enter via the result's metadata.auto_enter flag (absent on
old recordings and dropdown fills, so those compile exactly as they ran)."""
from test_agent_tools import Node
from test_compile_coverage import _item, _write

from automation.pipeline.agent_tools import _is_dropdown_filter, build_tools
from automation.pipeline.script_compile import compile_recording


# --------------------------- dropdown-filter detection ---------------------------


def test_react_select_filter_is_dropdown():
    assert _is_dropdown_filter(Node(attributes={"id": "react-select-6-input"}))


def test_combobox_role_and_aria_autocomplete_are_dropdowns():
    assert _is_dropdown_filter(Node(attributes={"role": "combobox"}))
    assert _is_dropdown_filter(Node(attributes={"aria-autocomplete": "list"}))
    assert not _is_dropdown_filter(Node(attributes={"aria-autocomplete": "none"}))


def test_plain_search_input_is_not_dropdown():
    assert not _is_dropdown_filter(Node(attributes={"id": "SearchBox137",
                                                    "placeholder": "Search"}))
    assert not _is_dropdown_filter(Node())


# --------------------------- registry override ---------------------------


def test_input_action_is_overridden_with_auto_enter_variant():
    reg = build_tools().registry.registry.actions
    assert "input" in reg
    assert "press Enter automatically" in reg["input"].description


# --------------------------- compile mirroring ---------------------------

_FIELD = {"node_name": "INPUT", "ax_name": "Search",
          "attributes": {"id": "SearchBox137", "placeholder": "Search"},
          "x_path": "html/body/div/input"}


def _input_item(auto_enter=None):
    result = {"extracted_content": "Typed '290 CREW'"}
    if auto_enter is not None:
        result["metadata"] = {"auto_enter": auto_enter}
    return _item({"input": {"index": 5, "text": "290 CREW", "clear": True}},
                 result=[result], element=_FIELD)


def _compile(tmp_path, item):
    return compile_recording(_write(tmp_path, [item]), emit_start_goto=False)


def test_auto_enter_flag_compiles_fill_plus_press(tmp_path):
    steps = _compile(tmp_path, _input_item(auto_enter=True))
    assert [s["action"] for s in steps] == ["fill", "press"]
    assert steps[0]["value"] == "290 CREW"
    assert steps[1] == {"action": "press", "keys": "Enter"}


def test_old_recording_without_flag_compiles_fill_only(tmp_path):
    steps = _compile(tmp_path, _input_item())
    assert [s["action"] for s in steps] == ["fill"]


def test_suppressed_enter_compiles_fill_only(tmp_path):
    steps = _compile(tmp_path, _input_item(auto_enter=False))
    assert [s["action"] for s in steps] == ["fill"]
