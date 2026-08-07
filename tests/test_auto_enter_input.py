"""The auto-Enter `input` replacement: dropdown-filter detection, registry override, the
dropdown-filter typing refusal, and compile_recording mirroring the Enter via the result's
metadata.auto_enter flag (absent on old recordings and dropdown fills, so those compile
exactly as they ran)."""
from test_agent_tools import Node
from test_compile_coverage import _item, _write
from test_stubborn_fill import _FakeField, _FakeSession, _type

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


def test_dialog_popup_combobox_is_a_date_picker_not_a_filter():
    # The Fluent DatePicker (attributes verbatim from run 20260807_093003's DOB field) is a
    # typeable input whose popup is a CALENDAR: typed text is its value, not discarded
    # filter text — it must never match, or every date fill gets refused into calendar
    # navigation. Option-list popups keep matching.
    assert not _is_dropdown_filter(Node(attributes={
        "type": "text", "id": "DatePicker2223-label", "role": "combobox",
        "aria-expanded": "false", "aria-haspopup": "dialog",
        "placeholder": "DD/MM/YYYY", "class": "ms-TextField-field field-1140"}))
    assert _is_dropdown_filter(Node(attributes={"role": "combobox",
                                                "aria-haspopup": "listbox"}))
    assert _is_dropdown_filter(Node(attributes={"role": "combobox",
                                                "aria-haspopup": "true"}))


# --------------------------- registry override ---------------------------


def test_input_action_is_overridden_with_auto_enter_variant():
    reg = build_tools().registry.registry.actions
    assert "input" in reg
    assert "press Enter automatically" in reg["input"].description


# --------------------------- dropdown-filter refusal ---------------------------
# Motivating failure (run 20260805_123407_334719): the agent typed 'Assets transferred'
# into a react-select filter, never clicked an option, and reported the field as set. The
# receipt's "click the option you want" was advisory; nothing was selected and the modal's
# dependent fields never rendered. Typing into a filter selects NOTHING — refuse it and
# name select_dropdown, the action that picks AND verifies.


async def test_dropdown_filter_typing_is_refused():
    session = _FakeSession(_FakeField(""), node_attrs={"id": "react-select-9-input"})
    res = await _type(session, text="Assets transferred")

    assert session.types == []                    # nothing was typed
    assert session.enters == 0                    # and no Enter went out
    assert session.field.keys == []               # no keyboard-clear side effect either
    assert (res.metadata or {}).get("no_fill") is True
    # error channel: multi_act stops the step's remaining queued actions on it — a
    # refused fill must not let an already-queued Send/Save fire (run 20260807_095537).
    assert "select_dropdown" in res.error
    assert "Assets transferred" in res.error


async def test_date_picker_typing_is_not_refused_and_enter_stays_suppressed():
    # The refusal's counter-case (run 20260807_093003): the DOB DatePicker is role=combobox
    # but its typed text IS the value — the fill must go through and read back. Enter stays
    # SUPPRESSED (4523b58 parity, the sequence that always filled dates cleanly): with the
    # calendar callout open, Enter is handled by the picker and can commit the callout's
    # highlighted date over the typed text — invisibly, since the read-back runs pre-Enter.
    session = _FakeSession(_FakeField(""), node_attrs={
        "role": "combobox", "aria-haspopup": "dialog", "placeholder": "DD/MM/YYYY"})
    res = await _type(session, text="30/06/1982")

    assert not (res.metadata or {}).get("no_fill")
    assert res.error is None
    assert session.field.value == "30/06/1982"
    assert session.types == [("30/06/1982", False)]  # typed, with the keyboard clear
    assert session.enters == 0                       # date picker: no Enter, blur commits
    assert (res.metadata or {}).get("auto_enter") is False   # replays compile fill-only
    assert "date picker — Enter suppressed" in res.extracted_content


async def test_dropdown_filter_clear_only_is_still_allowed():
    # text="" is a pure clear (stuck-filter recovery) — it selects nothing wrong and must
    # keep working; only actual filter TEXT gets the refusal.
    session = _FakeSession(_FakeField("stale"), node_attrs={"role": "combobox"})
    res = await _type(session, text="")

    assert not (res.metadata or {}).get("no_fill")
    assert session.types == [("", False)]         # the clear still ran
    assert session.enters == 0                    # dropdown: Enter stays suppressed


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


def test_date_picker_fill_compiles_without_press_even_if_recorded_with_enter(tmp_path):
    # Recordings made in the brief window when the live tool pressed Enter after dates
    # (2026-08-07 morning) carry auto_enter=True on DatePicker fills; replaying that
    # press risks the callout committing its highlighted date over the typed one.
    item = _item(
        {"input": {"index": 5, "text": "30/06/1982", "clear": True}},
        result=[{"extracted_content": "Typed '30/06/1982' and pressed Enter",
                 "metadata": {"auto_enter": True}}],
        element={"node_name": "INPUT", "ax_name": "Date of birth",
                 "attributes": {"role": "combobox", "aria-haspopup": "dialog",
                                "placeholder": "DD/MM/YYYY", "id": "DatePicker2223-label"},
                 "x_path": "html/body/div/input"},
    )
    steps = _compile(tmp_path, item)
    assert [s["action"] for s in steps] == ["fill"]
    assert steps[0]["value"] == "30/06/1982"


def test_refused_dropdown_fill_compiles_to_nothing(tmp_path):
    # The tool typed NOTHING (dropdown-filter refusal) — compiling the recorded action
    # would bake a phantom fill into the replay, the fill-shaped twin of the no_click
    # phantom-click bug.
    item = _item(
        {"input": {"index": 5, "text": "Assets transferred", "clear": True}},
        result=[{"extracted_content": "REFUSED — dropdown filter",
                 "metadata": {"no_fill": True}}],
        element={"node_name": "INPUT", "ax_name": "",
                 "attributes": {"id": "react-select-9-input", "role": "combobox"},
                 "x_path": "html/body/div/input"},
    )
    assert _compile(tmp_path, item) == []
