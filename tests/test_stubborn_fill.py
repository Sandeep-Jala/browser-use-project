"""Fills into fields that refuse to be cleared from JS.

The failure that motivated this: the pay-forecast grid's amount cells kept their old figure.
browser-use clears a field by assigning `this.value = ""`, but React's value tracker wraps the
value property on the element instance — that assignment moves the tracker with the DOM, the
`input` event that follows looks like a no-op, and React never learns of the change. Its state
still holds the old amount and it repaints that over whatever gets typed next.

The doubles below encode exactly that rule: a `_FakeField` accepts typed text only if it was
first cleared with REAL KEY EVENTS; a JS clear leaves the framework state behind. So a test
here fails if the keyboard clear, the read-back, or the repair round stops happening."""

from types import SimpleNamespace

import pytest
from browser_use.dom.views import EnhancedDOMTreeNode, NodeType

from automation.pipeline import agent_tools
from automation.pipeline.agent_tools import build_tools
from automation.pipeline.script_compile import _fill_with_retry, value_took


# --------------------------------- value_took ---------------------------------

@pytest.mark.parametrize("typed, actual, took", [
    ("5000", "5000", True),
    ("5000", "£5,000.00", True),        # the page regrouped what it accepted
    ("4000", "4000.00", True),
    ("5000", "3200", False),            # the cell kept its old amount
    ("5000", "15000", False),           # NOT a substring pass: a different number
    ("5000", "", False),
    ("Bailey", "Bailey Stevenson", True),  # autocompleted around what we typed
    ("abc", "5000", False),
    ("", "", True),
    ("", "leftover", False),
])
def test_value_took(typed, actual, took):
    assert value_took(typed, actual) is took


# ------------------------------- live-run doubles -------------------------------

class _FakeField:
    """An input whose framework state only moves for real keystrokes.

    `mode` picks which real-world field this is:
      "react"      — the pay-forecast cell: a JS clear is invisible, keystrokes work.
      "no_select"  — also ignores select-all, so only End+Backspace empties it.
      "hopeless"   — never accepts anything (the case the agent must be TOLD about).
    """

    def __init__(self, value="3200", mode="react"):
        self.value = value
        self.mode = mode
        self._state = value        # what the framework thinks the value is
        self._selected = False
        self._kbd_cleared = False
        self.keys: list[str] = []

    def js_clear(self):
        self.value = ""            # DOM only — `_state` is untouched, which is the bug

    def type(self, text):
        # The framework overwrites the typing only while it still holds an OLD value the
        # keystrokes never told it about; once it agrees the field is empty, typing lands.
        stale = bool(self._state) and not self._kbd_cleared
        if self.mode == "hopeless" or stale:
            self.value = self._state   # framework repaints its own value over the typing
        else:
            self.value = self._state = text
        self._kbd_cleared = False

    def key(self, key, commands):
        self.keys.append(key)
        if commands and "selectAll" in commands:
            self._selected = self.mode != "no_select"
        elif key == "Delete" and self._selected:
            self._clear_for_real()
        elif key == "Backspace":
            self.value = self.value[:-1]
            if not self.value:
                self._clear_for_real()
        elif key == "End":
            self._selected = False

    def _clear_for_real(self):
        self.value = ""
        self._kbd_cleared = True
        if self.mode != "hopeless":
            self._state = ""


class _FakeCdp:
    """Just the three CDP domains a fill touches."""

    def __init__(self, field, resolvable=True):
        self.field = field
        self.session_id = "s1"
        self.resolvable = resolvable
        self.cdp_client = SimpleNamespace(send=SimpleNamespace(
            DOM=SimpleNamespace(resolveNode=self._resolve_node),
            Runtime=SimpleNamespace(callFunctionOn=self._call),
            Input=SimpleNamespace(dispatchKeyEvent=self._key),
        ))

    async def _resolve_node(self, params, session_id=None):
        return {"object": {"objectId": "obj-1"}} if self.resolvable else {"object": {}}

    async def _call(self, params, session_id=None):
        decl = params["functionDeclaration"]
        if "focus()" in decl:
            return {"result": {"value": True}}
        if "isConnected" in decl:
            # Stale-node probe: live fields answer True; tests model a re-rendered
            # (detached) node by setting field.connected = False.
            return {"result": {"value": getattr(self.field, "connected", True)}}
        if "closest" in decl:
            # Layer-popup probe: tests model a callout-hosted field by setting
            # field.in_popup = True.
            return {"result": {"value": getattr(self.field, "in_popup", False)}}
        return {"result": {"value": self.field.value}}

    async def _key(self, params, session_id=None):
        if params["type"] == "keyDown":
            self.field.key(params["key"], params.get("commands"))
        return {}


def _input_node(attributes=None):
    """A real EnhancedDOMTreeNode — TypeTextEvent re-validates the node it is given, so a
    stand-in namespace never reaches the event bus."""
    return EnhancedDOMTreeNode(
        node_id=11, backend_node_id=11, node_type=NodeType.ELEMENT_NODE, node_name="INPUT",
        node_value="", attributes=attributes or {}, is_scrollable=False, is_visible=True,
        absolute_position=None, target_id="t1", frame_id="f1", session_id="s1",
        content_document=None, shadow_root_type=None, shadow_roots=[], parent_node=None,
        children_nodes=[], ax_node=None, snapshot_node=None,
    )


class _FakeEvent:
    def __init__(self, metadata=None):
        self._metadata = metadata

    def __await__(self):
        async def done():
            return self
        return done().__await__()

    async def event_result(self, raise_if_any=False, raise_if_none=False):
        return self._metadata


class _FakeSession:
    """browser_session as the `input` action uses it, recording every TypeTextEvent."""

    def __init__(self, field, node_attrs=None, resolvable=True, clears_on_enter=False):
        self.field = field
        self.cdp = _FakeCdp(field, resolvable)
        self.types: list[tuple[str, bool]] = []
        self.clears_on_enter = clears_on_enter
        self.enters = 0
        self.node = _input_node(node_attrs)
        self.event_bus = SimpleNamespace(dispatch=self._dispatch)

    async def get_element_by_index(self, index):
        return self.node

    async def get_or_create_cdp_session(self):
        return self.cdp

    def _dispatch(self, event):
        if type(event).__name__ == "TypeTextEvent":
            self.types.append((event.text, event.clear))
            if event.clear:
                self.field.js_clear()
            self.field.type(event.text)
            return _FakeEvent({"actual_value": self.field.value})
        self.enters += 1               # SendKeysEvent
        if self.clears_on_enter:
            self.field.value = ""      # a search box that empties itself on submit
        return _FakeEvent(None)


async def _type(session, text="5000", clear=True):
    action = build_tools().registry.registry.actions["input"]
    params = action.param_model(index=3, text=text, clear=clear)
    return await action.function(params=params, browser_session=session)


# --------------------------------- live-run tests ---------------------------------

async def test_react_field_is_cleared_with_keystrokes_not_js():
    """The regression itself: a JS clear is never asked for, and the amount lands."""
    session = _FakeSession(_FakeField("3200"))
    res = await _type(session)

    assert session.field.value == "5000"
    # browser-use is handed clear=False — its JS `value = ""` must not run.
    assert session.types == [("5000", False)]
    assert session.field.keys[:2] == ["a", "Delete"]
    assert "WARNING" not in res.extracted_content
    assert session.enters == 1


async def test_field_that_ignores_select_all_is_emptied_with_backspace():
    session = _FakeSession(_FakeField("3200", mode="no_select"))
    res = await _type(session)

    assert session.field.value == "5000"
    assert "End" in session.field.keys and "Backspace" in session.field.keys
    assert "WARNING" not in res.extracted_content


async def test_reverted_value_is_repaired_and_reported():
    """A field that takes the text only on the second round: repaired, and said so."""
    field = _FakeField("3200")
    session = _FakeSession(field)

    # Make the first type land wrong, as if the row re-rendered under us.
    real_type = field.type
    calls = {"n": 0}

    def flaky(text):
        calls["n"] += 1
        if calls["n"] == 1:
            field.value = "3200"       # old amount repainted
            return
        real_type(text)
    field.type = flaky

    res = await _type(session)

    assert field.value == "5000"
    assert len(session.types) == 2                      # typed, verified, retyped
    assert "cleared and retyped" in res.extracted_content


async def test_field_that_never_takes_warns_the_agent_with_the_real_value():
    session = _FakeSession(_FakeField("3200", mode="hopeless"))
    res = await _type(session)

    assert "WARNING" in res.extracted_content
    assert "'3200'" in res.extracted_content            # what it actually reads
    # The recovery ladder the INPUT VALUE MISMATCH prompt rule expects to see.
    assert "Press Escape" in res.extracted_content
    assert "reload the page" in res.extracted_content
    # Two repair rounds were attempted before giving up.
    assert len(session.types) == 1 + agent_tools._FILL_REPAIR_ROUNDS


async def test_unresolvable_element_falls_back_to_browser_use_clear():
    """No object id (detached node): keep the old behaviour rather than skipping the clear."""
    session = _FakeSession(_FakeField("3200"), resolvable=False)
    await _type(session)

    assert session.types == [("5000", True)]            # clear delegated to browser-use


async def test_search_box_that_empties_on_submit_is_not_called_a_failure():
    """Verification reads the field BEFORE the auto-Enter, so a box that clears itself on
    submit isn't mistaken for one that refused the text."""
    session = _FakeSession(_FakeField(""), clears_on_enter=True)
    res = await _type(session, text="Bailey Stevenson")

    assert session.field.value == ""       # the box did empty, as that app does
    assert "WARNING" not in res.extracted_content


async def test_dropdown_filter_still_suppresses_enter():
    # Typed filter text is now REFUSED outright (test_auto_enter_input's refusal tests
    # own that contract) — refusal keeps Enter out of the combobox, the invariant this
    # test has always pinned.
    session = _FakeSession(_FakeField(""), node_attrs={"role": "combobox"})
    res = await _type(session, text="Bailey Stevenson")

    assert session.enters == 0
    assert "REFUSED" in res.error


# --------------------------------- replay tests ---------------------------------

class _FakeLocator:
    """Playwright Locator surface used by _fill_with_retry, over a _FakeField."""

    def __init__(self, field):
        self.field = field
        self.refilled = False

    async def fill(self, value, timeout=None):
        if value == "":
            self.field.js_clear()
        else:
            self.field.type(value)

    async def input_value(self, timeout=None):
        return self.field.value

    async def click(self, timeout=None):
        pass

    async def press(self, key, timeout=None):
        self.field.key("a" if key == "ControlOrMeta+a" else key,
                       ["selectAll"] if key == "ControlOrMeta+a" else None)

    async def press_sequentially(self, value, delay=None, timeout=None):
        self.refilled = True
        self.field.type(value)


def _patch_resolve(monkeypatch, loc):
    async def resolve(page, step, timeout_ms, require_editable=False):
        return loc, "#amount", None
    monkeypatch.setattr("automation.pipeline.script_compile._resolve", resolve)


async def test_replayed_fill_repairs_a_field_that_did_not_take(monkeypatch):
    loc = _FakeLocator(_FakeField("3200"))
    _patch_resolve(monkeypatch, loc)

    sel, _ = await _fill_with_retry(None, {"value": "5000", "clear": True}, 5000)

    assert sel == "#amount"
    assert loc.refilled is True             # Playwright's fill() alone was not enough
    assert loc.field.value == "5000"


async def test_replayed_fill_fails_loudly_when_the_value_never_takes(monkeypatch):
    """Silence here would save a wrong amount just as happily as a right one."""
    loc = _FakeLocator(_FakeField("3200", mode="hopeless"))
    _patch_resolve(monkeypatch, loc)

    with pytest.raises(Exception, match="fill did not take"):
        await _fill_with_retry(None, {"value": "5000", "clear": True}, 5000)
