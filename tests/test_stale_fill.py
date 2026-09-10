"""Stale-node fills self-heal instead of typing into a detached element.

Motivating failure (run 20260805_131827_339055, the duplicate-add loop): the agent
batched three dropdown picks + the Cost fill into one step. The picks re-rendered the
modal, so the fill's index pointed at a DETACHED node — keystrokes went to whatever held
focus, the read-back read the same dead node ('' -> a false "value did NOT take"
warning), and that warning convinced the agent the form was broken. It re-added the
(actually saved) entry ten times. The `input` action now probes `isConnected` before and
after typing: a detached target is re-resolved to its live twin (same tag + identity
attribute, composed-tree walk) and filled there, or refused honestly with the no_fill
stamp — never typed into blind."""
import json
from types import SimpleNamespace

from test_stubborn_fill import _FakeCdp, _FakeField, _FakeSession, _type

from automation.pipeline import agent_tools


class _StaleCdp(_FakeCdp):
    """_FakeCdp plus the Input.insertText channel the re-resolution fill uses."""

    def __init__(self, field, twin, resolvable=True):
        super().__init__(field, resolvable)
        self.twin = twin
        self.cdp_client.send.Input.insertText = self._insert

    async def _insert(self, params, session_id=None):
        self.twin["value"] = self.twin.get("value", "") + params["text"]
        return {}


class _StaleSession(_FakeSession):
    """A session whose indexed node is (or becomes) detached, with a live twin that the
    composed-tree re-find can discover. The twin is a plain dict the fake `_eval_js`
    (below) serves to the FIELD_REFIND_JS calls."""

    def __init__(self, field, twin, node_attrs=None, disconnect_on_type=False):
        super().__init__(field, node_attrs)
        self.twin = twin
        self.cdp = _StaleCdp(field, twin)
        self.disconnect_on_type = disconnect_on_type

    def _dispatch(self, event):
        res = super()._dispatch(event)
        if self.disconnect_on_type and type(event).__name__ == "TypeTextEvent":
            self.field.connected = False
        return res


def _fake_eval(twin):
    """Serve FIELD_REFIND_JS evaluations from the `twin` dict: 'focus' returns the match
    (or a non-1 count), 'read' returns the twin's current value."""

    async def eval_js(_session, expr, **_kw):
        if '"focus"' in expr:
            if twin.get("count", 1) != 1:
                return {"count": twin.get("count", 1)}
            return {"count": 1, "focused": True, "label": twin.get("label", "Cost"),
                    "attrs": twin.get("attrs", {})}
        if '"read"' in expr:
            return {"count": 1, "value": twin.get("value", "")}
        raise AssertionError(f"unexpected _eval_js expression: {expr[:120]}")

    return eval_js


async def test_detached_field_is_refound_and_filled(monkeypatch):
    twin = {"label": "Cost", "attrs": {"placeholder": "Cost", "type": "text"}}
    field = _FakeField("")
    field.connected = False
    session = _StaleSession(field, twin,
                            node_attrs={"placeholder": "Cost", "type": "text"})
    monkeypatch.setattr(agent_tools, "_eval_js", _fake_eval(twin))

    res = await _type(session, text="200")

    assert session.types == []                     # never typed at the dead node
    assert twin["value"] == "200"                  # the live twin got the text
    msg = res.extracted_content
    assert "re-rendered" in msg
    assert "'Cost'" in msg
    assert "'200'" in msg                          # read back from the live twin
    assert (res.metadata or {}).get("auto_enter") is True
    assert session.enters == 1                     # Enter still goes to the (focused) twin


async def test_detached_field_without_unique_twin_is_refused(monkeypatch):
    twin = {"count": 2}
    field = _FakeField("")
    field.connected = False
    session = _StaleSession(field, twin,
                            node_attrs={"placeholder": "Cost", "type": "text"})
    monkeypatch.setattr(agent_tools, "_eval_js", _fake_eval(twin))

    res = await _type(session, text="200")

    assert session.types == []
    # error channel: the refusal stops the step's remaining queued actions too.
    assert "STALE INDEX" in res.error
    assert (res.metadata or {}).get("no_fill") is True
    assert session.enters == 0


async def test_detached_field_with_no_identity_attr_is_refused(monkeypatch):
    field = _FakeField("")
    field.connected = False
    session = _StaleSession(field, {}, node_attrs={"type": "text"})
    monkeypatch.setattr(agent_tools, "_eval_js", _fake_eval({}))

    res = await _type(session, text="200")

    assert session.types == []
    assert "STALE INDEX" in res.error
    assert (res.metadata or {}).get("no_fill") is True


async def test_mid_step_detach_escalates_to_refind(monkeypatch):
    """Connected at type time, detached by the re-render before the post-check: the typed
    value sits on the dead node (read-back would say '200' — the false-clean case), so
    the fill must land on the live twin instead of returning a clean receipt."""
    twin = {"label": "Cost", "attrs": {"placeholder": "Cost", "type": "text"}}
    field = _FakeField("")
    session = _StaleSession(field, twin, node_attrs={"placeholder": "Cost", "type": "text"},
                            disconnect_on_type=True)
    monkeypatch.setattr(agent_tools, "_eval_js", _fake_eval(twin))

    res = await _type(session, text="200")

    assert twin["value"] == "200"
    assert "re-rendered" in res.extracted_content


async def test_refound_dropdown_filter_is_still_refused(monkeypatch):
    """Re-resolution must not become a side door around the dropdown-filter refusal: a
    twin that IS a combobox filter gets the select_dropdown redirect, not a fill."""
    twin = {"label": "Select", "attrs": {"role": "combobox", "id": "react-select-9-input"}}
    field = _FakeField("")
    field.connected = False
    session = _StaleSession(field, twin, node_attrs={"name": "type", "type": "text"})
    monkeypatch.setattr(agent_tools, "_eval_js", _fake_eval(twin))

    res = await _type(session, text="Assets transferred")

    assert twin.get("value", "") == ""             # nothing typed into the filter
    assert "select_dropdown" in res.error
    assert (res.metadata or {}).get("no_fill") is True


async def test_refound_date_picker_fills_but_suppresses_enter(monkeypatch):
    """A twin that is a date picker (combobox + dialog popup) is FILLED — it is not a
    filter — but keeps the date-picker Enter suppression: blur commits the date."""
    twin = {"label": "Date of birth",
            "attrs": {"role": "combobox", "aria-haspopup": "dialog",
                      "placeholder": "DD/MM/YYYY"}}
    field = _FakeField("")
    field.connected = False
    session = _StaleSession(field, twin, node_attrs={"placeholder": "DD/MM/YYYY",
                                                     "type": "text"})
    monkeypatch.setattr(agent_tools, "_eval_js", _fake_eval(twin))

    res = await _type(session, text="30/06/1982")

    assert twin["value"] == "30/06/1982"
    assert session.enters == 0
    assert (res.metadata or {}).get("auto_enter") is False


async def test_connected_field_never_triggers_refind(monkeypatch):
    """The normal path is untouched: a live field fills exactly as before and the
    re-find machinery is never consulted."""

    async def boom(_session, _expr, **_kw):
        raise AssertionError("re-find ran for a connected field")

    monkeypatch.setattr(agent_tools, "_eval_js", boom)
    session = _FakeSession(_FakeField("3200"))

    res = await _type(session, text="5000")

    assert session.field.value == "5000"
    assert "WARNING" not in res.extracted_content
