"""agent_tools: descendant icon-hint harvesting.

The failure that motivated this: Fluent UI icon buttons (`<button class="ms-Button--icon">`)
carry their meaning only in a child glyph, which browser-use serializes away — so the send,
edit, and delete icons all reach the agent as a nameless `<button/>`. `_descendant_icon_hints`
recovers that meaning from the child so nameless icon buttons become findable."""

from automation.pipeline import agent_tools
from automation.pipeline.agent_tools import _descendant_icon_hints, read_new_notifications


class Node:
    """Minimal stand-in for browser-use's EnhancedDOMTreeNode (only what the harvester reads)."""

    def __init__(self, node_name="DIV", attributes=None, node_value="", children=None):
        self.node_name = node_name
        self.attributes = attributes or {}
        self.node_value = node_value
        self.children = children or []


def _button(*children):
    return Node("BUTTON", {"class": "ms-Button ms-Button--icon", "type": "button"},
                children=list(children))


def test_fluent_data_icon_name_is_recovered():
    btn = _button(Node("I", {"data-icon-name": "MailReminder", "class": "ms-Icon"}))
    assert _descendant_icon_hints(btn) == "MailReminder"


def test_child_title_and_aria_label():
    btn = _button(Node("SPAN", {"title": "Send survey"}),
                  Node("I", {"aria-label": "envelope"}))
    hints = _descendant_icon_hints(btn).lower()
    assert "send survey" in hints and "envelope" in hints


def test_svg_title_text_and_use_href():
    svg = Node("SVG", children=[
        Node("TITLE", node_value="Send NPS survey request"),
        Node("USE", {"href": "#icon-send"}),
    ])
    hints = _descendant_icon_hints(_button(svg))
    assert "Send NPS survey request" in hints
    assert "icon-send" in hints


def test_icon_font_class_token():
    btn = _button(Node("I", {"class": "fa fa-envelope"}))
    assert _descendant_icon_hints(btn) == "envelope"
    btn2 = _button(Node("I", {"class": "ms-Icon ms-Icon--Send"}))
    assert _descendant_icon_hints(btn2) == "Send"


def test_dedup_and_empty():
    # Same hint from two children collapses to one.
    btn = _button(Node("I", {"data-icon-name": "Send"}), Node("I", {"title": "send"}))
    assert _descendant_icon_hints(btn).lower().split() == ["send"]
    # A button with no semantic children yields nothing (not a crash).
    assert _descendant_icon_hints(_button(Node("SPAN", node_value="   "))) == ""
    assert _descendant_icon_hints(Node("BUTTON")) == ""


def test_depth_is_bounded():
    # A hint buried deeper than the walk limit is not harvested (keeps the scan cheap).
    deep = Node("I", {"data-icon-name": "TooDeep"})
    for _ in range(6):
        deep = Node("DIV", children=[deep])
    assert "TooDeep" not in _descendant_icon_hints(_button(deep), depth=4)


# ------------------------------- read_new_notifications -------------------------------


async def test_notifications_strip_icon_glyphs_and_dedup(monkeypatch):
    """PUA icon glyphs are stripped, whitespace collapsed, batch de-duplicated, and an
    icon-only 'notification' drops out. (The JS observer needs a browser; the Python
    cleaning path is faked here by stubbing _eval_js.)"""
    G = "\ue7ba"  # a Fluent private-use icon glyph
    raw = [
        G + " Client has unpaid invoice.  Please add a payment link",  # icon + real text
        "\uea39",                                                      # icon only -> dropped
        "Client has unpaid invoice. Please add a payment link",         # dup of #1 cleaned
        "Saved successfully",
    ]

    async def fake_eval(_session, _expr):
        return raw
    monkeypatch.setattr(agent_tools, "_eval_js", fake_eval)

    out = await read_new_notifications(object())  # non-None session
    assert out == ["Client has unpaid invoice. Please add a payment link", "Saved successfully"]


async def test_notifications_none_session_and_bad_result(monkeypatch):
    assert await read_new_notifications(None) == []

    async def boom(_s, _e):
        raise RuntimeError("cdp down")
    monkeypatch.setattr(agent_tools, "_eval_js", boom)
    assert await read_new_notifications(object()) == []  # never raises

    async def not_a_list(_s, _e):
        return "oops"
    monkeypatch.setattr(agent_tools, "_eval_js", not_a_list)
    assert await read_new_notifications(object()) == []


# ------------------------------- element capture resilience -------------------------------


def test_captured_element_manual_fallback_for_offscreen_nodes():
    """The full DOMInteractedElement loader throws for off-screen/zero-size nodes (the
    exact nodes find_by_text exists to reach); the capture must degrade to a manual
    identity carrying the fields compile anchors on — never to None."""
    from types import SimpleNamespace

    from automation.pipeline.agent_tools import _captured_element

    class _Node(SimpleNamespace):
        @property
        def xpath(self):  # traversal fails off-screen, exactly like 0.13.3
            raise RuntimeError("no layout")

    node = _Node(node_name="BUTTON", attributes={"title": "View all"},
                 ax_node=None, backend_node_id=123)
    el = _captured_element(node, "View all")
    assert el is not None
    assert el["node_name"] == "BUTTON"
    assert el["attributes"] == {"title": "View all"}
    assert el["ax_name"] == "View all"          # matched label fills in for the ax name
    assert "x_path" not in el                   # unreachable xpath is simply omitted
