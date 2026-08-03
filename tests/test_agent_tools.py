"""agent_tools: descendant icon-hint harvesting.

The failure that motivated this: Fluent UI icon buttons (`<button class="ms-Button--icon">`)
carry their meaning only in a child glyph, which browser-use serializes away — so the send,
edit, and delete icons all reach the agent as a nameless `<button/>`. `_descendant_icon_hints`
recovers that meaning from the child so nameless icon buttons become findable."""

from automation.pipeline import agent_tools
from automation.pipeline.agent_tools import (
    _descendant_icon_hints,
    ensure_reveal_css,
    read_new_notifications,
)


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


# ------------------------------- ensure_reveal_css -------------------------------


async def test_reveal_css_evals_the_shared_installer(monkeypatch):
    """The helper must inject EXACTLY the shared script_compile constant — identity, not a
    lookalike — so authoring and replay install one and the same stylesheet."""
    calls = []

    async def fake_eval(_session, expr):
        calls.append(expr)
    monkeypatch.setattr(agent_tools, "_eval_js", fake_eval)

    await ensure_reveal_css(object())
    assert calls == [agent_tools._REVEAL_CSS_JS]


async def test_reveal_css_never_raises_and_skips_none_session(monkeypatch):
    calls = []

    async def boom(_s, _e):
        calls.append(1)
        raise RuntimeError("cdp down")
    monkeypatch.setattr(agent_tools, "_eval_js", boom)

    await ensure_reveal_css(object())  # swallowed
    assert calls == [1]
    await ensure_reveal_css(None)      # no session -> no eval
    assert calls == [1]


# ------------------------------- extract_data -------------------------------


def _registered_action(name):
    from automation.pipeline.agent_tools import build_tools

    action = build_tools().registry.registry.actions[name]
    return action.function, action.param_model


class _FakeDomNode:
    """Snapshot node double for _matching_nodes/_captured_element (the full
    DOMInteractedElement loader raises on it, exercising the manual-capture path)."""

    def __init__(self, text, attributes=None):
        self._text = text
        self.attributes = attributes or {}
        self.node_name = "H2"
        self.ax_node = None
        self.backend_node_id = 7
        self.children = []

    def get_all_children_text(self, max_depth=5):
        return self._text

    @property
    def xpath(self):
        raise RuntimeError("no layout")


class _FakeBrowserSession:
    def __init__(self, nodes):
        self._nodes = nodes

    async def get_browser_state_summary(self, include_screenshot=False):
        from types import SimpleNamespace

        return SimpleNamespace(dom_state=SimpleNamespace(selector_map=self._nodes),
                               url="https://duckduckgo.com")


async def test_extract_data_captures_value_and_replayable_element_metadata():
    fn, pm = _registered_action("extract_data")
    node = _FakeDomNode("Acting Office — Cloud Accounting",
                        attributes={"data-testid": "result-title"})
    res = await fn(params=pm(text="acting office", label="Top Result Title!"),
                   browser_session=_FakeBrowserSession({3: node}))

    ext = (res.metadata or {}).get("extract")
    assert ext is not None
    assert ext["label"] == "top_result_title"           # slugified
    assert ext["value"] == "Acting Office — Cloud Accounting"
    assert ext["query"] == "acting office"
    assert ext["interacted_element"]["node_name"] == "H2"
    assert ext["interacted_element"]["attributes"] == {"data-testid": "result-title"}
    assert "top_result_title = 'Acting Office" in res.extracted_content


async def test_extract_data_no_match_returns_guidance_without_metadata(monkeypatch):
    async def no_raw(_session, _expr):
        return {"count": 0}
    monkeypatch.setattr(agent_tools, "_eval_js", no_raw)

    fn, pm = _registered_action("extract_data")
    res = await fn(params=pm(text="ghost value", label="g"),
                   browser_session=_FakeBrowserSession({}))
    assert res.metadata is None                          # compiles to NOTHING
    assert "NOT captured" in res.extracted_content


async def test_extract_data_raw_dom_fallback_for_non_interactive_text(monkeypatch):
    async def raw(_session, _expr):
        return {"count": 1, "clicked": False, "name": "Acting Office result",
                "element": {"tag": "h2", "attrs": {"id": "r1"}}}
    monkeypatch.setattr(agent_tools, "_eval_js", raw)

    fn, pm = _registered_action("extract_data")
    res = await fn(params=pm(text="acting office", label="t"),
                   browser_session=_FakeBrowserSession({}))
    ext = res.metadata["extract"]
    assert ext["value"] == "Acting Office result"
    assert ext["interacted_element"] == {"node_name": "h2", "attributes": {"id": "r1"},
                                         "ax_name": "Acting Office result"}


async def test_verify_download_reports_session_downloads():
    """The download ground-truth probe: NONE with an empty session, CONFIRMED with the
    downloaded files' basenames — the receipt that stops an agent from re-clicking a
    download control whose click receipt lied with a timeout."""
    from types import SimpleNamespace

    fn, _pm = _registered_action("verify_download")

    res = await fn(browser_session=SimpleNamespace(downloaded_files=[]))
    assert "NONE" in res.extracted_content

    res = await fn(browser_session=SimpleNamespace(
        downloaded_files=["/tmp/x/FOOD LIMITED_Bailey Stevenson_Forecast report_27.pdf"]))
    assert "CONFIRMED" in res.extracted_content
    assert "Forecast report_27.pdf" in res.extracted_content
    assert "do NOT click" in res.extracted_content


async def test_extract_data_defers_form_controls_to_raw_dom_finders(monkeypatch):
    """A <select> snapshot match must NOT be read via children text — that concatenates
    every option label ('Random Male Female', the observed live failure). The tool defers
    to the raw-DOM finders, whose select branch reports the SELECTED option."""
    async def raw(_session, _expr):
        return {"count": 1, "clicked": False, "name": "Male",
                "element": {"tag": "select", "attrs": {"id": "gender"},
                            "xpath": "/html/body/select"}}
    monkeypatch.setattr(agent_tools, "_eval_js", raw)

    node = _FakeDomNode("Random Male Female", attributes={"id": "gender"})
    node.node_name = "SELECT"
    fn, pm = _registered_action("extract_data")
    res = await fn(params=pm(text="male", label="gender"),
                   browser_session=_FakeBrowserSession({6: node}))
    ext = res.metadata["extract"]
    assert ext["value"] == "Male"                        # not the option blob
    assert ext["interacted_element"]["node_name"] == "select"
    assert ext["interacted_element"]["x_path"] == "/html/body/select"


async def test_extract_data_control_only_match_with_no_raw_hit_is_honest_miss(monkeypatch):
    """When the only snapshot match is a form control and the raw-DOM finders find
    nothing either (an <input>'s current value has no text presence), the tool must say
    so instead of reporting a wrong blob."""
    async def no_raw(_session, _expr):
        return {"count": 0}
    monkeypatch.setattr(agent_tools, "_eval_js", no_raw)

    node = _FakeDomNode("", attributes={"id": "email", "value": "a@b.com"})
    node.node_name = "INPUT"
    fn, pm = _registered_action("extract_data")
    res = await fn(params=pm(text="a@b.com", label="email"),
                   browser_session=_FakeBrowserSession({4: node}))
    assert res.metadata is None
    assert "NOT captured" in res.extracted_content


async def test_extract_data_falls_through_to_static_text_finder(monkeypatch):
    """The live failure this guards: a value in a bare <h3> (fakenamegenerator's generated
    identity) is invisible to the interactive snapshot AND to RAW_FIND_JS's control
    selector — extract_data must fall through to the static-text finder instead of
    reporting 'nothing matched' forever."""
    exprs = []

    async def raw(_session, expr):
        exprs.append(expr)
        if len(exprs) == 1:                       # RAW_FIND_JS: no control matches
            return {"count": 0}
        return {"count": 1, "name": "Felix MacDonald",
                "element": {"tag": "h3", "attrs": {}, "xpath": "/html/body/div/h3"}}
    monkeypatch.setattr(agent_tools, "_eval_js", raw)

    fn, pm = _registered_action("extract_data")
    res = await fn(params=pm(text="felix macdonald", label="generated_name"),
                   browser_session=_FakeBrowserSession({}))

    assert len(exprs) == 2
    assert exprs[0] != exprs[1]                   # second call is the TEXT finder
    ext = res.metadata["extract"]
    assert ext["label"] == "generated_name"
    assert ext["value"] == "Felix MacDonald"
    assert ext["interacted_element"]["node_name"] == "h3"
    # The positional anchor rides along — without it the compiled extract would anchor
    # ONLY on text="Felix MacDonald", dead the moment the page regenerates its value.
    assert ext["interacted_element"]["x_path"] == "/html/body/div/h3"


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


# --------------------- find_by_text truthfulness on unclickables ---------------------
# The live failure these guard: browser-use REFUSES a click on a native <select> by
# RETURNING {'validation_error': ...} (never raising), and find_by_text used to discard
# that result and report "clicked the single match" anyway — two phantom receipts on the
# country <select> convinced the agent its already-applied selection kept failing.


class _FakeEvent:
    def __init__(self, result=None, exc=None):
        self._result, self._exc = result, exc

    def __await__(self):
        async def _dispatched():
            return None
        return _dispatched().__await__()

    async def event_result(self, **_kw):
        if self._exc:
            raise self._exc
        return self._result


class _FakeClickSession(_FakeBrowserSession):
    def __init__(self, nodes, result=None, exc=None):
        super().__init__(nodes)
        from types import SimpleNamespace

        self.event_bus = SimpleNamespace(dispatch=lambda _e: _FakeEvent(result, exc))


class _FakeSelectDomNode(_FakeDomNode):
    def __init__(self):
        super().__init__("Australia Austria United Kingdom", attributes={"id": "c"})
        self.node_name = "SELECT"
        self.tag_name = "select"


async def test_find_by_text_refuses_to_click_native_select():
    fn, pm = _registered_action("find_by_text")
    res = await fn(params=pm(text="united kingdom", click_first=True),
                   browser_session=_FakeBrowserSession({9: _FakeSelectDomNode()}))

    assert res.error is None
    assert "did NOT click" in res.extracted_content
    assert "select_dropdown(index=9" in res.extracted_content
    assert res.metadata == {"no_click": True}    # compiles to NOTHING, never a phantom click


async def test_find_by_text_refuses_to_click_file_input():
    fn, pm = _registered_action("find_by_text")
    node = _FakeDomNode("", attributes={"type": "file", "name": "csv upload"})
    node.node_name = "INPUT"
    node.tag_name = "input"
    res = await fn(params=pm(text="csv upload", click_first=True),
                   browser_session=_FakeBrowserSession({4: node}))

    assert res.error is None
    assert "did NOT click" in res.extracted_content
    assert "upload_file" in res.extracted_content
    assert res.metadata == {"no_click": True}


async def test_find_by_text_reports_refused_click_instead_of_lying(monkeypatch):
    """A returned validation_error dict is a REFUSAL, not a click — the receipt must say
    so and must not stamp interacted_element (which would compile a dead click step)."""
    monkeypatch.setattr(agent_tools, "ClickElementEvent", lambda node: ("click", node))
    fn, pm = _registered_action("find_by_text")
    node = _FakeDomNode("Continue", attributes={"id": "go"})
    node.tag_name = "h2"
    refusal = {"validation_error": "Cannot click on <select> elements. "
                                   "Use dropdown_options(index=...) action instead."}
    res = await fn(params=pm(text="continue", click_first=True),
                   browser_session=_FakeClickSession({4: node}, result=refusal))

    assert res.error is None
    assert "REFUSED" in res.extracted_content
    assert "Nothing was clicked" in res.extracted_content
    assert res.metadata == {"no_click": True}


async def test_find_by_text_refuses_self_match_click_on_text_input():
    """The live failure this guards (run 20260803_152301_114498): the agent typed the
    employee name into the WRONG react-select (value/placeholder invisible in the
    snapshot), then find_by_text(name, click_first=True) matched ONLY its own typed text
    inside that input and "clicked" it — a no-op wearing a success receipt, so the agent
    believed the employee was selected. A single match that is a text-entry field whose
    identity (aria-label/title/placeholder/name/id) does NOT carry the query must refuse."""
    fn, pm = _registered_action("find_by_text")
    node = _FakeDomNode("Calum Findlay",
                        attributes={"id": "react-select-2-input", "type": "text",
                                    "role": "combobox"})
    node.node_name = "INPUT"
    node.tag_name = "input"
    res = await fn(params=pm(text="Calum Findlay", click_first=True),
                   browser_session=_FakeBrowserSession({18958: node}))

    assert res.error is None
    assert "did NOT click" in res.extracted_content
    assert res.metadata == {"no_click": True}    # compiles to NOTHING, never a phantom click


async def test_find_by_text_still_clicks_input_matched_by_placeholder(monkeypatch):
    """The guard must NOT block the recovery path it steers toward: clicking an input
    found by its placeholder/label (e.g. find_by_text('Select Employee')) is the
    legitimate way to focus/open a combobox."""
    monkeypatch.setattr(agent_tools, "ClickElementEvent", lambda node: ("click", node))
    fn, pm = _registered_action("find_by_text")
    node = _FakeDomNode("", attributes={"placeholder": "Select Employee", "type": "text"})
    node.node_name = "INPUT"
    node.tag_name = "input"
    res = await fn(params=pm(text="select employee", click_first=True),
                   browser_session=_FakeClickSession({4: node}, result=None))

    assert res.error is None
    assert "clicked the single match" in res.extracted_content
    assert res.metadata and "interacted_element" in res.metadata


# ------------------------- select_dropdown read-back override -------------------------
# The built-in trusts the picker's self-report; on an ad-heavy page the confirmation can
# time out AFTER the selection took, and the empty failure receipt made the agent re-set
# the same option repeatedly. The override reads the element back and reports what the
# select actually shows.


class _FakeSelectSession:
    def __init__(self, result=None, exc=None):
        from types import SimpleNamespace

        # tag_name matters since the custom-combobox branch: only a native <select>
        # takes the SelectDropdownOptionEvent path these tests exercise.
        self._node = SimpleNamespace(tag_name="select")
        self.event_bus = SimpleNamespace(dispatch=lambda _e: _FakeEvent(result, exc))

    async def get_element_by_index(self, _index):
        return self._node


def _patch_select_readback(monkeypatch, state):
    monkeypatch.setattr(agent_tools, "SelectDropdownOptionEvent", lambda **kw: kw)

    async def field_handle(_session, _node):
        return ("cdp", "obj")

    async def select_state(_handle):
        return state

    monkeypatch.setattr(agent_tools, "_field_handle", field_handle)
    monkeypatch.setattr(agent_tools, "_select_state", select_state)


def test_select_dropdown_override_is_registered():
    from automation.pipeline.agent_tools import build_tools

    action = build_tools().registry.registry.actions["select_dropdown"]
    assert action.function.__module__ == "automation.pipeline.agent_tools"


async def test_select_dropdown_timeout_is_rescued_by_readback(monkeypatch):
    _patch_select_readback(monkeypatch, ("uk", "United Kingdom"))
    fn, pm = _registered_action("select_dropdown")
    res = await fn(params=pm(index=9, text="United Kingdom"),
                   browser_session=_FakeSelectSession(exc=TimeoutError("no confirmation")))

    assert res.error is None
    assert "read-back confirms" in res.extracted_content
    assert "do NOT set it again" in res.extracted_content


async def test_select_dropdown_mismatch_is_reported_truthfully(monkeypatch):
    _patch_select_readback(monkeypatch, ("au", "Australia"))
    fn, pm = _registered_action("select_dropdown")
    res = await fn(params=pm(index=9, text="United Kingdom"),
                   browser_session=_FakeSelectSession(result={"success": "true"}))

    assert res.error is not None
    assert "did NOT take" in res.error
    assert "Australia" in res.error
    assert "dropdown_options(index=9)" in res.error


async def test_select_dropdown_success_echoes_readback(monkeypatch):
    _patch_select_readback(monkeypatch, ("uk", "United Kingdom"))
    fn, pm = _registered_action("select_dropdown")
    res = await fn(params=pm(index=9, text="United Kingdom"),
                   browser_session=_FakeSelectSession(result={"success": "true"}))

    assert res.error is None
    assert "now reads 'United Kingdom'" in res.extracted_content



# ------------------------------- custom-combobox picks -------------------------------
# Motivating failure (payroll run 20260803_112915): react-select comboboxes have no
# native-<select> path, so the agent hand-rolled open/type/click across batched steps and
# died after 17 steps of dropdown thrashing on "Join with". select_dropdown's non-native
# branch owns the whole transaction; these tests cover its option matcher, the in-page
# JS contract (against a real react-select-like widget), and the decision ladder.

_RS_PAGE = """
<div class="field">
  <label>Frequency</label>
  <div class="rs" id="c1">
    <div class="rs-control">
      <div class="rs-value" id="v1">Monthly</div>
      <input id="react-select-3-input" role="combobox">
    </div>
  </div>
</div>
<div class="field">
  <label>Join with</label>
  <div class="rs" id="c2">
    <div class="rs-control">
      <div class="rs-value" id="v2">Select...</div>
      <input id="react-select-4-input" role="combobox">
    </div>
  </div>
</div>
<script>
  // Faithful to react-select where it matters: the menu mounts on the CONTROL's
  // mousedown, options select on the OPTION's mousedown (never on click), and the
  // menu unmounts on selection.
  document.querySelectorAll('.rs-control').forEach(function (ctl) {
    ctl.addEventListener('mousedown', function () {
      var container = ctl.parentElement;
      if (container.querySelector('.rs-menu')) return;
      var menu = document.createElement('div');
      menu.className = 'rs-menu';
      var instance = ctl.querySelector('input').id.replace('-input', '');
      ['P45', 'P46', 'Existing employee'].forEach(function (t, i) {
        var o = document.createElement('div');
        o.id = instance + '-option-' + i;
        o.textContent = t;
        o.addEventListener('mousedown', function () {
          container.querySelector('.rs-value').textContent = t;
          menu.remove();
          window.__picked = t;
        });
        menu.appendChild(o);
      });
      container.appendChild(menu);
    });
  });
</script>
"""


def _cb_expr(op, input_id, wanted=None):
    import json as _json
    return agent_tools._CB_OPS_JS % {
        "id": _json.dumps(input_id), "op": _json.dumps(op), "wanted": _json.dumps(wanted)}


def _resolve_expr():
    return "(el) => (" + agent_tools._CB_RESOLVE_JS + ").call(el)"


def test_choose_option_exact_beats_partial():
    opts = [{"id": "a", "text": "Existing employee record"},
            {"id": "b", "text": "Existing employee"}]
    assert agent_tools._choose_option(opts, "existing employee")["id"] == "b"


def test_choose_option_unique_partial_matches_padded_cards():
    # react-select option cards pad the label with detail lines (tax code, id, pay).
    opts = [{"id": "a", "text": "AM Aaran Macleod Tax Code : 1257L Cum Employee ID : 5302"},
            {"id": "b", "text": "HS Haiden Stevenson Tax Code : 1257L Cum"}]
    assert agent_tools._choose_option(opts, "Haiden Stevenson")["id"] == "b"


def test_choose_option_ambiguous_or_absent_returns_none():
    opts = [{"id": "a", "text": "Existing employee A"},
            {"id": "b", "text": "Existing employee B"}]
    assert agent_tools._choose_option(opts, "Existing employee") is None
    assert agent_tools._choose_option(opts, "New starter") is None
    assert agent_tools._choose_option([], "anything") is None


async def test_cb_resolve_from_input_placeholder_and_container():
    from playwright.async_api import async_playwright

    from tests.test_heal_promotion import _launch

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content(_RS_PAGE)
        # From the combobox input itself.
        got = await page.eval_on_selector("#react-select-4-input", _resolve_expr())
        assert got == {"input_id": "react-select-4-input"}
        # From the visible placeholder/value div — the label-to-input association the
        # snapshot can't provide (inputs render nameless).
        got = await page.eval_on_selector("#v2", _resolve_expr())
        assert got == {"input_id": "react-select-4-input"}
        # From the widget container.
        got = await page.eval_on_selector("#c1", _resolve_expr())
        assert got == {"input_id": "react-select-3-input"}
        # The root stamp lands on the resolved widget, not the page.
        assert await page.eval_on_selector(
            "#c1", "el => el.hasAttribute('data-ao-cb-root')")


async def test_cb_resolve_refuses_multi_combobox_wrappers():
    from playwright.async_api import async_playwright

    from tests.test_heal_promotion import _launch

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content(_RS_PAGE)
        got = await page.eval_on_selector("body", _resolve_expr())
        assert got["error"] == "ambiguous"
        assert got["count"] == 2
        got = await page.eval_on_selector("label", _resolve_expr())
        # The 'Frequency' label's parent .field contains exactly one combobox — resolves.
        assert got == {"input_id": "react-select-3-input"}


async def test_cb_open_list_pick_verify_against_react_select_semantics():
    """The full transaction the tool drives: open fires the control's mousedown, options
    are read by instance prefix, pick fires the option's mousedown (el.click() would be
    a no-op here), and state shows the committed value with the menu closed."""
    from playwright.async_api import async_playwright

    from tests.test_heal_promotion import _launch

    async with async_playwright() as pw:
        browser = await _launch(pw)
        page = await browser.new_page()
        await page.set_content(_RS_PAGE)
        await page.eval_on_selector("#v2", _resolve_expr())

        assert (await page.evaluate(_cb_expr("open", "react-select-4-input")))["ok"]
        got = await page.evaluate(_cb_expr("options", "react-select-4-input"))
        assert [o["text"] for o in got["options"]] == ["P45", "P46", "Existing employee"]

        chosen = agent_tools._choose_option(got["options"], "existing employee")
        picked = await page.evaluate(
            _cb_expr("pick", "react-select-4-input", chosen["id"]))
        assert picked["clicked"] is True
        assert picked["attrs"]["id"] == "react-select-4-option-2"
        assert await page.evaluate("window.__picked") == "Existing employee"

        state = await page.evaluate(_cb_expr("state", "react-select-4-input"))
        assert state["menu_open"] is False
        assert "Existing employee" in state["display"]


def _scripted_combobox(monkeypatch, resolve, ops):
    """Wire _combobox_select's collaborators to canned responses. `ops` maps op-name to
    a value or a list consumed in order."""
    async def call_on_field(_handle, _decl, args=None):
        return resolve
    monkeypatch.setattr(agent_tools, "_call_on_field", call_on_field)

    async def eval_js(_session, expr, **_kw):
        for op in ("open", "options", "pick", "state"):
            if f'"{op}"' in expr.partition("var input")[0]:
                v = ops.get(op)
                if isinstance(v, list):
                    return v.pop(0) if len(v) > 1 else v[0]
                return v
        return {"ok": True}  # the focus() expression
    monkeypatch.setattr(agent_tools, "_eval_js", eval_js)


class _CbNode:
    tag_name = "div"
    backend_node_id = 1
    attributes: dict = {}


class _CbSession:
    async def get_element_by_index(self, _index):
        return _CbNode()

    async def get_or_create_cdp_session(self):
        raise RuntimeError("no cdp in unit test")  # filter typing becomes a no-op


def _cb_action():
    fn, pm = _registered_action("select_dropdown")

    async def run(monkeypatch, text="Existing employee"):
        async def field_handle(_s, _n):
            return ("cdp", "obj")
        monkeypatch.setattr(agent_tools, "_field_handle", field_handle)
        return await fn(params=pm(index=7, text=text), browser_session=_CbSession())
    return run


async def test_combobox_select_success_receipt_and_replayable_metadata(monkeypatch):
    opts = {"options": [{"id": "react-select-4-option-2", "text": "Existing employee"}]}
    _scripted_combobox(
        monkeypatch, {"input_id": "react-select-4-input"},
        {"open": {"ok": True}, "options": opts,
         "pick": {"clicked": True, "tag": "div",
                  "attrs": {"id": "react-select-4-option-2", "role": "option"}},
         "state": {"menu_open": False, "display": "Join with Existing employee"}})
    res = await _cb_action()(monkeypatch)
    assert res.error is None
    assert "Selected 'Existing employee'" in res.extracted_content
    assert "Do NOT set it again" in res.extracted_content
    el = res.metadata["interacted_element"]
    assert el["attributes"]["id"] == "react-select-4-option-2"
    assert el["ax_name"] == "Existing employee"


async def test_combobox_select_lists_real_options_on_a_miss(monkeypatch):
    # THE receipt that would have saved the live run: the agent hunted 'Aaran Macleod'
    # while the menu only ever offered P45/P46/Existing employee.
    opts = {"options": [{"id": "o0", "text": "P45"}, {"id": "o1", "text": "P46"},
                        {"id": "o2", "text": "Existing employee"}]}
    _scripted_combobox(
        monkeypatch, {"input_id": "react-select-4-input"},
        {"open": {"ok": True}, "options": opts})
    res = await _cb_action()(monkeypatch, text="Aaran Macleod")
    assert res.error is not None
    assert "no such option" in res.error
    assert "'P45', 'P46', 'Existing employee'" in res.error
    assert "Do NOT hunt the page for 'Aaran Macleod'" in res.error


async def test_combobox_select_read_back_mismatch_fails_honestly(monkeypatch):
    opts = {"options": [{"id": "o2", "text": "Existing employee"}]}
    _scripted_combobox(
        monkeypatch, {"input_id": "react-select-4-input"},
        {"open": {"ok": True}, "options": opts,
         "pick": {"clicked": True, "tag": "div", "attrs": {"id": "o2"}},
         "state": {"menu_open": False, "display": "Select..."}})
    res = await _cb_action()(monkeypatch)
    assert res.error is not None
    assert "does not show it" in res.error
    assert "Do not report it as set" in res.error


async def test_combobox_select_ambiguous_container_guides_the_agent(monkeypatch):
    _scripted_combobox(monkeypatch, {"error": "ambiguous", "count": 2}, {})
    res = await _cb_action()(monkeypatch)
    assert res.error is not None
    assert "2 different comboboxes" in res.error
    assert "find_by_text" in res.error


async def test_combobox_select_non_combobox_element_says_so(monkeypatch):
    _scripted_combobox(monkeypatch, {"error": "none"}, {})
    res = await _cb_action()(monkeypatch)
    assert res.error is not None
    assert "no combobox input" in res.error
