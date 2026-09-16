"""copy_text / paste_text: delivering ONE value into a widget that splits it across boxes.

Motivating failure (run 20260825_090938, subtask 6e8c9bb7ee56a6aa): the OTP dialog has six
one-character inputs, so the agent typed six 1-character fills and the code it actually
used — "502956" — never appeared as a step value at all. The provenance guard flags typed
values by matching them against the run's findings on whole tokens and skips anything under
3 characters, so nothing was flagged, `runtime_values` came out empty, and the commit was
refused: the segment re-authored with the LLM every run at 107k tokens and 143 seconds
while its neighbours replayed in 2-6s at zero tokens.

Pasting delivers the whole value in one action, which puts it back in the recording as ONE
step value — bindable by the machinery that already exists. The rung ladder and its
read-back are what make that honest: box 1 of a six-box widget reads "5" after a PERFECT
paste, so the ordinary single-field check would call a correct paste a failure.

Fixture note: the six inputs and their aria-labels come from the real recording's
state_message, not from a hand-written belief about the markup.
"""
import json
from types import SimpleNamespace

from test_compile_coverage import _item, _write

from automation.pipeline import agent_tools
from automation.pipeline.runner import restore_result_metadata
from automation.pipeline.script_compile import (GROUP_CLEAR_JS, GROUP_VALUES_JS,
                                                PASTE_EVENT_JS,
                                                compile_recording, paste_group_empty,
                                                paste_took)
from automation.skills import codegen

OTP = "502956"
# aria-labels exactly as the app renders them (library/6e8c9bb7ee56a6aa.recording.json).
BOX_LABELS = [f"Please enter OTP character {n}" for n in range(1, 7)]


def _tool(name):
    return agent_tools.build_tools().registry.registry.actions[name]


class _Widget:
    """A six-box OTP input that accepts delivery on exactly ONE rung."""

    def __init__(self, accepts):
        self.accepts = accepts          # "paste event" | "browser paste" | "keystrokes" | None
        self.boxes = [""] * 6
        self.rungs_tried = []

    def deliver(self, rung, text):
        self.rungs_tried.append(rung)
        if rung == self.accepts:
            for i, char in enumerate(text[:6]):
                self.boxes[i] = char
        elif rung == "browser paste":
            # Measured in chromium: Chrome's paste command truncates into a maxlength=1
            # box. The ladder must clear that before the next rung, which is what
            # rungs_tried + the empty boxes below prove.
            self.boxes[0] = text[:1]

    def clear(self):
        self.rungs_tried.append("clear")
        self.boxes = [""] * 6

    def reading(self):
        return {"own": self.boxes[0], "group": list(self.boxes)}


class _Event:
    def __init__(self, on_await):
        self._on = on_await

    def __await__(self):
        async def _go():
            self._on()
        return _go().__await__()

    async def event_result(self, **_kw):
        return None


class _Session:
    def __init__(self, widget, node=None):
        self.widget = widget
        self.node = node if node is not None else object()
        self.event_bus = self

    async def get_element_by_index(self, _index):
        return self.node

    def dispatch(self, event):
        keys = getattr(event, "keys", "")
        return _Event(lambda: self.widget.deliver("keystrokes", keys))


def _wire(monkeypatch, widget, *, connected=True, clipboard=True):
    """Patch the field primitives so the tool talks to `widget` instead of a live page."""
    async def call_on_field(_handle, declaration, args=None):
        if PASTE_EVENT_JS in declaration:
            widget.deliver("paste event", (args or [""])[0])
            return "handled"
        if GROUP_VALUES_JS in declaration:
            return widget.reading()
        if GROUP_CLEAR_JS in declaration:
            widget.clear()
            return 6
        raise AssertionError(f"unexpected declaration: {declaration[:60]}")

    async def native_paste(_handle):
        widget.deliver("browser paste", OTP)
        return True

    async def field_handle(_session, _node):
        return ("cdp", "object-id")

    monkeypatch.setattr(agent_tools, "_call_on_field", call_on_field)
    monkeypatch.setattr(agent_tools, "_native_paste", native_paste)
    monkeypatch.setattr(agent_tools, "_field_handle", field_handle)
    monkeypatch.setattr(agent_tools, "_field_connected",
                        lambda _h: _async(connected))
    monkeypatch.setattr(agent_tools, "_clipboard_write",
                        lambda _s, _t: _async(clipboard))
    monkeypatch.setattr(agent_tools, "_is_dropdown_filter", lambda _n: False)
    monkeypatch.setattr(agent_tools, "_captured_element",
                        lambda _n, _l: {"node_name": "input",
                                        "attributes": {"aria-label": BOX_LABELS[0]},
                                        "ax_name": BOX_LABELS[0]})


def _async(value):
    async def _go():
        return value
    return _go()


# ------------------------------- the read-back rule -------------------------------


def test_group_readback_accepts_a_value_spread_across_the_boxes():
    # THE point of the group reader: box 1 shows "5" after a perfect paste.
    reading = {"own": "5", "group": list(OTP)}
    assert paste_took(OTP, reading) == (True, OTP)


def test_group_readback_accepts_a_single_field_that_took_it_whole():
    assert paste_took(OTP, {"own": OTP, "group": [OTP]}) == (True, OTP)


def test_group_readback_rejects_a_partial_landing():
    took, shows = paste_took(OTP, {"own": "5", "group": ["5", "", "", "", "", ""]})
    assert took is False and shows == "5"


def test_group_empty_is_the_only_safe_state_for_the_next_rung():
    assert paste_group_empty({"own": "", "group": [""] * 6}) is True
    # A partial landing must NOT be stacked on top of by the next rung.
    assert paste_group_empty({"own": "5", "group": ["5"] + [""] * 5}) is False


# ------------------------------- the rung ladder -------------------------------


async def test_paste_takes_on_the_synthetic_event_rung(monkeypatch):
    widget = _Widget(accepts="paste event")
    _wire(monkeypatch, widget)
    res = await _tool("paste_text").function(index=5, text=OTP,
                                             browser_session=_Session(widget))

    assert res.error is None
    assert "paste event" in res.extracted_content
    assert widget.boxes == list(OTP)
    assert widget.rungs_tried == ["paste event"]      # no rung tried after it took
    assert res.metadata["paste"]["value"] == OTP


async def test_paste_falls_through_to_keystrokes(monkeypatch):
    # No paste handler, but the widget advances focus box-to-box as you type. Keystrokes
    # come BEFORE the browser paste command: they cannot truncate, and they need neither a
    # clipboard permission nor a secure origin.
    widget = _Widget(accepts="keystrokes")
    _wire(monkeypatch, widget)
    res = await _tool("paste_text").function(index=5, text=OTP,
                                             browser_session=_Session(widget))

    assert res.error is None
    assert "keystrokes" in res.extracted_content
    assert widget.boxes == list(OTP)
    assert widget.rungs_tried == ["paste event", "clear", "keystrokes"]


async def test_paste_falls_through_to_the_browser_paste_command(monkeypatch):
    # The last rung: a widget that distributes on a REAL paste and ignores a synthetic one
    # (an isTrusted check) and cannot be typed into.
    widget = _Widget(accepts="browser paste")
    _wire(monkeypatch, widget)
    res = await _tool("paste_text").function(index=5, text=OTP,
                                             browser_session=_Session(widget))

    assert res.error is None
    assert "browser paste" in res.extracted_content
    assert widget.boxes == list(OTP)
    assert widget.rungs_tried == ["paste event", "clear", "keystrokes", "clear",
                                  "browser paste"]


async def test_a_truncating_rung_is_cleared_before_the_next_one(monkeypatch):
    # The poisoning case, measured live: a browser paste that lands only "5" must not be
    # left for the next rung to append to.
    widget = _Widget(accepts="keystrokes")
    _wire(monkeypatch, widget)
    await _tool("paste_text").function(index=5, text=OTP,
                                       browser_session=_Session(widget))
    assert widget.boxes == list(OTP)


async def test_paste_that_lands_on_no_rung_refuses_on_the_error_channel(monkeypatch):
    widget = _Widget(accepts=None)
    _wire(monkeypatch, widget)
    res = await _tool("paste_text").function(index=5, text=OTP,
                                             browser_session=_Session(widget))

    # ERROR channel so multi_act stops: a queued "Proceed Securely" must never fire on a
    # field that did not take the code.
    assert res.error and "did NOT land" in res.error
    assert res.metadata["no_fill"] is True          # phantom-action rule: compiles to nothing


async def test_paste_refuses_a_detached_element(monkeypatch):
    widget = _Widget(accepts="paste event")
    _wire(monkeypatch, widget, connected=False)
    res = await _tool("paste_text").function(index=5, text=OTP,
                                             browser_session=_Session(widget))

    assert res.error and "left the page" in res.error
    assert widget.rungs_tried == []


async def test_paste_with_no_text_delivers_what_copy_text_captured(monkeypatch):
    widget = _Widget(accepts="paste event")
    _wire(monkeypatch, widget)
    agent_tools._CLIPBOARD.update(value=OTP, label="otp")
    res = await _tool("paste_text").function(index=5, browser_session=_Session(widget))

    assert res.error is None and widget.boxes == list(OTP)


async def test_paste_with_nothing_to_deliver_says_so(monkeypatch):
    widget = _Widget(accepts="paste event")
    _wire(monkeypatch, widget)
    agent_tools._CLIPBOARD.update(value="", label="")
    res = await _tool("paste_text").function(index=5, browser_session=_Session(widget))

    assert res.error and "nothing to paste" in res.error


# ------------------------------- copy_text -------------------------------


async def test_copy_text_captures_on_the_extract_channel(monkeypatch):
    widget = _Widget(accepts=None)
    _wire(monkeypatch, widget)
    monkeypatch.setattr(agent_tools, "_field_value", lambda _h: _async(OTP))
    res = await _tool("copy_text").function(index=9, label="OTP",
                                            browser_session=_Session(widget))

    assert res.error is None
    # The SAME metadata shape extract_data stamps, so the value joins run_values/values.json
    # and a later segment's binding can resolve against it with no new plumbing.
    assert res.metadata["extract"] == {"label": "otp", "value": OTP, "query": "index 9",
                                       "interacted_element": res.metadata["extract"]["interacted_element"]}
    assert agent_tools._CLIPBOARD["value"] == OTP


async def test_copy_text_survives_a_refused_clipboard_write(monkeypatch):
    widget = _Widget(accepts=None)
    _wire(monkeypatch, widget, clipboard=False)
    monkeypatch.setattr(agent_tools, "_field_value", lambda _h: _async(OTP))
    res = await _tool("copy_text").function(index=9, label="otp",
                                            browser_session=_Session(widget))

    # The clipboard is a convenience, never a gate: the capture still succeeded.
    assert res.error is None
    assert res.metadata["extract"]["value"] == OTP
    assert "clipboard refused" in res.extracted_content


async def test_copy_text_with_nothing_readable_records_no_metadata(monkeypatch):
    widget = _Widget(accepts=None)
    _wire(monkeypatch, widget)
    monkeypatch.setattr(agent_tools, "_field_value", lambda _h: _async(""))
    res = await _tool("copy_text").function(index=9, label="otp",
                                            browser_session=_Session(widget))

    # A valueless copy must compile to NOTHING, not to a step that fails every replay.
    assert res.metadata is None
    assert "NOTHING was copied" in res.extracted_content


# ------------------------------- compile -------------------------------


def _element(label):
    return {"node_name": "input", "attributes": {"aria-label": label}, "ax_name": label}


def test_paste_compiles_to_a_step_carrying_the_whole_value(tmp_path):
    history = [_item({"paste_text": {"index": 5, "text": OTP}},
                     result=[{"metadata": {"paste": {"value": OTP, "rung": "paste event"},
                                           "interacted_element": _element(BOX_LABELS[0])}}])]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)

    assert len(steps) == 1
    assert steps[0]["action"] == "paste"
    assert steps[0]["value"] == OTP           # ONE bindable value, not six characters
    assert 'css=[aria-label="Please enter OTP character 1"]' in steps[0]["selectors"]


def test_a_refused_paste_compiles_to_nothing(tmp_path):
    history = [_item({"paste_text": {"index": 5, "text": OTP}},
                     result=[{"metadata": {"no_fill": True}}])]
    assert compile_recording(_write(tmp_path, history), emit_start_goto=False) == []


def test_copy_text_compiles_to_a_copy_step(tmp_path):
    history = [_item({"copy_text": {"index": 9, "label": "otp"}},
                     result=[{"metadata": {"extract": {
                         "label": "otp", "value": OTP, "query": "index 9",
                         "interacted_element": _element("One time password")}}}])]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)

    assert len(steps) == 1
    # `copy` is an extract that ALSO seeds the clipboard: replay re-reads the value fresh,
    # so the authoring run's code is provenance only.
    assert steps[0]["action"] == "copy" and steps[0]["label"] == "otp"


# ------------------------------- codegen -------------------------------


def test_paste_and_copy_transpile_to_api_verbs():
    steps = [
        {"action": "copy", "selectors": ['css=[aria-label="One time password"]'],
         "label": "otp"},
        {"action": "paste", "selectors": [f'css=[aria-label="{BOX_LABELS[0]}"]'],
         "value": "{{bound_1}}"},
    ]
    code, anchors = codegen.transpile("sid", steps, params={"bound_1": OTP})

    assert "await api.copy(" in code
    assert "await api.paste(" in code
    # The paste CALL takes the bound name; the authoring literal appears only as the
    # signature default (base.py refuses to instantiate a bound entry without resolving
    # every binding fresh, so a stale default can never replay).
    assert "await api.paste('please-enter-otp-character-1', bound_1)" in code
    assert OTP not in code.split("):", 1)[1]
    assert not codegen.lint_code(code)        # stays inside the api.* whitelist
    assert len(anchors) == 2


# ------------------------------- the dropped-click hole -------------------------------


def test_a_click_recovers_its_element_from_the_tool_stamp(tmp_path):
    # browser-use fills state.interacted_element from the snapshot it takes AFTER the
    # action, so the external-link click that opened the OTP tab recorded nulls and compile
    # dropped the one load-bearing click of that segment. Our click override now stamps the
    # PRE-click element; compile reads it.
    history = [_item({"click": {"index": 782}},
                     result=[{"metadata": {"interacted_element": {
                         "node_name": "a",
                         "attributes": {"title": "Open payroll review request as client"},
                         "ax_name": "Open payroll review request as client"}}}])]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)

    assert len(steps) == 1 and steps[0]["action"] == "click"
    assert steps[0]["expect_text"] == "Open payroll review request as client"


def test_a_click_with_no_identity_at_all_is_dropped_but_never_silently(tmp_path, caplog):
    history = [_item({"click": {"index": 782}}, result=[{"metadata": {}}])]
    with caplog.at_level("WARNING"):
        steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)

    assert steps == []
    assert "no element identity was captured" in caplog.text


def test_a_field_pasted_twice_keeps_the_last_value(tmp_path):
    # Same rule a repeated fill follows: the last write is the one the form kept.
    box = _element(BOX_LABELS[0])
    history = [
        _item({"paste_text": {"index": 5, "text": "111111"}},
              result=[{"metadata": {"paste": {"value": "111111", "rung": "paste event"},
                                    "interacted_element": box}}]),
        _item({"paste_text": {"index": 5, "text": OTP}},
              result=[{"metadata": {"paste": {"value": OTP, "rung": "paste event"},
                                    "interacted_element": box}}]),
    ]
    steps = compile_recording(_write(tmp_path, history), emit_start_goto=False)

    assert len(steps) == 1 and steps[0]["value"] == OTP


# ------------------------- the recording round trip -------------------------
# browser-use's save_history serializes results WITHOUT their `metadata` field, and
# metadata is where BOTH new tools put everything compile needs (the pasted value, the
# captured label, the element identity). runner.restore_result_metadata exists for exactly
# this — it was written when every find_by_text click was compiling to nothing — and these
# two tests are the proof that copy_text/paste_text ride the same pass.


def _saved_without_metadata(tmp_path, actions_and_results):
    rec = tmp_path / "rec.json"
    rec.write_text(json.dumps({"history": [
        {"state": {"url": "http://app/x", "interacted_element": []},
         "model_output": {"action": [action]},
         "result": [{"extracted_content": res.extracted_content or ""}]}
        for action, res in actions_and_results
    ]}))
    return rec


async def _record_a_copy_and_a_paste(monkeypatch):
    widget = _Widget(accepts="paste event")
    _wire(monkeypatch, widget)
    monkeypatch.setattr(agent_tools, "_field_value", lambda _h: _async(OTP))
    session = _Session(widget)
    copied = await _tool("copy_text").function(index=9, label="otp",
                                               browser_session=session)
    pasted = await _tool("paste_text").function(index=5, text=OTP,
                                                browser_session=session)
    return copied, pasted


async def test_both_tools_reach_the_compiler_through_the_metadata_restore(
        monkeypatch, tmp_path):
    copied, pasted = await _record_a_copy_and_a_paste(monkeypatch)
    rec = _saved_without_metadata(tmp_path, [
        ({"copy_text": {"index": 9, "label": "otp"}}, copied),
        ({"paste_text": {"index": 5, "text": OTP}}, pasted),
    ])
    in_memory = SimpleNamespace(history=[SimpleNamespace(result=[copied]),
                                         SimpleNamespace(result=[pasted])])

    assert restore_result_metadata(in_memory, rec) is True
    steps = compile_recording(rec, emit_start_goto=False)

    assert [s["action"] for s in steps] == ["copy", "paste"]
    assert steps[0]["label"] == "otp"
    assert steps[1]["value"] == OTP            # ONE bindable value, not six characters
    assert steps[0]["selectors"] and steps[1]["selectors"]


async def test_without_the_restore_pass_both_tools_compile_to_nothing(
        monkeypatch, tmp_path):
    # The failure this guards: metadata is the ONLY channel carrying the pasted value and
    # the element, so a recording saved without it yields an empty script — silently.
    copied, pasted = await _record_a_copy_and_a_paste(monkeypatch)
    rec = _saved_without_metadata(tmp_path, [
        ({"copy_text": {"index": 9, "label": "otp"}}, copied),
        ({"paste_text": {"index": 5, "text": OTP}}, pasted),
    ])

    assert compile_recording(rec, emit_start_goto=False) == []


# --------------- segment-scoped clipboard + no leaked CDP session (2026-08-26) ---------------


def test_registering_a_segment_clears_the_copy_buffer():
    """`paste_text(index)` with no `text` falls back to whatever copy_text last captured, so
    a value left over from an EARLIER subtask would be pasted silently instead of the step
    failing. An OTP is the case that matters: captured in one slice, consumed in the next,
    stale by the one after. set_live_network is the per-segment reset point the rest of this
    module's state already uses."""
    agent_tools._CLIPBOARD.update(value="444919", label="otp")

    agent_tools.set_live_network(object())
    try:
        assert agent_tools._CLIPBOARD == {"value": "", "label": ""}
    finally:
        agent_tools.clear_live_network()


async def test_the_native_paste_rung_detaches_its_cdp_session():
    """Rung 3 opens a CDP session per call; without a detach it stays attached to the page
    for the life of the context. Driven through the real ladder with a read-back that never
    matches, so rungs 1 and 2 fall through and rung 3 actually runs."""
    from automation.pipeline import script_compile as sc

    detached, sent = [], []

    class _Cdp:
        async def send(self, method, params=None):
            sent.append(method)
            return {}

        async def detach(self):
            detached.append(True)

    class _Keyboard:
        async def type(self, _text, delay=None):
            return None

    class _Page:
        keyboard = _Keyboard()
        context = type("_Ctx", (), {
            "new_cdp_session": staticmethod(lambda _page: _cdp_coro())})()

        async def wait_for_timeout(self, _ms):
            return None

        async def evaluate(self, _js, *_a):
            return None

    async def _cdp_coro():
        return _Cdp()

    class _Loc:
        async def evaluate(self, _js, *_a):
            return ""            # never matches -> every rung falls through

    took, _shows = await sc._paste_into(_Page(), _Loc(), "492564")

    assert took is False                      # the ladder genuinely exhausted
    assert sent == ["Input.dispatchKeyEvent"] * 2
    assert detached == [True], "rung 3 must detach its session"
