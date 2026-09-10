"""Twelve identical pencils: naming the row is the only way to pick one.

Run 20260903_102354_153542 subtask 4 ("click the pencil icon at the right end of the Feb-27
row"). The Pay Forecast grid renders one nameless "Net to gross" icon per row.

    find_by_text('Net to gross', click_first=True) -> 24 match(es) — ambiguous, nothing clicked

The refusal is right, and its receipt says "pick the right index from the list above and
click(index) NOW". But the list was twenty-four rows of `<i> text='Net to gross'` — every
line identical. The advice named a route the agent could not walk. It guessed an index off
list_actions (hit a `<td>`, no popup), then wrote JavaScript into `navigate` twice and lost
the page. Nineteen steps; three earlier authorings of the same slice took three.

Two fixes, and the second is the one with the measured precedent:

1. The candidate listing carries each match's ROW, using the same `_row_context_label` built
   in August so an anonymous checkbox click could name the employee it ticked. That makes
   the receipt's existing advice executable for the first time.

2. `near_text` scopes the search to the row holding that text — no index hop at all. This is
   `select_dropdown(near_text='From')` again (2026-08-27): both of the "help it FIND the
   control" attempts before that were built, shipped and NEVER CALLED, because each ended in
   "now go call something with an index" — the hop the agent skips. What worked was removing
   the index. A row is the same shape of problem as a label beside a combobox.

Scoping FAILS CLOSED: when near_text matches no candidate's row, nothing is clicked and the
receipt says so. Clicking the wrong row is the failure this whole area exists to prevent —
run 20260817_124339 paid AARAN instead of Harris Duncan off a stale positional index.
"""
import pytest

from automation.pipeline import agent_tools
from tests.test_agent_tools import (_FakeBrowserSession, _FakeClickSession,
                                    _FakeDomNode, _registered_action,
                                    _stamped_dialog)


def _grid(*rows):
    """One nameless "Net to gross" icon per row, exactly like the Pay Forecast grid.
    Returns (session, {index: row_text})."""
    nodes, row_of = {}, {}
    for i, row in enumerate(rows):
        idx = 20400 + i * 5
        node = _FakeDomNode("", attributes={"aria-label": "Net to gross"})
        node.node_name = "I"
        node.tag_name = "i"   # _FakeDomNode carries only node_name; _line reads tag_name
        nodes[idx] = node
        row_of[idx] = row
    return _FakeBrowserSession(nodes), row_of


def _stub_rows(monkeypatch, row_of):
    """Stand in for the live CDP row probe (_row_context_label needs a real handle)."""
    async def fake(_session, node):
        for idx, row in row_of.items():
            if _session._nodes.get(idx) is node:
                return row
        return None
    monkeypatch.setattr(agent_tools, "_row_context_label", fake)


ROWS = ("Dec-26 £5,000.00 ", "Jan-27 £5,000.00 ", "Feb-27 £4,131.20 ")


# ------------------------------- 1. the listing names the row -------------------------------


async def test_an_ambiguous_listing_names_each_candidates_row(monkeypatch):
    session, row_of = _grid(*ROWS)
    _stub_rows(monkeypatch, row_of)
    fn, pm = _registered_action("find_by_text")

    res = await fn(params=pm(text="Net to gross", click_first=True), browser_session=session)

    body = res.error or ""
    assert "3 match(es)" in body
    # Without this the three lines are byte-identical and the receipt's "pick the right
    # index" is unfollowable.
    assert "Dec-26" in body and "Jan-27" in body and "Feb-27" in body


async def test_the_row_text_is_stripped_of_fluent_glyphs(monkeypatch):
    """Fluent draws its icons as literal Private Use Area text nodes (the 2026-08-21 pencil
    trap), which would otherwise land in the receipt as mojibake."""
    session, row_of = _grid(*ROWS)
    _stub_rows(monkeypatch, row_of)
    fn, pm = _registered_action("find_by_text")

    res = await fn(params=pm(text="Net to gross", click_first=True), browser_session=session)

    assert "" not in (res.error or "")


async def test_an_unreadable_row_just_leaves_the_line_as_it_was(monkeypatch):
    """Garnish, never a gate: the probe is best-effort and the listing must still render."""
    async def fake(_session, _node):
        raise RuntimeError("no handle")
    monkeypatch.setattr(agent_tools, "_row_context_label", fake)
    session, _ = _grid(*ROWS)
    fn, pm = _registered_action("find_by_text")

    res = await fn(params=pm(text="Net to gross", click_first=True), browser_session=session)

    assert "3 match(es)" in (res.error or "")


# ------------------------------- 2. near_text scopes the search -------------------------------


async def test_near_text_narrows_twelve_identical_icons_to_one(monkeypatch):
    session, row_of = _grid(*ROWS)
    _stub_rows(monkeypatch, row_of)
    fn, pm = _registered_action("find_by_text")

    res = await fn(params=pm(text="Net to gross", near_text="Feb-27"),
                   browser_session=session)

    body = res.extracted_content or ""
    assert "1 match(es)" in body            # twelve identical icons down to one
    assert "index=20410" in body            # and it is the Feb-27 one
    assert "index=20400" not in body and "index=20405" not in body


async def test_near_text_leaves_click_first_a_single_match_to_click(monkeypatch):
    """The point of scoping is that click_first can then act — no ambiguity refusal, no
    index for the agent to pick. (The click itself needs a live CDP session, so this
    asserts the tool stopped refusing, which is the behaviour scoping adds.)"""
    session, row_of = _grid(*ROWS)
    _stub_rows(monkeypatch, row_of)
    fn, pm = _registered_action("find_by_text")

    res = await fn(params=pm(text="Net to gross", near_text="Feb-27", click_first=True),
                   browser_session=session)

    assert "ambiguous" not in (res.error or "").lower()
    assert "match(es)" not in (res.error or "")     # it went to the click, not the listing


async def test_near_text_that_matches_no_row_clicks_nothing(monkeypatch):
    """Fails CLOSED. A scope that silently fell back to "all candidates" would click the
    wrong row, which is the exact failure the row machinery exists to prevent."""
    session, row_of = _grid(*ROWS)
    _stub_rows(monkeypatch, row_of)
    fn, pm = _registered_action("find_by_text")

    res = await fn(params=pm(text="Net to gross", near_text="Aug-99", click_first=True),
                   browser_session=session)

    assert res.error
    assert "aug-99" in res.error.lower()
    assert (res.metadata or {}).get("no_click") is True


async def test_near_text_is_matched_by_token_not_substring(monkeypatch):
    """Same rule as `text` itself: punctuation ignored, case-insensitive."""
    session, row_of = _grid(*ROWS)
    _stub_rows(monkeypatch, row_of)
    fn, pm = _registered_action("find_by_text")

    res = await fn(params=pm(text="Net to gross", near_text="feb 27"),
                   browser_session=session)

    body = res.extracted_content or ""
    assert "1 match(es)" in body and "index=20410" in body


async def test_near_text_still_reports_several_when_the_row_is_shared(monkeypatch):
    """Scoping is not a promise of uniqueness — two controls in one row stay ambiguous, and
    the listing then has to distinguish them some other way."""
    session, row_of = _grid("Feb-27 gross", "Feb-27 net")
    _stub_rows(monkeypatch, row_of)
    fn, pm = _registered_action("find_by_text")

    res = await fn(params=pm(text="Net to gross", near_text="Feb-27", click_first=True),
                   browser_session=session)

    assert res.error and "2 match(es)" in res.error


async def test_no_near_text_behaves_exactly_as_before(monkeypatch):
    session, row_of = _grid(*ROWS)
    _stub_rows(monkeypatch, row_of)
    fn, pm = _registered_action("find_by_text")

    res = await fn(params=pm(text="Net to gross", click_first=True), browser_session=session)

    assert res.error and "ambiguous" not in res.error.lower() or "3 match(es)" in res.error


def test_the_tool_description_advertises_near_text():
    """A capability is inert until the text describing the tool stops omitting it — the
    2026-08-27 rule, learned when list_actions gained field decoding and nothing changed."""
    from automation.pipeline.agent_tools import build_tools

    desc = build_tools().registry.registry.actions["find_by_text"].description

    assert "near_text" in desc


@pytest.mark.parametrize("row,tokens,expected", [
    ("Feb-27 £4,131.20", ["feb", "27"], True),
    ("Feb-27 £4,131.20", ["mar", "27"], False),
    ("FEB-27", ["feb", "27"], True),
    ("Feb-27 ", ["feb", "27"], True),
    (None, ["feb", "27"], False),
    ("", ["feb"], False),
])
def test_the_row_predicate(row, tokens, expected):
    assert agent_tools._row_matches(row, tokens) is expected


# ------------------------------- the three channels -------------------------------
#
# A capability is inert until every channel that describes the tool stops omitting it —
# the 2026-08-27 rule, learned when list_actions gained field decoding and nothing changed
# because prompts.py still framed it as icon-only. The action description is asserted above;
# these are the other two.


async def test_the_ambiguity_refusal_offers_near_text(monkeypatch):
    """The just-in-time channel. This receipt is what the agent reads at the exact moment
    it is stuck with N identical candidates — the moment run 20260903_102354 improvised."""
    session, row_of = _grid(*ROWS)
    _stub_rows(monkeypatch, row_of)
    fn, pm = _registered_action("find_by_text")

    res = await fn(params=pm(text="Net to gross", click_first=True), browser_session=session)

    assert "near_text" in (res.error or "")


def test_the_icon_button_prompt_section_offers_near_text():
    """That section opens by naming this exact problem — "several in a toolbar or table row
    appear to you as identical" — and until now offered no way to pick between them."""
    from automation.pipeline.prompts import SPEED_OPTIMIZATION_PROMPT

    # Bounded to THIS section: the next box-rule ends it. Without that the assertion is
    # satisfied by DROPDOWN / COMBOBOX PICKS further down, which mentions near_text for
    # select_dropdown and would pass while this section stayed silent.
    section = (SPEED_OPTIMIZATION_PROMPT
               .split("NAMED ICON BUTTONS")[1]
               .split("DIALOG IDENTITY CHECK")[0])

    assert "near_text" in section


# ------------------------------- scoping must not eat the fallback -------------------------------
#
# Introduced with near_text on 2026-09-03 and caught live the same day, run
# 20260903_122807_063006 subtask 16 steps 5 and 6:
#
#   find_by_text('Bulk upload FPS', near_text='Save & Next'): 0 element(s) carry
#   'Bulk upload FPS', but NONE of them sits in a row containing 'Save & Next' —
#   nothing was clicked. Rows seen: none readable.
#
# Nonsense: if ZERO elements carry the text, rows have nothing to do with it. Worse, the
# branch returned before the 0-match path ever ran — so passing near_text silently disabled
# the raw-DOM search and its scroll hunt, which is the only way to reach a 0-size or
# off-screen control. Scoping may only narrow candidates that exist.
#
# Falling through is safe: the raw path refuses ambiguity itself (RAW_FIND_JS returns
# {error:'ambiguous'} for >1), and the wrong-row hazard needs several same-named controls.


async def test_near_text_with_no_candidates_falls_through_to_the_raw_dom_search(monkeypatch):
    seen = []

    async def eval_js(_session, expr, **_kw):
        if "open:" in expr and "scrollTop" not in expr:
            return {"open": 0}
        if "scrollTop" in expr:
            return 0
        seen.append("raw")
        return {"count": 0}

    monkeypatch.setattr(agent_tools, "_eval_js", eval_js)
    fn, pm = _registered_action("find_by_text")

    res = await fn(params=pm(text="Bulk upload FPS", near_text="Save & Next"),
                   browser_session=_FakeBrowserSession({}))

    assert seen, "the raw-DOM search must still run when nothing matched the snapshot"
    body = (res.error or "") + (res.extracted_content or "")
    assert "sits in a row" not in body, body


async def test_the_row_refusal_still_fires_when_candidates_DO_exist(monkeypatch):
    """The guard itself is unchanged — it just may not speak for a miss it did not cause."""
    session, row_of = _grid(*ROWS)
    _stub_rows(monkeypatch, row_of)
    fn, pm = _registered_action("find_by_text")

    res = await fn(params=pm(text="Net to gross", near_text="Aug-99", click_first=True),
                   browser_session=session)

    assert res.error and "sits in a row" in res.error
    assert "3 element(s)" in res.error


# ------------------------------- scoping a LONE match -------------------------------
#
# Run 20260904_150433_366684 subtask 4, step 4 — the FPS submit button, five employees
# already ticked:
#
#   find_by_text('FPS', near_text='Brooklyn Millar'): 1 element(s) carry 'FPS', but NONE
#   of them sits in a row containing 'Brooklyn Millar' — nothing was clicked. Rows seen:
#   none readable.
#
# The ONE candidate was the right button. It lives in the panel FOOTER, inside no
# [role=row]/tr/li/List-cell at all, so its row text is empty and _row_matches fails
# closed. Refused, the agent fell back to guessing indexes off the (truncated) element
# listing and clicked the ms-Overlay, then the panel's close X — which closed the panel
# and discarded all five ticks.
#
# The agent supplied near_text because it is TAUGHT to: the slice says "click FPS at the
# bottom of the employee list", and both the prompt and this tool's own ambiguity refusal
# advertise near_text as the way to narrow. A redundant scope must not be punished.
#
# Scoping picks one control out of SEVERAL identically-named ones. With a single candidate
# there is nothing to pick, so scoping can only ever destroy a correct match — the same
# reasoning the 0-match fall-through above already runs on ("the wrong-row hazard needs
# several same-named controls to exist in the first place"), applied one case further.


def _panel_footer_button():
    """The FPS panel's submit button as run 20260904_150433 recorded it.

    `ms-Button--primary`, accessible name = the Fluent icon glyph + "FPS", NO id and NO
    aria-label, and it sits in the panel footer — in no row, so the row probe reads ''."""
    node = _FakeDomNode("\uf548\nFPS",
                        attributes={"class": "ms-Button ms-Button--primary btnMedium-1203"})
    node.node_name = "BUTTON"
    node.tag_name = "button"
    return node


def _rowless(monkeypatch):
    async def no_row(_session, _node):
        return None
    monkeypatch.setattr(agent_tools, "_row_context_label", no_row)


async def test_a_lone_match_is_clicked_not_scoped_away(monkeypatch):
    _rowless(monkeypatch)
    _stamped_dialog(monkeypatch, {"present": True, "open": 1})
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", None)
    monkeypatch.setattr(agent_tools, "ClickElementEvent", lambda node: ("click", node))
    fn, pm = _registered_action("find_by_text")
    session = _FakeClickSession({10218: _panel_footer_button()}, result=None)

    res = await fn(params=pm(text="FPS", near_text="Brooklyn Millar", click_first=True),
                   browser_session=session)

    assert res.error is None, res.error
    assert "clicked the single match" in (res.extracted_content or "")


async def test_a_skipped_scope_is_declared_in_the_receipt(monkeypatch):
    """Silently ignoring near_text would read as if the scope had been honoured — the
    agent asked for something in Brooklyn Millar's row and got a control in no row."""
    _rowless(monkeypatch)
    _stamped_dialog(monkeypatch, {"present": True, "open": 1})
    monkeypatch.setattr(agent_tools, "_LIVE_NETWORK", None)
    monkeypatch.setattr(agent_tools, "ClickElementEvent", lambda node: ("click", node))
    fn, pm = _registered_action("find_by_text")
    session = _FakeClickSession({10218: _panel_footer_button()}, result=None)

    res = await fn(params=pm(text="FPS", near_text="Brooklyn Millar", click_first=True),
                   browser_session=session)

    msg = res.extracted_content or ""
    assert "Brooklyn Millar" in msg and "not applied" in msg


async def test_two_matches_still_refuse_an_unmatched_row(monkeypatch):
    """The guard itself is untouched at the boundary: the moment a SECOND same-named
    control exists there is something to disambiguate, and an unmatched scope must still
    refuse rather than click a row the agent did not ask for."""
    session, row_of = _grid("Dec-26 £5,000.00 ", "Jan-27 £5,000.00 ")
    _stub_rows(monkeypatch, row_of)
    fn, pm = _registered_action("find_by_text")

    res = await fn(params=pm(text="Net to gross", near_text="Aug-99", click_first=True),
                   browser_session=session)

    assert res.error and "sits in a row" in res.error
    assert "2 element(s)" in res.error
    assert (res.metadata or {}).get("no_click") is True
