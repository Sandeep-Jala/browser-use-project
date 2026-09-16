"""A receipt must show the evidence that justified the match, not just the element's text.

Run 20260903_122807_063006 subtask 16 ("Then select Bulk upload FPS which is a button next to
Save & Next"). Step 1 clicked the RIGHT button and the agent disowned it:

    find_by_text('Bulk upload FPS'): clicked the single match
        index=1068 <button> text=' FPS' id='btnFPS'
    step 2 eval: "Mis-clicked the manual FPS button instead of Bulk upload FPS"

There is no "manual FPS" — the agent coined it. The app's button next to Save & Next is
labelled plain `FPS` (`[21701]<button id=btnSave/> Save & Next` / `[21702]<button id=btnFPS/>
FPS`, straight out of library/8dc07f3a936f920e.recording.json), and "Bulk upload" appears
NOWHERE in the app's DOM — every hit in the recordings is the task prompt embedded in the
history.

So how did the query match? find_by_text needs EVERY token, and `fps` is the only one the
text and id supply. `bulk` and `upload` came from `_descendant_icon_hints`, which harvests
`data-icon-name` / child titles / icon-font classes that browser-use drops — the button
carries a bulk-upload icon, which is exactly why the task calls it that. The agent's own
step-4 note agrees: "list_actions near 'FPS' returned Upload icon".

The match was right and the receipt hid the reason for it. The agent compared ' FPS' against
what it had asked for, declared a mis-click, and spent ten steps fighting to close the panel
it had just correctly opened.

`list_actions` has always reported these hints (its `_decoded` folds them in, in brackets).
find_by_text's line did not — the same one-of-two-twins inconsistency as the missing "those
clicks all landed" clause on the repeat budget.
"""
from tests.test_agent_tools import _FakeBrowserSession, _FakeDomNode, _registered_action


def _icon_button(text="FPS", icon="BulkUpload", attrs=None):
    """A Fluent icon button: bare text, meaning carried by a child `data-icon-name`."""
    node = _FakeDomNode(text, attributes=dict(attrs or {}, id="btnFPS"))
    node.node_name = "BUTTON"
    node.tag_name = "button"
    if icon:
        child = _FakeDomNode("", attributes={"data-icon-name": icon})
        child.node_name = "I"
        child.tag_name = "i"
        node.children = [child]
    return node


async def _find(session, text):
    fn, pm = _registered_action("find_by_text")
    return await fn(params=pm(text=text), browser_session=session)


async def test_the_receipt_names_the_icon_that_satisfied_the_query():
    """THE regression. Without this the agent is told it clicked ' FPS' when it asked for
    'Bulk upload FPS', and has no way to see its query WAS honoured."""
    session = _FakeBrowserSession({1068: _icon_button()})

    res = await _find(session, "Bulk upload FPS")

    body = res.extracted_content or ""
    assert "1 match(es)" in body, body
    assert "BulkUpload" in body


async def test_every_candidate_in_a_listing_carries_its_icon():
    """Icon buttons are exactly the case where the visible text cannot tell two candidates
    apart — the listing is useless without the decoded name."""
    session = _FakeBrowserSession({
        1068: _icon_button(icon="BulkUpload"),
        1070: _icon_button(icon="Upload"),
    })

    body = (await _find(session, "upload")).extracted_content or ""

    assert "BulkUpload" in body and "Upload" in body


async def test_a_control_with_no_icon_is_printed_exactly_as_before():
    """No empty `icon=''` noise on the ordinary named controls that make up most listings."""
    session = _FakeBrowserSession({1068: _icon_button(text="Save & Next", icon=None)})

    body = (await _find(session, "Save & Next")).extracted_content or ""

    assert "icon=" not in body


async def test_a_hint_already_visible_in_the_text_is_not_repeated():
    """The receipt should read like a sentence, not a stutter: when the icon name is already
    the control's text there is nothing new to tell the agent."""
    session = _FakeBrowserSession({1068: _icon_button(text="BulkUpload", icon="BulkUpload")})

    body = (await _find(session, "BulkUpload")).extracted_content or ""
    line = next(l for l in body.splitlines() if l.startswith("index="))

    assert line.count("BulkUpload") == 1, line


async def test_the_icon_never_replaces_the_control_s_real_text():
    """Additive. The text and id are what the agent checks a click receipt against, and the
    NAME MISMATCH warning downstream reads the control's own name, not this."""
    session = _FakeBrowserSession({1068: _icon_button()})

    body = (await _find(session, "Bulk upload FPS")).extracted_content or ""

    assert "text='FPS'" in body and "id='btnFPS'" in body
