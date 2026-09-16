"""list_actions must not answer a question it could not find, and must not be batched past.

Run 20260903_100115_102078, subtask 16 ("Then select Bulk upload FPS … unselect all
employees … tick the 4th Employee checkbox …"). The agent never clicked Bulk upload FPS.
Its first turn was:

    [1/4] list_actions: near_text: Select Employee   ← 30 controls
    [2/4] click: index: 1                            ← "Element index 1 not available"
    [3/4] list_actions: near_text: Anas Burns        ← 30 controls
    [4/4] click: index: 1                            ← "Element index 1 not available"

Two faults, both older than the last commit (`git diff HEAD` does not touch list_actions):

1. THE MISS WAS SILENT. The bulk-upload grid had never been opened, so none of "Select
   Employee", "Anas Burns" or "Employees" was on the page. With no anchor match the window
   filter is skipped entirely and the tool returns the page's FIRST 30 named controls under
   the heading "Clickable controls near 'Select Employee'". The agent was told three times
   that a grid it had not opened was in front of it, believed it, and started at sentence
   two of its job — hunting checkboxes that did not exist.

2. THE INDEX WAS GUESSED. All four actions came from ONE turn: `click(index=1)` was chosen
   before list_actions had returned anything. The real indices on that page were four
   digits (1180, 1290, 7413); `1` meant "the first thing you are about to list". A tool
   whose whole purpose is to hand over indices the agent does not have cannot allow actions
   queued behind it in the same turn.

The channel for fault 1 follows the house rule in _text_miss: it is decided by INTENT.
list_actions is a probe — "what is near this?" is a question whose answer can legitimately
be none — so a miss keeps the SUCCESS channel and says so honestly. Stopping the batch is
fault 2's job, and browser-use has a first-class flag for it (`terminates_sequence`, which
multi_act honours and `navigate`/`search`/`switch`/`evaluate` all set).
"""
from tests.test_agent_tools import _FakeBrowserSession, _FakeDomNode, _registered_action


def _page(*names):
    """A selector map with four-digit indices, like the live payrun page."""
    return {1180 + i * 30: _FakeDomNode(n) for i, n in enumerate(names)}


# ------------------------------- fault 1: the silent miss -------------------------------


async def test_a_landmark_that_is_not_on_the_page_lists_nothing():
    """The exact live call. "Select Employee" belongs to the bulk-upload grid, which was
    never opened — so the answer is "not here", not the page's first 30 controls."""
    fn, pm = _registered_action("list_actions")
    session = _FakeBrowserSession(_page("Save & Next", "Bulk upload FPS", "Add Expenses"))

    res = await fn(params=pm(near_text="Select Employee"), browser_session=session)

    body = res.extracted_content or ""
    assert "index=" not in body, body          # nothing offered to click
    assert "Save & Next" not in body           # and no unrelated control named


async def test_the_miss_says_the_section_may_not_be_open():
    """A receipt that names the likely cause, like find_by_text's. The agent's own step
    told it to open this section first; the tool should point back at that, not invite a
    different search string."""
    fn, pm = _registered_action("list_actions")
    session = _FakeBrowserSession(_page("Save & Next", "Bulk upload FPS"))

    res = await fn(params=pm(near_text="Select Employee"), browser_session=session)

    body = (res.extracted_content or "").lower()
    assert "select employee" in body           # quotes what was asked for
    assert "not on the page" in body
    assert "open" in body                      # names the likely cause


async def test_a_miss_keeps_the_success_channel():
    """Decided by INTENT, exactly like _text_miss: list_actions is a probe, and "is
    anything near this?" is a question whose answer can honestly be no. The batch is
    stopped by terminates_sequence instead — see below."""
    fn, pm = _registered_action("list_actions")

    res = await fn(params=pm(near_text="Select Employee"),
                   browser_session=_FakeBrowserSession(_page("Save & Next")))

    assert res.error is None


async def test_a_landmark_that_IS_present_still_lists_its_neighbours():
    """The tool's whole reason to exist — decoding nameless icon buttons near a row — is
    untouched. Only the miss changed."""
    fn, pm = _registered_action("list_actions")
    session = _FakeBrowserSession(_page("Select Employee", "Anas Burns", "Submit"))

    res = await fn(params=pm(near_text="Anas Burns"), browser_session=session)

    body = res.extracted_content or ""
    assert "index=" in body
    assert "Anas Burns" in body


async def test_no_near_text_still_lists_the_whole_page():
    """Calling it with no landmark is not a miss — there was nothing to find."""
    fn, pm = _registered_action("list_actions")

    res = await fn(params=pm(near_text=""),
                   browser_session=_FakeBrowserSession(_page("Save & Next", "FPS")))

    assert "index=" in (res.extracted_content or "")


# ------------------------------- fault 2: the guessed index -------------------------------


def test_list_actions_terminates_the_action_batch():
    """Anything queued behind it in the same turn was chosen WITHOUT the indices it
    returns. browser-use's multi_act breaks on this flag (agent/service.py), the same way
    it does for navigate/search/switch/evaluate."""
    from automation.pipeline.agent_tools import build_tools

    action = build_tools().registry.registry.actions["list_actions"]

    assert action.terminates_sequence is True
