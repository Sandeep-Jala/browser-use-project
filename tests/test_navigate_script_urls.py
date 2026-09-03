"""`navigate` is not a back door for running JavaScript.

Run 20260903_102354_153542, subtask 4 (Net to Gross). The agent could not address one of
twelve identical "Net to gross" pencil icons, so it wrote a DOM script and pushed it through
the only door left open:

    navigate: url: javascript:()=>{const rows=document.querySelectorAll('tr'); …}

browser-use's SecurityWatchdog blocked it — but only after the NavigateToUrlEvent had been
dispatched, and the block lands the tab on **about:blank**. The Pay Forecast page was gone,
and the agent spent the next twelve steps recovering, then did it a second time. Nineteen
steps for a slice that had authored in three, three times running.

`evaluate` is excluded from the registry on purpose (see this module's docstring: JS writes
are unrecordable, so a recording that leans on them compiles to a script missing those
actions). `navigate` was the loophole that gave it back — with the page as collateral.

So the refusal happens HERE, before the event is dispatched: the page keeps its state, and
the agent gets a receipt naming the tools that actually reach a control. `data:` is refused
for the same reason — it is not a page in this app and navigating to it replaces the one
the segment is working in.
"""
import pytest

from automation.pipeline import agent_tools
from tests.test_agent_tools import _registered_action


# ------------------------------- the predicate -------------------------------


@pytest.mark.parametrize("url", [
    "javascript:()=>{document.querySelector('i').click()}",
    "javascript:;",
    "JavaScript:void(0)",
    "  javascript:alert(1)",
    "data:text/html,<script>x()</script>",
])
def test_script_urls_are_refused(url):
    refusal = agent_tools._script_url_refusal(url)

    assert refusal is not None
    # ERROR channel: multi_act stops a step's remaining queued actions on an error, and the
    # agent that did this had two more actions queued behind it both times.
    assert refusal.error
    assert (refusal.metadata or {}).get("no_click") is True


@pytest.mark.parametrize("url", [
    "https://test.actingoffice.com/paye/clients/6a98/calculator",
    "http://localhost:3000/",
    "https://www.fakenamegenerator.com/gen-male-gd-uk.php",
    "",
])
def test_ordinary_urls_pass_through(url):
    assert agent_tools._script_url_refusal(url) is None


def test_the_refusal_names_what_to_use_instead():
    """A refusal that only says no sends the agent hunting for the next loophole. This one
    names the tools that reach a control, which is what it was trying to do."""
    msg = agent_tools._script_url_refusal("javascript:;").error.lower()

    assert "javascript" in msg
    assert "find_by_text" in msg
    assert "list_actions" in msg


def test_the_refusal_says_the_page_was_left_alone():
    """The damage was never the blocked script — it was the about:blank the block left
    behind. The agent must know its page is still there, or it 'recovers' from nothing."""
    msg = agent_tools._script_url_refusal("javascript:;").error.lower()

    assert "not navigate" in msg or "page is unchanged" in msg


# ------------------------------- wired into the action -------------------------------


async def test_the_navigate_action_refuses_a_script_url_without_a_session():
    """No browser_session is touched: the refusal must come BEFORE dispatch, which is the
    whole point — the watchdog's block is what destroys the page."""
    fn, pm = _registered_action("navigate")

    res = await fn(params=pm(url="javascript:;"), browser_session=None)

    assert res.error and "javascript" in res.error.lower()


def test_navigate_keeps_the_builtin_description_and_params():
    """Same name overrides the built-in (the `input` precedent). It must stay the same
    action from the agent's side — same description, same param model — or the override
    silently changes how every ordinary navigation is prompted."""
    from browser_use import Tools

    from automation.pipeline.agent_tools import build_tools

    ours = build_tools().registry.registry.actions["navigate"]
    stock = Tools().registry.registry.actions["navigate"]

    assert ours.description == stock.description
    assert ours.param_model is stock.param_model
    # And it still ends the batch, exactly as the built-in does.
    assert ours.terminates_sequence is stock.terminates_sequence is True
