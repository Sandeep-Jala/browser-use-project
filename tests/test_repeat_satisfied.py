"""A repetition that is already done is a request SATISFIED, not a request refused.

Run 20260903_114110_507756 subtask 15 ("exactly 5 more clicks"). repeat_click(times=5)
landed all five — receipt "5 of 5 asked", the panel advanced, every advance POST 200. The
agent verified the new employee, then rewrote its own memory:

    step 2 memory: "Completed 5 further Save & Next clicks; current employee ... Hydrax Raina."
    step 3 memory: "Current employee ... Hydrax Raina. Next: advance through five more
                    employees by clicking Save & Next five times."

Same fact, inverted, in one turn. It called repeat_click four more times and each was met
with:

    repeat_click REFUSED — did NOT click: Save & Next has already been clicked 5 time(s)...

and then reported the step incomplete, quoting exactly that: "repeat_click was refused and no
further navigation occurred ... this final action was not executed". A correct segment was
recorded FAILED and the run stopped with fifteen good subtasks behind it.

The agent asked for five clicks on that control. Five clicks on that control exist. Its
request HOLDS — so the honest answer is success with zero work done, not "REFUSED — did NOT
click", which is a sentence about failure. Note the plain-click twin (_refuse_if_over_budget)
already says "and those clicks all landed"; the repeat_click branch never got that clause.

Why dropping the ERROR channel here is safe, when every other refusal in this module needs
it: the error channel exists to stop batched follow-ups firing against a page the refused
action left unchanged-but-expected-to-change. Here NOTHING is clicked, so the page really is
unchanged and no index goes stale — and the budget stays enforced on the other route, where
_refuse_if_over_budget still refuses a plain click (the guard that stopped ten employees
being paid for a five-employee slice in run 20260902_105732).
"""
from automation.pipeline import agent_tools
from tests.test_agent_tools import (_FakeClickSession, _FakeEvent, _budget,
                                    _registered_action, _save_next)


async def _spend_the_budget(monkeypatch, budget=5, asked=5):
    """Run the real first repeat so the ledger is populated the way a live step's is."""
    _budget(monkeypatch, budget)
    fn, _pm = _registered_action("repeat_click")
    node = _save_next()
    session = _FakeClickSession({4: node})
    clicks = []
    session.event_bus.dispatch = lambda _e: (clicks.append(1), _FakeEvent(None))[1]
    first = await fn(index=4, times=asked, browser_session=session)
    assert first.error is None and len(clicks) == asked
    return fn, session, clicks


# ------------------------------- the satisfied call -------------------------------


async def test_a_repetition_already_done_answers_on_the_SUCCESS_channel(monkeypatch):
    """THE regression. "REFUSED — did NOT click" is what the agent quoted as its reason for
    reporting the step incomplete."""
    fn, session, clicks = await _spend_the_budget(monkeypatch)

    again = await fn(index=4, times=5, browser_session=session)

    assert again.error is None, again.error
    assert len(clicks) == 5, "a satisfied call must still click NOTHING"


async def test_it_says_plainly_that_nothing_more_was_needed(monkeypatch):
    fn, session, _ = await _spend_the_budget(monkeypatch)

    body = (await fn(index=4, times=5, browser_session=session)).extracted_content

    assert "0 additional" in body
    assert "5 time(s)" in body and "Save & Next" in body


async def test_it_states_the_earlier_clicks_LANDED(monkeypatch):
    """The clause the plain-click twin has and this branch did not. Without it the agent has
    no way to tell "already done" from "never happened"."""
    fn, session, _ = await _spend_the_budget(monkeypatch)

    body = (await fn(index=4, times=5, browser_session=session)).extracted_content

    assert "landed" in body


async def test_it_says_this_is_not_a_failure_of_the_step(monkeypatch):
    """The agent read the old message five times and concluded its job was incomplete. The
    receipt has to close that reading off explicitly."""
    fn, session, _ = await _spend_the_budget(monkeypatch)

    body = (await fn(index=4, times=5, browser_session=session)).extracted_content.lower()

    assert "not a failure" in body
    assert "success=true" in body


async def test_it_persists_so_the_fact_survives_a_memory_rewrite(monkeypatch):
    """The agent rewrote its memory between two consecutive steps and lost the completion.
    The success receipt already sets long_term_memory; this one must too, or the only line
    that survives is the one framing it as refused."""
    fn, session, _ = await _spend_the_budget(monkeypatch)

    res = await fn(index=4, times=5, browser_session=session)

    assert res.long_term_memory and "0 additional" in res.long_term_memory
    assert res.include_in_memory is True


async def test_a_satisfied_call_never_compiles(monkeypatch):
    """no_click, and NO `repeat` metadata: a call that clicked nothing must not add a step to
    the recording, nor tell compile to replay another N clicks."""
    fn, session, _ = await _spend_the_budget(monkeypatch)

    res = await fn(index=4, times=5, browser_session=session)

    assert res.metadata == {"no_click": True}


async def test_the_ledger_does_not_move(monkeypatch):
    fn, session, _ = await _spend_the_budget(monkeypatch)
    before = dict(agent_tools._CLICK_LEDGER)

    await fn(index=4, times=5, browser_session=session)

    assert agent_tools._CLICK_LEDGER == before


# ------------------------------- what must NOT change -------------------------------


async def test_a_SHORTFALL_is_still_an_error(monkeypatch):
    """Different branch, opposite meaning: fewer clicks landed than asked, so a queued
    follow-up must still be stopped and the step must NOT be reported done."""
    _budget(monkeypatch, None, verdicts=[(True, ""), (False, "the control went away")])
    fn, _pm = _registered_action("repeat_click")
    session = _FakeClickSession({4: _save_next()})
    session.event_bus.dispatch = lambda _e: _FakeEvent(None)

    res = await fn(index=4, times=5, browser_session=session)

    assert res.error and "only 2 of the 5" in res.error
    assert res.metadata == {"no_click": True}


async def test_a_plain_click_over_budget_is_still_REFUSED(monkeypatch):
    """The budget stays enforced on the other route. This is the guard that stopped ten
    employees being paid for a five-employee slice (run 20260902_105732) — softening the
    repeat_click branch must not soften this one."""
    await _spend_the_budget(monkeypatch)

    refusal = agent_tools._refuse_if_over_budget(_save_next(), "Save & Next")

    assert refusal is not None and refusal.error
    assert "did NOT click" in refusal.error
