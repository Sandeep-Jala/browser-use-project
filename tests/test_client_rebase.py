"""A replayed `goto` must land on the client the run is IN, not the one it was recorded in.

Run 20260903_093236_260802, in its own network log: the employee was created in Food
Alchemy (`POST api/Payroll/Clients/6a984504…/Employees/`), and four requests later the run
was reading FOOD LIMITED's employee list. Between them sat four identical hits on
`test.actingoffice.com/paye/clients/6a61d0…/calculator` — the four `api.goto()` lines in
library/d85bb1eda9ba8381.skill.py, which froze the authoring run's client id into the
script. The Pay Forecast slice then searched for the new employee in a client that had
never heard of them.

The identity layer was already client-agnostic and always had been: that entry's context is
`/paye/clients/*/rti/payrun` and its end_context `/paye/clients/*/calculator`, because
`_VOLATILE_SEGMENT` declares a long hex id to be instance data rather than page structure.
One recording is MEANT to serve every client. Only the compiled step disagreed — the
identity half of the system said `*` while the executable half said `6a61d0…`.

So a goto is rebased onto the live URL at replay: walk both paths together, substitute a
volatile segment with the live one, and stop at the first literal segment that disagrees
(past that point the paths have diverged and positions no longer correspond). Foreign
origins are never touched — the aux identity tab and the OTP portal are different sites,
and rewriting those would be the same bug pointed the other way.
"""
from automation.pipeline import subtask_store as ss
from automation.pipeline.script_compile import run_steps
from automation.skills.api import SkillApi
from tests.test_heal_promotion import _launch

APP = "https://test.actingoffice.com"
ALCHEMY = "6a984504a93cf7218de5fbc5"      # Food Alchemy LTD
LIMITED = "6a61d0ab5636abb464ba0e13"      # FOOD LIMITED


# ------------------------------- the rule -------------------------------


def test_the_recorded_client_is_replaced_by_the_live_one():
    """The exact live failure: recorded in FOOD LIMITED, replayed in Food Alchemy."""
    assert ss.rebase_to_live(f"{APP}/paye/clients/{LIMITED}/calculator",
                             f"{APP}/paye/clients/{ALCHEMY}/rti/payrun") == \
        f"{APP}/paye/clients/{ALCHEMY}/calculator"


def test_the_recorded_tail_survives_the_rebase():
    """Only the volatile segment moves. The goto still names the page it meant to open —
    /calculator — even though the live page is /rti/payrun."""
    out = ss.rebase_to_live(f"{APP}/paye/clients/{LIMITED}/reports/payrollsummary",
                            f"{APP}/paye/clients/{ALCHEMY}/")

    assert out == f"{APP}/paye/clients/{ALCHEMY}/reports/payrollsummary"


def test_a_goto_already_on_the_live_client_is_unchanged():
    url = f"{APP}/paye/clients/{ALCHEMY}/calculator"

    assert ss.rebase_to_live(url, f"{APP}/paye/clients/{ALCHEMY}/rti/payrun") == url


def test_a_foreign_origin_is_never_rewritten():
    """The aux identity tab and the OTP portal are different sites. Rebasing those would be
    this same bug pointed the other way — a goto dragged onto the wrong host."""
    aux = "https://www.fakenamegenerator.com/gen-male-gd-uk.php"

    assert ss.rebase_to_live(aux, f"{APP}/paye/clients/{ALCHEMY}/") == aux
    assert ss.rebase_to_live(f"{APP}/paye/clients/{LIMITED}/calculator", aux) == \
        f"{APP}/paye/clients/{LIMITED}/calculator"


def test_a_path_that_diverges_before_the_id_is_left_alone():
    """Different section of the app: the segments before the id disagree, so nothing in the
    recorded path corresponds to anything in the live one."""
    rec = f"{APP}/paye/clients/{LIMITED}/calculator"

    assert ss.rebase_to_live(rec, f"{APP}/admin/practice/{ALCHEMY}/settings") == rec


def test_a_url_with_no_volatile_segment_is_unchanged():
    assert ss.rebase_to_live(f"{APP}/admin", f"{APP}/paye/clients/{ALCHEMY}/") == \
        f"{APP}/admin"


def test_a_live_url_with_a_literal_where_the_recording_had_an_id_is_left_alone():
    """Nothing to take the id FROM: substituting a literal page name for a record id would
    invent a URL neither run ever visited."""
    rec = f"{APP}/paye/clients/{LIMITED}/calculator"

    assert ss.rebase_to_live(rec, f"{APP}/paye/clients/new") == rec


def test_an_unreadable_live_url_leaves_the_goto_alone():
    """about:blank and "" are the two shapes of "no location" — see _is_degenerate_url.
    Fails OPEN: with nothing to rebase onto, the recorded URL is the best guess there is."""
    rec = f"{APP}/paye/clients/{LIMITED}/calculator"

    assert ss.rebase_to_live(rec, "about:blank") == rec
    assert ss.rebase_to_live(rec, "") == rec


def test_query_and_fragment_come_from_the_recording():
    out = ss.rebase_to_live(f"{APP}/paye/clients/{LIMITED}/calculator?tab=2#rows",
                            f"{APP}/paye/clients/{ALCHEMY}/rti/payrun?other=1")

    assert out == f"{APP}/paye/clients/{ALCHEMY}/calculator?tab=2#rows"


def test_every_volatile_segment_on_the_shared_prefix_is_rebased():
    """Not just the first: a year- or period-scoped path carries more than one."""
    out = ss.rebase_to_live(f"{APP}/paye/clients/{LIMITED}/years/27/employees",
                            f"{APP}/paye/clients/{ALCHEMY}/years/28/employees")

    assert out == f"{APP}/paye/clients/{ALCHEMY}/years/28/employees"


# ------------------------------- both replay tiers -------------------------------
#
# The compiled goto reaches the browser through two doors and both were baking the id in:
# script_compile.run_steps (tier 0) and SkillApi.goto (tier 1, what codegen emits).


async def _served(pw):
    """A page on the app origin that answers every request, so real navigation works."""
    browser = await _launch(pw)
    page = await browser.new_page()
    await page.route("**/*", lambda route: route.fulfill(
        status=200, content_type="text/html", body="<h1>page</h1>"))
    await page.goto(f"{APP}/paye/clients/{ALCHEMY}/rti/payrun")
    return browser, page


async def test_tier0_replay_navigates_to_the_live_client():
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser, page = await _served(pw)
        out = await run_steps(page, [
            {"action": "goto", "url": f"{APP}/paye/clients/{LIMITED}/calculator"},
        ], timeout_ms=5000)

        assert out["failed_at"] is None, out["error"]
        assert page.url == f"{APP}/paye/clients/{ALCHEMY}/calculator"
        await browser.close()


async def test_tier1_replay_navigates_to_the_live_client():
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser, page = await _served(pw)
        api = SkillApi(page, {}, timeout_ms=5000)
        await api.goto(f"{APP}/paye/clients/{LIMITED}/calculator")

        assert api.page.url == f"{APP}/paye/clients/{ALCHEMY}/calculator"
        await browser.close()
