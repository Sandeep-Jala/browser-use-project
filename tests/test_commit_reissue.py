"""The COMMIT rung of the click recovery ladder.

Run 20260828_144426 subtask 0: the cached script is
`fill(search, "FOOD LIMITED") -> press Enter -> wait 2.0 -> click(food-limited)`. On a slow
connection the fill landed while the grid was still fetching its first page, the app dropped
the commit (no `search=FOOD%20LIMITED` request was ever issued), and the click then refused —
correctly — because the row it wanted was not in the unfiltered list.

The recovery ladder could not help: it re-clicks the nearest previous CLICK, which was
`payroll`, a link that lives in the modules flyout and is unreachable once that flyout has
closed. The step that actually PRODUCED the list is the fill, and nothing ever re-issued it.

These tests pin the rung that does: re-fill, re-press, retry the target once — and pin the
cases where it must NOT fire.
"""
from __future__ import annotations


class _FakeKeyboard:
    def __init__(self):
        self.pressed = []

    async def press(self, keys):
        self.pressed.append(keys)


class _FakePage:
    def __init__(self):
        self.keyboard = _FakeKeyboard()
        self.waits = []

    async def wait_for_timeout(self, ms):
        self.waits.append(int(ms))


# The shape that failed live: click(menus) click(payroll) fill(search) press(Enter) wait
# click(food-limited).
_SEARCH_STEPS = [
    {"action": "click", "selectors": ['css=[id="btn-menus-callout"]']},
    {"action": "click", "selectors": ['css=a[href="/paye"]']},
    {"action": "fill", "selectors": ['css=[placeholder="Search"]'], "value": "FOOD LIMITED"},
    {"action": "press", "keys": "Enter"},
    {"action": "wait", "seconds": 2.0},
    {"action": "click", "selectors": ['css=a:has-text("FOOD LIMITED")'],
     "expect_text": "FOOD LIMITED"},
]


def test_preceding_commit_finds_the_fill_across_waits_and_presses():
    from automation.pipeline.script_compile import _preceding_commit

    commit = _preceding_commit(_SEARCH_STEPS, 5)
    assert commit is not None
    fill_step, presses = commit
    assert fill_step["value"] == "FOOD LIMITED"
    assert presses == ["Enter"]


def test_preceding_commit_stops_at_an_intervening_click():
    """A click between the fill and the target means the fill is not what produced it —
    the existing predecessor-click rung owns that case."""
    from automation.pipeline.script_compile import _preceding_commit

    steps = _SEARCH_STEPS[:5] + [
        {"action": "click", "selectors": ['css=[id=open]']},
        {"action": "click", "selectors": ['css=a:has-text("FOOD LIMITED")']},
    ]
    assert _preceding_commit(steps, 6) is None


def test_preceding_commit_refuses_a_value_with_an_unresolved_noted_token():
    """`{{noted:x}}` resolves from the live extract ledger at step-execution time, which the
    recovery has no access to — re-typing the token itself would poison the field."""
    from automation.pipeline.script_compile import _preceding_commit

    steps = [
        {"action": "fill", "selectors": ['css=[name=code]'], "value": "{{noted:otp}}"},
        {"action": "press", "keys": "Enter"},
        {"action": "click", "selectors": ['css=[id=go]']},
    ]
    assert _preceding_commit(steps, 2) is None


async def test_flyout_recovery_reissues_the_commit_before_re_clicking(monkeypatch):
    from automation.pipeline import script_compile as sc

    seen = {"fills": [], "clicks": 0}

    async def fake_follow(page, step, timeout_ms):
        raise RuntimeError("no unique candidate matched: css=... -> no match")

    async def fake_fill(page, step, timeout_ms):
        seen["fills"].append(step["value"])
        return "fill-sel", None

    async def fake_click(page, step, timeout_ms):
        seen["clicks"] += 1
        return "click-sel", None

    monkeypatch.setattr(sc, "_click_and_follow", fake_follow)
    monkeypatch.setattr(sc, "_fill_with_retry", fake_fill)
    monkeypatch.setattr(sc, "_click_with_retry", fake_click)

    page = _FakePage()
    sel, healed, out = await sc._click_with_flyout_recovery(page, _SEARCH_STEPS, 5, 5000)
    assert sel == "click-sel"
    assert out is page
    assert seen["fills"] == ["FOOD LIMITED"]     # the fill was re-issued...
    assert page.keyboard.pressed == ["Enter"]    # ...with its commit key
    assert seen["clicks"] == 1                   # target retried once; NO predecessor re-click


async def test_flyout_recovery_still_re_clicks_when_no_fill_precedes(monkeypatch):
    """No fill in reach -> the ladder behaves exactly as it did before."""
    from automation.pipeline import script_compile as sc

    seen = {"fills": 0, "clicked": []}

    async def fake_follow(page, step, timeout_ms):
        raise RuntimeError("no unique candidate matched: css=... -> no match")

    async def fake_fill(page, step, timeout_ms):
        seen["fills"] += 1
        return "fill-sel", None

    async def fake_click(page, step, timeout_ms):
        seen["clicked"].append(step["selectors"][0])
        return "click-sel", None

    monkeypatch.setattr(sc, "_click_and_follow", fake_follow)
    monkeypatch.setattr(sc, "_fill_with_retry", fake_fill)
    monkeypatch.setattr(sc, "_click_with_retry", fake_click)

    steps = [
        {"action": "click", "selectors": ['css=[title="Standard data request."]']},
        {"action": "scroll_panels"},
        {"action": "click", "selectors": ['css=[role="row"]:has-text("X") div']},
    ]
    sel, _healed, _page = await sc._click_with_flyout_recovery(_FakePage(), steps, 2, 5000)
    assert sel == "click-sel"
    assert seen["fills"] == 0
    # predecessor reopened first, then the target retried
    assert seen["clicked"] == ['css=[title="Standard data request."]',
                               'css=[role="row"]:has-text("X") div']


async def test_skill_click_reissues_the_last_commit(monkeypatch):
    """Tier-1 parity: SkillApi has no step list, so the commit rides on the ledger."""
    from automation.skills import api as api_mod

    seen = {"fills": [], "clicks": 0}

    async def fake_fill(page, step, timeout_ms):
        seen["fills"].append(step["value"])
        return "fill-sel", None

    async def fake_click(page, step, timeout_ms):
        seen["clicks"] += 1
        if seen["clicks"] == 1:
            raise RuntimeError("no unique candidate matched: css=... -> no match")
        return "click-sel", None

    monkeypatch.setattr(api_mod, "_fill_with_retry", fake_fill)
    monkeypatch.setattr(api_mod, "_click_with_retry", fake_click)

    page = _FakePage()
    api = api_mod.SkillApi(page, {
        "search": {"selectors": ['css=[placeholder="Search"]']},
        "food-limited": {"selectors": ['css=a:has-text("FOOD LIMITED")'],
                         "expect_text": "FOOD LIMITED"},
    })
    await api.fill("search", "FOOD LIMITED")
    await api.press("Enter")
    await api.wait(0.0)
    await api.click("food-limited")

    assert seen["fills"] == ["FOOD LIMITED", "FOOD LIMITED"]   # original + re-issue
    assert page.keyboard.pressed == ["Enter", "Enter"]
    assert seen["clicks"] == 2


async def test_skill_commit_is_dropped_once_another_click_intervenes(monkeypatch):
    """A click between the fill and the failure means the fill no longer produced the
    target's list — the flyout rung owns it, and the field must not be retyped."""
    from automation.skills import api as api_mod

    seen = {"fills": [], "clicks": []}

    async def fake_fill(page, step, timeout_ms):
        seen["fills"].append(step["value"])
        return "fill-sel", None

    async def fake_click(page, step, timeout_ms):
        seen["clicks"].append(step["selectors"][0])
        if len(seen["clicks"]) == 2:                # the target's first attempt fails
            raise RuntimeError("no unique candidate matched: css=... -> no match")
        return "click-sel", None

    monkeypatch.setattr(api_mod, "_fill_with_retry", fake_fill)
    monkeypatch.setattr(api_mod, "_click_with_retry", fake_click)

    api = api_mod.SkillApi(_FakePage(), {
        "search": {"selectors": ['css=[placeholder="Search"]']},
        "opener": {"selectors": ['css=[id=opener]']},
        "target": {"selectors": ['css=[id=target]']},
    })
    await api.fill("search", "FOOD LIMITED")
    await api.click("opener")
    await api.click("target")

    assert seen["fills"] == ["FOOD LIMITED"]       # never retyped
    # opener, target (fails), opener again (flyout rung), target again
    assert seen["clicks"] == ['css=[id=opener]', 'css=[id=target]',
                              'css=[id=opener]', 'css=[id=target]']
