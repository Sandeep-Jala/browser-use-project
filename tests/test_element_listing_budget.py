"""The control list the agent reads is CHARACTER-CAPPED, and the cap silently hid the
control the step named.

Run 20260904_160415_098035 subtask 4, the FPS bulk-upload panel. Verified against the live
DOM the same afternoon: the panel's submit button is a Fluent primary button at 94.2% of
the page's interactive order (1521 of 1615), with only 92 interactive elements after it and
108 grid rows in front of it. browser-use serialises the listing to at most
`max_clickable_elements_length` characters (default 40000,
`browser_use/agent/views.py:92`), so it printed 956 of 1055 controls and cut mid-attribute:

    [5503]<i role=i

The submit button was in the dropped 99. What the agent DID read, at listing lines 488-492:

    'Save & Next'
    'FPS'
    '[2392]<div />'      <- the ms-Overlay, 1440x623, no name at all
    'FPS'
    '[2238]<button />'   <- ms-Panel-closeButton, ax_name null

Both `FPS` lines are bare STATIC TEXT — in that whole page state not one INDEXED control
is named FPS (a named control prints its label as an indented child line). The agent read
"the button under the FPS label is the FPS button", clicked 2238, and closed the panel with
its five ticked employees.

Two fixes, and they are layers, not alternatives:

1. Raise the cap so the button is listed at all. Measured, not guessed: the full listing on
   this page is ~44.2k characters (956 printed in 40016 chars = 41.9 chars/element x 1055),
   and the worst page in the whole recorded corpus is ~51.6k. The cap only truncates when
   exceeded, and browser-use keeps the state message in ONE REPLACED SLOT
   (`message_manager/service.py:559`), never a growing history — so the cost is a few
   thousand characters on the minority of steps that are dense, and zero on the rest.

2. When a listing IS still truncated, say so and say what to do instead. browser-use's own
   header already admits the truncation and it did not help: `prompts.py:276` prints
   `[End of page]` AFTER the cut, so the last thing the model reads is a severed attribute
   followed by "you have seen everything". The notice has to name the missing count and
   route the agent to find_by_text, which searches the page rather than the listing.
"""
import inspect
import json
import re
from pathlib import Path

from automation.pipeline import runner
from automation.pipeline.runner import Runner, truncated_listing_notice

# Never pin a recording FILENAME: the pipeline renames these as a segment's fate changes
# (`.recording.new.interrupted.json` -> `.recording.failed.json` -> `.recording.json` when
# it finally commits), so a hard-coded path passes until the run it documents succeeds and
# then fails for a reason that has nothing to do with the code. Select by CONTENT instead.


def _states(predicate):
    """Every recorded state message in library/ satisfying `predicate`, newest file first."""
    for rec in sorted(Path("library").glob("*recording*.json"),
                      key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            hist = json.loads(rec.read_text()).get("history") or []
        except (ValueError, OSError):
            continue
        for step in hist:
            state = step.get("state_message") or ""
            if state and predicate(state):
                yield state


def _one(predicate, what):
    state = next(_states(predicate), None)
    assert state is not None, f"no recording in library/ has {what}"
    return state


def _listing_body(state: str) -> str:
    """The serialised control listing only — header exclusive, "[End of page]" exclusive.

    Both bounds are required. Without the header the task text is swept in; without the
    marker the agent history is, and a sparse page then measures 585 chars/element instead
    of its real ~42."""
    head = re.search(r"Interactive elements[^:]*:\n", state)
    if not head or "[End of page]" not in state:
        return ""
    return state[head.end():].split("[End of page]", 1)[0]


def _full_listing_estimate(state: str) -> float:
    """How many characters this page's listing would need UNTRUNCATED.

    Per page, never pooled: chars-per-element is a property of what the page renders. The
    dense grid pages that actually threaten the budget sit at ~42 chars/element; a sparse
    form page runs double that on a tenth of the controls and must not be allowed to
    inflate the estimate for a page it says nothing about. Pages under 100 listed controls
    are ignored for the same reason."""
    stats = re.search(r"(\d+) interactive", state)
    body = _listing_body(state)
    printed = len(set(re.findall(r"\[(\d+)\]", body)))
    if not stats or printed < 100:
        return 0.0
    return len(body) / printed * int(stats.group(1))


def _truncated_state():
    """A real truncated step when the corpus still holds one, else one modelled on the
    exact strings browser-use emits (pinned by the test below).

    The corpus stops holding one as soon as the budget is high enough — that is the fix
    working — so the parser's tests must not depend on the failure still being reproducible
    in the recordings. Every literal here was read off a real state message first:
    `library/f2214d2e4d74c09b.recording.failed.json` step 2, 956 of 1055 printed, cut
    mid-attribute at `[5503]<i role=i`, with the two bare `FPS` labels stranded next to the
    overlay and the panel close button."""
    real = next(_states(lambda s: "truncated to" in s and "interactive" in s), None)
    if real:
        return real
    rows = "\n".join(f'\t\t[{5000 + n}]<div role=gridcell />' for n in range(956))
    return (
        "<page_stats>17 links, 1055 interactive, 0 iframes, 3 shadow(open), "
        "0 shadow(closed), 2668 total elements</page_stats>\n"
        "Interactive elements (truncated to 40000 characters):\n"
        "Save & Next\nFPS\n[2392]<div />\nFPS\n[2238]<button />\n"
        + rows + "\n\t\t\t[5503]<i role=i"
    )


def _untruncated_state():
    """A real step whose listing fitted — the control case."""
    return _one(lambda s: "truncated to" not in s and "interactive" in s,
                "an untruncated control listing")


def test_the_modelled_state_matches_the_string_browser_use_actually_emits():
    """The fixture above is only trustworthy while it mirrors the real emitter. browser-use
    builds the marker at agent/prompts.py:256; if that wording changes, the notice stops
    firing in production and this catches it here rather than in a run."""
    import inspect

    from browser_use.agent import prompts as bu_prompts

    src = inspect.getsource(bu_prompts)
    assert "f' (truncated to {self.max_clickable_elements_length} characters)'" in src
    assert runner._LISTING_TRUNCATED_RE.search(_truncated_state())


# ------------------------------- the notice -------------------------------


def test_a_real_truncated_panel_step_produces_a_notice():
    notice = truncated_listing_notice(_truncated_state())

    assert notice is not None
    # It must route to the tool that searches the PAGE, not the listing.
    assert "find_by_text" in notice


def test_the_notice_counts_what_is_missing_from_the_real_step():
    """Vague "some controls may be missing" is ignorable; "99 of 1055" is not. The numbers
    are DERIVED from the recording, not pinned, so this keeps meaning the same thing as the
    app's pages grow."""
    state = _truncated_state()
    notice = truncated_listing_notice(state)

    printed = len(set(re.findall(r"\[(\d+)\]", state)))
    total = int(re.search(r"(\d+) interactive", state).group(1))
    assert 0 < printed < total, f"{printed=} {total=} — not actually a truncated listing"
    assert str(total - printed) in notice and str(total) in notice


def test_the_notice_warns_that_an_index_under_a_label_is_not_that_labels_control():
    """The exact wrong inference: '[2238]<button/>' printed under a bare 'FPS' text line."""
    notice = truncated_listing_notice(_truncated_state())

    assert "not necessarily" in notice.lower()


def test_an_untruncated_step_produces_no_notice():
    assert truncated_listing_notice(_untruncated_state()) is None


def test_a_missing_or_empty_state_message_is_silent():
    assert truncated_listing_notice("") is None
    assert truncated_listing_notice(None) is None


# ------------------------------- the budget -------------------------------


def test_the_runner_pins_the_clickable_listing_budget():
    """A library upgrade or refactor must not silently drop it back to 40000."""
    src = inspect.getsource(Runner.run_agent_segment)
    assert "max_clickable_elements_length" in src


def test_the_configured_budget_clears_the_worst_page_in_the_corpus():
    """Self-updating against real recordings. If this app ever grows a page denser than the
    budget covers, this fails here instead of the run silently losing controls again — the
    grid gains an employee every run, so that is a when, not an if."""
    worst, where = 0.0, ""
    for rec in sorted(Path("library").glob("*recording*.json")):
        try:
            hist = json.loads(rec.read_text()).get("history") or []
        except (ValueError, OSError):
            continue
        for i, step in enumerate(hist):
            need = _full_listing_estimate(step.get("state_message") or "")
            if need > worst:
                worst, where = need, f"{rec.name} step {i}"

    # ~49.7k, the FPS panel at 1170 controls (e70874b26d951bb4 step 2).
    assert worst > 45_000, f"corpus not readable — worst estimate was only {worst:,.0f}"
    assert runner.CLICKABLE_LISTING_BUDGET >= worst * 1.25, (
        f"budget {runner.CLICKABLE_LISTING_BUDGET:,} does not clear the worst recorded "
        f"page with 25% headroom: {where} needs ~{worst:,.0f} chars")


def test_the_default_budget_would_NOT_have_cleared_it():
    """The counter-check: 40000 is what hid the button, so the test above must be measuring
    something the default genuinely fails. Without this, a mistake in the estimate could
    make the budget test vacuously true."""
    worst = max((_full_listing_estimate(step.get("state_message") or "")
                 for rec in Path("library").glob("*recording*.json")
                 for step in (json.loads(rec.read_text()).get("history") or [])),
                default=0.0)

    assert worst > 40_000, "the 40000 default would have sufficed — re-check the estimate"
