"""decompose tests: tier order, caching, derived matching, and the hallucination guard.
The LLM is stubbed with canned replies; derived matching is asserted to use NO LLM."""
import json

import pytest

from automation.pipeline import decompose
from automation.pipeline import subtask_store as ss
from automation.tasks import SubtaskDecl, TaskSpec


@pytest.fixture
def library(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(ss, "LIBRARY_MANIFEST", tmp_path / "library" / "manifest.json")
    monkeypatch.setattr(ss, "DECOMPOSITIONS_DIR", tmp_path / "decompositions")
    return tmp_path


class StubLLM:
    """Returns a canned completion; counts calls so tests can assert zero-LLM paths."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1

        class R:
            completion = self.reply

        return R()


class SeqLLM:
    """Returns queued completions in order (last one repeats); records received messages."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls = 0
        self.seen: list[list] = []

    async def ainvoke(self, messages):
        self.seen.append(list(messages))
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1

        class R:
            completion = reply

        return R()


PROMPT = ("go to Bookkeeping module, search and select 290 CREW LIMITED business name. "
          "add invoice for customer Suresh Gopi with qty 5 and click save")

GOOD_REPLY = json.dumps({"subtasks": [
    {"template_prompt": "go to Bookkeeping module, search and select {{business}} business name",
     "values": {"business": "290 CREW LIMITED"}, "is_save_step": False},
    {"template_prompt": "add invoice for customer {{customer}} with qty {{qty}} and click save",
     "values": {"customer": "Suresh Gopi", "qty": "5"}, "is_save_step": True},
]})


# ------------------------------- tier 1: spec passthrough -------------------------------


@pytest.mark.asyncio
async def test_spec_subtasks_win_and_are_cached(library):
    spec = TaskSpec(
        key="k", prompt=PROMPT, marker="Invoices",
        subtasks=(
            SubtaskDecl(prompt="go to Bookkeeping module, search and select {{business}} "
                               "business name", values={"business": "290 CREW LIMITED"}),
            SubtaskDecl(prompt="add invoice for customer {{customer}} with qty {{qty}} and "
                               "click save", values={"customer": "Suresh Gopi", "qty": "5"}),
        ),
    )
    llm = StubLLM(GOOD_REPLY)
    subs = await decompose.get_decomposition(PROMPT, llm=llm, spec=spec)

    assert llm.calls == 0
    assert [s.index for s in subs] == [0, 1]
    assert subs[0].instantiated_prompt.startswith(
        "go to Bookkeeping module, search and select 290 CREW LIMITED")
    # No subtask declared a marker -> the save defaults to the LAST subtask.
    assert subs[0].marker is None and subs[1].marker == "Invoices"
    # Cached canonically under the parent prompt's tid.
    cached = ss.load_decomposition(ss.task_id(PROMPT))
    assert cached and cached["source"] == "spec"


@pytest.mark.asyncio
async def test_invalid_spec_subtasks_fall_back_to_whole_prompt(library):
    spec = TaskSpec(key="k", prompt=PROMPT, marker="Invoices", subtasks=(
        SubtaskDecl(prompt="do {{thing}}"),  # token with no value: broken declaration
    ))
    subs = await decompose.get_decomposition(PROMPT, llm=None, spec=spec)
    assert len(subs) == 1
    assert subs[0].template_prompt == " ".join(PROMPT.split())
    assert subs[0].marker == "Invoices"


# ------------------------------- tier 2: exact cache -------------------------------


@pytest.mark.asyncio
async def test_exact_cache_hit_uses_no_llm(library):
    llm = StubLLM(GOOD_REPLY)
    first = await decompose.get_decomposition(PROMPT, llm=llm, marker="Invoices")
    assert llm.calls == 1 and len(first) == 2

    again = await decompose.get_decomposition(PROMPT, llm=llm, marker="Invoices")
    assert llm.calls == 1  # cache hit: no second call
    assert [s.template_prompt for s in again] == [s.template_prompt for s in first]


# ------------------------------- tier 3: derived match -------------------------------


@pytest.mark.asyncio
async def test_derived_match_reuses_templates_with_new_values_no_llm(library):
    seeded = StubLLM(GOOD_REPLY)
    await decompose.get_decomposition(PROMPT, llm=seeded, marker="Invoices")
    assert seeded.calls == 1

    new_prompt = ("go to Bookkeeping module, search and select ACME LTD business name. "
                  "add invoice for customer Mr Jones with qty 9 and click save")
    subs = await decompose.get_decomposition(new_prompt, llm=None, marker="Invoices")

    # Same template prompts (same library sids), new values read from the prompt.
    assert subs[0].values == {"business": "ACME LTD"}
    assert subs[1].values == {"customer": "Mr Jones", "qty": "9"}
    assert subs[1].marker == "Invoices"
    derived = ss.load_decomposition(ss.task_id(new_prompt))
    assert derived and derived["source"].startswith("derived:")


@pytest.mark.asyncio
async def test_structural_change_does_not_derive(library):
    seeded = StubLLM(GOOD_REPLY)
    await decompose.get_decomposition(PROMPT, llm=seeded, marker="Invoices")

    structural = ("go to Bookkeeping module, search and select ACME LTD business name. "
                  "add a credit note for supplier Mr Jones and click save")
    subs = await decompose.get_decomposition(structural, llm=None, marker="Refunds")
    # No derivation possible and no LLM -> single whole-prompt fallback.
    assert len(subs) == 1
    assert subs[0].marker == "Refunds"


# ------------------------------- tier 4: LLM + validation -------------------------------


@pytest.mark.asyncio
async def test_llm_decomposition_assigns_save_marker(library):
    subs = await decompose.get_decomposition(PROMPT, llm=StubLLM(GOOD_REPLY),
                                             marker="Invoices")
    assert len(subs) == 2
    assert subs[0].marker is None
    assert subs[1].marker == "Invoices"  # is_save_step: true


@pytest.mark.asyncio
async def test_hallucinated_value_rejected_then_fallback(library):
    bad = json.dumps({"subtasks": [
        {"template_prompt": "select {{business}}",
         "values": {"business": "NOT IN THE PROMPT LLC"}, "is_save_step": True},
    ]})
    llm = StubLLM(bad)
    subs = await decompose.get_decomposition(PROMPT, llm=llm, marker="Invoices")
    assert llm.calls == 3  # two retries, then give up
    assert len(subs) == 1  # whole-prompt fallback
    assert subs[0].marker == "Invoices"
    assert ss.load_decomposition(ss.task_id(PROMPT)) is None  # garbage is never cached


@pytest.mark.asyncio
async def test_rejected_attempt_feeds_error_back_to_the_retry(library):
    bad = json.dumps({"subtasks": [
        {"template_prompt": "select {{business}}",
         "values": {"business": "NOT IN THE PROMPT LLC"}, "is_save_step": True},
    ]})
    llm = SeqLLM(bad, GOOD_REPLY)
    subs = await decompose.get_decomposition(PROMPT, llm=llm, marker="Invoices")

    assert llm.calls == 2
    assert len(subs) == 2  # the corrected retry was accepted, not the fallback
    # Attempt 1 is clean; attempt 2's user message carries the specific rejection.
    first_user = str(llm.seen[0][-1].content)
    retry_user = str(llm.seen[1][-1].content)
    assert "REJECTED" not in first_user
    assert "REJECTED" in retry_user
    assert "NOT IN THE PROMPT LLC" in retry_user
    # The corrected split is cached like any tier-4 success.
    cached = ss.load_decomposition(ss.task_id(PROMPT))
    assert cached and cached["source"] == "llm"


@pytest.mark.asyncio
async def test_token_value_mismatch_rejected(library):
    bad = json.dumps({"subtasks": [
        {"template_prompt": "select {{business}} and {{ghost}}",
         "values": {"business": "290 CREW LIMITED"}, "is_save_step": True},
    ]})
    subs = await decompose.get_decomposition(PROMPT, llm=StubLLM(bad), marker="Invoices")
    assert len(subs) == 1  # fallback


# ------------------------------- the dropped-wording guard -------------------------------
# Regression: the NI clause of the payroll add-employee task ("NI number should be AB followed
# by a random 6 digit number and end with C") silently vanished from the split — a GENERATIVE
# instruction is neither a tokenizable literal (the hallucination guard bars it) nor a button
# label, so the LLM omitted it and the employee would have saved with a blank NI number.

NI_PROMPT = ("add employee with join date 06/04/2026, NI number should be AB followed by a "
             "random 6 digit number and end with C, NI category A, and save")

NI_DROPPED = json.dumps({"subtasks": [
    {"template_prompt": "add employee with join date {{join_date}}",
     "values": {"join_date": "06/04/2026"}, "is_save_step": False},
    {"template_prompt": "NI category {{ni_category}}, and save",
     "values": {"ni_category": "A"}, "is_save_step": True},
]})

NI_KEPT = json.dumps({"subtasks": [
    {"template_prompt": "add employee with join date {{join_date}}",
     "values": {"join_date": "06/04/2026"}, "is_save_step": False},
    {"template_prompt": "NI number should be AB followed by a random 6 digit number and end "
                        "with C, NI category {{ni_category}}, and save",
     "values": {"ni_category": "A"}, "is_save_step": True},
]})


def test_coverage_gap_spots_a_dropped_clause_and_passes_a_complete_split():
    raw = json.loads(NI_DROPPED)["subtasks"]
    gap = decompose.coverage_gap(raw, NI_PROMPT)
    assert gap and "random 6 digit number" in gap
    assert decompose.coverage_gap(json.loads(NI_KEPT)["subtasks"], NI_PROMPT) is None


def test_coverage_gap_tolerates_reworded_connectives():
    # The decomposer rewrites small joining words ("now" -> "Then"); only a long dropped run
    # is a lost instruction, so this must NOT be rejected.
    prompt = "click add request, now click save, and then click send"
    raw = [{"template_prompt": "click add request", "values": {}},
           {"template_prompt": "Then click save", "values": {}},
           {"template_prompt": "and then click send", "values": {}}]
    assert decompose.coverage_gap(raw, prompt) is None


def test_coverage_gap_is_not_fooled_by_vocabulary_reuse():
    # "number"/"date"/"end" all recur elsewhere in the prompt; a bag-of-words check would call
    # the dropped clause covered by its own words appearing in other subtasks.
    raw = json.loads(NI_DROPPED)["subtasks"] + [
        {"template_prompt": "set end date and reference number and random digit", "values": {}}]
    assert decompose.coverage_gap(raw, NI_PROMPT)


@pytest.mark.asyncio
async def test_dropped_wording_rejected_then_corrected_on_retry(library):
    llm = SeqLLM(NI_DROPPED, NI_KEPT)
    subs = await decompose.get_decomposition(NI_PROMPT, llm=llm, marker=None)

    assert llm.calls == 2
    assert "random 6 digit number" in subs[1].template_prompt
    retry_user = str(llm.seen[1][-1].content)
    assert "REJECTED" in retry_user and "dropped wording" in retry_user


@pytest.mark.asyncio
async def test_last_attempt_keeps_an_incomplete_split_over_whole_prompt_fallback(library):
    # Both attempts drop the clause. A structurally sound 2-node split still runs the task far
    # better than one giant whole-prompt node, so it is kept (loudly) rather than discarded.
    llm = SeqLLM(NI_DROPPED)
    subs = await decompose.get_decomposition(NI_PROMPT, llm=llm, marker=None)

    assert llm.calls == 3  # strict on 1 and 2, advisory on the last
    assert len(subs) == 2  # not the 1-node fallback


def test_new_tab_wording_implies_tab_url_but_in_app_links_are_left_alone():
    # Observed live: the fakenamegenerator subtask came back with tab_url omitted, which would
    # have navigated the APP page to that site and stranded every later subtask.
    assert decompose._implied_tab_url(
        "Open a new tab, go to https://www.fakenamegenerator.com/ , click Generate"
    ) == "https://www.fakenamegenerator.com/"
    # In-app deep links carry no new-tab wording and must NOT become helper tabs.
    assert decompose._implied_tab_url(
        "Navigate directly to https://appuat.actingoffice.com/admin/clients/business/69bd") is None
    assert decompose._implied_tab_url("open a new tab and check the totals") is None


@pytest.mark.asyncio
async def test_missing_tab_url_is_recovered_from_the_subtask_wording(library):
    reply = json.dumps({"subtasks": [
        {"template_prompt": "go to Bookkeeping module, search and select {{business}} "
                            "business name",
         "values": {"business": "290 CREW LIMITED"}, "is_save_step": False},
        {"template_prompt": "Open a new tab, go to https://www.fakenamegenerator.com/ and "
                            "note the identity",
         "values": {}, "is_save_step": False},
        {"template_prompt": "add invoice for customer {{customer}} with qty {{qty}} and "
                            "click save",
         "values": {"customer": "Suresh Gopi", "qty": "5"}, "is_save_step": True},
    ]})
    prompt = (PROMPT + " Open a new tab, go to https://www.fakenamegenerator.com/ and note "
              "the identity")
    subs = await decompose.get_decomposition(prompt, llm=StubLLM(reply), marker="Invoices")

    assert subs[1].tab_url == "https://www.fakenamegenerator.com/"
    assert subs[0].tab_url is None and subs[2].tab_url is None


@pytest.mark.asyncio
async def test_no_llm_no_cache_gives_whole_prompt_fallback(library):
    subs = await decompose.get_decomposition("just do the thing", llm=None, marker=None)
    assert len(subs) == 1
    assert subs[0].template_prompt == "just do the thing"
    assert subs[0].marker is None
    assert subs[0].kind == "action"
    assert subs[0].fallback is True


async def test_fallback_blob_never_takes_loop_kind(library):
    """A whole-task blob contains repeat+stop cues somewhere in its text (the pay-forecast
    retry clause), which classified the ENTIRE mega-task as a loop — the loop framing then
    told the agent it was mid-iteration and it hunted end-of-task controls from step 1
    (runs 20260805_155515/160602: 'Download' on /admin) and once declared the whole task
    done because the first repeat-until's stop condition held on a wandered-to page.
    Fallback blobs keep only the judge verdict; loop collapses to neutral action."""
    prompt = ("go to the section and add the record. keep repeating the check until the "
              "rows are shown. then download the report and save")
    subs = await decompose.get_decomposition(prompt, llm=None, marker=None)
    assert len(subs) == 1
    assert subs[0].fallback is True
    assert subs[0].kind == "action"


# ------------------------------- node kinds (action | judge) -------------------------------


def test_node_kind_heuristic():
    # Verification wording -> judge (cognitive: always LLM, never cached).
    assert decompose.node_kind("verify the CC field matches", None) == "judge"
    assert decompose.node_kind("Check that the mail is not sent", None) == "judge"
    # "remember that mail" is not producer wording (_PRODUCES_NOTED_RE wants a
    # determiner it can bind a value to: "remember THE mail"), so the judge net holds it.
    assert decompose.node_kind("remember that mail", None) == "judge"
    # ...but bare NOTING is an action now, not a judge: nothing here is compared, and a
    # replayed extract step re-reads the value live. See
    # test_producer_wording_is_an_action_not_a_judge. Add any verification verb and both
    # go back to judge (test_verification_wording_still_judges).
    assert decompose.node_kind("note the currently selected option", None) == "action"
    assert decompose.node_kind("capture the names of both users", None) == "action"
    assert decompose.node_kind("Confirm that the dropdown updates", None) == "judge"
    assert decompose.node_kind("make sure the panel opens", None) == "judge"
    assert decompose.node_kind("see if the icon works", None) == "judge"
    # Record types and action wording never classify as judge: "credit note" is a noun,
    # "check the option" is a click on a checkbox.
    assert decompose.node_kind("add credit note,select a customer and click save",
                               None) == "action"
    assert decompose.node_kind('check the option "no sharing" and submit', None) == "action"
    assert decompose.node_kind("go to inputs section,select sales", None) == "action"
    # The disambiguated RTI period pick: "period"/"dropdown"/"top bar" must not drift
    # into the judge net — a deterministic UI pick should stay cacheable.
    assert decompose.node_kind("using the period dropdown in the top bar, set the period "
                               "to May-26, then click Save & Next", None) == "action"
    # A marker-owning subtask is ALWAYS action — its network gate is machine ground truth.
    assert decompose.node_kind("verify and save the record", "Invoices") == "action"
    # An aux-tab subtask is action even with observational wording: its replayed extract
    # step re-reads the live DOM, so the observation stays fresh without the LLM.
    assert decompose.node_kind("note the title of the top result", None,
                               tab_url="https://duckduckgo.com") == "action"
    # An explicit declaration wins over the heuristic.
    assert decompose.node_kind("go to the reviews section", None, declared="judge") == "judge"
    assert decompose.node_kind("verify it worked", None, declared="action") == "action"
    assert decompose.node_kind("note the top result", None, declared="judge",
                               tab_url="https://duckduckgo.com") == "judge"


# The two RTI employee loops in the ORIGINAL wording (now carried by
# payroll_rti_process_old / payroll_food_limited_e2e_rti): imperative actions with the
# verification folded inside. Judge framing made the agent declare them done after ONE
# Save & Next (the observed wrong-employee bug) — they must classify "loop". Their
# "check that" clause used to be load-bearing: the 07-29 reword dropped it and the pass
# silently became a cacheable fixed-click action.
LOOP_OWEN = ("Process the existing employees one at a time by clicking Save & Next, and "
             "after each click check that the next employee has loaded, if the Save & "
             "Next button is disabled, Move on to the next employee. stopping as soon as "
             "{{employee}} is the employee shown")
LOOP_DAVID = ("Continue clicking Save & Next one employee at a time in the same way, and "
              "after each click check that the next employee has loaded, if the Save and "
              "next is disabled move on to the next employee, until {{employee}} is the "
              "employee shown")

# The same two loops in the 2026-08-05 imperative rewrite (live payroll_rti_process
# wording, verbatim templates): NO judge vocabulary anywhere. The judge-first gate
# demoted them to cacheable actions, and the frozen Save & Next replay saved the
# stop-target employee — they must classify "loop" on repetition + stop cues alone.
LOOP_ALAN = (
    "Now process the existing employees one at a time by repeating the following steps "
    "for each employee: first read the name of the employee currently shown; if the "
    "employee shown is {{employee}}, stop repeating and do not click Save & Next again; "
    "otherwise, if the error '{{error_message}}' is shown, click Add Payment, set Amount "
    "to {{amount}}, and click Save & Next; if Save & Next is disabled for this employee, "
    "do not click it and instead select the next employee in the list directly; in all "
    "other cases click Save & Next and wait until the next employee has fully loaded "
    "before doing anything else. Keep repeating those steps until {{employee}} is the "
    "employee shown")
LOOP_BRUCE = (
    "After saving, continue processing employees one at a time by repeating exactly the "
    "same steps as before: read the name of the employee currently shown; if the "
    "employee shown is {{employee}}, stop repeating and do not click Save & Next again; "
    "otherwise handle the minimum wage error and a disabled Save & Next the same way as "
    "before, and in all other cases click Save & Next and wait for the next employee to "
    "load. Keep repeating until {{employee}} is the employee shown")


def test_node_kind_loop_detection():
    # Judge phrase + iteration cues, with the judge phrase NOT the head directive -> loop.
    assert decompose.node_kind(LOOP_OWEN, None) == "loop"
    assert decompose.node_kind(LOOP_DAVID, None) == "loop"
    # A LEADING judge directive stays judge even when WHAT it checks iterates.
    assert decompose.node_kind(
        "Check that entries do not repeat across pages and each page loads",
        None) == "judge"
    assert decompose.node_kind("verify that each filter narrows the results",
                               None) == "judge"
    # Cue-free verification stays judge; cue-free iteration stays action.
    assert decompose.node_kind("verify the CC field matches", None) == "judge"
    assert decompose.node_kind(
        "Then go to Payroll & RTI, using the period dropdown in the top bar, change the "
        "period to the next month, and click Save & Next 3 times", None) == "action"
    # Judge-free loops: repetition cue + stop cue is a loop signature with NO judge
    # phrase (the 2026-08-05 rewrite; a frozen replay of it saved the stop-target
    # employee when this classified action).
    assert decompose.node_kind(LOOP_ALAN, None) == "loop"
    assert decompose.node_kind(LOOP_BRUCE, None) == "loop"
    # Either cue alone is everyday action filler, not a loop: a fixed click count with
    # "after each click ... stopping", and a bare "wait until X loads".
    assert decompose.node_kind(
        "change the date to the next month ({{date}}), and click Save & Next exactly "
        "{{times}} times, waiting for the screen to update after each click and "
        "stopping after the third click", None) == "action"
    assert decompose.node_kind(
        "click Save & Next and wait until the next employee has fully loaded",
        None) == "action"
    # The 2026-08-11 counter rewrite of the e2e's employee passes rides on this:
    # "exactly N clicks ... after each click" has neither a repeat cue ("for each"
    # requires adjacency) nor a stop cue, so both slices stay recordable actions.
    assert decompose.node_kind(
        "Click Save & Next for the next 5 employees: exactly 5 clicks, waiting for "
        "the next employee to fully load after each click.", None) == "action"
    assert decompose.node_kind(
        "Then click Save & Next for the next 14 employees in the same way: exactly "
        "14 more clicks, waiting for the next employee to fully load after each "
        "click.", None) == "action"
    # A MULTI-STEP iteration reads the same to the classifier as a single-click one, and
    # the "rest of the employees" phrasing carries neither cue — run 20260824_165824 filled
    # employee 1's three portal dialogs, clicked Next once, and closed the tab on employee
    # 2's freshly loaded empty form, exactly as the wording said.
    assert decompose.node_kind(
        "click the + button next to payment, enter 4000 in the amount field, click Save. "
        "Click Next for rest of the employees. Close this tab.", None) == "action"
    assert decompose.node_kind(
        "Now do this for each employee in the numbered list on the left, starting with "
        "the one already open: click the + button next to payment, enter 4000 in the "
        "amount field, click Save. Then click Next to load the following employee and "
        "repeat all of the above for them, until every employee in that list has been "
        "done. Then close this tab.", None) == "loop"
    # Marker precedence is unchanged: machine ground truth caches safely.
    assert decompose.node_kind(LOOP_OWEN, "Payroll") == "action"
    assert decompose.node_kind(LOOP_ALAN, "Payroll") == "action"
    # Explicit declarations still win in both directions.
    assert decompose.node_kind("go to the reviews section", None,
                               declared="loop") == "loop"
    assert decompose.node_kind(LOOP_OWEN, None, declared="judge") == "judge"


def test_conditional_guard_wording():
    # Leading "If" = branch guard: whether its actions run at all depends on live page
    # state, so hybrid runs it live and never caches it.
    assert decompose.is_conditional_guard(
        "If you see an error 'The employer pay is lower than the minimum wage rate', "
        "click Add Payment, set Amount to {{amount}}, and click Save & Next")
    assert decompose.is_conditional_guard("if a pop up appears, click Process")
    assert decompose.is_conditional_guard("Then, if an error shows, dismiss it")
    # An EMBEDDED conditional is a footnote to an unconditional procedure: cacheable.
    assert not decompose.is_conditional_guard(
        "go to Payroll & RTI, click Save & Next. If a pop up appears, select 'don't "
        "show this again' and click Process")
    assert not decompose.is_conditional_guard(
        "add invoice for customer {{customer}} and click save")


def test_downloads_file_wording():
    assert decompose.downloads_file("select download, select PDF")
    assert decompose.downloads_file("select download and select Excel")
    assert decompose.downloads_file("Export the report to csv")
    assert not decompose.downloads_file("go to employee section")
    assert not decompose.downloads_file(
        "note the generated identity; remember Name and Address")
    assert not decompose.downloads_file("add invoice for customer {{customer}} and save")


def test_consumes_noted_data_matches_consumers_not_producers():
    # CONSUMER wording — the segment fills the app with run-noted values, so a cached
    # replay would type the authoring run's stale ones (the observed Add Employee bug).
    assert decompose.consumes_noted_data(
        "Add Employee using the noted generated name and address, join date "
        "{{join_date}}, NI number {{ni_number}}, and save")
    assert decompose.consumes_noted_data("fill the form with the generated name")
    assert decompose.consumes_noted_data("enter the id noted earlier into the search box")
    assert decompose.consumes_noted_data("search for the title from the previous step")
    assert decompose.consumes_noted_data("compare it with the remembered address")
    # Relative-clause reference to a record an earlier segment created (the live case
    # where the employee pick got parameterized to the word "download").
    assert decompose.consumes_noted_data(
        "Then go to pay forecast, search for employee which we added in the combobox "
        "search employee, change pay from Dec-26 to {{pay}}, select download, select PDF")
    assert decompose.consumes_noted_data("open the request that was created and verify")
    assert decompose.consumes_noted_data("select the newly added employee")
    # PRODUCER wording — its replayed extract re-reads the live DOM (never stale), so it
    # must keep its zero-LLM replay.
    assert not decompose.consumes_noted_data(
        "Open a new tab, go to https://www.fakenamegenerator.com/ , set Name set to "
        "{{name_set}} and Country to {{country}}, click Generate, and note the generated "
        "identity; remember Name and Address")
    assert not decompose.consumes_noted_data(
        "search DuckDuckGo for {{query}} and note the title of the top result")
    # Plain action wording and app-domain vocabulary stay replayable.
    assert not decompose.consumes_noted_data(
        "go to Payroll module, search and select {{business}} business name, "
        "go to employee section")
    assert not decompose.consumes_noted_data(
        "add invoice for customer {{customer}} and click save")
    assert not decompose.consumes_noted_data("open the recorded payment and click void")
    # A PRODUCER slice — the wording IS the noting instruction — must not trip the
    # usage-word branch on its own opening phrase ("From the generated identity ...").
    # Observed live 2026-08-12: the fakenamegenerator slice classified consumer and
    # stopped replaying whenever seg 0 authored (findings present).
    assert not decompose.consumes_noted_data(
        "Open a new tab and go to the generator site. From the generated identity, "
        "note and remember exactly these details for use in all later steps: the "
        "Name, the Gender (Male), the Address, and the Date of Birth.")
    # ... but a slice that produces AND unambiguously consumes stays a consumer.
    assert decompose.consumes_noted_data(
        "note down the reference, then enter the noted name into the search box")


JUDGE_PROMPT = "go to the reviews section. verify the mail is not sent"
JUDGE_REPLY = json.dumps({"subtasks": [
    {"template_prompt": "go to the reviews section", "values": {}, "is_save_step": False},
    {"template_prompt": "verify the mail is not sent", "values": {}, "is_save_step": False},
]})


@pytest.mark.asyncio
async def test_judge_kind_assigned_and_survives_the_cache(library):
    subs = await decompose.get_decomposition(JUDGE_PROMPT, llm=StubLLM(JUDGE_REPLY),
                                             marker=None)
    assert [s.kind for s in subs] == ["action", "judge"]
    cached = ss.load_decomposition(ss.task_id(JUDGE_PROMPT))
    assert [d["kind"] for d in cached["subtasks"]] == ["action", "judge"]
    # Tier-2 rebuild from the cache preserves the kinds.
    again = await decompose.get_decomposition(JUDGE_PROMPT, llm=None, marker=None)
    assert [s.kind for s in again] == ["action", "judge"]


@pytest.mark.asyncio
async def test_marker_overrides_judge_wording_on_the_save_subtask(library):
    # With a parent marker and no declared save step, the LAST subtask becomes the save
    # owner — and a marker-owning node is action even with verification wording.
    subs = await decompose.get_decomposition(JUDGE_PROMPT, llm=StubLLM(JUDGE_REPLY),
                                             marker="Reviews")
    assert subs[1].marker == "Reviews"
    assert [s.kind for s in subs] == ["action", "action"]


@pytest.mark.asyncio
async def test_fallback_is_judge_for_markerless_verification_task(library):
    subs = await decompose.get_decomposition(
        "verify that the report shows the review", llm=None, marker=None)
    assert len(subs) == 1 and subs[0].kind == "judge"


@pytest.mark.asyncio
async def test_cached_kind_rederived_from_wording(library):
    """A cached decomposition carries the CLASSIFIER'S old verdict, not an author's
    declaration: tier 2 re-derives kinds from wording so a classifier fix reaches every
    already-cached task without --redecompose (the live case: the RTI employee loops sat
    in the cache as "judge" and kept running as one-shot observations)."""
    prompt = "go to the section. " + LOOP_OWEN.replace("{{employee}}", "Owen Millar")
    tid = ss.task_id(prompt)
    ss.save_decomposition(tid, {
        "parent_prompt": " ".join(prompt.split()),
        "source": "llm",
        "created": "2026-07-27T00:00:00",
        "subtasks": [
            {"template_prompt": "go to the section.", "values": {}, "marker": None,
             "postcondition": None, "kind": "judge", "tab_url": None},
            {"template_prompt": LOOP_OWEN, "values": {"employee": "Owen Millar"},
             "marker": None, "postcondition": None, "kind": "judge", "tab_url": None},
        ],
    })
    subs = await decompose.get_decomposition(prompt, llm=None, marker=None)
    assert [s.kind for s in subs] == ["action", "loop"]


@pytest.mark.asyncio
async def test_spec_declared_kind_still_wins(library):
    """Tier 1 is author-maintained: an explicit yaml `kind` is a real declaration and is
    NOT re-derived — only cached (derived) kinds are advisory."""
    prompt = "go to the archive area. archive the oldest record"
    spec = TaskSpec(key="k", prompt=prompt, subtasks=(
        SubtaskDecl(prompt="go to the archive area."),
        SubtaskDecl(prompt="archive the oldest record", kind="judge"),
    ))
    subs = await decompose.get_decomposition(prompt, llm=None, spec=spec)
    assert [s.kind for s in subs] == ["action", "judge"]


# ------------------------------- aux-tab subtasks (tab_url) -------------------------------


AUX_PROMPT = ("go to Bookkeeping module, search and select 290 CREW LIMITED business "
              "name. search DuckDuckGo for Acting Office and note the top result")
AUX_REPLY = json.dumps({"subtasks": [
    {"template_prompt": "go to Bookkeeping module, search and select {{business}} "
                        "business name",
     "values": {"business": "290 CREW LIMITED"}, "is_save_step": False},
    {"template_prompt": "search DuckDuckGo for {{q}} and note the top result",
     "values": {"q": "Acting Office"}, "is_save_step": False,
     "tab_url": "https://duckduckgo.com"},
]})


@pytest.mark.asyncio
async def test_tab_url_flows_llm_to_subtask_and_cache_roundtrip(library):
    llm = StubLLM(AUX_REPLY)
    subs = await decompose.get_decomposition(AUX_PROMPT, llm=llm, marker=None)
    assert subs[1].tab_url == "https://duckduckgo.com"
    # "note the ..." wording alone would make a judge node; tab_url keeps it cacheable.
    assert [s.kind for s in subs] == ["action", "action"]
    cached = ss.load_decomposition(ss.task_id(AUX_PROMPT))
    assert cached["subtasks"][1]["tab_url"] == "https://duckduckgo.com"
    # Tier-2 rebuild from the cache preserves it, zero LLM.
    again = await decompose.get_decomposition(AUX_PROMPT, llm=None, marker=None)
    assert again[1].tab_url == "https://duckduckgo.com"
    assert again[1].kind == "action"
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_old_cache_without_tab_url_still_loads(library):
    """Decompositions cached before the field existed must build unchanged."""
    data = {"parent_prompt": "do the legacy thing", "source": "llm",
            "subtasks": [{"template_prompt": "do the legacy thing", "values": {},
                          "marker": None, "postcondition": None, "kind": "action"}]}
    ss.save_decomposition(ss.task_id("do the legacy thing"), data)
    subs = await decompose.get_decomposition("do the legacy thing", llm=None)
    assert subs[0].tab_url is None and subs[0].kind == "action"


@pytest.mark.asyncio
async def test_derived_match_keeps_tab_url(library):
    await decompose.get_decomposition(AUX_PROMPT, llm=StubLLM(AUX_REPLY))
    new_prompt = (AUX_PROMPT.replace("290 CREW LIMITED", "ACME LTD")
                  .replace("Acting Office", "Best Beans"))
    subs = await decompose.get_decomposition(new_prompt, llm=None)
    assert subs[1].values == {"q": "Best Beans"}
    assert subs[1].tab_url == "https://duckduckgo.com"


@pytest.mark.asyncio
async def test_malformed_tab_url_rejected_then_corrected(library):
    """A structurally broken tab_url (no scheme) is mechanically rejected with the reason
    fed back, exactly like a hallucinated value — a garbled URL must never silently run."""
    prompt = "search DuckDuckGo for the business and note the top result"
    sub = {"template_prompt": prompt, "values": {}, "is_save_step": False}
    llm = SeqLLM(json.dumps({"subtasks": [{**sub, "tab_url": "duckduckgo.com"}]}),
                 json.dumps({"subtasks": [{**sub, "tab_url": "https://duckduckgo.com"}]}))
    subs = await decompose.get_decomposition(prompt, llm=llm)
    assert llm.calls == 2
    assert len(subs) == 1 and subs[0].tab_url == "https://duckduckgo.com"
    assert "tab_url" in str(llm.seen[1][-1].content)


# ------------------------------- declared verify checks -------------------------------


@pytest.mark.asyncio
async def test_spec_declared_verify_reaches_subtasks_and_cache_drops_it(library):
    """Tier 1 threads each slice's parsed checks onto its Subtask; the saved
    decomposition cache deliberately does NOT carry them (checks re-attach from
    the spec every load, so a stale cache can never resurrect old checks)."""
    from automation.pipeline.checks import Check

    prompt = "go to the payroll module. add the employee and save"
    spec = TaskSpec(key="k", prompt=prompt, subtasks=(
        SubtaskDecl(prompt="go to the payroll module.",
                    verify=(Check(kind="url_contains", arg="payroll"),)),
        SubtaskDecl(prompt="add the employee and save"),
    ))
    subs = await decompose.get_decomposition(prompt, llm=None, spec=spec)
    assert subs[0].verify == (Check(kind="url_contains", arg="payroll"),)
    assert subs[1].verify is None

    cached = ss.load_decomposition(ss.task_id(prompt))
    assert cached and all("verify" not in d for d in cached["subtasks"])
    rebuilt = decompose._build_subtasks(cached["subtasks"], None, trust_kind=False)
    assert all(s.verify is None for s in rebuilt)


@pytest.mark.asyncio
async def test_spec_declared_probe_reaches_subtasks_and_cache_drops_it(library):
    """Tier 1 threads a conditional slice's probe onto its Subtask; the saved cache
    deliberately does NOT carry it (same re-attach-from-spec contract as verify)."""
    from automation.pipeline.checks import Check

    prompt = ("go to the payroll module. "
              "If a popup appears, tick 'Don't show this again' and click Process.")
    probe = Check(kind="text_visible", arg="Don't show this again", timeout_s=3.0)
    spec = TaskSpec(key="k", prompt=prompt, subtasks=(
        SubtaskDecl(prompt="go to the payroll module."),
        SubtaskDecl(prompt="If a popup appears, tick 'Don't show this again' and "
                           "click Process.", probe=probe),
    ))
    subs = await decompose.get_decomposition(prompt, llm=None, spec=spec)
    assert subs[0].probe is None
    assert subs[1].probe == probe

    cached = ss.load_decomposition(ss.task_id(prompt))
    assert cached and all("probe" not in d for d in cached["subtasks"])
    rebuilt = decompose._build_subtasks(cached["subtasks"], None, trust_kind=False)
    assert all(s.probe is None for s in rebuilt)


# ------------------- producer wording is an action, not a judge -------------------
# The asymmetry this closes: the identity slice and the OTP slice carry the SAME judge
# phrase ("note and remember") and do the SAME job — capture a value later slices consume.
# The identity slice classified action only because it names an absolute URL, so the
# tab_url short-circuit fired before the judge regex was reached; the in-app OTP slice fell
# through to judge and paid 176s / 140k tokens re-observing itself every run
# (20260824_165824). The carve-out was always this rule, scoped to a foreign origin.

_OTP_SLICE = ("Now go to Data Request, and on the top row (S.No. 1, the newest request) "
              "click the ref. no. to open the Payroll Review panel, and click Get OTP, "
              "note and remember the OTP, close the review panel.")
_IDENTITY_SLICE = ("Open a new tab and go to https://www.fakenamegenerator.com/"
                   "gen-male-gd-uk.php. From the generated identity, note and remember "
                   "exactly these details for use in all later steps: the Name, the "
                   "Gender (Male), the Address, and the Date of Birth.")


def test_producer_wording_is_an_action_not_a_judge():
    assert decompose.node_kind(_OTP_SLICE, None) == "action"
    # Same wording, same verdict, with and without the aux-tab short-circuit.
    assert decompose.node_kind(_IDENTITY_SLICE, None) == "action"
    assert decompose.node_kind(
        _IDENTITY_SLICE, None,
        tab_url="https://www.fakenamegenerator.com/gen-male-gd-uk.php") == "action"
    assert decompose.produces_noted_data(_OTP_SLICE)
    assert decompose.produces_noted_data(_IDENTITY_SLICE)


def test_copy_wording_is_a_producer_now_that_copy_text_captures():
    """2026-08-25: the OTP producer slice was reworded to "click Get OTP, copy the 6 digit
    number (OTP)". `copy` was not a noting verb, so produces_noted_data returned False and
    the commit guard that refuses to cache a producer whose recording carries no capture
    step never ran — a run that copied nothing would have cached a producer that notes
    nothing, leaving every consumer's binding unresolvable. copy_text stamps the same
    extract channel extract_data does, so copying IS noting."""
    copy_slice = ("Now go to Data Request, and on the top row (S.No. 1, the newest "
                  "request) click the ref. no. to open the Payroll Review panel, and "
                  "click Get OTP, copy the 6 digit number (OTP), close the review panel.")
    assert decompose.produces_noted_data(copy_slice)
    assert decompose.node_kind(copy_slice, None) == "action"
    # A producer is still NOT a consumer without an unambiguous consumer phrase.
    assert not decompose.consumes_noted_data(copy_slice)
    # ...and the widening reaches only the noting shape: an ordinary click is untouched.
    assert not decompose.produces_noted_data("click Save and close the dialog")


def test_the_paste_consumer_slice_still_consumes():
    """Its counterpart: the reworded consumer must keep every classification the loop
    reads off it — announced tab included, or close_extra_tabs sweeps the tab the next
    subtask needs."""
    consumer = ("Now click on the external link button next to the ref. no, A new tab "
                "will be open click Already have an OTP, paste the OTP from the previous "
                "step into the first code box and click proceed Securely.")
    assert decompose.consumes_noted_data(consumer)
    assert not decompose.produces_noted_data(consumer)
    assert decompose.announces_new_tab(consumer)
    assert not decompose.is_conditional_guard(consumer)
    assert decompose.node_kind(consumer, None) == "action"


def test_verification_wording_still_judges():
    """The line the producer rule must not cross: a COMPARISON is what a recording cannot
    replay, so anything that verifies stays a judge — including a slice that notes a value
    AND checks it."""
    assert decompose.node_kind("verify the CC field matches", None) == "judge"
    assert decompose.node_kind(
        "note the currently selected option and confirm it is Account Manager",
        None) == "judge"
    assert decompose.node_kind(
        "capture the balance and check that it equals the invoice total", None) == "judge"
    assert decompose.node_kind("Check that the mail is not sent", None) == "judge"
    # ...and the registry's own verification slice, which must not have moved.
    assert decompose.node_kind(
        "Now go to Data Request, and on the top row (S.No. 1, the newest request) click "
        "the ref. no. link to open the Payroll Review panel, click Verify all.",
        None) == "judge"


def test_producer_wording_still_loses_to_loop_and_marker():
    """Precedence is unchanged: iteration and machine ground truth both outrank it."""
    assert decompose.node_kind(
        "note the employee shown, click Save & Next, and keep repeating until the last "
        "employee is reached", None) == "loop"
    assert decompose.node_kind(_OTP_SLICE, "Payroll") == "action"
    assert decompose.node_kind(_OTP_SLICE, None, declared="judge") == "judge"
