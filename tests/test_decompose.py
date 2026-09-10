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


@pytest.mark.asyncio
async def test_an_authored_split_is_not_capped_at_max_subtasks(library):
    """MAX_SUBTASKS catches a RUNAWAY split — an LLM that kept emitting nodes. An author
    who wrote N slices in tasks.yaml meant N slices.

    The regression this guards: payroll_detailed_review_fps_part2 grew to 32 declared
    slices (ten monthly payrun+FPS passes) and every run of it silently became ONE
    whole-prompt blob — no per-segment gates, no cached segments, a logged ERROR nobody
    was watching. Over the ceiling the failure is total and quiet, which is why it is
    worth a test rather than a bigger constant."""
    over = decompose.MAX_SUBTASKS + 8
    slices = tuple(SubtaskDecl(prompt=f"do step {i}") for i in range(over))
    prompt = " ".join(d.prompt for d in slices)
    spec = TaskSpec(key="k", prompt=prompt, subtasks=slices)

    subs = await decompose.get_decomposition(prompt, llm=None, spec=spec)
    assert len(subs) == over
    assert not any(s.fallback for s in subs)
    assert subs[-1].template_prompt == f"do step {over - 1}"


@pytest.mark.asyncio
async def test_an_llm_split_is_still_capped_at_max_subtasks(library):
    """The waiver is for authored splits only — the hallucination guard is untouched."""
    over = decompose.MAX_SUBTASKS + 8
    prompt = " ".join(f"do step {i}" for i in range(over))
    reply = json.dumps({"subtasks": [
        {"template_prompt": f"do step {i}", "values": {}, "is_save_step": False}
        for i in range(over)
    ]})
    subs = await decompose.get_decomposition(prompt, llm=StubLLM(reply), marker=None)
    assert len(subs) == 1 and subs[0].fallback is True


@pytest.mark.asyncio
async def test_an_empty_authored_split_is_still_rejected(library):
    """Waiving the ceiling must not waive the floor: zero slices is still a broken
    declaration, not a zero-step task."""
    assert decompose._validate([], "anything", authored=True)


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


async def test_marker_overrides_judge_wording_on_the_save_subtask(library):
    # With a parent marker and no declared save step, the LAST subtask becomes the save
    # owner — and a marker-owning node is action even with verification wording.
    subs = await decompose.get_decomposition(JUDGE_PROMPT, llm=StubLLM(JUDGE_REPLY),
                                             marker="Reviews")
    assert subs[1].marker == "Reviews"
    assert [s.kind for s in subs] == ["action", "action"]


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








def test_remaining_alone_is_not_an_iteration_cue():
    """The cue is an iteration preposition PLUS a depleting set, not the bare adjective —
    a one-shot control that happens to be named "remaining" is still an action."""
    assert decompose.node_kind(
        "tick the remaining periods checkbox and click Save", None) == "action"


def test_a_marker_still_outranks_exhaustion_wording():
    """Machine ground truth caches safely regardless of wording (node_kind's marker rule)."""
    assert decompose.node_kind(
        "click Next for all of the remaining employees", "Payroll") == "action"


# ---------------- kind is DECLARED, never inferred (2026-08-28) ----------------
# Four regex nets used to read the prompt: _JUDGE_RE ("verify", "check that", even "note"),
# a leading "If", producer phrasing, and _NOTED_DATA_RE ("the noted ..."). Each could
# silently stop a segment being recorded — "tick the Select Employee checkbox ... and click
# Verify" was held out of the library because the BUTTON is named Verify. These tests pin
# the ABSENCE of that inference, which is the guarantee now.


def test_verification_wording_no_longer_makes_a_judge():
    for wording in (
        "Verify that the totals match the invoice",
        "Check that entries do not repeat across pages",
        "Make sure the balance is zero",
        "note and remember the OTP shown in the dialog",
        "tick the Select Employee checkbox and click Verify",
        "confirm the employee was saved",
    ):
        assert decompose.node_kind(wording, None) == "action", wording


def test_repeat_wording_no_longer_makes_a_loop():
    for wording in (
        "Click Save & Next one at a time until every employee is done",
        "keep clicking Next until the last row is shown",
        "click Next for all of the remaining employees",
        "repeat for each employee in the list",
    ):
        assert decompose.node_kind(wording, None) == "action", wording


def test_a_declared_kind_is_the_only_thing_that_counts():
    assert decompose.node_kind("anything at all", None, "judge") == "judge"
    assert decompose.node_kind("Verify the totals", None, "action") == "action"
    # 'loop' was removed; tasks.py rejects it on load, and node_kind treats any
    # unrecognised value as undeclared rather than guessing.
    assert decompose.node_kind("click Next until done", None, "loop") == "action"
    assert decompose.node_kind("Verify the totals", None, "Judge") == "action"


def test_marker_and_tab_url_no_longer_need_to_force_action():
    """They used to short-circuit the wording nets; action is simply the default now."""
    assert decompose.node_kind("Verify the save", "/Invoices") == "action"
    assert decompose.node_kind("note the top result", None, None, "https://example.com") == "action"



@pytest.mark.asyncio
async def test_spec_declared_write_waiver_reaches_subtasks_and_cache_drops_it(library):
    """Tier 1 threads a slice's declared write-rule waiver onto its Subtask; the saved
    cache deliberately does NOT carry it (same re-attach-from-spec contract as verify
    and probe — a stale cache must never resurrect a superseded waiver)."""
    prompt = ("go to the payroll module. "
              "click Submit; if it shows an error, click cancel.")
    spec = TaskSpec(key="k", prompt=prompt, subtasks=(
        SubtaskDecl(prompt="go to the payroll module."),
        SubtaskDecl(prompt="click Submit; if it shows an error, click cancel.",
                    allow_write_refusal=True),
    ))
    subs = await decompose.get_decomposition(prompt, llm=None, spec=spec)
    assert subs[0].allow_write_refusal is False
    assert subs[1].allow_write_refusal is True

    cached = ss.load_decomposition(ss.task_id(prompt))
    assert cached and all("allow_write_refusal" not in d for d in cached["subtasks"])
    rebuilt = decompose._build_subtasks(cached["subtasks"], None, trust_kind=False)
    assert all(s.allow_write_refusal is False for s in rebuilt)
