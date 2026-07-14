"""decompose tests: tier order, caching, derived matching, and the hallucination guard.
The LLM is stubbed with canned replies; derived matching is asserted to use NO LLM."""
import json

import pytest

from automation.pipeline import decompose
from automation.pipeline import subtask_store as ss
from automation.pipeline import task_store as ts
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
    cached = ss.load_decomposition(ts.task_id(PROMPT))
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
    derived = ss.load_decomposition(ts.task_id(new_prompt))
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
    assert llm.calls == 2  # one retry, then give up
    assert len(subs) == 1  # whole-prompt fallback
    assert subs[0].marker == "Invoices"
    assert ss.load_decomposition(ts.task_id(PROMPT)) is None  # garbage is never cached


@pytest.mark.asyncio
async def test_token_value_mismatch_rejected(library):
    bad = json.dumps({"subtasks": [
        {"template_prompt": "select {{business}} and {{ghost}}",
         "values": {"business": "290 CREW LIMITED"}, "is_save_step": True},
    ]})
    subs = await decompose.get_decomposition(PROMPT, llm=StubLLM(bad), marker="Invoices")
    assert len(subs) == 1  # fallback


@pytest.mark.asyncio
async def test_no_llm_no_cache_gives_whole_prompt_fallback(library):
    subs = await decompose.get_decomposition("just do the thing", llm=None, marker=None)
    assert len(subs) == 1
    assert subs[0].template_prompt == "just do the thing"
    assert subs[0].marker is None
