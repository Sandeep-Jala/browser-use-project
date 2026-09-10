"""Semantic router: alias tier, embedding candidacy gates, LLM slot-verify, learning."""
import json

import pytest

from automation.pipeline import router
from automation.pipeline import subtask_store as ss
from automation.pipeline.decompose import Subtask
from automation.pipeline.router import Route, route


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(ss, "LIBRARY_MANIFEST", tmp_path / "library" / "manifest.json")
    (tmp_path / "library").mkdir()
    return tmp_path


class StubEmbedder:
    """Deterministic 'embeddings': a fixed vector per known text, orthogonal otherwise."""

    def __init__(self, table):
        self.table = table
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [self.table.get(t, [0.0, 0.0, 1.0]) for t in texts]


class StubLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1

        class R:
            completion = self.reply

        return R()


CANON_PROMPT = "add invoice, select a customer {{customer}}"
NEW_PROMPT = "create an invoice for client {{client}}"
CTX = "/books/clients/*/inputs/sales"


def _seed_canonical(sid="canon1", prompt=CANON_PROMPT, params=None, context=CTX):
    ss.code_path(sid).write_text("async def run(api):\n    await api.wait(1.0)\n")
    ss.update_manifest(sid, prompt, create=True, context=context,
                       params=params if params is not None else {"customer": "Suresh Gopi"})
    return sid


def _sub(prompt=NEW_PROMPT, values=None):
    return Subtask(index=0, template_prompt=prompt,
                   values=values if values is not None else {"client": "Mr Jones"})


def _embedder(sim=0.95):
    # canonical text ~aligned with the query at cosine `sim`.
    return StubEmbedder({
        router._routable_text(CANON_PROMPT): [1.0, 0.0, 0.0],
        router._routable_text(NEW_PROMPT): [sim, (1 - sim**2) ** 0.5, 0.0],
    })


GOOD_VERIFY = json.dumps({"same": True, "slots": {"client": "customer"}})


async def test_semantic_match_verifies_learns_alias_and_remaps_values(stores):
    sid = _seed_canonical()
    llm = StubLLM(GOOD_VERIFY)
    r = await route(_sub(), "alias1", CTX, llm, embedder=_embedder())
    assert isinstance(r, Route) and r.sid == sid and r.via == "semantic"
    assert r.values == {"customer": "Mr Jones"}          # re-keyed to canonical params
    assert llm.calls == 1
    # The alias was learned: the same wording now routes with ZERO embed/LLM cost.
    llm2 = StubLLM(GOOD_VERIFY)
    r2 = await route(_sub(values={"client": "Acme"}), "alias1", CTX, llm2, embedder=None)
    assert r2 is not None and r2.via == "alias" and r2.values == {"customer": "Acme"}
    assert llm2.calls == 0


async def test_verifier_rejection_is_cached_negative(stores):
    _seed_canonical()
    llm = StubLLM(json.dumps({"same": False, "slots": {}}))
    assert await route(_sub(), "alias1", CTX, llm, embedder=_embedder()) is None
    assert llm.calls == 1
    # Second attempt: the negative verdict is cached — no second LLM call.
    assert await route(_sub(), "alias1", CTX, llm, embedder=_embedder()) is None
    assert llm.calls == 1


async def test_low_similarity_or_margin_refuses_without_llm(stores):
    _seed_canonical()
    llm = StubLLM(GOOD_VERIFY)
    assert await route(_sub(), "a", CTX, llm, embedder=_embedder(sim=0.5)) is None
    assert llm.calls == 0                                # below the floor: not even asked
    # Two near-identical candidates -> ambiguous -> refused.
    _seed_canonical(sid="canon2", prompt="add invoice, select the customer {{customer}}")
    emb = _embedder()
    emb.table[router._routable_text(
        "add invoice, select the customer {{customer}}")] = [1.0, 0.0, 0.0]
    assert await route(_sub(), "a", CTX, llm, embedder=emb) is None
    assert llm.calls == 0


async def test_context_gate_and_no_embedder(stores, monkeypatch):
    _seed_canonical(context="/somewhere/else")
    llm = StubLLM(GOOD_VERIFY)
    assert await route(_sub(), "a", CTX, llm, embedder=_embedder()) is None
    _seed_canonical(sid="canon3")                        # right context now
    # No local model available -> alias-only mode (never let a test download the real one).
    monkeypatch.setattr(router, "_default_embedder", lambda name: None)
    assert await route(_sub(), "a", CTX, llm, embedder=None) is None


async def test_bad_slot_mapping_rejected(stores):
    _seed_canonical()
    # Empty mapping AND the default ("Suresh Gopi") is nowhere in the new wording:
    # the param is neither tokenized nor stated -> never replay guessed values.
    llm = StubLLM(json.dumps({"same": True, "slots": {}}))
    assert await route(_sub(), "a", CTX, llm, embedder=_embedder()) is None


async def test_unmapped_param_fills_from_verbatim_default(stores):
    """A canonical param (e.g. a parameterized find_click label "View all") needs no slot
    when the new wording states the same value literally — the default fills it."""
    sid = _seed_canonical(prompt='click the {{label}} icon in Reviews',
                          params={"label": "View all"})
    new = "clik the View all icon in the Reviews area"
    emb = StubEmbedder({
        router._routable_text('click the {{label}} icon in Reviews'): [1.0, 0.0, 0.0],
        router._routable_text(new): [0.97, 0.24, 0.0],
    })
    llm = StubLLM(json.dumps({"same": True, "slots": {}}))
    r = await route(Subtask(index=0, template_prompt=new), "a2", CTX, llm, embedder=emb)
    assert r is not None and r.sid == sid
    assert r.values == {"label": "View all"}
    # The learned alias replays the same fill with zero LLM/embedding cost.
    r2 = await route(Subtask(index=0, template_prompt=new), "a2", CTX, None, embedder=None)
    assert r2 is not None and r2.via == "alias" and r2.values == {"label": "View all"}


async def test_parameterless_canonical_routes_with_empty_values(stores):
    sid = _seed_canonical(prompt="go to inputs section, select sales", params={})
    emb = StubEmbedder({
        router._routable_text("go to inputs section, select sales"): [1.0, 0.0, 0.0],
        router._routable_text("open the sales area under inputs"): [0.97, 0.24, 0.0],
    })
    llm = StubLLM(json.dumps({"same": True, "slots": {}}))
    r = await route(Subtask(index=0, template_prompt="open the sales area under inputs"),
                    "a", CTX, llm, embedder=emb)
    assert r is not None and r.sid == sid and r.values == {}


async def test_stale_alias_falls_through(stores):
    ss.save_alias("alias1", {"sid": "gone", "slots": {}})
    # Canonical archived/missing -> alias is stale -> no route (and no crash).
    assert await route(_sub(), "alias1", CTX, None, embedder=None) is None


def test_vector_cache_reused_and_model_change_recomputes(stores):
    _seed_canonical()
    emb = _embedder()
    texts = {"canon1": router._routable_text(CANON_PROMPT)}
    v1 = router._candidate_vectors(emb, "m1", texts)
    assert emb.calls == 1
    v2 = router._candidate_vectors(emb, "m1", texts)     # cache hit: no new embed call
    assert emb.calls == 1 and v2 == v1
    router._candidate_vectors(emb, "m2", texts)          # model changed: recompute
    assert emb.calls == 2
