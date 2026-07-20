"""Semantic subtask router: map a differently-worded subtask to an existing library skill.

The library's identity is a hash of the exact (tokenized) wording — so "add invoice, select
a customer {{customer}}" and "create an invoice for client {{client}}" author two duplicate
entries even though they are the same procedure. This router closes that gap in three
tiers, cheapest first:

  1. ALIAS TABLE (free, deterministic): a wording that was semantically resolved ONCE is
     recorded under its own sid in library/aliases.json with the canonical sid and the
     slot mapping. Every later run is a dict lookup — the router gets faster with use.
  2. EMBEDDINGS (local, ~ms): a small local model (fastembed, lazy-loaded; the router
     degrades to alias-only when unavailable) ranks same-context library entries by
     cosine similarity of their value-stripped prompts. A candidate must clear a floor
     AND a margin over the runner-up — lookalikes with close scores are refused here.
  3. LLM VERIFY (one call, once per new wording): the expander model confirms "same
     procedure" and maps the new tokens onto the canonical params
     (ROUTER_VERIFY_SYSTEM_PROMPT). Values are then re-keyed to the canonical param
     names, so the skill's template alignment binds them by NAME. A yes is written to
     the alias table; a no costs nothing again for this wording+context (negative cache).

Safety: embeddings NEVER decide alone — the verifier is the gate against false friends,
and an accepted route is still judged by the segment gate at replay like any other skill.
A routed replay that fails falls back to authoring under the NEW wording's own sid
(never overwriting the canonical entry), which then shadows the alias.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Any

from browser_use.llm.messages import SystemMessage, UserMessage

from automation.pipeline import subtask_store as sstore
from automation.pipeline.adapt import _parse_json_reply
from automation.pipeline.prompts import ROUTER_VERIFY_SYSTEM_PROMPT

logger = logging.getLogger("framework.router")

# A candidate below the floor is not even worth a verify call; one that doesn't beat the
# runner-up by the margin is ambiguous (lookalike territory) and is refused outright.
SIM_FLOOR = 0.80
SIM_MARGIN = 0.03

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_embedders: dict[str, Any] = {}


def _default_embedder(model_name: str) -> Any | None:
    """Lazy local embedding model (fastembed/ONNX). None when unavailable — the router
    then works alias-only, and the deterministic tiers are unaffected."""
    if model_name in _embedders:
        return _embedders[model_name]
    try:
        from fastembed import TextEmbedding

        embedder = TextEmbedding(model_name=model_name)
    except Exception as exc:  # noqa: BLE001 - no local model = no semantic tier
        logger.warning("semantic router: local embedding model unavailable (%s); "
                       "alias-only routing", exc)
        embedder = None
    _embedders[model_name] = embedder
    return embedder


def _embed_texts(embedder: Any, texts: list[str]) -> list[list[float]]:
    return [[float(x) for x in v] for v in embedder.embed(texts)]


def _cos(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return num / den if den else 0.0


def _routable_text(template_prompt: str) -> str:
    """The wording that carries the procedure's meaning: slot NAMES are noise, so every
    {{token}} collapses to one placeholder before embedding."""
    return sstore.TOKEN_RE.sub("<value>", " ".join(template_prompt.split()).lower())


def _load_vectors() -> dict[str, Any]:
    p = sstore.embeddings_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001 - a corrupt cache just recomputes
        return {}


def _candidate_vectors(embedder: Any, model_name: str,
                       candidates: dict[str, str]) -> dict[str, list[float]]:
    """Vectors for {sid: routable_text}, cached in library/embeddings.json (recomputed
    wholesale when the model changes; orphans of archived entries are pruned)."""
    cache = _load_vectors()
    vectors: dict[str, list[float]] = \
        cache.get("vectors", {}) if cache.get("model") == model_name else {}
    missing = [sid for sid in candidates if sid not in vectors]
    if missing:
        for sid, vec in zip(missing, _embed_texts(embedder,
                                                  [candidates[s] for s in missing])):
            vectors[sid] = vec
        keep = {sid: v for sid, v in vectors.items()
                if sid in candidates or sid in sstore.load_manifest()}
        sstore._atomic_write_json(sstore.embeddings_path(),
                                  {"model": model_name, "vectors": keep})
        vectors = keep
    return vectors


@dataclass
class Route:
    sid: str                      # the canonical library entry to replay
    values: dict[str, str]        # canonical-param-keyed values for skill loading
    via: str                      # "alias" | "semantic"


def _filled_values(slots: dict[str, str], sub: Any,
                   canonical_params: dict[str, str]) -> dict[str, str] | None:
    """Concrete values for EVERY canonical param, or None.

    Slot-mapped params take the new wording's values. An UNMAPPED param may fall back to
    its recorded default only when that default appears VERBATIM in the new instruction —
    proof the new wording wants the same value (e.g. the canonical "click the {{label}}
    icon" with default "View all" matched by a wording that just says "click the View all
    icon"). A param that is neither tokenized nor stated stays uncovered -> no route:
    never replay guessed values."""
    values: dict[str, str] = {}
    prompt_lower = str(getattr(sub, "instantiated_prompt", "") or "").lower()
    for new, canon in slots.items():
        if new not in (sub.values or {}):
            return None
        values[canon] = (sub.values or {})[new]
    for name, default in (canonical_params or {}).items():
        if name in values:
            continue
        if str(default).strip() and str(default).lower() in prompt_lower:
            values[name] = str(default)
        else:
            return None
    return values


def _alias_route(alias_sid: str, sub: Any) -> Route | None:
    alias = sstore.load_aliases().get(alias_sid)
    if not alias or alias.get("same") is False:
        return None                       # unknown, or a cached NEGATIVE verdict
    target = alias.get("sid") or ""
    if not sstore.has_script(target):
        return None                       # canonical entry got archived; alias is stale
    canonical_params = (sstore.load_manifest().get(target) or {}).get("params") or {}
    values = _filled_values(alias.get("slots") or {}, sub, canonical_params)
    if values is None:
        return None                       # wording's values no longer cover the params
    return Route(sid=target, values=values, via="alias")


async def _verify(llm: Any, canonical_prompt: str, canonical_params: dict[str, str],
                  sub: Any) -> dict[str, str] | None:
    """One LLM call: same procedure? Returns the slot mapping {new_token: canonical_param}
    or None. The mapping must be injective and stay within the canonical params; full
    coverage is then enforced by _filled_values (verbatim defaults may fill the gaps)."""
    result = await llm.ainvoke([
        SystemMessage(content=ROUTER_VERIFY_SYSTEM_PROMPT),
        UserMessage(content=(
            f"CANONICAL (params: {sorted(canonical_params)}):\n{canonical_prompt}\n\n"
            f"NEW (values: {json.dumps(sub.values or {})}):\n{sub.template_prompt}")),
    ])
    data = _parse_json_reply(result.completion or "") or {}
    if not data.get("same"):
        return None
    slots = {str(k): str(v) for k, v in (data.get("slots") or {}).items()}
    mapped = list(slots.values())
    if sorted(mapped) != sorted(set(mapped)) or set(mapped) - set(canonical_params):
        logger.info("router verify: slot mapping %s is not injective into params %s",
                    slots, list(canonical_params))
        return None
    return slots


async def route(sub: Any, alias_sid: str, context: str, llm: Any, *,
                embedder: Any = None, model_name: str = _DEFAULT_MODEL) -> Route | None:
    """Resolve a subtask with NO direct library entry to an existing same-context entry,
    or None (author fresh). `alias_sid` is the subtask's own (missed) identity hash —
    the alias table's key."""
    hit = _alias_route(alias_sid, sub)
    if hit is not None:
        logger.info("router: alias hit %s -> %s", alias_sid, hit.sid)
        return hit
    if sstore.load_aliases().get(alias_sid, {}).get("same") is False:
        return None                       # negative cache: this wording was refused before

    manifest = sstore.load_manifest()
    candidates = {
        sid: _routable_text(entry.get("template_prompt") or "")
        for sid, entry in manifest.items()
        if entry.get("context") == context and sstore.has_script(sid)
    }
    if not candidates:
        return None
    embedder = embedder if embedder is not None else _default_embedder(model_name)
    if embedder is None:
        return None

    query = _routable_text(sub.template_prompt)
    vectors = _candidate_vectors(embedder, model_name, candidates)
    qvec = _embed_texts(embedder, [query])[0]
    ranked = sorted(((_cos(qvec, vectors[sid]), sid) for sid in candidates),
                    reverse=True)
    best_sim, best_sid = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
    if best_sim < SIM_FLOOR or best_sim - runner_up < SIM_MARGIN:
        logger.info("router: no confident candidate (best %.3f, margin %.3f)",
                    best_sim, best_sim - runner_up)
        return None
    if llm is None:
        return None                       # embeddings never decide alone

    entry = manifest.get(best_sid) or {}
    canonical_params = entry.get("params") or {}
    slots = await _verify(llm, entry.get("template_prompt") or "", canonical_params, sub)
    values = None if slots is None else _filled_values(slots, sub, canonical_params)
    if values is None:
        # Negative verdicts are cached too: this wording never pays the verify call again.
        sstore.save_alias(alias_sid, {"same": False, "sid": best_sid})
        return None
    sstore.save_alias(alias_sid, {"sid": best_sid, "slots": slots,
                                  "similarity": round(best_sim, 4)})
    logger.info("router: %r semantically routed to %s (sim %.3f); alias learned",
                sub.template_prompt[:60], best_sid, best_sim)
    return Route(sid=best_sid, values=values, via="semantic")
