"""Task decomposition for the hybrid subtask engine.

Resolves a task prompt into an ordered list of `Subtask`s, each carrying a TOKENIZED
template prompt ({{param}} placeholders) + this task's concrete values. The tokenized prompt
is the subtask's shared-library identity (see subtask_store), so decomposition is where
cross-task reuse is decided.

Resolution order (get_decomposition):
  1. spec.subtasks declared in the task registry  -> build directly, cache canonically
  2. exact cache hit: decompositions/<subtask_store.task_id(prompt)>.json
  3. derived match (deterministic, NO LLM): a cached decomposition whose parent prompt equals
     this prompt with only values swapped is re-instantiated with the new values — "same task,
     different customer/qty" reuses the SAME library entries at zero token cost
  4. one LLM call (DECOMPOSE_SYSTEM_PROMPT), validated against hallucination; on repeated
     failure, a single whole-prompt subtask (the engine degenerates to whole-task behavior)

The cache is immutable per prompt hash; --redecompose regenerates it.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from browser_use.llm.messages import SystemMessage, UserMessage

from automation.pipeline import subtask_store as sstore
from automation.pipeline.adapt import _parse_json_reply, _template_regex
from automation.pipeline.prompts import DECOMPOSE_SYSTEM_PROMPT

logger = logging.getLogger("framework.decompose")

# The token grammar is owned by subtask_store (library identity depends on it) — one
# definition, so the decomposer and the id normalizer can never drift apart.
_TOKEN = sstore.TOKEN_RE
MAX_SUBTASKS = 15

# Verification wording that marks a subtask as a JUDGE node: its success is a judgment call
# (compare/observe values), which does not survive compilation into a selector script — a
# replayed judge segment would walk the clicks with nobody looking and report a hollow pass.
# Judge nodes therefore always run with the LLM and are never committed to the library
# (see hybrid.run_hybrid_task). Wording is matched on the TEMPLATE prompt (values lifted);
# bare "note"/"check" are avoided: "credit note" is a record type and "check the option"
# is a click, so only their verification phrasings match.
_JUDGE_RE = re.compile(
    r"\b(verify|verifies|confirm|ensure|validate|compare)\b"
    r"|\bcheck (that|whether|if|it)\b"
    r"|\bmake sure\b"
    r"|\bsee (if|whether|that)\b"
    r"|\bnote (the|down|it)\b"
    r"|\bremember\b"
    r"|\bcapture the\b",
    re.IGNORECASE,
)


def node_kind(template_prompt: str, marker: str | None,
              declared: str | None = None) -> str:
    """Resolve a subtask's node kind: "action" (replayable) or "judge" (cognitive).

    An explicit declaration (spec/cache) wins; a marker-owning subtask is ALWAYS action —
    its network gate is machine ground truth, so caching it is safe regardless of wording;
    otherwise verification wording makes it a judge node. A false positive here only costs
    caching (the segment authors every run); a false negative would cost correctness
    (hollow replay), so the wording net is cast deliberately wide.
    """
    if declared in ("action", "judge"):
        return declared
    if marker:
        return "action"
    return "judge" if _JUDGE_RE.search(template_prompt) else "action"


@dataclass
class Subtask:
    """One executable segment of a task."""

    index: int
    template_prompt: str                 # tokenized — the library identity half
    values: dict[str, str] = field(default_factory=dict)
    marker: str | None = None            # set on the save-owning subtask only
    postcondition: dict[str, Any] | None = None
    # "action" (replayable from the library) | "judge" (cognitive: always LLM, never
    # cached — see node_kind). Assigned by _build_subtasks after markers are settled.
    kind: str = "action"

    @property
    def instantiated_prompt(self) -> str:
        """The concrete prompt for THIS task: tokens replaced by this task's values."""
        return _TOKEN.sub(
            lambda m: self.values.get(m.group(1), m.group(0)), self.template_prompt
        )


def _tokens_of(template_prompt: str) -> set[str]:
    return set(_TOKEN.findall(template_prompt))


def _build_subtasks(raw: list[dict[str, Any]], marker: str | None) -> list[Subtask]:
    """Materialize Subtasks from cache/spec/LLM dicts, assign the save-owning marker, and
    settle each node's kind (action/judge — see node_kind).

    Exactly one subtask owns the parent marker: an explicitly-declared one wins, else the
    last subtask (the save is the final act of a create flow). Kinds are resolved AFTER
    markers so the save-owning subtask can never be classified as a judge node.
    """
    subs = [
        Subtask(
            index=i,
            template_prompt=" ".join(str(d.get("template_prompt", "")).split()),
            values={str(k): str(v) for k, v in (d.get("values") or {}).items()},
            marker=d.get("marker"),
            postcondition=d.get("postcondition"),
        )
        for i, d in enumerate(raw)
    ]
    if marker and not any(s.marker for s in subs):
        subs[-1].marker = marker
    for s, d in zip(subs, raw):
        s.kind = node_kind(s.template_prompt, s.marker, d.get("kind"))
    return subs


def _as_cache(prompt: str, source: str, subs: list[Subtask]) -> dict[str, Any]:
    return {
        "parent_prompt": " ".join(prompt.split()),
        "source": source,
        "created": datetime.now().isoformat(timespec="seconds"),
        "subtasks": [
            {"template_prompt": s.template_prompt, "values": s.values,
             "marker": s.marker, "postcondition": s.postcondition, "kind": s.kind}
            for s in subs
        ],
    }


def _validate(raw: list[Any], prompt: str) -> str | None:
    """Why this decomposition is unusable, or None if it is sound.

    The hallucination guard mirrors adapt.parameterize: a value the parent prompt never
    contains could not be read out of a future prompt either — and here it also means the
    LLM invented work. Token/value closure guarantees instantiated prompts are concrete.
    """
    if not isinstance(raw, list) or not (1 <= len(raw) <= MAX_SUBTASKS):
        return f"expected 1..{MAX_SUBTASKS} subtasks, got {len(raw) if isinstance(raw, list) else type(raw).__name__}"
    prompt_lower = " ".join(prompt.split()).lower()
    for i, d in enumerate(raw):
        if not isinstance(d, dict) or not str(d.get("template_prompt", "")).strip():
            return f"subtask {i}: missing template_prompt"
        template = " ".join(str(d["template_prompt"]).split())
        values = d.get("values") or {}
        if not isinstance(values, dict):
            return f"subtask {i}: values is not a dict"
        tokens = _tokens_of(template)
        if tokens != set(values.keys()):
            return (f"subtask {i}: tokens {sorted(tokens)} do not close over values "
                    f"{sorted(values.keys())}")
        for name, value in values.items():
            if not str(value).strip():
                return f"subtask {i}: empty value for {{{{{name}}}}}"
            if str(value).lower() not in prompt_lower:
                return (f"subtask {i}: value {value!r} for {{{{{name}}}}} does not appear "
                        f"in the task prompt (hallucination guard)")
    return None


def whole_prompt_fallback(prompt: str, marker: str | None) -> list[Subtask]:
    """A single subtask covering the entire prompt — hybrid degenerates safely to today's
    whole-task behavior when decomposition is unavailable or invalid. The kind heuristic
    still applies: a markerless verification task falls back to ONE judge node, so it is
    never hollow-replayed even in degenerate form."""
    template = " ".join(prompt.split())
    return [Subtask(index=0, template_prompt=template, marker=marker,
                    kind=node_kind(template, marker))]


# ------------------------------- derived matching (tier 3) -------------------------------


def match_cached_decomposition(prompt: str) -> dict[str, Any] | None:
    """Find a cached decomposition whose parent prompt equals `prompt` with only values
    swapped; return a NEW cache dict with the new values, or None. Deterministic, no LLM.

    Each cached decomposition's parent prompt is compiled into a regex where every value
    becomes a named capture group (namespaced per subtask so two subtasks may reuse a param
    name), exactly the adapt._template_regex mechanics the whole-task template tier uses.
    """
    norm_new = " ".join(prompt.split())
    for tid, cached in sstore.all_decompositions().items():
        parent = cached.get("parent_prompt") or ""
        namespaced: dict[str, str] = {}
        for i, sub in enumerate(cached.get("subtasks") or []):
            for name, value in (sub.get("values") or {}).items():
                namespaced[f"s{i}__{name}"] = str(value)
        if not namespaced:
            # A value-free decomposition can still match — but only the exact same prompt,
            # which tier 2 (exact cache) already handles.
            continue
        rx = _template_regex(parent, namespaced)
        if rx is None:
            continue
        m = rx.match(norm_new)
        if not m:
            continue
        groups = {k: v.strip() for k, v in m.groupdict().items() if v is not None}
        new_subs: list[dict[str, Any]] = []
        for i, sub in enumerate(cached.get("subtasks") or []):
            new_values = {
                name: groups.get(f"s{i}__{name}", str(value))
                for name, value in (sub.get("values") or {}).items()
            }
            new_subs.append({**sub, "values": new_values})
        logger.info("decomposition derived from cached %s (values re-read from prompt)", tid)
        return {
            "parent_prompt": norm_new,
            "source": f"derived:{tid}",
            "created": datetime.now().isoformat(timespec="seconds"),
            "subtasks": new_subs,
        }
    return None


# ------------------------------- LLM decomposition (tier 4) -------------------------------


async def _llm_decompose(prompt: str, llm: Any) -> list[dict[str, Any]] | None:
    """One DECOMPOSE_SYSTEM_PROMPT call -> validated raw subtask dicts, or None."""
    result = await llm.ainvoke(
        [SystemMessage(content=DECOMPOSE_SYSTEM_PROMPT),
         UserMessage(content=f"Split this task:\n\n{prompt}")]
    )
    data = _parse_json_reply(result.completion or "")
    raw = (data or {}).get("subtasks")
    problem = _validate(raw, prompt) if raw else "no subtasks in reply"
    if problem:
        logger.warning("LLM decomposition rejected: %s", problem)
        return None
    # Translate is_save_step into a marker slot (the caller substitutes the real marker).
    out: list[dict[str, Any]] = []
    for d in raw:
        out.append({
            "template_prompt": " ".join(str(d["template_prompt"]).split()),
            "values": {str(k): str(v) for k, v in (d.get("values") or {}).items()},
            "is_save_step": bool(d.get("is_save_step")),
        })
    return out


# ------------------------------- resolution entry point -------------------------------


async def get_decomposition(
    prompt: str, llm: Any = None, spec: Any = None, marker: str | None = None,
    redecompose: bool = False,
) -> list[Subtask]:
    """Resolve `prompt` into subtasks; see module docstring for the 4-tier order.

    `spec` is the TaskSpec when the prompt came from the registry (its `subtasks` tuple wins
    outright); `marker` is the parent ground-truth marker (defaults to spec.marker) and is
    assigned to the save-owning subtask. Never raises: the worst case is the single
    whole-prompt fallback subtask.
    """
    if marker is None and spec is not None:
        marker = getattr(spec, "marker", None)
    tid = sstore.task_id(prompt)

    # Tier 1: explicit declaration on the TaskSpec.
    declared = getattr(spec, "subtasks", None) if spec is not None else None
    if declared:
        raw = [
            {"template_prompt": d.prompt, "values": dict(d.values or {}),
             "marker": d.marker, "postcondition": d.postcondition,
             "kind": getattr(d, "kind", None)}
            for d in declared
        ]
        problem = _validate(raw, prompt)
        if problem:
            # A registry declaration is author-maintained; a broken one should be loud.
            logger.error("spec.subtasks for %s invalid (%s); using whole-prompt fallback",
                         getattr(spec, "key", tid), problem)
            return whole_prompt_fallback(prompt, marker)
        subs = _build_subtasks(raw, marker)
        sstore.save_decomposition(tid, _as_cache(prompt, "spec", subs))
        return subs

    # Tier 2: exact cache.
    if not redecompose:
        cached = sstore.load_decomposition(tid)
        if cached:
            return _build_subtasks(cached["subtasks"], marker)

        # Tier 3: derived from a cached decomposition of the same prompt shape.
        derived = match_cached_decomposition(prompt)
        if derived:
            problem = _validate(derived["subtasks"], prompt)
            if problem is None:
                sstore.save_decomposition(tid, derived)
                return _build_subtasks(derived["subtasks"], marker)
            logger.warning("derived decomposition invalid (%s); falling through", problem)

    # Tier 4: LLM, once per novel prompt shape (retry once on validation failure).
    if llm is not None:
        for attempt in (1, 2):
            try:
                raw = await _llm_decompose(prompt, llm)
            except Exception as exc:  # noqa: BLE001 - decomposition must never crash a run
                logger.warning("LLM decomposition attempt %d failed: %s", attempt, exc)
                raw = None
            if raw:
                save_owners = [d for d in raw if d.pop("is_save_step", False)]
                if marker and len(save_owners) == 1:
                    save_owners[0]["marker"] = marker
                # Zero or several claimed save steps: _build_subtasks defaults to the last.
                subs = _build_subtasks(raw, marker)
                sstore.save_decomposition(tid, _as_cache(prompt, "llm", subs))
                return subs

    logger.info("no decomposition available for %s; whole-prompt fallback", tid)
    return whole_prompt_fallback(prompt, marker)
