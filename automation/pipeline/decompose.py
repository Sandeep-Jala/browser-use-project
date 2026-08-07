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
  4. LLM (DECOMPOSE_SYSTEM_PROMPT), validated against hallucination; a rejected attempt is
     retried ONCE with the rejection reason fed back; on repeated failure, a single
     whole-prompt subtask (the engine degenerates to whole-task behavior)

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
# Ceiling on a split, to catch a runaway/hallucinated decomposition. Raised 15 -> 24 on
# 2026-07-27: combined end-to-end chains (a payroll e2e followed by the RTI payrun)
# legitimately need ~20 nodes, and over the ceiling _validate rejects both LLM attempts and
# the run degrades to whole_prompt_fallback — one giant node with no per-segment gates.
# Keep in sync with the "2 to N subtasks" bullet in prompts.DECOMPOSE_SYSTEM_PROMPT: if the
# prompt states a smaller number the LLM self-limits and raising this constant does nothing.
MAX_SUBTASKS = 24

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

# Iteration wording that, COMBINED with a judge phrase, marks a LOOP node instead: an
# imperative action repeated until a stated stop condition holds ("process employees one
# at a time ... check that the next employee has loaded ... stopping as soon as X is
# shown"). "keep"/"continue" count only with a following gerund ("continue clicking"), so
# verification prose like "values keep their order" never matches.
_LOOP_CUE_RE = re.compile(
    r"\b(?:one at a time|at a time|each|until|stopping|(?:keep|continue)\s+\w+ing)\b",
    re.IGNORECASE,
)

# Loop signature that needs NO judge phrase. The judge-first gate in node_kind misses
# imperative rewrites ("keep repeating ... until X is shown") that carry no verification
# vocabulary — observed 2026-08-05: the RTI employee loop reworded without "check that"
# classified action, was committed to the library, and its frozen Save & Next replay
# saved the stop-target employee. BOTH cues are required: "each"/"until"/"stopping"
# alone are everyday action filler ("after each click", "wait until the page loads",
# "stopping after the third click").
_LOOP_REPEAT_RE = re.compile(
    r"\b(?:at a time|repeat(?:ing|ed|s)?|for each|(?:keep|continue)\s+\w+ing)\b",
    re.IGNORECASE,
)
_LOOP_STOP_RE = re.compile(
    r"\b(?:until|as soon as|stop(?:ping|s)?)\b",
    re.IGNORECASE,
)

# A judge phrase that IS the subtask's head directive ("Verify that each filter...", "Then
# check that entries do not repeat...") keeps the node a judge even when iteration cues
# appear in WHAT it checks — only a leading imperative action with verification folded
# inside ("Process ... and after each click check that ...") makes a loop.
_LEADING_JUDGE_RE = re.compile(
    r"^\s*(?:(?:and|then|now|next|first|finally|also|please)[,\s]+)*"
    r"(?:verify|verifies|confirm|ensure|validate|compare|check|make sure|"
    r"see (?:if|whether|that)|note|remember|capture)\b",
    re.IGNORECASE,
)


def node_kind(template_prompt: str, marker: str | None,
              declared: str | None = None, tab_url: str | None = None) -> str:
    """Resolve a subtask's node kind: "action" (replayable), "judge" (cognitive), or
    "loop" (repeat-until: acts like an action, but always live and never cached).

    An explicit declaration (spec/cache) wins; a marker-owning subtask is ALWAYS action —
    its network gate is machine ground truth, so caching it is safe regardless of wording;
    an aux-tab subtask (tab_url) is action too: its wording is usually observational
    ("note the top result"), but its replay is not hollow — the compiled extract step
    re-reads the live DOM every run, so the observation stays fresh without the LLM;
    otherwise verification wording makes it a judge node — unless iteration cues say the
    verification is folded INTO a repeated action ("Save & Next ... check that ... until X
    is shown"): that is a loop node. A loop must ACT (observation framing made the agent
    declare the loop done after one iteration), yet can never be cached: the iteration
    count is live page state, so a replayed loop would walk a fixed number of steps and
    land anywhere. Loop wording needs no judge phrase, though: an imperative rewrite
    ("keep repeating ... until X is shown") carries none, and routing it through the
    judge gate demoted it to a cacheable action whose frozen Save & Next replay saved
    the stop-target employee (2026-08-05) — so a repetition cue PLUS a stop cue is a
    loop signature on its own. A false positive here only costs caching (the segment
    authors every run); a false negative costs correctness (hollow replay / frozen
    iteration count), so the wording net is cast deliberately wide.
    """
    if declared in ("action", "judge", "loop"):
        return declared
    if marker or tab_url:
        return "action"
    if _JUDGE_RE.search(template_prompt):
        if _LOOP_CUE_RE.search(template_prompt) \
                and not _LEADING_JUDGE_RE.match(template_prompt):
            return "loop"
        return "judge"
    if _LOOP_REPEAT_RE.search(template_prompt) \
            and _LOOP_STOP_RE.search(template_prompt) \
            and not _LEADING_JUDGE_RE.match(template_prompt):
        return "loop"
    return "action"


# Wording that CONSUMES data noted by an EARLIER subtask ("using the noted generated name
# and address"). Such values exist only at run time, so a cached recording can only carry
# the AUTHORING run's concrete ones (adapt.parameterize lifts only values the prompt spells
# out) — replaying it types stale data into the app, and a marker gate would even bless the
# save. The net is consumer-shaped on purpose: PRODUCER wording ("note the generated
# identity; remember Name and Address") must NOT match, or the extract segments it labels
# would lose their zero-LLM replay (replayed extracts re-read the live DOM — never stale).
_NOTED_DATA_RE = re.compile(
    # Determiner + participle: "the noted name", "these remembered values" — participles
    # that unambiguously mean run-noted data. "generated"/"recorded"/"saved" are excluded
    # here: "the generated identity" appears inside producer wording and "the recorded
    # payment" is app-domain vocabulary.
    r"\b(?:the|that|those|these|its|their)\s+"
    r"(?:noted|remembered|captured|extracted|observed)\b"
    # Usage word + determiner + participle: the ambiguous participles count only when the
    # phrase says the data is being USED ("with the generated name", "enter the saved id").
    r"|\b(?:using|use|with|from|for|enter|fill|add|type|paste|search)\s+"
    r"(?:the|that|those|these)\s+"
    r"(?:noted|remembered|captured|extracted|observed|generated|recorded|saved|copied)\b"
    # Participle + back-reference adverb: "the name generated earlier", "values noted above".
    r"|\b(?:noted|remembered|captured|extracted|observed|generated|recorded|saved|copied)\s+"
    r"(?:earlier|previously|above|before)\b"
    # Explicit cross-segment reference: "the title from the previous step/tab/search".
    r"|\bfrom\s+the\s+(?:previous|prior|earlier|last)\s+"
    r"(?:step|subtask|tab|page|site|search|result)s?\b"
    # Relative-clause reference to a record an earlier segment made: "the employee which
    # we added", "the request that was created" — its NAME/identity exists only at run
    # time, so a cached pick can never carry it (observed live: the parameterizer bound
    # the employee pick to the word "download", the only verbatim-checkable token left).
    r"|\b(?:which|that|whom?)\s+(?:we|you|i|was|were)\s+(?:just\s+)?"
    r"(?:added|created|generated|made|noted|saved)\b"
    r"|\b(?:newly|just)[\s-](?:added|created|generated)\b",
    re.IGNORECASE,
)


# Wording whose deliverable is a FILE DOWNLOAD ("select download, select PDF", "export to
# Excel"). Such a subtask gets the authoritative download GATE (hybrid.segment_gate): the
# segment passed iff a file actually arrived in its window — the click that triggers a
# download reports a watchdog TIMEOUT on every honest success, so neither the agent's
# self-report nor the steps floor can be trusted in either direction.
_DOWNLOAD_RE = re.compile(r"\b(download|export)\b", re.IGNORECASE)


def downloads_file(template_prompt: str) -> bool:
    """True when the subtask's wording says it downloads/exports a file."""
    return bool(_DOWNLOAD_RE.search(template_prompt))


def consumes_noted_data(template_prompt: str) -> bool:
    """True when the subtask's wording says it USES data noted by an earlier subtask.

    The hybrid loop combines this with "did an earlier segment actually record findings
    this run" to conclude that a library replay would type stale values: the segment then
    runs with the agent (which receives the fresh observations) and is never committed.
    Matched on the TEMPLATE prompt — the reference wording is procedure, not a value, so
    the decomposer keeps it literal there.
    """
    return bool(_NOTED_DATA_RE.search(template_prompt))


# Wording that opens with a conditional guard ("If you see an error ..., click Add
# Payment"): whether its actions run AT ALL depends on live page state. A recording made
# on a run where the guard was TRUE would replay the branch unconditionally on every run
# (the FALSE branch already refuses commit via the zero-step compile rule), so hybrid runs
# these with the agent and never commits them. Leading-"If" only: an embedded conditional
# ("go to Payroll & RTI ... If a pop up appears, dismiss it") is a footnote to an
# unconditional procedure, and stays cacheable.
_CONDITIONAL_RE = re.compile(r"^\s*(?:(?:and|then|now)[,\s]+)*if\b", re.IGNORECASE)


def is_conditional_guard(template_prompt: str) -> bool:
    """True when the subtask's wording is a conditional branch guard (leading "If ...")."""
    return bool(_CONDITIONAL_RE.match(template_prompt))


@dataclass
class Subtask:
    """One executable segment of a task."""

    index: int
    template_prompt: str                 # tokenized — the library identity half
    values: dict[str, str] = field(default_factory=dict)
    marker: str | None = None            # set on the save-owning subtask only
    postcondition: dict[str, Any] | None = None
    # Run this subtask in a separate helper tab opened at this URL (same browser context).
    # The tab closes when the subtask ends; the main app page is never navigated.
    tab_url: str | None = None
    # "action" (replayable from the library) | "judge" (cognitive: always LLM, never
    # cached) | "loop" (repeat-until: action framing, always LLM, never cached — see
    # node_kind). Assigned by _build_subtasks after markers are settled.
    kind: str = "action"
    # True only for the whole-prompt fallback blob: the entire task as one subtask.
    # Downstream it degrades gating/framing to neutral (steps gate, no download block,
    # whole-task step budget) — specialized framings are calibrated for FRAGMENTS and
    # derailed the blob runs (see whole_prompt_fallback).
    fallback: bool = False
    # Declared deterministic checks (tuple of checks.Check) — tier-1 spec subtasks only;
    # evaluated on top of the segment's base gate. Never cached (_as_cache drops them).
    verify: tuple[Any, ...] | None = None

    @property
    def instantiated_prompt(self) -> str:
        """The concrete prompt for THIS task: tokens replaced by this task's values."""
        return _TOKEN.sub(
            lambda m: self.values.get(m.group(1), m.group(0)), self.template_prompt
        )


def _tokens_of(template_prompt: str) -> set[str]:
    return set(_TOKEN.findall(template_prompt))


# A subtask that says to open a NEW TAB at an absolute URL is an aux-tab subtask, whether or
# not the LLM remembered to emit "tab_url". Left unset, the helper-tab machinery never engages
# and the agent navigates the APP page to that site — the one thing aux tabs exist to prevent
# (there is no navigate-back subtask, so every later subtask runs on the wrong page). Keyed on
# explicit new-tab wording so in-app deep links ("navigate directly to <app url>", "come back
# to <app url>") are left alone.
_NEW_TAB_RE = re.compile(r"\b(?:new|another|separate)\s+tab\b", re.I)
_ABS_URL_RE = re.compile(r"https?://[^\s,;)'\"]+", re.I)


def _implied_tab_url(template_prompt: str) -> str | None:
    if not _NEW_TAB_RE.search(template_prompt):
        return None
    for candidate in _ABS_URL_RE.findall(template_prompt):
        url = candidate.rstrip(".,;:")
        if sstore.is_absolute_http_url(url):
            return url
    return None


def _build_subtasks(raw: list[dict[str, Any]], marker: str | None,
                    trust_kind: bool = True) -> list[Subtask]:
    """Materialize Subtasks from cache/spec/LLM dicts, assign the save-owning marker, and
    settle each node's kind (action/judge/loop — see node_kind).

    Exactly one subtask owns the parent marker: an explicitly-declared one wins, else the
    last subtask (the save is the final act of a create flow). Kinds are resolved AFTER
    markers so the save-owning subtask can never be classified as a judge node. With
    `trust_kind=False` a stored "kind" is advisory only and is re-derived from wording:
    cached decompositions carry the CLASSIFIER'S old verdict, and re-deriving is what lets
    a classifier fix reach every already-cached task without --redecompose.
    """
    subs = [
        Subtask(
            index=i,
            template_prompt=" ".join(str(d.get("template_prompt", "")).split()),
            values={str(k): str(v) for k, v in (d.get("values") or {}).items()},
            marker=d.get("marker"),
            postcondition=d.get("postcondition"),
            tab_url=d.get("tab_url"),
            verify=d.get("verify") or None,
        )
        for i, d in enumerate(raw)
    ]
    if marker and not any(s.marker for s in subs):
        subs[-1].marker = marker
    for s in subs:
        if not s.tab_url and (implied := _implied_tab_url(s.template_prompt)):
            logger.info("subtask %d: deriving tab_url %s from its new-tab wording", s.index,
                        implied)
            s.tab_url = implied
    for s, d in zip(subs, raw):
        s.kind = node_kind(s.template_prompt, s.marker,
                           d.get("kind") if trust_kind else None, s.tab_url)
    return subs


def _as_cache(prompt: str, source: str, subs: list[Subtask]) -> dict[str, Any]:
    return {
        "parent_prompt": " ".join(prompt.split()),
        "source": source,
        "created": datetime.now().isoformat(timespec="seconds"),
        "subtasks": [
            # Deliberately no "verify": declared checks re-attach from the spec on every
            # load, so a stale cache can never resurrect superseded checks.
            {"template_prompt": s.template_prompt, "values": s.values,
             "marker": s.marker, "postcondition": s.postcondition, "kind": s.kind,
             "tab_url": s.tab_url}
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
        tab_url = d.get("tab_url")
        if tab_url is not None and not sstore.is_absolute_http_url(tab_url):
            # An invented/garbled helper-tab URL must never silently run.
            return f"subtask {i}: tab_url {tab_url!r} is not an absolute http(s) URL"
    return None


# A dropped span this long is a lost instruction, not a reworded connective. DECOMPOSE_SYSTEM_
# PROMPT already forbids dropping ("Substituting every subtask's values back into its
# template_prompt must reproduce the task's original wording ... Do not reword, add, or drop
# actions"), but every OTHER rule there is also enforced mechanically by _validate — this one
# was not, so a silently dropped clause reached the run. Observed live: "NI number should be AB
# followed by a random 6 digit number and end with C" vanished from the add-employee subtask
# because a GENERATIVE instruction is neither a literal value (the hallucination guard bars
# tokenizing it) nor a button label, so the LLM simply omitted it and the employee would have
# saved with a blank NI number.
_MIN_DROPPED_SPAN = 5


def _coverage_words(text: str) -> list[str]:
    """Normalized word sequence used for the drop check — punctuation is noise here."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).split()


def coverage_gap(raw: list[dict[str, Any]], prompt: str) -> str | None:
    """The longest run of parent-prompt wording that no subtask covers, or None.

    Matching is by BIGRAM, not by word: individual words ("number", "date", "end") recur all
    over a task prompt and a bag-of-words check would call a dropped clause covered by its
    own vocabulary appearing elsewhere.
    """
    covered = set()
    for d in raw:
        template = str(d.get("template_prompt", ""))
        for name, value in (d.get("values") or {}).items():
            template = template.replace("{{%s}}" % name, str(value))
        words = _coverage_words(template)
        covered.update(zip(words, words[1:]))

    words = _coverage_words(prompt)
    run: list[str] = []
    worst: list[str] = []
    for first, second in zip(words, words[1:]):
        if (first, second) in covered:
            run = []
            continue
        run.append(first if not run else second)
        if len(run) > len(worst):
            worst = list(run)
    if len(worst) < _MIN_DROPPED_SPAN:
        return None
    return (f"dropped wording from the task: {' '.join(worst)!r} appears in no subtask. "
            "Every instruction in the task must survive into exactly one subtask")


def whole_prompt_fallback(prompt: str, marker: str | None) -> list[Subtask]:
    """A single subtask covering the entire prompt — hybrid degenerates safely to
    whole-task behavior when decomposition is unavailable or invalid.

    The judge verdict is kept (a markerless verification task falls back to ONE judge
    node, never hollow-replayed), but "loop" collapses to "action": repeat+stop cues
    ANYWHERE in a mega-task's text classified the whole blob a loop, and the loop
    framing — "you are mid-iteration; done when the stop condition holds" — made the
    agent skip the task's opening and once declare the entire task complete because the
    FIRST embedded repeat-until's stop condition held on a page it wandered onto
    (runs 20260805_155515/161958). A blob is a whole procedure, not an iteration."""
    template = " ".join(prompt.split())
    k = node_kind(template, marker)
    return [Subtask(index=0, template_prompt=template, marker=marker,
                    kind=k if k == "judge" else "action", fallback=True)]


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


async def _llm_decompose(
    prompt: str, llm: Any, feedback: str | None = None, strict_coverage: bool = True,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """One DECOMPOSE_SYSTEM_PROMPT call -> (validated raw subtask dicts, rejection reason).

    `feedback` is the rejection reason from the previous attempt, folded into the user
    message so the retry corrects that specific mistake — a blind resample repeats it.

    `strict_coverage` rejects a split that drops task wording. It is OFF on the final attempt
    on purpose: a structurally sound split missing one clause still runs the task far better
    than the whole-prompt fallback that a rejection would leave us with, so the gap is logged
    and kept rather than traded for one giant node.
    """
    user = f"Split this task:\n\n{prompt}"
    if feedback:
        user += (
            f"\n\nYour previous split was REJECTED: {feedback}\n"
            "Produce a corrected split of the SAME task that fixes exactly this problem. "
            "A value must be an exact substring of the task text; a phrase referring to "
            "data discovered at runtime is procedure wording — keep it literal in "
            'template_prompt, and use "values": {} when the task spells out no data.'
        )
    result = await llm.ainvoke(
        [SystemMessage(content=DECOMPOSE_SYSTEM_PROMPT), UserMessage(content=user)]
    )
    data = _parse_json_reply(result.completion or "")
    raw = (data or {}).get("subtasks")
    problem = _validate(raw, prompt) if raw else "no subtasks in reply"
    if problem:
        logger.warning("LLM decomposition rejected: %s", problem)
        return None, problem
    gap = coverage_gap(raw, prompt)
    if gap:
        if strict_coverage:
            logger.warning("LLM decomposition rejected: %s", gap)
            return None, gap
        logger.error("decomposition KEPT despite dropped wording (%s) — the subtask that "
                     "should carry it will run without that instruction", gap)
    # Translate is_save_step into a marker slot (the caller substitutes the real marker).
    out: list[dict[str, Any]] = []
    for d in raw:
        out.append({
            "template_prompt": " ".join(str(d["template_prompt"]).split()),
            "values": {str(k): str(v) for k, v in (d.get("values") or {}).items()},
            "is_save_step": bool(d.get("is_save_step")),
            "tab_url": d.get("tab_url"),
        })
    return out, None


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
             "kind": getattr(d, "kind", None), "tab_url": getattr(d, "tab_url", None),
             "verify": getattr(d, "verify", None)}
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
            return _build_subtasks(cached["subtasks"], marker, trust_kind=False)

        # Tier 3: derived from a cached decomposition of the same prompt shape.
        derived = match_cached_decomposition(prompt)
        if derived:
            problem = _validate(derived["subtasks"], prompt)
            if problem is None:
                sstore.save_decomposition(tid, derived)
                return _build_subtasks(derived["subtasks"], marker, trust_kind=False)
            logger.warning("derived decomposition invalid (%s); falling through", problem)

    # Tier 4: LLM, once per novel prompt shape (retries feed the rejection reason back).
    # Three attempts, not two: a long chain typically burns one on a structural rejection
    # (token closure), which used to leave the dropped-wording check no retry to spend — the
    # gap was then only reported, never corrected. The last attempt is coverage-advisory, so
    # the extra attempt costs tokens only on prompts that are already failing.
    if llm is not None:
        problem: str | None = None
        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                raw, problem = await _llm_decompose(prompt, llm, feedback=problem,
                                                    strict_coverage=(attempt < attempts))
            except Exception as exc:  # noqa: BLE001 - decomposition must never crash a run
                logger.warning("LLM decomposition attempt %d failed: %s", attempt, exc)
                raw = None  # a transport error carries no feedback; keep any prior reason
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
