"""Task/subtask identity + the shared subtask library.

The library holds one entry per (parameterized subtask prompt, starting page context) pair,
shared GLOBALLY across parent tasks — the "select {{business}} business" prefix every task
starts with is ONE entry here, authored once and replayed everywhere. It is the only
recording store: there is no whole-task script tier above it.

Identity: `subtask_id(template_prompt, context)` hashes the TOKENIZED prompt (values lifted
into {{param}} tokens), so "add invoice for customer Suresh Gopi" and "... for customer Mr
Jones" resolve to the same entry; the `context` half is the normalized URL the page is on
when the subtask starts, disambiguating same-worded subtasks that begin on different pages.

`task_id(prompt)` is the coarser twin: a stable hash of a PARENT task prompt, used to key its
cached decomposition and to label runs.

Layout (LIBRARY_DIR):
  {sid}.skill.py        tier-1 code skill transpiled from the committed steps (the
                        transient {sid}.steps.json body is deleted after codegen)
  {sid}.template.json   adapt.parameterize output ({{param}} tokens + defaults)
  {sid}.recording.json  raw agent history of the authoring segment
  {sid}.meta.json       mutable per-run stats (uses/fail_count) — kept OUT of the manifest
                        so ~100k replays/day don't serialize on one atomically-rewritten file
  manifest.json         identity registry (template_prompt, params, context, end_context...),
                        written only when an entry is created or archived
  archive/              retired entries, timestamped

DECOMPOSITIONS_DIR holds one cached decomposition per PARENT prompt hash (task_id).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("framework.subtask_store")

LIBRARY_DIR = Path("library")
LIBRARY_MANIFEST = LIBRARY_DIR / "manifest.json"
DECOMPOSITIONS_DIR = Path("decompositions")

# Volatile URL path segments that must not split library identity: pure digit runs, GUIDs,
# and long hex ids are all instance data (record ids, session ids), not page structure.
_VOLATILE_SEGMENT = re.compile(
    r"^(\d+|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{16,})$",
    re.IGNORECASE,
)

# {{param}} tokens in a template prompt — the ONE definition of the decomposer's token
# grammar (decompose.py reuses it; group 1 is the token name). Token NAMES are erased
# from library identity: the LLM decomposer names tokens freely ({{business}} one run,
# {{business_name}} the next), and a name difference must not split two identically-worded
# subtasks into separate library entries.
TOKEN_RE = re.compile(r"\{\{([a-z][a-z0-9_]*)\}\}")


def normalize_context(url: str) -> str:
    """Stable page-state key from a live URL.

    Lowercased path plus '#fragment' (SPA routes live in the fragment here); origin and query
    stripped; volatile segments (record ids, GUIDs, hex runs) replaced by '*' so two visits
    to the same page through different records share a context.
      https://app/x/bookkeeping/12345/inputs/sales?tab=2#invoices
        -> /x/bookkeeping/*/inputs/sales#invoices
    """
    from urllib.parse import urlsplit

    parts = urlsplit(url or "")
    path = parts.path or "/"
    segments = [
        "*" if _VOLATILE_SEGMENT.match(seg) else seg.lower()
        for seg in path.split("/")
    ]
    norm = "/".join(segments) or "/"
    frag = (parts.fragment or "").strip()
    if frag:
        # The fragment is itself a path on SPA routers — normalize its segments too.
        frag_norm = "/".join(
            "*" if _VOLATILE_SEGMENT.match(seg) else seg.lower()
            for seg in frag.split("/")
        )
        norm += f"#{frag_norm}"
    return norm


def rebase_to_live(recorded_url: str, live_url: str) -> str:
    """`recorded_url` with its volatile path segments taken from `live_url` instead.

    A compiled `goto` carries the authoring run's absolute URL, which means it carries that
    run's CLIENT. library/d85bb1eda9ba8381 (Pay Forecast) was authored in FOOD LIMITED and
    its four `api.goto()` lines named `/paye/clients/6a61d0…/calculator` outright; replayed
    in Food Alchemy it teleported the run to the other client and looked for the freshly
    added employee in a company that had never heard of them (run 20260903_093236_260802
    — the create POST goes to Clients/6a984504…, the next four requests to 6a61d0…).

    The identity layer was already right and always had been: that entry's context is
    `/paye/clients/*/rti/payrun` because _VOLATILE_SEGMENT calls a long hex id instance
    data, not page structure, so ONE recording is meant to serve every client. Only the
    executable half disagreed. This is the two halves saying the same thing — the same
    predicate that writes the `*` decides what a replay may substitute.

    The walk: compare paths segment by segment; where BOTH are volatile take the live one;
    where both are literal and equal keep going; on any other disagreement stop and keep the
    recorded tail verbatim. Stopping matters — past a divergence the two paths describe
    different parts of the app and position no longer means anything, so an id there has
    nothing to correspond to. The recorded query and fragment always survive: they say what
    the goto wanted, not where it was.

    Never rewrites across origins. The aux identity tab and the OTP portal are different
    SITES, and dragging a goto onto whatever host happens to be live would be exactly this
    bug pointed the other way. An unreadable or degenerate live URL (about:blank, "") also
    returns the recording unchanged: with nothing to rebase onto, the recorded URL is the
    best guess available, and a navigation is not the place to fail closed.
    """
    from urllib.parse import urlsplit, urlunsplit

    rec, live = urlsplit(recorded_url or ""), urlsplit(live_url or "")
    if not rec.netloc or rec.netloc != live.netloc or rec.scheme != live.scheme:
        return recorded_url
    rec_segments = (rec.path or "/").split("/")
    live_segments = (live.path or "/").split("/")
    out = list(rec_segments)
    for i, seg in enumerate(rec_segments):
        if i >= len(live_segments):
            break
        if _VOLATILE_SEGMENT.match(seg) and _VOLATILE_SEGMENT.match(live_segments[i]):
            out[i] = live_segments[i]
        elif seg.lower() != live_segments[i].lower():
            break
    path = "/".join(out)
    if path == rec.path:
        return recorded_url
    logger.info("goto rebased onto the live page: %s -> %s", rec.path, path)
    return urlunsplit((rec.scheme, rec.netloc, path, rec.query, rec.fragment))


def normalize_aux_context(url: str) -> str:
    """Page-state key for a helper-tab (aux) URL: normalize_context host-QUALIFIED.

    normalize_context strips the origin (one-app assumption), but aux tabs are foreign
    origins — without the host, google.com/ and bing.com/ would both key as '/'. Aux
    contexts therefore start with a hostname while main contexts start with '/', a
    self-describing discriminator (evaluate_gate picks its normalizer by that prefix).
      https://www.google.com/search?q=x -> www.google.com/search
    """
    from urllib.parse import urlsplit

    host = (urlsplit(url or "").hostname or "").lower()
    return f"{host}{normalize_context(url)}"


def is_absolute_http_url(url: Any) -> bool:
    """True for an absolute http(s) URL with a dotted host — the only tab_url shape the
    engine will open. Everything else (relative paths, invented pseudo-URLs like
    'https://module.search') is rejected loud at declaration/validation time, never at
    run time."""
    if not isinstance(url, str):
        return False
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    return parts.scheme in ("http", "https") and "." in (parts.hostname or "")


def task_id(prompt: str) -> str:
    """Stable short id for a PARENT task prompt (whitespace/case-insensitive).

    Keys the cached decomposition (decompositions/<tid>.json) and labels runs. Identity is
    the ORIGINAL user prompt, so editing a task's wording by even one word gives it a new id
    and orphans its cached decomposition — see tests/test_tasks.py::test_task_ids_stable.
    """
    norm = " ".join(prompt.split()).lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def normalize_template(template_prompt: str) -> str:
    """The wording half of a subtask's identity, normalized hard so decompositions of
    DIFFERENT parent tasks converge on one entry: whitespace collapsed, lowercased, token
    names erased ({{business}} == {{business_name}} — wording carries the identity, not
    what the LLM called the slot), and trailing punctuation dropped (where the decomposer
    cuts a span decides whether it ends in '.')."""
    norm = " ".join(template_prompt.split()).lower()
    return TOKEN_RE.sub("{{*}}", norm).rstrip(" .,;")


def subtask_id(template_prompt: str, context: str) -> str:
    """Stable short id for a (tokenized subtask prompt, starting context) pair."""
    norm = normalize_template(template_prompt)
    return hashlib.sha256(f"{norm}\n{context}".encode("utf-8")).hexdigest()[:16]


def find_same_template_entry(template_prompt: str,
                             exclude_sid: str) -> tuple[str, dict[str, Any]] | None:
    """First manifest entry with the same normalized wording under a DIFFERENT sid.

    That is an identity fork: sids key on wording + start context, so the same subtask
    recorded from another starting page is invisible to a direct lookup. Surfaced (not
    auto-replayed — navigating to a stored URL can cross run-specific records) so the
    log can say a recording exists but is unreachable from here."""
    tnorm = normalize_template(template_prompt)
    for osid, entry in load_manifest().items():
        if osid != exclude_sid \
                and normalize_template(str(entry.get("template_prompt") or "")) == tnorm:
            return osid, entry
    return None


def steps_path(sid: str) -> Path:
    return LIBRARY_DIR / f"{sid}.steps.json"


def template_path(sid: str) -> Path:
    return LIBRARY_DIR / f"{sid}.template.json"


def recording_path(sid: str) -> Path:
    return LIBRARY_DIR / f"{sid}.recording.json"


def meta_path(sid: str) -> Path:
    return LIBRARY_DIR / f"{sid}.meta.json"


def code_path(sid: str) -> Path:
    """Tier-1 body: the generated skill function (skills/codegen.py)."""
    return LIBRARY_DIR / f"{sid}.skill.py"


def anchors_path(sid: str) -> Path:
    """Tier-1 element identities: handle -> ranked selectors + fingerprint."""
    return LIBRARY_DIR / f"{sid}.anchors.json"


def aliases_path() -> Path:
    """Semantic-router alias table: {alias_sid: {"sid": canonical, "slots": {...}}}."""
    return LIBRARY_DIR / "aliases.json"


def embeddings_path() -> Path:
    """Semantic-router vector cache: {"model": name, "vectors": {sid: [floats]}}."""
    return LIBRARY_DIR / "embeddings.json"


def load_aliases() -> dict[str, Any]:
    p = aliases_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001 - a corrupt alias table just means no aliases
        return {}


def save_alias(alias_sid: str, data: dict[str, Any]) -> None:
    aliases = load_aliases()
    aliases[alias_sid] = data
    _atomic_write_json(aliases_path(), aliases)


def has_script(sid: str) -> bool:
    """True when the entry is REGISTERED and has an executable body. Tier-1 code is the
    normal case; a steps file exists only for entries the transpiler couldn't express (see
    codegen).

    The manifest check is what makes a refused commit stick. A body on disk with no manifest
    entry is an ORPHAN — the segment ran, its recording was compiled, and then a commit
    guard rejected it (an unbindable runtime value, an unanchorable step). Run
    20260828_124929 replayed such an orphan: the provenance guard had refused the Data
    Request recording because it baked in a previous run's employee name, but the steps file
    it left behind still answered True here, so the rejected script replayed anyway — ticking
    the wrong employee and reporting ok=True. A guard that leaves its own bypass on disk is
    not a guard."""
    return (code_path(sid).exists() or steps_path(sid).exists()) and sid in load_manifest()


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def archive_entry(sid: str) -> list[Path]:
    """Move a library entry's files into library/archive/ (timestamped). The manifest entry
    is dropped; meta is removed (a re-authored entry starts with fresh stats). Returns the
    archived paths (empty if nothing existed)."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = LIBRARY_DIR / "archive"
    moved: list[Path] = []
    for src, kind in ((steps_path(sid), "steps"), (template_path(sid), "template"),
                      (recording_path(sid), "recording"),
                      (recording_path(sid).with_suffix(".failed.json"), "recording-failed"),
                      (code_path(sid), "skill"), (anchors_path(sid), "anchors")):
        if not src.exists():
            continue
        archive_dir.mkdir(parents=True, exist_ok=True)
        dst = archive_dir / f"{sid}.{kind}.{stamp}{src.suffix}"
        os.replace(src, dst)
        moved.append(dst)
    meta_path(sid).unlink(missing_ok=True)
    manifest = load_manifest()
    if sid in manifest:
        del manifest[sid]
        _atomic_write_json(LIBRARY_MANIFEST, manifest)
    return moved


def load_manifest() -> dict[str, Any]:
    """Read library/manifest.json; {} if missing or corrupt."""
    if not LIBRARY_MANIFEST.exists():
        return {}
    try:
        return json.loads(LIBRARY_MANIFEST.read_text())
    except Exception:  # noqa: BLE001 - a corrupt manifest just means an empty library
        return {}


def update_manifest(sid: str, template_prompt: str, *, create: bool = False,
                    **fields: Any) -> None:
    """Record/refresh a library entry in manifest.json (atomic write). Called only on
    author/archive — per-run counters go in the entry's meta file, not here.

    `create=True` is required to REGISTER a new entry; without it an update to an
    unregistered sid is dropped. Only the commit path may register, because only it has the
    full entry (context, start_url, steps, params, bindings). The post-replay learners —
    healed-selector promotion, end-title pinning — carry ONE field each, and before
    2026-08-28 they would happily conjure an entry out of that single field: run
    20260828_124929 ended up with a manifest entry holding nothing but an `end_title`
    learned from a leftover tab, which then became the expected end state for every later
    run of that subtask. A learner may refine a registered entry; it may not invent one."""
    data = load_manifest()
    entry = data.get(sid)
    if entry is None:
        if not create:
            logger.debug("update_manifest: %s is not registered; dropping %s",
                         sid, sorted(fields))
            return
        entry = {}
    entry.setdefault("template_prompt", " ".join(template_prompt.split()))
    entry.setdefault("created", datetime.now().isoformat(timespec="seconds"))
    entry.update(fields)
    entry["updated"] = datetime.now().isoformat(timespec="seconds")
    data[sid] = entry
    _atomic_write_json(LIBRARY_MANIFEST, data)


def load_meta(sid: str) -> dict[str, Any]:
    p = meta_path(sid)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001 - corrupt meta just means fresh stats
        return {}


def bump_meta(sid: str, **deltas: int) -> dict[str, Any]:
    """Increment per-run counters (uses, fail_count, ...) in the entry's meta file.
    A successful use resets fail_count: only CONSECUTIVE failures should retire an entry."""
    meta = load_meta(sid)
    for key, delta in deltas.items():
        meta[key] = int(meta.get(key, 0)) + int(delta)
    if deltas.get("uses"):
        meta["fail_count"] = 0
        meta["last_used"] = datetime.now().isoformat(timespec="seconds")
    _atomic_write_json(meta_path(sid), meta)
    return meta


def archive_if_failing(sid: str, threshold: int = 2) -> bool:
    """Retire the entry once its consecutive fail_count reaches `threshold`, so the next run
    authors a clean replacement instead of re-fighting a stale recording. True if archived."""
    if int(load_meta(sid).get("fail_count", 0)) >= threshold:
        archive_entry(sid)
        return True
    return False


# ------------------------------- decomposition cache -------------------------------


def decomposition_path(tid: str) -> Path:
    return DECOMPOSITIONS_DIR / f"{tid}.json"


def load_decomposition(tid: str) -> dict[str, Any] | None:
    p = decomposition_path(tid)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except Exception:  # noqa: BLE001 - corrupt cache entry = no cache entry
        return None
    return data if isinstance(data, dict) and data.get("subtasks") else None


def save_decomposition(tid: str, data: dict[str, Any]) -> None:
    _atomic_write_json(decomposition_path(tid), data)


def all_decompositions() -> dict[str, dict[str, Any]]:
    """Every cached decomposition keyed by tid (for derived matching)."""
    if not DECOMPOSITIONS_DIR.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for p in DECOMPOSITIONS_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        if isinstance(data, dict) and data.get("subtasks"):
            out[p.stem] = data
    return out
