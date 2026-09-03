"""Deterministic transpiler: committed steps/template -> a tier-1 code skill.

No LLM writes this code. The committed steps are already distilled, evidence-backed, and
parameterized (script_compile + adapt.parameterize), so the transpile is mechanical:
every step becomes one `await api.<verb>(...)` line, element identities move into the
ANCHOR BUNDLE (<sid>.anchors.json, semantic handle -> ranked selectors + fingerprint),
and template {{params}} become the function's keyword arguments. Zero hallucination risk,
no validation replay needed — the artifact is exactly as trustworthy as the steps it came
from. LLM codegen becomes worthwhile only when control-flow evidence exists (branches,
loops); this transpiler is the bootstrap that gives that future codegen a proven API to
target.

The generated file is real, hand-editable code — but edits must stay inside the api.*
whitelist enforced by `lint_code` (checked again at every load): no imports, no calls
except `api.<verb>(...)`, no attribute access except on `api`. That keeps executing the
file exactly as safe as interpreting the JSON it replaced.
"""
from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any

from automation.pipeline import adapt
from automation.pipeline import subtask_store as sstore
from automation.pipeline.script_compile import _atomic_write

logger = logging.getLogger("framework.skills.codegen")

_TOKEN = re.compile(r"\{\{([a-z0-9_]+)\}\}")
# The self-scoped token (hybrid._bind_self_noted). Deliberately NOT matched by _TOKEN —
# it is not a param, so the template/anchor substitution machinery must leave it alone.
_NOTED_TOKEN = re.compile(r"\{\{noted:([A-Za-z0-9_]+)\}\}")
# A click on a dropdown OPTION (react-select) — recognized so a preceding `type` collapses
# into api.select_option(label), the by-value pick primitive.
_OPTION_CLICK = re.compile(r'^role=option\[|\[id\$="-option')


def _is_option_click(step: dict[str, Any]) -> bool:
    return step.get("action") == "click" and any(
        _OPTION_CLICK.search(s) for s in step.get("selectors") or [])


def _value_expr(text: str, params: set[str]) -> str:
    """The Python expression for a (possibly tokenized) recorded value: a bare {{param}}
    becomes the argument name, embedded tokens become an f-string, literals stay literal.

    A {{noted:label}} token is the SELF-scoped kind (hybrid._bind_self_noted): the value
    was extracted by an earlier step of this same skill, so it compiles to a live read of
    the extract ledger — api.noted(label) — never to the authoring run's literal."""
    noted = _NOTED_TOKEN.findall(text)
    if noted:
        if len(noted) > 1 or text.strip() != "{{noted:%s}}" % noted[0]:
            # A noted value spliced into a larger string has no api verb to express it;
            # refusing here drops the entry to tier-0, where resolve_noted handles it.
            raise ValueError(f"cannot transpile embedded noted token in {text!r}")
        return f"api.noted({noted[0]!r})"
    tokens = [t for t in _TOKEN.findall(text) if t in params]
    if not tokens:
        return repr(text)
    if len(tokens) == 1 and text.strip() == "{{" + tokens[0] + "}}":
        return tokens[0]
    body = text.replace("{", "{{").replace("}", "}}")
    for t in tokens:
        body = body.replace("{{{{" + t + "}}}}", "{" + t + "}")
    return "f" + repr(body)


def _handle_for(step: dict[str, Any], used: set[str]) -> str:
    """A stable, human-readable handle for a step's target element. Prefers the param the
    selectors are tokenized on, then the recorded label/attr identity, then any quoted
    label inside the first selector (query-fallback clicks carry no fingerprint)."""
    sels = step.get("selectors") or []
    token = next((t for s in sels for t in _TOKEN.findall(s)), None)
    fp = step.get("fingerprint") or {}
    attrs = fp.get("attrs") or {}
    quoted = re.search(r'"([^"]+)"', sels[0]) if sels else None
    base = f"{token}-target" if token else next(
        (str(c) for c in (fp.get("text"), attrs.get("aria-label"), attrs.get("name"),
                          attrs.get("placeholder"), attrs.get("title"), attrs.get("id"),
                          quoted.group(1) if quoted else None)
         if c and str(c).strip()),
        step.get("action", "el"),
    )
    slug = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")[:40] or "el"
    handle, n = slug, 2
    while handle in used:
        handle, n = f"{slug}-{n}", n + 1
    used.add(handle)
    return handle


def _anchor(step: dict[str, Any]) -> dict[str, Any]:
    anchor: dict[str, Any] = {"selectors": list(step.get("selectors") or [])}
    if step.get("fingerprint"):
        anchor["fingerprint"] = step["fingerprint"]
    if step.get("hidden_ok"):
        # The recorded click reached a legitimately-invisible control (hover-revealed /
        # 0-size); replay keeps the hidden-dispatch permission (script_compile).
        anchor["hidden_ok"] = True
    if step.get("opens_tab"):
        # The recorded click spawned a tab; replay must follow it there and must never
        # re-click (see script_compile._click_and_follow). Lose this on the way into the
        # anchor bundle and the tier-1 skill replays the pre-fix behaviour.
        anchor["opens_tab"] = True
    if step.get("query"):
        # Extract steps keep their recorded query as the semantic re-find fallback.
        anchor["query"] = step["query"]
    if step.get("expect_text"):
        # Landed-click name guard travels with the anchor (api.click reads it via
        # _step_for; value-parameterized anchors get it overwritten at instantiation —
        # see base._substituted_anchors).
        anchor["expect_text"] = step["expect_text"]
    return anchor


def transpile(sid: str, steps: list[dict[str, Any]], *, source_prompt: str = "",
              params: dict[str, str] | None = None) -> tuple[str, dict[str, Any]]:
    """steps (+ template params) -> (code_text, anchors). Raises on any step it cannot
    express — the caller then simply keeps the tier-0 body."""
    params = dict(params or {})
    param_set = set(params)
    anchors: dict[str, Any] = {}
    used: set[str] = set()
    lines: list[str] = []
    optional_open = False
    i = 0
    while i < len(steps):
        step, action = steps[i], steps[i].get("action")
        nxt = steps[i + 1] if i + 1 < len(steps) else None
        if step.get("optional") and not optional_open:
            # From here on the compiled steps are the slice's declared error branch, not
            # its work (script_compile.compile_recording's `optional_from`). One mark, not
            # a per-call flag: the branch is always the trailing tail.
            lines.append("    await api.begin_optional()")
            optional_open = True
        if action == "type" and nxt is not None and _is_option_click(nxt):
            expr = _value_expr(str(step.get("text", "")), param_set)
            lines.append(f"    await api.select_option({expr})")
            i += 2
            continue
        if action == "goto":
            lines.append(f"    await api.goto({step['url']!r})")
        elif action == "click":
            handle = _handle_for(step, used)
            anchors[handle] = _anchor(step)
            count = int(step.get("count", 1))
            if step.get("until_done"):
                # Recorded as "until it stops advancing" (repeat_click times=0): replay the
                # INTENT, not the authoring run's number, or a longer list under-runs.
                lines.append(f"    await api.repeat_until_done({handle!r}, "
                             f"{float(step.get('repeat_wait_s', 0.0))!r})")
            elif count > 1:
                # A recorded "exactly N clicks" cadence: one faithful repeat verb, the
                # recorded inter-click wait as its floor (api adds the readiness poll).
                lines.append(f"    await api.repeat_click({handle!r}, {count}, "
                             f"{float(step.get('repeat_wait_s', 0.0))!r})")
            else:
                lines.append(f"    await api.click({handle!r})")
        elif action == "click_indexed":
            handle = _handle_for(step, used)
            anchor: dict[str, Any] = {"selector_template": step["selector_template"]}
            if step.get("fingerprint"):
                anchor["fingerprint"] = step["fingerprint"]
            anchors[handle] = anchor
            lines.append(f"    await api.click_indexed({handle!r}, "
                         f"{int(step.get('start', 0))}, {int(step['count'])})")
        elif action == "fill":
            handle = _handle_for(step, used)
            anchors[handle] = _anchor(step)
            expr = _value_expr(str(step.get("value", "")), param_set)
            tail = "" if step.get("clear", True) else ", clear=False"
            lines.append(f"    await api.fill({handle!r}, {expr}{tail})")
        elif action == "select":
            handle = _handle_for(step, used)
            anchors[handle] = _anchor(step)
            expr = _value_expr(str(step.get("value", "")), param_set)
            lines.append(f"    await api.select({handle!r}, {expr})")
        elif action == "type":
            lines.append(f"    await api.type_text("
                         f"{_value_expr(str(step.get('text', '')), param_set)})")
        elif action == "press":
            lines.append(f"    await api.press({step['keys']!r})")
        elif action == "wait":
            lines.append(f"    await api.wait({float(step.get('seconds', 0))!r})")
        elif action == "scroll":
            tail = "" if step.get("down", True) else ", down=False"
            lines.append(f"    await api.scroll({float(step.get('pages', 0.5))!r}{tail})")
        elif action == "close_tab":
            lines.append("    await api.close_tab()")
        elif action == "find_click":
            expr = _value_expr(str(step.get("text", "")), param_set)
            lines.append(f"    await api.find_click({expr})")
        elif action == "paste":
            handle = _handle_for(step, used)
            anchors[handle] = _anchor(step)
            expr = _value_expr(str(step.get("value", "")), param_set)
            lines.append(f"    await api.paste({handle!r}, {expr})")
        elif action in ("extract", "copy"):
            if not step.get("selectors"):
                # Query-only extract: no element identity to anchor on — the entry stays
                # tier-0, where run_steps owns the semantic re-find.
                raise ValueError(f"cannot transpile query-only {action} step")
            handle = _handle_for(step, used)
            anchors[handle] = _anchor(step)
            lines.append(f"    await api.{action}({handle!r}, "
                         f"{str(step.get('label') or 'value')!r})")
        elif action == "upload":
            handle = _handle_for(step, used)
            anchors[handle] = _anchor(step)
            expr = _value_expr(str(step.get("value", "")), param_set)
            lines.append(f"    await api.upload({handle!r}, {expr})")
        else:
            raise ValueError(f"cannot transpile step action {action!r}")
        i += 1
    if not lines:
        raise ValueError("nothing to transpile")

    sig = "".join(f", {name}={default!r}" for name, default in params.items())
    doc = (f"Generated tier-1 skill {sid} (deterministic transpile of the committed "
           f"steps).\n\nsource: {source_prompt or '(no template)'}\n\nElement identities "
           f"live in {sid}.anchors.json — healing edits THAT file, never this one. Hand "
           f"edits here are allowed but must stay inside the api.* whitelist "
           f"(skills.codegen.lint_code).")
    code = (f'"""{doc}\n"""\n\n\n'
            f"async def run(api{(', *' + sig) if sig else ''}):\n"
            + "\n".join(lines) + "\n")
    return code, anchors


# ------------------------------- the safety whitelist -------------------------------

_ALLOWED_NODES = {
    ast.Module, ast.Expr, ast.Constant, ast.AsyncFunctionDef, ast.arguments, ast.arg,
    ast.Await, ast.Call, ast.Attribute, ast.Name, ast.Load, ast.Store, ast.keyword,
    ast.JoinedStr, ast.FormattedValue, ast.If, ast.Compare, ast.BoolOp, ast.And, ast.Or,
    ast.UnaryOp, ast.Not, ast.Eq, ast.NotEq, ast.In, ast.NotIn, ast.Is, ast.IsNot,
    ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Assign, ast.Return, ast.Pass, ast.For,
}


def lint_code(code: str) -> list[str]:
    """Why this skill code is unsafe to execute, [] when it is clean.

    Whitelist, not blacklist: exactly one `async def run(api, ...)`, and the only thing
    the body may DO is await `api.<verb>(...)` — plus inert control flow (if/for/assign
    over literals and args). Anything else (imports, other calls, attribute escapes,
    lambdas, subscripts) is rejected. Enforced at generation AND at every load, so a
    hand-edited file gets the same gate as a generated one.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    problems: list[str] = []
    fns = [n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)]
    others = [n for n in tree.body if not isinstance(n, (ast.AsyncFunctionDef, ast.Expr))]
    if len(fns) != 1 or fns[0].name != "run" or others:
        problems.append("file must contain exactly one `async def run(api, ...)` "
                        "(plus an optional docstring)")
    else:
        args = fns[0].args
        if [a.arg for a in args.args] != ["api"] or args.vararg or args.kwarg:
            problems.append("run's only positional argument must be `api` "
                            "(params go keyword-only)")
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            problems.append(f"disallowed syntax: {type(node).__name__}")
        elif isinstance(node, ast.Call):
            f = node.func
            if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                    and f.value.id == "api"):
                problems.append("only api.<verb>(...) calls are allowed")
        elif isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id == "api"):
                problems.append("attribute access is only allowed on `api`")
    return sorted(set(problems))


# ------------------------------- commit-time upgrade -------------------------------


def compile_code_skill(sid: str) -> Path | None:
    """Best-effort tier-1 upgrade of a just-committed library entry: transpile its steps
    (template steps when parameterized) into <sid>.skill.py + <sid>.anchors.json.

    Failure is non-fatal — the tier-0 steps stay authoritative — but it must never leave
    STALE code shadowing fresh steps, so any failure removes the code artifacts."""
    try:
        steps = json.loads(sstore.steps_path(sid).read_text())
        source_prompt, params = "", {}
        tpath = sstore.template_path(sid)
        if tpath.exists():
            template = adapt.load_template(tpath)
            steps = template.get("steps") or steps
            params = template.get("params") or {}
            source_prompt = template.get("source_prompt", "")
        code, anchors = transpile(sid, steps, source_prompt=source_prompt, params=params)
        problems = lint_code(code)
        if problems:
            raise ValueError(f"emitted code failed its own lint: {problems}")
        _atomic_write(sstore.code_path(sid), code)
        _atomic_write(sstore.anchors_path(sid), json.dumps(anchors, indent=2))
        # The code body SUBSUMES the steps body (it was derived from it deterministically);
        # keeping both invites drift, so steps.json survives only for entries the
        # transpiler can't express. Recompilation always starts from the recording (or the
        # template's tokenized steps), never from this file.
        sstore.steps_path(sid).unlink(missing_ok=True)
        logger.info("segment %s: tier-1 code skill compiled (%d anchors)", sid, len(anchors))
        return sstore.code_path(sid)
    except Exception as exc:  # noqa: BLE001 - the upgrade is optional; staleness is not
        sstore.code_path(sid).unlink(missing_ok=True)
        sstore.anchors_path(sid).unlink(missing_ok=True)
        logger.warning("tier-1 compile skipped for %s: %s", sid, exc)
        return None
