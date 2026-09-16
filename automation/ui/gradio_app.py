"""Auto Agent's four tabs, as a Gradio `Blocks`. Layout and event wiring only.

Everything with a decision in it lives in `ops.py`, which imports no gradio — that is what keeps
the logic testable without standing up a server.

**The one rule that makes the step editor work.** `@gr.render(inputs=editor_state)` re-runs its
whole body whenever `editor_state` changes. If a keystroke wrote into that state, every character
would tear down and rebuild N step cards and the textarea would lose focus on each one. So:

    editor_state is an output of STRUCTURAL handlers only — add/remove/reorder a step, change a
    step's type, add/remove a check. Never of a keystroke.

Typing instead fires `revalidate`, whose only outputs are markdown boxes that no layout depends
on, so nothing is rebuilt. And every structural handler HARVESTS the live widget values into a new
form dict BEFORE it mutates the structure, so text typed but not yet saved survives the re-render.
Correctness comes from the harvest; `key=` is then only a DOM-reuse optimisation, and the design
holds whether or not Gradio preserves keyed components across a reorder.

**Every editable widget inside the render passes `interactive=True` explicitly, and must.** Left to
`interactive=None`, Gradio infers editability from whether a component is an input to some event at
config time — and for components created inside `@gr.render`, that inference resolves to DISABLED
once the block re-renders. The first render looked fine; loading a prompt turned the whole editor
read-only, radios and textareas alike. Nothing about the wiring was wrong, so it is invisible in
any server-side test: a `form_input`-style programmatic write still works, because setting `.value`
ignores the `disabled` attribute. Only a real click finds it. If you add a widget here and it will
not take input, this is why.

The whole UI is built inside one function on purpose: every component is then in lexical scope for
every handler, which is what lets the Prompts tab fill the Run tab's box without module globals.
"""
from __future__ import annotations

import copy
from typing import Any

import gradio as gr

from automation.ui import ops, store, uploads
from automation.ui.model import (CHECK_LABELS, PROBE_DEFAULT_TIMEOUT_S,
                                 VERIFY_DEFAULT_TIMEOUT_S, PromptModel)
from automation.ui.paths import Paths
from automation.ui.supervisor import BrowserBusy, RunBusy, RunRequest, RunSupervisor

_CHECK_CHOICES = [(label, kind) for kind, label in CHECK_LABELS.items()]
_STEP_CHOICES = [("Action", "action"), ("Check", "check"), ("Conditional", "conditional")]
_STEP_HELP = {
    "action": "Work. Recorded once, then replayed for free on later runs.",
    "check": "A verification the agent judges live every run — never cached.",
    "conditional": "Runs only when the page shows what you name. Skipped at zero AI cost.",
}
_LOG_TAIL = 400


def build_blocks(paths: Paths, supervisor: RunSupervisor) -> gr.Blocks:
    with gr.Blocks(title="Auto Agent", fill_width=True) as demo:
        gr.Markdown("## Auto Agent")

        with gr.Tabs() as tabs:

            # ══ Prompts ══════════════════════════════════════════════════════════════════
            with gr.Tab("Prompts", id="prompts"):
                editor_state = gr.State(ops.model_to_form(PromptModel()))
                is_new = gr.State(True)
                picked_prompt = gr.State("")
                picked_task = gr.State("")

                with gr.Row():
                    with gr.Column(scale=1):
                        broken_md = gr.Markdown(visible=False)
                        gr.Markdown("### Your prompts")
                        prompt_df = gr.Dataframe(
                            headers=["name", "shape", "save gate", "tags", "edited"],
                            datatype=["str"] * 5, type="array", interactive=False,
                            max_height=240, wrap=True)
                        with gr.Row():
                            new_btn = gr.Button("New prompt", variant="primary", size="sm")
                            del_btn = gr.Button("Delete", variant="stop", size="sm")
                        prompts_msg = gr.Markdown()

                        gr.Markdown("### Built-in tasks *(read-only — tasks.yaml)*")
                        task_df = gr.Dataframe(
                            headers=["name", "shape", "save gate", "tags"],
                            datatype=["str"] * 4, type="array", interactive=False,
                            max_height=200, wrap=True)
                        with gr.Row():
                            copy_name = gr.Textbox(placeholder="name for the copy…",
                                                   show_label=False, scale=2)
                            copy_btn = gr.Button("Copy to an editable prompt",
                                                 size="sm", scale=1)

                    with gr.Column(scale=2):
                        banner = gr.Markdown()
                        saved_key = gr.State("")
                        # Set only by "Save and run". A separate State from `saved_key` so a
                        # plain Save does not also yank the user onto the Run tab.
                        jump_key = gr.State("")

                        @gr.render(inputs=editor_state)
                        def editor(form: dict[str, Any]) -> None:
                            form = form or ops.model_to_form(PromptModel())
                            steps = form.get("steps") or []
                            comps: dict[tuple, Any] = {}
                            step_errs: list[gr.Markdown] = []
                            ups, dns, rms, add_chks, rm_chks = [], [], [], [], []

                            # `marker` and `tags` have no widget on purpose. They are NOT dropped:
                            # `harvest` deep-copies the form and overwrites only the paths in
                            # `comps`, so a prompt that already declares them keeps them across an
                            # edit and a save. Giving them a widget is the only thing that changed.
                            comps[("key",)] = gr.Textbox(
                                form.get("key", ""), label="Name", key="f-key",
                                interactive=True)

                            if not steps:
                                comps[("prompt",)] = gr.Textbox(
                                    form.get("prompt", ""), label="One instruction", lines=4,
                                    key="f-prompt", interactive=True)
                                mode_btn = gr.Button("Switch to step-by-step", size="sm",
                                                     key="f-mode")
                            else:
                                mode_btn = gr.Button("Join into one instruction", size="sm",
                                                     key="f-mode")

                            for i, step in enumerate(steps):
                                uid = step["uid"]
                                with gr.Group(key=f"g-{uid}"):
                                    with gr.Row(key=f"r-{uid}"):
                                        comps[("steps", i, "step_type")] = gr.Radio(
                                            _STEP_CHOICES, value=step.get("step_type", "action"),
                                            show_label=False, key=f"{uid}-ty", scale=4,
                                            interactive=True,
                                            info=_STEP_HELP.get(step.get("step_type", "action")))
                                        ups.append(gr.Button("↑", size="sm", min_width=40,
                                                             scale=0, key=f"{uid}-up"))
                                        dns.append(gr.Button("↓", size="sm", min_width=40,
                                                             scale=0, key=f"{uid}-dn"))
                                        rms.append(gr.Button("✕", size="sm", min_width=40,
                                                             scale=0, variant="stop",
                                                             key=f"{uid}-rm"))

                                    comps[("steps", i, "prompt")] = gr.Textbox(
                                        step.get("prompt", ""), label=f"Step {i + 1}", lines=2,
                                        key=f"{uid}-p", interactive=True)
                                    step_errs.append(gr.Markdown(visible=False, key=f"{uid}-e"))

                                    if step.get("step_type") == "conditional":
                                        with gr.Row(key=f"pr-{uid}"):
                                            comps[("steps", i, "probe_kind")] = gr.Dropdown(
                                                _CHECK_CHOICES,
                                                value=step.get("probe_kind", "text_visible"),
                                                label="Only run this step if…",
                                                key=f"{uid}-pk", scale=2, interactive=True)
                                            comps[("steps", i, "probe_text")] = gr.Textbox(
                                                step.get("probe_text", ""), show_label=False,
                                                key=f"{uid}-pt", scale=2, interactive=True)
                                            comps[("steps", i, "probe_timeout_s")] = gr.Number(
                                                step.get("probe_timeout_s"), label="timeout s",
                                                placeholder=str(PROBE_DEFAULT_TIMEOUT_S),
                                                minimum=0, scale=0, key=f"{uid}-po",
                                                interactive=True)

                                    checks = step.get("verify") or []
                                    if checks:
                                        gr.Markdown("*Must be true when this step ends*",
                                                    key=f"{uid}-vh")
                                    per_step_rm = []
                                    for j, chk in enumerate(checks):
                                        cuid = chk["uid"]
                                        with gr.Row(key=f"cr-{cuid}"):
                                            comps[("steps", i, "verify", j, "kind")] = \
                                                gr.Dropdown(_CHECK_CHOICES,
                                                            value=chk.get("kind", "text_visible"),
                                                            show_label=False, scale=2,
                                                            key=f"{cuid}-k", interactive=True)
                                            comps[("steps", i, "verify", j, "text")] = \
                                                gr.Textbox(chk.get("text", ""), show_label=False,
                                                           scale=2, key=f"{cuid}-t",
                                                           interactive=True)
                                            comps[("steps", i, "verify", j, "timeout_s")] = \
                                                gr.Number(chk.get("timeout_s"), show_label=False,
                                                          placeholder=str(
                                                              VERIFY_DEFAULT_TIMEOUT_S),
                                                          minimum=0, scale=0, key=f"{cuid}-o",
                                                          interactive=True)
                                            per_step_rm.append(
                                                gr.Button("✕", size="sm", min_width=40, scale=0,
                                                          key=f"{cuid}-rm"))
                                    rm_chks.append(per_step_rm)
                                    add_chks.append(gr.Button("Add a check", size="sm",
                                                              key=f"{uid}-ac"))

                                    with gr.Accordion("Advanced", open=False, key=f"{uid}-ad"):
                                        with gr.Row(key=f"ar-{uid}"):
                                            comps[("steps", i, "marker")] = gr.Textbox(
                                                step.get("marker", ""),
                                                label="Save gate for this step",
                                                key=f"{uid}-m", scale=2, interactive=True)
                                            comps[("steps", i, "tab_url")] = gr.Textbox(
                                                step.get("tab_url", ""),
                                                label="Run this step in a helper tab at",
                                                key=f"{uid}-tu", scale=2, interactive=True)
                                        comps[("steps", i, "allow_write_refusal")] = gr.Checkbox(
                                            bool(step.get("allow_write_refusal")),
                                            label="may end on a refused save", key=f"{uid}-aw",
                                            interactive=True)
                                        vals = step.get("values") or {}
                                        comps[("steps", i, "values")] = gr.Dataframe(
                                            value=[[k, v] for k, v in vals.items()] or [["", ""]],
                                            headers=["{{token}}", "value"],
                                            datatype=["str", "str"], type="array",
                                            row_count=(max(1, len(vals)), "dynamic"),
                                            col_count=(2, "fixed"), label="Values",
                                            key=f"{uid}-va", interactive=True)

                            add_step_btn = gr.Button("Add a step", size="sm", key="f-add")
                            with gr.Row():
                                save_btn = gr.Button("Save", variant="primary", key="f-save")
                                save_run_btn = gr.Button("Save and run", key="f-srun")


                            live = set(comps.values())

                            def harvest(data, _f=form) -> dict[str, Any]:
                                out = copy.deepcopy(_f)
                                for path, comp in comps.items():
                                    _put(out, path, data[comp])
                                return out

                            # -- structural: harvest first, then mutate --
                            for i in range(len(steps)):
                                ups[i].click(lambda d, i=i: ops.move_step(harvest(d), i, -1),
                                             live, editor_state, api_visibility="private")
                                dns[i].click(lambda d, i=i: ops.move_step(harvest(d), i, +1),
                                             live, editor_state, api_visibility="private")
                                rms[i].click(lambda d, i=i: ops.remove_step(harvest(d), i),
                                             live, editor_state, api_visibility="private")
                                add_chks[i].click(lambda d, i=i: ops.add_check(harvest(d), i),
                                                  live, editor_state, api_visibility="private")
                                for j, btn in enumerate(rm_chks[i]):
                                    btn.click(
                                        lambda d, i=i, j=j: ops.remove_check(harvest(d), i, j),
                                        live, editor_state, api_visibility="private")
                                comps[("steps", i, "step_type")].change(
                                    harvest, live, editor_state, api_visibility="private")

                            add_step_btn.click(lambda d: ops.add_step(harvest(d)), live,
                                               editor_state, api_visibility="private")
                            mode_btn.click(
                                lambda d: (_to_one if steps else _to_steps)(harvest(d)),
                                live, editor_state, api_visibility="private")

                            # -- keystrokes: validation only; nothing here is rebuilt --
                            def revalidate(data):
                                text, per_step, _ = ops.validation_report(
                                    ops.model_from_form(harvest(data)))
                                return [gr.update(value=text)] + [
                                    gr.update(value=per_step.get(k, ""), visible=k in per_step)
                                    for k in range(len(steps))]

                            for path, c in comps.items():
                                if path[-1] == "step_type":
                                    continue      # already an editor_state trigger
                                c.change(revalidate, live, [banner] + step_errs,
                                         trigger_mode="always_last", show_progress="hidden",
                                         api_visibility="private")

                            # -- save --
                            def do_save(data):
                                ok, msg, key = ops.save_prompt(harvest(data),
                                                               is_new=bool(data[is_new]))
                                rows, note = ops.prompt_rows()
                                return (("✅ " if ok else "⚠️ ") + msg, rows,
                                        gr.update(value=note, visible=bool(note)),
                                        False if ok else gr.skip(), key if ok else "")

                            outs = [prompts_msg, prompt_df, broken_md, is_new, saved_key]
                            save_btn.click(do_save, live | {is_new}, outs,
                                           api_visibility="private")
                            # The tab switch itself is wired at the bottom of build_blocks, where
                            # `tabs` and the Run tab's dropdown are in scope — this button is
                            # created inside the render and they are not visible to each other.
                            save_run_btn.click(do_save, live | {is_new}, outs,
                                               api_visibility="private").then(
                                lambda k: k, saved_key, jump_key, api_visibility="private")

            # ══ Files ════════════════════════════════════════════════════════════════════
            with gr.Tab("Files", id="files"):
                gr.Markdown(f"Files a prompt can name, in `{paths.uploads_dir}`. Refer to one by "
                            f"its **bare name** — the scanner resolves it before the run logs in.")
                files_df = gr.Dataframe(
                    headers=["name", "write this in a prompt", "size", "used by", "warning"],
                    datatype=["str"] * 5, type="array", interactive=False, wrap=True,
                    max_height=320)
                with gr.Row():
                    up_file = gr.File(label="Add a file", type="filepath", scale=2)
                    with gr.Column(scale=1):
                        overwrite = gr.Checkbox(False, label="replace if it exists")
                        up_btn = gr.Button("Upload", variant="primary", size="sm")
                with gr.Row():
                    del_name = gr.Textbox(label="Delete this file", placeholder="exact name…",
                                          scale=2)
                    force_del = gr.Checkbox(False, label="delete even if a prompt uses it",
                                            scale=1)
                    del_file_btn = gr.Button("Delete", variant="stop", size="sm", scale=0)
                files_msg = gr.Markdown()

                def do_upload(tmp, ow):
                    ok, msg = ops.save_upload_path(tmp, overwrite=bool(ow))
                    return ops.file_rows(), ("✅ " if ok else "⚠️ ") + msg
                up_btn.click(do_upload, [up_file, overwrite], [files_df, files_msg],
                             api_visibility="private")

                def do_delete_file(name, force):
                    ok, msg = ops.delete_upload((name or "").strip(), force=bool(force))
                    return ops.file_rows(), ("✅ " if ok else "⚠️ ") + msg
                del_file_btn.click(do_delete_file, [del_name, force_del],
                                   [files_df, files_msg], api_visibility="private")

            # ══ Run ══════════════════════════════════════════════════════════════════════
            with gr.Tab("Run", id="run"):
                with gr.Group():
                    with gr.Row():
                        task_dd = gr.Dropdown(ops.run_options(), label="Prompt",
                                              allow_custom_value=False, scale=2)
                        free_text = gr.Textbox(label="…or type an instruction", scale=3)
                    with gr.Row():
                        marker_in = gr.Textbox(label="Save gate", scale=2,
                                               placeholder="blank = as configured, "
                                                           "'none' = off")
                        reauthor_in = gr.Textbox(label="Re-record steps (indexes or text)",
                                                 scale=2)
                    with gr.Row():
                        fresh_cb = gr.Checkbox(False, label="Re-record every step")
                        redecomp_cb = gr.Checkbox(False, label="Re-plan the steps")
                        record_cb = gr.Checkbox(False, label="Save a video")
                        hosts_cb = gr.Checkbox(False, label="Log every host")
                        force_cb = gr.Checkbox(False, label="Start even if a browser is open")
                    with gr.Row():
                        start_btn = gr.Button("Start run", variant="primary")
                        refresh_opts = gr.Button("Refresh list", size="sm")

                phase_md = gr.Markdown("**Idle**")
                run_msg = gr.Markdown()
                step_df = gr.Dataframe(headers=ops.SEGMENT_HEADERS,
                                       datatype=["str"] * len(ops.SEGMENT_HEADERS),
                                       type="array", interactive=False, wrap=True,
                                       max_height=300)
                with gr.Row():
                    pause_btn = gr.Button("Pause", interactive=False, size="sm")
                    resume_btn = gr.Button("Resume", interactive=False, size="sm")
                    stop_btn = gr.Button("Stop", interactive=False, variant="stop", size="sm")
                    kill_btn = gr.Button("Force kill", size="sm")
                steer_in = gr.Textbox(
                    label="Tell it what to do instead, then press Resume", interactive=False,
                    info="a replaying step has no model to steer — it refuses out loud")
                log_box = gr.Textbox(label="Console", lines=18, max_lines=18, interactive=False,
                                     autoscroll=True)

                tail = gr.State({"seq": 0, "dropped": 0, "lines": []})
                timer = gr.Timer(1.0)

                refresh_opts.click(lambda: gr.update(choices=ops.run_options()), None, task_dd,
                                   api_visibility="private")

                async def do_start(task, free, marker, reauthor, fresh, redec, rec, hosts, force):
                    chosen = (free or "").strip() or (task or "").strip()
                    if not chosen:
                        return "⚠️ pick a prompt or type an instruction"
                    req = RunRequest(task=chosen, fresh=bool(fresh),
                                     marker=(marker or "").strip() or None,
                                     redecompose=bool(redec),
                                     reauthor=(reauthor or "").strip() or None,
                                     log_all_hosts=bool(hosts), record=bool(rec))
                    try:
                        await supervisor.start(req, force=bool(force))
                    except (RunBusy, BrowserBusy) as exc:
                        return f"⚠️ {exc}"
                    except Exception as exc:  # noqa: BLE001
                        return f"⚠️ could not start: {exc}"
                    return f"started **{chosen}**"

                start_btn.click(do_start,
                                [task_dd, free_text, marker_in, reauthor_in, fresh_cb,
                                 redecomp_cb, record_cb, hosts_cb, force_cb],
                                run_msg, api_visibility="private")

                def tick(t):
                    snap = supervisor.snapshot()
                    chunk = supervisor.log_since(t["seq"])
                    lines = list(t["lines"])
                    if chunk["dropped"] > t["dropped"]:
                        lines.append(f"… {chunk['dropped'] - t['dropped']} earlier lines dropped …")
                    lines = (lines + [l["text"].rstrip("\n") for l in chunk["lines"]])[-_LOG_TAIL:]
                    new_t = {"seq": chunk["next"], "dropped": chunk["dropped"], "lines": lines}

                    state = snap["state"]
                    armed = bool(snap["control"]["armed"]) and state == "running"
                    head = f"**{snap['phase']}**"
                    if snap.get("task_label"):
                        head += f" · {snap['task_label']}"
                    if snap.get("run_id"):
                        head += f" · `{snap['run_id']}`"
                    for w in snap.get("warnings") or []:
                        head += f"\n\n⚠️ {w}"
                    if snap.get("reason"):
                        head += f"\n\n{snap['reason']}"

                    return (new_t, head,
                            ops.segment_rows(snap.get("progress"), finished=state == "done"),
                            "\n".join(lines) if chunk["lines"] else gr.skip(),
                            gr.update(interactive=armed), gr.update(interactive=armed),
                            gr.update(interactive=armed), gr.update(interactive=armed))

                timer.tick(tick, tail,
                           [tail, phase_md, step_df, log_box,
                            pause_btn, resume_btn, stop_btn, steer_in],
                           show_progress="hidden", api_visibility="private")

                def send(cmd, instruction=""):
                    try:
                        supervisor.control(cmd, instruction or "")
                    except (ValueError, RuntimeError) as exc:
                        return f"⚠️ {exc}"
                    return f"sent **{cmd}**" + (f" — “{instruction}”" if instruction else "")

                pause_btn.click(lambda: send("pause"), None, run_msg, api_visibility="private")
                resume_btn.click(lambda s: send("resume", (s or "").strip()), steer_in, run_msg,
                                 api_visibility="private")
                stop_btn.click(lambda: send("stop"), None, run_msg, api_visibility="private")

                def do_kill():
                    out = supervisor.kill()
                    return "sent SIGINT to the run" if out.get("signalled") else "nothing running"
                kill_btn.click(do_kill, None, run_msg, api_visibility="private")

            # ══ History ══════════════════════════════════════════════════════════════════
            with gr.Tab("History", id="history"):
                gr.Markdown(f"Runs in `{paths.artifacts_dir}`")
                runs_df = gr.Dataframe(
                    headers=["status", "run", "started", "task", "took", "steps",
                             "replayed", "tokens", "size", "note"],
                    datatype=["str"] * 10, type="array", interactive=False, wrap=True,
                    max_height=300)
                with gr.Row():
                    reload_btn = gr.Button("Reload", size="sm")
                    del_run_in = gr.Textbox(placeholder="run id to delete…", show_label=False,
                                            scale=2)
                    del_run_btn = gr.Button("Delete run", variant="stop", size="sm")
                runs_msg = gr.Markdown()
                report_btn = gr.Button("Open the full report in a tab", visible=False)
                report_html = gr.HTML()

                def load_runs():
                    return ops.run_rows(paths.artifacts_dir, supervisor.snapshot()["run_id"])

                def show_run(rows, evt: gr.SelectData):
                    i = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                    try:
                        run_id = rows[i][1]
                    except (IndexError, TypeError):
                        return gr.skip(), gr.skip(), gr.skip()
                    try:
                        ops.report_path(paths.artifacts_dir, run_id, "report.html")
                    except (ValueError, FileNotFoundError):
                        return (f"`{run_id}` has no report.html — it never finished cleanly.",
                                "", gr.update(visible=False))
                    url = f"/report/{run_id}/report.html"
                    return (f"showing `{run_id}`",
                            f"<iframe src='{url}' style='width:100%;height:70vh;border:0'>"
                            f"</iframe>",
                            gr.update(visible=True, link=url))

                runs_df.select(show_run, runs_df, [runs_msg, report_html, report_btn],
                               api_visibility="private")
                reload_btn.click(load_runs, None, runs_df, api_visibility="private")

                def do_delete_run(run_id):
                    ok, msg = ops.delete_runs(paths.artifacts_dir, [(run_id or "").strip()])
                    return load_runs(), ("✅ " if ok else "⚠️ ") + msg, "", gr.update(visible=False)
                del_run_btn.click(do_delete_run, del_run_in,
                                  [runs_df, runs_msg, report_html, report_btn],
                                  api_visibility="private")

        # ── cross-tab wiring, and the initial fill ───────────────────────────────────────
        jump_key.change(lambda k: (gr.Tabs(selected="run"), k) if k
                        else (gr.skip(), gr.skip()),
                        jump_key, [tabs, task_dd], api_visibility="private")

        def boot():
            rows, note = ops.prompt_rows()
            return (rows, ops.task_rows(), gr.update(value=note, visible=bool(note)),
                    ops.file_rows(), load_runs(),
                    gr.update(choices=ops.run_options()))

        demo.load(boot, None,
                  [prompt_df, task_df, broken_md, files_df, runs_df, task_dd],
                  api_visibility="private")

        # Selecting a row on either prompts table loads it / arms the copy button.
        def pick(rows, evt: gr.SelectData) -> str:
            i = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            try:
                return rows[i][0]
            except (IndexError, TypeError):
                return ""

        def open_prompt(rows, evt: gr.SelectData):
            key = pick(rows, evt)
            if not key:
                return gr.skip(), gr.skip(), gr.skip()
            try:
                return ops.model_to_form(store.read_prompt(key)), False, f"editing **{key}**"
            except Exception as exc:  # noqa: BLE001 - a broken file must still open
                return gr.skip(), gr.skip(), f"⚠️ could not open {key}: {exc}"

        prompt_df.select(open_prompt, prompt_df, [editor_state, is_new, prompts_msg],
                         api_visibility="private")
        prompt_df.select(pick, prompt_df, picked_prompt, api_visibility="private")
        task_df.select(pick, task_df, picked_task, api_visibility="private")

        new_btn.click(lambda: (ops.model_to_form(PromptModel()), True,
                               "**New prompt** — name it and add a step."),
                      None, [editor_state, is_new, prompts_msg], api_visibility="private")

        def delete_prompt(key):
            ok, msg = ops.delete_prompt(key)
            rows, note = ops.prompt_rows()
            return (rows, gr.update(value=note, visible=bool(note)),
                    ("✅ " if ok else "⚠️ ") + msg, gr.update(choices=ops.run_options()))
        del_btn.click(delete_prompt, picked_prompt,
                      [prompt_df, broken_md, prompts_msg, task_dd], api_visibility="private")

        def copy_task(task_key, name):
            ok, msg, key = ops.import_task(task_key, name)
            rows, note = ops.prompt_rows()
            if not ok:
                return (rows, gr.update(value=note, visible=bool(note)), f"⚠️ {msg}",
                        gr.skip(), gr.skip(), gr.skip())
            return (rows, gr.update(value=note, visible=bool(note)), f"✅ {msg}",
                    ops.model_to_form(store.read_prompt(key)), False,
                    gr.update(choices=ops.run_options()))
        copy_btn.click(copy_task, [picked_task, copy_name],
                       [prompt_df, broken_md, prompts_msg, editor_state, is_new, task_dd],
                       api_visibility="private")

    return demo


# ── helpers ───────────────────────────────────────────────────────────────────────────────

def _put(form: dict[str, Any], path: tuple, value: Any) -> None:
    """Write one harvested widget value back into the form dict at `path`."""
    if path[0] != "steps":
        form[path[0]] = value
        return
    _, i, field, *rest = path
    step = form["steps"][i]
    if field == "verify":
        step["verify"][rest[0]][rest[1]] = value
    elif field == "values":
        step["values"] = {str(r[0]).strip(): str(r[1]).strip()
                          for r in (value or []) if r and str(r[0]).strip()}
    else:
        step[field] = value


def _to_steps(form: dict[str, Any]) -> dict[str, Any]:
    step = ops.blank_step()
    step["prompt"] = (form.get("prompt") or "").strip()
    form["steps"], form["prompt"] = [step], ""
    return form


def _to_one(form: dict[str, Any]) -> dict[str, Any]:
    form["prompt"] = " ".join((s.get("prompt") or "").strip()
                              for s in form.get("steps") or []).strip()
    form["steps"] = []
    return form
