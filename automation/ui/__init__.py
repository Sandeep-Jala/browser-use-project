"""Auto Agent — the local web UI for authoring and running prompts.

Three invariants hold across every module in this package. Each exists because breaking it
caused, or would cause, a concrete failure:

1. **Never import `browser_use`.** Importing it installs a `StreamHandler` on the ROOT logger at
   import time (`browser_use/__init__.py` calls `setup_logging()`), which would hijack the
   server's own logging, and it costs seconds of import. Everything the UI needs is
   browser-use-free: `automation.tasks`, `pipeline.checks`, `pipeline.subtask_store`,
   `pipeline.files`, `pipeline.control`, `pipeline.report`. Never `pipeline.hybrid` or
   `pipeline.runner`.

2. **Run the server with the repo root as cwd.** `tasks.yaml`, `prompts/`, `library/`,
   `decompositions/`, `artifacts/` and `automation/uploads/` are all RELATIVE paths in the
   modules the UI reuses, and the run subprocess inherits the directory. `__main__.py` chdirs
   once at startup so there is one answer for both the server and its children.

3. **One run at a time.** The operator control channel is a module global plus ONE file
   (`artifacts/control.json`, per install, not per run), so two concurrent runs would steal each
   other's pause and stop commands — and there is only one browser to drive anyway. The
   supervisor enforces it with a lock, not a state check.

The UI also never writes `tasks.yaml`: its comments are the authoring record. User prompts live
in `prompts/<key>.yaml`, one per file, in the same schema.

Layout: `gradio_app.py` is the Gradio `Blocks` — layout and event wiring, nothing else. `ops.py`
below it holds every callback that has a decision in it and imports no gradio at all. That seam is
deliberate: it is what lets `tests/test_ui_ops.py` and `tests/test_ui_editor.py` pin the behaviour
as plain functions, without standing up a server or poking at a Blocks' internals.
"""
