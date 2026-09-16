"""`python -m automation.ui` — start Auto Agent.

Binds 127.0.0.1 only, deliberately and not configurably: this server starts runs that drive a
browser already logged into a live system, and it deletes run directories. Letting it listen on the
LAN would be a different product with a different security model.

**Gradio is mounted onto our own FastAPI app rather than started with `demo.launch()`**, for two
reasons that both outlive the convenience:

* `_LocalOnly` survives. Gradio has no Host-header check, and binding to loopback does not stop a
  page on the internet resolving its own hostname to 127.0.0.1 and driving this server. Cheap
  insurance for something that spawns browser runs and `rmtree`s directories.
* Run reports get their own guarded route instead of Gradio's blanket `allowed_paths`, which would
  expose every file under `artifacts/` — `network.json` and `console.json` included — to anything
  that reaches the port.

A consequence worth stating: `share=False` is not passed anywhere because `launch()` is never
called, so there is no code path that could mint a public share link.

The `chdir` is load-bearing. `tasks.yaml`, `prompts/`, `library/`, `decompositions/`, `artifacts/`
and `automation/uploads/` are all relative paths in the modules the UI reuses, and the run
subprocess inherits the working directory — so the server and its children must agree, and the repo
root is the only answer that matches the CLI.
"""
from __future__ import annotations

import argparse
import logging
import os
import threading
import webbrowser
from typing import Callable

from automation.ui.paths import REPO_ROOT, Paths

DEFAULT_PORT = 8765

_ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]", "::1")


class _LocalOnly:
    """Reject a request whose Host header is not this machine.

    DNS rebinding: a page on the internet can resolve its own hostname to 127.0.0.1 and then talk
    to whatever is listening. Binding to loopback does not stop that; checking Host does.

    Raw ASGI rather than a Starlette/FastAPI middleware class so it sits in front of everything,
    Gradio's own routes included, with no framework of its own.
    """

    def __init__(self, app: Callable) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope.get("type") == "http":
            host = ""
            for key, value in scope.get("headers") or []:
                if key == b"host":
                    host = value.decode("latin-1")
                    break
            hostname = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
            if hostname and hostname not in _ALLOWED_HOSTS:
                body = b"Auto Agent only answers on 127.0.0.1"
                await send({"type": "http.response.start", "status": 403,
                            "headers": [(b"content-type", b"text/plain; charset=utf-8"),
                                        (b"content-length", str(len(body)).encode())]})
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


def build_app(paths: Paths, supervisor):
    """The ASGI app: our routes first, Gradio mounted at `/`, `_LocalOnly` wrapped around both."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    import gradio as gr

    from automation.ui import ops, uploads
    from automation.ui.gradio_app import build_blocks

    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @api.get("/report/{run_id}/{name}")
    def artifact(run_id: str, name: str):
        """One run's files. The `/{name}` segment is not padding: report.html references run.mp4
        RELATIVELY, so the sibling only resolves if the report is served from a directory-shaped
        URL. Guards live in ops.report_path — run-id shape, filename shape, then resolve() before
        the containment test, so a symlink planted in the run dir cannot point out of it."""
        try:
            return FileResponse(ops.report_path(paths.artifacts_dir, run_id, name))
        except (ValueError, FileNotFoundError):
            raise HTTPException(status_code=404, detail="no such file")

    demo = build_blocks(paths, supervisor)
    demo.queue(default_concurrency_limit=None)   # Gradio defaults to 1, which would put the
                                                 # 1 s live tick behind a slow save. The
                                                 # one-run-at-a-time invariant is held by
                                                 # supervisor._lock, not by this queue.
    app = gr.mount_gradio_app(api, demo, path="/", ssr_mode=False,
                              max_file_size=uploads.MAX_UPLOAD_BYTES,
                              show_error=True, footer_links=[],
                              # Gradio builds its OWN FastAPI app internally and mounts it here;
                              # without this it publishes a 39 KB OpenAPI schema of its routes,
                              # /login among them, which this app never uses. Silencing the
                              # parent's docs is not enough — the schema comes from the child.
                              app_kwargs={"docs_url": None, "redoc_url": None,
                                          "openapi_url": None})
    return _LocalOnly(app)


def _refuse_if_port_taken(port: int) -> None:
    """Exit with a sentence the user can act on if something already holds the port.

    Probes with a real bind rather than a connect: a connect test would miss a socket bound but
    not accepting, and would also report "free" for a port held by a process that is wedged.
    SO_REUSEADDR is deliberately NOT set — uvicorn will not set it either, so the probe has to
    fail under exactly the conditions uvicorn would.
    """
    import socket
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", port))
    except OSError as exc:
        owner = ""
        try:
            import psutil
            for conn in psutil.net_connections(kind="inet"):
                if conn.laddr and conn.laddr.port == port and conn.status == "LISTEN" and conn.pid:
                    owner = f" It is held by pid {conn.pid} ({psutil.Process(conn.pid).name()})."
                    break
        except Exception:  # noqa: BLE001 - naming the owner is a courtesy, never a requirement
            pass
        raise SystemExit(
            f"[!] could not listen on port {port}: {exc}.{owner} Something else is using it — "
            f"stop it, or pass --port to pick another. NOTE: whatever is answering on "
            f"http://127.0.0.1:{port}/ right now is that other process, not this one.") from None
    finally:
        probe.close()


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="auto-agent",
        description="Auto Agent — author, run and steer automation prompts in a browser.")
    parser.add_argument("--port", type=int,
                        default=int(os.getenv("AUTO_AGENT_PORT", str(DEFAULT_PORT))),
                        help=f"port on 127.0.0.1 (default {DEFAULT_PORT})")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open a browser window on startup")
    args = parser.parse_args()

    # First, before any other work: a start that cannot bind should do NOTHING observable.
    # Adopting a running task and logging "re-adopted run pid N" from a process that is about to
    # exit reads like this server took charge of that run. It did not.
    _refuse_if_port_taken(args.port)

    os.chdir(REPO_ROOT)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Before gradio is imported: it reads this at import time. No telemetry and no version ping
    # from a tool that drives a browser logged into a live system.
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

    paths = Paths.discover()
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    paths.ui_state_dir.mkdir(parents=True, exist_ok=True)
    paths.prompts_dir.mkdir(parents=True, exist_ok=True)

    from automation.ui.supervisor import RunSupervisor
    supervisor = RunSupervisor(paths)
    # One supervisor per process, never per session: it owns artifacts/control.json and the run
    # subprocess. Re-attach to a run this server started before it was restarted.
    supervisor.adopt()

    url = f"http://127.0.0.1:{args.port}/"
    print(f"[*] Auto Agent on {url}")
    print(f"[*] prompts: {paths.prompts_dir}")
    print(f"[*] runs:    {paths.artifacts_dir}")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()

    import uvicorn
    uvicorn.run(build_app(paths, supervisor), host="127.0.0.1", port=args.port,
                log_level="warning")


if __name__ == "__main__":
    cli()
