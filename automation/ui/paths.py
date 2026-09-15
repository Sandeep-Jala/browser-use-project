"""Where everything lives, resolved once.

Every path the framework uses is RELATIVE (`tasks.yaml`, `prompts/`, `library/`,
`decompositions/`, `artifacts/`, `automation/uploads/`), which is correct for a CLI run from the
repo root and a trap for a long-lived server. This is the one module that turns literals into
paths, so there is a single answer to "relative to what" — and `__main__.py` chdirs to
`repo_root` at startup so the answer is the same for the run subprocess, which inherits it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from automation.config import Config
from automation.pipeline.control import CONTROL_FILENAME

# automation/ui/paths.py -> automation/ui -> automation -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    tasks_file: Path
    prompts_dir: Path
    uploads_dir: Path
    artifacts_dir: Path
    library_dir: Path
    errors_dir: Path
    control_path: Path
    ui_state_dir: Path      # artifacts/.auto_agent: console logs + the run-adoption file

    @classmethod
    def discover(cls, config: Config | None = None) -> "Paths":
        """Resolve against the repo root, honouring ARTIFACTS_DIR / ERRORS_DIR from the
        environment so the UI writes where the CLI writes."""
        from automation import tasks as tasks_mod
        from automation.pipeline import files as pfiles
        from automation.pipeline import subtask_store as sstore

        cfg = config if config is not None else Config.from_env()
        root = REPO_ROOT

        def _abs(p: Path) -> Path:
            return p if p.is_absolute() else root / p

        artifacts = _abs(Path(cfg.artifacts_dir))
        return cls(
            repo_root=root,
            tasks_file=_abs(tasks_mod.TASKS_FILE),
            prompts_dir=_abs(tasks_mod.PROMPTS_DIR),
            uploads_dir=_abs(pfiles.UPLOADS_DIR),
            artifacts_dir=artifacts,
            library_dir=_abs(sstore.LIBRARY_DIR),
            errors_dir=_abs(Path(cfg.errors_dir)),
            control_path=artifacts / CONTROL_FILENAME,
            ui_state_dir=artifacts / ".auto_agent",
        )
