"""Central configuration for the automation framework.

A single `Config` object is loaded once from `.env` (via python-dotenv) and passed
explicitly through the pipeline (login -> session -> runner). Keeping everything here
means the Groq model, vision toggle, and output directories each have exactly one place
to change.

Two groups of settings live here:
  * Login settings consumed by `browser/login.py` (login_url, credentials, headless,
    cdp_port, errors_dir).
  * Framework settings for the browser-use agent and telemetry (provider/model/key,
    use_vision, artifacts_dir, prompt expansion).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# --- LLM providers ---
# Azure OpenAI (gpt-4.1-mini) is the active provider. Its /openai/v1 endpoint is
# OpenAI-compatible, so browser-use's ChatOpenAI drives it with just a custom base_url.
# ONE model everywhere: gpt-4.1-mini runs the agent loop, the prompt expander, AND the
# QA judge, so behaviour stays consistent. Groq remains available as an alternate
# (LLM_PROVIDER=groq in .env).
DEFAULT_AZURE_MODEL = "gpt-4.1-mini"
AZURE_BASE_URL = "https://actingoffice-foundry.openai.azure.com/openai/v1"
DEFAULT_GROQ_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
# Prompt-expander / QA-judge model: same Azure deployment, independent of LLM_PROVIDER.
DEFAULT_EXPANDER_MODEL = DEFAULT_AZURE_MODEL


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a truthy/falsey env var like HEADFUL=true."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_history_items(name: str, default: int | None = None) -> int | None:
    """Parse MAX_HISTORY_ITEMS. browser-use requires None or an int > 5, so any unset/blank/
    invalid/out-of-range value falls back to `default`."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        return default
    return value if value > 5 else default


@dataclass
class Config:
    """Resolved configuration for one run of the framework."""

    # --- Login (consumed by login.py) ---
    login_url: str
    login_email: str
    login_password: str
    headless: bool
    cdp_port: int
    errors_dir: Path

    # --- Framework / agent ---
    # Active LLM provider: "azure" (default) | "groq".
    llm_provider: str
    azure_api_key: str | None
    azure_model: str
    azure_base_url: str
    groq_api_key: str | None
    groq_model: str
    use_vision: bool
    # Image detail sent to the model each step: "high" | "low" | "auto". Default "high": a
    # sharper screenshot lets the model actually read icon/label text before acting —
    # misread labels are a direct misclick source for a small model. Costs ~$0.01 extra per
    # 25-step segment; set VISION_DETAIL_LEVEL=auto/low in .env to trade back.
    vision_detail_level: str
    artifacts_dir: Path
    # Cap on how many past steps the agent keeps in context (None = unlimited). Cuts per-step
    # tokens (the resent history grows each step). browser-use requires None or > 5.
    max_history_items: int | None
    # browser-use's plan_update layer: the model re-emits its full plan in every step's
    # output. Set ENABLE_PLANNING=false to A/B off its token weight.
    enable_planning: bool
    # Model for the subtask decomposer, the end-of-run judge, and adapt.parameterize.
    expander_model: str
    # Agent step budget for ONE subtask segment: a subtask is ~a tenth of a task, so 25
    # leaves room to recover from missteps without runaway cost.
    subtask_max_steps: int
    # Semantic subtask router (pipeline/router.py): maps a differently-worded subtask to
    # an existing library skill via alias table -> local embeddings -> one LLM verify.
    # SEMANTIC_ROUTER=false disables it; without the local model installed it silently
    # degrades to alias-only.
    semantic_router: bool
    # Local embedding model for the router (fastembed name; downloaded once to its cache).
    embedding_model: str
    # Inject a stylesheet (script_compile.REVEAL_CSS) into every page that forces the app's
    # hover-revealed / 0-size controls visible, so they enter the agent's snapshot and pass
    # replay's visibility gate instead of relying on the RAW_FIND_JS blind-click fallback.
    # Flip per-ENVIRONMENT, not per-run: recordings authored with it on compile to
    # visibility-requiring click steps that fail replay with it off.
    reveal_hidden_controls: bool

    @classmethod
    def from_env(cls, env_path: str | os.PathLike[str] | None = None) -> "Config":
        """Build a Config from a `.env` file (defaults to ./.env)."""
        load_dotenv(dotenv_path=env_path, override=False)

        # HEADFUL=true means a visible browser, i.e. headless=False.
        headful = _env_bool("HEADFUL", default=False)

        return cls(
            login_url=os.getenv("LOGIN_URL", ""),
            login_email=os.getenv("LOGIN_EMAIL", ""),
            login_password=os.getenv("LOGIN_PASSWORD", ""),
            headless=not headful,
            cdp_port=int(os.getenv("CDP_PORT", "9222")),
            errors_dir=Path(os.getenv("ERRORS_DIR", "errors")),
            llm_provider=os.getenv("LLM_PROVIDER", "azure"),
            azure_api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_model=os.getenv("AZURE_OPENAI_MODEL", DEFAULT_AZURE_MODEL),
            azure_base_url=os.getenv("AZURE_OPENAI_ENDPOINT", AZURE_BASE_URL),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            groq_model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            use_vision=_env_bool("USE_VISION", default=True),
            vision_detail_level=os.getenv("VISION_DETAIL_LEVEL", "high").strip().lower(),
            artifacts_dir=Path(os.getenv("ARTIFACTS_DIR", "artifacts")),
            max_history_items=_env_history_items("MAX_HISTORY_ITEMS", default=20),
            enable_planning=_env_bool("ENABLE_PLANNING", default=True),
            expander_model=os.getenv("EXPANDER_MODEL", DEFAULT_EXPANDER_MODEL),
            subtask_max_steps=int(os.getenv("SUBTASK_MAX_STEPS", "25")),
            semantic_router=_env_bool("SEMANTIC_ROUTER", default=True),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            reveal_hidden_controls=_env_bool("REVEAL_HIDDEN_CONTROLS", default=True),
        )

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist yet."""
        self.errors_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        # Hybrid subtask engine stores (module constants in subtask_store).
        from automation.pipeline import subtask_store as _ss

        _ss.LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
        _ss.DECOMPOSITIONS_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def cdp_url(self) -> str:
        return f"http://localhost:{self.cdp_port}"

    @property
    def active_model(self) -> str:
        """The model id for the currently selected provider (for logging)."""
        if self.llm_provider.lower() == "groq":
            return self.groq_model
        return self.azure_model
