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
# We develop against OpenRouter (OpenAI-compatible) to avoid Groq's free-tier daily
# token cap, but Groq is the intended FINAL provider. Switch with LLM_PROVIDER in .env;
# each provider's model is a single swap point below.
#
# ONE model everywhere: Llama 4 Maverick drives the agent loop, the prompt expander, AND the
# QA judge. Maverick (128 experts) grounds this app far better than Scout did, and the same
# id is available on both OpenRouter and Groq, so dev mirrors the Groq production target.
DEFAULT_GROQ_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-4-maverick"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Prompt-expander / QA-judge model: same Maverick (always via OpenRouter, independent of
# LLM_PROVIDER). One model for everything keeps behaviour consistent and avoids an Anthropic
# dependency.
DEFAULT_EXPANDER_MODEL = "meta-llama/llama-4-maverick"


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a truthy/falsey env var like HEADFUL=true."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_history_items(name: str) -> int | None:
    """Parse MAX_HISTORY_ITEMS. browser-use requires None or an int > 5, so any unset/blank/
    invalid/out-of-range value falls back to None (unlimited history)."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw.strip())
    except ValueError:
        return None
    return value if value > 5 else None


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
    # Active LLM provider: "openrouter" (dev) | "groq" (final).
    llm_provider: str
    groq_api_key: str | None
    groq_model: str
    openrouter_api_key: str | None
    openrouter_model: str
    openrouter_base_url: str
    use_vision: bool
    artifacts_dir: Path
    # Cap on how many past steps the agent keeps in context (None = unlimited). Cuts per-step
    # tokens (the resent history grows each step). browser-use requires None or > 5.
    max_history_items: int | None
    # Prompt expansion: rewrite the task into step-by-step instructions before running.
    expand_prompt: bool
    expander_model: str

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
            llm_provider=os.getenv("LLM_PROVIDER", "openrouter"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            groq_model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            openrouter_api_key=os.getenv("OPEN_ROUTER_KEY"),
            openrouter_model=os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
            use_vision=_env_bool("USE_VISION", default=True),
            artifacts_dir=Path(os.getenv("ARTIFACTS_DIR", "artifacts")),
            max_history_items=_env_history_items("MAX_HISTORY_ITEMS"),
            expand_prompt=_env_bool("EXPAND_PROMPT", default=True),
            expander_model=os.getenv("EXPANDER_MODEL", DEFAULT_EXPANDER_MODEL),
        )

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist yet."""
        self.errors_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cdp_url(self) -> str:
        return f"http://localhost:{self.cdp_port}"

    @property
    def active_model(self) -> str:
        """The model id for the currently selected provider (for logging)."""
        if self.llm_provider.lower() == "groq":
            return self.groq_model
        return self.openrouter_model
