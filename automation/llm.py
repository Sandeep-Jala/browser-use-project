"""LLM provider construction.

One place to turn a `Config` into a browser-use chat model. We develop against
OpenRouter (OpenAI-compatible endpoint) to dodge Groq's free-tier daily token cap, but
Groq is the intended final provider — flip `LLM_PROVIDER=groq` in `.env` to switch back
with no other code changes.
"""
from __future__ import annotations

from browser_use.llm.base import BaseChatModel
from browser_use.llm.groq.chat import ChatGroq
from browser_use.llm.openai.chat import ChatOpenAI

from automation.config import Config


def build_llm(config: Config) -> BaseChatModel:
    """Build the chat model for the provider selected in `config.llm_provider`."""
    provider = config.llm_provider.lower()

    if provider == "openrouter":
        if not config.openrouter_api_key:
            raise SystemExit("OPEN_ROUTER_KEY is not set in .env (LLM_PROVIDER=openrouter)")
        # OpenRouter is OpenAI-compatible: ChatOpenAI + custom base_url.
        return ChatOpenAI(
            model=config.openrouter_model,
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
        )

    if provider == "groq":
        if not config.groq_api_key:
            raise SystemExit("GROQ_API_KEY is not set in .env (LLM_PROVIDER=groq)")
        return ChatGroq(model=config.groq_model, api_key=config.groq_api_key)

    raise SystemExit(f"Unknown LLM_PROVIDER: {config.llm_provider!r} (expected 'openrouter' or 'groq')")


def build_expander_llm(config: Config) -> BaseChatModel | None:
    """Build the model used for one-shot prompt expansion and the QA judge.

    Always via OpenRouter (the expander model id is an OpenRouter id), independent of the
    agent's LLM_PROVIDER. Defaults to the same Maverick model as the agent loop. Returns None
    if no OpenRouter key is configured.
    """
    if not config.openrouter_api_key:
        return None
    return ChatOpenAI(
        model=config.expander_model,
        api_key=config.openrouter_api_key,
        base_url=config.openrouter_base_url,
    )
