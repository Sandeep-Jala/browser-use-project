"""LLM provider construction.

One place to turn a `Config` into a browser-use chat model. The active provider is Azure
OpenAI (gpt-4.1-mini) via its OpenAI-compatible /openai/v1 endpoint — ChatOpenAI plus a
custom base_url. Groq remains available: flip `LLM_PROVIDER=groq` in `.env` to switch
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

    if provider == "azure":
        if not config.azure_api_key:
            raise SystemExit("AZURE_OPENAI_KEY is not set in .env (LLM_PROVIDER=azure)")
        # Azure's /openai/v1 endpoint is OpenAI-compatible: ChatOpenAI + custom base_url,
        # with the deployment name as the model id.
        return ChatOpenAI(
            model=config.azure_model,
            api_key=config.azure_api_key,
            base_url=config.azure_base_url,
        )

    if provider == "groq":
        if not config.groq_api_key:
            raise SystemExit("GROQ_API_KEY is not set in .env (LLM_PROVIDER=groq)")
        return ChatGroq(model=config.groq_model, api_key=config.groq_api_key)

    raise SystemExit(f"Unknown LLM_PROVIDER: {config.llm_provider!r} (expected 'azure' or 'groq')")


def build_expander_llm(config: Config) -> BaseChatModel | None:
    """Build the model used for one-shot prompt expansion and the QA judge.

    Always on the Azure endpoint (same gpt-4.1-mini deployment as the agent), independent of
    the agent's LLM_PROVIDER. Returns None if no Azure key is configured.
    """
    if not config.azure_api_key:
        return None
    return ChatOpenAI(
        model=config.expander_model,
        api_key=config.azure_api_key,
        base_url=config.azure_base_url,
    )
