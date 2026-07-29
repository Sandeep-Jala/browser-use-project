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
            temperature=0.0,
            timeout=45.0,
            # Reliability params pinned rather than inherited from browser-use defaults
            # (these ARE 0.13.3's defaults — pinned so an upgrade can't silently move them).
            # frequency_penalty is deliberately left at its 0.3 default: it exists to stop
            # gpt-4.1-mini's runaway "\t" generation — do not zero it.
            max_retries=5,
            # Applied only to reasoning models (o4-mini): "low" (browser-use's default)
            # produced shallow moves — saving before mandatory fields, a fixed NI where
            # the task said random. The cap must rise with the effort: completion tokens
            # INCLUDE the hidden reasoning tokens, and medium effort at 4096 risks
            # finish_reason='length' with empty content.
            reasoning_effort="low",
            max_completion_tokens=8192,
            # max_completion_tokens= 4096,
            # Best-effort determinism: at temperature 0, Azure still varies across backend
            # replicas; a fixed seed narrows step-to-step decision flakiness.
            seed=42,
        )

    if provider == "groq":
        if not config.groq_api_key:
            raise SystemExit("GROQ_API_KEY is not set in .env (LLM_PROVIDER=groq)")
        return ChatGroq(model=config.groq_model, api_key=config.groq_api_key, temperature=0.0, timeout=45.0)

    raise SystemExit(f"Unknown LLM_PROVIDER: {config.llm_provider!r} (expected 'azure' or 'groq')")


def build_expander_llm(config: Config) -> BaseChatModel | None:
    """Build the model used for one-shot prompt expansion and the QA judge.

    Always on the Azure endpoint (same gpt-4.1-mini deployment as the agent), independent of
    the agent's LLM_PROVIDER. Returns None if no Azure key is configured.
    """
    if not config.azure_api_key:
        return None
    # Same explicit reliability/determinism params as build_llm (see the comments there).
    return ChatOpenAI(
        model=config.expander_model,
        api_key=config.azure_api_key,
        base_url=config.azure_base_url,
        temperature=0.0,
        timeout=45.0,
        max_retries=5,
        reasoning_effort="medium",
        max_completion_tokens=8192,
        seed=42,
    )
