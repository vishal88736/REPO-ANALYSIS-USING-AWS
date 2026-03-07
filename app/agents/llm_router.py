"""
LLM Router — ALL tasks go to Bedrock qwen.qwen3-coder-next.
  262K context window, 8K output per call.
  Groq is FALLBACK ONLY (12K TPM / 100K daily limit too low for primary use).

Fallback chain: Bedrock → Gemini → Groq heavy → Groq fast
"""

import logging
from enum import Enum
from typing import TypeVar, Type

from pydantic import BaseModel

from app.config import settings
from app.agents.groq_client import GroqClient
from app.core.exceptions import LLMError

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class TaskType(str, Enum):
    FILE_ANALYSIS = "file_analysis"
    ARCHITECTURE = "architecture"
    MERMAID = "mermaid"
    QUERY_PLANNING = "query_planning"
    QUERY_ANSWER = "query_answer"
    RAG_ANSWER = "rag_answer"
    COMPREHENSIVE_SUMMARY = "comprehensive_summary"
    DIAGRAM_ENRICHMENT = "diagram_enrichment"


# ═══════════════════════════════════════════════════════════════════
# ALL tasks → Bedrock (primary)
# Groq only used as emergency fallback when Bedrock is down/throttled
# ═══════════════════════════════════════════════════════════════════

DEFAULT_ROUTING = {
    TaskType.FILE_ANALYSIS:         "bedrock",
    TaskType.ARCHITECTURE:          "bedrock",
    TaskType.MERMAID:               "bedrock",
    TaskType.QUERY_PLANNING:        "bedrock",
    TaskType.QUERY_ANSWER:          "bedrock",
    TaskType.RAG_ANSWER:            "bedrock",
    TaskType.COMPREHENSIVE_SUMMARY: "bedrock",
    TaskType.DIAGRAM_ENRICHMENT:    "bedrock",
}


class LLMRouter:
    """
    Central LLM dispatch — Bedrock qwen.qwen3-coder-next for everything.

    Why not Groq?
      - 12K TPM = can only process ~3 files/minute
      - 100K/day = only ~25 file analyses per day
      - Bedrock has no TPM limit, just $/token

    Fallback: Bedrock → Gemini → Groq heavy → Groq fast
    """

    def __init__(self, groq: GroqClient):
        self._groq = groq
        self._bedrock = None
        self._bedrock_init_attempted = False
        self._gemini = None
        self._gemini_init_attempted = False

    # ── Lazy init: Bedrock ──

    def _get_bedrock(self):
        if self._bedrock is not None:
            return self._bedrock
        if self._bedrock_init_attempted:
            return None

        self._bedrock_init_attempted = True
        if not settings.bedrock_available:
            logger.warning("Bedrock not configured — falling back to Groq (WILL BE SLOW)")
            return None

        try:
            from app.agents.bedrock_client import BedrockClient
            self._bedrock = BedrockClient()
            return self._bedrock
        except Exception as e:
            logger.error("Bedrock init failed: %s — falling back to Groq", e)
            return None

    # ── Lazy init: Gemini ──

    def _get_gemini(self):
        if self._gemini is not None:
            return self._gemini
        if self._gemini_init_attempted:
            return None

        self._gemini_init_attempted = True
        if not settings.gemini_available:
            return None

        try:
            from app.agents.gemini_client import GeminiClient
            self._gemini = GeminiClient()
            logger.info("Gemini initialized: %s", settings.gemini_model)
            return self._gemini
        except Exception as e:
            logger.warning("Gemini init failed: %s", e)
            return None

    # ── Main chat ──

    async def chat(
        self,
        task: TaskType,
        prompt: str,
        system: str = "You are a helpful assistant.",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        temp = temperature if temperature is not None else settings.bedrock_temperature

        # PRIMARY: Bedrock
        bedrock = self._get_bedrock()
        if bedrock:
            try:
                logger.debug("Routing %s → Bedrock qwen3-coder-next", task.value)
                return await bedrock.chat(prompt, system, temp, max_tokens)
            except Exception as e:
                logger.warning("Bedrock failed for %s: %s — trying fallbacks", task.value, e)

        # FALLBACK chain
        return await self._fallback_chat(task, prompt, system, temp, max_tokens)

    async def _fallback_chat(
        self, task, prompt, system, temperature, max_tokens, original_error=None,
    ) -> str:
        """Fallback: Gemini → Groq heavy → Groq fast."""

        # Try Gemini
        gemini = self._get_gemini()
        if gemini:
            try:
                logger.info("Fallback %s → Gemini", task.value)
                return await gemini.generate(prompt, system, temperature)
            except Exception as e:
                logger.warning("Gemini fallback failed: %s", e)

        # Try Groq heavy (will be slow due to 12K TPM)
        if settings.groq_available:
            try:
                logger.info("Fallback %s → Groq heavy (12K TPM — expect slowness)", task.value)
                return await self._groq.chat(prompt, system, temperature, max_tokens)
            except Exception:
                pass

            # Last resort: Groq fast
            try:
                logger.info("Fallback %s → Groq fast", task.value)
                return await self._groq.chat(prompt, system, temperature, max_tokens, use_fast_model=True)
            except Exception as e:
                raise LLMError(f"All providers failed for {task.value}: {e}") from (original_error or e)

        raise LLMError(f"Bedrock failed and no fallback configured for {task.value}")

    # ── Structured chat (JSON → Pydantic) ──

    async def structured_chat(
        self,
        task: TaskType,
        prompt: str,
        system: str,
        response_model: Type[T],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        temp = temperature if temperature is not None else settings.bedrock_temperature

        # PRIMARY: Bedrock
        bedrock = self._get_bedrock()
        if bedrock:
            try:
                logger.debug("Structured %s → Bedrock", task.value)
                return await bedrock.structured_chat(prompt, system, response_model, temp, max_tokens)
            except Exception as e:
                logger.warning("Structured Bedrock failed for %s: %s", task.value, e)

        # Fallback: Gemini → Groq
        gemini = self._get_gemini()
        if gemini:
            try:
                return await gemini.structured_generate(prompt, system, response_model, temp)
            except Exception:
                pass

        if settings.groq_available:
            try:
                return await self._groq.structured_chat(
                    prompt=prompt, system=system, response_model=response_model,
                    temperature=temp, max_tokens=max_tokens,
                )
            except Exception:
                return await self._groq.structured_chat(
                    prompt=prompt, system=system, response_model=response_model,
                    temperature=temp, max_tokens=max_tokens, use_fast_model=True,
                )

        raise LLMError(f"All providers failed for structured {task.value}")