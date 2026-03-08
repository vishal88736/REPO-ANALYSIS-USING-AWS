"""
LLM Router — routes all tasks to Bedrock (primary).
Fallback chain: Bedrock → Gemini → Groq (optional)
"""

import logging
from enum import Enum
from typing import TypeVar, Type

from pydantic import BaseModel

from app.config import settings
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


class LLMRouter:
    """
    Central LLM dispatch.

    Primary: AWS Bedrock (Qwen3-Coder-Next)
    Optional fallback: Gemini → Groq
    """

    def __init__(self, groq=None):
        self._groq = groq
        self._bedrock = None
        self._bedrock_init_attempted = False
        self._gemini = None
        self._gemini_init_attempted = False

    # ------------------------------------------------
    # Bedrock lazy initialization
    # ------------------------------------------------

    def _get_bedrock(self):
        if self._bedrock is not None:
            return self._bedrock

        if self._bedrock_init_attempted:
            return None

        self._bedrock_init_attempted = True

        if not settings.bedrock_available:
            logger.warning("Bedrock credentials not configured")
            return None

        try:
            from app.agents.bedrock_client import BedrockClient

            self._bedrock = BedrockClient()
            logger.info("Bedrock initialized: %s", settings.bedrock_model_id)
            return self._bedrock

        except Exception as e:
            logger.error("Bedrock initialization failed: %s", e)
            return None

    # ------------------------------------------------
    # Gemini lazy initialization
    # ------------------------------------------------

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
            logger.warning("Gemini initialization failed: %s", e)
            return None

    # ------------------------------------------------
    # Standard chat
    # ------------------------------------------------

    async def chat(
        self,
        task: TaskType,
        prompt: str,
        system: str = "You are a helpful assistant.",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:

        temp = temperature if temperature is not None else settings.bedrock_temperature

        bedrock = self._get_bedrock()

        if bedrock:
            try:
                logger.debug("Routing %s → Bedrock", task.value)

                return await bedrock.chat(
                    prompt=prompt,
                    system=system,
                    temperature=temp,
                    max_tokens=max_tokens,
                )

            except Exception as e:
                logger.warning("Bedrock failed: %s", e)

        return await self._fallback_chat(task, prompt, system, temp, max_tokens)

    # ------------------------------------------------
    # Structured chat
    # ------------------------------------------------

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

        bedrock = self._get_bedrock()

        if bedrock:
            try:
                logger.debug("Structured routing %s → Bedrock", task.value)

                return await bedrock.structured_chat(
                    prompt=prompt,
                    system=system,
                    response_model=response_model,
                    temperature=temp,
                    max_tokens=max_tokens,
                )

            except Exception as e:
                logger.warning("Bedrock structured call failed: %s", e)

        return await self._fallback_structured(task, prompt, system, response_model, temp, max_tokens)

    # ------------------------------------------------
    # Fallback logic
    # ------------------------------------------------

    async def _fallback_chat(self, task, prompt, system, temperature, max_tokens):

        gemini = self._get_gemini()

        if gemini:
            try:
                logger.info("Fallback %s → Gemini", task.value)
                return await gemini.generate(prompt, system, temperature)
            except Exception:
                pass

        if self._groq and settings.groq_available:
            try:
                logger.info("Fallback %s → Groq", task.value)
                return await self._groq.chat(prompt, system, temperature, max_tokens)
            except Exception as e:
                raise LLMError(f"All providers failed: {e}")

        raise LLMError("No LLM provider available")

    async def _fallback_structured(
        self,
        task,
        prompt,
        system,
        response_model,
        temperature,
        max_tokens,
    ):

        gemini = self._get_gemini()

        if gemini:
            try:
                return await gemini.structured_generate(
                    prompt,
                    system,
                    response_model,
                    temperature,
                )
            except Exception:
                pass

        if self._groq and settings.groq_available:
            try:
                return await self._groq.structured_chat(
                    prompt,
                    system,
                    response_model,
                    temperature,
                    max_tokens,
                )
            except Exception as e:
                raise LLMError(f"All providers failed: {e}")

        raise LLMError("No structured LLM provider available")