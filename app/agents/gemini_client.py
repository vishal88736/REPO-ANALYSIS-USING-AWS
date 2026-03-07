"""
Feature 13: Google Gemini Client.
Async wrapper using google-generativeai SDK.
Gemini supports large context windows — used for architecture reasoning,
query planning, and fallback when Groq rate limits.
"""

import json
import logging
import asyncio
from typing import TypeVar, Type
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from app.config import settings
from app.core.exceptions import LLMError, LLMResponseValidationError

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_executor = ThreadPoolExecutor(max_workers=4)


class GeminiClient:
    """
    Async Gemini wrapper.
    The google-generativeai SDK is synchronous, so we wrap calls
    with asyncio.run_in_executor + ThreadPoolExecutor.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key or settings.gemini_api_key
        self._model_name = model or settings.gemini_model

        if not self._api_key:
            raise LLMError("GEMINI_API_KEY is not configured")

        try:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            self._model = genai.GenerativeModel(self._model_name)
            self._genai = genai
        except ImportError:
            raise LLMError("google-generativeai package not installed. Run: pip install google-generativeai")

        self.total_requests = 0
        self.total_tokens = 0

    def _sync_generate(self, prompt: str, system: str = "", temperature: float = 0.3) -> str:
        """Synchronous Gemini call."""
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        try:
            response = self._model.generate_content(
                full_prompt,
                generation_config=self._genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=4096,
                ),
            )
            self.total_requests += 1
            text = response.text or ""

            # Track tokens if available
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                self.total_tokens += getattr(response.usage_metadata, "total_token_count", 0)

            return text
        except Exception as e:
            logger.error("Gemini API error: %s", e)
            raise LLMError(f"Gemini API error: {e}")

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
    ) -> str:
        """Async Gemini call via thread executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor,
            self._sync_generate,
            prompt,
            system,
            temperature,
        )

    async def structured_generate(
        self,
        prompt: str,
        system: str,
        response_model: Type[T],
        temperature: float = 0.3,
    ) -> T:
        """Generate structured JSON output validated against Pydantic model."""
        schema_desc = json.dumps(response_model.model_json_schema(), indent=2)
        enhanced_system = (
            f"{system}\n\n"
            f"Respond with ONLY valid JSON matching this schema:\n"
            f"```json\n{schema_desc}\n```\n"
            f"No markdown fences. No extra text."
        )

        content = await self.generate(prompt, enhanced_system, temperature)

        try:
            cleaned = content.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
            data = json.loads(cleaned)
            return response_model.model_validate(data)
        except (json.JSONDecodeError, Exception) as e:
            logger.error("Gemini JSON parse failed: %s", e)
            raise LLMResponseValidationError(f"Gemini returned invalid JSON: {e}", details=content[:500])

    def stats(self) -> dict:
        return {
            "provider": "gemini",
            "model": self._model_name,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
        }