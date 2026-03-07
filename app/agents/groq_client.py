"""
Groq Client — optional cloud LLM provider.
Only initialized when GROQ_API_KEY is set.
"""

import json
import logging
import asyncio
from typing import TypeVar, Type

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings
from app.core.exceptions import LLMError, LLMResponseValidationError

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class TokenUsageTracker:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0

    def record(self, prompt_tokens: int, completion_tokens: int):
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_requests += 1

    def summary(self) -> str:
        total = self.total_prompt_tokens + self.total_completion_tokens
        return f"groq[reqs={self.total_requests}, prompt={self.total_prompt_tokens}, completion={self.total_completion_tokens}, total={total}]"


class GroqClient:
    def __init__(self):
        if not settings.groq_api_key:
            raise LLMError("GROQ_API_KEY is not configured")

        from groq import AsyncGroq
        self._client = AsyncGroq(api_key=settings.groq_api_key)
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_llm_calls)
        self.token_usage = TokenUsageTracker()

    @retry(
        stop=stop_after_attempt(settings.llm_retry_attempts),
        wait=wait_exponential(multiplier=settings.llm_retry_delay, min=2, max=60),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    async def chat(
        self,
        prompt: str,
        system: str = "You are a helpful assistant.",
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_fast_model: bool = False,
    ) -> str:
        model = settings.groq_fast_model if use_fast_model else settings.groq_model
        temp = temperature if temperature is not None else settings.groq_temperature
        max_tok = max_tokens or settings.groq_max_tokens

        async with self._semaphore:
            response = await self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temp,
                max_tokens=max_tok,
            )

            content = response.choices[0].message.content or ""
            usage = response.usage
            if usage:
                self.token_usage.record(usage.prompt_tokens, usage.completion_tokens)
                logger.info(
                    "Groq API success | model=%s | prompt_tok=%d | completion_tok=%d",
                    model, usage.prompt_tokens, usage.completion_tokens,
                )
            return content

    async def structured_chat(
        self,
        prompt: str,
        system: str,
        response_model: Type[T],
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_fast_model: bool = False,
    ) -> T:
        schema = json.dumps(response_model.model_json_schema(), indent=2)
        enhanced_system = (
            f"{system}\n\nRespond with ONLY valid JSON matching this schema:\n"
            f"```json\n{schema}\n```\n"
            f"No markdown, no extra text. Just the JSON object."
        )

        content = await self.chat(prompt, enhanced_system, temperature, max_tokens, use_fast_model)

        try:
            cleaned = content.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:])
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()

            data = json.loads(cleaned)
            return response_model.model_validate(data)
        except (json.JSONDecodeError, Exception) as e:
            raise LLMResponseValidationError(
                f"Groq returned invalid JSON: {e}",
                details=content[:500],
            )