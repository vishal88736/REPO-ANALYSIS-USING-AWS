"""
AWS Bedrock client — Qwen3-Coder-Next (262K context, 8K output).
Used for: architecture summary, 5000-word report, RAG answers.
Uses the Converse API for uniform model access.
"""

import json
import logging
import asyncio
from typing import TypeVar, Type

import boto3
from botocore.config import Config as BotoConfig
from pydantic import BaseModel

from app.config import settings
from app.core.exceptions import LLMError

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class BedrockClient:
    """
    Async wrapper around AWS Bedrock Converse API.
    Optimized for qwen.qwen3-coder-next:
      - 262,144 token context window (can fit entire codebases)
      - 8,192 token max output (split large outputs into parallel calls)
    """

    def __init__(
        self,
        region: str | None = None,
        model_id: str | None = None,
        max_concurrent: int = 10,
    ):
        self._region = region or settings.aws_region
        self._model_id = model_id or settings.bedrock_model_id
        self._max_output = settings.bedrock_max_tokens  # 8192 for qwen3-coder-next
        self._timeout = settings.bedrock_timeout

        boto_config = BotoConfig(
            region_name=self._region,
            read_timeout=self._timeout,
            connect_timeout=30,
            retries={"max_attempts": 3, "mode": "adaptive"},
        )

        self._client = boto3.client(
            "bedrock-runtime",
            region_name=self._region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            config=boto_config,
        )
        self._semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(
            "Bedrock client: model=%s region=%s context=262K output=%d",
            self._model_id, self._region, self._max_output,
        )

    async def chat(
        self,
        prompt: str,
        system: str = "You are a helpful assistant.",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion to Bedrock Qwen3-Coder-Next."""
        temp = temperature if temperature is not None else settings.bedrock_temperature
        tokens = min(max_tokens or self._max_output, self._max_output)

        messages = [{"role": "user", "content": [{"text": prompt}]}]
        system_list = [{"text": system}]

        async with self._semaphore:
            try:
                response = await asyncio.to_thread(
                    self._client.converse,
                    modelId=self._model_id,
                    messages=messages,
                    system=system_list,
                    inferenceConfig={
                        "temperature": temp,
                        "maxTokens": tokens,
                    },
                )

                output = response.get("output", {})
                message = output.get("message", {})
                content_blocks = message.get("content", [])

                text = ""
                for block in content_blocks:
                    if "text" in block:
                        text += block["text"]

                usage = response.get("usage", {})
                input_tokens = usage.get("inputTokens", 0)
                output_tokens = usage.get("outputTokens", 0)
                stop_reason = response.get("stopReason", "")

                logger.info(
                    "Bedrock OK | model=%s | in=%d | out=%d | stop=%s",
                    self._model_id, input_tokens, output_tokens, stop_reason,
                )

                if stop_reason == "max_tokens":
                    logger.warning(
                        "Bedrock hit max_tokens (%d) — output may be truncated", tokens,
                    )

                return text

            except self._client.exceptions.ThrottlingException as e:
                logger.warning("Bedrock throttled: %s — will retry", e)
                raise LLMError(f"Bedrock throttled: {e}") from e

            except self._client.exceptions.ModelTimeoutException as e:
                logger.warning("Bedrock timeout: %s", e)
                raise LLMError(f"Bedrock timeout: {e}") from e

            except Exception as e:
                logger.error("Bedrock failed: %s", e)
                raise LLMError(f"Bedrock call failed: {e}") from e

    async def structured_chat(
        self,
        prompt: str,
        system: str,
        response_model: Type[T],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """Get structured JSON from Qwen3-Coder-Next, parsed into Pydantic model."""
        schema_hint = json.dumps(response_model.model_json_schema(), indent=2)
        full_prompt = (
            f"{prompt}\n\n"
            f"IMPORTANT: Return ONLY valid JSON matching this schema (no markdown, no explanation):\n"
            f"{schema_hint}"
        )
        full_system = f"{system}\nYou MUST return only valid JSON. No markdown fences, no text before/after."

        raw = await self.chat(full_prompt, full_system, temperature, max_tokens)

        # Extract JSON
        first = raw.find("{")
        last = raw.rfind("}")
        if first == -1 or last == -1:
            # Try array
            first = raw.find("[")
            last = raw.rfind("]")
        if first == -1 or last == -1:
            raise LLMError(f"Bedrock returned no JSON: {raw[:300]}")

        json_str = raw[first:last + 1]
        try:
            data = json.loads(json_str)
            return response_model.model_validate(data)
        except Exception as e:
            raise LLMError(f"JSON parse failed: {e}\nRaw: {json_str[:500]}") from e