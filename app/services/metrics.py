"""
Feature 12 + 17: Token Usage Metrics.
Per-stage, per-provider tracking.
"""

import logging
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)


class PipelineMetrics:
    """Track token usage per stage and per provider."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._stages: dict[str, dict] = defaultdict(lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "requests": 0,
            "provider": "",
        })
        self._providers: dict[str, dict] = defaultdict(lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "requests": 0,
        })

    async def record(
        self,
        stage: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
    ):
        async with self._lock:
            self._stages[stage]["prompt_tokens"] += prompt_tokens
            self._stages[stage]["completion_tokens"] += completion_tokens
            self._stages[stage]["requests"] += 1
            self._stages[stage]["provider"] = provider

            self._providers[provider]["prompt_tokens"] += prompt_tokens
            self._providers[provider]["completion_tokens"] += completion_tokens
            self._providers[provider]["requests"] += 1

    def summary(self) -> dict:
        total_prompt = sum(s["prompt_tokens"] for s in self._stages.values())
        total_completion = sum(s["completion_tokens"] for s in self._stages.values())

        return {
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "by_stage": dict(self._stages),
            "by_provider": dict(self._providers),
        }

    def log_summary(self):
        s = self.summary()
        logger.info("=" * 60)
        logger.info("TOKEN USAGE SUMMARY")
        logger.info("  Total: %d tokens (prompt=%d, completion=%d)",
                     s["total_tokens"], s["total_prompt_tokens"], s["total_completion_tokens"])
        for stage, data in s["by_stage"].items():
            logger.info("  [%s] %d tokens (%d requests) via %s",
                         stage, data["prompt_tokens"] + data["completion_tokens"],
                         data["requests"], data["provider"])
        for provider, data in s["by_provider"].items():
            logger.info("  Provider %s: %d tokens (%d requests)",
                         provider, data["prompt_tokens"] + data["completion_tokens"],
                         data["requests"])
        logger.info("=" * 60)