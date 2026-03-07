"""
VYUHA Bedrock Assembler — sends the FULL prompt to Bedrock Qwen3-480B.
No staging, no splitting. One shot, full diagram_spec.
"""

import json
import logging

from app.agents.llm_router import LLMRouter, TaskType
from app.vyuha.scanner_models import ParsedRepo
from app.vyuha.vyuha_prompt import VYUHA_SYSTEM_PROMPT, build_vyuha_prompt
from app.vyuha.assembler import build_diagram_spec as staged_build

logger = logging.getLogger(__name__)


async def build_diagram_spec_bedrock(
    repo: ParsedRepo,
    router: LLMRouter,
) -> dict:
    """
    Build VYUHA diagram_spec using Bedrock 480B in a single call.
    Falls back to the staged pipeline if Bedrock is unavailable.
    """
    bedrock = router._get_bedrock()

    if bedrock:
        logger.info("VYUHA: Using Bedrock single-shot for %s (%d nodes)",
                     repo.meta.repo_name, len(repo.nodes))

        # Serialize parsed repo to JSON
        repo_json = repo.model_dump_json(indent=1)

        # Build full prompt
        prompt = build_vyuha_prompt(repo_json)

        logger.info("VYUHA prompt: ~%d chars", len(prompt))

        try:
            result = await bedrock.raw_json_chat(
                prompt=prompt,
                system=VYUHA_SYSTEM_PROMPT,
                temperature=0.15,
                max_tokens=50000,
            )

            # Validate essential structure
            required_keys = {"meta", "architecture", "logical_flow", "summary"}
            if not required_keys.issubset(result.keys()):
                missing = required_keys - result.keys()
                logger.warning("VYUHA output missing keys: %s — falling back", missing)
                raise ValueError(f"Missing: {missing}")

            logger.info(
                "VYUHA Bedrock OK | arch_nodes=%d | flow_steps=%d",
                len(result.get("architecture", {}).get("nodes", [])),
                len(result.get("logical_flow", {}).get("steps", [])),
            )

            return result

        except Exception as e:
            logger.warning("VYUHA Bedrock failed: %s — falling back to staged pipeline", e)

    # Fallback: staged pipeline (works with any model)
    logger.info("VYUHA: Using staged pipeline (no Bedrock available)")
    spec = await staged_build(repo, router)
    return spec.model_dump(by_alias=True)