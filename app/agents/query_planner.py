"""
Query Planning Agent — strict version.
Only suggests files that actually exist in the repo map.
"""

import json
import logging

from app.agents.llm_router import LLMRouter, TaskType
from app.schemas.analysis import QueryPlan, RepoMap, CompactFileSummary

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """\
You are a query planning agent. Given a user question about a codebase, \
analyze the repository map and file summaries to determine which files \
are most relevant to answering the question.

CRITICAL RULES:
1. ONLY suggest files that exist in the provided file list. \
Do NOT invent file paths.
2. Use the exact file paths from the summaries — do not modify them.
3. Be conservative — suggest 3-10 files maximum.

Respond with ONLY valid JSON with these fields:
- "relevant_files": list of ACTUAL file paths from the summaries (max 10)
- "relevant_modules": list of directory names that contain relevant files
- "reasoning": brief explanation of why these files are relevant
- "needs_raw_code": boolean — true only if the question requires seeing actual code syntax\
"""


async def plan_query(
    router: LLMRouter,
    question: str,
    repo_map: RepoMap | None,
    compact_summaries: list[CompactFileSummary],
) -> QueryPlan:
    # Build context from repo map
    map_context = ""
    actual_files = set()

    if repo_map:
        map_context = f"REPOSITORY STRUCTURE ({repo_map.total_files} files):\n"
        map_context += f"Languages: {json.dumps(repo_map.languages)}\n"
        map_context += "Files:\n"
        for path in sorted(repo_map.files.keys()):
            actual_files.add(path)
            map_context += f"  {path}\n"

    # Build summary context
    summary_context = "FILE SUMMARIES (use ONLY these file paths):\n"
    for s in compact_summaries[:80]:
        actual_files.add(s.file_path)
        summary_context += (
            f"  {s.file_path}: {s.purpose} "
            f"funcs=[{', '.join(s.functions[:5])}] "
            f"deps=[{', '.join(s.key_dependencies[:3])}]\n"
        )

    prompt = f"""\
USER QUESTION: {question}

{map_context}

{summary_context}

REMINDER: Only suggest file paths that appear in the lists above.
"""

    try:
        result = await router.structured_chat(
            task=TaskType.QUERY_PLANNING,
            prompt=prompt,
            system=PLANNER_SYSTEM_PROMPT,
            response_model=QueryPlan,
            temperature=0.2,
        )

        # POST-VALIDATION: Remove files that don't exist
        if actual_files:
            validated = [f for f in result.relevant_files if f in actual_files]
            removed = len(result.relevant_files) - len(validated)
            if removed > 0:
                logger.warning("Removed %d hallucinated file paths from query plan", removed)
            result.relevant_files = validated if validated else [s.file_path for s in compact_summaries[:5]]

        return result
    except Exception as e:
        logger.error("Query planning failed: %s", e)
        return QueryPlan(
            relevant_files=[s.file_path for s in compact_summaries[:10]],
            relevant_modules=[],
            reasoning=f"Planning failed ({e}), using top files",
            needs_raw_code=False,
        )