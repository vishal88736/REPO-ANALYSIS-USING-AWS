"""
RAG Answer Agent — strict anti-hallucination + progressive context loading.
"""

import logging
from pathlib import Path

from app.agents.llm_router import LLMRouter, TaskType
from app.agents.progressive_loader import build_stage1_context, build_stage2_context, build_stage3_context
from app.schemas.analysis import QueryPlan, RepoMap, CompactFileSummary

logger = logging.getLogger(__name__)

RAG_ANSWER_SYSTEM_PROMPT = """\
You are an expert codebase assistant. You answer questions about \
a software repository using ONLY the provided context.

CRITICAL RULES:
1. Answer ONLY based on the provided context. If the context doesn't contain \
enough information, say "Based on the available context, I cannot fully answer this" \
and explain what you DO know.
2. Reference ONLY specific file paths and function names that appear in the context. \
Do NOT invent files, functions, or code that aren't shown.
3. When referencing code, quote it from the context — do NOT write code from memory.
4. Be concise but thorough.
5. If the user asks about something not in the codebase, say so clearly.
6. Format response in clear markdown with code blocks where appropriate.
7. NEVER make up API endpoints, function signatures, or file paths.\
"""


async def answer_query(
    router: LLMRouter,
    question: str,
    retrieved_chunks: list[dict],
    dependency_context: str = "",
    architecture_summary: str = "",
    repo_map: RepoMap | None = None,
    compact_summaries: list[CompactFileSummary] | None = None,
    query_plan: QueryPlan | None = None,
    repo_path: Path | None = None,
) -> str:
    """Answer with progressive context loading and anti-hallucination."""
    context_parts = []

    # Stage 1: Repo map
    if repo_map:
        stage1 = build_stage1_context(repo_map)
        context_parts.append(f"## Repository Structure\n{stage1}\n")

    if architecture_summary:
        context_parts.append(f"## Architecture Overview\n{architecture_summary}\n")

    # Stage 2: Relevant file summaries
    if query_plan and compact_summaries:
        stage2 = build_stage2_context(query_plan, compact_summaries)
        context_parts.append(f"## Relevant Files\n{stage2}\n")

    if dependency_context:
        context_parts.append(f"## Dependency Context\n{dependency_context}\n")

    # Retrieved chunks
    context_parts.append("## Retrieved Code Sections\n")
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk.get("metadata", {})
        fp = meta.get("file_path", "unknown")
        ctype = meta.get("type", "code")
        text = chunk.get("text", "")
        context_parts.append(f"### [{i}] {fp} ({ctype})\n{text}\n")

    # Stage 3: Raw code
    if query_plan and query_plan.needs_raw_code and repo_path:
        stage3 = build_stage3_context(query_plan, repo_path)
        if stage3:
            context_parts.append(f"## Raw Code\n{stage3}\n")

    full_context = "\n".join(context_parts)

    prompt = f"""\
Using the following context from a codebase analysis, answer the user's question.

{full_context}

---
USER QUESTION: {question}
---

IMPORTANT: Only reference files, functions, and code that appear in the context above. \
If you're not sure, say so. Do NOT make up information.\
"""

    try:
        return await router.chat(TaskType.RAG_ANSWER, prompt, RAG_ANSWER_SYSTEM_PROMPT, max_tokens=2048)
    except Exception as e:
        logger.error("RAG answer failed: %s", e)
        return f"Failed to generate answer: {e}"