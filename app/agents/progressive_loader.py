"""
Feature 5: Progressive Context Loading.
Feature 15: Large Context Reasoning with Gemini.

Loads context in 3 stages:
  Stage 1: Repo map (structure only, no code)
  Stage 2: Compact summaries of candidate files
  Stage 3: Raw code for only the most relevant files
"""

import logging
from pathlib import Path

from app.schemas.analysis import (
    QueryPlan, RepoMap, CompactFileSummary, FileAnalysisResult,
)
from app.services.token_utils import estimate_tokens, truncate_to_budget
from app.config import settings

logger = logging.getLogger(__name__)


def build_stage1_context(repo_map: RepoMap | None) -> str:
    """Stage 1: Repository structure only."""
    if not repo_map:
        return "No repository map available."

    lines = [
        f"Repository: {repo_map.total_files} files, ~{repo_map.total_tokens_estimate} tokens total",
        f"Languages: {', '.join(f'{k}({v})' for k, v in sorted(repo_map.languages.items(), key=lambda x: -x[1])[:10])}",
        "",
        "Directory structure:",
    ]
    for path in repo_map.directory_tree[:60]:
        lines.append(f"  {path}")

    return "\n".join(lines)


def build_stage2_context(
    plan: QueryPlan,
    compact_summaries: list[CompactFileSummary],
) -> str:
    """Stage 2: Summaries of relevant files only."""
    relevant_paths = set(plan.relevant_files)
    relevant_modules = set(plan.relevant_modules)

    selected = []
    for s in compact_summaries:
        if s.file_path in relevant_paths:
            selected.append(s)
        elif any(mod in s.file_path for mod in relevant_modules):
            selected.append(s)

    if not selected:
        # Fallback to first N summaries
        selected = compact_summaries[:10]

    lines = ["Relevant file summaries:"]
    for s in selected:
        lines.append(f"\n### {s.file_path}")
        lines.append(f"Purpose: {s.purpose}")
        if s.functions:
            lines.append(f"Functions: {', '.join(s.functions)}")
        if s.classes:
            lines.append(f"Classes: {', '.join(s.classes)}")
        if s.imports:
            lines.append(f"Imports: {', '.join(s.imports[:10])}")
        if s.key_dependencies:
            lines.append(f"Dependencies: {', '.join(s.key_dependencies)}")

    return "\n".join(lines)


def build_stage3_context(
    plan: QueryPlan,
    repo_path: Path | None,
    max_files: int = 5,
) -> str:
    """Stage 3: Raw code for the most critical files."""
    if not plan.needs_raw_code or not repo_path:
        return ""

    lines = ["Raw code for most relevant files:"]
    files_loaded = 0
    total_tokens = 0
    budget = settings.max_prompt_tokens

    for fp in plan.relevant_files[:max_files]:
        full_path = repo_path / fp
        if not full_path.exists():
            continue

        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            content_tokens = estimate_tokens(content)

            if total_tokens + content_tokens > budget:
                content = truncate_to_budget(content, budget - total_tokens)
                content_tokens = estimate_tokens(content)

            lines.append(f"\n### FILE: {fp}")
            lines.append(f"```\n{content}\n```")
            total_tokens += content_tokens
            files_loaded += 1

            if total_tokens >= budget:
                break
        except Exception as e:
            logger.warning("Failed to load %s: %s", fp, e)

    logger.info("Stage 3: Loaded %d files, ~%d tokens", files_loaded, total_tokens)
    return "\n".join(lines)