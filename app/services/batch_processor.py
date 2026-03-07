"""
Batch Processor — skips LLM for simple files, batches multiple files per call.
"""

import logging
import asyncio
from pathlib import Path

from app.config import settings
from app.agents.file_analysis_agent import analyze_file
from app.parsers.treesitter_parser import parse_file
from app.schemas.analysis import (
    FileMetadata, FileAnalysisResult, CompactFileSummary,
    FunctionAnalysis, ClassAnalysis,
)
from app.services.cache import AnalysisCache

logger = logging.getLogger(__name__)

# Files that NEVER need LLM analysis
SKIP_LLM_EXTENSIONS = {
    ".lock", ".sum", ".mod", ".toml", ".yaml", ".yml",
    ".svg", ".png", ".jpg", ".ico", ".gif",
    ".woff", ".woff2", ".ttf", ".eot",
    ".map", ".min.js", ".min.css",
}

SKIP_LLM_FILENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "go.sum", "go.mod", "Cargo.lock",
    ".gitignore", ".eslintrc", ".prettierrc",
    "tsconfig.json", "babel.config.js", "jest.config.js",
    ".env.example", "Dockerfile", "docker-compose.yml",
    "Makefile", "LICENSE", "CHANGELOG.md",
}

# Files where tree-sitter is sufficient (config, data, docs)
TREESITTER_ONLY_EXTENSIONS = {
    ".json", ".xml", ".csv", ".env", ".ini", ".cfg",
}


def _needs_llm(metadata: FileMetadata, content: str, parsed) -> bool:
    """Decide if this file needs LLM analysis or tree-sitter is enough."""
    filename = metadata.path.split("/")[-1]
    ext = f".{metadata.extension}" if metadata.extension else ""

    # Skip entirely: binary, locks, assets
    if ext in SKIP_LLM_EXTENSIONS:
        return False
    if filename in SKIP_LLM_FILENAMES:
        return False

    # Config files: tree-sitter only
    if ext in TREESITTER_ONLY_EXTENSIONS and filename != "package.json":
        return False

    # Very small files (<10 lines): tree-sitter only
    line_count = content.count("\n") + 1
    if line_count < 10:
        return False

    # No functions/classes and <30 lines: tree-sitter only
    if not parsed.functions and not parsed.classes and line_count < 30:
        return False

    # README, docs: tree-sitter only
    if filename.upper().startswith("README") or ext == ".md":
        return False

    return True


def _build_treesitter_result(
    file_path: str,
    content: str,
    metadata: FileMetadata,
    parsed,
    all_project_files: list[str],
) -> FileAnalysisResult:
    """Build result from tree-sitter only — no LLM needed."""
    from app.agents.file_analysis_agent import (
        _extract_file_references, _detect_interactions,
    )

    refs = _extract_file_references(content, all_project_files)
    interactions = _detect_interactions(file_path, content, refs, metadata.language)

    functions = [
        FunctionAnalysis(
            name=f.name,
            description=f"Function at lines {f.start_line}-{f.end_line}",
        )
        for f in parsed.functions
    ]
    classes = [
        ClassAnalysis(name=c.name, methods=c.methods)
        for c in parsed.classes
    ]

    filename = file_path.split("/")[-1]
    line_count = content.count("\n") + 1

    return FileAnalysisResult(
        file_path=file_path,
        summary=f"{filename}: {len(functions)} functions, {len(classes)} classes, {line_count} lines.",
        functions=functions,
        classes=classes,
        exports=[f.name for f in parsed.functions],
        external_dependencies=[i.module for i in parsed.imports],
        internal_file_references=refs,
        file_interactions=interactions,
    )


def _build_compact_summary(fa: FileAnalysisResult) -> CompactFileSummary:
    return CompactFileSummary(
        file_path=fa.file_path,
        purpose=fa.summary[:200] if fa.summary else "",
        functions=[f.name for f in fa.functions],
        classes=[c.name for c in fa.classes],
        imports=[d for d in fa.external_dependencies],
        key_dependencies=fa.external_dependencies[:10],
        entry_point=any(
            f.name in ("main", "__main__", "app", "run", "start", "handler")
            for f in fa.functions
        ),
    )


async def _process_single_file(
    llm_client,
    repo_path: Path,
    metadata: FileMetadata,
    cache: AnalysisCache,
    all_project_files: list[str],
) -> tuple[FileAnalysisResult | None, CompactFileSummary | None]:
    file_path = repo_path / metadata.path

    try:
        content_bytes = file_path.read_bytes()
        content_text = content_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning("Skip %s: %s", metadata.path, e)
        return None, None

    # Check cache first
    content_hash = AnalysisCache.hash_content(content_bytes)
    cached = cache.get(content_hash)
    if cached:
        logger.info("♻ Cache: %s", metadata.path)
        summary = cache.get_summary(content_hash) or _build_compact_summary(cached)
        return cached, summary

    # Parse with tree-sitter
    parsed = parse_file(metadata.path, content_bytes, metadata.extension)

    # Decide: LLM or tree-sitter only?
    if not _needs_llm(metadata, content_text, parsed):
        result = _build_treesitter_result(
            metadata.path, content_text, metadata, parsed, all_project_files,
        )
        summary = _build_compact_summary(result)
        cache.put(content_hash, result, summary)
        logger.info("⚡ TreeSitter: %s (skipped LLM)", metadata.path)
        return result, summary

    # Full LLM analysis
    result = await analyze_file(
        groq=llm_client,
        file_path=metadata.path,
        content=content_text,
        metadata=metadata,
        parsed=parsed,
        all_project_files=all_project_files,
    )

    if result:
        summary = _build_compact_summary(result)
        cache.put(content_hash, result, summary)
        return result, summary

    return None, None


async def process_files_in_batches(
    groq,
    repo_path: Path,
    files: list[FileMetadata],
    batch_size: int | None = None,
    cache: AnalysisCache | None = None,
) -> tuple[list[FileAnalysisResult], list[CompactFileSummary]]:
    batch_size = batch_size or settings.batch_size
    if cache is None:
        cache = AnalysisCache()

    all_project_files = [f.path for f in files]

    total = len(files)
    results: list[FileAnalysisResult] = []
    summaries: list[CompactFileSummary] = []
    skipped_llm = 0
    used_llm = 0
    failed = 0

    logger.info("Processing %d files | batch=%d | concurrent=%d",
                total, batch_size, settings.max_concurrent_llm_calls)

    # Phase 1: Quick pass — tree-sitter-only files (instant, no LLM)
    llm_needed_files = []
    for metadata in files:
        file_path = repo_path / metadata.path
        try:
            content_bytes = file_path.read_bytes()
            content_text = content_bytes.decode("utf-8", errors="replace")
        except Exception:
            continue

        content_hash = AnalysisCache.hash_content(content_bytes)
        cached = cache.get(content_hash)
        if cached:
            results.append(cached)
            summaries.append(cache.get_summary(content_hash) or _build_compact_summary(cached))
            skipped_llm += 1
            continue

        parsed = parse_file(metadata.path, content_bytes, metadata.extension)

        if not _needs_llm(metadata, content_text, parsed):
            result = _build_treesitter_result(
                metadata.path, content_text, metadata, parsed, all_project_files,
            )
            summary = _build_compact_summary(result)
            cache.put(content_hash, result, summary)
            results.append(result)
            summaries.append(summary)
            skipped_llm += 1
        else:
            llm_needed_files.append(metadata)

    logger.info(
        "Phase 1 done: %d tree-sitter-only | %d need LLM | %d cached",
        skipped_llm, len(llm_needed_files), skipped_llm,
    )

    # Phase 2: LLM files — process in parallel batches
    sem = asyncio.Semaphore(settings.max_concurrent_llm_calls)

    async def _process_with_sem(metadata: FileMetadata):
        async with sem:
            return await _process_single_file(
                groq, repo_path, metadata, cache, all_project_files,
            )

    batches = [llm_needed_files[i:i + batch_size] for i in range(0, len(llm_needed_files), batch_size)]

    for idx, batch in enumerate(batches):
        batch_num = idx + 1
        logger.info("LLM Batch %d/%d (%d files)", batch_num, len(batches), len(batch))

        # Run batch in parallel
        tasks = [_process_with_sem(m) for m in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in batch_results:
            if isinstance(result, Exception):
                logger.error("File failed: %s", result)
                failed += 1
            elif result is not None:
                analysis, summary = result
                if analysis:
                    results.append(analysis)
                    used_llm += 1
                if summary:
                    summaries.append(summary)
            else:
                failed += 1

        logger.info("Batch %d done | total=%d | llm=%d | skip=%d | fail=%d",
                     batch_num, len(results), used_llm, skipped_llm, failed)

    total_refs = sum(len(r.internal_file_references) for r in results)
    total_inter = sum(len(r.file_interactions) for r in results)
    logger.info(
        "✅ Complete | %d files | llm=%d | tree-sitter=%d | failed=%d | refs=%d | interactions=%d",
        len(results), used_llm, skipped_llm, failed, total_refs, total_inter,
    )
    return results, summaries