"""
Orchestrator — ALL LLM calls go to Bedrock qwen.qwen3-coder-next.
No Groq dependency (12K TPM too low for pipeline).

qwen.qwen3-coder-next: 262K context, 8K output per call.
  - File analysis: parallel calls (10 concurrent via semaphore)
  - 5000-word summary: 8 parallel calls × ~600 words each
  - Architecture + diagrams generated inside summary
  - RAG: stripped code indexed, queries answered by Bedrock
"""

import logging
import uuid
import asyncio
from pathlib import Path

from app.agents.groq_client import GroqClient
from app.agents.llm_router import LLMRouter, TaskType
from app.services.scanner import clone_repository, scan_repository, generate_repo_map
from app.services.analysis_store import AnalysisStore
from app.services.cache import AnalysisCache
from app.rag.vector_store import VectorStore
from app.utils.code_stripper import strip_for_llm
from app.schemas.analysis import (
    FileMetadata, FileAnalysisResult, CompactFileSummary,
    FullAnalysisReport, ArchitectureSummary,
)
from app.parsers.treesitter_parser import parse_file
from app.config import settings

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# FILE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

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


def _needs_llm(metadata: FileMetadata, content: str, parsed) -> bool:
    filename = metadata.path.split("/")[-1]
    ext = f".{metadata.extension}" if metadata.extension else ""

    if ext in SKIP_LLM_EXTENSIONS:
        return False
    if filename in SKIP_LLM_FILENAMES:
        return False
    if ext in {".json", ".xml", ".csv", ".env", ".ini", ".cfg"} and filename != "package.json":
        return False

    line_count = content.count("\n") + 1

    if line_count < 10:
        return False
    if not parsed.functions and not parsed.classes and line_count < 30:
        return False
    if filename.upper().startswith("README") or ext == ".md":
        return False

    return True


def _build_compact(fa: FileAnalysisResult) -> CompactFileSummary:
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


# ═══════════════════════════════════════════════════════════════════
# PARALLEL FILE AGENT
# ═══════════════════════════════════════════════════════════════════

async def _analyze_single_file(
    groq,
    repo_path: Path,
    metadata: FileMetadata,
    cache: AnalysisCache,
    all_project_files: list[str],
    semaphore: asyncio.Semaphore,
):
    async with semaphore:

        file_path = repo_path / metadata.path

        try:
            content_bytes = file_path.read_bytes()
            content_text = content_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning("Skip %s: %s", metadata.path, e)
            return None, None, None, None

        stripped_content, strip_stats = strip_for_llm(content_text, metadata.path)

        content_hash = AnalysisCache.hash_content(content_bytes)
        cached = cache.get(content_hash)

        if cached:
            compact = cache.get_summary(content_hash) or _build_compact(cached)
            return cached, compact, metadata.path, stripped_content

        parsed = parse_file(metadata.path, content_bytes, metadata.extension)

        if not _needs_llm(metadata, content_text, parsed):

            from app.agents.file_analysis_agent import _extract_file_references, _detect_interactions
            from app.schemas.analysis import FunctionAnalysis, ClassAnalysis

            refs = _extract_file_references(content_text, all_project_files)
            interactions = _detect_interactions(
                metadata.path, content_text, refs, metadata.language
            )

            functions = [
                FunctionAnalysis(
                    name=f.name,
                    description=f"Function at lines {f.start_line}-{f.end_line}"
                )
                for f in parsed.functions
            ]

            classes = [
                ClassAnalysis(name=c.name, methods=c.methods)
                for c in parsed.classes
            ]

            filename = metadata.path.split("/")[-1]

            result = FileAnalysisResult(
                file_path=metadata.path,
                summary=f"{filename}: {len(functions)} functions, {len(classes)} classes.",
                functions=functions,
                classes=classes,
                exports=[f.name for f in parsed.functions],
                external_dependencies=[i.module for i in parsed.imports],
                internal_file_references=refs,
                file_interactions=interactions,
            )

            compact = _build_compact(result)
            cache.put(content_hash, result, compact)

            logger.info(
                "⚡ TreeSitter: %s (-%s%%)",
                metadata.path,
                strip_stats["saved_percent"]
            )

            return result, compact, metadata.path, stripped_content

        from app.agents.file_analysis_agent import analyze_file

        try:
            result = await analyze_file(
                groq=groq,
                file_path=metadata.path,
                content=content_text,
                metadata=metadata,
                parsed=parsed,
                all_project_files=all_project_files,
            )

            if result:
                compact = _build_compact(result)
                cache.put(content_hash, result, compact)

                logger.info(
                    "✓ Bedrock: %s (-%s%%)",
                    metadata.path,
                    strip_stats["saved_percent"]
                )

                return result, compact, metadata.path, stripped_content

        except Exception as e:
            logger.error("✗ %s: %s", metadata.path, e)

        return None, None, metadata.path, stripped_content


# ═══════════════════════════════════════════════════════════════════
# COMPREHENSIVE SUMMARY (WITH DIAGRAMS)
# ═══════════════════════════════════════════════════════════════════

async def generate_comprehensive_summary(
    router: LLMRouter,
    file_analyses: list[FileAnalysisResult],
    repo_url: str,
) -> str:

    all_files = []
    all_functions = []
    all_classes = []
    all_deps = set()
    all_interactions = []

    for fa in file_analyses:

        all_files.append(f"- {fa.file_path}: {fa.summary[:120]}")

        for func in fa.functions:
            calls_str = f" → {', '.join(func.calls[:3])}" if func.calls else ""
            all_functions.append(
                f"- {fa.file_path}::{func.name}: {func.description[:80]}{calls_str}"
            )

        for cls in fa.classes:
            all_classes.append(
                f"- {fa.file_path}::{cls.name}: methods={', '.join(cls.methods[:5])}"
            )

        all_deps.update(fa.external_dependencies)

        for inter in fa.file_interactions:
            all_interactions.append(
                f"- {inter.source_file} --[{inter.interaction_type}]--> {inter.target_file}"
            )

    shared_context = f"""REPOSITORY: {repo_url}

FILES ({len(all_files)}):
{chr(10).join(all_files)}

FUNCTIONS ({len(all_functions)}):
{chr(10).join(all_functions)}

CLASSES ({len(all_classes)}):
{chr(10).join(all_classes)}

DEPENDENCIES ({len(all_deps)}):
{', '.join(sorted(all_deps))}

INTERACTIONS ({len(all_interactions)}):
{chr(10).join(all_interactions)}
"""

    system = (
        "You are a senior software architect writing a crisp technical report. "
        "Be exhaustive but concise."
    )

    prompts = [
        f"""{shared_context}

Write Section 1: PROJECT OVERVIEW (~500 words)
Explain purpose, target users, and repository structure.
""",
        f"""{shared_context}

Write Section 2: ARCHITECTURE & DESIGN PATTERNS (~500 words)
Explain system architecture and module relationships.
""",
        f"""{shared_context}

Write Section 3: TECHNOLOGY STACK (~400 words)
Explain languages, frameworks, dependencies.
""",
        f"""{shared_context}

Write Section 4: ARCHITECTURE DIAGRAM

Include a Mermaid architecture diagram:

```mermaid
graph TD
User --> API
API --> Service
Service --> Database
Service --> ExternalAPI
```
""",
        f"""{shared_context}

Write Section 5: CORE FUNCTIONS & CLASSES (~700 words)
Explain key functions and classes.
""",
        f"""{shared_context}

Write Section 6: NETWORK FLOW

Include this Mermaid sequence diagram:
```mermaid
sequenceDiagram
    participant User
    participant API
    participant Service
    participant DB
    User->>API: HTTP Request
    API->>Service: process()
    Service->>DB: query()
    DB-->>Service: results
    Service-->>API: response
    API-->>User: JSON
```
""",
        f"""{shared_context}

Write Section 7: API & INTEGRATION POINTS (~500 words)
Explain external integrations.
""",
        f"""{shared_context}

Write Section 8: RISK ANALYSIS (~500 words)
Explain complexity risks and improvements.
""",
    ]

    results = await asyncio.gather(
        *[
            router.chat(
                TaskType.COMPREHENSIVE_SUMMARY,
                p,
                system,
                temperature=0.3,
                max_tokens=8000,
            )
            for p in prompts
        ],
        return_exceptions=True,
    )

    titles = [
        "## 1. Project Overview",
        "## 2. Architecture & Design Patterns",
        "## 3. Technology Stack",
        "## 4. Architecture Diagram",
        "## 5. Core Functions & Classes",
        "## 6. Network & Data Flow Diagram",
        "## 7. API & Integration Points",
        "## 8. Risk Analysis & Recommendations",
    ]

    sections = []

    for title, result in zip(titles, results):
        if isinstance(result, Exception):
            sections.append(f"{title}\n\n(Generation failed: {result})")
            logger.error("Section failed: %s — %s", title, result)
        else:
            sections.append(f"{title}\n\n{result}")

    header = (
        f"# Repository Analysis: {repo_url}\n\n"
        f"Files: {len(all_files)} | "
        f"Functions: {len(all_functions)} | "
        f"Classes: {len(all_classes)} | "
        f"Dependencies: {len(all_deps)}\n\n"
        f"---\n"
    )

    full_report = header + "\n\n---\n\n".join(sections)

    logger.info("📝 Report generated: %d words", len(full_report.split()))

    return full_report


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

async def run_analysis_pipeline(
    analysis_id: str,
    repo_url: str,
    store: AnalysisStore,
) -> None:
    """
    Full analysis pipeline:
      1. Clone repo
      2. Scan files → repo map
      3. Parallel file analysis (10 concurrent)
      4. Generate comprehensive summary
      5. Index stripped code into RAG vector store
      6. Persist report + graph + compact summaries
    """
    try:
        # ── 1. Clone ──
        await store.set_status(analysis_id, "cloning")
        logger.info("[%s] Cloning %s", analysis_id, repo_url)
        repo_path = clone_repository(repo_url)

        # ── 2. Scan ──
        await store.set_status(analysis_id, "scanning")
        files = scan_repository(repo_path)
        repo_map = generate_repo_map(repo_path, files)
        await store.save_repo_map(analysis_id, repo_map)
        logger.info("[%s] %d files found", analysis_id, len(files))

        if not files:
            raise ValueError("No analysable files found in repository.")

        # ── 3. Parallel file analysis ──
        await store.set_status(analysis_id, "analyzing")

        groq = GroqClient()
        router = LLMRouter(groq)
        cache = AnalysisCache()
        semaphore = asyncio.Semaphore(10)
        all_project_files = [f.path for f in files]

        tasks = [
            _analyze_single_file(
                groq=groq,
                repo_path=repo_path,
                metadata=meta,
                cache=cache,
                all_project_files=all_project_files,
                semaphore=semaphore,
            )
            for meta in files
        ]

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        file_analyses: list[FileAnalysisResult] = []
        compact_summaries: list[CompactFileSummary] = []
        stripped_contents: dict[str, str] = {}

        for item in raw_results:
            if isinstance(item, Exception):
                logger.error("File task raised: %s", item)
                continue
            result, compact, path, stripped = item
            if result:
                file_analyses.append(result)
            if compact:
                compact_summaries.append(compact)
            if path and stripped:
                stripped_contents[path] = stripped

        logger.info(
            "[%s] Analysed %d/%d files", analysis_id, len(file_analyses), len(files)
        )

        # ── 4. Comprehensive summary ──
        await store.set_status(analysis_id, "summarizing")
        overview = await generate_comprehensive_summary(router, file_analyses, repo_url)

        # ── 5. RAG indexing ──
        await store.set_status(analysis_id, "indexing")
        try:
            from app.api.dependencies import get_vector_store
            vector_store: VectorStore = get_vector_store()
            await vector_store.index_files(
                analysis_id=analysis_id,
                stripped_contents=stripped_contents,
                file_analyses=file_analyses,
            )
            logger.info(
                "[%s] RAG indexing complete (%d files)", analysis_id, len(stripped_contents)
            )
        except Exception as e:
            logger.warning("[%s] RAG indexing failed (non-fatal): %s", analysis_id, e)

        # ── 6. Build dependency graph ──
        graph = None
        try:
            from app.graph.dependency_graph import build_dependency_graph
            graph = build_dependency_graph(file_analyses)
        except Exception as e:
            logger.warning("[%s] Graph build failed (non-fatal): %s", analysis_id, e)

        # ── 7. Persist ──
        await store.save_compact_summaries(analysis_id, compact_summaries)

        report = FullAnalysisReport(
            analysis_id=analysis_id,
            repository_url=repo_url,
            total_files=len(files),
            file_analyses=file_analyses,
            compact_summaries=compact_summaries,
            architecture_summary=ArchitectureSummary(overview=overview),
            repo_map=repo_map,
            status="completed",
        )

        await store.save_report(report, graph)
        logger.info("[%s] ✅ Pipeline complete", analysis_id)

    except Exception as e:
        logger.exception("[%s] ❌ Pipeline failed: %s", analysis_id, e)
        await store.save_error(analysis_id, repo_url, str(e))


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def create_analysis_id() -> str:
    return str(uuid.uuid4())