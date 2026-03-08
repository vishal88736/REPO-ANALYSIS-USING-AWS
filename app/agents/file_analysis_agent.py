"""
File Analysis Agent — sends STRIPPED code to LLM (Bedrock via LLMRouter).
"""

import re
import logging

from app.utils.code_stripper import strip_for_llm
from app.schemas.analysis import (
    FileAnalysisResult,
    FileMetadata,
    ParsedStructure,
    FileInteraction,
    FunctionAnalysis,
    ClassAnalysis,
)
from app.agents.llm_router import TaskType

logger = logging.getLogger(__name__)


def _extract_file_references(content: str, all_project_files: list[str]) -> list[str]:
    refs = set()
    basenames = {}

    for f in all_project_files:
        bn = f.split("/")[-1]
        basenames[bn] = f
        basenames[f] = f

    for match in re.findall(r"""['"]([^'"]*?\.(?:js|ts|py|go|java|json|css|html))['"]\s*""", content):
        clean = match.lstrip("./")

        if clean in basenames:
            refs.add(basenames[clean])

        bn = clean.split("/")[-1]

        if bn in basenames:
            refs.add(basenames[bn])

    for match in re.findall(r"""files\s*:\s*\[(.*?)\]""", content, re.DOTALL):
        for fref in re.findall(r"""['"]([^'"]+)['"]""", match):
            clean = fref.lstrip("./")

            if clean in basenames:
                refs.add(basenames[clean])

    for match in re.findall(r"""['"]([^'"]+\.js)['"]""", content):
        clean = match.lstrip("./")

        if clean in basenames:
            refs.add(basenames[clean])

    return sorted(refs)


def _detect_interactions(file_path: str, content: str, refs: list[str], language: str):
    interactions = []

    for ref in refs:
        ref_bn = ref.split("/")[-1]

        if "executeScript" in content and ref_bn in content:
            itype = "injects"
            desc = f"{file_path} injects {ref} via chrome.scripting.executeScript"

        elif file_path.endswith("manifest.json"):
            itype = "configures"
            desc = f"manifest.json declares {ref}"

        elif "require(" in content or "import " in content:
            itype = "imports"
            desc = f"{file_path} imports {ref}"

        else:
            itype = "references"
            desc = f"{file_path} references {ref}"

        interactions.append(
            FileInteraction(
                source_file=file_path,
                target_file=ref,
                interaction_type=itype,
                description=desc,
            )
        )

    return interactions


async def analyze_file(
    router,
    file_path: str,
    content: str,
    metadata: FileMetadata,
    parsed: ParsedStructure,
    all_project_files: list[str] | None = None,
) -> FileAnalysisResult:

    if all_project_files is None:
        all_project_files = []

    # Static analysis using full content
    static_refs = _extract_file_references(content, all_project_files)

    static_interactions = _detect_interactions(
        file_path,
        content,
        static_refs,
        metadata.language,
    )

    # Strip comments + blank lines for LLM
    stripped_content, strip_stats = strip_for_llm(content, file_path)

    logger.info(
        "📦 %s: %d→%d lines (-%s%% tokens saved)",
        file_path,
        strip_stats["original_lines"],
        strip_stats["stripped_lines"],
        strip_stats["saved_percent"],
    )

    ts_funcs = [f.name for f in parsed.functions]
    ts_classes = [c.name for c in parsed.classes]
    ts_imports = [i.module for i in parsed.imports]

    prompt = f"""Analyze this {metadata.language} file.

File: {file_path}
Functions found: {', '.join(ts_funcs) if ts_funcs else 'none'}
Classes found: {', '.join(ts_classes) if ts_classes else 'none'}
Imports: {', '.join(ts_imports) if ts_imports else 'none'}
Other project files: {', '.join(all_project_files)}

Code:
{stripped_content}

Return JSON:
{{"file_path": "{file_path}", "summary": "what this file does", "functions": [{{"name": "func_name", "description": "what it does", "calls": ["functions it calls"], "imports_used": []}}], "classes": [{{"name": "class_name", "methods": ["method1"]}}], "exports": [], "external_dependencies": ["third party packages"], "internal_file_references": ["other project files used"], "file_interactions": [{{"source_file": "{file_path}", "target_file": "other.js", "interaction_type": "imports", "description": "how"}}]}}"""

    system = "You are a code analyzer. Return only a JSON object."

    try:
        result = await router.structured_chat(
            task=TaskType.FILE_ANALYSIS,
            prompt=prompt,
            system=system,
            response_model=FileAnalysisResult,
            temperature=0.1,
            max_tokens=4096,
        )

        result.file_path = file_path

        all_refs = set(static_refs)
        all_refs.update(result.internal_file_references)
        all_refs.discard(file_path)

        result.internal_file_references = sorted(all_refs)

        existing = {(i.source_file, i.target_file) for i in static_interactions}

        merged = list(static_interactions)

        for i in result.file_interactions:
            if (i.source_file, i.target_file) not in existing:
                merged.append(i)

        result.file_interactions = merged

        if parsed.functions:
            ts_names = {f.name for f in parsed.functions}

            result.functions = [f for f in result.functions if f.name in ts_names]

            llm_names = {f.name for f in result.functions}

            for ts_func in parsed.functions:
                if ts_func.name not in llm_names:
                    result.functions.append(
                        FunctionAnalysis(
                            name=ts_func.name,
                            description=f"Function at lines {ts_func.start_line}-{ts_func.end_line}",
                        )
                    )

        if parsed.classes:
            ts_names = {c.name for c in parsed.classes}
            result.classes = [c for c in result.classes if c.name in ts_names]

        logger.info(
            "✓ %s (funcs=%d, refs=%d, interactions=%d)",
            file_path,
            len(result.functions),
            len(result.internal_file_references),
            len(result.file_interactions),
        )

        return result

    except Exception as e:
        logger.error("✗ %s: %s — fallback to static analysis", file_path, e)
        return _fallback(file_path, content, parsed, static_refs, static_interactions)


def _fallback(file_path, content, parsed, refs, interactions):

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

    return FileAnalysisResult(
        file_path=file_path,
        summary=f"File with {len(functions)} functions, {len(classes)} classes.",
        functions=functions,
        classes=classes,
        exports=[f.name for f in parsed.functions],
        external_dependencies=[i.module for i in parsed.imports],
        internal_file_references=refs,
        file_interactions=interactions,
    )