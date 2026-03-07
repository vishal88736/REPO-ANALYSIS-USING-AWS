"""Architecture Agent — handles LLM response format mismatches."""

import json
import logging

from app.agents.llm_router import LLMRouter, TaskType
from app.schemas.analysis import (
    ArchitectureSummary, FileAnalysisResult, FileInteraction,
    ExecutionFlow, ExecutionStep, DataFlow, DataFlowStep, TechnologyProfile,
    EntryPoint,
)

logger = logging.getLogger(__name__)


def _collect_interactions(fas: list[FileAnalysisResult]) -> list[FileInteraction]:
    seen = set()
    result = []
    for fa in fas:
        for i in fa.file_interactions:
            k = (i.source_file, i.target_file, i.interaction_type)
            if k not in seen:
                seen.add(k)
                result.append(i)
    return result


def _detect_platform(fas: list[FileAnalysisResult], deps: set[str]) -> TechnologyProfile:
    p = TechnologyProfile()
    files = {fa.file_path for fa in fas}
    calls = set()
    for fa in fas:
        for f in fa.functions:
            calls.update(f.calls)

    lang_count = {}
    for fa in fas:
        ext = fa.file_path.split(".")[-1] if "." in fa.file_path else ""
        lmap = {"js": "JavaScript", "ts": "TypeScript", "py": "Python", "go": "Go", "java": "Java"}
        if ext in lmap:
            lang_count[lmap[ext]] = lang_count.get(lmap[ext], 0) + 1
    if lang_count:
        p.primary_language = max(lang_count, key=lang_count.get)

    if "manifest.json" in files:
        for fa in fas:
            if fa.file_path == "manifest.json" and "chrome" in fa.summary.lower():
                p.platform = "Chrome Extension"
                p.platform_category = "Browser Extension"
                p.runtime_environment = "Browser (Chrome)"
                p.apis_used.append("Chrome Extension API")
                break

    if any("chrome." in c for c in calls):
        if not p.platform:
            p.platform = "Chrome Extension"
            p.platform_category = "Browser Extension"
            p.runtime_environment = "Browser (Chrome)"
        for c in calls:
            if "chrome." in c and c not in p.apis_used:
                p.apis_used.append(c)

    if any("fetch" in c.lower() for c in calls):
        p.apis_used.append("Fetch API")
    if any("jspdf" in str(d).lower() for d in deps):
        p.apis_used.append("jsPDF")

    for dep, (plat, cat, rt) in {
        "fastapi": ("FastAPI", "Web API", "Python"), "flask": ("Flask", "Web App", "Python"),
        "express": ("Express.js", "Web API", "Node.js"), "react": ("React", "SPA", "Browser"),
    }.items():
        if dep in deps and not p.platform:
            p.platform, p.platform_category, p.runtime_environment = plat, cat, rt

    p.libraries = sorted(deps)[:20]
    p.summary = f"{p.platform or 'Application'} using {p.primary_language or 'unknown'}. APIs: {', '.join(p.apis_used[:5]) or 'standard'}."
    return p


def _build_context(fas: list[FileAnalysisResult]) -> str:
    lines = []
    for fa in fas:
        funcs = ", ".join(f"{f.name}({', '.join(f.calls[:3])})" for f in fa.functions)
        lines.append(
            f"FILE: {fa.file_path}\n"
            f"  Summary: {fa.summary[:200]}\n"
            f"  Functions: {funcs or 'none'}\n"
            f"  Deps: {', '.join(fa.external_dependencies[:5]) or 'none'}\n"
            f"  Refs: {', '.join(fa.internal_file_references[:5]) or 'none'}"
        )
    return "\n".join(lines)


def _normalize_key_components(raw: list) -> list[str]:
    """
    Normalize key_components to list[str].
    LLMs return mixed formats:
      - ["string1", "string2"]                          → pass through
      - [{"file": "x", "description": "y"}, ...]       → "x: y"
      - [{"name": "x", "role": "y"}, ...]              → "x: y"
      - [{"component": "x", ...}, ...]                 → "x: ..."
    """
    result = []
    for item in raw:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            # Try common key patterns
            name = (
                item.get("file") or
                item.get("name") or
                item.get("component") or
                item.get("path") or
                item.get("label") or
                ""
            )
            desc = (
                item.get("description") or
                item.get("role") or
                item.get("summary") or
                item.get("purpose") or
                ""
            )
            if name and desc:
                result.append(f"{name}: {desc}")
            elif name:
                result.append(str(name))
            elif desc:
                result.append(str(desc))
            else:
                # Last resort: join all values
                result.append(", ".join(str(v) for v in item.values() if v))
        else:
            result.append(str(item))
    return result


def _normalize_tech_stack(raw: list) -> list[str]:
    """Normalize technology_stack to list[str]."""
    result = []
    for item in raw:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            name = item.get("name") or item.get("technology") or item.get("tool") or ""
            if name:
                result.append(str(name))
            else:
                result.append(", ".join(str(v) for v in item.values() if v))
        else:
            result.append(str(item))
    return result


def _normalize_design_patterns(raw: list) -> list[str]:
    """Normalize design_patterns to list[str]."""
    result = []
    for item in raw:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            name = item.get("name") or item.get("pattern") or ""
            desc = item.get("description") or ""
            if name and desc:
                result.append(f"{name}: {desc}")
            elif name:
                result.append(str(name))
            else:
                result.append(", ".join(str(v) for v in item.values() if v))
        else:
            result.append(str(item))
    return result


def _normalize_entry_points(raw: list, actual_files: set[str]) -> list[EntryPoint]:
    """Normalize entry_points — handles both dict and string formats."""
    result = []
    for item in raw:
        if isinstance(item, dict):
            fp = item.get("file_path") or item.get("file") or item.get("path") or ""
            if fp in actual_files:
                result.append(EntryPoint(
                    file_path=fp,
                    function_name=item.get("function_name") or item.get("function") or "",
                    reason=item.get("reason") or item.get("description") or "",
                ))
        elif isinstance(item, str):
            # Could be just a file path
            if item in actual_files:
                result.append(EntryPoint(file_path=item, function_name="", reason=""))
    return result


async def _call_1_overview(router: LLMRouter, fas: list[FileAnalysisResult], deps: set, platform: TechnologyProfile) -> dict:
    ctx = _build_context(fas)
    file_list = [fa.file_path for fa in fas]

    prompt = f"""This is a {platform.platform or 'software'} project with files: {', '.join(file_list)}

{ctx}

Return JSON with:
- "overview": 2 paragraphs about what this project does
- "key_components": list of STRINGS like ["manifest.json: configures the extension", "background.js: handles events"]
- "design_patterns": list of STRINGS like ["Event-driven programming", "MVC pattern"]
- "entry_points": list of {{"file_path": "...", "function_name": "...", "reason": "..."}}
- "technology_stack": list of STRINGS like ["JavaScript", "Chrome Extension API", "jsPDF"]

IMPORTANT: key_components, design_patterns, and technology_stack must be arrays of STRINGS, not objects."""

    system = "You are a software architect. Return only a JSON object."

    try:
        text = await router.chat(TaskType.ARCHITECTURE, prompt, system, temperature=0.2, max_tokens=4096)
        if not text.strip():
            raise ValueError("Empty response")

        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last > first:
            json_str = text[first:last + 1]
            return json.loads(json_str)

        raise ValueError("No JSON in response")
    except Exception as e:
        logger.error("Overview failed: %s", e)
        return {
            "overview": f"A {platform.platform or 'software'} project with {len(fas)} files. {platform.summary}",
            "key_components": [f"{fa.file_path}: {fa.summary[:80]}" for fa in fas],
            "design_patterns": [],
            "entry_points": [],
            "technology_stack": sorted(deps)[:10],
        }


async def _call_2_execution(router: LLMRouter, fas: list[FileAnalysisResult], interactions: list[FileInteraction], platform: TechnologyProfile) -> ExecutionFlow:
    inter_str = "\n".join(f"  {i.source_file} --{i.interaction_type}--> {i.target_file}" for i in interactions)
    ctx = _build_context(fas)

    prompt = f"""This {platform.platform or 'app'} has these file interactions:
{inter_str}

{ctx}

Describe the runtime execution flow. Return JSON:
{{"trigger": "what starts it", "steps": [{{"step_number": 1, "actor": "who", "action": "what", "target": "which file", "data_involved": "what data", "description": "explain"}}], "output": "final result", "summary": "full flow description"}}"""

    system = "You are a software analyst. Return only a JSON object."

    try:
        result = await router.structured_chat(
            TaskType.ARCHITECTURE, prompt, system,
            ExecutionFlow, temperature=0.2, max_tokens=4096,
        )
        return result
    except Exception as e:
        logger.error("Execution flow failed: %s", e)
        return ExecutionFlow(summary=f"Execution flow: {e}")


async def _call_3_dataflow(router: LLMRouter, fas: list[FileAnalysisResult], platform: TechnologyProfile) -> tuple[DataFlow, str]:
    ctx = _build_context(fas)

    prompt = f"""How does data flow through this {platform.platform or 'application'}?

{ctx}

Return JSON:
{{"data_flow": {{"steps": [{{"source": "where from", "transform": "what happens", "destination": "where to", "data_type": "e.g. JSON"}}], "summary": "description"}}, "component_interaction_summary": "how components work together"}}"""

    system = "You are a software analyst. Return only a JSON object."

    try:
        text = await router.chat(TaskType.ARCHITECTURE, prompt, system, temperature=0.2, max_tokens=4096)
        if not text.strip():
            raise ValueError("Empty")

        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last > first:
            data = json.loads(text[first:last + 1])
            df_data = data.get("data_flow", {})
            df = DataFlow(
                steps=[DataFlowStep(**s) for s in df_data.get("steps", [])],
                summary=df_data.get("summary", ""),
            )
            return df, data.get("component_interaction_summary", "")

        raise ValueError("No JSON")
    except Exception as e:
        logger.error("Data flow failed: %s", e)
        return DataFlow(), ""


async def generate_architecture_summary(
    router: LLMRouter,
    file_analyses: list[FileAnalysisResult],
) -> ArchitectureSummary:
    deps = set()
    actual = set()
    for fa in file_analyses:
        actual.add(fa.file_path)
        deps.update(fa.external_dependencies)

    interactions = _collect_interactions(file_analyses)
    platform = _detect_platform(file_analyses, deps)

    logger.info("Architecture: platform=%s, files=%d, interactions=%d",
                platform.platform, len(actual), len(interactions))

    # 3 sequential calls
    overview = await _call_1_overview(router, file_analyses, deps, platform)
    exec_flow = await _call_2_execution(router, file_analyses, interactions, platform)
    data_flow, comp_summary = await _call_3_dataflow(router, file_analyses, platform)

    # ── Normalize ALL list fields ──
    # This handles LLMs returning dicts instead of strings
    key_components = _normalize_key_components(overview.get("key_components", []))
    design_patterns = _normalize_design_patterns(overview.get("design_patterns", []))
    technology_stack = _normalize_tech_stack(overview.get("technology_stack", []))
    entry_points = _normalize_entry_points(overview.get("entry_points", []), actual)

    return ArchitectureSummary(
        overview=overview.get("overview", ""),
        key_components=key_components,
        design_patterns=design_patterns,
        entry_points=entry_points,
        technology_stack=technology_stack,
        technology_profile=platform,
        file_interactions=interactions,
        execution_flow=exec_flow,
        data_flow=data_flow,
        component_interaction_summary=comp_summary,
    )