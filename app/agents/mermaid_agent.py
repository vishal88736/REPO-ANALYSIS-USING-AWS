"""
Mermaid Agent — data-driven diagrams + component interaction diagram.
"""

import json
import re
import logging

from app.agents.llm_router import LLMRouter, TaskType
from app.schemas.analysis import (
    MermaidDiagram, FileAnalysisResult, FileInteraction, ArchitectureSummary,
)
from app.schemas.graph_models import DependencyGraph

logger = logging.getLogger(__name__)

MERMAID_SYSTEM_PROMPT = """\
You are a diagram specialist. Produce clean Mermaid flowchart syntax.

CRITICAL RULES:
1. Use ONLY the nodes and edges provided. Do NOT invent ANY files or functions.
2. If data is empty, return a single-node diagram saying "No data found".
3. Output ONLY raw Mermaid syntax. No markdown fences, no explanations, no notes.
4. NEVER add nodes not in the provided data.
5. NEVER explain the diagram after the syntax.\
"""


def _clean_mermaid(content: str) -> str:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:])
    if cleaned.startswith("mermaid"):
        cleaned = cleaned[7:].strip()
    if "```" in cleaned:
        cleaned = cleaned[:cleaned.index("```")].strip()

    # Remove trailing explanations
    lines = cleaned.split("\n")
    diagram_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Note") or stripped.startswith("I've") or stripped.startswith("This "):
            break
        diagram_lines.append(line)

    return "\n".join(diagram_lines).strip()


def _sanitize_id(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


# ============================================================
# FILE DEPENDENCY FLOW — built from actual file_interactions
# ============================================================

def _build_file_flow_from_interactions(
    file_analyses: list[FileAnalysisResult],
) -> str:
    """Build file flow from detected file interactions — no LLM needed."""
    actual_files = {fa.file_path for fa in file_analyses}

    # Collect all interactions
    all_interactions: list[FileInteraction] = []
    for fa in file_analyses:
        for inter in fa.file_interactions:
            all_interactions.append(inter)

    lines = ["graph TD"]

    if not all_interactions:
        # No interactions — just list files
        for fp in sorted(actual_files):
            node_id = _sanitize_id(fp)
            short = fp.split("/")[-1]
            lines.append(f"    {node_id}[\"{short}\"]")
        return "\n".join(lines)

    # Add edges with interaction labels
    seen_edges = set()
    for inter in all_interactions:
        if inter.source_file not in actual_files or inter.target_file not in actual_files:
            continue
        edge_key = f"{inter.source_file}→{inter.target_file}"
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        src_id = _sanitize_id(inter.source_file)
        tgt_id = _sanitize_id(inter.target_file)
        src_label = inter.source_file.split("/")[-1]
        tgt_label = inter.target_file.split("/")[-1]
        edge_label = inter.interaction_type

        lines.append(f"    {src_id}[\"{src_label}\"] -->|{edge_label}| {tgt_id}[\"{tgt_label}\"]")

    # Add isolated files (no interactions)
    referenced = set()
    for inter in all_interactions:
        referenced.add(inter.source_file)
        referenced.add(inter.target_file)

    for fp in sorted(actual_files - referenced):
        node_id = _sanitize_id(fp)
        short = fp.split("/")[-1]
        lines.append(f"    {node_id}[\"{short}\"]")

    return "\n".join(lines)


async def generate_file_flow_diagram(
    router: LLMRouter,
    graph: DependencyGraph,
    file_analyses: list[FileAnalysisResult],
) -> MermaidDiagram:
    """File dependency diagram built from actual file_interactions."""
    syntax = _build_file_flow_from_interactions(file_analyses)
    return MermaidDiagram(
        title="File Dependency Flow",
        diagram_type="file_flow",
        mermaid_syntax=syntax,
    )


# ============================================================
# FUNCTION CALL FLOW — built from actual function data
# ============================================================

def _build_function_flow(file_analyses: list[FileAnalysisResult]) -> str:
    lines = ["graph LR"]
    seen_edges = set()

    call_data = []
    for fa in file_analyses:
        for func in fa.functions:
            for called in func.calls:
                caller = f"{fa.file_path.split('/')[-1]}::{func.name}"
                call_data.append((caller, called))

    if not call_data:
        return 'graph LR\n    NoData["No function call relationships detected"]'

    for caller, callee in call_data[:50]:
        edge_key = f"{caller}→{callee}"
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        caller_id = _sanitize_id(caller)
        callee_id = _sanitize_id(callee)
        lines.append(f"    {caller_id}[\"{caller}\"] --> {callee_id}[\"{callee}\"]")

    return "\n".join(lines)


async def generate_function_flow_diagram(
    router: LLMRouter,
    file_analyses: list[FileAnalysisResult],
) -> MermaidDiagram:
    syntax = _build_function_flow(file_analyses)
    return MermaidDiagram(
        title="Function Call Flow",
        diagram_type="function_flow",
        mermaid_syntax=syntax,
    )


# ============================================================
# ENTRY POINT FLOW — built from actual entry points + calls
# ============================================================

async def generate_entry_point_diagram(
    router: LLMRouter,
    file_analyses: list[FileAnalysisResult],
    entry_points: list[dict],
) -> MermaidDiagram:
    if not entry_points:
        return MermaidDiagram(
            title="Entry Point Flow",
            diagram_type="entry_point_flow",
            mermaid_syntax='graph TD\n    NoEntry["No entry points detected"]',
        )

    lines = ["graph TD"]
    entry_files = {ep.get("file_path", "") for ep in entry_points}

    for ep in entry_points:
        ep_file = ep.get("file_path", "unknown")
        ep_func = ep.get("function_name", "entry")
        ep_reason = ep.get("reason", "")
        ep_id = _sanitize_id(f"{ep_file}__{ep_func}")
        ep_label = f"{ep_file.split('/')[-1]}::{ep_func}"
        lines.append(f"    {ep_id}[\"{ep_label}\"]")

        # Add calls from this entry point
        for fa in file_analyses:
            if fa.file_path == ep_file:
                for func in fa.functions:
                    if func.name == ep_func or ep_func in ("(anonymous)", "", "main"):
                        for called in func.calls:
                            called_id = _sanitize_id(called)
                            lines.append(f"    {ep_id} --> {called_id}[\"{called}\"]")

    return MermaidDiagram(
        title="Entry Point Flow",
        diagram_type="entry_point_flow",
        mermaid_syntax="\n".join(lines),
    )


# ============================================================
# NEW: COMPONENT INTERACTION DIAGRAM — high-level view
# ============================================================

async def generate_component_interaction_diagram(
    router: LLMRouter,
    architecture: ArchitectureSummary,
    file_analyses: list[FileAnalysisResult],
) -> MermaidDiagram:
    """
    High-level component interaction diagram.
    Shows runtime flow between components.
    """
    actual_files = {fa.file_path for fa in file_analyses}

    # If we have execution flow, build from it
    if architecture.execution_flow and architecture.execution_flow.steps:
        lines = ["graph TD"]

        prev_id = None
        for step in architecture.execution_flow.steps:
            step_id = _sanitize_id(f"step_{step.step_number}")
            label = f"{step.actor}: {step.action}"
            if len(label) > 50:
                label = label[:47] + "..."
            lines.append(f"    {step_id}[\"{label}\"]")

            if prev_id:
                data_label = step.data_involved if step.data_involved else ""
                if data_label:
                    lines.append(f"    {prev_id} -->|{data_label}| {step_id}")
                else:
                    lines.append(f"    {prev_id} --> {step_id}")
            prev_id = step_id

        return MermaidDiagram(
            title="Component Interaction / Execution Flow",
            diagram_type="component_interaction",
            mermaid_syntax="\n".join(lines),
        )

    # Fallback: build from file interactions
    if architecture.file_interactions:
        lines = ["graph TD"]
        for inter in architecture.file_interactions:
            src_id = _sanitize_id(inter.source_file)
            tgt_id = _sanitize_id(inter.target_file)
            src_label = inter.source_file.split("/")[-1]
            tgt_label = inter.target_file.split("/")[-1]
            lines.append(
                f"    {src_id}[\"{src_label}\"] -->|{inter.interaction_type}| {tgt_id}[\"{tgt_label}\"]"
            )
        return MermaidDiagram(
            title="Component Interaction Flow",
            diagram_type="component_interaction",
            mermaid_syntax="\n".join(lines),
        )

    return MermaidDiagram(
        title="Component Interaction Flow",
        diagram_type="component_interaction",
        mermaid_syntax='graph TD\n    NoData["No component interactions detected"]',
    )