"""
VYUHA Assembler — combines all stages into the final diagram_spec JSON.
Pure Python assembly + 3 small LLM calls.
"""

import logging
from datetime import datetime, timezone
from collections import Counter

from app.agents.llm_router import LLMRouter
from app.vyuha.scanner_models import ParsedRepo
from app.vyuha.models import (
    DiagramSpec, DiagramMeta, ArchitectureDiagram, LogicalFlowDiagram,
    DiagramSummary, LanguageBreakdown, KeyComponent, Risk, Suggestion,
    DiagramNode, DiagramEdge, Group,
)
from app.vyuha.node_classifier import classify_nodes
from app.vyuha.group_assigner import assign_groups
from app.vyuha.flow_tracer import (
    trace_primary_flow, build_flow_trigger,
    build_flow_nodes_and_edges,
)
from app.vyuha.edge_builder import build_architecture_edges
from app.vyuha.llm_enricher import enrich_summary, enrich_descriptions, enrich_flow_insight

logger = logging.getLogger(__name__)


def _detect_direction(repo: ParsedRepo, arch_style: str) -> str:
    """Determine diagram direction from architecture style."""
    lr_styles = {"Microservices", "Event-driven", "Pipeline", "CLI Tool"}
    return "LEFT_TO_RIGHT" if arch_style in lr_styles else "TOP_TO_BOTTOM"


def _build_language_breakdown(repo: ParsedRepo) -> list[LanguageBreakdown]:
    lang_count = Counter(n.language for n in repo.nodes if n.language)
    total = sum(lang_count.values()) or 1
    return [
        LanguageBreakdown(language=lang, percentage=round(count / total * 100))
        for lang, count in lang_count.most_common(5)
    ]


def _build_risks(nodes: list[DiagramNode]) -> list[Risk]:
    risks = []
    for n in nodes:
        if n.risk_level == "high":
            risks.append(Risk(
                node_id=n.id,
                node_name=n.label,
                risk_type="high_complexity",
                description=f"{n.label} has high risk: {n.description or 'complex component'}",
                severity="high",
            ))
    return risks[:10]


def _build_suggestions(nodes: list[DiagramNode], entry_ids: set[str]) -> list[Suggestion]:
    suggestions = []
    # Suggest entry points first
    for n in nodes:
        if n.is_entry_point:
            suggestions.append(Suggestion(
                label=f"Start here: {n.label}",
                node_id=n.id,
                reason=f"Entry point ({n.badge or n.node_type})",
            ))
    # Suggest high-risk nodes
    for n in nodes:
        if n.risk_level == "high" and not n.is_entry_point:
            suggestions.append(Suggestion(
                label=f"Review: {n.label}",
                node_id=n.id,
                reason="High risk component",
            ))
    return suggestions[:5]


async def build_diagram_spec(
    repo: ParsedRepo,
    router: LLMRouter,
) -> DiagramSpec:
    """
    Build the complete VYUHA diagram_spec from a parsed repository.
    Pipeline:
      Stage 1: classify_nodes (Python)
      Stage 2: assign_groups (Python)
      Stage 3: trace_primary_flow (Python)
      Stage 4: build_architecture_edges (Python)
      Stage 5: LLM enrichment (3 small calls)
      Assembly: combine everything (Python)
    """
    logger.info("VYUHA: Building diagram_spec for %s (%d nodes, %d edges)",
                repo.meta.repo_name, len(repo.nodes), len(repo.edges))

    # ── Stage 1: Node Classification ─────────────────────────────
    included_nodes, external_nodes = classify_nodes(repo)
    all_diagram_nodes = included_nodes + external_nodes
    node_map = {n.id: n for n in all_diagram_nodes}

    # ── Stage 2: Group Assignment ────────────────────────────────
    groups = assign_groups(included_nodes, external_nodes)

    # ── Stage 3: Flow Tracing ────────────────────────────────────
    flow_path_ids, flow_entry = trace_primary_flow(repo, node_map)

    # ── Stage 4: Edge Building ───────────────────────────────────
    arch_edges = build_architecture_edges(repo, included_nodes, external_nodes, flow_path_ids)

    # ── Stage 5: LLM Enrichment (3 calls) ────────────────────────
    summary_data = await enrich_summary(router, repo, included_nodes)
    all_diagram_nodes = await enrich_descriptions(router, all_diagram_nodes, repo)

    flow_step_labels = []
    if flow_path_ids:
        for fid in flow_path_ids:
            parsed = next((n for n in repo.nodes if n.id == fid), None)
            if parsed:
                flow_step_labels.append(parsed.name)
    flow_insight = await enrich_flow_insight(router, flow_step_labels, repo)

    # ── Assembly ─────────────────────────────────────────────────

    arch_style = summary_data.get("architecture_style", "Monolith")
    direction = _detect_direction(repo, arch_style)
    has_runtime = repo.runtime_summary.has_runtime_data if repo.runtime_summary else False

    # Meta
    meta = DiagramMeta(
        repo_name=repo.meta.repo_name,
        repo_url=repo.meta.repo_url,
        languages=repo.meta.languages,
        generated_at=datetime.now(timezone.utc).isoformat(),
        total_nodes_scanned=len(repo.nodes),
        total_nodes_in_diagram=len(all_diagram_nodes),
        has_runtime_data=has_runtime,
    )

    # Architecture diagram
    architecture = ArchitectureDiagram(
        title=f"System Architecture — {repo.meta.repo_name}",
        description=summary_data.get("one_liner", ""),
        diagram_type="architecture",
        direction=direction,
        groups=groups,
        nodes=all_diagram_nodes,
        edges=arch_edges,
        key_insight=summary_data.get("key_insight", ""),
    )

    # Logical flow diagram
    logical_flow = LogicalFlowDiagram(
        title="Primary Execution Flow",
        flow_name="primary_flow",
        diagram_type="flow",
        direction="LEFT_TO_RIGHT",
        key_insight=flow_insight,
    )

    if flow_entry and flow_path_ids:
        logical_flow.trigger = build_flow_trigger(flow_entry)
        flow_nodes, flow_edges, flow_steps = build_flow_nodes_and_edges(
            flow_path_ids, repo, node_map,
        )
        logical_flow.nodes = flow_nodes
        logical_flow.edges = flow_edges
        logical_flow.steps = flow_steps
        logical_flow.happy_path = [n.id for n in flow_nodes]
        logical_flow.description = (
            f"Traces the primary flow from {flow_entry.kind} "
            f"through {len(flow_steps)} steps."
        )

    # Summary
    entry_ids = {ep.node_id for ep in repo.entry_points}
    primary_lang = repo.meta.languages[0] if repo.meta.languages else "Unknown"

    entry_counts = Counter(ep.kind for ep in repo.entry_points)
    entry_summary_parts = [f"{count} {kind}" for kind, count in entry_counts.items()]

    summary = DiagramSummary(
        one_liner=summary_data.get("one_liner", f"{repo.meta.repo_name} project"),
        architecture_style=arch_style,
        primary_language=primary_lang,
        language_breakdown=_build_language_breakdown(repo),
        entry_points_summary=", ".join(entry_summary_parts) if entry_summary_parts else "No entry points detected",
        key_components=[
            KeyComponent(
                name=n.label,
                role=n.description or f"{n.node_type} in {n.sublabel}",
                importance="critical" if n.is_entry_point or n.risk_level == "high" else "supporting",
            )
            for n in all_diagram_nodes[:10]
        ],
        risks=_build_risks(all_diagram_nodes),
        suggested_exploration=_build_suggestions(all_diagram_nodes, entry_ids),
    )

    spec = DiagramSpec(
        meta=meta,
        architecture=architecture,
        logical_flow=logical_flow,
        domain_map=None,
        summary=summary,
    )

    logger.info("VYUHA: diagram_spec complete | arch_nodes=%d | arch_edges=%d | flow_steps=%d",
                len(all_diagram_nodes), len(arch_edges), len(logical_flow.steps))

    return spec