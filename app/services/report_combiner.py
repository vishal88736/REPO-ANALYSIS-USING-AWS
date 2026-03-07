"""
Report Combiner — generates ALL diagrams in parallel.
Architecture + Network Flow + Mermaid diagrams all run simultaneously.
"""

import logging
import asyncio

from app.agents.llm_router import LLMRouter
from app.agents.architecture_agent import generate_architecture_summary
from app.agents.diagram_agent import (
    build_architecture_diagram,
    build_network_flow_diagram,
    enrich_diagrams,
)
from app.agents.mermaid_agent import (
    generate_file_flow_diagram,
    generate_function_flow_diagram,
    generate_entry_point_diagram,
    generate_component_interaction_diagram,
)
from app.graph.dependency_graph import build_dependency_graph
from app.schemas.analysis import (
    FileAnalysisResult, MermaidDiagram, FullAnalysisReport,
    CompactFileSummary, RepoMap,
)
from app.schemas.graph_models import DependencyGraph

logger = logging.getLogger(__name__)


async def combine_and_generate_report(
    groq,
    router: LLMRouter,
    analysis_id: str,
    repository_url: str,
    file_analyses: list[FileAnalysisResult],
    compact_summaries: list[CompactFileSummary] | None = None,
    repo_map: RepoMap | None = None,
) -> tuple[FullAnalysisReport, DependencyGraph]:
    logger.info("Combining %d file reports", len(file_analyses))

    # Dependency graph (pure Python, instant)
    dep_graph = build_dependency_graph(file_analyses)

    # Architecture summary (3 sequential LLM calls — needed before diagrams)
    arch_summary = await generate_architecture_summary(router, file_analyses)

    # ═══ ALL DIAGRAMS IN PARALLEL ═══
    # Pure Python diagrams (instant) + LLM mermaid diagrams (parallel)

    # Build advanced diagrams (pure Python — <1 second)
    adv_arch = build_architecture_diagram(file_analyses, arch_summary)
    net_flow = build_network_flow_diagram(file_analyses, arch_summary)

    logger.info("Built: arch=%d nodes/%d edges | network=%d nodes/%d flows",
                adv_arch.total_nodes, adv_arch.total_edges,
                net_flow.total_nodes, net_flow.total_flows)

    # Run LLM diagram enrichment + all 4 mermaid diagrams IN PARALLEL
    ep_dicts = [ep.model_dump() for ep in arch_summary.entry_points]

    all_tasks = await asyncio.gather(
        # LLM enrichment for advanced diagrams (2 small calls)
        enrich_diagrams(router, adv_arch, net_flow, arch_summary),
        # 4 mermaid diagrams (4 LLM calls)
        generate_file_flow_diagram(router, dep_graph, file_analyses),
        generate_function_flow_diagram(router, file_analyses),
        generate_entry_point_diagram(router, file_analyses, ep_dicts),
        generate_component_interaction_diagram(router, arch_summary, file_analyses),
        return_exceptions=True,
    )

    # Unpack results
    enrichment_result = all_tasks[0]
    mermaid_results = all_tasks[1:]

    if not isinstance(enrichment_result, Exception):
        adv_arch, net_flow = enrichment_result
    else:
        logger.warning("Diagram enrichment failed: %s", enrichment_result)

    diagrams = [d for d in mermaid_results if isinstance(d, MermaidDiagram)]

    report = FullAnalysisReport(
        analysis_id=analysis_id,
        repository_url=repository_url,
        total_files=len(file_analyses),
        file_analyses=file_analyses,
        compact_summaries=compact_summaries or [],
        architecture_summary=arch_summary,
        mermaid_diagrams=diagrams,
        repo_map=repo_map,
        advanced_architecture=adv_arch,
        network_flow=net_flow,
        status="completed",
    )

    logger.info(
        "Report generated for %s (%d mermaid + adv arch + network flow) — ALL PARALLEL",
        analysis_id, len(diagrams),
    )
    return report, dep_graph