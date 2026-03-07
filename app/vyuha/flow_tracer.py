"""
Stage 3: Flow Tracing — traces the primary execution flow.
Uses call graph edges to find the longest path from entry point to terminal.
"""

import logging
from collections import defaultdict

from app.vyuha.scanner_models import ParsedRepo, ParsedEdge, EntryPoint
from app.vyuha.models import (
    DiagramNode, DiagramEdge, FlowStep, FlowTrigger,
    ErrorPath, Group,
)
from app.vyuha.node_classifier import _make_label, GROUP_COLORS

logger = logging.getLogger(__name__)


def _build_call_graph(edges: list[ParsedEdge]) -> dict[str, list[str]]:
    """Build adjacency list from call edges."""
    graph = defaultdict(list)
    for e in edges:
        if e.kind in ("calls", "depends_on"):
            graph[e.source_id].append(e.target_id)
    return dict(graph)


def _find_longest_path(graph: dict[str, list[str]], start: str, max_depth: int = 12) -> list[str]:
    """DFS to find longest path from start node."""
    best_path = [start]
    visited = set()

    def dfs(node: str, path: list[str]):
        nonlocal best_path
        if len(path) > len(best_path):
            best_path = list(path)
        if len(path) >= max_depth:
            return
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, path + [neighbor])
        visited.discard(node)

    dfs(start, [start])
    return best_path


def _pick_primary_entry(repo: ParsedRepo, call_graph: dict[str, list[str]]) -> EntryPoint | None:
    """Pick the entry point with the most downstream calls."""
    if not repo.entry_points:
        return None

    best_ep = None
    best_depth = 0

    for ep in repo.entry_points:
        path = _find_longest_path(call_graph, ep.node_id, max_depth=15)
        if len(path) > best_depth:
            best_depth = len(path)
            best_ep = ep

    return best_ep


def trace_primary_flow(
    repo: ParsedRepo,
    node_map: dict[str, DiagramNode],
) -> tuple[list[str], EntryPoint | None]:
    """
    Trace the primary flow through the codebase.
    Returns: (ordered node_ids on the path, entry_point used)
    """
    call_graph = _build_call_graph(repo.edges)
    entry = _pick_primary_entry(repo, call_graph)

    if not entry:
        logger.warning("No entry points found — cannot trace flow")
        return [], None

    path_node_ids = _find_longest_path(call_graph, entry.node_id, max_depth=12)

    logger.info("Primary flow: %s → %d steps", entry.node_id, len(path_node_ids))
    return path_node_ids, entry


def build_flow_trigger(entry: EntryPoint) -> FlowTrigger:
    """Build trigger from entry point."""
    kind_map = {
        "http_handler": "http_request",
        "grpc_handler": "grpc_call",
        "cli": "cli_command",
        "main": "function_call",
        "event_listener": "event",
        "cron": "cron",
    }
    label = ""
    if entry.method and entry.route:
        label = f"{entry.method} {entry.route}"
    elif entry.route:
        label = entry.route
    else:
        label = entry.kind

    return FlowTrigger(
        kind=kind_map.get(entry.kind, "function_call"),
        label=label[:24],
        description=entry.description or f"Flow starts at {entry.kind}",
    )


def build_flow_nodes_and_edges(
    path_ids: list[str],
    repo: ParsedRepo,
    node_map: dict[str, DiagramNode],
) -> tuple[list[DiagramNode], list[DiagramEdge], list[FlowStep]]:
    """Build flow-specific nodes, edges, and steps from the traced path."""
    # Map source node IDs to diagram nodes
    source_to_diag = {n.source_node_id: n for n in node_map.values() if n.source_node_id}

    flow_nodes = []
    flow_edges = []
    flow_steps = []

    prev_flow_id = None
    for i, source_id in enumerate(path_ids):
        diag_node = source_to_diag.get(source_id)

        if diag_node:
            flow_id = f"flow_{diag_node.id}"
            flow_node = diag_node.model_copy(update={"id": flow_id})
        else:
            # Node not in included set — create minimal node
            parsed = next((n for n in repo.nodes if n.id == source_id), None)
            if not parsed:
                continue
            flow_id = f"flow_{_make_label(parsed.name).replace(' ', '_')}"
            flow_node = DiagramNode(
                id=flow_id,
                label=_make_label(parsed.name),
                sublabel=parsed.file.split("/")[-1] if parsed.file else "",
                node_type="function",
                source_node_id=source_id,
                language=parsed.language,
            )

        flow_nodes.append(flow_node)
        flow_steps.append(FlowStep(
            step_number=i + 1,
            node_id=flow_id,
            label=flow_node.label,
            can_fail=flow_node.risk_level == "high",
            failure_reason="High complexity or error-prone" if flow_node.risk_level == "high" else "",
        ))

        if prev_flow_id:
            flow_edges.append(DiagramEdge(
                id=f"fe_{i}",
                **{"from": prev_flow_id, "to": flow_id},
                edge_type="call",
                animated=True,
                is_primary=True,
            ))
        prev_flow_id = flow_id

    return flow_nodes, flow_edges, flow_steps