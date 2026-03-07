"""
Stage 4: Build edges for the architecture diagram.
Maps ParsedEdges to DiagramEdges, filtering to only included nodes.
"""

import logging
from collections import defaultdict

from app.vyuha.scanner_models import ParsedRepo, ParsedEdge
from app.vyuha.models import DiagramNode, DiagramEdge

logger = logging.getLogger(__name__)


def _edge_type_from_kind(kind: str, is_async: bool) -> str:
    mapping = {
        "calls": "async_call" if is_async else "call",
        "imports": "dependency",
        "implements": "implements",
        "extends": "dependency",
        "contains": "dependency",
        "depends_on": "call",
    }
    return mapping.get(kind, "call")


def build_architecture_edges(
    repo: ParsedRepo,
    included_nodes: list[DiagramNode],
    external_nodes: list[DiagramNode],
    primary_flow_ids: list[str],
) -> list[DiagramEdge]:
    """
    Build edges for the architecture diagram.
    Only includes edges where BOTH endpoints are in the diagram.
    """
    all_nodes = included_nodes + external_nodes

    # Build MULTIPLE lookups to maximize matching
    # 1. source_node_id → diagram_id (direct match)
    source_to_diag: dict[str, str] = {}
    for node in all_nodes:
        if node.source_node_id:
            source_to_diag[node.source_node_id] = node.id

    # 2. Also index by label and sublabel for fuzzy matching
    label_to_diag: dict[str, str] = {}
    for node in all_nodes:
        label_to_diag[node.label.lower()] = node.id
        if node.sublabel:
            label_to_diag[node.sublabel.lower()] = node.id

    # 3. Also index parsed node names → diagram node
    #    This handles cases where edge source_id is "file:background.js"
    #    but diagram source_node_id is different
    parsed_name_to_diag: dict[str, str] = {}
    for parsed_node in repo.nodes:
        if parsed_node.id in source_to_diag:
            diag_id = source_to_diag[parsed_node.id]
            # Index by name too
            parsed_name_to_diag[parsed_node.name.lower()] = diag_id
            if parsed_node.file:
                parsed_name_to_diag[parsed_node.file.lower()] = diag_id
                # Also the file basename
                basename = parsed_node.file.split("/")[-1].lower()
                parsed_name_to_diag[basename] = diag_id

    def _resolve(node_id: str) -> str | None:
        """Try to resolve a parsed edge node_id to a diagram node id."""
        # Direct match
        if node_id in source_to_diag:
            return source_to_diag[node_id]

        # Strip prefix and try (e.g. "file:background.js" → "background.js")
        if ":" in node_id:
            stripped = node_id.split(":", 1)[1]
            if stripped in source_to_diag:
                return source_to_diag[stripped]
            # Try lowercase
            if stripped.lower() in parsed_name_to_diag:
                return parsed_name_to_diag[stripped.lower()]
            # Try basename
            basename = stripped.split("/")[-1].lower()
            if basename in parsed_name_to_diag:
                return parsed_name_to_diag[basename]

        # Try the raw id as lowercase
        if node_id.lower() in parsed_name_to_diag:
            return parsed_name_to_diag[node_id.lower()]

        # Try label match
        if node_id.lower() in label_to_diag:
            return label_to_diag[node_id.lower()]

        return None

    edges = []
    seen = set()
    primary_set = set(primary_flow_ids)
    edge_counter = 0
    unresolved_src = 0
    unresolved_tgt = 0

    for parsed_edge in repo.edges:
        src_diag = _resolve(parsed_edge.source_id)
        tgt_diag = _resolve(parsed_edge.target_id)

        if not src_diag:
            unresolved_src += 1
            continue
        if not tgt_diag:
            unresolved_tgt += 1
            continue
        if src_diag == tgt_diag:
            continue

        edge_key = (src_diag, tgt_diag)
        if edge_key in seen:
            continue
        seen.add(edge_key)

        edge_counter += 1
        is_on_primary = (
            parsed_edge.source_id in primary_set and
            parsed_edge.target_id in primary_set
        )

        edges.append(DiagramEdge(
            id=f"e_{edge_counter}",
            **{"from": src_diag, "to": tgt_diag},
            label="",
            edge_type=_edge_type_from_kind(parsed_edge.kind, parsed_edge.is_async),
            animated=is_on_primary,
            is_primary=is_on_primary,
        ))

    # Also build edges from file_interactions in the original analysis
    # These are the most reliable edges (statically detected)
    for node in repo.nodes:
        pass  # Already in parsed edges

    # Build edges from external deps used_by
    ext_name_to_diag = {}
    for dep in repo.external_deps:
        for ext_node in external_nodes:
            if dep.name.replace("-", "_").replace(".", "_") in ext_node.id:
                ext_name_to_diag[dep.name] = ext_node.id
                break

    for dep in repo.external_deps:
        ext_diag = ext_name_to_diag.get(dep.name)
        if not ext_diag:
            continue
        for user_id in dep.used_by:
            user_diag = _resolve(user_id)
            if not user_diag:
                continue
            edge_key = (user_diag, ext_diag)
            if edge_key in seen:
                continue
            seen.add(edge_key)
            edge_counter += 1
            edges.append(DiagramEdge(
                id=f"e_{edge_counter}",
                **{"from": user_diag, "to": ext_diag},
                label="",
                edge_type="call",
                animated=False,
                is_primary=False,
            ))

    logger.info(
        "Architecture edges: %d built | %d parsed | unresolved: src=%d tgt=%d | diagram_nodes=%d",
        len(edges), len(repo.edges), unresolved_src, unresolved_tgt, len(all_nodes),
    )

    # Debug: log the lookup tables if 0 edges
    if len(edges) == 0 and len(repo.edges) > 0:
        logger.warning("0 edges built! Debugging lookups:")
        logger.warning("  source_to_diag keys (first 10): %s", list(source_to_diag.keys())[:10])
        logger.warning("  parsed_name_to_diag keys (first 10): %s", list(parsed_name_to_diag.keys())[:10])
        sample_edges = repo.edges[:5]
        for e in sample_edges:
            logger.warning("  Sample edge: %s → %s (resolved: %s → %s)",
                           e.source_id, e.target_id, _resolve(e.source_id), _resolve(e.target_id))

    return edges