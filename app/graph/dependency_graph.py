"""Dependency graph builder — uses file_interactions for accurate edges."""

import logging
import os
from collections import defaultdict

from app.schemas.analysis import FileAnalysisResult
from app.schemas.graph_models import DependencyGraph, DependencyNode, DependencyEdge

logger = logging.getLogger(__name__)


def _resolve_import(import_module: str, all_paths: set[str]) -> str | None:
    potential = import_module.replace(".", "/")
    for ext in (".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".java", ".rs"):
        candidate = potential + ext
        if candidate in all_paths:
            return candidate
    init = potential + "/__init__.py"
    if init in all_paths:
        return init
    if "/" in import_module:
        for ext in (".js", ".ts", ".jsx", ".tsx"):
            candidate = import_module.lstrip("./") + ext
            if candidate in all_paths:
                return candidate
    return None


def build_dependency_graph(file_analyses: list[FileAnalysisResult]) -> DependencyGraph:
    nodes, edges = [], []
    adjacency: dict[str, list[str]] = defaultdict(list)
    all_paths = {fa.file_path for fa in file_analyses}
    file_funcs: dict[str, set[str]] = {}

    for fa in file_analyses:
        nodes.append(DependencyNode(
            id=fa.file_path, type="file",
            label=os.path.basename(fa.file_path), file_path=fa.file_path,
        ))
        funcs = set()
        for func in fa.functions:
            fid = f"{fa.file_path}::{func.name}"
            nodes.append(DependencyNode(id=fid, type="function", label=func.name, file_path=fa.file_path))
            funcs.add(func.name)
        file_funcs[fa.file_path] = funcs

        for cls in fa.classes:
            cid = f"{fa.file_path}::{cls.name}"
            nodes.append(DependencyNode(id=cid, type="class", label=cls.name, file_path=fa.file_path))

    # === NEW: Add edges from file_interactions (most accurate) ===
    seen_file_edges = set()
    for fa in file_analyses:
        for inter in fa.file_interactions:
            src = inter.source_file
            tgt = inter.target_file
            if src in all_paths and tgt in all_paths and src != tgt:
                edge_key = (src, tgt, inter.interaction_type)
                if edge_key not in seen_file_edges:
                    seen_file_edges.add(edge_key)
                    edges.append(DependencyEdge(
                        source=src, target=tgt,
                        relationship=inter.interaction_type or "imports",
                    ))
                    adjacency[src].append(tgt)

    # Also add edges from internal_file_references (backup)
    for fa in file_analyses:
        for ref in fa.internal_file_references:
            if ref in all_paths and ref != fa.file_path:
                edge_key = (fa.file_path, ref, "references")
                if edge_key not in seen_file_edges:
                    seen_file_edges.add(edge_key)
                    edges.append(DependencyEdge(
                        source=fa.file_path, target=ref, relationship="references",
                    ))
                    adjacency[fa.file_path].append(ref)

    # Legacy: external_dependencies import resolution
    for fa in file_analyses:
        for dep in fa.external_dependencies:
            target = _resolve_import(dep, all_paths)
            if target and target != fa.file_path:
                edge_key = (fa.file_path, target, "imports")
                if edge_key not in seen_file_edges:
                    seen_file_edges.add(edge_key)
                    edges.append(DependencyEdge(source=fa.file_path, target=target, relationship="imports"))
                    adjacency[fa.file_path].append(target)

    # Function-level call edges
    for fa in file_analyses:
        for func in fa.functions:
            caller_id = f"{fa.file_path}::{func.name}"
            for called in func.calls:
                for other_path, other_funcs in file_funcs.items():
                    if called in other_funcs:
                        callee_id = f"{other_path}::{called}"
                        edges.append(DependencyEdge(source=caller_id, target=callee_id, relationship="calls"))
                        adjacency[caller_id].append(callee_id)
                        break

    logger.info("Dependency graph: %d nodes, %d edges (file_interactions=%d)",
                len(nodes), len(edges), len(seen_file_edges))
    return DependencyGraph(nodes=nodes, edges=edges, adjacency_list=dict(adjacency))


def get_dependency_context_for_file(graph: DependencyGraph, file_path: str, depth: int = 2) -> str:
    visited = set()
    lines = []

    def _walk(node_id: str, d: int):
        if d > depth or node_id in visited:
            return
        visited.add(node_id)
        for neighbor in graph.adjacency_list.get(node_id, []):
            lines.append(f"{'  ' * d}{node_id} → {neighbor}")
            _walk(neighbor, d + 1)

    _walk(file_path, 0)
    return "\n".join(lines) if lines else f"No dependencies found for {file_path}"