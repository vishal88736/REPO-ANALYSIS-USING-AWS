"""
Feature 4: Hybrid Retrieval.
Combines: vector similarity + keyword search + dependency graph traversal.
Merges and ranks results using a scoring function.
"""

import logging
from collections import defaultdict

from app.rag.vector_store import VectorStore
from app.schemas.analysis import CompactFileSummary
from app.schemas.graph_models import DependencyGraph

logger = logging.getLogger(__name__)


def _keyword_search(
    query: str,
    compact_summaries: list[CompactFileSummary],
    top_k: int = 10,
) -> list[dict]:
    """Simple keyword matching over file names and summaries."""
    query_lower = query.lower()
    keywords = set(query_lower.split())

    scored = []
    for s in compact_summaries:
        score = 0.0

        # File path matches
        path_lower = s.file_path.lower()
        for kw in keywords:
            if kw in path_lower:
                score += 2.0

        # Purpose matches
        purpose_lower = s.purpose.lower()
        for kw in keywords:
            if kw in purpose_lower:
                score += 1.5

        # Function name matches
        for func in s.functions:
            for kw in keywords:
                if kw in func.lower():
                    score += 1.0

        # Class name matches
        for cls in s.classes:
            for kw in keywords:
                if kw in cls.lower():
                    score += 1.0

        if score > 0:
            scored.append({
                "text": f"File: {s.file_path}\nPurpose: {s.purpose}\n"
                        f"Functions: {', '.join(s.functions)}\n"
                        f"Classes: {', '.join(s.classes)}",
                "metadata": {"file_path": s.file_path, "type": "keyword_match"},
                "distance": 1.0 / (1.0 + score),  # Lower is better
                "score": score,
            })

    scored.sort(key=lambda x: -x["score"])
    return scored[:top_k]


def _graph_search(
    query: str,
    compact_summaries: list[CompactFileSummary],
    graph: DependencyGraph | None,
    seed_files: list[str],
    max_depth: int = 2,
) -> list[dict]:
    """Traverse dependency graph from seed files."""
    if not graph or not seed_files:
        return []

    visited = set()
    results = []
    summary_map = {s.file_path: s for s in compact_summaries}

    def walk(node_id: str, depth: int):
        if depth > max_depth or node_id in visited:
            return
        visited.add(node_id)

        if node_id in summary_map:
            s = summary_map[node_id]
            results.append({
                "text": f"File: {s.file_path}\nPurpose: {s.purpose}\n"
                        f"Functions: {', '.join(s.functions)}",
                "metadata": {"file_path": s.file_path, "type": "graph_neighbor", "depth": depth},
                "distance": 0.5 + depth * 0.2,
                "score": max(0, 3.0 - depth),
            })

        for neighbor in graph.adjacency_list.get(node_id, []):
            # Only follow file-level edges
            if "::" not in neighbor:
                walk(neighbor, depth + 1)

    for seed in seed_files:
        walk(seed, 0)

    return results


async def hybrid_retrieve(
    vector_store: VectorStore,
    analysis_id: str,
    query: str,
    compact_summaries: list[CompactFileSummary],
    graph: DependencyGraph | None = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Merge results from 3 retrieval methods and rank.

    Scoring:
      - Vector result: score = (1 - distance) * 3
      - Keyword result: score = keyword_score
      - Graph result: score = graph_score

    Deduplicate by file_path, keep highest score.
    """
    # 1. Vector similarity search
    vector_results = await vector_store.search(analysis_id, query, top_k=top_k)
    for r in vector_results:
        r["score"] = (1.0 - min(r.get("distance", 1.0), 1.0)) * 3.0

    # 2. Keyword search
    keyword_results = _keyword_search(query, compact_summaries, top_k=top_k)

    # 3. Graph search (seed from top vector results)
    seed_files = [
        r["metadata"].get("file_path", "")
        for r in vector_results[:3]
        if r["metadata"].get("file_path", "__global__") != "__global__"
    ]
    graph_results = _graph_search(query, compact_summaries, graph, seed_files)

    # Merge and deduplicate
    all_results = vector_results + keyword_results + graph_results
    file_best: dict[str, dict] = {}

    for r in all_results:
        fp = r.get("metadata", {}).get("file_path", "unknown")
        if fp not in file_best or r.get("score", 0) > file_best[fp].get("score", 0):
            file_best[fp] = r

    # Sort by score descending
    merged = sorted(file_best.values(), key=lambda x: -x.get("score", 0))

    logger.info(
        "Hybrid retrieval: vector=%d, keyword=%d, graph=%d → merged=%d",
        len(vector_results), len(keyword_results), len(graph_results), len(merged),
    )

    return merged[:top_k]