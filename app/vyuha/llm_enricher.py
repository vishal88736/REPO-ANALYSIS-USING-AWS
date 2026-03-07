"""
Stage 5: LLM Enrichment — small focused calls.
"""

import json
import logging

from app.agents.llm_router import LLMRouter, TaskType
from app.vyuha.models import DiagramNode
from app.vyuha.scanner_models import ParsedRepo

logger = logging.getLogger(__name__)


async def enrich_summary(
    router: LLMRouter,
    repo: ParsedRepo,
    included_nodes: list[DiagramNode],
) -> dict:
    """LLM Call 1: one-liner, architecture style, key components."""
    file_list = sorted(list({n.file for n in repo.nodes if n.file}))[:20]
    entry_list = [f"{ep.kind}: {ep.route or ep.node_id}" for ep in repo.entry_points[:5]]
    dep_list = [d.name for d in repo.external_deps[:10]]

    prompt = f"""This repo is called "{repo.meta.repo_name}".
Languages: {', '.join(repo.meta.languages)}
Files: {', '.join(file_list)}
Entry points: {', '.join(entry_list) or 'none'}
Dependencies: {', '.join(dep_list) or 'none'}

Return JSON:
{{"one_liner": "max 12 words what this repo does", "architecture_style": "Monolith|Microservices|Event-driven|Layered MVC|Pipeline|Library / SDK|CLI Tool|Plugin-based", "key_insight": "one sentence about the most important architectural decision"}}"""

    system = "You are a software architect. Return only JSON."

    try:
        text = await router.chat(TaskType.ARCHITECTURE, prompt, system, temperature=0.2, max_tokens=1024)
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last > first:
            return json.loads(text[first:last + 1])
    except Exception as e:
        logger.warning("Summary enrichment failed: %s", e)

    return {
        "one_liner": f"{repo.meta.repo_name} — {', '.join(repo.meta.languages)} project",
        "architecture_style": "Monolith",
        "key_insight": f"Project with {repo.meta.total_files} files and {len(repo.entry_points)} entry points.",
    }


async def enrich_descriptions(
    router: LLMRouter,
    nodes: list[DiagramNode],
    repo: ParsedRepo,
) -> list[DiagramNode]:
    """LLM Call 2: short descriptions for nodes without one."""
    needs_desc = [n for n in nodes if not n.description]
    if not needs_desc:
        return nodes

    node_map = {n.id: n for n in nodes}

    for i in range(0, len(needs_desc), 10):
        batch = needs_desc[i:i + 10]
        node_list = []
        for n in batch:
            source = next((s for s in repo.nodes if s.id == n.source_node_id), None)
            snippet = source.source_snippet[:100] if source and source.source_snippet else ""
            node_list.append(f"- {n.id}: {n.label} ({n.node_type}): {snippet}")

        prompt = f"""Give a 1-sentence description for each component:
{chr(10).join(node_list)}

Return JSON: {{"descriptions": [{{"id": "node_id", "desc": "one sentence"}}]}}"""

        try:
            text = await router.chat(TaskType.ARCHITECTURE, prompt,
                                      "Return only JSON.", temperature=0.2, max_tokens=2048)
            first = text.find("{")
            last = text.rfind("}")
            if first != -1 and last > first:
                data = json.loads(text[first:last + 1])
                for item in data.get("descriptions", []):
                    nid = item.get("id", "")
                    desc = item.get("desc", "")
                    if nid in node_map and desc:
                        node_map[nid].description = desc[:100]
        except Exception as e:
            logger.warning("Description enrichment failed: %s", e)

    return list(node_map.values())


async def enrich_flow_insight(
    router: LLMRouter,
    flow_steps: list[str],
    repo: ParsedRepo,
) -> str:
    """LLM Call 3: key insight about the primary flow."""
    if not flow_steps:
        return ""

    prompt = f"""This flow goes through these steps: {' → '.join(flow_steps[:10])}
In repo "{repo.meta.repo_name}" using {', '.join(repo.meta.languages)}.

What is the single most important thing a developer should know about this flow?
Answer in 1 sentence only."""

    try:
        text = await router.chat(TaskType.ARCHITECTURE, prompt,
                                  "Answer in 1 sentence.", temperature=0.3, max_tokens=256)
        return text.strip()[:200]
    except Exception as e:
        logger.warning("Flow insight failed: %s", e)
        return ""