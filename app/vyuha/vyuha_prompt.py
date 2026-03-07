"""
VYUHA Prompt — the full diagram-spec generation prompt.
This is sent AS-IS to Bedrock Qwen3-480B which can handle it in one shot.
For smaller models, use the staged pipeline in assembler.py instead.
"""

VYUHA_SYSTEM_PROMPT = """\
You are VYUHA's repository scanning and analysis agent.

Your ONLY job is to analyze everything the scanner found and produce
a single structured JSON output that a diagram renderer can consume
directly to generate Eraser.io-quality architecture and flow diagrams.

Output ONLY valid JSON. No markdown code blocks. No explanation.
No preamble. The first character must be { and the last must be }.
Maximum output size: 50,000 tokens.
"""


def build_vyuha_prompt(parsed_repo_json: str) -> str:
    """
    Build the full VYUHA prompt with the parsed repo data embedded.
    This is sent to Bedrock Qwen3-480B in a single call.
    """
    return f"""\
You receive a parsed repository. Analyze it and produce a diagram_spec JSON.

## PARSED REPOSITORY DATA

{parsed_repo_json}

## OUTPUT REQUIREMENTS

Produce a JSON with these exact top-level keys:
- "meta": repo metadata + diagram stats
- "architecture": architecture diagram with groups, nodes, edges
- "logical_flow": primary execution flow traced end-to-end
- "domain_map": null (unless AI enrichment is available)
- "summary": plain English overview

## NODE SELECTION RULES

SMALL REPO (< 50 nodes): show all packages + exported functions
MEDIUM REPO (50-300 nodes): show packages and services only
LARGE REPO (> 300 nodes): show top-level services only

Always include: entry points, external dependencies, high-risk nodes, primary call path
Always exclude: test files, generated code, vendor code, logging/formatting utilities, trivial functions (<3 lines)

## NODE TYPES
"entry_point" — HTTP handlers, main(), CLI commands
"service" — packages, microservices, major modules
"function" — individual functions (small repos or flow diagrams only)
"class" — classes, structs
"database" — DB, cache, store
"queue" — message queue, event bus
"external" — third-party API
"cloud" — AWS/GCP/Azure service
"interface" — Go interface, TS interface

## EDGE TYPES
"call" — synchronous function calls
"async_call" — goroutines, async/await
"data_flow" — data transformations, animated
"dependency" — imports, dotted
"implements" — interface implementations
"error_path" — failure paths, red animated

## GROUP COLORS
API Layer: "#6C8EFF", Core Services: "#4FFFB0", Domain Logic: "#C084FC"
Data Layer: "#38BDF8", External: "#FF9F4A", Workers: "#FB923C"
Shared: "#94A3B8", Auth: "#F472B6"

## ARCHITECTURE PATTERNS
LAYERED MONOLITH: TOP_TO_BOTTOM, groups = API → Service → Repository → Database
MICROSERVICES: LEFT_TO_RIGHT, groups = [Service A] [Service B] → [Shared Infra]
EVENT-DRIVEN: LEFT_TO_RIGHT, groups = Producers → Event Bus → Consumers → Stores
CLI TOOL: LEFT_TO_RIGHT, groups = Commands → Core Logic → Output
LIBRARY/SDK: TOP_TO_BOTTOM, groups = Public API → Implementation → Utilities
DATA PIPELINE: LEFT_TO_RIGHT, groups = Source → Transform → Validate → Sink

## RULES
- Every node belongs to exactly ONE group
- No group with 1 node (merge) or >8 nodes (split)
- Architecture diagram: 6-20 nodes
- Flow diagram: 5-12 nodes, ordered by step_number
- Node labels: max 24 chars, human-readable
- Edge labels: max 20 chars, or empty if obvious
- Flow diagram: always LEFT_TO_RIGHT
- Flow happy_path: ordered node ids of success path
- key_insight: genuinely insightful 1-sentence observation

## REQUIRED JSON STRUCTURE

{{
  "meta": {{
    "repo_name": "string",
    "repo_url": "string",
    "languages": ["string"],
    "generated_at": "datetime",
    "total_nodes_scanned": 0,
    "total_nodes_in_diagram": 0,
    "has_runtime_data": false
  }},
  "architecture": {{
    "title": "System Architecture — repo_name",
    "description": "2-3 sentences",
    "diagram_type": "architecture",
    "direction": "TOP_TO_BOTTOM or LEFT_TO_RIGHT",
    "groups": [{{
      "id": "grp_xxx", "label": "Group Name", "description": "purpose",
      "color": "#hex", "node_ids": ["id1"], "layer_order": 1
    }}],
    "nodes": [{{
      "id": "diag_xxx", "label": "ShortName", "sublabel": "path/file.ext",
      "node_type": "service", "source_node_id": "", "badge": "",
      "provider": "", "domain": "", "risk_level": "", "description": "",
      "is_entry_point": false, "language": ""
    }}],
    "edges": [{{
      "id": "e1", "from": "node_id", "to": "node_id", "label": "",
      "edge_type": "call", "animated": false, "is_primary": false
    }}],
    "key_insight": "1 sentence"
  }},
  "logical_flow": {{
    "title": "Flow Name",
    "description": "2-3 sentences",
    "flow_name": "slug_name",
    "diagram_type": "flow",
    "direction": "LEFT_TO_RIGHT",
    "trigger": {{"kind": "http_request", "label": "POST /path", "description": ""}},
    "steps": [{{"step_number": 1, "node_id": "flow_xxx", "label": "Step Name",
      "duration_hint": "", "can_fail": false, "failure_reason": ""}}],
    "groups": [...],
    "nodes": [...],
    "edges": [...],
    "happy_path": ["flow_id1", "flow_id2"],
    "error_paths": [{{"trigger_node_id": "", "affected_nodes": [],
      "error_type": "", "description": ""}}],
    "key_insight": "1 sentence"
  }},
  "domain_map": null,
  "summary": {{
    "one_liner": "max 12 words",
    "architecture_style": "Monolith|Microservices|Event-driven|...",
    "primary_language": "Go",
    "language_breakdown": [{{"language": "Go", "percentage": 100}}],
    "entry_points_summary": "3 HTTP endpoints, 1 CLI command",
    "key_components": [{{"name": "Name", "role": "what it does", "importance": "critical|important|supporting"}}],
    "risks": [{{"node_id": "", "node_name": "", "risk_type": "", "description": "", "severity": "high|medium|low"}}],
    "suggested_exploration": [{{"label": "", "node_id": "", "reason": ""}}]
  }}
}}
"""