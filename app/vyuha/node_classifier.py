"""
Stage 1: Node Classification — pure Python, zero LLM.
Determines which nodes to include, what type they are, and which group they belong to.
"""

import re
import logging

from app.vyuha.scanner_models import ParsedNode, ParsedRepo, EntryPoint, ExternalDep
from app.vyuha.models import DiagramNode

logger = logging.getLogger(__name__)

# Exclusion patterns
EXCLUDE_NAMES = re.compile(
    r"^(log|logger|metric|trace|format|parse|toString|toJSON|toStr|"
    r"marshal|unmarshal|serialize|deserialize|getter|setter|init__|"
    r"__repr__|__str__|__init__|setUp|tearDown|test_|Test|_test|"
    r"mock_|Mock|stub_|Stub|_pb2|_grpc|\.pb\.go)$",
    re.IGNORECASE,
)

EXCLUDE_FILE_PATTERNS = re.compile(
    r"(test|spec|mock|stub|fixture|_pb2|\.pb\.go|vendor/|node_modules/|"
    r"__pycache__|\.min\.js|\.generated\.|\.gen\.)",
    re.IGNORECASE,
)


# ── Group color palette ──────────────────────────────────────────

GROUP_COLORS = {
    "API Layer":       "#6C8EFF",
    "Core Services":   "#4FFFB0",
    "Domain Logic":    "#C084FC",
    "Data Layer":      "#38BDF8",
    "Infrastructure":  "#38BDF8",
    "External":        "#FF9F4A",
    "Workers":         "#FB923C",
    "Shared / Util":   "#94A3B8",
    "Auth / Security": "#F472B6",
}


# ── Node type detection ──────────────────────────────────────────

def _detect_node_type(node: ParsedNode, entry_ids: set[str]) -> str:
    """Map a parsed node to a diagram node_type."""
    if node.id in entry_ids:
        return "entry_point"
    if node.kind in ("interface",):
        return "interface"
    if node.kind in ("class", "struct"):
        return "class"
    if node.kind in ("function", "method"):
        return "function"
    if node.kind in ("package", "module"):
        return "service"
    return "service"


def _detect_badge(node: ParsedNode, entry_map: dict[str, EntryPoint]) -> str:
    """Detect badge for entry points."""
    ep = entry_map.get(node.id)
    if not ep:
        return ""
    kind_badges = {
        "http_handler": "REST",
        "grpc_handler": "gRPC",
        "cli": "CLI",
        "main": "Main",
        "event_listener": "Event",
        "cron": "Cron",
    }
    return kind_badges.get(ep.kind, "")


def _detect_domain(node: ParsedNode) -> str:
    """Heuristic domain detection from file path and name."""
    combined = f"{node.file} {node.name} {node.qualified_name}".lower()

    domain_patterns = {
        "auth":    r"auth|login|token|session|jwt|oauth|permission|role|credential",
        "billing": r"payment|billing|charge|invoice|subscription|stripe|razorpay|price",
        "data":    r"database|repository|dao|store|cache|redis|postgres|mysql|mongo|query|migration",
        "api":     r"handler|controller|endpoint|route|middleware|request|response|http|grpc|api",
        "infra":   r"config|logger|metric|telemetry|docker|k8s|deploy|ci|health|probe",
        "worker":  r"worker|consumer|producer|queue|job|cron|scheduler|background|async",
        "core":    r"service|processor|engine|manager|orchestrator|domain|business|logic",
    }

    for domain, pattern in domain_patterns.items():
        if re.search(pattern, combined):
            return domain

    return "core"


def _detect_risk(node: ParsedNode) -> str:
    """Heuristic risk detection."""
    if node.complexity and node.complexity > 15:
        return "high"
    if node.line_count and node.line_count > 200:
        return "high"
    if node.runtime and node.runtime.error_rate > 0.05:
        return "high"
    if node.runtime and node.runtime.status == "failing":
        return "high"
    if node.runtime and (node.runtime.status == "degraded" or node.runtime.error_rate > 0.01):
        return "medium"
    if node.complexity and node.complexity > 8:
        return "medium"
    return "low"


def _make_label(name: str, max_len: int = 24) -> str:
    """Clean label: short, readable."""
    # Remove common prefixes
    for prefix in ("internal/", "pkg/", "src/", "lib/", "app/"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    # Take last segment if path-like
    if "/" in name:
        name = name.split("/")[-1]
    if "." in name and not name.endswith((".go", ".py", ".js", ".ts")):
        name = name.split(".")[-1]
    # Truncate
    if len(name) > max_len:
        name = name[:max_len - 1] + "…"
    return name


def _make_sublabel(node: ParsedNode) -> str:
    """Short file reference."""
    if not node.file:
        return ""
    parts = node.file.replace("\\", "/").split("/")
    if len(parts) <= 2:
        return node.file
    return "/".join(parts[-2:])


# In the _should_exclude function, change the file kind check:

def _should_exclude(node: ParsedNode) -> bool:
    """Should this node be excluded from diagrams?"""
    if EXCLUDE_FILE_PATTERNS.search(node.file or ""):
        return True
    if EXCLUDE_NAMES.search(node.name):
        return True
    if node.line_count and node.line_count < 3:
        return True
    # Exclude raw "file" kind but keep "module" (which represents meaningful files)
    if node.kind == "file":
        return True
    return False

# ── External dependency → DiagramNode ────────────────────────────

def _external_dep_to_node(dep: ExternalDep) -> DiagramNode:
    """Convert an external dependency to a diagram node."""
    # Detect node type
    category_map = {
        "db": "database", "database": "database", "cache": "database",
        "queue": "queue", "messaging": "queue", "stream": "queue",
        "cloud": "cloud", "aws": "cloud", "gcp": "cloud", "azure": "cloud",
    }
    node_type = category_map.get(dep.category, "external")

    # Detect provider badge
    provider_badges = {
        "aws": "AWS", "gcp": "GCP", "azure": "Azure",
        "stripe": "Stripe", "redis": "Redis", "kafka": "Kafka",
        "postgres": "PG", "mysql": "MySQL", "mongo": "Mongo",
    }
    badge = provider_badges.get(dep.provider, "")

    label = _make_label(dep.name)
    if label == label.lower():
        label = label.upper()  # External deps in UPPERCASE

    return DiagramNode(
        id=f"ext_{dep.name.replace('-', '_').replace('.', '_')}",
        label=label,
        sublabel=dep.import_path[:32] if dep.import_path else "",
        node_type=node_type,
        source_node_id="",
        badge=badge,
        provider=dep.provider,
        domain="external",
        description=f"External dependency: {dep.name} ({dep.category})",
        is_entry_point=False,
    )


# ── Main classification function ─────────────────────────────────

def classify_nodes(repo: ParsedRepo) -> tuple[list[DiagramNode], list[DiagramNode]]:
    """
    Classify all nodes into diagram nodes.
    Returns: (included_nodes, external_nodes)

    Uses ONLY heuristics — no LLM.
    """
    total = len(repo.nodes)
    is_small = total < 50
    is_large = total > 300

    entry_ids = {ep.node_id for ep in repo.entry_points}
    entry_map = {ep.node_id: ep for ep in repo.entry_points}

    # High-priority nodes (always include)
    high_priority_ids = set(entry_ids)
    if repo.runtime_summary:
        high_priority_ids.update(repo.runtime_summary.top_slow_nodes)
        high_priority_ids.update(repo.runtime_summary.top_error_nodes)

    included = []
    for node in repo.nodes:
        # Always include high-priority
        if node.id in high_priority_ids:
            pass
        elif _should_exclude(node):
            continue
        elif is_large and node.kind in ("function", "method"):
            # Large repos: skip functions, keep packages/classes
            continue
        elif not is_small and node.kind in ("function", "method") and not node.is_exported:
            # Medium repos: skip unexported functions
            continue

        diagram_node = DiagramNode(
            id=f"diag_{node.name.replace('.', '_').replace('/', '_').replace('-', '_')}",
            label=_make_label(node.name),
            sublabel=_make_sublabel(node),
            node_type=_detect_node_type(node, entry_ids),
            source_node_id=node.id,
            badge=_detect_badge(node, entry_map),
            domain=node.domain or _detect_domain(node),
            risk_level=node.risk_level or _detect_risk(node),
            description=node.description,
            is_entry_point=node.id in entry_ids,
            language=node.language,
        )

        if node.runtime:
            diagram_node.runtime = {
                "status": node.runtime.status,
                "latency_ms": node.runtime.avg_latency_ms,
                "error_rate": node.runtime.error_rate,
            }

        included.append(diagram_node)

    # External deps
    externals = [_external_dep_to_node(dep) for dep in repo.external_deps]

    logger.info("Node classification: %d scanned → %d included + %d external",
                total, len(included), len(externals))
    return included, externals