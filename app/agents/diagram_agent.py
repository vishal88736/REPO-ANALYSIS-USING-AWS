"""
Diagram Agent — builds Advanced Architecture + Network Flow diagrams.
90% pure Python from file analysis data. 2 small LLM calls for insights.
"""

import re
import json
import logging
from collections import defaultdict

from app.agents.llm_router import LLMRouter, TaskType
from app.schemas.analysis import (
    FileAnalysisResult, ArchitectureSummary, FileInteraction,
    EntryPoint, TechnologyProfile,
)
from app.schemas.graph_models import (
    AdvancedArchitectureDiagram, ArchNode, ArchEdge, ArchLayer,
    ArchCluster, ArchLegendItem, DiagStyle,
    NetworkFlowDiagram, NetworkNode, NetworkFlow, NetworkSequence,
    NetworkZone, FlowPort,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# COLORS + STYLES
# ═══════════════════════════════════════════════════════════════════

LAYER_COLORS = {
    "presentation": "#6C8EFF",
    "api":           "#818CF8",
    "business":      "#4FFFB0",
    "data":          "#38BDF8",
    "infrastructure":"#C084FC",
    "external":      "#FF9F4A",
}

NODE_STYLES = {
    "entry_point":    DiagStyle(color="#FFFFFF", background="#6C8EFF", icon="⚡", shape="hexagon"),
    "service":        DiagStyle(color="#FFFFFF", background="#4FFFB0", icon="⚙️", shape="rectangle"),
    "function":       DiagStyle(color="#FFFFFF", background="#34D399", icon="ƒ", shape="rounded"),
    "class":          DiagStyle(color="#FFFFFF", background="#C084FC", icon="◆", shape="rectangle"),
    "database":       DiagStyle(color="#FFFFFF", background="#38BDF8", icon="🗄️", shape="cylinder"),
    "cache":          DiagStyle(color="#FFFFFF", background="#FB923C", icon="⚡", shape="cylinder"),
    "queue":          DiagStyle(color="#FFFFFF", background="#FBBF24", icon="📨", shape="parallelogram"),
    "external_api":   DiagStyle(color="#FFFFFF", background="#FF9F4A", icon="🌐", shape="cloud"),
    "cloud_service":  DiagStyle(color="#FFFFFF", background="#F97316", icon="☁️", shape="cloud"),
    "config":         DiagStyle(color="#FFFFFF", background="#94A3B8", icon="📋", shape="note"),
    "worker":         DiagStyle(color="#FFFFFF", background="#A78BFA", icon="🔄", shape="hexagon"),
    "gateway":        DiagStyle(color="#FFFFFF", background="#6366F1", icon="🚪", shape="trapezoid"),
    "storage":        DiagStyle(color="#FFFFFF", background="#22D3EE", icon="💾", shape="cylinder"),
}

EDGE_COLORS = {
    "call":           "#6C8EFF",
    "async_call":     "#C084FC",
    "data_flow":      "#4FFFB0",
    "event":          "#FBBF24",
    "dependency":     "#94A3B8",
    "http_request":   "#6C8EFF",
    "grpc_call":      "#818CF8",
    "configures":     "#94A3B8",
    "injects":        "#F472B6",
    "reads_from":     "#38BDF8",
    "writes_to":      "#EF4444",
    "publishes":      "#FBBF24",
    "subscribes":     "#FB923C",
}

ZONE_COLORS = {
    "public":   "#FEE2E2",
    "dmz":      "#FEF3C7",
    "internal": "#DBEAFE",
    "private":  "#E0E7FF",
    "external": "#FFE4E6",
}


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def _make_safe_id(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '_', text).strip('_')


def _detect_arch_node_type(fa: FileAnalysisResult, arch: ArchitectureSummary) -> str:
    fp = fa.file_path.lower()
    name = fa.file_path.split("/")[-1].lower()
    combined = f"{fp} {fa.summary}".lower()
    deps = {d.lower() for d in fa.external_dependencies}

    ep_files = {ep.file_path for ep in arch.entry_points}
    if fa.file_path in ep_files:
        return "entry_point"

    db_kw = {"database", "repository", "dao", "store", "model", "migration", "schema", "query", "orm"}
    if any(k in combined for k in db_kw) or deps & {"pg", "mysql", "mongodb", "sqlite3", "prisma", "sequelize", "sqlalchemy", "mongoose"}:
        return "database"

    if any(k in combined for k in {"cache", "redis", "memcache"}) or deps & {"redis", "ioredis", "memcached"}:
        return "cache"

    if any(k in combined for k in {"queue", "consumer", "producer", "worker", "job", "event_bus"}) or deps & {"kafka", "rabbitmq", "bull", "celery", "sqs", "amqplib"}:
        return "queue"

    if any(k in combined for k in {"client", "sdk", "api_client", "external"}) or deps & {"axios", "requests", "httpx", "got", "node-fetch"}:
        if "handler" not in combined and "route" not in combined:
            return "external_api"

    if name in {"config.py", "config.js", "config.ts", "settings.py", ".env", "config.go"}:
        return "config"
    if name.endswith((".json", ".yaml", ".yml", ".toml")) and "manifest" not in name:
        return "config"

    if any(k in combined for k in {"worker", "background", "cron", "scheduler", "job"}):
        return "worker"

    if any(k in combined for k in {"gateway", "proxy", "router", "middleware", "interceptor"}):
        return "gateway"

    if any(k in combined for k in {"handler", "controller", "route", "endpoint", "api", "view"}):
        return "service"

    if len(fa.classes) > len(fa.functions):
        return "class"

    return "service"


def _detect_layer(node_type: str, fp: str) -> str:
    layer_map = {
        "entry_point": "api", "gateway": "api", "service": "business",
        "function": "business", "class": "business", "worker": "business",
        "database": "data", "cache": "data", "queue": "infrastructure",
        "config": "infrastructure", "storage": "data",
        "external_api": "external", "cloud_service": "external",
    }
    fp_lower = fp.lower()
    if any(p in fp_lower for p in ("route", "handler", "controller", "api/", "endpoint")):
        return "api"
    if any(p in fp_lower for p in ("model", "schema", "migration", "db/", "database")):
        return "data"
    if any(p in fp_lower for p in ("config", "infra", "deploy", "docker")):
        return "infrastructure"
    return layer_map.get(node_type, "business")


def _detect_domain(fa: FileAnalysisResult) -> str:
    combined = f"{fa.file_path} {fa.summary}".lower()
    patterns = {
        "auth": r"auth|login|token|session|jwt|oauth|permission|credential|signup|register",
        "billing": r"payment|billing|charge|invoice|subscription|stripe|razorpay|checkout|price",
        "data": r"database|repository|dao|store|migration|query|model|schema",
        "api": r"handler|controller|endpoint|route|middleware|api|view|request|response",
        "infra": r"config|logger|metric|deploy|docker|ci|health|probe|util|helper|common",
        "worker": r"worker|consumer|producer|queue|job|cron|scheduler|background|task",
    }
    for domain, pattern in patterns.items():
        if re.search(pattern, combined):
            return domain
    return "core"


def _detect_importance(fa: FileAnalysisResult, arch: ArchitectureSummary) -> str:
    ep_files = {ep.file_path for ep in arch.entry_points}
    if fa.file_path in ep_files:
        return "critical"
    if len(fa.functions) > 5 or len(fa.classes) > 2:
        return "important"
    ref_count = sum(1 for i in arch.file_interactions if i.target_file == fa.file_path)
    if ref_count >= 3:
        return "important"
    return "normal"


def _detect_edge_type(interaction: FileInteraction, fas_map: dict) -> str:
    itype = interaction.interaction_type.lower()
    desc = interaction.description.lower()
    if "inject" in itype or "inject" in desc or "executescript" in desc:
        return "injects"
    if "configur" in itype or "manifest" in interaction.source_file.lower():
        return "configures"
    if "import" in itype or "require" in desc:
        return "dependency"
    if "event" in desc or "publish" in desc or "emit" in desc:
        return "event"
    if "fetch" in desc or "http" in desc or "request" in desc:
        return "http_request"
    if "grpc" in desc:
        return "grpc_call"
    if "async" in desc or "await" in desc:
        return "async_call"
    return "call"


def _detect_protocol(fa: FileAnalysisResult) -> str:
    deps = {d.lower() for d in fa.external_dependencies}
    combined = f"{fa.file_path} {fa.summary}".lower()
    if deps & {"grpc", "@grpc/grpc-js"}:
        return "grpc"
    if any(k in combined for k in {"websocket", "socket.io", "ws"}):
        return "ws"
    if deps & {"kafkajs", "kafka", "confluent-kafka"}:
        return "kafka"
    if deps & {"amqplib", "pika", "rabbitmq"}:
        return "amqp"
    if deps & {"redis", "ioredis"}:
        return "redis"
    if any(k in combined for k in {"http", "rest", "api", "fetch", "axios", "request"}):
        return "http"
    return ""


def _detect_runtime(fa: FileAnalysisResult) -> str:
    ext = fa.file_path.split(".")[-1] if "." in fa.file_path else ""
    return {"py": "python", "js": "node", "ts": "node", "go": "go", "java": "jvm", "rs": "rust", "rb": "ruby"}.get(ext, "")


def _detect_arch_style(arch: ArchitectureSummary, fas: list[FileAnalysisResult]) -> str:
    platform = arch.technology_profile.platform.lower()
    patterns = arch.design_patterns
    all_deps = set()
    for fa in fas:
        all_deps.update(d.lower() for d in fa.external_dependencies)
    if any("event" in p.lower() for p in patterns) or all_deps & {"kafka", "rabbitmq", "bull", "celery"}:
        return "event_driven"
    if len(arch.entry_points) > 3:
        return "microservices"
    if "extension" in platform:
        return "plugin"
    if any("pipe" in p.lower() for p in patterns):
        return "pipeline"
    if any("layer" in p.lower() or "mvc" in p.lower() for p in patterns):
        return "layered"
    return "monolith"


def _get_cluster(node_id: str, clusters_map: dict[str, list[str]]) -> str:
    for name, nids in clusters_map.items():
        if node_id in nids:
            return name
    return ""


# ═══════════════════════════════════════════════════════════════════
# BUILD ADVANCED ARCHITECTURE DIAGRAM
# ═══════════════════════════════════════════════════════════════════

def build_architecture_diagram(
    file_analyses: list[FileAnalysisResult],
    arch: ArchitectureSummary,
) -> AdvancedArchitectureDiagram:
    nodes: list[ArchNode] = []
    edges: list[ArchEdge] = []
    layers_map: dict[str, list[str]] = defaultdict(list)
    clusters_map: dict[str, list[str]] = defaultdict(list)
    fas_map = {fa.file_path: fa for fa in file_analyses}

    for fa in file_analyses:
        ntype = _detect_arch_node_type(fa, arch)
        layer = _detect_layer(ntype, fa.file_path)
        domain = _detect_domain(fa)
        importance = _detect_importance(fa, arch)
        node_id = f"arch_{_make_safe_id(fa.file_path)}"
        name = fa.file_path.split("/")[-1]

        nodes.append(ArchNode(
            id=node_id,
            label=name[:24],
            sublabel=fa.file_path if len(fa.file_path) <= 32 else "/".join(fa.file_path.split("/")[-2:]),
            node_type=ntype, layer=layer, domain=domain,
            file_path=fa.file_path,
            language=fa.file_path.split(".")[-1] if "." in fa.file_path else "",
            description=fa.summary[:200],
            importance=importance,
            risk_level="high" if len(fa.functions) > 10 else ("medium" if len(fa.functions) > 5 else "low"),
            is_entry_point=fa.file_path in {ep.file_path for ep in arch.entry_points},
            functions_count=len(fa.functions),
            tags=[ntype, layer, domain],
            style=NODE_STYLES.get(ntype, DiagStyle()),
        ))
        layers_map[layer].append(node_id)
        clusters_map[domain].append(node_id)

    # External deps as nodes
    all_ext_deps = set()
    dep_users: dict[str, list[str]] = defaultdict(list)
    for fa in file_analyses:
        for dep in fa.external_dependencies:
            all_ext_deps.add(dep)
            dep_users[dep].append(f"arch_{_make_safe_id(fa.file_path)}")

    for dep in sorted(all_ext_deps):
        dep_lower = dep.lower()
        ntype = "external_api"
        if any(k in dep_lower for k in ("pg", "mysql", "mongo", "sqlite", "prisma", "sequelize", "sqlalchemy")):
            ntype = "database"
        elif any(k in dep_lower for k in ("redis", "memcache")):
            ntype = "cache"
        elif any(k in dep_lower for k in ("kafka", "rabbitmq", "bull", "celery", "sqs")):
            ntype = "queue"
        elif any(k in dep_lower for k in ("aws", "gcp", "azure", "s3", "lambda")):
            ntype = "cloud_service"

        node_id = f"ext_{_make_safe_id(dep)}"
        label = dep[:24].upper() if dep == dep.lower() else dep[:24]
        nodes.append(ArchNode(
            id=node_id, label=label, node_type=ntype,
            layer="external", domain="external", is_external=True,
            description=f"External: {dep}",
            tags=[ntype, "external"],
            style=NODE_STYLES.get(ntype, NODE_STYLES["external_api"]),
        ))
        layers_map["external"].append(node_id)
        clusters_map["external"].append(node_id)

    # Edges from file interactions
    node_ids = {n.id for n in nodes}
    edge_counter = 0
    for inter in arch.file_interactions:
        src_id = f"arch_{_make_safe_id(inter.source_file)}"
        tgt_id = f"arch_{_make_safe_id(inter.target_file)}"
        if src_id not in node_ids or tgt_id not in node_ids or src_id == tgt_id:
            continue
        edge_counter += 1
        etype = _detect_edge_type(inter, fas_map)
        edges.append(ArchEdge(
            id=f"ae_{edge_counter}", source=src_id, target=tgt_id,
            label=inter.interaction_type[:20] if inter.interaction_type not in ("imports", "references") else "",
            edge_type=etype,
            is_async="async" in inter.description.lower(),
            style=DiagStyle(color=EDGE_COLORS.get(etype, "#94A3B8")),
        ))

    # Edges from ext deps
    for dep, users in dep_users.items():
        ext_id = f"ext_{_make_safe_id(dep)}"
        if ext_id not in node_ids:
            continue
        for user_id in users:
            if user_id not in node_ids:
                continue
            edge_counter += 1
            edges.append(ArchEdge(
                id=f"ae_{edge_counter}", source=user_id, target=ext_id,
                edge_type="dependency",
                style=DiagStyle(color=EDGE_COLORS["dependency"]),
            ))

    # Layers
    layer_order = {"presentation": 0, "api": 1, "business": 2, "data": 3, "infrastructure": 4, "external": 5}
    layers = [
        ArchLayer(
            id=f"layer_{name}", label=name.replace("_", " ").title(),
            order=layer_order.get(name, 99),
            color=LAYER_COLORS.get(name, "#94A3B8"),
            node_ids=nids,
        )
        for name, nids in sorted(layers_map.items(), key=lambda x: layer_order.get(x[0], 99)) if nids
    ]

    # Clusters
    cluster_labels = {
        "auth": "Authentication", "billing": "Billing & Payments",
        "data": "Data Access", "api": "API / Routes",
        "infra": "Infrastructure", "worker": "Background Workers",
        "core": "Core Business Logic", "external": "External Services",
    }
    clusters = [
        ArchCluster(
            id=f"cluster_{name}", label=cluster_labels.get(name, name.title()),
            cluster_type="bounded_context",
            color=LAYER_COLORS.get(name, "#94A3B8"),
            node_ids=nids,
        )
        for name, nids in clusters_map.items() if nids
    ]

    # Coupling
    cross = sum(1 for e in edges if _get_cluster(e.source, clusters_map) != _get_cluster(e.target, clusters_map))
    coupling = round(cross / max(len(edges), 1), 2)

    # Legend
    used_types = {n.node_type for n in nodes}
    legend = [
        ArchLegendItem(
            icon=NODE_STYLES.get(t, DiagStyle()).icon,
            color=NODE_STYLES.get(t, DiagStyle()).background,
            label=t.replace("_", " ").title(),
        )
        for t in sorted(used_types)
    ]

    style = arch.technology_profile.platform_category or ""
    direction = "LR" if any(k in style.lower() for k in ("microservice", "pipeline", "event")) else "TB"

    return AdvancedArchitectureDiagram(
        title=f"Architecture — {arch.technology_profile.platform or 'Application'}",
        description=arch.overview[:300],
        architecture_style=_detect_arch_style(arch, file_analyses),
        direction=direction, nodes=nodes, edges=edges, layers=layers, clusters=clusters,
        critical_path=[n.id for n in nodes if n.importance == "critical"],
        entry_points=[n.id for n in nodes if n.is_entry_point],
        external_dependencies=[n.id for n in nodes if n.is_external],
        hot_spots=[n.id for n in nodes if n.risk_level == "high"],
        total_nodes=len(nodes), total_edges=len(edges),
        max_depth=max((l.order for l in layers), default=0),
        coupling_score=coupling, legend=legend,
        metadata={"platform": arch.technology_profile.platform},
    )


# ═══════════════════════════════════════════════════════════════════
# BUILD NETWORK FLOW DIAGRAM
# ═══════════════════════════════════════════════════════════════════

def build_network_flow_diagram(
    file_analyses: list[FileAnalysisResult],
    arch: ArchitectureSummary,
) -> NetworkFlowDiagram:
    nodes: list[NetworkNode] = []
    flows: list[NetworkFlow] = []
    zones_map: dict[str, list[str]] = defaultdict(list)
    node_ids_set = set()

    # Client node
    client_id = "net_client"
    nodes.append(NetworkNode(
        id=client_id, label="Client", node_type="client",
        environment="browser", is_user_facing=True, is_external=True,
        description="End user / browser / API consumer",
        style=DiagStyle(color="#FFFFFF", background="#6C8EFF", icon="👤", shape="rounded"),
    ))
    node_ids_set.add(client_id)
    zones_map["public"].append(client_id)

    ep_files = {ep.file_path for ep in arch.entry_points}
    flow_counter = 0
    protocols_used = set()
    databases = []
    queues = []
    external_eps = []

    for fa in file_analyses:
        ntype = _detect_arch_node_type(fa, arch)
        protocol = _detect_protocol(fa)
        if protocol:
            protocols_used.add(protocol)

        node_id = f"net_{_make_safe_id(fa.file_path)}"
        name = fa.file_path.split("/")[-1]

        net_type_map = {
            "entry_point": "api_server", "gateway": "gateway",
            "service": "service", "worker": "worker",
            "database": "database", "cache": "cache", "queue": "queue",
            "config": "service", "external_api": "external_api",
            "cloud_service": "external_api", "class": "service", "function": "service",
        }
        net_type = net_type_map.get(ntype, "service")

        ports = []
        if fa.file_path in ep_files:
            ports.append(FlowPort(
                id=f"port_{_make_safe_id(fa.file_path)}",
                label="HTTP", port_type="inbound", protocol=protocol or "http",
            ))

        env = "server"
        summary_lower = fa.summary.lower()
        if any(k in summary_lower for k in ("browser", "chrome", "dom", "window")):
            env = "browser"
        elif any(k in summary_lower for k in ("container", "docker")):
            env = "container"

        nodes.append(NetworkNode(
            id=node_id, label=name[:24], sublabel=fa.file_path,
            node_type=net_type, ports=ports, environment=env,
            runtime=_detect_runtime(fa),
            is_user_facing=fa.file_path in ep_files,
            description=fa.summary[:200],
            style=NODE_STYLES.get(ntype, DiagStyle()),
        ))
        node_ids_set.add(node_id)

        if fa.file_path in ep_files:
            zones_map["dmz"].append(node_id)
        elif ntype in ("database", "cache", "queue"):
            zones_map["private"].append(node_id)
        elif ntype in ("external_api", "cloud_service"):
            zones_map["external"].append(node_id)
            external_eps.append(node_id)
        else:
            zones_map["internal"].append(node_id)

        if net_type == "database":
            databases.append(node_id)
        elif net_type == "queue":
            queues.append(node_id)

        # Client → entry point
        if fa.file_path in ep_files:
            flow_counter += 1
            flows.append(NetworkFlow(
                id=f"nf_{flow_counter}", label=f"Request → {name}",
                flow_type="request_response", protocol=protocol or "http",
                source=client_id, target=node_id, data_format="json",
                is_encrypted=True, is_critical=True, order=flow_counter,
                style=DiagStyle(color=EDGE_COLORS.get("http_request", "#6C8EFF")),
            ))

    # Flows from file interactions
    fas_map = {fa.file_path: fa for fa in file_analyses}
    for inter in arch.file_interactions:
        src_id = f"net_{_make_safe_id(inter.source_file)}"
        tgt_id = f"net_{_make_safe_id(inter.target_file)}"
        if src_id not in node_ids_set or tgt_id not in node_ids_set or src_id == tgt_id:
            continue
        flow_counter += 1
        etype = _detect_edge_type(inter, fas_map)
        protocol = ""
        if etype == "http_request": protocol = "http"
        elif etype == "grpc_call": protocol = "grpc"
        elif etype == "event": protocol = "kafka"
        is_async = "async" in inter.description.lower() or etype in ("event", "async_call")
        flows.append(NetworkFlow(
            id=f"nf_{flow_counter}", label=inter.interaction_type[:20],
            flow_type="event_driven" if is_async else "request_response",
            protocol=protocol, source=src_id, target=tgt_id,
            data_description=inter.description[:100],
            is_critical=etype in ("http_request", "call", "injects"),
            order=flow_counter,
            style=DiagStyle(color=EDGE_COLORS.get(etype, "#94A3B8")),
        ))

    # External dep nodes + flows
    seen_ext_edges = set()
    for dep in sorted({d for fa in file_analyses for d in fa.external_dependencies}):
        dep_id = f"net_ext_{_make_safe_id(dep)}"
        dep_lower = dep.lower()
        net_type = "external_api"
        if any(k in dep_lower for k in ("pg", "mysql", "mongo", "sqlite", "prisma")):
            net_type = "database"
            databases.append(dep_id)
        elif any(k in dep_lower for k in ("redis", "memcache")):
            net_type = "cache"
        elif any(k in dep_lower for k in ("kafka", "rabbitmq", "bull", "sqs")):
            net_type = "queue"
            queues.append(dep_id)

        nodes.append(NetworkNode(
            id=dep_id, label=dep[:24], node_type=net_type,
            is_external=True, description=f"External: {dep}",
            style=NODE_STYLES.get(net_type, NODE_STYLES["external_api"]),
        ))
        node_ids_set.add(dep_id)
        zones_map["external"].append(dep_id)
        external_eps.append(dep_id)

        for fa in file_analyses:
            if dep in fa.external_dependencies:
                src_id = f"net_{_make_safe_id(fa.file_path)}"
                edge_key = (src_id, dep_id)
                if src_id in node_ids_set and edge_key not in seen_ext_edges:
                    seen_ext_edges.add(edge_key)
                    flow_counter += 1
                    flows.append(NetworkFlow(
                        id=f"nf_{flow_counter}", label=f"→ {dep[:16]}",
                        flow_type="request_response",
                        source=src_id, target=dep_id, order=flow_counter,
                        style=DiagStyle(color=EDGE_COLORS["dependency"]),
                    ))
                    break

    # Zones
    zones = [
        NetworkZone(
            id=f"zone_{name}", label=name.replace("_", " ").title(),
            zone_type=name, color=ZONE_COLORS.get(name, "#F3F4F6"),
            node_ids=nids,
            trust_level="untrusted" if name == "public" else ("semi_trusted" if name == "dmz" else "trusted"),
        )
        for name, nids in zones_map.items() if nids
    ]

    # Primary sequence
    primary_flow_ids = [f.id for f in flows if f.is_critical][:10]
    sequences = []
    if primary_flow_ids:
        sequences.append(NetworkSequence(
            id="seq_primary", label="Primary Request Flow",
            description="Main user-facing request path",
            trigger="User request", outcome="Response returned",
            flow_ids=primary_flow_ids, happy_path=True,
        ))

    return NetworkFlowDiagram(
        title=f"Network Flow — {arch.technology_profile.platform or 'Application'}",
        description=f"Data flow across {len(nodes)} components in {len(zones)} network zones.",
        nodes=nodes, flows=flows, sequences=sequences, zones=zones,
        primary_sequence_id="seq_primary" if sequences else "",
        total_nodes=len(nodes), total_flows=len(flows),
        protocols_used=sorted(protocols_used),
        external_endpoints=external_eps, databases=databases,
        message_queues=queues,
        metadata={"platform": arch.technology_profile.platform},
    )


# ═══════════════════════════════════════════════════════════════════
# LLM ENRICHMENT (2 small calls)
# ═══════════════════════════════════════════════════════════════════

async def enrich_diagrams(
    router: LLMRouter,
    arch_diag: AdvancedArchitectureDiagram,
    net_diag: NetworkFlowDiagram,
    arch: ArchitectureSummary,
) -> tuple[AdvancedArchitectureDiagram, NetworkFlowDiagram]:

    # Call 1: Architecture insight
    arch_nodes_str = ", ".join(f"{n.label}({n.node_type})" for n in arch_diag.nodes[:15])
    prompt1 = f"""This {arch.technology_profile.platform or 'application'} has these components:
{arch_nodes_str}
Style: {arch_diag.architecture_style}
Layers: {', '.join(l.label for l in arch_diag.layers)}
Coupling: {arch_diag.coupling_score}

Return JSON:
{{"key_insight": "1 sentence about architecture", "description": "2-3 sentence overview"}}"""

    try:
        text = await router.chat(TaskType.ARCHITECTURE, prompt1, "Return only JSON.", temperature=0.2, max_tokens=512)
        first, last = text.find("{"), text.rfind("}")
        if first != -1 and last > first:
            data = json.loads(text[first:last + 1])
            arch_diag.key_insight = data.get("key_insight", "")[:200]
            if data.get("description"):
                arch_diag.description = data["description"][:300]
    except Exception as e:
        logger.warning("Arch insight failed: %s", e)

    # Call 2: Network flow insight
    protocols = ", ".join(net_diag.protocols_used) or "none detected"
    zones_str = ", ".join(f"{z.label}({len(z.node_ids)})" for z in net_diag.zones)
    prompt2 = f"""System has {net_diag.total_nodes} network components, zones: {zones_str}
Protocols: {protocols}, External: {len(net_diag.external_endpoints)}, DBs: {len(net_diag.databases)}

Return JSON:
{{"key_insight": "1 sentence about network architecture", "security_notes": ["note1"], "performance_notes": ["note1"]}}"""

    try:
        text = await router.chat(TaskType.ARCHITECTURE, prompt2, "Return only JSON.", temperature=0.2, max_tokens=512)
        first, last = text.find("{"), text.rfind("}")
        if first != -1 and last > first:
            data = json.loads(text[first:last + 1])
            net_diag.key_insight = data.get("key_insight", "")[:200]
            net_diag.security_notes = [str(n)[:100] for n in data.get("security_notes", [])][:5]
            net_diag.performance_notes = [str(n)[:100] for n in data.get("performance_notes", [])][:5]
    except Exception as e:
        logger.warning("Network insight failed: %s", e)

    return arch_diag, net_diag