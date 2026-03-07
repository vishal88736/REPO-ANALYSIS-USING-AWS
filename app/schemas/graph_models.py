"""Pydantic models for dependency graph + advanced diagram data."""

from __future__ import annotations
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════
# EXISTING — Dependency Graph (unchanged)
# ═══════════════════════════════════════════════════════════════════

class DependencyEdge(BaseModel):
    source: str
    target: str
    relationship: str = "imports"


class DependencyNode(BaseModel):
    id: str
    type: str = "file"
    label: str = ""
    file_path: str = ""


class DependencyGraph(BaseModel):
    nodes: list[DependencyNode] = Field(default_factory=list)
    edges: list[DependencyEdge] = Field(default_factory=list)
    adjacency_list: dict[str, list[str]] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# NEW — Shared Diagram Primitives
# ═══════════════════════════════════════════════════════════════════

class DiagStyle(BaseModel):
    color: str = ""
    background: str = ""
    border_color: str = ""
    border_style: str = "solid"
    icon: str = ""
    opacity: float = 1.0
    shape: str = "rectangle"


# ═══════════════════════════════════════════════════════════════════
# NEW — Advanced Architecture Diagram
# ═══════════════════════════════════════════════════════════════════

class ArchNode(BaseModel):
    id: str
    label: str
    sublabel: str = ""
    node_type: str = "service"
    layer: str = ""
    domain: str = ""
    file_path: str = ""
    language: str = ""
    description: str = ""
    importance: str = "normal"
    risk_level: str = "low"
    is_entry_point: bool = False
    is_external: bool = False
    functions_count: int = 0
    lines_of_code: int = 0
    complexity_score: int = 0
    tags: list[str] = Field(default_factory=list)
    style: DiagStyle = Field(default_factory=DiagStyle)
    metadata: dict = Field(default_factory=dict)


class ArchEdge(BaseModel):
    id: str
    source: str
    target: str
    label: str = ""
    edge_type: str = "dependency"
    protocol: str = ""
    method: str = ""
    is_bidirectional: bool = False
    is_async: bool = False
    is_critical_path: bool = False
    data_format: str = ""
    frequency: str = ""
    style: DiagStyle = Field(default_factory=DiagStyle)
    metadata: dict = Field(default_factory=dict)


class ArchLayer(BaseModel):
    id: str
    label: str
    description: str = ""
    order: int = 0
    color: str = ""
    node_ids: list[str] = Field(default_factory=list)


class ArchCluster(BaseModel):
    id: str
    label: str
    description: str = ""
    cluster_type: str = "bounded_context"
    color: str = ""
    node_ids: list[str] = Field(default_factory=list)
    parent_cluster_id: str = ""


class ArchLegendItem(BaseModel):
    icon: str = ""
    color: str = ""
    label: str = ""
    description: str = ""


class AdvancedArchitectureDiagram(BaseModel):
    title: str = ""
    description: str = ""
    architecture_style: str = ""
    direction: str = "TB"
    nodes: list[ArchNode] = Field(default_factory=list)
    edges: list[ArchEdge] = Field(default_factory=list)
    layers: list[ArchLayer] = Field(default_factory=list)
    clusters: list[ArchCluster] = Field(default_factory=list)
    critical_path: list[str] = Field(default_factory=list)
    entry_points: list[str] = Field(default_factory=list)
    external_dependencies: list[str] = Field(default_factory=list)
    hot_spots: list[str] = Field(default_factory=list)
    total_nodes: int = 0
    total_edges: int = 0
    max_depth: int = 0
    coupling_score: float = 0.0
    legend: list[ArchLegendItem] = Field(default_factory=list)
    key_insight: str = ""
    metadata: dict = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# NEW — Network Flow Diagram
# ═══════════════════════════════════════════════════════════════════

class FlowPort(BaseModel):
    id: str
    label: str = ""
    port_type: str = "inbound"
    protocol: str = ""
    port_number: int = 0
    path: str = ""


class NetworkNode(BaseModel):
    id: str
    label: str
    sublabel: str = ""
    node_type: str = "service"
    host: str = ""
    ports: list[FlowPort] = Field(default_factory=list)
    environment: str = ""
    runtime: str = ""
    is_external: bool = False
    is_user_facing: bool = False
    health_status: str = ""
    replicas: int = 1
    description: str = ""
    style: DiagStyle = Field(default_factory=DiagStyle)
    metadata: dict = Field(default_factory=dict)


class NetworkFlow(BaseModel):
    id: str
    label: str = ""
    flow_type: str = "request_response"
    protocol: str = ""
    method: str = ""
    path: str = ""
    source: str = ""
    target: str = ""
    data_format: str = ""
    data_description: str = ""
    is_encrypted: bool = False
    is_authenticated: bool = False
    auth_type: str = ""
    avg_latency_ms: float = 0.0
    is_critical: bool = False
    can_fail: bool = False
    failure_impact: str = ""
    retry_policy: str = ""
    rate_limit: str = ""
    order: int = 0
    style: DiagStyle = Field(default_factory=DiagStyle)
    metadata: dict = Field(default_factory=dict)


class NetworkSequence(BaseModel):
    id: str
    label: str = ""
    description: str = ""
    trigger: str = ""
    outcome: str = ""
    flow_ids: list[str] = Field(default_factory=list)
    happy_path: bool = True
    error_scenario: str = ""


class NetworkZone(BaseModel):
    id: str
    label: str = ""
    zone_type: str = "internal"
    color: str = ""
    node_ids: list[str] = Field(default_factory=list)
    trust_level: str = ""


class NetworkFlowDiagram(BaseModel):
    title: str = ""
    description: str = ""
    nodes: list[NetworkNode] = Field(default_factory=list)
    flows: list[NetworkFlow] = Field(default_factory=list)
    sequences: list[NetworkSequence] = Field(default_factory=list)
    zones: list[NetworkZone] = Field(default_factory=list)
    primary_sequence_id: str = ""
    error_sequences: list[str] = Field(default_factory=list)
    total_nodes: int = 0
    total_flows: int = 0
    protocols_used: list[str] = Field(default_factory=list)
    external_endpoints: list[str] = Field(default_factory=list)
    databases: list[str] = Field(default_factory=list)
    message_queues: list[str] = Field(default_factory=list)
    key_insight: str = ""
    security_notes: list[str] = Field(default_factory=list)
    performance_notes: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)