"""
VYUHA diagram_spec Pydantic models — matches the full spec exactly.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Shared Components ────────────────────────────────────────────

class RuntimeInfo(BaseModel):
    status: str = ""
    latency_ms: float = 0.0
    error_rate: float = 0.0


class DiagramNode(BaseModel):
    id: str
    label: str
    sublabel: str = ""
    node_type: str = "service"
    source_node_id: str = ""
    badge: str = ""
    provider: str = ""
    domain: str = ""
    risk_level: str = ""
    description: str = ""
    is_entry_point: bool = False
    language: str = ""
    runtime: Optional[RuntimeInfo] = None


class DiagramEdge(BaseModel):
    id: str
    from_node: str = Field(..., alias="from")
    to_node: str = Field(..., alias="to")
    label: str = ""
    edge_type: str = "call"
    animated: bool = False
    is_primary: bool = False

    model_config = {"populate_by_name": True}


class Group(BaseModel):
    id: str
    label: str
    description: str = ""
    color: str = "#6C8EFF"
    node_ids: list[str] = Field(default_factory=list)
    layer_order: int = 1


# ── Meta ─────────────────────────────────────────────────────────

class DiagramMeta(BaseModel):
    repo_name: str
    repo_url: str = ""
    languages: list[str] = Field(default_factory=list)
    generated_at: str = ""
    total_nodes_scanned: int = 0
    total_nodes_in_diagram: int = 0
    has_runtime_data: bool = False


# ── Architecture Diagram ─────────────────────────────────────────

class ArchitectureDiagram(BaseModel):
    title: str = ""
    description: str = ""
    diagram_type: str = "architecture"
    direction: str = "TOP_TO_BOTTOM"
    groups: list[Group] = Field(default_factory=list)
    nodes: list[DiagramNode] = Field(default_factory=list)
    edges: list[DiagramEdge] = Field(default_factory=list)
    key_insight: str = ""


# ── Logical Flow Diagram ─────────────────────────────────────────

class FlowTrigger(BaseModel):
    kind: str = ""
    label: str = ""
    description: str = ""


class FlowStep(BaseModel):
    step_number: int = 0
    node_id: str = ""
    label: str = ""
    duration_hint: str = ""
    can_fail: bool = False
    failure_reason: str = ""


class ErrorPath(BaseModel):
    trigger_node_id: str = ""
    affected_nodes: list[str] = Field(default_factory=list)
    error_type: str = ""
    description: str = ""


class LogicalFlowDiagram(BaseModel):
    title: str = ""
    description: str = ""
    flow_name: str = ""
    diagram_type: str = "flow"
    direction: str = "LEFT_TO_RIGHT"
    trigger: FlowTrigger = Field(default_factory=FlowTrigger)
    steps: list[FlowStep] = Field(default_factory=list)
    groups: list[Group] = Field(default_factory=list)
    nodes: list[DiagramNode] = Field(default_factory=list)
    edges: list[DiagramEdge] = Field(default_factory=list)
    happy_path: list[str] = Field(default_factory=list)
    error_paths: list[ErrorPath] = Field(default_factory=list)
    key_insight: str = ""


# ── Domain Map ────────────────────────────────────────────────────

class DomainGroup(BaseModel):
    domain_name: str = ""
    label: str = ""
    color: str = ""
    node_count: int = 0
    nodes: list[DiagramNode] = Field(default_factory=list)
    coupling_score: float = 0.0


class DomainMap(BaseModel):
    title: str = ""
    description: str = ""
    diagram_type: str = "architecture"
    direction: str = "TOP_TO_BOTTOM"
    domains: list[DomainGroup] = Field(default_factory=list)
    cross_domain_edges: list[DiagramEdge] = Field(default_factory=list)
    key_insight: str = ""


# ── Summary ───────────────────────────────────────────────────────

class LanguageBreakdown(BaseModel):
    language: str
    percentage: int = 0


class KeyComponent(BaseModel):
    name: str
    role: str = ""
    importance: str = "supporting"


class Risk(BaseModel):
    node_id: str = ""
    node_name: str = ""
    risk_type: str = ""
    description: str = ""
    severity: str = "low"


class Suggestion(BaseModel):
    label: str = ""
    node_id: str = ""
    reason: str = ""


class DiagramSummary(BaseModel):
    one_liner: str = ""
    architecture_style: str = ""
    primary_language: str = ""
    language_breakdown: list[LanguageBreakdown] = Field(default_factory=list)
    entry_points_summary: str = ""
    key_components: list[KeyComponent] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    suggested_exploration: list[Suggestion] = Field(default_factory=list)


# ── Full Diagram Spec ─────────────────────────────────────────────

class DiagramSpec(BaseModel):
    meta: DiagramMeta
    architecture: ArchitectureDiagram = Field(default_factory=ArchitectureDiagram)
    logical_flow: LogicalFlowDiagram = Field(default_factory=LogicalFlowDiagram)
    domain_map: Optional[DomainMap] = None
    summary: DiagramSummary = Field(default_factory=DiagramSummary)