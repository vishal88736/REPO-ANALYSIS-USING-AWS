"""
Scanner input models — what tree-sitter and AST parsers produce.
These feed into the VYUHA pipeline.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


class Param(BaseModel):
    name: str = ""
    type: str = ""


class NodeRuntime(BaseModel):
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    error_rate: float = 0.0
    calls_per_min: float = 0.0
    status: str = "healthy"


class ParsedNode(BaseModel):
    id: str
    kind: str  # function, method, class, struct, interface, file, package, module
    name: str
    qualified_name: str = ""
    file: str = ""
    line_start: int = 0
    line_end: int = 0
    language: str = ""
    is_exported: bool = False
    is_async: bool = False
    signature: str = ""
    source_snippet: str = ""
    params: list[Param] = Field(default_factory=list)
    return_type: str = ""
    decorators: list[str] = Field(default_factory=list)
    complexity: int = 0
    line_count: int = 0

    # AI enrichment (filled by pipeline)
    semantic_role: str = ""
    domain: str = ""
    description: str = ""
    risk_level: str = ""
    tags: list[str] = Field(default_factory=list)

    runtime: Optional[NodeRuntime] = None


class ParsedEdge(BaseModel):
    id: str
    kind: str  # calls, imports, implements, extends, contains, depends_on
    source_id: str
    target_id: str
    line: int = 0
    is_async: bool = False
    is_resolved: bool = True
    call_count: int = 0


class Package(BaseModel):
    id: str
    name: str
    import_path: str = ""
    files: list[str] = Field(default_factory=list)
    node_ids: list[str] = Field(default_factory=list)
    language: str = ""
    is_internal: bool = True
    is_vendor: bool = False


class EntryPoint(BaseModel):
    node_id: str
    kind: str = ""  # http_handler, grpc_handler, cli, main, event_listener, cron
    route: str = ""
    method: str = ""
    description: str = ""


class ExternalDep(BaseModel):
    name: str
    import_path: str = ""
    category: str = ""
    provider: str = ""
    used_by: list[str] = Field(default_factory=list)


class RuntimeSummary(BaseModel):
    has_runtime_data: bool = False
    observation_window: str = ""
    total_events: int = 0
    error_rate: float = 0.0
    top_slow_nodes: list[str] = Field(default_factory=list)
    top_error_nodes: list[str] = Field(default_factory=list)


class RepoMeta(BaseModel):
    repo_name: str
    repo_url: str = ""
    root_path: str = ""
    languages: list[str] = Field(default_factory=list)
    total_files: int = 0
    total_nodes: int = 0
    scanned_at: str = ""


class ParsedRepo(BaseModel):
    meta: RepoMeta
    nodes: list[ParsedNode] = Field(default_factory=list)
    edges: list[ParsedEdge] = Field(default_factory=list)
    packages: list[Package] = Field(default_factory=list)
    entry_points: list[EntryPoint] = Field(default_factory=list)
    external_deps: list[ExternalDep] = Field(default_factory=list)
    runtime_summary: Optional[RuntimeSummary] = None