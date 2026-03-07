"""Pydantic models for file analysis, agent outputs, and reports."""

from __future__ import annotations
from pydantic import BaseModel, Field


class FileMetadata(BaseModel):
    path: str = Field(..., description="Relative file path from repo root")
    size_bytes: int = Field(..., ge=0)
    language: str = Field(default="unknown")
    extension: str = Field(default="")


class ParsedFunction(BaseModel):
    name: str
    start_line: int
    end_line: int
    parameters: list[str] = Field(default_factory=list)


class ParsedClass(BaseModel):
    name: str
    start_line: int
    end_line: int
    methods: list[str] = Field(default_factory=list)


class ParsedImport(BaseModel):
    module: str
    names: list[str] = Field(default_factory=list)


class ParsedStructure(BaseModel):
    functions: list[ParsedFunction] = Field(default_factory=list)
    classes: list[ParsedClass] = Field(default_factory=list)
    imports: list[ParsedImport] = Field(default_factory=list)


class FunctionAnalysis(BaseModel):
    name: str
    description: str = ""
    calls: list[str] = Field(default_factory=list)
    imports_used: list[str] = Field(default_factory=list)


class ClassAnalysis(BaseModel):
    name: str
    methods: list[str] = Field(default_factory=list)


# === Enhanced: File-level interaction tracking ===

class FileInteraction(BaseModel):
    """Describes how one file uses/references another."""
    source_file: str
    target_file: str
    interaction_type: str = ""  # "injects", "imports", "loads", "calls_into", "configures"
    description: str = ""


class FileAnalysisResult(BaseModel):
    file_path: str
    summary: str = ""
    functions: list[FunctionAnalysis] = Field(default_factory=list)
    classes: list[ClassAnalysis] = Field(default_factory=list)
    exports: list[str] = Field(default_factory=list)
    external_dependencies: list[str] = Field(default_factory=list)
    # NEW: which other project files does this file reference?
    internal_file_references: list[str] = Field(default_factory=list)
    # NEW: how does this file interact with others?
    file_interactions: list[FileInteraction] = Field(default_factory=list)


# === Feature 2: Compact File Summary ===

class CompactFileSummary(BaseModel):
    file_path: str
    purpose: str = ""
    functions: list[str] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)
    key_dependencies: list[str] = Field(default_factory=list)
    entry_point: bool = False


# === Feature 1: Repository Map ===

class RepoMapEntry(BaseModel):
    language: str = "unknown"
    size_bytes: int = 0
    tokens_estimate: int = 0
    extension: str = ""
    directory: str = ""


class RepoMap(BaseModel):
    files: dict[str, RepoMapEntry] = Field(default_factory=dict)
    total_files: int = 0
    total_tokens_estimate: int = 0
    languages: dict[str, int] = Field(default_factory=dict)
    directory_tree: list[str] = Field(default_factory=list)


# === Query Plan ===

class QueryPlan(BaseModel):
    relevant_files: list[str] = Field(default_factory=list)
    relevant_modules: list[str] = Field(default_factory=list)
    reasoning: str = ""
    needs_raw_code: bool = False


class EntryPoint(BaseModel):
    file_path: str
    function_name: str = ""
    reason: str = ""


# === NEW: Execution Flow ===

class ExecutionStep(BaseModel):
    """A single step in the execution workflow."""
    step_number: int = 0
    actor: str = ""        # "user", "browser", "background.js", etc.
    action: str = ""       # what happens
    target: str = ""       # which file/component is involved
    data_involved: str = ""  # what data is passed/transformed
    description: str = ""


class ExecutionFlow(BaseModel):
    """End-to-end runtime workflow from trigger to output."""
    trigger: str = ""
    steps: list[ExecutionStep] = Field(default_factory=list)
    output: str = ""
    summary: str = ""


# === NEW: Data Flow ===

class DataFlowStep(BaseModel):
    """Describes one data transformation."""
    source: str = ""       # where data comes from
    transform: str = ""    # what happens to it
    destination: str = ""  # where it goes
    data_type: str = ""    # "HTML", "JSON", "PDF bytes", etc.


class DataFlow(BaseModel):
    """How data moves through the system."""
    steps: list[DataFlowStep] = Field(default_factory=list)
    summary: str = ""


# === NEW: Enhanced Technology Stack ===

class TechnologyProfile(BaseModel):
    """Rich technology classification beyond just listing libraries."""
    platform: str = ""           # "Chrome Extension", "Node.js Server", "React SPA"
    platform_category: str = ""  # "Browser Extension", "Web Application", "CLI Tool"
    primary_language: str = ""
    runtime_environment: str = ""  # "Browser", "Node.js", "Python Runtime"
    apis_used: list[str] = Field(default_factory=list)  # "Chrome Extension API", "Fetch API"
    libraries: list[str] = Field(default_factory=list)
    build_tools: list[str] = Field(default_factory=list)
    summary: str = ""


class ArchitectureSummary(BaseModel):
    overview: str = ""
    key_components: list[str] = Field(default_factory=list)
    design_patterns: list[str] = Field(default_factory=list)
    entry_points: list[EntryPoint] = Field(default_factory=list)
    technology_stack: list[str] = Field(default_factory=list)
    # NEW fields
    technology_profile: TechnologyProfile = Field(default_factory=TechnologyProfile)
    file_interactions: list[FileInteraction] = Field(default_factory=list)
    execution_flow: ExecutionFlow = Field(default_factory=ExecutionFlow)
    data_flow: DataFlow = Field(default_factory=DataFlow)
    component_interaction_summary: str = ""


class MermaidDiagram(BaseModel):
    title: str
    diagram_type: str = "file_flow"
    mermaid_syntax: str = ""


# === NEW: Import advanced diagram models from graph_models ===

from app.schemas.graph_models import (
    AdvancedArchitectureDiagram,
    NetworkFlowDiagram,
)


class FullAnalysisReport(BaseModel):
    analysis_id: str
    repository_url: str
    total_files: int = 0
    file_analyses: list[FileAnalysisResult] = Field(default_factory=list)
    compact_summaries: list[CompactFileSummary] = Field(default_factory=list)
    architecture_summary: ArchitectureSummary = Field(default_factory=ArchitectureSummary)
    mermaid_diagrams: list[MermaidDiagram] = Field(default_factory=list)
    repo_map: RepoMap | None = None
    # NEW: Advanced diagrams
    advanced_architecture: AdvancedArchitectureDiagram = Field(
        default_factory=AdvancedArchitectureDiagram
    )
    network_flow: NetworkFlowDiagram = Field(
        default_factory=NetworkFlowDiagram
    )
    status: str = "pending"
    error_message: str | None = None