"""Pydantic models for API request/response."""

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    repository_url: str = Field(
        ...,
        description="Full GitHub repository URL",
        examples=["https://github.com/pallets/flask"],
    )


class AnalyzeResponse(BaseModel):
    analysis_id: str
    status: str
    message: str


class QueryRequest(BaseModel):
    analysis_id: str
    question: str = Field(..., min_length=3)


class QueryResponse(BaseModel):
    analysis_id: str
    question: str
    answer: str
    sources: list[str] = Field(default_factory=list)


class ReportResponse(BaseModel):
    """
    ONE output only: ~5000 words containing explanation + architecture diagram
    + network flow diagram as mermaid code blocks. No info loss.
    """
    analysis_id: str
    repository_url: str
    status: str
    total_files: int
    report: str = ""   # ← the entire 5000-word output (explanation + diagrams)


class StatusResponse(BaseModel):
    analysis_id: str
    status: str


class ErrorResponse(BaseModel):
    detail: str