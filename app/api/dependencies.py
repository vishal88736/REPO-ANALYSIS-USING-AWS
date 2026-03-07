"""FastAPI dependency injection."""

from fastapi import Request
from app.services.analysis_store import AnalysisStore
from app.rag.vector_store import VectorStore


def get_analysis_store(request: Request) -> AnalysisStore:
    return request.app.state.analysis_store


def get_vector_store() -> VectorStore:
    return VectorStore()