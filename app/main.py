"""FastAPI application entry point."""

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.config import settings
from app.core.logging_config import setup_logging
from app.api.routes import router
from app.api.vyuha_routes import vyuha_router
from app.services.analysis_store import AnalysisStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level)
    settings.clone_path.mkdir(parents=True, exist_ok=True)
    settings.vector_store_path.mkdir(parents=True, exist_ok=True)
    settings.analysis_store_path.mkdir(parents=True, exist_ok=True)
    settings.cache_path.mkdir(parents=True, exist_ok=True)
    app.state.analysis_store = AnalysisStore(settings.analysis_store_path)
    yield


app = FastAPI(
    title="VYUHA — Multi-Agent Repo Analyzer",
    description=(
        "Analyzes GitHub repositories using Bedrock Qwen3-480B / Ollama / Groq / Gemini "
        "with token optimization, caching, hybrid RAG, progressive loading, "
        "multi-LLM routing, and VYUHA diagram-spec generation."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

# Standard analysis + query endpoints
app.include_router(router, prefix="/api/v1")

# VYUHA diagram-spec endpoint
app.include_router(vyuha_router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)