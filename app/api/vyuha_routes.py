"""VYUHA API endpoint — Bedrock single-shot or staged fallback."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_analysis_store
from app.services.analysis_store import AnalysisStore
from app.agents.llm_router import LLMRouter
from app.vyuha.scanner_adapter import file_analyses_to_parsed_repo
from app.vyuha.bedrock_assembler import build_diagram_spec_bedrock

logger = logging.getLogger(__name__)
vyuha_router = APIRouter(tags=["VYUHA Diagrams"])


@vyuha_router.get("/vyuha/{analysis_id}")
async def get_vyuha_diagram(
    analysis_id: str,
    store: AnalysisStore = Depends(get_analysis_store),
):
    """
    Generate VYUHA diagram_spec.
    - With Bedrock: single-shot full prompt → Qwen3-480B
    - Without Bedrock: staged pipeline (works with any model)
    """
    status = await store.get_status(analysis_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    if status != "completed":
        raise HTTPException(status_code=400, detail=f"Not ready. Status: {status}")

    report = await store.load_report(analysis_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")

    parts = report.repository_url.rstrip("/").split("/")
    repo_name = parts[-1] if parts else "unknown"

    parsed_repo = file_analyses_to_parsed_repo(
        repo_name=repo_name,
        repo_url=report.repository_url,
        file_analyses=report.file_analyses,
        architecture=report.architecture_summary,
    )

    router = LLMRouter()
    result = await build_diagram_spec_bedrock(parsed_repo, router)

    return result