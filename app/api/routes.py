
"""
REST API:
  GET  /report/{id} → one 5000-word output (explanation + arch diagram + network diagram)
  POST /query       → RAG-powered code questions
"""

import logging

from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, status

from app.api.dependencies import get_analysis_store, get_vector_store
from app.api.security import verify_api_key
from app.schemas.api_models import (
    AnalyzeRequest, AnalyzeResponse, QueryRequest, QueryResponse,
    ReportResponse, StatusResponse, ErrorResponse,
)
from app.services.analysis_store import AnalysisStore
from app.services.orchestrator import run_analysis_pipeline, create_analysis_id
from app.agents.groq_client import GroqClient
from app.agents.llm_router import LLMRouter
from app.agents.rag_answer_agent import answer_query
from app.agents.query_planner import plan_query
from app.graph.dependency_graph import get_dependency_context_for_file
from app.rag.vector_store import VectorStore
from app.rag.hybrid_retriever import hybrid_retrieve

logger = logging.getLogger(__name__)

# 🔐 Protect all routes with API key
router = APIRouter(
    tags=["Analysis"],
    dependencies=[Depends(verify_api_key)]
)


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={400: {"model": ErrorResponse}},
)
async def start_analysis(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    store: AnalysisStore = Depends(get_analysis_store),
):
    url = request.repository_url.strip()

    if not url.startswith("https://github.com/"):
        raise HTTPException(status_code=400, detail="Only GitHub URLs supported")

    analysis_id = create_analysis_id()

    await store.set_status(analysis_id, "pending")

    background_tasks.add_task(
        run_analysis_pipeline,
        analysis_id,
        url,
        store
    )

    return AnalyzeResponse(
        analysis_id=analysis_id,
        status="pending",
        message=f"Analysis started. Track: GET /api/v1/status/{analysis_id}",
    )


@router.get("/status/{analysis_id}", response_model=StatusResponse)
async def get_status(
    analysis_id: str,
    store: AnalysisStore = Depends(get_analysis_store),
):
    s = await store.get_status(analysis_id)

    if s is None:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis '{analysis_id}' not found."
        )

    return StatusResponse(
        analysis_id=analysis_id,
        status=s
    )


@router.get(
    "/report/{analysis_id}",
    response_model=ReportResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_report(
    analysis_id: str,
    store: AnalysisStore = Depends(get_analysis_store),
):
    s = await store.get_status(analysis_id)

    if s is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    if s not in ("completed", "failed"):
        raise HTTPException(
            status_code=202,
            detail=f"Still processing. Status: {s}"
        )

    report = await store.load_report(analysis_id)

    if not report:
        raise HTTPException(
            status_code=404,
            detail="Report file not found."
        )

    if report.status == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {report.error_message}"
        )

    return ReportResponse(
        analysis_id=report.analysis_id,
        repository_url=report.repository_url,
        status=report.status,
        total_files=report.total_files,
        report=report.architecture_summary.overview,
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={404: {"model": ErrorResponse}},
)
async def query_codebase(
    request: QueryRequest,
    store: AnalysisStore = Depends(get_analysis_store),
    vector_store: VectorStore = Depends(get_vector_store),
):
    s = await store.get_status(request.analysis_id)

    if s is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    if s != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Not ready. Status: {s}"
        )

    report = await store.load_report(request.analysis_id)
    graph = await store.load_graph(request.analysis_id)
    repo_map = await store.load_repo_map(request.analysis_id)
    compact_summaries = await store.load_compact_summaries(request.analysis_id)

    groq = GroqClient()
    llm_router = LLMRouter(groq)

    query_plan = await plan_query(
        llm_router,
        request.question,
        repo_map,
        compact_summaries
    )

    chunks = await hybrid_retrieve(
        vector_store=vector_store,
        analysis_id=request.analysis_id,
        query=request.question,
        compact_summaries=compact_summaries,
        graph=graph,
    )

    dep_context = ""

    if graph and chunks:
        relevant_files = {
            c.get("metadata", {}).get("file_path", "")
            for c in chunks[:5]
            if c.get("metadata", {}).get("file_path", "__global__") != "__global__"
        }

        dep_context = "\n".join(
            get_dependency_context_for_file(graph, fp)
            for fp in relevant_files
        )

    arch = report.architecture_summary.overview if report else ""

    answer = await answer_query(
        router=llm_router,
        question=request.question,
        retrieved_chunks=chunks,
        dependency_context=dep_context,
        architecture_summary=arch,
        repo_map=repo_map,
        compact_summaries=compact_summaries,
        query_plan=query_plan,
    )

    sources = list({
        c.get("metadata", {}).get("file_path", "")
        for c in chunks
        if c.get("metadata", {}).get("file_path", "__global__") != "__global__"
    })

    return QueryResponse(
        analysis_id=request.analysis_id,
        question=request.question,
        answer=answer,
        sources=sources,
    )
