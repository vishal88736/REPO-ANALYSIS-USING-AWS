"""
Analysis Store — persists reports, graphs, repo maps, and compact summaries.
Features 1 + 2: Store repo map and compact summaries for reuse.
"""

import logging
import asyncio
from pathlib import Path

import orjson

from app.schemas.analysis import FullAnalysisReport, RepoMap, CompactFileSummary
from app.schemas.graph_models import DependencyGraph

logger = logging.getLogger(__name__)


class AnalysisStore:
    def __init__(self, store_dir: Path):
        self._store_dir = store_dir
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._status: dict[str, str] = {}

    def _report_path(self, aid: str) -> Path:
        return self._store_dir / f"{aid}_report.json"

    def _graph_path(self, aid: str) -> Path:
        return self._store_dir / f"{aid}_graph.json"

    def _repo_map_path(self, aid: str) -> Path:
        return self._store_dir / f"{aid}_repo_map.json"

    def _summaries_path(self, aid: str) -> Path:
        return self._store_dir / f"{aid}_summaries.json"

    async def set_status(self, aid: str, status: str):
        async with self._lock:
            self._status[aid] = status

    async def get_status(self, aid: str) -> str | None:
        async with self._lock:
            return self._status.get(aid)

    async def save_report(self, report: FullAnalysisReport, graph: DependencyGraph | None = None):
        self._report_path(report.analysis_id).write_bytes(
            orjson.dumps(report.model_dump(), option=orjson.OPT_INDENT_2)
        )
        if graph:
            self._graph_path(report.analysis_id).write_bytes(
                orjson.dumps(graph.model_dump(), option=orjson.OPT_INDENT_2)
            )
        await self.set_status(report.analysis_id, report.status)

    async def save_repo_map(self, aid: str, repo_map: RepoMap):
        """Feature 1: Persist repo map."""
        self._repo_map_path(aid).write_bytes(
            orjson.dumps(repo_map.model_dump(), option=orjson.OPT_INDENT_2)
        )

    async def load_repo_map(self, aid: str) -> RepoMap | None:
        path = self._repo_map_path(aid)
        if not path.exists():
            return None
        try:
            return RepoMap.model_validate(orjson.loads(path.read_bytes()))
        except Exception as e:
            logger.error("Load repo map %s failed: %s", aid, e)
            return None

    async def save_compact_summaries(self, aid: str, summaries: list[CompactFileSummary]):
        """Feature 2: Persist compact summaries."""
        data = [s.model_dump() for s in summaries]
        self._summaries_path(aid).write_bytes(
            orjson.dumps(data, option=orjson.OPT_INDENT_2)
        )

    async def load_compact_summaries(self, aid: str) -> list[CompactFileSummary]:
        path = self._summaries_path(aid)
        if not path.exists():
            return []
        try:
            data = orjson.loads(path.read_bytes())
            return [CompactFileSummary.model_validate(item) for item in data]
        except Exception as e:
            logger.error("Load summaries %s failed: %s", aid, e)
            return []

    async def load_report(self, aid: str) -> FullAnalysisReport | None:
        path = self._report_path(aid)
        if not path.exists():
            return None
        try:
            return FullAnalysisReport.model_validate(orjson.loads(path.read_bytes()))
        except Exception as e:
            logger.error("Load report %s failed: %s", aid, e)
            return None

    async def load_graph(self, aid: str) -> DependencyGraph | None:
        path = self._graph_path(aid)
        if not path.exists():
            return None
        try:
            return DependencyGraph.model_validate(orjson.loads(path.read_bytes()))
        except Exception as e:
            logger.error("Load graph %s failed: %s", aid, e)
            return None

    async def save_error(self, aid: str, url: str, error: str):
        report = FullAnalysisReport(analysis_id=aid, repository_url=url, status="failed", error_message=error)
        await self.save_report(report)