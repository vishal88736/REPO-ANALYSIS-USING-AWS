"""
Feature 8: Caching Layer.
SHA-256 hash-based cache for file analyses.
If a file hasn't changed, reuse cached analysis.
"""

import hashlib
import logging
from pathlib import Path

import orjson

from app.config import settings
from app.schemas.analysis import FileAnalysisResult, CompactFileSummary

logger = logging.getLogger(__name__)


class AnalysisCache:
    """
    File-level cache keyed by content hash.
    Stores: FileAnalysisResult + CompactFileSummary per file.
    """

    def __init__(self, cache_dir: Path | None = None):
        self._dir = cache_dir or settings.cache_path
        self._dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    @staticmethod
    def hash_content(content: bytes) -> str:
        """SHA-256 hash of file content."""
        return hashlib.sha256(content).hexdigest()

    def _cache_path(self, content_hash: str) -> Path:
        # Use first 2 chars as subdirectory for filesystem performance
        subdir = self._dir / content_hash[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{content_hash}.json"

    def get(self, content_hash: str) -> FileAnalysisResult | None:
        """Retrieve cached analysis by content hash."""
        path = self._cache_path(content_hash)
        if not path.exists():
            self._misses += 1
            return None

        try:
            data = orjson.loads(path.read_bytes())
            self._hits += 1
            logger.debug("Cache HIT: %s", content_hash[:12])
            return FileAnalysisResult.model_validate(data["analysis"])
        except Exception as e:
            logger.warning("Cache read failed for %s: %s", content_hash[:12], e)
            self._misses += 1
            return None

    def get_summary(self, content_hash: str) -> CompactFileSummary | None:
        """Retrieve cached compact summary."""
        path = self._cache_path(content_hash)
        if not path.exists():
            return None

        try:
            data = orjson.loads(path.read_bytes())
            if "summary" in data:
                return CompactFileSummary.model_validate(data["summary"])
            return None
        except Exception:
            return None

    def put(
        self,
        content_hash: str,
        analysis: FileAnalysisResult,
        summary: CompactFileSummary | None = None,
    ):
        """Store analysis + optional compact summary."""
        path = self._cache_path(content_hash)
        data: dict = {"analysis": analysis.model_dump()}
        if summary:
            data["summary"] = summary.model_dump()

        try:
            path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
            logger.debug("Cache STORE: %s", content_hash[:12])
        except Exception as e:
            logger.warning("Cache write failed: %s", e)

    def stats(self) -> dict:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self._hits / max(self._hits + self._misses, 1):.1%}",
        }