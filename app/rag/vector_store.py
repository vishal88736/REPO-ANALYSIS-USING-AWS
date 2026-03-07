"""
RAG Vector Store — indexes STRIPPED SOURCE CODE as the primary database.
The actual code (comments/blanks removed) is the ground truth for RAG queries.
Also indexes summaries, functions, interactions for semantic search.
"""

import logging
import hashlib
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.config import settings
from app.schemas.analysis import FileAnalysisResult, ArchitectureSummary

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, persist_dir: Path | None = None):
        self._persist_dir = persist_dir or settings.vector_store_path
        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model,
        )

    def _collection_name(self, analysis_id: str) -> str:
        safe = analysis_id.replace("-", "_")[:60]
        return f"a_{safe}"

    def _get_or_create_collection(self, analysis_id: str):
        return self._client.get_or_create_collection(
            name=self._collection_name(analysis_id),
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def _chunk_text(self, text: str, max_size: int | None = None) -> list[str]:
        max_size = max_size or settings.rag_chunk_size
        overlap = settings.rag_chunk_overlap
        if len(text) <= max_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    def _chunk_code(self, code: str, file_path: str, max_size: int = 1200) -> list[str]:
        """Chunk code intelligently — split on function/class boundaries when possible."""
        lines = code.split("\n")
        if len(code) <= max_size:
            return [f"# File: {file_path}\n{code}"]

        chunks = []
        current = [f"# File: {file_path}"]
        current_size = len(current[0])

        for line in lines:
            line_size = len(line) + 1
            if current_size + line_size > max_size and len(current) > 1:
                chunks.append("\n".join(current))
                # Keep file header + overlap (last 3 lines)
                overlap_lines = current[-3:] if len(current) > 3 else current[-1:]
                current = [f"# File: {file_path} (cont.)"] + overlap_lines
                current_size = sum(len(l) + 1 for l in current)
            current.append(line)
            current_size += line_size

        if len(current) > 1:
            chunks.append("\n".join(current))

        return chunks

    def _make_id(self, *parts: str) -> str:
        return hashlib.sha256("::".join(parts).encode()).hexdigest()[:32]

    async def index_analysis(
        self,
        analysis_id: str,
        file_analyses: list[FileAnalysisResult],
        architecture_summary: ArchitectureSummary,
        stripped_code_map: dict[str, str] | None = None,
    ) -> int:
        """
        Index everything into ChromaDB:
        1. STRIPPED SOURCE CODE (primary RAG database)
        2. File summaries + functions + classes
        3. Architecture overview + flows
        4. File interactions
        """
        collection = self._get_or_create_collection(analysis_id)

        documents, metadatas, ids = [], [], []
        seen_ids: set[str] = set()

        def _add_doc(doc_id: str, document: str, metadata: dict):
            if not document or not document.strip():
                return
            if doc_id in seen_ids:
                counter = 1
                while f"{doc_id}_{counter}" in seen_ids:
                    counter += 1
                doc_id = f"{doc_id}_{counter}"
            seen_ids.add(doc_id)
            ids.append(doc_id)
            documents.append(document)
            metadatas.append(metadata)

        # ═══ 1. STRIPPED SOURCE CODE — the main RAG database ═══
        code_chunks_total = 0
        if stripped_code_map:
            for file_path, stripped_code in stripped_code_map.items():
                if not stripped_code.strip():
                    continue
                code_chunks = self._chunk_code(stripped_code, file_path)
                for i, chunk in enumerate(code_chunks):
                    _add_doc(
                        self._make_id("code", file_path, str(i)),
                        chunk,
                        {
                            "type": "source_code",
                            "file_path": file_path,
                            "chunk_index": i,
                            "total_chunks": len(code_chunks),
                        },
                    )
                    code_chunks_total += 1

        # ═══ 2. Architecture summaries ═══
        if architecture_summary.overview:
            for i, chunk in enumerate(self._chunk_text(architecture_summary.overview)):
                _add_doc(
                    self._make_id("arch", "overview", str(i)),
                    chunk,
                    {"type": "architecture_summary", "file_path": "__global__", "chunk_index": i},
                )

        if architecture_summary.component_interaction_summary:
            for i, chunk in enumerate(self._chunk_text(architecture_summary.component_interaction_summary)):
                _add_doc(
                    self._make_id("arch", "component_interaction", str(i)),
                    chunk,
                    {"type": "component_interaction", "file_path": "__global__", "chunk_index": i},
                )

        if architecture_summary.execution_flow and architecture_summary.execution_flow.summary:
            _add_doc(
                self._make_id("arch", "execution_flow"),
                architecture_summary.execution_flow.summary,
                {"type": "execution_flow", "file_path": "__global__", "chunk_index": 0},
            )

        if architecture_summary.data_flow and architecture_summary.data_flow.summary:
            _add_doc(
                self._make_id("arch", "data_flow"),
                architecture_summary.data_flow.summary,
                {"type": "data_flow", "file_path": "__global__", "chunk_index": 0},
            )

        # ═══ 3. File summaries + functions + classes ═══
        for fa in file_analyses:
            if fa.summary:
                for i, chunk in enumerate(self._chunk_text(fa.summary)):
                    _add_doc(
                        self._make_id("file", fa.file_path, "summary", str(i)),
                        f"File: {fa.file_path}\n{chunk}",
                        {"type": "file_summary", "file_path": fa.file_path, "chunk_index": i},
                    )

            for func in fa.functions:
                text = f"Function `{func.name}` in {fa.file_path}: {func.description}"
                if func.calls:
                    text += f"\nCalls: {', '.join(func.calls)}"
                if func.imports_used:
                    text += f"\nUses: {', '.join(func.imports_used)}"
                _add_doc(
                    self._make_id("func", fa.file_path, func.name),
                    text,
                    {"type": "function", "file_path": fa.file_path, "function_name": func.name},
                )

            for cls in fa.classes:
                text = f"Class `{cls.name}` in {fa.file_path}. Methods: {', '.join(cls.methods)}"
                _add_doc(
                    self._make_id("cls", fa.file_path, cls.name),
                    text,
                    {"type": "class", "file_path": fa.file_path, "class_name": cls.name},
                )

            # ═══ 4. File interactions ═══
            for inter in fa.file_interactions:
                text = (
                    f"File interaction: {inter.source_file} --[{inter.interaction_type}]--> "
                    f"{inter.target_file}: {inter.description}"
                )
                _add_doc(
                    self._make_id("interaction", inter.source_file, inter.target_file, inter.interaction_type),
                    text,
                    {"type": "file_interaction", "file_path": inter.source_file},
                )

        # Batch upsert
        if documents:
            batch_size = 500
            for i in range(0, len(documents), batch_size):
                end = min(i + batch_size, len(documents))
                collection.upsert(
                    ids=ids[i:end],
                    documents=documents[i:end],
                    metadatas=metadatas[i:end],
                )

        logger.info(
            "Indexed %d docs (code=%d, other=%d) for %s",
            len(documents), code_chunks_total, len(documents) - code_chunks_total, analysis_id,
        )
        return len(documents)

    async def search(
        self,
        analysis_id: str,
        query: str,
        top_k: int | None = None,
        filter_type: str | None = None,
    ) -> list[dict]:
        top_k = top_k or settings.rag_top_k
        collection = self._get_or_create_collection(analysis_id)

        where = {"type": filter_type} if filter_type else None

        try:
            results = collection.query(query_texts=[query], n_results=top_k, where=where)
        except Exception as e:
            logger.error("Vector search failed: %s", e)
            return []

        chunks = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else 0.0
                chunks.append({"text": doc, "metadata": meta, "distance": dist})
        return chunks

    async def search_code(self, analysis_id: str, query: str, top_k: int = 10) -> list[dict]:
        """Search only source code chunks — for precise code-level RAG answers."""
        return await self.search(analysis_id, query, top_k=top_k, filter_type="source_code")

    def delete_collection(self, analysis_id: str):
        try:
            self._client.delete_collection(self._collection_name(analysis_id))
        except Exception:
            pass