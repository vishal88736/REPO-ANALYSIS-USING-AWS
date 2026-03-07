"""
Feature 9: Smart Chunking.
Chunk code by function/class boundaries instead of arbitrary character splits.
"""

import logging

from app.schemas.analysis import ParsedStructure
from app.config import settings

logger = logging.getLogger(__name__)


def smart_chunk_code(
    content: str,
    parsed: ParsedStructure,
    max_chunk_size: int | None = None,
) -> list[dict]:
    """
    Split code into semantically meaningful chunks.
    Each chunk represents a function, class, or module section.
    Returns list of {"text": str, "type": str, "name": str}
    """
    max_size = max_chunk_size or settings.rag_chunk_size
    lines = content.split("\n")
    chunks: list[dict] = []
    used_lines: set[int] = set()

    # Extract function chunks
    for func in parsed.functions:
        start = max(0, func.start_line - 1)
        end = min(len(lines), func.end_line)
        func_text = "\n".join(lines[start:end])

        if len(func_text) <= max_size:
            chunks.append({
                "text": func_text,
                "type": "function",
                "name": func.name,
                "start_line": func.start_line,
                "end_line": func.end_line,
            })
        else:
            # Function too large — split into sub-chunks
            for i in range(0, len(func_text), max_size - 100):
                sub = func_text[i:i + max_size]
                chunks.append({
                    "text": sub,
                    "type": "function_part",
                    "name": f"{func.name}_part{i // max_size}",
                    "start_line": func.start_line,
                    "end_line": func.end_line,
                })

        for ln in range(start, end):
            used_lines.add(ln)

    # Extract class chunks
    for cls in parsed.classes:
        start = max(0, cls.start_line - 1)
        end = min(len(lines), cls.end_line)
        cls_text = "\n".join(lines[start:end])

        if len(cls_text) <= max_size:
            chunks.append({
                "text": cls_text,
                "type": "class",
                "name": cls.name,
                "start_line": cls.start_line,
                "end_line": cls.end_line,
            })
        else:
            for i in range(0, len(cls_text), max_size - 100):
                sub = cls_text[i:i + max_size]
                chunks.append({
                    "text": sub,
                    "type": "class_part",
                    "name": f"{cls.name}_part{i // max_size}",
                    "start_line": cls.start_line,
                    "end_line": cls.end_line,
                })

        for ln in range(start, end):
            used_lines.add(ln)

    # Remaining lines (imports, module-level code) as one chunk
    remaining = []
    for i, line in enumerate(lines):
        if i not in used_lines:
            remaining.append(line)

    remaining_text = "\n".join(remaining).strip()
    if remaining_text:
        if len(remaining_text) <= max_size:
            chunks.append({
                "text": remaining_text,
                "type": "module",
                "name": "module_level",
                "start_line": 0,
                "end_line": 0,
            })
        else:
            for i in range(0, len(remaining_text), max_size - 100):
                sub = remaining_text[i:i + max_size]
                chunks.append({
                    "text": sub,
                    "type": "module_part",
                    "name": f"module_part{i // max_size}",
                    "start_line": 0,
                    "end_line": 0,
                })

    logger.debug("Smart chunked: %d chunks from code", len(chunks))
    return chunks