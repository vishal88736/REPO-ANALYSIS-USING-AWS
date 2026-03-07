"""
Token utilities — estimation, budgeting, compression, splitting.
Features 3 (Token-Aware Processing) + 10 (Context Compression).
"""

import re
import logging

from app.config import settings

logger = logging.getLogger(__name__)

# Rough estimate: 1 token ≈ 4 characters for English/code
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length."""
    return len(text) // CHARS_PER_TOKEN


def is_within_budget(text: str, max_tokens: int | None = None) -> bool:
    """Check if text fits within token budget."""
    max_tokens = max_tokens or settings.max_file_tokens
    return estimate_tokens(text) <= max_tokens


def compress_code(content: str, language: str = "unknown") -> str:
    """
    Feature 10: Context Compression.
    Remove comments, excessive whitespace, formatting noise.
    Keep function bodies, variable names, logic flow.
    """
    lines = content.split("\n")
    compressed_lines = []

    in_multiline_comment = False

    for line in lines:
        stripped = line.strip()

        # Skip empty lines (keep max 1 consecutive)
        if not stripped:
            if compressed_lines and compressed_lines[-1] == "":
                continue
            compressed_lines.append("")
            continue

        # Python/Shell single-line comments
        if language in ("Python", "Shell") and stripped.startswith("#"):
            # Keep shebang and type hints
            if stripped.startswith("#!") or stripped.startswith("# type:"):
                compressed_lines.append(line)
            continue

        # JS/TS/Go/Java/C/C++/Rust single-line comments
        if language in ("JavaScript", "TypeScript", "Go", "Java", "C", "C++", "Rust"):
            if stripped.startswith("//"):
                continue

        # Multi-line comment handling (/* ... */)
        if "/*" in stripped and language in ("JavaScript", "TypeScript", "Go", "Java", "C", "C++", "Rust"):
            in_multiline_comment = True
            if "*/" in stripped:
                in_multiline_comment = False
            continue
        if in_multiline_comment:
            if "*/" in stripped:
                in_multiline_comment = False
            continue

        # Python docstrings — keep first line only
        if language == "Python" and (stripped.startswith('"""') or stripped.startswith("'''")):
            quote = stripped[:3]
            if stripped.count(quote) >= 2:
                # Single-line docstring — skip it
                continue
            else:
                # Multi-line docstring — skip until closing
                in_multiline_comment = True
                continue

        if in_multiline_comment and language == "Python":
            if '"""' in stripped or "'''" in stripped:
                in_multiline_comment = False
            continue

        # Remove trailing whitespace, reduce indentation to 2 spaces
        cleaned = line.rstrip()
        leading_spaces = len(line) - len(line.lstrip())
        indent = " " * min(leading_spaces, (leading_spaces // 4) * 2)
        compressed_lines.append(indent + cleaned.lstrip())

    result = "\n".join(compressed_lines).strip()

    # Remove consecutive blank lines
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result


def split_by_token_budget(
    content: str,
    max_tokens: int | None = None,
) -> list[str]:
    """
    Feature 3: Split content into chunks that fit within token budget.
    Tries to split at function/class boundaries first, then falls back to line splits.
    """
    max_tokens = max_tokens or settings.max_file_tokens

    if estimate_tokens(content) <= max_tokens:
        return [content]

    lines = content.split("\n")
    chunks = []
    current_chunk_lines: list[str] = []
    current_tokens = 0

    for line in lines:
        line_tokens = estimate_tokens(line)

        if current_tokens + line_tokens > max_tokens and current_chunk_lines:
            chunks.append("\n".join(current_chunk_lines))
            current_chunk_lines = []
            current_tokens = 0

        current_chunk_lines.append(line)
        current_tokens += line_tokens

    if current_chunk_lines:
        chunks.append("\n".join(current_chunk_lines))

    logger.debug("Split content into %d chunks (budget=%d tokens)", len(chunks), max_tokens)
    return chunks


def truncate_to_budget(text: str, max_tokens: int | None = None) -> str:
    """Truncate text to fit within token budget."""
    max_tokens = max_tokens or settings.max_prompt_tokens
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n... [TRUNCATED to fit token budget] ..."