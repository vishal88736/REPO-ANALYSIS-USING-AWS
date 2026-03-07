"""
Repository Scanner — clones GitHub repo, scans files, generates repo map.
Feature 1: Repo Map Generation.
Feature 6: Enhanced File Filtering.
"""

import logging
import os
from pathlib import Path
from collections import defaultdict

import git

from app.config import settings
from app.core.exceptions import RepositoryCloneError
from app.schemas.analysis import FileMetadata, RepoMap, RepoMapEntry
from app.services.token_utils import CHARS_PER_TOKEN

logger = logging.getLogger(__name__)

# Feature 6: Enhanced filtering
IGNORED_DIRS = {
    ".git", "node_modules", "venv", ".venv", "env", ".env",
    "dist", "build", "__pycache__", ".tox", ".mypy_cache",
    ".pytest_cache", "vendor", "target", ".idea", ".vscode",
    "coverage", ".next", ".nuxt", "out", ".cache", ".turbo",
    "bower_components", ".gradle", ".mvn", "bin", "obj",
    ".terraform", ".serverless", "cdk.out",
}

# Feature 6: Skip binary/asset extensions
IGNORED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".bmp",
    ".mp3", ".mp4", ".avi", ".mov", ".wav",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".map", ".min.js", ".min.css",
    ".lock", ".sum",
    ".woff", ".woff2", ".ttf", ".eot",
    ".pyc", ".pyo", ".class", ".o", ".so", ".dll", ".exe",
    ".DS_Store",
}

# Feature 6: Skip specific filenames
IGNORED_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "composer.lock", "Gemfile.lock", "Cargo.lock",
    "go.sum", "poetry.lock", "Pipfile.lock",
}

CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".java",
    ".rs", ".c", ".cpp", ".cc", ".h", ".hpp",
    ".rb", ".php", ".swift", ".kt", ".scala",
    ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg",
    ".md", ".rst", ".txt", ".sql", ".graphql", ".proto",
    ".sh", ".bash", ".zsh", ".dockerfile", ".tf", ".hcl",
}

LANG_MAP = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".go": "Go",
    ".java": "Java", ".rs": "Rust", ".c": "C", ".cpp": "C++",
    ".cc": "C++", ".h": "C", ".hpp": "C++", ".rb": "Ruby",
    ".php": "PHP", ".swift": "Swift", ".kt": "Kotlin",
    ".scala": "Scala", ".yaml": "YAML", ".yml": "YAML",
    ".json": "JSON", ".toml": "TOML", ".md": "Markdown",
    ".sql": "SQL", ".sh": "Shell", ".dockerfile": "Dockerfile",
}


def _repo_dir_name(url: str) -> str:
    parts = url.rstrip("/").rstrip(".git").split("/")
    return f"{parts[-2]}__{parts[-1]}" if len(parts) >= 2 else parts[-1]

def clone_repository(url: str) -> Path:
    """Clone (shallow) or reuse cached repo safely."""
    dir_name = _repo_dir_name(url)
    clone_path = settings.clone_path / dir_name

    # Case 1: Valid cached repo
    if clone_path.exists() and (clone_path / ".git").exists():
        logger.info("Using cached clone: %s", clone_path)
        try:
            repo = git.Repo(clone_path)
            repo.remotes.origin.pull()
        except Exception as e:
            logger.warning("Pull failed, using cache: %s", e)
        return clone_path

    # Case 2: Folder exists but NOT a git repo → remove it
    if clone_path.exists() and not (clone_path / ".git").exists():
        logger.warning("Invalid repo directory detected, deleting: %s", clone_path)
        import shutil
        shutil.rmtree(clone_path)

    # Case 3: Fresh clone
    try:
        logger.info("Cloning: %s → %s", url, clone_path)
        git.Repo.clone_from(
            url,
            str(clone_path),
            depth=1,
            single_branch=True
        )
        return clone_path

    except git.GitCommandError as e:
        raise RepositoryCloneError(
            f"Clone failed: {url}",
            details=str(e)
        )


def _should_skip_file(filename: str, ext: str) -> bool:
    """Feature 6: Enhanced file filtering."""
    if filename in IGNORED_FILES:
        return True
    if ext in IGNORED_EXTENSIONS:
        return True
    # Skip minified files
    if filename.endswith(".min.js") or filename.endswith(".min.css"):
        return True
    return False


def scan_repository(repo_path: Path) -> list[FileMetadata]:
    """Recursively scan repo for code files with enhanced filtering."""
    files = []
    max_size = settings.max_file_size_kb * 1024

    for root, dirs, filenames in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

        for filename in filenames:
            full_path = Path(root) / filename
            ext = full_path.suffix.lower()
            if filename.lower() == "dockerfile":
                ext = ".dockerfile"

            # Feature 6: Enhanced filtering
            if _should_skip_file(filename, ext):
                continue
            if ext not in CODE_EXTENSIONS:
                continue

            try:
                size = full_path.stat().st_size
            except OSError:
                continue
            if size > max_size or size == 0:
                continue

            files.append(FileMetadata(
                path=str(full_path.relative_to(repo_path)),
                size_bytes=size,
                language=LANG_MAP.get(ext, "unknown"),
                extension=ext,
            ))

    logger.info("Scanned %d files in %s", len(files), repo_path)
    return files


def generate_repo_map(repo_path: Path, files: list[FileMetadata]) -> RepoMap:
    """
    Feature 1: Generate repository map from scanned files.
    Stores directory tree, file paths, sizes, types, token estimates.
    """
    file_entries: dict[str, RepoMapEntry] = {}
    lang_count: dict[str, int] = defaultdict(int)
    total_tokens = 0
    directories: set[str] = set()

    for f in files:
        tokens_est = f.size_bytes // CHARS_PER_TOKEN
        total_tokens += tokens_est

        directory = str(Path(f.path).parent)
        directories.add(directory)

        file_entries[f.path] = RepoMapEntry(
            language=f.language,
            size_bytes=f.size_bytes,
            tokens_estimate=tokens_est,
            extension=f.extension,
            directory=directory,
        )
        lang_count[f.language] += 1

    # Build directory tree (sorted)
    dir_tree = sorted(directories)

    repo_map = RepoMap(
        files=file_entries,
        total_files=len(files),
        total_tokens_estimate=total_tokens,
        languages=dict(lang_count),
        directory_tree=dir_tree,
    )

    logger.info(
        "Repo map: %d files, ~%d tokens, %d languages",
        repo_map.total_files, repo_map.total_tokens_estimate, len(repo_map.languages),
    )
    return repo_map