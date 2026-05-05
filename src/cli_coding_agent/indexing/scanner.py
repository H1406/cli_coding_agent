from __future__ import annotations

import hashlib
from pathlib import Path

from cli_coding_agent.indexing.models import RepoFileCandidate

DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
}

DEFAULT_EXCLUDED_FILES = {
    ".DS_Store",
}

DEFAULT_EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".lock",
}

LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".md": "markdown",
    ".toml": "toml",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".txt": "text",
    ".sh": "shell",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".html": "html",
    ".css": "css",
    ".sql": "sql",
}


def scan_repository(
    root: str | Path,
    previous_hashes: dict[str, str] | None = None,
) -> list[RepoFileCandidate]:
    root_path = Path(root).resolve()
    previous_hashes = previous_hashes or {}
    candidates: list[RepoFileCandidate] = []

    for path in sorted(root_path.rglob("*")):
        if not path.is_file():
            continue
        if _should_skip(path, root_path):
            continue

        data = path.read_bytes()
        if _is_binary(data):
            continue

        relative_path = path.relative_to(root_path).as_posix()
        content_hash = hashlib.sha256(data).hexdigest()
        candidates.append(
            RepoFileCandidate(
                path=path,
                relative_path=relative_path,
                content_hash=content_hash,
                language=detect_language(path),
                size_bytes=len(data),
                changed=previous_hashes.get(relative_path) != content_hash,
            )
        )

    return candidates


def changed_files(
    root: str | Path,
    previous_hashes: dict[str, str] | None = None,
) -> list[RepoFileCandidate]:
    return [candidate for candidate in scan_repository(root, previous_hashes) if candidate.changed]


def detect_language(path: Path) -> str:
    return LANGUAGE_BY_SUFFIX.get(path.suffix.lower(), "text")


def _should_skip(path: Path, root: Path) -> bool:
    if path.name in DEFAULT_EXCLUDED_FILES:
        return True
    if path.suffix.lower() in DEFAULT_EXCLUDED_SUFFIXES:
        return True

    relative_parts = path.relative_to(root).parts
    return any(part in DEFAULT_EXCLUDED_DIRS for part in relative_parts[:-1])


def _is_binary(data: bytes) -> bool:
    if not data:
        return False
    if b"\x00" in data:
        return True
    sample = data[:1024]
    non_text = sum(byte < 9 or 13 < byte < 32 for byte in sample)
    return non_text / max(len(sample), 1) > 0.3

