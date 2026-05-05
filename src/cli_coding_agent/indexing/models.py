from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RepoFileCandidate:
    path: Path
    relative_path: str
    content_hash: str
    language: str
    size_bytes: int
    changed: bool


@dataclass(frozen=True, slots=True)
class RepoChunk:
    file_path: str
    chunk_index: int
    content: str
    token_count: int
    start_line: int
    end_line: int
    language: str


@dataclass(frozen=True, slots=True)
class RepoFileRecord:
    id: str
    repo_id: str
    path: str
    content_hash: str
    language: str
    size_bytes: int
    indexed_at: datetime

