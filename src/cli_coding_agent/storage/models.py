from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class SessionRecord:
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class MessageRecord:
    id: str
    session_id: str
    role: str
    content: str
    created_at: datetime
    sequence_no: int


@dataclass(frozen=True, slots=True)
class ConversationSummaryRecord:
    id: str
    session_id: str
    summary_text: str
    up_to_message_id: str
    up_to_sequence_no: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ConversationMemoryRecord:
    id: str
    session_id: str
    source_type: str
    source_id: str
    role: str | None
    content: str
    embedding_model: str
    embedding_version: str
    vector: list[float]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RetrievalEventRecord:
    id: str
    session_id: str | None
    repo_id: str | None
    query_text: str
    event_type: str
    payload_json: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RepoIndexRunRecord:
    id: str
    repo_id: str
    branch: str | None
    commit_sha: str | None
    started_at: datetime
    completed_at: datetime | None


@dataclass(frozen=True, slots=True)
class RepoFileRecord:
    id: str
    repo_id: str
    path: str
    content_hash: str
    language: str
    size_bytes: int
    indexed_at: datetime


@dataclass(frozen=True, slots=True)
class RepoChunkRecord:
    id: str
    repo_file_id: str
    file_path: str
    chunk_index: int
    content: str
    token_count: int
    start_line: int
    end_line: int
    language: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RepoChunkEmbeddingRecord:
    id: str
    repo_chunk_id: str
    embedding_model: str
    embedding_version: str
    vector: list[float]
    created_at: datetime
