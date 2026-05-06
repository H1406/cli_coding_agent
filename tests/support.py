from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from cli_coding_agent.indexing.models import RepoFileCandidate
from cli_coding_agent.storage.models import (
    ConversationMemoryRecord,
    ConversationSummaryRecord,
    MessageRecord,
    RepoChunkEmbeddingRecord,
    RepoChunkRecord,
    RepoFileRecord,
    RepoIndexRunRecord,
    RetrievalEventRecord,
    SessionRecord,
)


@dataclass
class InMemoryConversationStore:
    sessions: dict[str, SessionRecord]
    messages: list[MessageRecord]
    repo_index_runs: list[RepoIndexRunRecord]
    summaries: dict[str, ConversationSummaryRecord]
    conversation_memories: dict[tuple[str, str, str], ConversationMemoryRecord]
    retrieval_events: list[RetrievalEventRecord]
    repo_files: dict[tuple[str, str], RepoFileRecord]
    repo_chunks: dict[str, list[RepoChunkRecord]]
    repo_embeddings: dict[str, RepoChunkEmbeddingRecord]

    def __init__(self) -> None:
        self.sessions = {}
        self.messages = []
        self.repo_index_runs = []
        self.summaries = {}
        self.conversation_memories = {}
        self.retrieval_events = []
        self.repo_files = {}
        self.repo_chunks = {}
        self.repo_embeddings = {}

    def create_session(self, title: str, session_id: str | None = None) -> SessionRecord:
        resolved_id = session_id or f"session-{len(self.sessions) + 1}"
        timestamp = datetime.now(timezone.utc)
        record = SessionRecord(
            id=resolved_id,
            title=title,
            created_at=timestamp,
            updated_at=timestamp,
        )
        self.sessions[resolved_id] = record
        return record

    def get_session(self, session_id: str) -> SessionRecord | None:
        return self.sessions.get(session_id)

    def list_messages(self, session_id: str, limit: int | None = None) -> list[MessageRecord]:
        matching = [message for message in self.messages if message.session_id == session_id]
        return matching if limit is None else matching[:limit]

    def append_message(self, session_id: str, role: str, content: str) -> MessageRecord:
        sequence_no = len([m for m in self.messages if m.session_id == session_id]) + 1
        record = MessageRecord(
            id=f"message-{len(self.messages) + 1}",
            session_id=session_id,
            role=role,
            content=content,
            created_at=datetime.now(timezone.utc),
            sequence_no=sequence_no,
        )
        self.messages.append(record)
        return record

    def upsert_conversation_summary(
        self,
        session_id: str,
        summary_text: str,
        up_to_message_id: str,
        up_to_sequence_no: int,
    ) -> ConversationSummaryRecord:
        record = ConversationSummaryRecord(
            id=f"summary-{session_id}",
            session_id=session_id,
            summary_text=summary_text,
            up_to_message_id=up_to_message_id,
            up_to_sequence_no=up_to_sequence_no,
            created_at=datetime.now(timezone.utc),
        )
        self.summaries[session_id] = record
        return record

    def get_latest_conversation_summary(
        self,
        session_id: str,
    ) -> ConversationSummaryRecord | None:
        return self.summaries.get(session_id)

    def upsert_conversation_memory(
        self,
        session_id: str,
        source_type: str,
        source_id: str,
        role: str | None,
        content: str,
        vector: list[float],
        *,
        embedding_model: str,
        embedding_version: str,
    ) -> ConversationMemoryRecord:
        key = (session_id, source_type, source_id)
        record = ConversationMemoryRecord(
            id=f"memory-{source_type}-{source_id}",
            session_id=session_id,
            source_type=source_type,
            source_id=source_id,
            role=role,
            content=content,
            embedding_model=embedding_model,
            embedding_version=embedding_version,
            vector=vector,
            created_at=datetime.now(timezone.utc),
        )
        self.conversation_memories[key] = record
        return record

    def list_conversation_memories(self, session_id: str) -> list[ConversationMemoryRecord]:
        return [
            record
            for (memory_session_id, _, _), record in self.conversation_memories.items()
            if memory_session_id == session_id
        ]

    def create_retrieval_event(
        self,
        *,
        event_type: str,
        query_text: str,
        payload_json: str,
        session_id: str | None = None,
        repo_id: str | None = None,
    ) -> RetrievalEventRecord:
        record = RetrievalEventRecord(
            id=f"retrieval-{len(self.retrieval_events) + 1}",
            session_id=session_id,
            repo_id=repo_id,
            query_text=query_text,
            event_type=event_type,
            payload_json=payload_json,
            created_at=datetime.now(timezone.utc),
        )
        self.retrieval_events.append(record)
        return record

    def list_repo_files(self, repo_id: str) -> list[RepoFileRecord]:
        return sorted(
            [record for record in self.repo_files.values() if record.repo_id == repo_id],
            key=lambda record: record.path,
        )

    def create_repo_index_run(
        self,
        repo_id: str,
        branch: str | None,
        commit_sha: str | None,
    ) -> RepoIndexRunRecord:
        record = RepoIndexRunRecord(
            id=f"index-run-{len(self.repo_index_runs) + 1}",
            repo_id=repo_id,
            branch=branch,
            commit_sha=commit_sha,
            started_at=datetime.now(timezone.utc),
            completed_at=None,
        )
        self.repo_index_runs.append(record)
        return record

    def complete_repo_index_run(self, index_run_id: str) -> None:
        self.repo_index_runs = [
            RepoIndexRunRecord(
                id=run.id,
                repo_id=run.repo_id,
                branch=run.branch,
                commit_sha=run.commit_sha,
                started_at=run.started_at,
                completed_at=datetime.now(timezone.utc) if run.id == index_run_id else run.completed_at,
            )
            for run in self.repo_index_runs
        ]

    def upsert_repo_file(self, repo_id: str, candidate: RepoFileCandidate) -> RepoFileRecord:
        key = (repo_id, candidate.relative_path)
        existing = self.repo_files.get(key)
        record = RepoFileRecord(
            id=existing.id if existing else f"repo-file-{len(self.repo_files) + 1}",
            repo_id=repo_id,
            path=candidate.relative_path,
            content_hash=candidate.content_hash,
            language=candidate.language,
            size_bytes=candidate.size_bytes,
            indexed_at=datetime.now(timezone.utc),
        )
        self.repo_files[key] = record
        return record

    def replace_repo_file_chunks(
        self,
        repo_file_id: str,
        file_path: str,
        language: str,
        chunks,
    ) -> list[RepoChunkRecord]:
        records: list[RepoChunkRecord] = []
        for chunk in chunks:
            records.append(
                RepoChunkRecord(
                    id=f"{repo_file_id}-chunk-{chunk.chunk_index}",
                    repo_file_id=repo_file_id,
                    file_path=file_path,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    token_count=chunk.token_count,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    language=language,
                    created_at=datetime.now(timezone.utc),
                )
            )
        self.repo_chunks[repo_file_id] = records
        for chunk_id in list(self.repo_embeddings):
            if chunk_id.startswith(f"{repo_file_id}-chunk-"):
                del self.repo_embeddings[chunk_id]
        return records

    def delete_repo_file(self, repo_id: str, path: str) -> None:
        record = self.repo_files.pop((repo_id, path), None)
        if record is None:
            return
        self.repo_chunks.pop(record.id, None)
        for chunk_id in list(self.repo_embeddings):
            if chunk_id.startswith(f"{record.id}-chunk-"):
                del self.repo_embeddings[chunk_id]

    def replace_chunk_embeddings(
        self,
        chunk_records: list[RepoChunkRecord],
        vectors: list[list[float]],
        *,
        embedding_model: str,
        embedding_version: str,
    ) -> list[RepoChunkEmbeddingRecord]:
        records: list[RepoChunkEmbeddingRecord] = []
        for chunk_record, vector in zip(chunk_records, vectors, strict=True):
            record = RepoChunkEmbeddingRecord(
                id=f"embedding-{chunk_record.id}",
                repo_chunk_id=chunk_record.id,
                embedding_model=embedding_model,
                embedding_version=embedding_version,
                vector=vector,
                created_at=datetime.now(timezone.utc),
            )
            self.repo_embeddings[chunk_record.id] = record
            records.append(record)
        return records

    def list_repo_chunk_embeddings(
        self,
        repo_id: str,
    ) -> list[tuple[RepoChunkRecord, RepoChunkEmbeddingRecord]]:
        results: list[tuple[RepoChunkRecord, RepoChunkEmbeddingRecord]] = []
        for repo_file in self.list_repo_files(repo_id):
            for chunk_record in self.repo_chunks.get(repo_file.id, []):
                embedding = self.repo_embeddings.get(chunk_record.id)
                if embedding is not None:
                    results.append((chunk_record, embedding))
        return results
