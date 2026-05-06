from __future__ import annotations

from abc import ABC, abstractmethod

from cli_coding_agent.indexing.models import RepoChunk, RepoFileCandidate
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


class ConversationStore(ABC):
    @abstractmethod
    def create_session(self, title: str, session_id: str | None = None) -> SessionRecord:
        raise NotImplementedError

    @abstractmethod
    def get_session(self, session_id: str) -> SessionRecord | None:
        raise NotImplementedError

    @abstractmethod
    def list_messages(self, session_id: str, limit: int | None = None) -> list[MessageRecord]:
        raise NotImplementedError

    @abstractmethod
    def append_message(self, session_id: str, role: str, content: str) -> MessageRecord:
        raise NotImplementedError

    @abstractmethod
    def upsert_conversation_summary(
        self,
        session_id: str,
        summary_text: str,
        up_to_message_id: str,
        up_to_sequence_no: int,
    ) -> ConversationSummaryRecord:
        raise NotImplementedError

    @abstractmethod
    def get_latest_conversation_summary(
        self,
        session_id: str,
    ) -> ConversationSummaryRecord | None:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def list_conversation_memories(self, session_id: str) -> list[ConversationMemoryRecord]:
        raise NotImplementedError

    @abstractmethod
    def create_retrieval_event(
        self,
        *,
        event_type: str,
        query_text: str,
        payload_json: str,
        session_id: str | None = None,
        repo_id: str | None = None,
    ) -> RetrievalEventRecord:
        raise NotImplementedError

    @abstractmethod
    def list_repo_files(self, repo_id: str) -> list[RepoFileRecord]:
        raise NotImplementedError

    @abstractmethod
    def create_repo_index_run(
        self,
        repo_id: str,
        branch: str | None,
        commit_sha: str | None,
    ) -> RepoIndexRunRecord:
        raise NotImplementedError

    @abstractmethod
    def complete_repo_index_run(self, index_run_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def upsert_repo_file(self, repo_id: str, candidate: RepoFileCandidate) -> RepoFileRecord:
        raise NotImplementedError

    @abstractmethod
    def replace_repo_file_chunks(
        self,
        repo_file_id: str,
        file_path: str,
        language: str,
        chunks: list[RepoChunk],
    ) -> list[RepoChunkRecord]:
        raise NotImplementedError

    @abstractmethod
    def delete_repo_file(self, repo_id: str, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def replace_chunk_embeddings(
        self,
        chunk_records: list[RepoChunkRecord],
        vectors: list[list[float]],
        *,
        embedding_model: str,
        embedding_version: str,
    ) -> list[RepoChunkEmbeddingRecord]:
        raise NotImplementedError

    @abstractmethod
    def list_repo_chunk_embeddings(
        self,
        repo_id: str,
    ) -> list[tuple[RepoChunkRecord, RepoChunkEmbeddingRecord]]:
        raise NotImplementedError
