from __future__ import annotations

from dataclasses import dataclass

from cli_coding_agent.embeddings import EmbeddingClient, cosine_similarity
from cli_coding_agent.storage import ConversationMemoryRecord, ConversationStore, RepoChunkRecord


@dataclass(frozen=True, slots=True)
class RetrievedRepoChunk:
    chunk: RepoChunkRecord
    score: float


class RepoRetriever:
    def __init__(self, store: ConversationStore, embedder: EmbeddingClient) -> None:
        self.store = store
        self.embedder = embedder

    def search(self, repo_id: str, query: str, limit: int = 3) -> list[RetrievedRepoChunk]:
        query_vector = self.embedder.embed_text(query)
        candidates = self.store.list_repo_chunk_embeddings(repo_id)

        ranked = [
            RetrievedRepoChunk(
                chunk=chunk_record,
                score=cosine_similarity(query_vector, embedding_record.vector),
            )
            for chunk_record, embedding_record in candidates
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:limit]


@dataclass(frozen=True, slots=True)
class RetrievedConversationMemory:
    memory: ConversationMemoryRecord
    score: float


class ConversationRetriever:
    def __init__(self, store: ConversationStore, embedder: EmbeddingClient) -> None:
        self.store = store
        self.embedder = embedder

    def search(
        self,
        session_id: str,
        query: str,
        limit: int = 3,
    ) -> list[RetrievedConversationMemory]:
        query_vector = self.embedder.embed_text(query)
        candidates = self.store.list_conversation_memories(session_id)
        ranked = [
            RetrievedConversationMemory(
                memory=memory,
                score=cosine_similarity(query_vector, memory.vector),
            )
            for memory in candidates
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:limit]
