from __future__ import annotations

from dataclasses import dataclass

from cli_coding_agent.embeddings import EmbeddingClient
from cli_coding_agent.storage import (
    ConversationMemoryRecord,
    ConversationStore,
    ConversationSummaryRecord,
    MessageRecord,
)


@dataclass(frozen=True, slots=True)
class MemoryRefreshResult:
    summary: ConversationSummaryRecord | None
    memories_indexed: int


class ConversationMemoryService:
    def __init__(
        self,
        store: ConversationStore,
        embedder: EmbeddingClient,
        *,
        summary_trigger_messages: int = 6,
        recent_message_limit: int = 4,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.summary_trigger_messages = summary_trigger_messages
        self.recent_message_limit = recent_message_limit

    def refresh_session_memory(self, session_id: str) -> MemoryRefreshResult:
        messages = self.store.list_messages(session_id)
        summary = self._refresh_summary(session_id, messages)
        indexed = self._refresh_message_memories(session_id, messages)
        if summary is not None:
            self.store.upsert_conversation_memory(
                session_id=session_id,
                source_type="summary",
                source_id=summary.id,
                role="system",
                content=summary.summary_text,
                vector=self.embedder.embed_text(summary.summary_text),
                embedding_model=self.embedder.model_name,
                embedding_version=self.embedder.version,
            )
            indexed += 1
        return MemoryRefreshResult(summary=summary, memories_indexed=indexed)

    def _refresh_summary(
        self,
        session_id: str,
        messages: list[MessageRecord],
    ) -> ConversationSummaryRecord | None:
        if len(messages) < self.summary_trigger_messages:
            return self.store.get_latest_conversation_summary(session_id)

        cutoff = max(0, len(messages) - self.recent_message_limit)
        if cutoff <= 0:
            return self.store.get_latest_conversation_summary(session_id)

        summary_messages = messages[:cutoff]
        latest = self.store.get_latest_conversation_summary(session_id)
        if latest is not None and latest.up_to_sequence_no >= summary_messages[-1].sequence_no:
            return latest

        summary_text = self._build_summary(summary_messages)
        return self.store.upsert_conversation_summary(
            session_id=session_id,
            summary_text=summary_text,
            up_to_message_id=summary_messages[-1].id,
            up_to_sequence_no=summary_messages[-1].sequence_no,
        )

    def _refresh_message_memories(
        self,
        session_id: str,
        messages: list[MessageRecord],
    ) -> int:
        indexed = 0
        for message in messages:
            if not self._should_embed_message(message):
                continue
            self.store.upsert_conversation_memory(
                session_id=session_id,
                source_type="message",
                source_id=message.id,
                role=message.role,
                content=message.content,
                vector=self.embedder.embed_text(message.content),
                embedding_model=self.embedder.model_name,
                embedding_version=self.embedder.version,
            )
            indexed += 1
        return indexed

    def _build_summary(self, messages: list[MessageRecord]) -> str:
        user_points = [self._compact(message.content) for message in messages if message.role == "user"]
        assistant_points = [
            self._compact(message.content) for message in messages if message.role == "assistant"
        ]
        lines = ["Current conversation summary:"]
        if user_points:
            lines.append("User requests:")
            lines.extend(f"- {item}" for item in user_points[-3:])
        if assistant_points:
            lines.append("Assistant responses:")
            lines.extend(f"- {item}" for item in assistant_points[-3:])
        return "\n".join(lines)

    def _should_embed_message(self, message: MessageRecord) -> bool:
        stripped = message.content.strip()
        if len(stripped) < 24:
            return False
        return True

    def _compact(self, text: str, limit: int = 160) -> str:
        one_line = " ".join(text.split())
        if len(one_line) <= limit:
            return one_line
        return f"{one_line[:limit - 3]}..."


def format_recent_messages(messages: list[MessageRecord]) -> str:
    if not messages:
        return ""
    lines = ["Recent conversation:"]
    for message in messages:
        role = "User" if message.role == "user" else "Assistant"
        lines.append(f"{role}: {message.content}")
    return "\n".join(lines)


def trim_message_window(
    messages: list[MessageRecord],
    *,
    limit: int,
) -> list[MessageRecord]:
    if limit <= 0:
        return []
    return messages[-limit:]


def unique_memories(memories: list[ConversationMemoryRecord]) -> list[ConversationMemoryRecord]:
    seen: set[tuple[str, str]] = set()
    unique: list[ConversationMemoryRecord] = []
    for memory in memories:
        key = (memory.source_type, memory.source_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(memory)
    return unique
