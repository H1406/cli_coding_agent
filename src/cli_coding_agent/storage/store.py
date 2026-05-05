from __future__ import annotations

from abc import ABC, abstractmethod

from cli_coding_agent.storage.models import MessageRecord, SessionRecord


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
