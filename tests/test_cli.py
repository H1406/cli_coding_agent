from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cli_coding_agent.agent import CodingAgent
from cli_coding_agent.config import AgentConfig
from cli_coding_agent.prompts import load_instructions
from cli_coding_agent.storage.models import MessageRecord, SessionRecord


@dataclass
class InMemoryConversationStore:
    sessions: dict[str, SessionRecord]
    messages: list[MessageRecord]

    def __init__(self) -> None:
        self.sessions = {}
        self.messages = []

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
        if limit is None:
            return matching
        return matching[:limit]

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


def test_load_instructions_from_relative_repo_path(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)

    repo_instruction_path = Path(__file__).resolve().parents[1] / "instruction.txt"
    original_text = repo_instruction_path.read_text(encoding="utf-8")

    assert load_instructions("instruction.txt") == original_text.strip()


def test_agent_builds_prompt_with_instructions(tmp_path) -> None:
    instruction_path = tmp_path / "instruction.txt"
    instruction_path.write_text("You are a coding agent.", encoding="utf-8")
    store = InMemoryConversationStore()

    agent = CodingAgent(
        AgentConfig(
            agent_name="test-agent",
            provider="huggingface",
            model="test-model",
            instructions_path=str(instruction_path),
            huggingface_model_id="Qwen/Qwen2.5-Coder-1.5B",
            database_url="postgresql://example",
        ),
        store,
    )

    result = agent.build_prompt("Inspect the repository")

    assert "You are a coding agent." in result
    assert "Inspect the repository" in result
    assert "Assistant response:" in result


def test_agent_run_passes_built_prompt_to_model(tmp_path, monkeypatch) -> None:
    instruction_path = tmp_path / "instruction.txt"
    instruction_path.write_text("Follow the repository rules.", encoding="utf-8")
    store = InMemoryConversationStore()

    agent = CodingAgent(
        AgentConfig(
            agent_name="test-agent",
            provider="huggingface",
            model="test-model",
            instructions_path=str(instruction_path),
            huggingface_model_id="Qwen/Qwen2.5-Coder-1.5B",
            database_url="postgresql://example",
        ),
        store,
    )

    captured = {}

    def fake_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return "stubbed output"

    monkeypatch.setattr(agent, "llm", fake_llm)

    result = agent.run("Inspect the repository")

    assert "Instructions loaded: True" in result
    assert captured["prompt"].startswith("Follow the repository rules.")
    assert "Session ID: session-1" in result
    assert "stubbed output" in result
    assert [message.role for message in store.messages] == ["user", "assistant"]
    assert store.messages[0].content == "Inspect the repository"
    assert store.messages[1].content == "stubbed output"


def test_agent_reuses_explicit_session_id(tmp_path, monkeypatch) -> None:
    instruction_path = tmp_path / "instruction.txt"
    instruction_path.write_text("Persist turns.", encoding="utf-8")
    store = InMemoryConversationStore()
    store.create_session(title="existing", session_id="session-42")

    agent = CodingAgent(
        AgentConfig(
            agent_name="test-agent",
            provider="huggingface",
            model="test-model",
            instructions_path=str(instruction_path),
            huggingface_model_id="Qwen/Qwen2.5-Coder-1.5B",
            database_url="postgresql://example",
            session_id="session-42",
        ),
        store,
    )

    monkeypatch.setattr(agent, "llm", lambda prompt: "second reply")

    result = agent.run("Resume work")

    assert "Session ID: session-42" in result
    assert [message.sequence_no for message in store.messages] == [1, 2]


def test_config_reads_database_url_and_session_id(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_DATABASE_URL", "postgresql://db/test")
    monkeypatch.setenv("AGENT_SESSION_ID", "session-from-env")

    config = AgentConfig.from_env()

    assert config.database_url == "postgresql://db/test"
    assert config.session_id == "session-from-env"
