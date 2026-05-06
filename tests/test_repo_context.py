from __future__ import annotations

from pathlib import Path

from cli_coding_agent.agent import CodingAgent
from cli_coding_agent.config import AgentConfig
from cli_coding_agent.indexing.service import RepositoryIndexingService
from cli_coding_agent.retrieval import ConversationRetriever, RepoRetriever

from tests.support import InMemoryConversationStore


def test_indexing_service_persists_repo_files_chunks_and_embeddings(tmp_path: Path) -> None:
    store = InMemoryConversationStore()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "service.py").write_text(
        "def create_session():\n"
        "    return 'session'\n\n"
        "def persist_message():\n"
        "    return 'stored'\n",
        encoding="utf-8",
    )

    agent = CodingAgent(
        AgentConfig(
            agent_name="test-agent",
            provider="huggingface",
            model="test-model",
            instructions_path=str(tmp_path / "missing.txt"),
            huggingface_model_id="Qwen/Qwen2.5-Coder-1.5B",
            database_url="postgresql://example",
            repo_root=str(repo_root),
            repo_id="repo-1",
        ),
        store,
    )

    result = agent.indexing_service.index_repository(repo_root, "repo-1")

    assert result.files_scanned == 1
    assert result.files_changed == 1
    assert result.chunks_written >= 1
    assert len(store.list_repo_files("repo-1")) == 1
    assert store.list_repo_chunk_embeddings("repo-1")
    assert store.repo_index_runs[0].completed_at is not None


def test_repo_retriever_returns_relevant_chunk_for_query(tmp_path: Path) -> None:
    store = InMemoryConversationStore()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "agent.py").write_text(
        "def store_message():\n"
        "    return 'message stored'\n\n"
        "def load_index():\n"
        "    return 'index loaded'\n",
        encoding="utf-8",
    )

    service = RepositoryIndexingService(store, CodingAgent(
        AgentConfig(
            agent_name="test-agent",
            provider="huggingface",
            model="test-model",
            instructions_path=str(tmp_path / "missing.txt"),
            huggingface_model_id="Qwen/Qwen2.5-Coder-1.5B",
            database_url="postgresql://example",
            repo_root=str(repo_root),
            repo_id="repo-1",
        ),
        store,
    ).embedder)
    service.index_repository(repo_root, "repo-1")

    retriever = RepoRetriever(store, CodingAgent(
        AgentConfig(
            agent_name="test-agent",
            provider="huggingface",
            model="test-model",
            instructions_path=str(tmp_path / "missing.txt"),
            huggingface_model_id="Qwen/Qwen2.5-Coder-1.5B",
            database_url="postgresql://example",
            repo_root=str(repo_root),
            repo_id="repo-1",
        ),
        store,
    ).embedder)
    results = retriever.search("repo-1", "where do we store messages?", limit=1)

    assert len(results) == 1
    assert "store_message" in results[0].chunk.content


def test_agent_build_prompt_includes_retrieved_repo_context(tmp_path: Path) -> None:
    store = InMemoryConversationStore()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    instruction_path = tmp_path / "instruction.txt"
    instruction_path.write_text("Use the provided repo context.", encoding="utf-8")
    (repo_root / "storage.py").write_text(
        "def persist_turn():\n"
        "    return 'saved'\n",
        encoding="utf-8",
    )

    agent = CodingAgent(
        AgentConfig(
            agent_name="test-agent",
            provider="huggingface",
            model="test-model",
            instructions_path=str(instruction_path),
            huggingface_model_id="Qwen/Qwen2.5-Coder-1.5B",
            database_url="postgresql://example",
            repo_root=str(repo_root),
            repo_id="repo-1",
        ),
        store,
    )
    agent.indexing_service.index_repository(repo_root, "repo-1")
    repo_chunks = agent.repo_retriever.search("repo-1", "how are turns persisted?", limit=1)

    prompt = agent.build_prompt("Explain persistence", repo_chunks=repo_chunks)

    assert "Retrieved repo context:" in prompt
    assert "Source: storage.py lines" in prompt
    assert "persist_turn" in prompt


def test_conversation_memory_service_creates_summary_and_retrieves_memory(tmp_path: Path) -> None:
    store = InMemoryConversationStore()
    instruction_path = tmp_path / "instruction.txt"
    instruction_path.write_text("Use memory.", encoding="utf-8")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "helper.py").write_text("def helper():\n    return True\n", encoding="utf-8")

    agent = CodingAgent(
        AgentConfig(
            agent_name="test-agent",
            provider="huggingface",
            model="test-model",
            instructions_path=str(instruction_path),
            huggingface_model_id="Qwen/Qwen2.5-Coder-1.5B",
            database_url="postgresql://example",
            repo_root=str(repo_root),
            repo_id="repo-1",
            summary_trigger_messages=4,
            recent_message_limit=2,
        ),
        store,
    )
    session_id = agent.ensure_session()

    store.append_message(session_id, "user", "We need historical conversations stored for later retrieval.")
    store.append_message(session_id, "assistant", "I will store and retrieve the session history.")
    store.append_message(session_id, "user", "Please remember the persistence requirement across sessions.")
    store.append_message(session_id, "assistant", "Persisting cross-session memory is now a requirement.")

    refresh = agent.memory_service.refresh_session_memory(session_id)
    retriever = ConversationRetriever(store, agent.embedder)
    hits = retriever.search(session_id, "what did we decide about persistence?", limit=3)

    assert refresh.summary is not None
    assert "Current conversation summary:" in refresh.summary.summary_text
    assert hits
    assert any("persistence" in item.memory.content.lower() for item in hits)
