from __future__ import annotations

from pathlib import Path

from cli_coding_agent.agent import CodingAgent
from cli_coding_agent.cli import build_parser
from cli_coding_agent.config import AgentConfig
from cli_coding_agent.prompts import load_instructions
from tests.support import InMemoryConversationStore


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
            repo_root=str(tmp_path),
            repo_id="repo-test",
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
            repo_root=str(tmp_path),
            repo_id="repo-test",
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
            repo_root=str(tmp_path),
            repo_id="repo-test",
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
    monkeypatch.setenv("AGENT_REPO_ROOT", "/tmp/repo")
    monkeypatch.setenv("AGENT_REPO_ID", "repo-from-env")
    monkeypatch.setenv("AGENT_REPO_CONTEXT_LIMIT", "5")
    monkeypatch.setenv("AGENT_CONVERSATION_CONTEXT_LIMIT", "2")
    monkeypatch.setenv("AGENT_RECENT_MESSAGE_LIMIT", "6")
    monkeypatch.setenv("AGENT_SUMMARY_TRIGGER_MESSAGES", "8")
    monkeypatch.setenv("AGENT_PROMPT_TOKEN_BUDGET", "900")
    monkeypatch.setenv("AGENT_MAX_ACTION_STEPS", "10")
    monkeypatch.setenv("AGENT_TOOL_RUN_TIMEOUT_SECONDS", "15")

    config = AgentConfig.from_env()

    assert config.database_url == "postgresql://db/test"
    assert config.session_id == "session-from-env"
    assert config.repo_root == "/tmp/repo"
    assert config.repo_id == "repo-from-env"
    assert config.repo_context_limit == 5
    assert config.conversation_context_limit == 2
    assert config.recent_message_limit == 6
    assert config.summary_trigger_messages == 8
    assert config.prompt_token_budget == 900
    assert config.max_action_steps == 10
    assert config.tool_run_timeout_seconds == 15


def test_parser_accepts_repo_root_flag() -> None:
    parser = build_parser()

    args = parser.parse_args(["--repo-root", "/tmp/project"])

    assert args.repo_root == "/tmp/project"
