from __future__ import annotations

from pathlib import Path

from cli_coding_agent.agent import CodingAgent
from cli_coding_agent.config import AgentConfig
from cli_coding_agent.tools import ToolRegistry
from tests.support import InMemoryConversationStore


def test_tool_registry_read_write_and_run(tmp_path: Path) -> None:
    registry = ToolRegistry.default(tmp_path, run_timeout_seconds=5)
    file_path = tmp_path / "example.txt"
    file_path.write_text("hello", encoding="utf-8")

    read_result = registry.run("READ", {"path": "example.txt"})
    write_result = registry.run("WRITE", {"path": "nested/out.txt", "content": "updated"})
    run_result = registry.run("RUN", {"cmd": "pwd"})

    assert read_result.ok is True
    assert read_result.output == "hello"
    assert write_result.ok is True
    assert (tmp_path / "nested" / "out.txt").read_text(encoding="utf-8") == "updated"
    assert run_result.ok is True
    assert str(tmp_path) in run_result.output


def test_agent_run_turn_executes_action_before_final(tmp_path: Path, monkeypatch) -> None:
    store = InMemoryConversationStore()
    instruction_path = tmp_path / "instruction.txt"
    instruction_path.write_text("Use tools when needed.", encoding="utf-8")
    (tmp_path / "data.txt").write_text("important context", encoding="utf-8")

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

    responses = iter(
        [
            '{"action": {"tool": "READ", "input": {"path": "data.txt"}}}',
            '{"final": "I read the file and found the important context."}',
        ]
    )
    monkeypatch.setattr(agent, "llm", lambda prompt: next(responses))

    result = agent.run_turn("Inspect the file before answering")

    assert result == "I read the file and found the important context."
    assert [message.role for message in store.messages] == [
        "user",
        "assistant_action",
        "tool",
        "assistant",
    ]
    assert "important context" in store.messages[2].content
