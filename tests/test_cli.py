from cli_coding_agent.agent import CodingAgent
from cli_coding_agent.config import AgentConfig


def test_agent_runs_with_prompt(tmp_path) -> None:
    instruction_path = tmp_path / "instruction.txt"
    instruction_path.write_text("You are a coding agent.", encoding="utf-8")

    agent = CodingAgent(
        AgentConfig(
            agent_name="test-agent",
            model="test-model",
            instructions_path=str(instruction_path),
        )
    )

    result = agent.run("Inspect the repository")

    assert "test-agent" in result
    assert "Inspect the repository" in result
