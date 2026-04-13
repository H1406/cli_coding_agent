from __future__ import annotations

from cli_coding_agent.config import AgentConfig
from cli_coding_agent.prompts import load_instructions
from cli_coding_agent.tools import ToolRegistry
import subprocess


class CodingAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.instructions = load_instructions(config.instructions_path)
        self.tools = ToolRegistry.default()

    def llm(self, prompt: str) -> str:
        result = subprocess.run(
            ["ollama", "run", self.config.model],
            input=prompt,
            capture_output=True,
            text=True
        )

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("RETURN CODE:", result.returncode)

        return result.stdout

    def run(self, user_prompt: str) -> str:
        prompt = user_prompt.strip()
        if not prompt:
            return "No prompt provided."

        tool_names = ", ".join(self.tools.names())
        return (
            f"[{self.config.agent_name}] ready.\n"
            f"Model: {self.config.model}\n"
            f"Instructions loaded: {bool(self.instructions)}\n"
            f"Available tools: {tool_names}\n\n"
            f"User prompt:\n{prompt}\n\n"
            f"Agent response:\n{self.llm(user_prompt)}\n"
        )
