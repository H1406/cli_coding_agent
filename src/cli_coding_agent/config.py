from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class AgentConfig:
    agent_name: str
    model: str
    instructions_path: str
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, instructions_path: str | None = None) -> "AgentConfig":
        return cls(
            agent_name=os.getenv("AGENT_NAME", "local-coding-agent"),
            model=os.getenv("AGENT_MODEL", "deepseek-coder:6.7b-instruct"),
            instructions_path=instructions_path
            or os.getenv("AGENT_INSTRUCTIONS_PATH", "instruction.txt"),
            log_level=os.getenv("AGENT_LOG_LEVEL", "INFO"),
        )
