from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class AgentConfig:
    agent_name: str
    provider: str
    model: str
    instructions_path: str
    huggingface_model_id: str
    database_url: str
    session_id: str | None = None
    log_level: str = "INFO"

    @classmethod
    def from_env(
        cls,
        instructions_path: str | None = None,
        session_id: str | None = None,
    ) -> "AgentConfig":
        return cls(
            agent_name=os.getenv("AGENT_NAME", "local-coding-agent"),
            provider=os.getenv("AGENT_PROVIDER", "huggingface"),
            model=os.getenv("AGENT_MODEL", "Qwen/Qwen2.5-Coder-1.5B"),
            instructions_path=instructions_path
            or os.getenv("AGENT_INSTRUCTIONS_PATH", "instruction.txt"),
            huggingface_model_id=os.getenv(
                "HUGGINGFACE_MODEL_ID", "Qwen/Qwen2.5-Coder-1.5B"
            ),
            database_url=os.getenv(
                "AGENT_DATABASE_URL",
                "postgresql://phanhieu@localhost:5432/coding_agent_db",
            ),
            session_id=session_id or os.getenv("AGENT_SESSION_ID"),
            log_level=os.getenv("AGENT_LOG_LEVEL", "INFO"),
        )
