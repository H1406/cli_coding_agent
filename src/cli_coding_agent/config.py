from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AgentConfig:
    agent_name: str
    provider: str
    model: str
    instructions_path: str
    huggingface_model_id: str
    database_url: str
    repo_root: str
    repo_id: str
    repo_context_limit: int = 3
    conversation_context_limit: int = 3
    recent_message_limit: int = 4
    summary_trigger_messages: int = 6
    prompt_token_budget: int = 1800
    max_action_steps: int = 8
    tool_run_timeout_seconds: int = 20
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
            repo_root=os.getenv("AGENT_REPO_ROOT", str(Path.cwd())),
            repo_id=os.getenv("AGENT_REPO_ID", Path.cwd().resolve().as_posix()),
            repo_context_limit=int(os.getenv("AGENT_REPO_CONTEXT_LIMIT", "3")),
            conversation_context_limit=int(os.getenv("AGENT_CONVERSATION_CONTEXT_LIMIT", "3")),
            recent_message_limit=int(os.getenv("AGENT_RECENT_MESSAGE_LIMIT", "4")),
            summary_trigger_messages=int(os.getenv("AGENT_SUMMARY_TRIGGER_MESSAGES", "6")),
            prompt_token_budget=int(os.getenv("AGENT_PROMPT_TOKEN_BUDGET", "1800")),
            max_action_steps=int(os.getenv("AGENT_MAX_ACTION_STEPS", "8")),
            tool_run_timeout_seconds=int(os.getenv("AGENT_TOOL_RUN_TIMEOUT_SECONDS", "20")),
            session_id=session_id or os.getenv("AGENT_SESSION_ID"),
            log_level=os.getenv("AGENT_LOG_LEVEL", "INFO"),
        )
