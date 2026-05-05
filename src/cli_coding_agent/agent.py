from __future__ import annotations

import importlib
from datetime import datetime, timezone
from typing import Any

from cli_coding_agent.config import AgentConfig
from cli_coding_agent.prompts import load_instructions
from cli_coding_agent.storage import ConversationStore
from cli_coding_agent.tools import ToolRegistry


class CodingAgent:
    def __init__(self, config: AgentConfig, store: ConversationStore) -> None:
        self.config = config
        self.store = store
        self.instructions = load_instructions(config.instructions_path)
        self.tools = ToolRegistry.default()
        self.device = self._detect_device()
        self.model: Any | None = None
        self.tokenizer: Any | None = None

    def _detect_device(self) -> str:
        try:
            torch = self._import_dependency("torch")
        except RuntimeError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _import_dependency(self, module_name: str) -> Any:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Missing required dependency '{module_name}'. "
                "Install the Hugging Face runtime dependencies before running the model."
            ) from exc

    def _load_model(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        torch = self._import_dependency("torch")
        transformers = self._import_dependency("transformers")
        auto_model = transformers.AutoModelForCausalLM
        auto_tokenizer = transformers.AutoTokenizer

        self.model = auto_model.from_pretrained(
            self.config.huggingface_model_id,
            device_map=self.device,
            torch_dtype="auto"
        )
        self.model.eval()
        self.tokenizer = auto_tokenizer.from_pretrained(self.config.huggingface_model_id)

    def build_prompt(self, user_prompt: str) -> str:
        prompt = user_prompt.strip()
        if not self.instructions:
            return prompt
        return (
            f"{self.instructions}\n\n"
            "User request:\n"
            f"{prompt}\n\n"
            "Assistant response:"
        )

    def llm(self, prompt: str) -> str:
        self._load_model()
        assert self.model is not None
        assert self.tokenizer is not None

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device=self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def _ensure_session(self) -> str:
        if self.config.session_id:
            session = self.store.get_session(self.config.session_id)
            if session is None:
                session = self.store.create_session(
                    title=self._default_session_title(),
                    session_id=self.config.session_id,
                )
            return session.id

        session = self.store.create_session(title=self._default_session_title())
        self.config.session_id = session.id
        return session.id

    def _default_session_title(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return f"{self.config.agent_name} session {timestamp}"

    def run(self, user_prompt: str) -> str:
        prompt = user_prompt.strip()
        if not prompt:
            return "No prompt provided."

        session_id = self._ensure_session()
        self.store.append_message(session_id=session_id, role="user", content=prompt)

        tool_names = ", ".join(self.tools.names())
        full_prompt = self.build_prompt(prompt)
        response = self.llm(full_prompt)
        self.store.append_message(session_id=session_id, role="assistant", content=response)
        return (
            f"[{self.config.agent_name}] ready.\n"
            f"Model: {self.config.model}\n"
            f"Session ID: {session_id}\n"
            f"Instructions loaded: {bool(self.instructions)}\n"
            f"Available tools: {tool_names}\n\n"
            f"User prompt:\n{prompt}\n\n"
            f"Agent response:\n{response}\n"
        )
