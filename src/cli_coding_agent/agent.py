from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone
from typing import Any

from cli_coding_agent.config import AgentConfig
from cli_coding_agent.embeddings import HashEmbeddingClient
from cli_coding_agent.indexing.service import RepositoryIndexingService
from cli_coding_agent.memory import (
    ConversationMemoryService,
    format_recent_messages,
    trim_message_window,
    unique_memories,
)
from cli_coding_agent.prompts import load_instructions
from cli_coding_agent.retrieval import (
    ConversationRetriever,
    RepoRetriever,
    RetrievedConversationMemory,
    RetrievedRepoChunk,
)
from cli_coding_agent.storage import ConversationStore, ConversationSummaryRecord, MessageRecord
from cli_coding_agent.tools import ToolRegistry


class CodingAgent:
    def __init__(self, config: AgentConfig, store: ConversationStore) -> None:
        self.config = config
        self.store = store
        self.instructions = load_instructions(config.instructions_path)
        self.tools = ToolRegistry.default()
        self.embedder = HashEmbeddingClient()
        self.indexing_service = RepositoryIndexingService(store, self.embedder)
        self.repo_retriever = RepoRetriever(store, self.embedder)
        self.conversation_retriever = ConversationRetriever(store, self.embedder)
        self.memory_service = ConversationMemoryService(
            store,
            self.embedder,
            summary_trigger_messages=config.summary_trigger_messages,
            recent_message_limit=config.recent_message_limit,
        )
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

    def build_prompt(
        self,
        user_prompt: str,
        recent_messages: list[MessageRecord] | None = None,
        summary: ConversationSummaryRecord | None = None,
        conversation_memories: list[RetrievedConversationMemory] | None = None,
        repo_chunks: list[RetrievedRepoChunk] | None = None,
    ) -> str:
        prompt = user_prompt.strip()
        recent_messages = recent_messages or []
        conversation_memories = conversation_memories or []
        repo_chunks = repo_chunks or []
        sections = self._build_context_sections(
            summary=summary,
            recent_messages=recent_messages,
            conversation_memories=conversation_memories,
            repo_chunks=repo_chunks,
        )
        context_block = self._fit_sections_to_budget(sections, self.config.prompt_token_budget)
        if not self.instructions:
            if not context_block:
                return prompt
            return f"{context_block}\n\nUser request:\n{prompt}\n\nAssistant response:"
        return (
            f"{self.instructions}\n\n"
            f"{context_block}"
            "User request:\n"
            f"{prompt}\n\n"
            "Assistant response:"
        )

    def _build_context_sections(
        self,
        *,
        summary: ConversationSummaryRecord | None,
        recent_messages: list[MessageRecord],
        conversation_memories: list[RetrievedConversationMemory],
        repo_chunks: list[RetrievedRepoChunk],
    ) -> list[str]:
        sections: list[str] = []
        if summary is not None:
            sections.append(f"Conversation summary:\n{summary.summary_text}")

        recent_block = format_recent_messages(recent_messages)
        if recent_block:
            sections.append(recent_block)

        memory_block = self._format_conversation_memories(conversation_memories)
        if memory_block:
            sections.append(memory_block)

        repo_block = self._format_repo_context(repo_chunks)
        if repo_block:
            sections.append(repo_block)

        return sections

    def _fit_sections_to_budget(self, sections: list[str], token_budget: int) -> str:
        included: list[str] = []
        used = 0
        for section in sections:
            estimate = self._estimate_tokens(section)
            if included and used + estimate > token_budget:
                continue
            included.append(section)
            used += estimate
        if not included:
            return ""
        return "\n\n".join(included) + "\n\n"

    def _estimate_tokens(self, text: str) -> int:
        words = text.split()
        if not words:
            return 0
        return max(1, int(len(words) * 1.3))

    def _format_conversation_memories(
        self,
        conversation_memories: list[RetrievedConversationMemory],
    ) -> str:
        if not conversation_memories:
            return ""

        sections = ["Retrieved conversation memory:"]
        for item in conversation_memories:
            label = item.memory.source_type
            role = f" ({item.memory.role})" if item.memory.role else ""
            sections.append(f"Source: {label}{role}")
            sections.append(item.memory.content)
            sections.append("")
        return "\n".join(sections)

    def _format_repo_context(self, repo_chunks: list[RetrievedRepoChunk]) -> str:
        if not repo_chunks:
            return ""

        sections = ["Retrieved repo context:"]
        for item in repo_chunks:
            sections.append(
                f"Source: {item.chunk.file_path} lines {item.chunk.start_line}-{item.chunk.end_line}"
            )
            sections.append(item.chunk.content)
            sections.append("")
        return "\n".join(sections)

    def _retrieve_repo_context(self, prompt: str) -> list[RetrievedRepoChunk]:
        self.indexing_service.index_repository(self.config.repo_root, self.config.repo_id)
        return self.repo_retriever.search(
            self.config.repo_id,
            prompt,
            limit=self.config.repo_context_limit,
        )

    def _retrieve_conversation_context(
        self,
        session_id: str,
        prompt: str,
    ) -> tuple[ConversationSummaryRecord | None, list[MessageRecord], list[RetrievedConversationMemory]]:
        summary = self.store.get_latest_conversation_summary(session_id)
        recent_messages = trim_message_window(
            self.store.list_messages(session_id),
            limit=self.config.recent_message_limit,
        )
        memories = unique_memories(
            [
                item.memory
                for item in self.conversation_retriever.search(
                    session_id,
                    prompt,
                    limit=self.config.conversation_context_limit,
                )
            ]
        )
        retrieved = [
            RetrievedConversationMemory(memory=memory, score=0.0)
            for memory in memories
        ]
        return summary, recent_messages, retrieved

    def _log_retrieval_event(
        self,
        *,
        session_id: str,
        prompt: str,
        conversation_memories: list[RetrievedConversationMemory],
        repo_chunks: list[RetrievedRepoChunk],
    ) -> None:
        payload = {
            "conversation_memories": [
                {
                    "source_type": item.memory.source_type,
                    "source_id": item.memory.source_id,
                }
                for item in conversation_memories
            ],
            "repo_chunks": [
                {
                    "file_path": item.chunk.file_path,
                    "chunk_index": item.chunk.chunk_index,
                }
                for item in repo_chunks
            ],
        }
        self.store.create_retrieval_event(
            event_type="prompt_context",
            query_text=prompt,
            payload_json=json.dumps(payload),
            session_id=session_id,
            repo_id=self.config.repo_id,
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

    def ensure_session(self) -> str:
        return self._ensure_session()

    def _default_session_title(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return f"{self.config.agent_name} session {timestamp}"

    def run(self, user_prompt: str) -> str:
        prompt = user_prompt.strip()
        if not prompt:
            return "No prompt provided."

        result = self.run_turn(prompt)
        tool_names = ", ".join(self.tools.names())
        return (
            f"[{self.config.agent_name}] ready.\n"
            f"Model: {self.config.model}\n"
            f"Session ID: {self.config.session_id}\n"
            f"Indexed repo: {self.config.repo_id}\n"
            f"Instructions loaded: {bool(self.instructions)}\n"
            f"Available tools: {tool_names}\n\n"
            f"User prompt:\n{prompt}\n\n"
            f"Agent response:\n{result}\n"
        )

    def run_turn(self, user_prompt: str) -> str:
        prompt = user_prompt.strip()
        if not prompt:
            return "No prompt provided."

        session_id = self._ensure_session()
        self.store.append_message(session_id=session_id, role="user", content=prompt)
        self.memory_service.refresh_session_memory(session_id)
        summary, recent_messages, conversation_memories = self._retrieve_conversation_context(
            session_id,
            prompt,
        )
        repo_chunks = self._retrieve_repo_context(prompt)
        self._log_retrieval_event(
            session_id=session_id,
            prompt=prompt,
            conversation_memories=conversation_memories,
            repo_chunks=repo_chunks,
        )
        full_prompt = self.build_prompt(
            prompt,
            recent_messages=recent_messages,
            summary=summary,
            conversation_memories=conversation_memories,
            repo_chunks=repo_chunks,
        )
        response = self.llm(full_prompt)
        self.store.append_message(session_id=session_id, role="assistant", content=response)
        self.memory_service.refresh_session_memory(session_id)
        return response
