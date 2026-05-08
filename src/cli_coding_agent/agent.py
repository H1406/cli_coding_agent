from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
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
from cli_coding_agent.tools import ToolRegistry, ToolResult


@dataclass(frozen=True, slots=True)
class AgentAction:
    tool: str
    tool_input: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ParsedAgentOutput:
    final: str | None = None
    action: AgentAction | None = None
    raw_text: str = ""


class CodingAgent:
    def __init__(self, config: AgentConfig, store: ConversationStore) -> None:
        self.config = config
        self.store = store
        self.instructions = load_instructions(config.instructions_path)
        self.tools = ToolRegistry.default(
            config.repo_root,
            run_timeout_seconds=config.tool_run_timeout_seconds,
        )
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
        tool_history: list[str] | None = None,
    ) -> str:
        prompt = user_prompt.strip()
        recent_messages = recent_messages or []
        conversation_memories = conversation_memories or []
        repo_chunks = repo_chunks or []
        tool_history = tool_history or []
        sections = self._build_context_sections(
            summary=summary,
            recent_messages=recent_messages,
            conversation_memories=conversation_memories,
            repo_chunks=repo_chunks,
            tool_history=tool_history,
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
        tool_history: list[str],
    ) -> list[str]:
        sections: list[str] = []
        sections.append(self._tool_protocol())
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

        if tool_history:
            sections.append("Tool interaction history:\n" + "\n\n".join(tool_history))

        return sections

    def _tool_protocol(self) -> str:
        return (
            "Tool protocol:\n"
            "You may either return a final answer or a single JSON action.\n"
            "Final answer format:\n"
            '{"final": "your answer"}\n'
            "Action format:\n"
            '{"action": {"tool": "READ|WRITE|RUN", "input": {...}}}\n'
            "Only return JSON when choosing an action. Use a final answer when you are done."
        )

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
        outputs = self.model.generate(**inputs, max_new_tokens=300)
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
        response = self._run_action_loop(
            session_id=session_id,
            user_prompt=prompt,
            summary=summary,
            recent_messages=recent_messages,
            conversation_memories=conversation_memories,
            repo_chunks=repo_chunks,
        )
        self.memory_service.refresh_session_memory(session_id)
        return response

    def _run_action_loop(
        self,
        *,
        session_id: str,
        user_prompt: str,
        summary: ConversationSummaryRecord | None,
        recent_messages: list[MessageRecord],
        conversation_memories: list[RetrievedConversationMemory],
        repo_chunks: list[RetrievedRepoChunk],
    ) -> str:
        tool_history: list[str] = []

        for _ in range(self.config.max_action_steps):
            prompt = self.build_prompt(
                user_prompt,
                recent_messages=recent_messages,
                summary=summary,
                conversation_memories=conversation_memories,
                repo_chunks=repo_chunks,
                tool_history=tool_history,
            )
            raw_output = self.llm(prompt)
            parsed = self._parse_agent_output(raw_output)

            if parsed.final is not None:
                self.store.append_message(session_id=session_id, role="assistant", content=parsed.final)
                return parsed.final

            if parsed.action is None:
                self.store.append_message(session_id=session_id, role="assistant", content=parsed.raw_text)
                return parsed.raw_text

            self.store.append_message(
                session_id=session_id,
                role="assistant_action",
                content=json.dumps(
                    {
                        "tool": parsed.action.tool,
                        "input": parsed.action.tool_input,
                    }
                ),
            )
            result = self.tools.run(parsed.action.tool, parsed.action.tool_input)
            self.store.append_message(session_id=session_id, role="tool", content=result.output)
            tool_history.append(self._format_tool_exchange(parsed.action, result))
            self.memory_service.refresh_session_memory(session_id)
            summary, recent_messages, conversation_memories = self._retrieve_conversation_context(
            session_id,
            user_prompt,
        )
        repo_chunks = self._retrieve_repo_context(user_prompt) 

        fallback = "Stopped after reaching the maximum number of action steps."
        self.store.append_message(session_id=session_id, role="assistant", content=fallback)
        return fallback

    def _parse_agent_output(self, raw_output: str) -> ParsedAgentOutput:
        text = raw_output.strip()
        payload = self._extract_json_object(text)
        if payload is None:
            return ParsedAgentOutput(final=text, raw_text=text)

        if isinstance(payload.get("final"), str):
            return ParsedAgentOutput(final=payload["final"].strip(), raw_text=text)

        action_payload = payload.get("action")
        if isinstance(action_payload, dict):
            tool = action_payload.get("tool")
            tool_input = action_payload.get("input", {})
            if isinstance(tool, str) and isinstance(tool_input, dict):
                return ParsedAgentOutput(
                    action=AgentAction(tool=tool.strip().upper(), tool_input=tool_input),
                    raw_text=text,
                )

        return ParsedAgentOutput(final=text, raw_text=text)

    def _extract_json_object(self, text: str) -> dict[str, Any] | None:
        if not text:
            return None
        candidates = [text]
        if "```" in text:
            for block in text.split("```"):
                stripped = block.strip()
                if stripped.startswith("json"):
                    stripped = stripped[4:].strip()
                candidates.append(stripped)

        for candidate in candidates:
            if not candidate.startswith("{"):
                continue
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    def _format_tool_exchange(self, action: AgentAction, result: ToolResult) -> str:
        status = "ok" if result.ok else "error"
        return (
            f"Action: {action.tool} {json.dumps(action.tool_input, ensure_ascii=True)}\n"
            f"Observation ({status}):\n{result.output}"
        )
