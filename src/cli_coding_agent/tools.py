from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


class ToolResult:
    def __init__(self, ok: bool, tool: str, output: str) -> None:
        self.ok = ok
        self.tool = tool
        self.output = output

    def as_prompt_block(self) -> str:
        status = "ok" if self.ok else "error"
        return f"Tool `{self.tool}` result ({status}):\n{self.output}"


class Tool:
    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

    def run(self, tool_input: dict[str, Any]) -> ToolResult:
        raise NotImplementedError


class ReadTool(Tool):
    def __init__(self, repo_root: Path) -> None:
        super().__init__(name="READ", description="Read a UTF-8 text file from the repo")
        self.repo_root = repo_root

    def run(self, tool_input: dict[str, Any]) -> ToolResult:
        path_value = tool_input.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            return ToolResult(False, self.name, "Missing required string input: path")

        try:
            path = resolve_repo_path(self.repo_root, path_value)
        except ValueError as exc:
            return ToolResult(False, self.name, str(exc))

        if not path.exists():
            return ToolResult(False, self.name, f"File does not exist: {path_value}")
        if not path.is_file():
            return ToolResult(False, self.name, f"Path is not a file: {path_value}")

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(False, self.name, f"File is not UTF-8 text: {path_value}")

        return ToolResult(True, self.name, content)


class WriteTool(Tool):
    def __init__(self, repo_root: Path) -> None:
        super().__init__(name="WRITE", description="Write full UTF-8 text content to a file in the repo")
        self.repo_root = repo_root

    def run(self, tool_input: dict[str, Any]) -> ToolResult:
        path_value = tool_input.get("path")
        content = tool_input.get("content")
        if not isinstance(path_value, str) or not path_value.strip():
            return ToolResult(False, self.name, "Missing required string input: path")
        if not isinstance(content, str):
            return ToolResult(False, self.name, "Missing required string input: content")

        try:
            path = resolve_repo_path(self.repo_root, path_value)
        except ValueError as exc:
            return ToolResult(False, self.name, str(exc))

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolResult(True, self.name, f"Wrote {len(content)} characters to {path_value}")


class RunTool(Tool):
    def __init__(self, repo_root: Path, timeout_seconds: int = 20) -> None:
        super().__init__(name="RUN", description="Run a shell command from the repo root")
        self.repo_root = repo_root
        self.timeout_seconds = timeout_seconds

    def run(self, tool_input: dict[str, Any]) -> ToolResult:
        command = tool_input.get("cmd")
        if not isinstance(command, str) or not command.strip():
            return ToolResult(False, self.name, "Missing required string input: cmd")

        try:
            completed = subprocess.run(
                command,
                cwd=self.repo_root,
                shell=True,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, self.name, f"Command timed out after {self.timeout_seconds}s")

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        payload = {
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
        ok = completed.returncode == 0
        return ToolResult(ok, self.name, json.dumps(payload, indent=2))


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    @classmethod
    def default(cls, repo_root: str | Path, *, run_timeout_seconds: int = 20) -> "ToolRegistry":
        root = Path(repo_root).resolve()
        return cls(
            [
                ReadTool(root),
                WriteTool(root),
                RunTool(root, timeout_seconds=run_timeout_seconds),
            ]
        )

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def describe_tools(self) -> str:
        return "\n".join(f"- {tool.name}: {tool.description}" for tool in self._tools.values())

    def run(self, name: str, tool_input: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(False, name, f"Unknown tool: {name}")
        return tool.run(tool_input)


def resolve_repo_path(repo_root: Path, path_value: str) -> Path:
    candidate = Path(path_value).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"Refusing to access path outside repo root: {path_value}") from exc
    return resolved
