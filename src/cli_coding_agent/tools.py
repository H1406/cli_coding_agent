from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = tools

    @classmethod
    def default(cls) -> "ToolRegistry":
        return cls(
            [
                Tool(name="READ", description="Read a file"),
                Tool(name="WRITE", description="Write a file"),
                Tool(name="RUN", description="Run a shell command"),
            ]
        )

    def names(self) -> list[str]:
        return [tool.name for tool in self._tools]
