from __future__ import annotations

from pathlib import Path


def load_instructions(path: str) -> str:
    instruction_file = Path(path)
    if not instruction_file.exists():
        return ""
    return instruction_file.read_text(encoding="utf-8").strip()
