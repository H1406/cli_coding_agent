from __future__ import annotations

from pathlib import Path


def resolve_instruction_path(path: str) -> Path:
    instruction_file = Path(path).expanduser()
    if instruction_file.is_absolute():
        return instruction_file

    cwd_candidate = Path.cwd() / instruction_file
    if cwd_candidate.exists():
        return cwd_candidate

    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / instruction_file


def load_instructions(path: str) -> str:
    instruction_file = resolve_instruction_path(path)
    if not instruction_file.exists():
        return ""
    return instruction_file.read_text(encoding="utf-8").strip()
