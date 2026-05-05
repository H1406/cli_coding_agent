from __future__ import annotations

import hashlib
from pathlib import Path

from cli_coding_agent.indexing import changed_files, chunk_file, scan_repository


def test_scan_repository_skips_ignored_dirs_and_binary_files(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "src").mkdir()

    (tmp_path / ".git" / "config").write_text("ignored", encoding="utf-8")
    (tmp_path / "node_modules" / "lib.js").write_text("ignored", encoding="utf-8")
    (tmp_path / "src" / "app.py").write_text("print('hello')\n", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "notes.md").write_text("# Title\n\nbody\n", encoding="utf-8")

    candidates = scan_repository(tmp_path)

    assert [candidate.relative_path for candidate in candidates] == ["notes.md", "src/app.py"]


def test_changed_files_only_returns_modified_content(tmp_path: Path) -> None:
    file_path = tmp_path / "main.py"
    file_path.write_text("print('hello')\n", encoding="utf-8")
    initial_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()

    unchanged = changed_files(tmp_path, {"main.py": initial_hash})
    assert unchanged == []

    file_path.write_text("print('updated')\n", encoding="utf-8")
    updated = changed_files(tmp_path, {"main.py": initial_hash})

    assert len(updated) == 1
    assert updated[0].relative_path == "main.py"
    assert updated[0].changed is True


def test_chunk_markdown_by_heading_with_line_metadata(tmp_path: Path) -> None:
    doc = tmp_path / "guide.md"
    doc.write_text(
        "# Intro\n\nOne two three.\n\n## Details\n\nFour five six.\n",
        encoding="utf-8",
    )

    candidate = scan_repository(tmp_path)[0]
    chunks = chunk_file(candidate, target_tokens=20, overlap_tokens=0)

    assert len(chunks) == 2
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 4
    assert chunks[1].start_line == 5
    assert chunks[1].end_line == 7
    assert chunks[1].content.startswith("## Details")


def test_chunk_code_uses_overlap_and_preserves_path(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text(
        "\n".join(
            [
                "def alpha():",
                "    first = 1",
                "    second = 2",
                "    return first + second",
                "",
                "def beta():",
                "    third = 3",
                "    fourth = 4",
                "    return third + fourth",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    candidate = scan_repository(tmp_path)[0]
    chunks = chunk_file(candidate, target_tokens=8, overlap_tokens=3)

    assert len(chunks) >= 2
    assert chunks[0].file_path == "module.py"
    assert chunks[0].start_line == 1
    assert chunks[1].start_line <= chunks[0].end_line
    assert "def beta():" in chunks[-1].content
