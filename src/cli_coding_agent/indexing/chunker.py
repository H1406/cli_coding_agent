from __future__ import annotations

from cli_coding_agent.indexing.models import RepoChunk, RepoFileCandidate


def chunk_file(
    candidate: RepoFileCandidate,
    *,
    target_tokens: int = 500,
    overlap_tokens: int = 80,
) -> list[RepoChunk]:
    text = candidate.path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        return []

    if candidate.language == "markdown":
        boundaries = _markdown_boundaries(lines, target_tokens)
    else:
        boundaries = _code_boundaries(lines, target_tokens, overlap_tokens)

    chunks: list[RepoChunk] = []
    for chunk_index, (start_idx, end_idx) in enumerate(boundaries):
        chunk_lines = lines[start_idx:end_idx]
        content = "\n".join(chunk_lines).strip()
        if not content:
            continue
        chunks.append(
            RepoChunk(
                file_path=candidate.relative_path,
                chunk_index=chunk_index,
                content=content,
                token_count=_estimate_tokens(content),
                start_line=start_idx + 1,
                end_line=end_idx,
                language=candidate.language,
            )
        )

    return chunks


def _markdown_boundaries(lines: list[str], target_tokens: int) -> list[tuple[int, int]]:
    sections: list[tuple[int, int]] = []
    start_idx = 0

    for idx, line in enumerate(lines):
        if idx == 0:
            continue
        if line.startswith("#") and idx > start_idx:
            sections.append((start_idx, idx))
            start_idx = idx
    sections.append((start_idx, len(lines)))

    boundaries: list[tuple[int, int]] = []
    for start_idx, end_idx in sections:
        section_lines = lines[start_idx:end_idx]
        if _estimate_tokens("\n".join(section_lines)) <= target_tokens:
            boundaries.append((start_idx, end_idx))
            continue
        boundaries.extend(_line_window_boundaries(lines, start_idx, end_idx, target_tokens, 0))

    return boundaries


def _code_boundaries(
    lines: list[str],
    target_tokens: int,
    overlap_tokens: int,
) -> list[tuple[int, int]]:
    boundaries: list[tuple[int, int]] = []
    start_idx = 0

    while start_idx < len(lines):
        current_start = start_idx
        end_idx = _find_chunk_end(lines, start_idx, target_tokens)
        boundaries.append((start_idx, end_idx))
        if end_idx >= len(lines):
            break
        next_start = _find_overlap_start(lines, end_idx, overlap_tokens)
        start_idx = next_start if next_start > current_start else end_idx

    return boundaries


def _line_window_boundaries(
    lines: list[str],
    start_idx: int,
    end_idx: int,
    target_tokens: int,
    overlap_tokens: int,
) -> list[tuple[int, int]]:
    boundaries: list[tuple[int, int]] = []
    cursor = start_idx
    while cursor < end_idx:
        current_start = cursor
        local_end = _find_chunk_end(lines, cursor, target_tokens, hard_stop=end_idx)
        boundaries.append((cursor, local_end))
        if local_end >= end_idx:
            break
        if overlap_tokens > 0:
            next_start = _find_overlap_start(
                lines,
                local_end,
                overlap_tokens,
                min_index=start_idx,
            )
            cursor = next_start if next_start > current_start else local_end
        else:
            cursor = local_end
    return boundaries


def _find_chunk_end(
    lines: list[str],
    start_idx: int,
    target_tokens: int,
    hard_stop: int | None = None,
) -> int:
    token_total = 0
    best_break = start_idx + 1
    limit = hard_stop or len(lines)

    for idx in range(start_idx, limit):
        token_total += _estimate_tokens(lines[idx])
        if _is_preferred_boundary(lines[idx]):
            best_break = idx + 1
        if token_total >= target_tokens:
            return max(best_break, idx + 1 if best_break == start_idx + 1 else best_break)

    return limit


def _find_overlap_start(
    lines: list[str],
    end_idx: int,
    overlap_tokens: int,
    *,
    min_index: int = 0,
) -> int:
    token_total = 0
    start_idx = end_idx
    for idx in range(end_idx - 1, min_index - 1, -1):
        token_total += _estimate_tokens(lines[idx])
        start_idx = idx
        if token_total >= overlap_tokens:
            return start_idx
    return min_index


def _estimate_tokens(text: str) -> int:
    words = text.split()
    if not words:
        return 0
    return max(1, int(len(words) * 1.3))


def _is_preferred_boundary(line: str) -> bool:
    stripped = line.strip()
    return (
        stripped == ""
        or stripped.startswith(("def ", "class ", "async def ", "# ", "## ", "### "))
    )
