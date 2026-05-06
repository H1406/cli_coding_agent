from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from cli_coding_agent.embeddings import EmbeddingClient
from cli_coding_agent.indexing import chunk_file, scan_repository
from cli_coding_agent.storage import ConversationStore


@dataclass(frozen=True, slots=True)
class RepoSnapshot:
    branch: str | None
    commit_sha: str | None


@dataclass(frozen=True, slots=True)
class IndexingResult:
    repo_id: str
    files_scanned: int
    files_changed: int
    files_deleted: int
    chunks_written: int
    branch: str | None
    commit_sha: str | None


class RepositoryIndexingService:
    def __init__(self, store: ConversationStore, embedder: EmbeddingClient) -> None:
        self.store = store
        self.embedder = embedder

    def index_repository(self, repo_root: str | Path, repo_id: str) -> IndexingResult:
        root_path = Path(repo_root).resolve()
        previous_files = {record.path: record.content_hash for record in self.store.list_repo_files(repo_id)}
        candidates = scan_repository(root_path, previous_files)
        changed_candidates = [candidate for candidate in candidates if candidate.changed]
        current_paths = {candidate.relative_path for candidate in candidates}
        deleted_paths = sorted(set(previous_files) - current_paths)
        snapshot = _git_snapshot(root_path)

        index_run = self.store.create_repo_index_run(
            repo_id=repo_id,
            branch=snapshot.branch,
            commit_sha=snapshot.commit_sha,
        )
        chunks_written = 0

        try:
            for deleted_path in deleted_paths:
                self.store.delete_repo_file(repo_id, deleted_path)

            for candidate in changed_candidates:
                repo_file = self.store.upsert_repo_file(repo_id, candidate)
                chunks = chunk_file(candidate)
                chunk_records = self.store.replace_repo_file_chunks(
                    repo_file_id=repo_file.id,
                    file_path=candidate.relative_path,
                    language=candidate.language,
                    chunks=chunks,
                )
                vectors = self.embedder.embed_many_texts([chunk.content for chunk in chunk_records])
                if chunk_records:
                    self.store.replace_chunk_embeddings(
                        chunk_records,
                        vectors,
                        embedding_model=self.embedder.model_name,
                        embedding_version=self.embedder.version,
                    )
                chunks_written += len(chunk_records)
        finally:
            self.store.complete_repo_index_run(index_run.id)

        return IndexingResult(
            repo_id=repo_id,
            files_scanned=len(candidates),
            files_changed=len(changed_candidates),
            files_deleted=len(deleted_paths),
            chunks_written=chunks_written,
            branch=snapshot.branch,
            commit_sha=snapshot.commit_sha,
        )


def _git_snapshot(repo_root: Path) -> RepoSnapshot:
    branch = _git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    commit_sha = _git_output(["git", "rev-parse", "HEAD"], repo_root)
    if branch == "HEAD":
        branch = None
    return RepoSnapshot(branch=branch, commit_sha=commit_sha)


def _git_output(command: list[str], cwd: Path) -> str | None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None
