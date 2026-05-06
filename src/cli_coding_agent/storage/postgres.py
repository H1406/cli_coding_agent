from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

from cli_coding_agent.indexing.models import RepoChunk, RepoFileCandidate
from cli_coding_agent.storage.models import (
    ConversationMemoryRecord,
    ConversationSummaryRecord,
    MessageRecord,
    RepoChunkEmbeddingRecord,
    RepoChunkRecord,
    RepoFileRecord,
    RepoIndexRunRecord,
    RetrievalEventRecord,
    SessionRecord,
)
from cli_coding_agent.storage.store import ConversationStore


class PostgresConversationStore(ConversationStore):
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._schema_initialized = False

    def _import_psycopg(self) -> Any:
        try:
            import psycopg
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing required dependency 'psycopg'. Install PostgreSQL runtime "
                "dependencies before using conversation storage."
            ) from exc
        return psycopg

    def _connect(self) -> Any:
        psycopg = self._import_psycopg()
        connection = psycopg.connect(self.database_url)
        self._ensure_schema(connection)
        return connection

    def _ensure_schema(self, connection: Any) -> None:
        if self._schema_initialized:
            return

        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    sequence_no INTEGER NOT NULL,
                    CONSTRAINT messages_session_sequence_unique UNIQUE (session_id, sequence_no)
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS messages_session_sequence_idx
                ON messages (session_id, sequence_no)
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    summary_text TEXT NOT NULL,
                    up_to_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                    up_to_sequence_no INTEGER NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT conversation_summaries_session_unique UNIQUE (session_id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_memories (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    role TEXT,
                    content TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_version TEXT NOT NULL,
                    vector_json JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT conversation_memories_source_unique
                    UNIQUE (session_id, source_type, source_id, embedding_model, embedding_version)
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS conversation_memories_session_idx
                ON conversation_memories (session_id, source_type, created_at DESC)
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS retrieval_events (
                    id TEXT PRIMARY KEY,
                    session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
                    repo_id TEXT,
                    query_text TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS repo_index_runs (
                    id TEXT PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    branch TEXT,
                    commit_sha TEXT,
                    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    completed_at TIMESTAMPTZ
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS repo_index_runs_repo_started_idx
                ON repo_index_runs (repo_id, started_at DESC)
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS repo_files (
                    id TEXT PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    language TEXT NOT NULL,
                    size_bytes BIGINT NOT NULL,
                    indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT repo_files_repo_path_unique UNIQUE (repo_id, path)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS repo_chunks (
                    id TEXT PRIMARY KEY,
                    repo_file_id TEXT NOT NULL REFERENCES repo_files(id) ON DELETE CASCADE,
                    file_path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    language TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT repo_chunks_file_chunk_unique UNIQUE (repo_file_id, chunk_index)
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS repo_chunks_file_idx
                ON repo_chunks (repo_file_id, chunk_index)
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS repo_chunk_embeddings (
                    id TEXT PRIMARY KEY,
                    repo_chunk_id TEXT NOT NULL REFERENCES repo_chunks(id) ON DELETE CASCADE,
                    embedding_model TEXT NOT NULL,
                    embedding_version TEXT NOT NULL,
                    vector_json JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT repo_chunk_embeddings_chunk_model_unique
                    UNIQUE (repo_chunk_id, embedding_model, embedding_version)
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS repo_chunk_embeddings_chunk_idx
                ON repo_chunk_embeddings (repo_chunk_id)
                """
            )
        connection.commit()
        self._schema_initialized = True

    def create_session(self, title: str, session_id: str | None = None) -> SessionRecord:
        resolved_session_id = session_id or str(uuid.uuid4())
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO sessions (id, title)
                VALUES (%s, %s)
                RETURNING id, title, created_at, updated_at
                """,
                (resolved_session_id, title),
            )
            row = cursor.fetchone()
            assert row is not None
            return self._row_to_session(row)

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM sessions
                WHERE id = %s
                """,
                (session_id,),
            )
            row = cursor.fetchone()
            return None if row is None else self._row_to_session(row)

    def list_messages(self, session_id: str, limit: int | None = None) -> list[MessageRecord]:
        query = """
            SELECT id, session_id, role, content, created_at, sequence_no
            FROM messages
            WHERE session_id = %s
            ORDER BY sequence_no ASC
        """
        params: tuple[Any, ...] = (session_id,)
        if limit is not None:
            query += " LIMIT %s"
            params = (session_id, limit)

        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(query, params)
            return [self._row_to_message(row) for row in cursor.fetchall()]

    def append_message(self, session_id: str, role: str, content: str) -> MessageRecord:
        message_id = str(uuid.uuid4())
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT id FROM sessions WHERE id = %s FOR UPDATE", (session_id,))
                if cursor.fetchone() is None:
                    raise ValueError(f"Session '{session_id}' does not exist.")

                cursor.execute(
                    """
                    SELECT COALESCE(MAX(sequence_no), 0) + 1
                    FROM messages
                    WHERE session_id = %s
                    """,
                    (session_id,),
                )
                row = cursor.fetchone()
                assert row is not None
                sequence_no = row[0]

                cursor.execute(
                    """
                    INSERT INTO messages (id, session_id, role, content, sequence_no)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id, session_id, role, content, created_at, sequence_no
                    """,
                    (message_id, session_id, role, content, sequence_no),
                )
                inserted = cursor.fetchone()
                assert inserted is not None
                return self._row_to_message(inserted)

    def upsert_conversation_summary(
        self,
        session_id: str,
        summary_text: str,
        up_to_message_id: str,
        up_to_sequence_no: int,
    ) -> ConversationSummaryRecord:
        summary_id = str(uuid.uuid4())
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO conversation_summaries (
                    id,
                    session_id,
                    summary_text,
                    up_to_message_id,
                    up_to_sequence_no
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (session_id) DO UPDATE SET
                    summary_text = EXCLUDED.summary_text,
                    up_to_message_id = EXCLUDED.up_to_message_id,
                    up_to_sequence_no = EXCLUDED.up_to_sequence_no,
                    created_at = NOW()
                RETURNING
                    id,
                    session_id,
                    summary_text,
                    up_to_message_id,
                    up_to_sequence_no,
                    created_at
                """,
                (summary_id, session_id, summary_text, up_to_message_id, up_to_sequence_no),
            )
            row = cursor.fetchone()
            assert row is not None
            return self._row_to_conversation_summary(row)

    def get_latest_conversation_summary(
        self,
        session_id: str,
    ) -> ConversationSummaryRecord | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, session_id, summary_text, up_to_message_id, up_to_sequence_no, created_at
                FROM conversation_summaries
                WHERE session_id = %s
                """,
                (session_id,),
            )
            row = cursor.fetchone()
            return None if row is None else self._row_to_conversation_summary(row)

    def upsert_conversation_memory(
        self,
        session_id: str,
        source_type: str,
        source_id: str,
        role: str | None,
        content: str,
        vector: list[float],
        *,
        embedding_model: str,
        embedding_version: str,
    ) -> ConversationMemoryRecord:
        memory_id = str(uuid.uuid4())
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO conversation_memories (
                    id,
                    session_id,
                    source_type,
                    source_id,
                    role,
                    content,
                    embedding_model,
                    embedding_version,
                    vector_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (session_id, source_type, source_id, embedding_model, embedding_version)
                DO UPDATE SET
                    role = EXCLUDED.role,
                    content = EXCLUDED.content,
                    vector_json = EXCLUDED.vector_json,
                    created_at = NOW()
                RETURNING
                    id,
                    session_id,
                    source_type,
                    source_id,
                    role,
                    content,
                    embedding_model,
                    embedding_version,
                    vector_json,
                    created_at
                """,
                (
                    memory_id,
                    session_id,
                    source_type,
                    source_id,
                    role,
                    content,
                    embedding_model,
                    embedding_version,
                    json.dumps(vector),
                ),
            )
            row = cursor.fetchone()
            assert row is not None
            return self._row_to_conversation_memory(row)

    def list_conversation_memories(self, session_id: str) -> list[ConversationMemoryRecord]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    session_id,
                    source_type,
                    source_id,
                    role,
                    content,
                    embedding_model,
                    embedding_version,
                    vector_json,
                    created_at
                FROM conversation_memories
                WHERE session_id = %s
                ORDER BY created_at DESC
                """,
                (session_id,),
            )
            return [self._row_to_conversation_memory(row) for row in cursor.fetchall()]

    def create_retrieval_event(
        self,
        *,
        event_type: str,
        query_text: str,
        payload_json: str,
        session_id: str | None = None,
        repo_id: str | None = None,
    ) -> RetrievalEventRecord:
        event_id = str(uuid.uuid4())
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO retrieval_events (
                    id,
                    session_id,
                    repo_id,
                    query_text,
                    event_type,
                    payload_json
                )
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                RETURNING
                    id,
                    session_id,
                    repo_id,
                    query_text,
                    event_type,
                    payload_json,
                    created_at
                """,
                (event_id, session_id, repo_id, query_text, event_type, payload_json),
            )
            row = cursor.fetchone()
            assert row is not None
            return self._row_to_retrieval_event(row)

    def list_repo_files(self, repo_id: str) -> list[RepoFileRecord]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, repo_id, path, content_hash, language, size_bytes, indexed_at
                FROM repo_files
                WHERE repo_id = %s
                ORDER BY path ASC
                """,
                (repo_id,),
            )
            return [self._row_to_repo_file(row) for row in cursor.fetchall()]

    def create_repo_index_run(
        self,
        repo_id: str,
        branch: str | None,
        commit_sha: str | None,
    ) -> RepoIndexRunRecord:
        run_id = str(uuid.uuid4())
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO repo_index_runs (id, repo_id, branch, commit_sha)
                VALUES (%s, %s, %s, %s)
                RETURNING id, repo_id, branch, commit_sha, started_at, completed_at
                """,
                (run_id, repo_id, branch, commit_sha),
            )
            row = cursor.fetchone()
            assert row is not None
            return self._row_to_repo_index_run(row)

    def complete_repo_index_run(self, index_run_id: str) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE repo_index_runs
                SET completed_at = NOW()
                WHERE id = %s
                """,
                (index_run_id,),
            )

    def upsert_repo_file(self, repo_id: str, candidate: RepoFileCandidate) -> RepoFileRecord:
        repo_file_id = str(uuid.uuid4())
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO repo_files (id, repo_id, path, content_hash, language, size_bytes)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (repo_id, path) DO UPDATE SET
                    content_hash = EXCLUDED.content_hash,
                    language = EXCLUDED.language,
                    size_bytes = EXCLUDED.size_bytes,
                    indexed_at = NOW()
                RETURNING id, repo_id, path, content_hash, language, size_bytes, indexed_at
                """,
                (
                    repo_file_id,
                    repo_id,
                    candidate.relative_path,
                    candidate.content_hash,
                    candidate.language,
                    candidate.size_bytes,
                ),
            )
            row = cursor.fetchone()
            assert row is not None
            return self._row_to_repo_file(row)

    def replace_repo_file_chunks(
        self,
        repo_file_id: str,
        file_path: str,
        language: str,
        chunks: list[RepoChunk],
    ) -> list[RepoChunkRecord]:
        inserted: list[RepoChunkRecord] = []
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM repo_chunks WHERE repo_file_id = %s", (repo_file_id,))
                for chunk in chunks:
                    chunk_id = str(uuid.uuid4())
                    cursor.execute(
                        """
                        INSERT INTO repo_chunks (
                            id,
                            repo_file_id,
                            file_path,
                            chunk_index,
                            content,
                            token_count,
                            start_line,
                            end_line,
                            language
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING
                            id,
                            repo_file_id,
                            file_path,
                            chunk_index,
                            content,
                            token_count,
                            start_line,
                            end_line,
                            language,
                            created_at
                        """,
                        (
                            chunk_id,
                            repo_file_id,
                            file_path,
                            chunk.chunk_index,
                            chunk.content,
                            chunk.token_count,
                            chunk.start_line,
                            chunk.end_line,
                            language,
                        ),
                    )
                    row = cursor.fetchone()
                    assert row is not None
                    inserted.append(self._row_to_repo_chunk(row))
        return inserted

    def delete_repo_file(self, repo_id: str, path: str) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM repo_files
                WHERE repo_id = %s AND path = %s
                """,
                (repo_id, path),
            )

    def replace_chunk_embeddings(
        self,
        chunk_records: list[RepoChunkRecord],
        vectors: list[list[float]],
        *,
        embedding_model: str,
        embedding_version: str,
    ) -> list[RepoChunkEmbeddingRecord]:
        if len(chunk_records) != len(vectors):
            raise ValueError("Chunk records and vectors must have the same length.")

        inserted: list[RepoChunkEmbeddingRecord] = []
        with self._connect() as connection:
            with connection.cursor() as cursor:
                for chunk_record, vector in zip(chunk_records, vectors, strict=True):
                    cursor.execute(
                        """
                        DELETE FROM repo_chunk_embeddings
                        WHERE repo_chunk_id = %s
                        """,
                        (chunk_record.id,),
                    )
                    embedding_id = str(uuid.uuid4())
                    cursor.execute(
                        """
                        INSERT INTO repo_chunk_embeddings (
                            id,
                            repo_chunk_id,
                            embedding_model,
                            embedding_version,
                            vector_json
                        )
                        VALUES (%s, %s, %s, %s, %s::jsonb)
                        RETURNING
                            id,
                            repo_chunk_id,
                            embedding_model,
                            embedding_version,
                            vector_json,
                            created_at
                        """,
                        (
                            embedding_id,
                            chunk_record.id,
                            embedding_model,
                            embedding_version,
                            json.dumps(vector),
                        ),
                    )
                    row = cursor.fetchone()
                    assert row is not None
                    inserted.append(self._row_to_repo_chunk_embedding(row))
        return inserted

    def list_repo_chunk_embeddings(
        self,
        repo_id: str,
    ) -> list[tuple[RepoChunkRecord, RepoChunkEmbeddingRecord]]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    c.id,
                    c.repo_file_id,
                    c.file_path,
                    c.chunk_index,
                    c.content,
                    c.token_count,
                    c.start_line,
                    c.end_line,
                    c.language,
                    c.created_at,
                    e.id,
                    e.repo_chunk_id,
                    e.embedding_model,
                    e.embedding_version,
                    e.vector_json,
                    e.created_at
                FROM repo_chunks c
                INNER JOIN repo_files f ON f.id = c.repo_file_id
                INNER JOIN repo_chunk_embeddings e ON e.repo_chunk_id = c.id
                WHERE f.repo_id = %s
                ORDER BY c.file_path ASC, c.chunk_index ASC
                """,
                (repo_id,),
            )
            results: list[tuple[RepoChunkRecord, RepoChunkEmbeddingRecord]] = []
            for row in cursor.fetchall():
                results.append(
                    (
                        self._row_to_repo_chunk(row[:10]),
                        self._row_to_repo_chunk_embedding(row[10:]),
                    )
                )
            return results

    def _row_to_session(self, row: tuple[str, str, datetime, datetime]) -> SessionRecord:
        return SessionRecord(id=row[0], title=row[1], created_at=row[2], updated_at=row[3])

    def _row_to_message(self, row: tuple[str, str, str, str, datetime, int]) -> MessageRecord:
        return MessageRecord(
            id=row[0],
            session_id=row[1],
            role=row[2],
            content=row[3],
            created_at=row[4],
            sequence_no=row[5],
        )

    def _row_to_conversation_summary(
        self,
        row: tuple[str, str, str, str, int, datetime],
    ) -> ConversationSummaryRecord:
        return ConversationSummaryRecord(
            id=row[0],
            session_id=row[1],
            summary_text=row[2],
            up_to_message_id=row[3],
            up_to_sequence_no=row[4],
            created_at=row[5],
        )

    def _row_to_conversation_memory(
        self,
        row: tuple[str, str, str, str, str | None, str, str, str, list[float] | str, datetime],
    ) -> ConversationMemoryRecord:
        vector = row[8]
        parsed = json.loads(vector) if isinstance(vector, str) else vector
        return ConversationMemoryRecord(
            id=row[0],
            session_id=row[1],
            source_type=row[2],
            source_id=row[3],
            role=row[4],
            content=row[5],
            embedding_model=row[6],
            embedding_version=row[7],
            vector=[float(value) for value in parsed],
            created_at=row[9],
        )

    def _row_to_retrieval_event(
        self,
        row: tuple[str, str | None, str | None, str, str, str | dict[str, Any], datetime],
    ) -> RetrievalEventRecord:
        payload = row[5]
        if not isinstance(payload, str):
            payload = json.dumps(payload)
        return RetrievalEventRecord(
            id=row[0],
            session_id=row[1],
            repo_id=row[2],
            query_text=row[3],
            event_type=row[4],
            payload_json=payload,
            created_at=row[6],
        )

    def _row_to_repo_index_run(
        self,
        row: tuple[str, str, str | None, str | None, datetime, datetime | None],
    ) -> RepoIndexRunRecord:
        return RepoIndexRunRecord(
            id=row[0],
            repo_id=row[1],
            branch=row[2],
            commit_sha=row[3],
            started_at=row[4],
            completed_at=row[5],
        )

    def _row_to_repo_file(
        self,
        row: tuple[str, str, str, str, str, int, datetime],
    ) -> RepoFileRecord:
        return RepoFileRecord(
            id=row[0],
            repo_id=row[1],
            path=row[2],
            content_hash=row[3],
            language=row[4],
            size_bytes=row[5],
            indexed_at=row[6],
        )

    def _row_to_repo_chunk(
        self,
        row: tuple[str, str, str, int, str, int, int, int, str, datetime],
    ) -> RepoChunkRecord:
        return RepoChunkRecord(
            id=row[0],
            repo_file_id=row[1],
            file_path=row[2],
            chunk_index=row[3],
            content=row[4],
            token_count=row[5],
            start_line=row[6],
            end_line=row[7],
            language=row[8],
            created_at=row[9],
        )

    def _row_to_repo_chunk_embedding(
        self,
        row: tuple[str, str, str, str, list[float] | str, datetime],
    ) -> RepoChunkEmbeddingRecord:
        vector = row[4]
        if isinstance(vector, str):
            parsed = json.loads(vector)
        else:
            parsed = vector
        return RepoChunkEmbeddingRecord(
            id=row[0],
            repo_chunk_id=row[1],
            embedding_model=row[2],
            embedding_version=row[3],
            vector=[float(value) for value in parsed],
            created_at=row[5],
        )
