from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from cli_coding_agent.storage.models import MessageRecord, SessionRecord
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
            if row is None:
                return None
            return self._row_to_session(row)

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
            rows = cursor.fetchall()
            return [self._row_to_message(row) for row in rows]

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

    def _row_to_session(self, row: tuple[str, str, datetime, datetime]) -> SessionRecord:
        return SessionRecord(
            id=row[0],
            title=row[1],
            created_at=row[2],
            updated_at=row[3],
        )

    def _row_to_message(
        self,
        row: tuple[str, str, str, str, datetime, int],
    ) -> MessageRecord:
        return MessageRecord(
            id=row[0],
            session_id=row[1],
            role=row[2],
            content=row[3],
            created_at=row[4],
            sequence_no=row[5],
        )
