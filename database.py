"""Async-friendly, thread-safe persistence for the clip pipeline.

The public methods do blocking MySQL work in a worker thread, keeping the
Streamlink/Telegram event loop responsive. Connections are short lived, which
also makes the repository safe for Streamlit's independent process.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import mysql.connector
from mysql.connector import Error

from config import Settings

ClipStatus = Literal[
    "UPLOADED", "PROCESSING", "CANCEL_REQUESTED", "PENDING_APPROVAL", "APPROVED_QUEUED",
    "PUBLISHING", "PUBLISHED", "PUBLISH_FAILED", "DISCARDED"
]
ClipSource = Literal["TWITCH", "PS_APP"]


@dataclass(frozen=True, slots=True)
class ClipRecord:
    id: int
    filename: str
    status: str
    source: str
    timestamp: datetime
    telegram_message_id: str | None
    processing_progress: float


@dataclass(frozen=True, slots=True)
class PublicationAttemptRecord:
    id: int
    clip_id: int
    platform: str
    attempt_number: int
    status: str
    detail: str | None
    started_at: datetime
    finished_at: datetime | None


class ClipRepository:
    """Repository whose schema is initialized explicitly at application start."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _connect(self, include_database: bool = True):
        options = {
            "host": self.settings.mysql_host,
            "port": self.settings.mysql_port,
            "user": self.settings.mysql_user,
            "password": self.settings.mysql_password,
            "autocommit": True,
        }
        if include_database:
            options["database"] = self.settings.mysql_database
        return mysql.connector.connect(**options)

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize)

    def _initialize(self) -> None:
        # Creating the DB separately lets a freshly installed local MySQL work.
        connection = self._connect(include_database=False)
        try:
            cursor = connection.cursor()
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{self.settings.mysql_database.replace('`', '')}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            cursor.close()
        finally:
            connection.close()
        connection = self._connect()
        try:
            cursor = connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS clips (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    filename VARCHAR(512) NOT NULL,
                    status ENUM('UPLOADED','PROCESSING','CANCEL_REQUESTED','PENDING_APPROVAL','APPROVED_QUEUED','PUBLISHING','PUBLISHED','PUBLISH_FAILED','DISCARDED') NOT NULL,
                    source ENUM('TWITCH','PS_APP') NOT NULL,
                    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    telegram_message_id VARCHAR(64) NULL,
                    processing_progress DECIMAL(5,2) NOT NULL DEFAULT 0,
                    INDEX idx_clips_status_timestamp (status, timestamp)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            # CREATE TABLE does not widen an enum on existing installations.
            cursor.execute("SHOW COLUMNS FROM clips LIKE 'status'")
            status_column = cursor.fetchone()
            if status_column and "CANCEL_REQUESTED" not in str(status_column[1]):
                cursor.execute("""
                    ALTER TABLE clips MODIFY COLUMN status
                    ENUM('UPLOADED','PROCESSING','CANCEL_REQUESTED','PENDING_APPROVAL','APPROVED_QUEUED','PUBLISHING','PUBLISHED','PUBLISH_FAILED','DISCARDED') NOT NULL
                """)
            cursor.execute("SHOW COLUMNS FROM clips LIKE 'processing_progress'")
            if cursor.fetchone() is None:
                cursor.execute(
                    "ALTER TABLE clips ADD COLUMN processing_progress "
                    "DECIMAL(5,2) NOT NULL DEFAULT 0 AFTER telegram_message_id"
                )
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS publication_attempts (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    clip_id INT NOT NULL,
                    platform VARCHAR(64) NOT NULL,
                    attempt_number INT NOT NULL,
                    status VARCHAR(32) NOT NULL,
                    detail TEXT NULL,
                    started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    finished_at DATETIME NULL,
                    UNIQUE KEY uq_publication_attempt (clip_id, platform, attempt_number),
                    INDEX idx_publication_clip_started (clip_id, started_at),
                    CONSTRAINT fk_publication_clip FOREIGN KEY (clip_id) REFERENCES clips(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cursor.close()
        finally:
            connection.close()

    async def create_clip(self, filename: str, status: ClipStatus, source: ClipSource) -> int:
        return await asyncio.to_thread(self._create_clip, filename, status, source)

    def _create_clip(self, filename: str, status: str, source: str) -> int:
        connection = self._connect()
        try:
            cursor = connection.cursor()
            cursor.execute(
                "INSERT INTO clips (filename, status, source) VALUES (%s, %s, %s)",
                (filename, status, source),
            )
            clip_id = int(cursor.lastrowid)
            cursor.close()
            return clip_id
        finally:
            connection.close()

    async def update_clip(
        self, clip_id: int, *, status: ClipStatus | None = None, filename: str | None = None,
        telegram_message_id: str | None = None,
    ) -> None:
        await asyncio.to_thread(self._update_clip, clip_id, status, filename, telegram_message_id)

    async def transition_clip_status(
        self,
        clip_id: int,
        *,
        from_statuses: tuple[ClipStatus, ...],
        to_status: ClipStatus,
        filename: str | None = None,
    ) -> bool:
        """Atomically move a clip only while it remains in an expected state."""
        return await asyncio.to_thread(
            self._transition_clip_status, clip_id, from_statuses, to_status, filename
        )

    def _transition_clip_status(
        self,
        clip_id: int,
        from_statuses: tuple[str, ...],
        to_status: str,
        filename: str | None,
    ) -> bool:
        if not from_statuses:
            return False
        placeholders = ", ".join(["%s"] * len(from_statuses))
        connection = self._connect()
        try:
            cursor = connection.cursor()
            assignments = "status = %s"
            values: list[object] = [to_status]
            if filename is not None:
                assignments += ", filename = %s"
                values.append(filename)
            values.extend((clip_id, *from_statuses))
            cursor.execute(
                f"UPDATE clips SET {assignments} "
                f"WHERE id = %s AND status IN ({placeholders})",
                tuple(values),
            )
            changed = cursor.rowcount == 1
            cursor.close()
            return changed
        finally:
            connection.close()

    def _update_clip(self, clip_id: int, status: str | None, filename: str | None, telegram_message_id: str | None) -> None:
        fields: list[str] = []
        values: list[object] = []
        if status is not None:
            fields.append("status = %s")
            values.append(status)
        if filename is not None:
            fields.append("filename = %s")
            values.append(filename)
        if telegram_message_id is not None:
            fields.append("telegram_message_id = %s")
            values.append(telegram_message_id)
        if not fields:
            return
        values.append(clip_id)
        connection = self._connect()
        try:
            cursor = connection.cursor()
            cursor.execute(f"UPDATE clips SET {', '.join(fields)} WHERE id = %s", tuple(values))
            if cursor.rowcount != 1:
                raise LookupError(f"Clip {clip_id} does not exist.")
            cursor.close()
        finally:
            connection.close()

    async def get_clip(self, clip_id: int) -> ClipRecord | None:
        return await asyncio.to_thread(self._get_clip, clip_id)

    async def clip_has_status(self, clip_id: int, status: ClipStatus) -> bool:
        return await asyncio.to_thread(self._clip_has_status, clip_id, status)

    async def processing_should_stop(self, clip_id: int) -> bool:
        """Return true when a render no longer owns the PROCESSING state."""
        return await asyncio.to_thread(self._processing_should_stop, clip_id)

    def _processing_should_stop(self, clip_id: int) -> bool:
        connection = self._connect()
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT status FROM clips WHERE id = %s LIMIT 1", (clip_id,))
            row = cursor.fetchone()
            cursor.close()
            return row is None or str(row[0]) != "PROCESSING"
        finally:
            connection.close()

    async def set_processing_progress(self, clip_id: int, progress: float) -> None:
        await asyncio.to_thread(self._set_processing_progress, clip_id, progress)

    def _set_processing_progress(self, clip_id: int, progress: float) -> None:
        percentage = max(0.0, min(float(progress), 100.0))
        connection = self._connect()
        try:
            cursor = connection.cursor()
            cursor.execute(
                "UPDATE clips SET processing_progress = %s "
                "WHERE id = %s AND status = 'PROCESSING'",
                (percentage, clip_id),
            )
            cursor.close()
        finally:
            connection.close()

    def _clip_has_status(self, clip_id: int, status: str) -> bool:
        connection = self._connect()
        try:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT 1 FROM clips WHERE id = %s AND status = %s LIMIT 1",
                (clip_id, status),
            )
            matches = cursor.fetchone() is not None
            cursor.close()
            return matches
        finally:
            connection.close()

    def _get_clip(self, clip_id: int) -> ClipRecord | None:
        connection = self._connect()
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute(
                "SELECT id, filename, status, source, timestamp, telegram_message_id, "
                "processing_progress FROM clips WHERE id = %s",
                (clip_id,),
            )
            row = cursor.fetchone()
            cursor.close()
            return ClipRecord(**row) if row else None
        finally:
            connection.close()

    async def list_clips(self, limit: int = 100) -> list[ClipRecord]:
        return await asyncio.to_thread(self._list_clips, limit)

    def _list_clips(self, limit: int) -> list[ClipRecord]:
        connection = self._connect()
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute(
                "SELECT id, filename, status, source, timestamp, telegram_message_id, "
                "processing_progress FROM clips "
                "ORDER BY timestamp DESC LIMIT %s", (max(1, min(limit, 500)),)
            )
            rows = cursor.fetchall()
            cursor.close()
            return [ClipRecord(**row) for row in rows]
        finally:
            connection.close()

    async def list_clips_by_status(self, status: ClipStatus, limit: int = 500) -> list[ClipRecord]:
        return await asyncio.to_thread(self._list_clips_by_status, status, limit)

    def _list_clips_by_status(self, status: str, limit: int) -> list[ClipRecord]:
        connection = self._connect()
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute(
                "SELECT id, filename, status, source, timestamp, telegram_message_id, "
                "processing_progress FROM clips "
                "WHERE status = %s ORDER BY timestamp ASC LIMIT %s",
                (status, max(1, min(limit, 2000))),
            )
            rows = cursor.fetchall()
            cursor.close()
            return [ClipRecord(**row) for row in rows]
        finally:
            connection.close()

    async def list_clips_by_status_page(
        self, status: ClipStatus, *, limit: int = 5, offset: int = 0,
    ) -> tuple[list[ClipRecord], int]:
        """Return one newest-first page and the unpaginated status total."""
        pages, totals = await self.list_clip_pages_by_status({status: offset}, limit=limit)
        return pages[status], totals[status]

    async def list_clip_pages_by_status(
        self, status_offsets: dict[ClipStatus, int], *, limit: int = 5,
    ) -> tuple[dict[str, list[ClipRecord]], dict[str, int]]:
        """Load independent status pages while sharing one database connection."""
        return await asyncio.to_thread(
            self._list_clip_pages_by_status, status_offsets, limit
        )

    def _list_clip_pages_by_status(
        self, status_offsets: dict[str, int], limit: int,
    ) -> tuple[dict[str, list[ClipRecord]], dict[str, int]]:
        if not status_offsets:
            return {}, {}
        page_size = max(1, min(limit, 50))
        statuses = tuple(status_offsets)
        connection = self._connect()
        try:
            cursor = connection.cursor(dictionary=True)
            placeholders = ", ".join(["%s"] * len(statuses))
            cursor.execute(
                f"SELECT status, COUNT(*) AS total FROM clips "
                f"WHERE status IN ({placeholders}) GROUP BY status",
                statuses,
            )
            totals = {status: 0 for status in statuses}
            totals.update({str(row["status"]): int(row["total"]) for row in cursor.fetchall()})

            pages: dict[str, list[ClipRecord]] = {}
            for status, offset in status_offsets.items():
                total = totals[status]
                last_offset = ((total - 1) // page_size) * page_size if total else 0
                safe_offset = min(max(0, offset), last_offset)
                cursor.execute(
                    "SELECT id, filename, status, source, timestamp, telegram_message_id, "
                    "processing_progress "
                    "FROM clips WHERE status = %s "
                    "ORDER BY timestamp DESC, id DESC LIMIT %s OFFSET %s",
                    (status, page_size, safe_offset),
                )
                pages[status] = [ClipRecord(**row) for row in cursor.fetchall()]
            cursor.close()
            return pages, totals
        finally:
            connection.close()

    async def start_publication_attempt(self, clip_id: int, platform: str) -> tuple[int, int]:
        return await asyncio.to_thread(self._start_publication_attempt, clip_id, platform)

    def _start_publication_attempt(self, clip_id: int, platform: str) -> tuple[int, int]:
        connection = self._connect()
        try:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT COALESCE(MAX(attempt_number), 0) + 1 FROM publication_attempts "
                "WHERE clip_id = %s AND platform = %s",
                (clip_id, platform),
            )
            attempt_number = int(cursor.fetchone()[0])
            cursor.execute(
                "INSERT INTO publication_attempts "
                "(clip_id, platform, attempt_number, status) VALUES (%s, %s, %s, 'STARTED')",
                (clip_id, platform, attempt_number),
            )
            attempt_id = int(cursor.lastrowid)
            cursor.close()
            return attempt_id, attempt_number
        finally:
            connection.close()

    async def finish_publication_attempt(self, attempt_id: int, status: str, detail: str) -> None:
        await asyncio.to_thread(self._finish_publication_attempt, attempt_id, status, detail)

    def _finish_publication_attempt(self, attempt_id: int, status: str, detail: str) -> None:
        connection = self._connect()
        try:
            cursor = connection.cursor()
            cursor.execute(
                "UPDATE publication_attempts SET status = %s, detail = %s, finished_at = NOW() "
                "WHERE id = %s",
                (status.upper(), detail[:65535], attempt_id),
            )
            cursor.close()
        finally:
            connection.close()

    def set_publication_attempt_progress(self, attempt_id: int, status: str, detail: str) -> None:
        """Update a running attempt from a blocking platform worker thread."""
        connection = self._connect()
        try:
            cursor = connection.cursor()
            cursor.execute(
                "UPDATE publication_attempts SET status = %s, detail = %s WHERE id = %s",
                (status.upper(), detail[:65535], attempt_id),
            )
            cursor.close()
        finally:
            connection.close()

    async def list_publication_attempts(
        self, clip_id: int, limit: int = 20,
    ) -> list[PublicationAttemptRecord]:
        return await asyncio.to_thread(self._list_publication_attempts, clip_id, limit)

    def _list_publication_attempts(
        self, clip_id: int, limit: int,
    ) -> list[PublicationAttemptRecord]:
        connection = self._connect()
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute(
                "SELECT id, clip_id, platform, attempt_number, status, detail, started_at, finished_at "
                "FROM publication_attempts WHERE clip_id = %s ORDER BY id DESC LIMIT %s",
                (clip_id, max(1, min(limit, 100))),
            )
            rows = cursor.fetchall()
            cursor.close()
            return [PublicationAttemptRecord(**row) for row in rows]
        finally:
            connection.close()

    async def get_published_platforms(self, clip_id: int) -> set[str]:
        return await asyncio.to_thread(self._get_published_platforms, clip_id)

    def _get_published_platforms(self, clip_id: int) -> set[str]:
        connection = self._connect()
        try:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT DISTINCT platform FROM publication_attempts "
                "WHERE clip_id = %s AND status IN ('PUBLISHED', 'DRAFT_READY')",
                (clip_id,),
            )
            platforms = {str(row[0]) for row in cursor.fetchall()}
            cursor.close()
            return platforms
        finally:
            connection.close()

    async def recover_interrupted_publications(self) -> None:
        """Make a crash visible without blindly risking duplicate posts."""
        await asyncio.to_thread(self._recover_interrupted_publications)

    def _recover_interrupted_publications(self) -> None:
        connection = self._connect()
        try:
            cursor = connection.cursor()
            cursor.execute(
                "UPDATE publication_attempts SET status = 'FAILED', "
                "detail = 'Application stopped before this attempt completed.', finished_at = NOW() "
                "WHERE status = 'STARTED'"
            )
            cursor.execute("UPDATE clips SET status = 'PUBLISH_FAILED' WHERE status = 'PUBLISHING'")
            cursor.close()
        finally:
            connection.close()

    async def test_connection(self) -> str:
        return await asyncio.to_thread(self._test_connection)

    def _test_connection(self) -> str:
        try:
            connection = self._connect()
            cursor = connection.cursor()
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()[0]
            cursor.close()
            connection.close()
            return str(version)
        except Error as exc:
            raise RuntimeError(f"MySQL connection failed: {exc}") from exc
