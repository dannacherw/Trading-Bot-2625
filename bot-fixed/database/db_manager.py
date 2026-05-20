"""
database/db_manager.py
Async SQLite database manager. Single connection with WAL mode for
concurrent reads. Handles schema migrations automatically on startup.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import aiosqlite
from loguru import logger

from core.exceptions import DatabaseError, MigrationError
from database.schema import CREATE_MIGRATIONS_TABLE, MIGRATIONS, SCHEMA_VERSION


class DatabaseManager:
    """
    Manages a single aiosqlite connection with WAL mode.
    Use as an async context manager or call connect()/close() explicitly.
    """

    def __init__(self, db_path: str = "data/trading_bot.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        if self._conn is not None:
            return
        self._conn = await aiosqlite.connect(str(self._db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA cache_size=-64000")  # 64 MB
        await self._conn.commit()
        logger.info("Database connected: {}", self._db_path.resolve())
        await self._run_migrations()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            logger.info("Database connection closed")

    async def __aenter__(self) -> "DatabaseManager":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Migrations
    # ------------------------------------------------------------------

    async def _run_migrations(self) -> None:
        conn = self._get_conn()
        try:
            await conn.execute(CREATE_MIGRATIONS_TABLE)
            await conn.commit()

            cursor = await conn.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            )
            applied = {row[0] for row in await cursor.fetchall()}

            for version in sorted(MIGRATIONS):
                if version not in applied:
                    logger.info("Applying migration v{}", version)
                    try:
                        await conn.executescript(MIGRATIONS[version])
                        await conn.execute(
                            "INSERT INTO schema_migrations(version) VALUES(?)", (version,)
                        )
                        await conn.commit()
                        logger.info("Migration v{} applied successfully", version)
                    except Exception as exc:
                        await conn.rollback()
                        raise MigrationError(f"Migration v{version} failed: {exc}") from exc
        except MigrationError:
            raise
        except Exception as exc:
            raise DatabaseError(f"Migration check failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise DatabaseError("Database not connected. Call connect() first.")
        return self._conn

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Yield connection inside an explicit transaction."""
        conn = self._get_conn()
        try:
            yield conn
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise

    # ------------------------------------------------------------------
    # Execute helpers
    # ------------------------------------------------------------------

    async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        conn = self._get_conn()
        try:
            await conn.execute(sql, params)
            await conn.commit()
        except Exception as exc:
            raise DatabaseError(f"Execute failed: {exc}\nSQL: {sql}") from exc

    async def execute_many(self, sql: str, params_list: list[tuple[Any, ...]]) -> None:
        conn = self._get_conn()
        try:
            await conn.executemany(sql, params_list)
            await conn.commit()
        except Exception as exc:
            raise DatabaseError(f"Execute many failed: {exc}") from exc

    async def fetch_all(
        self, sql: str, params: tuple[Any, ...] = ()
    ) -> list[dict[str, Any]]:
        conn = self._get_conn()
        try:
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as exc:
            raise DatabaseError(f"Fetch all failed: {exc}") from exc

    async def fetch_one(
        self, sql: str, params: tuple[Any, ...] = ()
    ) -> dict[str, Any] | None:
        conn = self._get_conn()
        try:
            cursor = await conn.execute(sql, params)
            row = await cursor.fetchone()
            return dict(row) if row else None
        except Exception as exc:
            raise DatabaseError(f"Fetch one failed: {exc}") from exc

    async def fetch_scalar(
        self, sql: str, params: tuple[Any, ...] = ()
    ) -> Any:
        row = await self.fetch_one(sql, params)
        if row is None:
            return None
        return next(iter(row.values()))

    # ------------------------------------------------------------------
    # JSON helpers (for raw_json columns)
    # ------------------------------------------------------------------

    @staticmethod
    def to_json(obj: Any) -> str:
        return json.dumps(obj, default=str)

    @staticmethod
    def from_json(s: str) -> Any:
        return json.loads(s)
