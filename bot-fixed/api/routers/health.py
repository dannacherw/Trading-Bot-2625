"""api/routers/health.py — System health and bot status endpoints."""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Annotated, Any

import aiosqlite
from fastapi import APIRouter, Depends

from api.auth import require_viewer

router = APIRouter()


async def get_db() -> Any:
    import os
    from api.app import get_db as _get_db
    async for conn in _get_db():
        yield conn


DBConn = Annotated[aiosqlite.Connection, Depends(get_db)]


@router.get("/health")
async def get_health(
    db: DBConn,
    _user: str = Depends(require_viewer),
) -> dict:
    """
    System health snapshot.
    Reads the most recent bot_health_log row and augments with live SQLite stats.
    """
    today = date.today().isoformat()

    cur = await db.execute(
        "SELECT * FROM kill_switch_events WHERE resolved_at IS NULL ORDER BY triggered_at DESC"
    )
    active_halts = [dict(r) for r in await cur.fetchall()]

    cur = await db.execute(
        "SELECT * FROM scan_window_log ORDER BY logged_at DESC LIMIT 1"
    )
    last_window = dict(await cur.fetchone() or {})

    cur = await db.execute(
        "SELECT MAX(scanned_at) as ts, COUNT(*) as n FROM scan_results WHERE scanned_at LIKE ?",
        (f"{today}%",),
    )
    row = await cur.fetchone()
    scan_info = {"last_scan_at": row["ts"], "scans_today": row["n"]} if row else {}

    cur = await db.execute(
        "SELECT COUNT(*) as n FROM positions WHERE status IN ('OPEN','PARTIALLY_CLOSED')"
    )
    row = await cur.fetchone()
    open_positions = row["n"] if row else 0

    return {
        "timestamp":         datetime.now(tz=timezone.utc).isoformat(),
        "kill_switch_active": len(active_halts) > 0,
        "active_halts":      active_halts,
        "open_positions":    open_positions,
        "last_scan_window":  last_window,
        **scan_info,
    }


@router.get("/health/kill-switch")
async def get_kill_switch_events(
    db: DBConn,
    limit: int = 50,
    _user: str = Depends(require_viewer),
) -> list[dict]:
    """Recent kill switch events (resolved and unresolved)."""
    cur = await db.execute(
        "SELECT * FROM kill_switch_events ORDER BY triggered_at DESC LIMIT ?",
        (limit,),
    )
    return [dict(r) for r in await cur.fetchall()]
