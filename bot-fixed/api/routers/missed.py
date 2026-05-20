"""api/routers/missed.py — Missed trade analysis endpoints."""
from __future__ import annotations

from datetime import date
from typing import Annotated, Any

import aiosqlite
from fastapi import APIRouter, Depends, Query

from api.auth import require_viewer

router = APIRouter()


async def get_db() -> Any:
    from api.app import get_db as _get_db
    async for conn in _get_db():
        yield conn


DBConn = Annotated[aiosqlite.Connection, Depends(get_db)]


@router.get("/today")
async def get_missed_today(
    db: DBConn,
    _user: str = Depends(require_viewer),
) -> list[dict]:
    """All missed trade audit results for today."""
    today = date.today().isoformat()
    cur = await db.execute(
        """
        SELECT * FROM missed_trade_audit
        WHERE audit_date = ?
        ORDER BY max_favorable_excursion_r DESC
        """,
        (today,),
    )
    return [dict(r) for r in await cur.fetchall()]


@router.get("/investigate")
async def get_should_investigate(
    db: DBConn,
    limit: int = Query(50, le=200),
    _user: str = Depends(require_viewer),
) -> list[dict]:
    """
    Missed trades flagged for investigation — hit +2R after rejection.
    Sorted by MFE descending (most painful misses first).
    """
    cur = await db.execute(
        """
        SELECT * FROM missed_trade_audit
        WHERE should_investigate = 1
        ORDER BY audit_date DESC, max_favorable_excursion_r DESC
        LIMIT ?
        """,
        (limit,),
    )
    return [dict(r) for r in await cur.fetchall()]


@router.get("/filter-stats")
async def get_filter_stats(
    db: DBConn,
    days: int = Query(30, ge=1, le=365),
    _user: str = Depends(require_viewer),
) -> list[dict]:
    """
    Per-filter false-negative statistics for the last N days.
    Powers the Filter Analytics tab on the dashboard.
    """
    end_date   = date.today().isoformat()
    start_date = (date.today().replace(day=1).__class__.fromordinal(
        date.today().toordinal() - days
    )).isoformat()

    cur = await db.execute(
        """
        SELECT
            rejection_filter,
            COUNT(*)                               AS total_rejected,
            SUM(hit_1r)                            AS hit_1r_count,
            SUM(hit_2r)                            AS hit_2r_count,
            SUM(hit_stop)                          AS hit_stop_count,
            SUM(rejection_was_correct)             AS correct_rejections,
            SUM(should_investigate)                AS should_investigate_count,
            ROUND(AVG(max_favorable_excursion_r),3) AS avg_mfe_r,
            ROUND(AVG(max_adverse_excursion_r),3)   AS avg_mae_r
        FROM missed_trade_audit
        WHERE audit_date >= ? AND audit_date <= ?
        GROUP BY rejection_filter
        ORDER BY hit_1r_count DESC
        """,
        (start_date, end_date),
    )
    return [dict(r) for r in await cur.fetchall()]


@router.get("/history")
async def get_missed_history(
    db: DBConn,
    start: str = Query(None),
    end:   str = Query(None),
    limit: int = Query(200, le=1000),
    _user: str = Depends(require_viewer),
) -> list[dict]:
    """Missed trade history with optional date range filter."""
    conditions = []
    params: list[Any] = []
    if start:
        conditions.append("audit_date >= ?")
        params.append(start)
    if end:
        conditions.append("audit_date <= ?")
        params.append(end)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)

    cur = await db.execute(
        f"""
        SELECT * FROM missed_trade_audit
        {where}
        ORDER BY audit_date DESC, max_favorable_excursion_r DESC
        LIMIT ?
        """,
        params,
    )
    return [dict(r) for r in await cur.fetchall()]
