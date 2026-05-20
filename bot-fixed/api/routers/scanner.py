"""
api/routers/scanner.py
Scanner dashboard endpoints.
"""
from __future__ import annotations

import json
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
async def get_scan_results_today(
    db: DBConn,
    passed: bool | None = Query(None, description="Filter by pass/fail. Omit for all."),
    limit: int = Query(200, le=1000),
    _user: str = Depends(require_viewer),
) -> list[dict]:
    """
    All scan results for today, ranked by composite score descending.
    Returns the raw_json payload which contains the full ScanResult model.
    """
    today = date.today().isoformat()
    where = f"date(scanned_at) = '{today}'"
    if passed is not None:
        where += f" AND passes_filters = {1 if passed else 0}"

    cur = await db.execute(
        f"""
        SELECT raw_json, composite_score, passes_filters, filter_failure_reason
        FROM scan_results
        WHERE {where}
        ORDER BY composite_score DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = await cur.fetchall()
    results = []
    for row in rows:
        try:
            data = json.loads(row["raw_json"])
            data["passes_filters"]      = bool(row["passes_filters"])
            data["composite_score"]     = row["composite_score"]
            data["filter_failure_reason"] = row["filter_failure_reason"]
            results.append(data)
        except Exception:
            continue
    return results


@router.get("/watchlist")
async def get_watchlist_today(
    db: DBConn,
    _user: str = Depends(require_viewer),
) -> list[dict]:
    """Top-ranked passing scan results for today (the live watchlist)."""
    today = date.today().isoformat()
    cur = await db.execute(
        """
        SELECT raw_json, composite_score
        FROM scan_results
        WHERE date(scanned_at) = ? AND passes_filters = 1
        ORDER BY composite_score DESC
        LIMIT 20
        """,
        (today,),
    )
    rows = await cur.fetchall()
    results = []
    for row in rows:
        try:
            data = json.loads(row["raw_json"])
            data["composite_score"] = row["composite_score"]
            results.append(data)
        except Exception:
            continue
    return results


@router.get("/summary")
async def get_scan_summary(
    db: DBConn,
    _user: str = Depends(require_viewer),
) -> dict:
    """Quick counts for the scanner dashboard header."""
    today = date.today().isoformat()
    cur = await db.execute(
        """
        SELECT
            COUNT(*) as total,
            SUM(passes_filters) as passed,
            COUNT(*) - SUM(passes_filters) as rejected,
            MAX(scanned_at) as last_scan_at,
            AVG(CASE WHEN passes_filters=1 THEN composite_score END) as avg_score
        FROM scan_results
        WHERE date(scanned_at) = ?
        """,
        (today,),
    )
    row = await cur.fetchone()
    return {
        "date":          today,
        "total_scanned": row["total"] if row else 0,
        "passed":        row["passed"] if row else 0,
        "rejected":      row["rejected"] if row else 0,
        "last_scan_at":  row["last_scan_at"] if row else None,
        "avg_score":     round(float(row["avg_score"] or 0), 4) if row else 0.0,
    }
