"""api/routers/risk.py — Risk dashboard endpoints."""
from __future__ import annotations

from datetime import date
from typing import Annotated, Any

import aiosqlite
from fastapi import APIRouter, Depends

from api.auth import require_admin, require_viewer

router = APIRouter()


async def get_db() -> Any:
    from api.app import get_db as _get_db
    async for conn in _get_db():
        yield conn


DBConn = Annotated[aiosqlite.Connection, Depends(get_db)]


@router.get("/state")
async def get_risk_state(
    db: DBConn,
    _user: str = Depends(require_viewer),
) -> dict:
    """Current risk state: daily P&L, loss limit remaining, open positions."""
    today = date.today().isoformat()

    # Daily P&L
    cur = await db.execute(
        "SELECT COALESCE(SUM(net_pnl),0) as pnl FROM trades WHERE exit_time LIKE ?",
        (f"{today}%",),
    )
    row = await cur.fetchone()
    daily_pnl = float(row["pnl"]) if row else 0.0

    # Open positions
    cur = await db.execute(
        """
        SELECT position_id, symbol, side, entry_price, quantity,
               stop_price, target_1_price
        FROM positions WHERE status IN ('OPEN','PARTIALLY_CLOSED')
        """
    )
    open_positions = [dict(r) for r in await cur.fetchall()]

    # Kill switch status
    cur = await db.execute(
        "SELECT * FROM kill_switch_events WHERE resolved_at IS NULL ORDER BY triggered_at"
    )
    active_halts = [dict(r) for r in await cur.fetchall()]

    # Estimated open risk (sum of dollar_risk across positions)
    # dollar_risk = (entry_price - stop_price) * quantity
    open_risk_dollars = sum(
        abs(p["entry_price"] - p["stop_price"]) * p["quantity"]
        for p in open_positions
        if p.get("stop_price")
    )

    return {
        "date":               today,
        "daily_pnl":          round(daily_pnl, 2),
        "open_positions":     len(open_positions),
        "positions":          open_positions,
        "open_risk_dollars":  round(open_risk_dollars, 2),
        "kill_switch_active": len(active_halts) > 0,
        "active_halts":       active_halts,
    }


@router.get("/kill-switch/events")
async def get_kill_switch_history(
    db: DBConn,
    limit: int = 100,
    _user: str = Depends(require_viewer),
) -> list[dict]:
    """All kill switch events (resolved and unresolved) ordered by time."""
    cur = await db.execute(
        "SELECT * FROM kill_switch_events ORDER BY triggered_at DESC LIMIT ?",
        (limit,),
    )
    return [dict(r) for r in await cur.fetchall()]


@router.get("/daily-performance")
async def get_daily_performance_history(
    db: DBConn,
    days: int = 30,
    _user: str = Depends(require_viewer),
) -> list[dict]:
    """
    Daily performance summary for the last N days, computed from the
    trades table. Does not require the Daily_Performance sheet to be populated.
    """
    cur = await db.execute(
        """
        SELECT
            date(exit_time) AS trade_date,
            COUNT(*)                                        AS total_trades,
            SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END)  AS wins,
            ROUND(SUM(net_pnl), 2)                         AS net_pnl,
            ROUND(AVG(r_multiple), 4)                      AS avg_r,
            ROUND(MIN(net_pnl), 2)                         AS max_loss,
            ROUND(MAX(net_pnl), 2)                         AS max_win
        FROM trades
        WHERE exit_time >= date('now', ?)
        GROUP BY date(exit_time)
        ORDER BY trade_date DESC
        """,
        (f"-{days} days",),
    )
    rows = await cur.fetchall()
    results = []
    for row in rows:
        d = dict(row)
        d["win_rate"] = round(d["wins"] / d["total_trades"], 4) if d["total_trades"] else 0.0
        results.append(d)
    return results
