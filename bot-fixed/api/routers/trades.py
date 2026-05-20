"""api/routers/trades.py — Trade blotter and performance endpoints."""
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
async def get_trades_today(
    db: DBConn,
    _user: str = Depends(require_viewer),
) -> list[dict]:
    """All completed trades for today."""
    today = date.today().isoformat()
    cur = await db.execute(
        """
        SELECT trade_id, position_id, symbol, side, entry_price, exit_price,
               quantity, entry_time, exit_time, exit_reason,
               gross_pnl, commission, net_pnl, r_multiple, hold_duration_secs
        FROM trades
        WHERE exit_time LIKE ?
        ORDER BY exit_time ASC
        """,
        (f"{today}%",),
    )
    return [dict(r) for r in await cur.fetchall()]


@router.get("/summary/today")
async def get_daily_summary(
    db: DBConn,
    _user: str = Depends(require_viewer),
) -> dict:
    """Aggregate P&L statistics for today."""
    today = date.today().isoformat()
    cur = await db.execute(
        """
        SELECT
            COUNT(*)                                         AS total_trades,
            SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END)   AS wins,
            SUM(CASE WHEN net_pnl <= 0 THEN 1 ELSE 0 END)   AS losses,
            COALESCE(SUM(gross_pnl), 0)                      AS gross_pnl,
            COALESCE(SUM(net_pnl), 0)                        AS net_pnl,
            COALESCE(AVG(r_multiple), 0)                     AS avg_r,
            COALESCE(SUM(r_multiple), 0)                     AS total_r,
            COALESCE(MIN(net_pnl), 0)                        AS max_loss,
            COALESCE(MAX(net_pnl), 0)                        AS max_win,
            COALESCE(SUM(commission), 0)                     AS total_commission
        FROM trades
        WHERE exit_time LIKE ?
        """,
        (f"{today}%",),
    )
    row = await cur.fetchone()
    if row is None:
        return {"date": today, "total_trades": 0}

    d = dict(row)
    d["date"]     = today
    d["win_rate"] = round(d["wins"] / d["total_trades"], 4) if d["total_trades"] else 0.0
    d["net_pnl"]  = round(float(d["net_pnl"]), 2)
    d["avg_r"]    = round(float(d["avg_r"]), 4)
    return d


@router.get("/history")
async def get_trade_history(
    db: DBConn,
    start: str = Query(None, description="Start date YYYY-MM-DD"),
    end:   str = Query(None, description="End date YYYY-MM-DD"),
    limit: int = Query(500, le=5000),
    _user: str = Depends(require_viewer),
) -> list[dict]:
    """Trade history with optional date range filter."""
    conditions = []
    params: list[Any] = []

    if start:
        conditions.append("exit_time >= ?")
        params.append(f"{start}T00:00:00")
    if end:
        conditions.append("exit_time <= ?")
        params.append(f"{end}T23:59:59")

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)

    cur = await db.execute(
        f"""
        SELECT trade_id, symbol, side, entry_price, exit_price, quantity,
               entry_time, exit_time, exit_reason, net_pnl, r_multiple,
               hold_duration_secs
        FROM trades {where}
        ORDER BY exit_time DESC
        LIMIT ?
        """,
        params,
    )
    return [dict(r) for r in await cur.fetchall()]


@router.get("/{trade_id}")
async def get_trade_detail(
    trade_id: str,
    db: DBConn,
    _user: str = Depends(require_viewer),
) -> dict:
    """
    Full trade detail page: trade record + originating signal + scan metrics
    for the symbol on that day.
    """
    # Trade record
    cur = await db.execute(
        "SELECT * FROM trades WHERE trade_id = ?", (trade_id,)
    )
    trade = await cur.fetchone()
    if not trade:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Trade not found")
    data = dict(trade)

    # Originating position → signal
    cur2 = await db.execute(
        "SELECT * FROM positions WHERE position_id = ?", (data["position_id"],)
    )
    pos = await cur2.fetchone()
    if pos:
        import json
        pos_data = json.loads(pos["raw_json"]) if "raw_json" in pos.keys() else dict(pos)
        sig_id = pos_data.get("signal_id") or dict(pos).get("signal_id")
        if sig_id:
            cur3 = await db.execute("SELECT * FROM signals WHERE signal_id = ?", (sig_id,))
            sig = await cur3.fetchone()
            data["signal"] = dict(sig) if sig else None

    # Scan result for this symbol on trade date
    trade_date = data["entry_time"][:10]
    cur4 = await db.execute(
        """
        SELECT raw_json, composite_score FROM scan_results
        WHERE symbol = ? AND date(scanned_at) = ?
        ORDER BY composite_score DESC LIMIT 1
        """,
        (data["symbol"], trade_date),
    )
    scan = await cur4.fetchone()
    if scan:
        import json
        data["scan_result"] = json.loads(scan["raw_json"])

    return data
