"""api/routers/positions.py — Open position endpoints."""
from __future__ import annotations

from typing import Annotated, Any

import aiosqlite
from fastapi import APIRouter, Depends

from api.auth import require_viewer

router = APIRouter()


async def get_db() -> Any:
    from api.app import get_db as _get_db
    async for conn in _get_db():
        yield conn


DBConn = Annotated[aiosqlite.Connection, Depends(get_db)]


@router.get("")
async def get_open_positions(
    db: DBConn,
    _user: str = Depends(require_viewer),
) -> list[dict]:
    """All currently open or partially-closed positions."""
    cur = await db.execute(
        """
        SELECT position_id, symbol, side, status, entry_price, entry_time,
               quantity, remaining_quantity, stop_price, target_1_price,
               target_2_price, breakeven_price, trailing_stop_price,
               realized_pnl, commission, signal_id
        FROM positions
        WHERE status IN ('OPEN', 'PARTIALLY_CLOSED')
        ORDER BY entry_time DESC
        """
    )
    rows = await cur.fetchall()
    return [dict(r) for r in rows]


@router.get("/{position_id}")
async def get_position_detail(
    position_id: str,
    db: DBConn,
    _user: str = Depends(require_viewer),
) -> dict:
    """Full detail for a single position including its signal."""
    cur = await db.execute(
        "SELECT raw_json FROM positions WHERE position_id = ?",
        (position_id,),
    )
    row = await cur.fetchone()
    if row is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Position not found")

    import json
    position_data = json.loads(row["raw_json"])

    # Attach the originating signal if available
    sig_id = position_data.get("signal_id")
    if sig_id:
        cur2 = await db.execute(
            "SELECT * FROM signals WHERE signal_id = ?", (sig_id,)
        )
        sig_row = await cur2.fetchone()
        position_data["signal"] = dict(sig_row) if sig_row else None

    return position_data
