"""
api/app.py
FastAPI application — trading bot command center backend.

Reads directly from the SQLite database on the same machine as the bot.
All endpoints are read-only except for the admin kill-switch routes.

WebSocket /ws/live pushes real-time updates to the dashboard every 5 seconds
using the same SQLite data (no message queue needed for single-machine deploy).

Running:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

Environment variables:
    DB_PATH              — path to trading_bot.db (default: data/trading_bot.db)
    API_ADMIN_USER/PASS  — admin credentials
    API_VIEWER_USER/PASS — viewer credentials (comma-separated for multiple)
    ALLOWED_ORIGINS      — comma-separated CORS origins (default: *)
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated, AsyncGenerator

import aiosqlite
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.auth import require_admin, require_viewer
from api.routers import health, missed, positions, risk, scanner, trades

# ---------------------------------------------------------------------------
# Database dependency
# ---------------------------------------------------------------------------

_DB_PATH = os.getenv("DB_PATH", "data/trading_bot.db")


async def get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    """
    FastAPI dependency that yields a read-only aiosqlite connection.
    Each request gets its own connection from the pool.
    WAL mode is set so reads don't block the bot's writes.
    """
    async with aiosqlite.connect(_DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA query_only=ON")   # Read-only safety
        yield conn


DBConn = Annotated[aiosqlite.Connection, Depends(get_db)]


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class _WSManager:
    """Tracks all active WebSocket connections and broadcasts to all of them."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)
        logger.debug("WS: client connected (total: {})", len(self._connections))

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.remove(ws)
        logger.debug("WS: client disconnected (total: {})", len(self._connections))

    async def broadcast(self, data: dict) -> None:
        payload = json.dumps(data, default=str)
        dead: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._connections.remove(ws)

    @property
    def has_clients(self) -> bool:
        return len(self._connections) > 0


_ws_manager = _WSManager()


# ---------------------------------------------------------------------------
# App lifespan — background broadcast loop
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Start the WebSocket broadcast loop on startup."""
    task = asyncio.create_task(_broadcast_loop(), name="ws-broadcast")
    logger.info("Trading bot API started — DB: {}", _DB_PATH)
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        logger.info("Trading bot API stopped")


async def _broadcast_loop() -> None:
    """
    Every 5 seconds, if any WebSocket clients are connected,
    fetch a lightweight summary payload from SQLite and broadcast it.
    This keeps the dashboard live without polling from the client.
    """
    while True:
        try:
            await asyncio.sleep(5)
            if not _ws_manager.has_clients:
                continue
            payload = await _build_live_payload()
            await _ws_manager.broadcast(payload)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.debug("WS broadcast error: {}", exc)


async def _build_live_payload() -> dict:
    """
    Build a lightweight live update packet from SQLite.
    Deliberately minimal — just the numbers the dashboard needs to refresh.
    """
    today = date.today().isoformat()
    try:
        async with aiosqlite.connect(_DB_PATH) as conn:
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA journal_mode=WAL")

            # Open positions
            cur = await conn.execute(
                "SELECT COUNT(*) as n FROM positions WHERE status IN ('OPEN','PARTIALLY_CLOSED')"
            )
            row = await cur.fetchone()
            open_positions = row["n"] if row else 0

            # Today's P&L
            cur = await conn.execute(
                "SELECT COALESCE(SUM(net_pnl),0) as pnl, COUNT(*) as n "
                "FROM trades WHERE exit_time LIKE ?",
                (f"{today}%",),
            )
            row = await cur.fetchone()
            pnl   = round(float(row["pnl"]), 2) if row else 0.0
            n_trades = int(row["n"]) if row else 0

            # Latest scan time
            cur = await conn.execute(
                "SELECT MAX(scanned_at) as ts FROM scan_results WHERE scanned_at LIKE ?",
                (f"{today}%",),
            )
            row = await cur.fetchone()
            last_scan = row["ts"] if row else None

            # Kill switch events (unresolved)
            cur = await conn.execute(
                "SELECT COUNT(*) as n FROM kill_switch_events WHERE resolved_at IS NULL"
            )
            row = await cur.fetchone()
            halted = (row["n"] > 0) if row else False

        return {
            "type":            "live_update",
            "timestamp":       datetime.now(tz=timezone.utc).isoformat(),
            "open_positions":  open_positions,
            "pnl_today":       pnl,
            "trades_today":    n_trades,
            "last_scan_at":    last_scan,
            "kill_switch_active": halted,
        }
    except Exception as exc:
        return {"type": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    application = FastAPI(
        title="Quant Trading Bot — Command Center",
        description=(
            "Live monitoring, scanner data, trade blotter, and risk dashboard "
            "for the VWAP momentum trading system."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — restrict to configured origins in production
    raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
    origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Routers
    application.include_router(health.router,    prefix="/api",          tags=["System"])
    application.include_router(scanner.router,   prefix="/api/scanner",  tags=["Scanner"])
    application.include_router(positions.router, prefix="/api/positions",tags=["Positions"])
    application.include_router(trades.router,    prefix="/api/trades",   tags=["Trades"])
    application.include_router(risk.router,      prefix="/api/risk",     tags=["Risk"])
    application.include_router(missed.router,    prefix="/api/missed",   tags=["Analysis"])

    # WebSocket live feed
    @application.websocket("/ws/live")
    async def websocket_live(websocket: WebSocket) -> None:
        await _ws_manager.connect(websocket)
        try:
            # Send initial payload immediately on connect
            payload = await _build_live_payload()
            await websocket.send_text(json.dumps(payload, default=str))
            # Keep alive — client can send pings
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            _ws_manager.disconnect(websocket)

    @application.get("/", include_in_schema=False)
    async def root() -> dict:
        return {"service": "Quant Trading Bot API", "status": "running"}

    return application


app = create_app()
