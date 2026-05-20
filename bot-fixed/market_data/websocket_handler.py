"""
market_data/websocket_handler.py
Async WebSocket handler for Polygon.io real-time streaming.
Handles connection, authentication, subscription, and reconnection.
"""
from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import websockets
from loguru import logger

from core.exceptions import WebSocketError
from core.models import Bar, Quote
from core.enums import BarTimeframe

_POLYGON_WS_URL = "wss://socket.polygon.io/stocks"


class PolygonWebSocketHandler:
    """
    Manages a persistent WebSocket connection to Polygon.io.

    Supports subscriptions for:
      - T.* — trades (every print)
      - Q.* — quotes (NBBO updates)
      - A.* — per-second aggregates
      - AM.* — per-minute aggregates

    Reconnects automatically with exponential back-off.
    """

    def __init__(
        self,
        api_key: str | None = None,
        on_bar: Callable[[Bar], None] | None = None,
        on_quote: Callable[[Quote], None] | None = None,
        on_raw: Callable[[dict], None] | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv("POLYGON_API_KEY", "")
        self._on_bar = on_bar
        self._on_quote = on_quote
        self._on_raw = on_raw
        self._ws: Any | None = None
        self._running = False
        self._subscriptions: set[str] = set()
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def start(self, symbols: list[str], feeds: list[str] = ("AM",)) -> None:
        """
        Connect, authenticate, subscribe, and start message loop.
        feeds: list of feed prefixes — "AM" (1-min aggs), "Q" (quotes), "T" (trades)
        """
        self._running = True
        for sym in symbols:
            for feed in feeds:
                self._subscriptions.add(f"{feed}.{sym}")
        asyncio.create_task(self._run_loop())
        logger.info("WebSocket handler started for {} symbols", len(symbols))

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
        logger.info("WebSocket handler stopped")

    async def subscribe(self, symbols: list[str], feed: str = "AM") -> None:
        for sym in symbols:
            self._subscriptions.add(f"{feed}.{sym}")
        if self._ws:
            await self._send({"action": "subscribe", "params": ",".join(
                f"{feed}.{sym}" for sym in symbols
            )})

    async def unsubscribe(self, symbols: list[str], feed: str = "AM") -> None:
        for sym in symbols:
            self._subscriptions.discard(f"{feed}.{sym}")
        if self._ws:
            await self._send({"action": "unsubscribe", "params": ",".join(
                f"{feed}.{sym}" for sym in symbols
            )})

    # ------------------------------------------------------------------
    # Internal connection loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self._connect_and_run()
            except websockets.exceptions.ConnectionClosed as exc:
                if not self._running:
                    break
                logger.warning("WebSocket closed ({}), reconnecting in {}s...", exc, self._reconnect_delay)
            except Exception as exc:
                if not self._running:
                    break
                logger.error("WebSocket error: {}, reconnecting in {}s...", exc, self._reconnect_delay)
            await asyncio.sleep(self._reconnect_delay)
            self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)

    async def _connect_and_run(self) -> None:
        logger.info("Connecting to Polygon WebSocket...")
        async with websockets.connect(_POLYGON_WS_URL, ping_interval=20) as ws:
            self._ws = ws
            self._reconnect_delay = 1.0  # Reset on successful connect

            async for raw_msg in ws:
                messages = json.loads(raw_msg)
                for msg in messages:
                    await self._dispatch(msg)

    async def _dispatch(self, msg: dict) -> None:
        event = msg.get("ev")

        if event == "connected":
            logger.info("Polygon WS connected, authenticating...")
            await self._send({"action": "auth", "params": self._api_key})

        elif event == "auth_success":
            logger.info("Polygon WS authenticated, subscribing...")
            if self._subscriptions:
                await self._send({
                    "action": "subscribe",
                    "params": ",".join(self._subscriptions),
                })

        elif event == "auth_failed":
            raise WebSocketError("Polygon WebSocket authentication failed")

        elif event in ("AM", "A"):
            # Aggregate (minute or second bar)
            if self._on_bar:
                bar = self._agg_to_bar(msg)
                self._on_bar(bar)

        elif event == "Q":
            # Quote
            if self._on_quote:
                quote = self._raw_to_quote(msg)
                self._on_quote(quote)

        elif event == "T":
            # Trade
            if self._on_raw:
                self._on_raw(msg)

        elif event in ("status", "subscription"):
            logger.debug("WS status: {}", msg)

    async def _send(self, payload: dict) -> None:
        if self._ws:
            await self._ws.send(json.dumps(payload))

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _agg_to_bar(msg: dict) -> Bar:
        ts = datetime.fromtimestamp(msg.get("s", msg.get("t", 0)) / 1000, tz=timezone.utc)
        return Bar(
            symbol=msg["sym"],
            timestamp=ts,
            timeframe=BarTimeframe.MINUTE_1,
            open=float(msg.get("o", 0)),
            high=float(msg.get("h", 0)),
            low=float(msg.get("l", 0)),
            close=float(msg.get("c", 0)),
            volume=int(msg.get("v", 0)),
            vwap=float(msg["vw"]) if "vw" in msg else None,
        )

    @staticmethod
    def _raw_to_quote(msg: dict) -> Quote:
        ts = datetime.fromtimestamp(msg.get("t", 0) / 1_000_000_000, tz=timezone.utc)
        return Quote(
            symbol=msg["sym"],
            timestamp=ts,
            bid=float(msg.get("bp", 0)),
            ask=float(msg.get("ap", 0)),
            bid_size=int(msg.get("bs", 0)),
            ask_size=int(msg.get("as", 0)),
        )
