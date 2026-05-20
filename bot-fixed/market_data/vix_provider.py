"""
market_data/vix_provider.py
VIX level fetcher for the volatility kill-switch circuit breaker.

Architecture:
  - Polygon /v2/aggs/ticker/I:VIX/... is the primary source (indices are
    available on Starter+ via the Indices data set). We hit the snapshot
    endpoint which returns the latest bar without pagination.
  - Yahoo Finance quote for ^VIX is the fallback — free, no auth, but
    scrape-dependent and subject to formatting changes.
  - A background asyncio task (VIXMonitor) polls every 5 minutes and
    pushes the result to the KillSwitchMonitor.
  - The last known value is always cached so callers get an answer even
    during a transient API failure.

Usage:
    monitor = VIXMonitor(polygon_api_key="...", kill_switch=kill_switch_monitor)
    asyncio.create_task(monitor.start())   # run in background
    vix = monitor.last_vix                 # read anytime
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import aiohttp
from loguru import logger

if TYPE_CHECKING:
    from risk.kill_switch import KillSwitchMonitor

# Poll interval in seconds — 5 minutes is fine for a kill-switch input
_POLL_INTERVAL_SECONDS = 300

# Polygon indices snapshot endpoint
_POLYGON_VIX_URL = (
    "https://api.polygon.io/v2/snapshot/locale/us/markets/indices/tickers"
)
_POLYGON_VIX_TICKER = "I:VIX"

# Yahoo Finance quote endpoint (no auth required)
_YAHOO_VIX_URL = (
    "https://query2.finance.yahoo.com/v8/finance/chart/%5EVIX"
)


# ---------------------------------------------------------------------------
# Low-level fetchers
# ---------------------------------------------------------------------------

async def _fetch_vix_polygon(
    session: aiohttp.ClientSession,
    api_key: str,
) -> float | None:
    """
    Fetch VIX from Polygon /v2/snapshot/indices endpoint.
    Returns the current value or None on failure / missing data.
    """
    try:
        async with session.get(
            _POLYGON_VIX_URL,
            params={"tickers": _POLYGON_VIX_TICKER, "apiKey": api_key},
            timeout=aiohttp.ClientTimeout(total=8),
        ) as resp:
            if resp.status == 403:
                logger.debug("VIX: Polygon indices not available on this tier")
                return None
            if resp.status != 200:
                logger.debug("VIX: Polygon HTTP {}", resp.status)
                return None
            data = await resp.json()
            tickers = data.get("tickers", [])
            if not tickers:
                return None
            value = tickers[0].get("value") or tickers[0].get("day", {}).get("c")
            if value is not None:
                return float(value)
    except Exception as exc:
        logger.debug("VIX Polygon fetch error: {}", exc)
    return None


async def _fetch_vix_yahoo(session: aiohttp.ClientSession) -> float | None:
    """
    Fetch VIX from Yahoo Finance chart API for ^VIX.
    Returns the most recent closing price or None on failure.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; trading-bot/1.0)",
        "Accept":     "application/json",
    }
    try:
        async with session.get(
            _YAHOO_VIX_URL,
            params={"interval": "5m", "range": "1d"},
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                logger.debug("VIX: Yahoo HTTP {}", resp.status)
                return None
            data = await resp.json()
            result = data.get("chart", {}).get("result", [])
            if not result:
                return None
            # Most recent close from the 5-minute bar
            closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
            # Filter out None values (gaps in intraday data)
            valid = [c for c in closes if c is not None]
            if valid:
                return float(valid[-1])
    except Exception as exc:
        logger.debug("VIX Yahoo fetch error: {}", exc)
    return None


# ---------------------------------------------------------------------------
# VIX Monitor — background task
# ---------------------------------------------------------------------------

class VIXMonitor:
    """
    Background task that fetches VIX every _POLL_INTERVAL_SECONDS and
    forwards the value to a KillSwitchMonitor for circuit breaker evaluation.

    The last known value is always accessible via .last_vix even after
    a failed fetch (stale-but-better-than-nothing behaviour).

    SPY data is also fetched here so a single background task handles
    both market-condition inputs required by the kill switch.
    """

    def __init__(
        self,
        polygon_api_key: str,
        kill_switch: "KillSwitchMonitor | None" = None,
        poll_interval: int = _POLL_INTERVAL_SECONDS,
    ) -> None:
        self._api_key      = polygon_api_key
        self._kill_switch  = kill_switch
        self._poll_interval = poll_interval
        self._session: aiohttp.ClientSession | None = None

        self._last_vix:       float | None = None
        self._last_spy:       float | None = None
        self._last_spy_open:  float | None = None
        self._last_fetched:   datetime | None = None
        self._task: asyncio.Task | None = None  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def last_vix(self) -> float | None:
        return self._last_vix

    @property
    def last_spy_move_pct(self) -> float | None:
        if self._last_spy and self._last_spy_open:
            return (self._last_spy - self._last_spy_open) / self._last_spy_open * 100.0
        return None

    @property
    def last_fetched(self) -> datetime | None:
        return self._last_fetched

    async def start(self) -> None:
        """
        Entry point for asyncio.create_task().
        Fetches immediately on start, then every poll_interval seconds.
        """
        self._session = aiohttp.ClientSession()
        try:
            # Initial fetch before sleeping
            await self._poll()
            while True:
                await asyncio.sleep(self._poll_interval)
                await self._poll()
        except asyncio.CancelledError:
            logger.debug("VIXMonitor: task cancelled")
        finally:
            if self._session:
                await self._session.close()
                self._session = None

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    # Internal poll
    # ------------------------------------------------------------------

    async def _poll(self) -> None:
        if self._session is None:
            return

        # Fetch VIX
        vix = await _fetch_vix_polygon(self._session, self._api_key)
        if vix is None:
            logger.debug("VIX: Polygon unavailable — trying Yahoo")
            vix = await _fetch_vix_yahoo(self._session)

        if vix is not None:
            self._last_vix = vix
            logger.debug("VIX: {:.2f}", vix)
        else:
            logger.warning("VIX: could not fetch from any source, using last={}", self._last_vix)

        # Fetch SPY open + current for kill-switch SPY-move check
        spy_current, spy_open = await self._fetch_spy(self._session)
        if spy_current is not None:
            self._last_spy = spy_current
        if spy_open is not None:
            self._last_spy_open = spy_open

        self._last_fetched = datetime.now(tz=timezone.utc)

        # Push to kill switch
        if self._kill_switch is not None:
            self._kill_switch.update_market_conditions(
                spy_price=self._last_spy,
                spy_open=self._last_spy_open,
                vix_level=self._last_vix,
            )

    async def _fetch_spy(
        self, session: aiohttp.ClientSession
    ) -> tuple[float | None, float | None]:
        """
        Return (current_price, open_price) for SPY.
        Uses Yahoo Finance daily bar — Polygon REST v2 aggs works too but
        this avoids an extra rate-limit bucket on Starter.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; trading-bot/1.0)",
            "Accept":     "application/json",
        }
        url = "https://query2.finance.yahoo.com/v8/finance/chart/SPY"
        try:
            async with session.get(
                url,
                params={"interval": "1d", "range": "1d"},
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status != 200:
                    return None, None
                data = await resp.json()
                result = data.get("chart", {}).get("result", [])
                if not result:
                    return None, None
                meta  = result[0].get("meta", {})
                open_ = result[0].get("indicators", {}).get("quote", [{}])[0].get("open", [])
                current = meta.get("regularMarketPrice") or meta.get("previousClose")
                open_price = open_[0] if open_ else None
                return (
                    float(current) if current else None,
                    float(open_price) if open_price else None,
                )
        except Exception as exc:
            logger.debug("SPY fetch error: {}", exc)
            return None, None
