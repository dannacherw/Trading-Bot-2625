"""
market_data/polygon_client.py
Polygon.io implementation of BaseDataProvider.
Supports REST API (all tiers) + WebSocket (Starter WS and above).
API key is loaded from POLYGON_API_KEY env var.
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import aiohttp
from loguru import logger

from core.enums import BarTimeframe
from core.exceptions import DataProviderError, InsufficientDataError
from core.models import Bar, PremarketMetrics, Quote
from market_data.base_provider import BaseDataProvider

_BASE_URL = "https://api.polygon.io"
_TIMEFRAME_MAP: dict[BarTimeframe, tuple[str, str]] = {
    BarTimeframe.MINUTE_1:  ("1",  "minute"),
    BarTimeframe.MINUTE_5:  ("5",  "minute"),
    BarTimeframe.MINUTE_15: ("15", "minute"),
    BarTimeframe.MINUTE_30: ("30", "minute"),
    BarTimeframe.HOUR_1:    ("1",  "hour"),
    BarTimeframe.DAY_1:     ("1",  "day"),
}


class PolygonClient(BaseDataProvider):
    """
    Polygon.io market data client.

    Rate limits (Starter): 5 req/min REST — handled via asyncio.sleep.
    Higher tiers: set POLYGON_UNLIMITED=1 to skip rate-limit pauses.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.getenv("POLYGON_API_KEY", "")
        if not self._api_key:
            logger.warning("POLYGON_API_KEY not set — market data calls will fail")
        self._session: aiohttp.ClientSession | None = None
        self._unlimited = bool(os.getenv("POLYGON_UNLIMITED", ""))
        self._request_semaphore = asyncio.Semaphore(5)  # max concurrent requests

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=aiohttp.ClientTimeout(total=15),
        )
        logger.info("PolygonClient connected")

    async def disconnect(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("PolygonClient disconnected")

    # ------------------------------------------------------------------
    # Internal request helper
    # ------------------------------------------------------------------

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if not self._session:
            raise DataProviderError("PolygonClient not connected. Call connect() first.")
        url = f"{_BASE_URL}{path}"
        async with self._request_semaphore:
            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status == 429:
                        retry_after = float(resp.headers.get("Retry-After", "60"))
                        logger.warning("Polygon rate limited — waiting {}s", retry_after)
                        await asyncio.sleep(retry_after)
                        return await self._get(path, params)
                    if resp.status == 403:
                        raise DataProviderError(
                            "Polygon API key invalid or insufficient tier for this endpoint"
                        )
                    if resp.status != 200:
                        text = await resp.text()
                        raise DataProviderError(
                            f"Polygon returned {resp.status}: {text[:200]}"
                        )
                    data = await resp.json()
                    if not self._unlimited:
                        await asyncio.sleep(0.12)  # ~5 req/s safety pause
                    return data
            except aiohttp.ClientError as exc:
                raise DataProviderError(f"Network error: {exc}") from exc

    async def _get_paginated(
        self, path: str, params: dict[str, Any], result_key: str = "results"
    ) -> list[dict]:
        """Follow Polygon's cursor-based pagination."""
        all_results: list[dict] = []
        params = dict(params)
        params.setdefault("limit", 50000)

        while True:
            data = await self._get(path, params)
            results = data.get(result_key, [])
            all_results.extend(results)
            next_url = data.get("next_url")
            if not next_url or not results:
                break
            # Extract cursor from next_url query string
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(next_url)
            qs = parse_qs(parsed.query)
            if "cursor" in qs:
                params = {"cursor": qs["cursor"][0]}
            else:
                break
        return all_results

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------

    async def get_tradable_universe(
        self,
        min_price: float = 5.0,
        min_avg_dollar_volume: float = 20_000_000,
        max_results: int = 1500,
    ) -> list[str]:
        data = await self._get(
            "/v3/reference/tickers",
            {
                "market": "stocks",
                "exchange": "XNYS,XNAS",  # NYSE + NASDAQ
                "active": "true",
                "type": "CS",             # Common stock only
                "sort": "market_cap",
                "order": "desc",
                "limit": min(max_results, 1000),
            },
        )
        tickers = [t["ticker"] for t in data.get("results", [])]
        logger.debug("Universe fetched: {} symbols", len(tickers))
        return tickers

    async def get_ticker_details(self, symbol: str) -> dict:
        data = await self._get(f"/v3/reference/tickers/{symbol}")
        return data.get("results", {})

    async def get_float_shares(self, symbol: str) -> int | None:
        """
        Fetch shares outstanding from Polygon ticker details.
        Returns None gracefully if API call fails or data is absent.
        Polygon returns `share_class_shares_outstanding` or `weighted_shares_outstanding`.
        """
        try:
            details = await self.get_ticker_details(symbol)
            # Prefer float (shares in market) over total shares outstanding
            float_val = (
                details.get("share_class_shares_outstanding")
                or details.get("weighted_shares_outstanding")
            )
            return int(float_val) if float_val else None
        except Exception as exc:
            logger.debug("Float data unavailable for {}: {}", symbol, exc)
            return None

    # ------------------------------------------------------------------
    # Historical
    # ------------------------------------------------------------------

    async def get_bars(
        self,
        symbol: str,
        timeframe: BarTimeframe,
        start: datetime,
        end: datetime,
    ) -> list[Bar]:
        mult, span = _TIMEFRAME_MAP[timeframe]
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
        results = await self._get_paginated(
            f"/v2/aggs/ticker/{symbol}/range/{mult}/{span}/{start_str}/{end_str}",
            {"adjusted": "true", "sort": "asc"},
        )
        return [self._agg_to_bar(symbol, r, timeframe) for r in results]

    async def get_prev_close(self, symbol: str) -> float:
        data = await self._get(f"/v2/aggs/ticker/{symbol}/prev", {"adjusted": "true"})
        results = data.get("results", [])
        if not results:
            raise InsufficientDataError(f"No prev close data for {symbol}")
        return float(results[0]["c"])

    async def get_atr(self, symbol: str, period: int = 14) -> float:
        end = datetime.now()
        start = end - timedelta(days=period * 2)
        bars = await self.get_bars(symbol, BarTimeframe.DAY_1, start, end)
        if len(bars) < period:
            raise InsufficientDataError(f"Insufficient bars to compute ATR for {symbol}")
        import numpy as np
        highs = np.array([b.high for b in bars])
        lows = np.array([b.low for b in bars])
        closes = np.array([b.close for b in bars])
        prev_closes = np.roll(closes, 1)
        prev_closes[0] = closes[0]
        tr = np.maximum(highs - lows, np.maximum(
            np.abs(highs - prev_closes), np.abs(lows - prev_closes)
        ))
        return float(np.mean(tr[-period:]))

    async def get_avg_daily_dollar_volume(
        self, symbol: str, lookback_days: int = 20
    ) -> float:
        end = datetime.now()
        start = end - timedelta(days=lookback_days * 2)
        bars = await self.get_bars(symbol, BarTimeframe.DAY_1, start, end)
        if not bars:
            return 0.0
        recent = bars[-lookback_days:]
        dvols = [b.close * b.volume for b in recent]
        return float(sum(dvols) / len(dvols))

    # ------------------------------------------------------------------
    # Premarket
    # ------------------------------------------------------------------

    async def get_premarket_bars(self, symbol: str, date: datetime) -> list[Bar]:
        date_str = date.strftime("%Y-%m-%d")
        results = await self._get_paginated(
            f"/v2/aggs/ticker/{symbol}/range/1/minute/{date_str}/{date_str}",
            {"adjusted": "true", "sort": "asc"},
        )
        pm_bars = []
        for r in results:
            ts = datetime.fromtimestamp(r["t"] / 1000, tz=timezone.utc)
            # 4:00–9:30 ET = 8:00–13:30 UTC
            hour_utc = ts.hour
            if 8 <= hour_utc < 13 or (hour_utc == 13 and ts.minute < 30):
                pm_bars.append(self._agg_to_bar(symbol, r, BarTimeframe.MINUTE_1))
        return pm_bars

    async def get_premarket_quote(self, symbol: str) -> Quote:
        data = await self._get(f"/v2/last/nbbo/{symbol}")
        result = data.get("results", {})
        now = datetime.now(tz=timezone.utc)
        return Quote(
            symbol=symbol,
            timestamp=now,
            bid=float(result.get("P", 0.0)),
            ask=float(result.get("p", 0.0)),
            bid_size=int(result.get("S", 0)),
            ask_size=int(result.get("s", 0)),
        )

    async def get_premarket_metrics(
        self, symbol: str, date: datetime
    ) -> PremarketMetrics:
        pm_bars = await self.get_premarket_bars(symbol, date)
        prev_close = await self.get_prev_close(symbol)
        quote = await self.get_premarket_quote(symbol)
        avg_dv = await self.get_avg_daily_dollar_volume(symbol)

        if not pm_bars:
            raise InsufficientDataError(f"No premarket bars for {symbol} on {date.date()}")

        pm_open = pm_bars[0].open
        pm_high = max(b.high for b in pm_bars)
        pm_low = min(b.low for b in pm_bars)
        pm_last = pm_bars[-1].close
        pm_vol = sum(b.volume for b in pm_bars)
        pm_dv = sum(b.close * b.volume for b in pm_bars)

        gap_pct = ((pm_last - prev_close) / prev_close * 100.0) if prev_close else 0.0
        pm_range = pm_high - pm_low
        range_pct = (pm_range / pm_low * 100.0) if pm_low else 0.0
        range_pos = ((pm_last - pm_low) / pm_range) if pm_range > 0 else 0.5

        # Trend quality: fraction of bars that close higher than previous
        closes = [b.close for b in pm_bars]
        up_bars = sum(1 for i in range(1, len(closes)) if closes[i] >= closes[i - 1])
        trend_quality = up_bars / max(len(closes) - 1, 1)

        # Relative volume: use avg daily vol / 6.5h session, scale to PM duration
        avg_bar_vol = avg_dv / (prev_close or 1) / 390  # avg min vol
        avg_pm_vol_estimate = avg_bar_vol * len(pm_bars)
        rel_vol = (pm_vol / avg_pm_vol_estimate) if avg_pm_vol_estimate > 0 else 1.0

        return PremarketMetrics(
            symbol=symbol,
            computed_at=datetime.now(tz=timezone.utc),
            prev_close=prev_close,
            premarket_open=pm_open,
            premarket_high=pm_high,
            premarket_low=pm_low,
            premarket_last=pm_last,
            premarket_volume=pm_vol,
            premarket_dollar_volume=pm_dv,
            gap_pct=gap_pct,
            relative_volume=rel_vol,
            spread_pct=quote.spread_pct,
            range_pct=range_pct,
            range_position=range_pos,
            trend_quality=trend_quality,
            avg_daily_dollar_volume=avg_dv,
        )

    # ------------------------------------------------------------------
    # Intraday / real-time
    # ------------------------------------------------------------------

    async def get_latest_bar(
        self, symbol: str, timeframe: BarTimeframe = BarTimeframe.MINUTE_1
    ) -> Bar:
        data = await self._get(
            f"/v2/aggs/ticker/{symbol}/prev", {"adjusted": "true"}
        )
        results = data.get("results", [])
        if not results:
            raise InsufficientDataError(f"No latest bar for {symbol}")
        return self._agg_to_bar(symbol, results[0], timeframe)

    async def get_current_quote(self, symbol: str) -> Quote:
        return await self.get_premarket_quote(symbol)

    async def get_intraday_bars(
        self, symbol: str, date: datetime, timeframe: BarTimeframe = BarTimeframe.MINUTE_1
    ) -> list[Bar]:
        date_str = date.strftime("%Y-%m-%d")
        mult, span = _TIMEFRAME_MAP[timeframe]
        results = await self._get_paginated(
            f"/v2/aggs/ticker/{symbol}/range/{mult}/{span}/{date_str}/{date_str}",
            {"adjusted": "true", "sort": "asc"},
        )
        return [self._agg_to_bar(symbol, r, timeframe) for r in results]

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------

    async def get_news(
        self,
        symbol: str,
        published_after: datetime | None = None,
        limit: int = 10,
    ) -> list[dict]:
        params: dict[str, Any] = {"ticker": symbol, "limit": limit, "order": "desc"}
        if published_after:
            params["published_utc.gte"] = published_after.strftime("%Y-%m-%dT%H:%M:%SZ")
        data = await self._get("/v2/reference/news", params)
        return data.get("results", [])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _agg_to_bar(symbol: str, agg: dict, timeframe: BarTimeframe) -> Bar:
        ts = datetime.fromtimestamp(agg["t"] / 1000, tz=timezone.utc)
        return Bar(
            symbol=symbol,
            timestamp=ts,
            timeframe=timeframe,
            open=float(agg["o"]),
            high=float(agg["h"]),
            low=float(agg["l"]),
            close=float(agg["c"]),
            volume=int(agg.get("v", 0)),
            vwap=float(agg["vw"]) if "vw" in agg else None,
        )

    @property
    def name(self) -> str:
        return "polygon"

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_news(self) -> bool:
        return True
