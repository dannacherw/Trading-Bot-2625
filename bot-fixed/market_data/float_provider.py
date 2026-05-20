"""
market_data/float_provider.py
Float shares data fetcher.

Strategy:
  1. Check in-process cache (TTL = one trading day)
  2. Try Polygon ticker details endpoint (Starter tier +)
  3. On failure / missing data: fall back to Yahoo Finance
  4. Persist result to SQLite float_data_cache for next session

Design notes:
  - Sentinel pattern: _MISSING (singleton instance) distinguishes a true
    cache miss from a cached None (float genuinely unknown).
  - DB cache also uses a separate "fetched" flag column so None can be
    stored without being confused with "not yet fetched".
"""
from __future__ import annotations

import asyncio
import re
from datetime import date, datetime, timezone
from typing import Final

import aiohttp
from loguru import logger


# ---------------------------------------------------------------------------
# Sentinel — distinguishes "not in cache" from "cached as unknown"
# ---------------------------------------------------------------------------

class _MissingType:
    """Singleton sentinel for cache miss."""
    _instance: "_MissingType | None" = None

    def __new__(cls) -> "_MissingType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "MISSING"


MISSING: Final[_MissingType] = _MissingType()


# ---------------------------------------------------------------------------
# In-process daily cache
# ---------------------------------------------------------------------------

class _FloatCache:
    """
    In-process TTL cache keyed by symbol, valid for one trading day.
    Stores int | None — None means float is genuinely unknown from all sources.
    """

    def __init__(self) -> None:
        # symbol -> (float_shares_or_None, date_str)
        self._store: dict[str, tuple[int | None, str]] = {}

    def get(self, symbol: str) -> int | None | _MissingType:
        """Return cached value, or MISSING if not cached / stale."""
        today = date.today().isoformat()
        entry = self._store.get(symbol)
        if entry is None:
            return MISSING
        value, stored_date = entry
        if stored_date != today:
            return MISSING
        return value  # May be None — that is a valid cached result

    def set(self, symbol: str, value: int | None) -> None:
        self._store[symbol] = (value, date.today().isoformat())

    def clear(self) -> None:
        self._store.clear()


_cache = _FloatCache()


# ---------------------------------------------------------------------------
# Polygon float fetch
# ---------------------------------------------------------------------------

async def _fetch_float_polygon(
    symbol: str,
    session: aiohttp.ClientSession,
    api_key: str,
) -> int | None:
    """
    Fetch float shares from Polygon /v3/reference/tickers/{symbol}.
    Uses share_class_shares_outstanding as the best available proxy.
    Falls back to weighted_shares_outstanding if class shares missing.
    Returns None if unavailable or request fails.
    """
    url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
    try:
        async with session.get(
            url,
            params={"apiKey": api_key},
            timeout=aiohttp.ClientTimeout(total=8),
        ) as resp:
            if resp.status == 429:
                logger.warning("Polygon rate limited fetching float for {}", symbol)
                return None
            if resp.status != 200:
                logger.debug("Polygon float: HTTP {} for {}", resp.status, symbol)
                return None
            data = await resp.json()
            results = data.get("results", {}) or {}
            float_val = (
                results.get("share_class_shares_outstanding")
                or results.get("weighted_shares_outstanding")
            )
            if float_val and isinstance(float_val, (int, float)) and float_val > 0:
                return int(float_val)
            return None
    except Exception as exc:
        logger.debug("Polygon float error for {}: {}", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Yahoo Finance float fetch (fallback)
# ---------------------------------------------------------------------------

_YAHOO_FLOAT_RE = re.compile(r'"floatShares"\s*:\s*\{"raw"\s*:\s*(\d+)')
_YAHOO_LOOSE_RE = re.compile(r'floatShares[":\s]+(\d{6,})')

async def _fetch_float_yahoo(
    symbol: str,
    session: aiohttp.ClientSession,
) -> int | None:
    """
    Fetch float shares from Yahoo Finance quoteSummary v10 API.
    Falls back to HTML scrape of key-statistics page on failure.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; trading-bot/1.0)",
        "Accept": "application/json",
    }

    # --- JSON API (preferred) ---
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
    try:
        async with session.get(
            url,
            params={"modules": "defaultKeyStatistics"},
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                try:
                    stats = data["quoteSummary"]["result"][0]["defaultKeyStatistics"]
                    raw = stats.get("floatShares", {}).get("raw")
                    if raw and isinstance(raw, (int, float)) and raw > 0:
                        return int(raw)
                except (KeyError, IndexError, TypeError):
                    pass
    except Exception as exc:
        logger.debug("Yahoo JSON float error for {}: {}", symbol, exc)

    # --- HTML fallback ---
    html_url = f"https://finance.yahoo.com/quote/{symbol}/key-statistics"
    try:
        async with session.get(
            html_url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status == 200:
                text = await resp.text()
                for pattern in (_YAHOO_FLOAT_RE, _YAHOO_LOOSE_RE):
                    m = pattern.search(text)
                    if m:
                        val = int(m.group(1))
                        if val > 100_000:  # Sanity: must be a real share count
                            return val
    except Exception as exc:
        logger.debug("Yahoo HTML float error for {}: {}", symbol, exc)

    return None


# ---------------------------------------------------------------------------
# Composite provider
# ---------------------------------------------------------------------------

class CompositeFloatProvider:
    """
    Float shares provider: in-process cache → SQLite → Polygon → Yahoo.

    Usage (as async context manager):
        async with CompositeFloatProvider(api_key, db_repo) as fp:
            shares = await fp.get_float_shares("AAPL")
            batch  = await fp.get_float_shares_batch(["AAPL","TSLA"])
    """

    def __init__(
        self,
        polygon_api_key: str,
        db_repo: "FloatCacheRepository | None" = None,
        concurrency: int = 10,
    ) -> None:
        self._api_key = polygon_api_key
        self._db_repo = db_repo
        self._sem = asyncio.Semaphore(concurrency)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "CompositeFloatProvider":
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def get_float_shares(self, symbol: str) -> int | None:
        """Return float shares or None if unavailable from all sources."""
        async with self._sem:
            return await self._get(symbol)

    async def get_float_shares_batch(
        self, symbols: list[str]
    ) -> dict[str, int | None]:
        """Fetch float for multiple symbols concurrently."""
        results = await asyncio.gather(
            *(self.get_float_shares(sym) for sym in symbols)
        )
        return dict(zip(symbols, results))

    async def _get(self, symbol: str) -> int | None:
        # 1. In-process cache (fast path — avoids DB round-trip)
        cached = _cache.get(symbol)
        if cached is not MISSING:
            logger.debug("Float in-process cache hit for {}: {}", symbol, cached)
            return cached  # type: ignore[return-value]  # May be None

        # 2. SQLite persistence cache
        if self._db_repo is not None:
            db_result = await self._db_repo.get_with_hit_flag(symbol)
            if db_result.was_found:
                _cache.set(symbol, db_result.value)
                return db_result.value

        # 3. Polygon primary fetch
        session = self._session
        owns_session = session is None
        if owns_session:
            session = aiohttp.ClientSession()

        try:
            value = await _fetch_float_polygon(symbol, session, self._api_key)  # type: ignore[arg-type]

            if value is None:
                logger.debug("Polygon float unavailable for {} — trying Yahoo", symbol)
                value = await _fetch_float_yahoo(symbol, session)  # type: ignore[arg-type]

            if value is not None:
                logger.debug("Float resolved for {}: {:,} shares", symbol, value)
            else:
                logger.debug("Float unavailable for {} from all sources", symbol)

            # Always cache — None means "genuinely unknown, don't retry today"
            _cache.set(symbol, value)
            if self._db_repo is not None:
                await self._db_repo.set(symbol, value)

            return value

        finally:
            if owns_session:
                await session.close()  # type: ignore[union-attr]
