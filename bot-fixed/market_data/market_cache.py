"""
market_data/market_cache.py
Thread-safe in-memory LRU cache for market data.
Prevents redundant API calls within a scanning cycle.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any

from loguru import logger


class MarketCache:
    """
    Async-safe in-memory cache with TTL and max-size eviction.
    Keys are arbitrary strings; values are any serialisable object.
    """

    def __init__(self, max_size: int = 2000, default_ttl_seconds: int = 60) -> None:
        self._cache: OrderedDict[str, tuple[Any, datetime]] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = timedelta(seconds=default_ttl_seconds)
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            value, expires_at = self._cache[key]
            if datetime.now(tz=timezone.utc) > expires_at:
                del self._cache[key]
                self._misses += 1
                return None
            # LRU: move to end
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> None:
        ttl = timedelta(seconds=ttl_seconds) if ttl_seconds is not None else self._default_ttl
        expires_at = datetime.now(tz=timezone.utc) + ttl
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, expires_at)
            # Evict oldest if over capacity
            while len(self._cache) > self._max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug("Cache evicted: {}", evicted_key)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._cache.pop(key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()
            logger.debug("Market cache cleared")

    async def invalidate_symbol(self, symbol: str) -> None:
        """Remove all cache entries for a given symbol."""
        async with self._lock:
            to_delete = [k for k in self._cache if k.startswith(f"{symbol}:")]
            for k in to_delete:
                del self._cache[k]

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 3),
        }

    # ------------------------------------------------------------------
    # Typed helpers for common cache patterns
    # ------------------------------------------------------------------

    async def get_bars(self, symbol: str, date_str: str) -> Any | None:
        return await self.get(f"{symbol}:bars:{date_str}")

    async def set_bars(self, symbol: str, date_str: str, bars: Any) -> None:
        await self.set(f"{symbol}:bars:{date_str}", bars, ttl_seconds=300)

    async def get_quote(self, symbol: str) -> Any | None:
        return await self.get(f"{symbol}:quote")

    async def set_quote(self, symbol: str, quote: Any) -> None:
        await self.set(f"{symbol}:quote", quote, ttl_seconds=5)

    async def get_metrics(self, symbol: str, date_str: str) -> Any | None:
        return await self.get(f"{symbol}:metrics:{date_str}")

    async def set_metrics(self, symbol: str, date_str: str, metrics: Any) -> None:
        await self.set(f"{symbol}:metrics:{date_str}", metrics, ttl_seconds=120)

    async def get_prev_close(self, symbol: str) -> float | None:
        return await self.get(f"{symbol}:prev_close")

    async def set_prev_close(self, symbol: str, close: float) -> None:
        await self.set(f"{symbol}:prev_close", close, ttl_seconds=86400)

    async def get_atr(self, symbol: str) -> float | None:
        return await self.get(f"{symbol}:atr")

    async def set_atr(self, symbol: str, atr: float) -> None:
        await self.set(f"{symbol}:atr", atr, ttl_seconds=86400)
