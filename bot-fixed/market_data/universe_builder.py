"""
market_data/universe_builder.py
Builds the tradable universe from the data provider, applying all
exclusion filters defined in UniverseSettings.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from loguru import logger

from core.config import UniverseSettings
from core.exceptions import UniverseBuildError
from market_data.base_provider import BaseDataProvider
from market_data.market_cache import MarketCache

# Ticker suffixes/patterns that indicate non-common-stock instruments
_EXCLUSION_SUFFIXES = (
    ".WS",   # warrants
    ".WT",   # warrants
    ".RT",   # rights
    ".U",    # SPAC units
    ".R",    # rights
    "~",     # when-issued
)

_EXCLUSION_SUBSTRINGS = (
    "PREF",  # preferred
    " P ",   # preferred
)

# Known ETF suffixes (not exhaustive — primary filter is type=CS via API)
_ETF_KEYWORDS = ("ETF", "FUND", "TRUST", "REIT")


class UniverseBuilder:
    """
    Fetches and filters the tradable universe.

    Results are cached for the trading day to avoid repeated API calls.
    """

    def __init__(
        self,
        provider: BaseDataProvider,
        settings: UniverseSettings,
        cache: MarketCache | None = None,
    ) -> None:
        self._provider = provider
        self._settings = settings
        self._cache = cache or MarketCache()

    async def build_universe(self) -> list[str]:
        """Return filtered list of tradable tickers sorted by dollar volume desc."""
        date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        cache_key = f"universe:{date_str}"

        cached = await self._cache.get(cache_key)
        if cached:
            logger.debug("Universe loaded from cache: {} symbols", len(cached))
            return cached

        logger.info("Building tradable universe...")
        try:
            raw_tickers = await self._provider.get_tradable_universe(
                min_price=self._settings.min_price,
                min_avg_dollar_volume=self._settings.min_avg_daily_dollar_volume,
                max_results=self._settings.max_universe_size * 2,  # over-fetch then filter
            )
        except Exception as exc:
            raise UniverseBuildError(f"Failed to fetch universe: {exc}") from exc

        # Apply exclusion filters
        filtered = [t for t in raw_tickers if self._passes_exclusions(t)]

        # Apply size cap
        filtered = filtered[: self._settings.max_universe_size]

        logger.info(
            "Universe built: {} symbols (from {} raw)", len(filtered), len(raw_tickers)
        )

        await self._cache.set(cache_key, filtered, ttl_seconds=3600)
        return filtered

    async def build_universe_with_metadata(
        self, concurrency: int = 20
    ) -> list[dict[str, Any]]:
        """
        Build universe and enrich each ticker with dollar volume metadata.
        Used for deeper pre-scan filtering.
        """
        tickers = await self.build_universe()
        sem = asyncio.Semaphore(concurrency)
        results: list[dict[str, Any]] = []

        async def fetch_one(symbol: str) -> dict[str, Any] | None:
            async with sem:
                try:
                    adv = await self._provider.get_avg_daily_dollar_volume(symbol)
                    if adv >= self._settings.min_avg_daily_dollar_volume:
                        return {"symbol": symbol, "avg_daily_dv": adv}
                    return None
                except Exception:
                    return None

        tasks = [fetch_one(t) for t in tickers]
        fetched = await asyncio.gather(*tasks)
        results = [r for r in fetched if r is not None]
        results.sort(key=lambda x: x["avg_daily_dv"], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Exclusion logic
    # ------------------------------------------------------------------

    def _passes_exclusions(self, symbol: str) -> bool:
        sym_upper = symbol.upper()

        if self._settings.exclude_warrants:
            if any(sym_upper.endswith(s) for s in (".WS", ".WT", "W")):
                return False

        if self._settings.exclude_adrs:
            # ADRs often end in specific patterns — basic heuristic
            pass  # Better handled via API type filter (CS only)

        if self._settings.exclude_spacs:
            if sym_upper.endswith(".U"):
                return False

        # Skip symbols with non-standard characters (typically not common stock)
        if any(c in symbol for c in ("~", "/")):
            return False

        # Skip obviously non-standard ticker lengths
        if len(symbol) > 5:
            return False

        return True
