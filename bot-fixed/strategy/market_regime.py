"""
strategy/market_regime.py
Intraday market regime filter.

Prevents taking VWAP pullback longs when the broader market or sector ETF
is in a clear downtrend — the single highest-impact improvement for win rate.

Benchmark is selected per symbol based on sector:
  Healthcare/Biotech → XBI  (sector ETF)
  Technology         → QQQ
  Small-cap          → IWM (detected via price < $20 and market cap heuristic)
  All others         → SPY

Two checks are run each cycle:
  1. Benchmark move from open: if down > threshold → reject all new longs
  2. Benchmark VWAP slope: if clearly negative → reject all new longs

Additionally computes relative strength of each symbol vs. its benchmark,
used as a required gate in entry_signals.py.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from core.config import MarketRegimeSettings
from core.models import Bar
from market_data.base_provider import BaseDataProvider
from market_data.market_cache import MarketCache
from scanner.metrics import compute_intraday_vwap, compute_vwap_slope


class MarketRegimeFilter:
    """
    Evaluates intraday market conditions before each entry signal.

    Usage:
        regime = MarketRegimeFilter(provider, settings, cache)
        ok, reason = await regime.is_favorable(symbol, sector="Healthcare")
        rs = await regime.get_relative_strength(symbol, symbol_bars)
    """

    def __init__(
        self,
        provider: BaseDataProvider,
        settings: MarketRegimeSettings,
        cache: MarketCache | None = None,
    ) -> None:
        self._provider = provider
        self._settings = settings
        self._cache = cache or MarketCache()
        # Stores most recent benchmark bars keyed by ETF symbol
        self._benchmark_bars: dict[str, list[Bar]] = {}
        self._last_benchmark_fetch: dict[str, datetime] = {}
        self._benchmark_fetch_interval_seconds = 60  # Refresh every minute

    # ------------------------------------------------------------------
    # Primary gate: is the market environment favorable?
    # ------------------------------------------------------------------

    async def is_favorable(
        self,
        symbol: str = "",
        sector: str = "",
    ) -> tuple[bool, str]:
        """
        Returns (True, "") if conditions allow new long entries.
        Returns (False, reason) if regime is unfavorable.
        """
        if not self._settings.enabled:
            return True, ""

        benchmark = self._resolve_benchmark(sector)
        bars = await self._get_benchmark_bars(benchmark)

        if not bars or len(bars) < 3:
            # Cannot determine regime — allow through but log
            logger.debug("Regime check: insufficient {} bars, allowing", benchmark)
            return True, ""

        # --- Check 1: benchmark move from open ---
        bench_open = bars[0].open
        bench_current = bars[-1].close
        bench_move_pct = (bench_current - bench_open) / bench_open * 100.0

        if bench_move_pct < self._settings.max_benchmark_down_pct:
            reason = (
                f"{benchmark} down {bench_move_pct:.2f}% from open "
                f"(threshold: {self._settings.max_benchmark_down_pct:.2f}%)"
            )
            logger.debug("Regime UNFAVORABLE: {}", reason)
            return False, reason

        # --- Check 2: benchmark VWAP slope ---
        vwap_slope = compute_vwap_slope(bars, lookback=self._settings.vwap_slope_lookback_bars)
        if vwap_slope < self._settings.min_benchmark_vwap_slope:
            reason = (
                f"{benchmark} VWAP slope {vwap_slope:.4f} < "
                f"threshold {self._settings.min_benchmark_vwap_slope:.4f}"
            )
            logger.debug("Regime UNFAVORABLE: {}", reason)
            return False, reason

        return True, ""

    # ------------------------------------------------------------------
    # Relative strength of symbol vs. its benchmark
    # ------------------------------------------------------------------

    async def get_relative_strength(
        self,
        symbol: str,
        symbol_bars: list[Bar],
        sector: str = "",
    ) -> float:
        """
        Beta-adjusted relative strength: symbol move % − benchmark move %.
        Positive = outperforming benchmark intraday.
        Returns 0.0 if benchmark data unavailable.
        """
        if not symbol_bars:
            return 0.0

        benchmark = self._resolve_benchmark(sector)
        bench_bars = await self._get_benchmark_bars(benchmark)
        if not bench_bars:
            return 0.0

        sym_move = (symbol_bars[-1].close - symbol_bars[0].open) / symbol_bars[0].open * 100.0
        bench_move = (bench_bars[-1].close - bench_bars[0].open) / bench_bars[0].open * 100.0
        rs = sym_move - bench_move

        logger.debug(
            "{} RS vs {}: sym={:.2f}% bench={:.2f}% → RS={:.2f}%",
            symbol, benchmark, sym_move, bench_move, rs,
        )
        return rs

    # ------------------------------------------------------------------
    # Benchmark resolution
    # ------------------------------------------------------------------

    def _resolve_benchmark(self, sector: str) -> str:
        """Map sector string to the appropriate benchmark ETF."""
        if not sector:
            return self._settings.default_benchmark
        # Case-insensitive lookup
        sector_lower = sector.lower()
        for sector_key, etf in self._settings.sector_benchmark_map.items():
            if sector_key.lower() in sector_lower or sector_lower in sector_key.lower():
                return etf
        return self._settings.default_benchmark

    # ------------------------------------------------------------------
    # Benchmark data fetching with cache
    # ------------------------------------------------------------------

    async def _get_benchmark_bars(self, benchmark: str) -> list[Bar]:
        """Fetch intraday bars for a benchmark ETF, with 60s cache."""
        now = datetime.now(tz=timezone.utc)
        last_fetch = self._last_benchmark_fetch.get(benchmark)

        if (
            last_fetch is not None
            and (now - last_fetch).total_seconds() < self._benchmark_fetch_interval_seconds
            and benchmark in self._benchmark_bars
        ):
            return self._benchmark_bars[benchmark]

        try:
            bars = await self._provider.get_intraday_bars(benchmark, now)
            self._benchmark_bars[benchmark] = bars
            self._last_benchmark_fetch[benchmark] = now
            return bars
        except Exception as exc:
            logger.warning("Failed to fetch {} bars for regime check: {}", benchmark, exc)
            return self._benchmark_bars.get(benchmark, [])

    # ------------------------------------------------------------------
    # Sector lookup (from ticker details)
    # ------------------------------------------------------------------

    async def get_sector_for_symbol(self, symbol: str) -> str:
        """
        Fetch sector for a symbol from provider ticker details.
        Returns "" if unavailable — regime defaults to SPY in that case.
        Cached for the day.
        """
        cache_key = f"sector:{symbol}"
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            details = await self._provider.get_ticker_details(symbol)
            sector = details.get("sic_description", "") or details.get("type", "")
            await self._cache.set(cache_key, sector, ttl_seconds=86400)
            return sector
        except Exception:
            return ""

    def reset_daily(self) -> None:
        """Clear all cached benchmark bars at the start of a new session."""
        self._benchmark_bars.clear()
        self._last_benchmark_fetch.clear()
