"""
scanner/premarket_scanner.py
Main premarket scanner orchestrator.
Runs between 8:00–9:25 ET, producing a ranked watchlist every scan cycle.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Callable

from loguru import logger

from catalysts.catalyst_engine import CatalystEngine
from core.config import ScannerConfig
from core.exceptions import ScannerError
from core.models import PremarketMetrics, ScanResult
from database.repository import ScanResultRepository
from market_data.base_provider import BaseDataProvider
from market_data.market_cache import MarketCache
from market_data.universe_builder import UniverseBuilder
from scanner.filters import apply_all_filters
from scanner.metrics import compute_premarket_metrics_from_bars
from scanner.percentile_scoring import rank_scan_results, score_population_percentile
from scanner.scan_window import ScanWindowSettings, check_scan_window
from scanner.tagging import tag_stock
from scanner.watchlist import Watchlist
from market_data.float_provider import CompositeFloatProvider


class PremarketScanner:
    """
    Continuously scans the premarket universe between 8:00–9:25 ET.

    Scan cycle:
      1. Fetch tradable universe
      2. For each symbol: fetch premarket bars + quote
      3. Compute metrics
      4. Apply filters
      5. Detect catalysts for passing symbols
      6. Score cross-sectionally
      7. Tag archetypes
      8. Build ranked watchlist
      9. Persist results
     10. Emit watchlist to strategy engine
    """

    def __init__(
        self,
        provider: BaseDataProvider,
        config: ScannerConfig,
        catalyst_engine: CatalystEngine,
        scan_result_repo: ScanResultRepository,
        cache: MarketCache | None = None,
        on_watchlist_ready: Callable[[Watchlist], None] | None = None,
        float_provider: CompositeFloatProvider | None = None,
        scan_window: ScanWindowSettings | None = None,
    ) -> None:
        self._provider = provider
        self._config = config
        self._catalyst_engine = catalyst_engine
        self._repo = scan_result_repo
        self._cache = cache or MarketCache()
        self._on_watchlist_ready = on_watchlist_ready
        self._float_provider = float_provider
        self._scan_window = scan_window or ScanWindowSettings()
        self._universe_builder = UniverseBuilder(provider, config.universe, self._cache)
        self._watchlist = Watchlist(config.watchlist)
        self._running = False
        self._last_scan_at: datetime | None = None
        self._scan_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        logger.info("Premarket scanner started")
        while self._running:
            # Enforce scan window (8:00–9:25 AM ET)
            window = check_scan_window(self._scan_window)
            if not window.allowed:
                if window.minutes_to_open is not None:
                    wait = min(window.minutes_to_open * 60, 60.0)
                    logger.info("Scan window not open: {} — sleeping {:.0f}s", window.reason, wait)
                    await asyncio.sleep(wait)
                else:
                    logger.info("Scan window closed for today: {}", window.reason)
                    await asyncio.sleep(300)  # Sleep 5min and re-check
                continue
            if window.in_warning_zone:
                logger.warning("⚠ Scanner in warning zone: {}", window.reason)
            try:
                await self._scan_cycle()
            except ScannerError as exc:
                logger.error("Scanner error: {}", exc)
            except Exception as exc:
                logger.exception("Unexpected scanner error: {}", exc)
            await asyncio.sleep(self._config.scan_interval_seconds)

    def stop(self) -> None:
        self._running = False
        logger.info("Premarket scanner stopped")

    # ------------------------------------------------------------------
    # Single scan cycle
    # ------------------------------------------------------------------

    async def _scan_cycle(self) -> None:
        start_time = datetime.now(tz=timezone.utc)
        self._scan_count += 1
        logger.info("Scan cycle #{} starting...", self._scan_count)

        # 1. Build universe
        universe = await self._universe_builder.build_universe()
        logger.info("Universe: {} symbols", len(universe))

        # 2. Scan all symbols concurrently (with concurrency limit)
        scan_results = await self._scan_universe(universe)
        logger.info(
            "Scanned {} symbols — {} passed filters",
            len(scan_results),
            sum(1 for r in scan_results if r.passes_filters),
        )

        # 3. Detect catalysts for ALL symbols (pass + fail that had small gaps)
        # This is the critical fix: original code only detected catalysts for
        # passing symbols, meaning a 4% gap stock with strong earnings was
        # rejected by the no-catalyst 5% floor BEFORE catalyst detection ran.
        # Now: detect catalysts for all symbols with gap >= min_gap_pct (3%),
        # then re-run the tiered gap filter with catalyst context.
        small_gap_candidates = [
            r for r in scan_results
            if not r.passes_filters
            and r.metrics.gap_pct >= self._config.filters.min_gap_pct
        ]
        passing = [r for r in scan_results if r.passes_filters]
        catalyst_detection_targets = passing + small_gap_candidates
        catalyst_scores = await self._detect_catalysts(catalyst_detection_targets)

        # 4. Update catalyst on scan results
        scan_results = self._apply_catalysts(scan_results, catalyst_scores)

        # 4b. Re-filter small-gap candidates that now have catalyst context
        #     (gives tiered gap filter a second chance with real catalyst data)
        small_gap_symbols = {r.symbol for r in small_gap_candidates}  # Precomputed once
        rescanned = []
        for result in scan_results:
            if not result.passes_filters and result.symbol in small_gap_symbols:
                catalyst = self._catalyst_engine.get_catalyst(result.symbol)
                passes, reason = apply_all_filters(
                    result.metrics, self._config.filters, catalyst=catalyst
                )
                if passes:
                    logger.info(
                        "Catalyst re-filter PASS: {} (gap={:.1f}% + {})",
                        result.symbol, result.metrics.gap_pct,
                        catalyst.category.value if catalyst else "unknown",
                    )
                    rescanned.append(result.model_copy(update={
                        "passes_filters": True,
                        "filter_failure_reason": None,
                    }))
                else:
                    rescanned.append(result)
            else:
                rescanned.append(result)
        scan_results = rescanned

        # 5. Cross-sectional percentile scoring (rank-based, outlier-resistant)
        scan_results = score_population_percentile(
            scan_results, catalyst_scores, self._config.weights, use_historical=True
        )
        scan_results = rank_scan_results(scan_results)

        # 6. Apply archetype tags
        scan_results = self._apply_tags(scan_results)

        # 7. Build watchlist
        self._watchlist.build(scan_results)

        # 8. Persist results
        await self._persist_results(scan_results)

        elapsed = (datetime.now(tz=timezone.utc) - start_time).total_seconds()
        logger.info(
            "Scan cycle #{} complete in {:.1f}s — watchlist: {} ({} focus)",
            self._scan_count, elapsed,
            len(self._watchlist), len(self._watchlist.focus_list),
        )

        self._last_scan_at = start_time

        # 9. Notify strategy engine
        if self._on_watchlist_ready:
            self._on_watchlist_ready(self._watchlist)

    # ------------------------------------------------------------------
    # Per-symbol scanning
    # ------------------------------------------------------------------

    async def _scan_universe(self, symbols: list[str]) -> list[ScanResult]:
        sem = asyncio.Semaphore(20)  # max 20 concurrent requests
        today = datetime.now(tz=timezone.utc)

        async def scan_one(symbol: str) -> ScanResult | None:
            async with sem:
                try:
                    return await self._scan_symbol(symbol, today)
                except Exception as exc:
                    logger.debug("Failed to scan {}: {}", symbol, exc)
                    return None

        tasks = [scan_one(sym) for sym in symbols]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

    async def _scan_symbol(self, symbol: str, date: datetime) -> ScanResult:
        date_str = date.strftime("%Y-%m-%d")

        # Check cache first
        cached = await self._cache.get_metrics(symbol, date_str)
        if cached:
            metrics = cached
        else:
            # Fetch premarket bars and quote
            pm_bars = await self._provider.get_premarket_bars(symbol, date)
            if not pm_bars:
                raise ScannerError(f"No premarket bars for {symbol}")

            quote = await self._provider.get_premarket_quote(symbol)
            prev_close = await self._provider.get_prev_close(symbol)
            avg_dv = await self._provider.get_avg_daily_dollar_volume(symbol)
            avg_vol = avg_dv / max(prev_close, 0.01)

            metrics = compute_premarket_metrics_from_bars(
                symbol=symbol,
                pm_bars=pm_bars,
                prev_close=prev_close,
                quote=quote,
                avg_daily_dollar_volume=avg_dv,
                avg_daily_volume=avg_vol,
            )
            # Enrich with real float data when provider is available
            if self._float_provider is not None and metrics.float_shares is None:
                float_shares = await self._float_provider.get_float_shares(symbol)
                if float_shares is not None and float_shares > 0:
                    pm_volume = metrics.premarket_volume
                    float_rot = (pm_volume / float_shares * 100.0) if float_shares > 0 else None
                    metrics = metrics.model_copy(update={
                        "float_shares": float_shares,
                        "pm_float_rotation_pct": float_rot,
                    })
            await self._cache.set_metrics(symbol, date_str, metrics)

        # Apply filters
        passes, failure_reason = apply_all_filters(metrics, self._config.filters)

        return ScanResult(
            symbol=symbol,
            scanned_at=date,
            metrics=metrics,
            composite_score=0.0,  # Set in cross-sectional scoring step
            passes_filters=passes,
            filter_failure_reason=failure_reason,
        )

    # ------------------------------------------------------------------
    # Catalyst detection
    # ------------------------------------------------------------------

    async def _detect_catalysts(
        self, passing_results: list[ScanResult]
    ) -> dict[str, float]:
        """Returns dict of symbol -> catalyst_score for passing symbols."""
        if not passing_results:
            return {}
        symbols = [r.symbol for r in passing_results]
        return await self._catalyst_engine.score_symbols(symbols)

    def _apply_catalysts(
        self,
        scan_results: list[ScanResult],
        catalyst_scores: dict[str, float],
    ) -> list[ScanResult]:
        updated = []
        for result in scan_results:
            catalyst = self._catalyst_engine.get_catalyst(result.symbol)
            updated.append(result.model_copy(update={"catalyst": catalyst}))
        return updated

    # ------------------------------------------------------------------
    # Tagging
    # ------------------------------------------------------------------

    def _apply_tags(self, scan_results: list[ScanResult]) -> list[ScanResult]:
        tagged = []
        for result in scan_results:
            if result.passes_filters:
                tags = tag_stock(
                    metrics=result.metrics,
                    composite_score=result.composite_score,
                    catalyst=result.catalyst,
                    thresholds=self._config.archetypes,
                )
                tagged.append(result.model_copy(update={"archetypes": tags}))
            else:
                tagged.append(result)
        return tagged

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _persist_results(self, scan_results: list[ScanResult]) -> None:
        for result in scan_results:
            if result.passes_filters:
                try:
                    await self._repo.save(result)
                except Exception as exc:
                    logger.warning("Failed to persist scan for {}: {}", result.symbol, exc)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def watchlist(self) -> Watchlist:
        return self._watchlist

    @property
    def last_scan_at(self) -> datetime | None:
        return self._last_scan_at

    async def scan_once(self) -> Watchlist:
        """Run a single scan cycle and return the resulting watchlist."""
        await self._scan_cycle()
        return self._watchlist
