"""
strategy/vwap_strategy.py — v2
VWAP pullback continuation strategy engine.

New in v2:
  - MarketRegimeFilter gate before every entry evaluation
  - FirstCandleAnalysis computed at 9:35 ET per symbol
  - Relative strength computed per symbol vs. sector benchmark
  - SignalOutcomeTracker wired to all signal/trade events
  - Sector fetched per symbol for benchmark resolution
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Callable

from loguru import logger

from core.config import StrategyConfig, StopLossSettings
from core.models import Bar, Quote, Signal, WatchlistItem
from market_data.base_provider import BaseDataProvider
from market_data.market_cache import MarketCache
from scanner.metrics import compute_intraday_vwap
from scanner.watchlist import Watchlist
from strategy.entry_signals import detect_vwap_pullback_entry
from strategy.first_candle import FirstCandleAnalysis, compute_first_candle, is_valid_setup
from strategy.market_regime import MarketRegimeFilter
from strategy.signal_tracker import SignalOutcomeTracker
from strategy.trade_manager import TradeManager


class VWAPStrategy:
    """
    Monitors watchlisted symbols for VWAP pullback continuation setups.
    Emits trade signals to the risk/execution layer.
    Integrates regime filter, first candle gate, RS check, and signal tracking.
    """

    # How many minutes after open to wait before computing first candle
    FIRST_CANDLE_READY_MINUTES = 5

    def __init__(
        self,
        provider: BaseDataProvider,
        config: StrategyConfig,
        trade_manager: TradeManager,
        cache: MarketCache | None = None,
        on_signal: Callable[[Signal], None] | None = None,
        loop_interval_seconds: int = 10,
        risk_engine: "RiskEngine | None" = None,
    ) -> None:
        self._provider = provider
        self._config = config
        self._trade_manager = trade_manager
        self._cache = cache or MarketCache()
        self._on_signal = on_signal
        self._loop_interval = loop_interval_seconds  # Configurable — was hardcoded 30s
        self._risk_engine = risk_engine  # Injected so sector map can be pushed up
        # StopLossSettings loaded from risk_config — injected at session start if available
        self._stop_settings: StopLossSettings = StopLossSettings()

        # Sub-modules
        self._regime = MarketRegimeFilter(provider, config.regime, self._cache)
        self._signal_tracker = SignalOutcomeTracker()

        self._watchlist: Watchlist | None = None
        self._running = False
        self._bar_buffer: dict[str, list[Bar]] = {}
        self._emitted_signals: dict[str, Signal] = {}  # symbol → signal (for tracking)
        self._first_candles: dict[str, FirstCandleAnalysis | None] = {}
        self._pm_bars: dict[str, list[Bar]] = {}       # premarket bars (for first candle vol baseline)
        self._sectors: dict[str, str] = {}             # symbol → sector string
        self._market_open: datetime | None = None
        self._first_candle_ready = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def set_watchlist(self, watchlist: Watchlist) -> None:
        """Called by scanner when a new watchlist is ready."""
        self._watchlist = watchlist
        for symbol in watchlist.symbols:
            if symbol not in self._bar_buffer:
                self._bar_buffer[symbol] = []
        logger.info(
            "Strategy watchlist updated: {} symbols ({} focus)",
            len(watchlist), len(watchlist.focus_list),
        )

    def set_premarket_bars(self, symbol: str, bars: list[Bar]) -> None:
        """Store premarket bars for a symbol (used as volume baseline for first candle)."""
        self._pm_bars[symbol] = bars

    async def start(self, market_open: datetime) -> None:
        self._market_open = market_open
        self._running = True
        self._first_candle_ready = False
        logger.info("VWAPStrategy started — market open: {}", market_open)
        await self._run_loop()

    def stop(self) -> None:
        self._running = False
        self._signal_tracker.log_daily_report()
        logger.info("VWAPStrategy stopped")

    # ------------------------------------------------------------------
    # Main polling loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self._process_cycle()
            except Exception as exc:
                logger.exception("Strategy loop error: {}", exc)
            await asyncio.sleep(self._loop_interval)  # Configurable via strategy_config.yaml

    async def _process_cycle(self) -> None:
        if not self._watchlist or not self._market_open:
            return

        now = datetime.now(tz=timezone.utc)
        minutes_since_open = (now - self._market_open).total_seconds() / 60.0

        # --- Compute first candles once, 5 min after open ---
        if not self._first_candle_ready and minutes_since_open >= self.FIRST_CANDLE_READY_MINUTES:
            await self._compute_all_first_candles()
            self._first_candle_ready = True

        focus_symbols = self._watchlist.focus_symbols
        if not focus_symbols:
            return

        # --- Check regime once per cycle (not per symbol) ---
        regime_ok, regime_reason = await self._regime.is_favorable()
        if not regime_ok:
            logger.info("Regime gate blocking new entries: {}", regime_reason)
            # Still evaluate exits even when regime is unfavorable
            await self._evaluate_exits()
            return

        sem = asyncio.Semaphore(10)

        async def fetch_and_evaluate(symbol: str) -> None:
            async with sem:
                await self._evaluate_symbol(symbol)

        await asyncio.gather(*[fetch_and_evaluate(sym) for sym in focus_symbols])
        await self._evaluate_exits()

    # ------------------------------------------------------------------
    # First candle computation
    # ------------------------------------------------------------------

    async def _compute_all_first_candles(self) -> None:
        """Compute first candle analysis for all watchlisted symbols."""
        if not self._watchlist or not self._market_open:
            return

        logger.info("Computing first candle analysis for {} symbols...",
                    len(self._watchlist.focus_symbols))

        today = datetime.now(tz=timezone.utc)
        sem = asyncio.Semaphore(10)

        async def compute_one(symbol: str) -> None:
            async with sem:
                try:
                    bars = await self._provider.get_intraday_bars(symbol, today)
                    pm_bars = self._pm_bars.get(symbol, [])
                    analysis = compute_first_candle(
                        symbol=symbol,
                        minute_bars=bars,
                        pm_bars=pm_bars,
                        market_open=self._market_open,  # type: ignore
                    )
                    self._first_candles[symbol] = analysis
                    if analysis:
                        logger.debug("{}", str(analysis))
                except Exception as exc:
                    logger.debug("First candle failed for {}: {}", symbol, exc)
                    self._first_candles[symbol] = None

        await asyncio.gather(*[compute_one(sym) for sym in self._watchlist.focus_symbols])

    # ------------------------------------------------------------------
    # Per-symbol entry evaluation
    # ------------------------------------------------------------------

    async def _evaluate_symbol(self, symbol: str) -> None:
        """Fetch latest bars, run all gates, check for entry signal."""
        if symbol in self._emitted_signals:
            return  # Already traded this symbol today

        # --- First candle gate ---
        if self._config.entry.require_first_candle_analysis:
            fc = self._first_candles.get(symbol)
            fc_ok, fc_reason = is_valid_setup(
                fc,
                min_close_position=self._config.entry.first_candle_min_close_position,
                min_volume_vs_pm=self._config.entry.first_candle_min_volume_vs_pm,
            )
            if not fc_ok and fc is not None:
                logger.debug("{}: first candle gate — {}", symbol, fc_reason)
                return

        try:
            today = datetime.now(tz=timezone.utc)
            bars = await self._provider.get_intraday_bars(symbol, today)
            if not bars:
                return
            self._bar_buffer[symbol] = bars

            quote = await self._provider.get_current_quote(symbol)
            atr = await self._get_cached_atr(symbol)

            # --- Relative strength vs. sector benchmark ---
            sector = await self._get_sector(symbol)
            relative_strength: float | None = None
            if self._config.entry.require_relative_strength:
                relative_strength = await self._regime.get_relative_strength(
                    symbol, bars, sector=sector
                )

            # --- Entry signal ---
            assert self._market_open is not None
            signal = detect_vwap_pullback_entry(
                symbol=symbol,
                bars=bars,
                quote=quote,
                market_open=self._market_open,
                settings=self._config.entry,
                atr=atr,
                relative_strength=relative_strength,
                stop_settings=self._stop_settings,
            )

            if signal:
                logger.info(
                    "SIGNAL ▶ {} {} @ {:.4f} | stop={:.4f} | T1={:.4f} | strength={} | RS={:.2f}%",
                    symbol, signal.signal_type.value, signal.entry_price,
                    signal.stop_price, signal.target_1_price, signal.strength.value,
                    relative_strength or 0.0,
                )
                self._emitted_signals[symbol] = signal

                # Register with tracker
                item = self._watchlist.get(symbol) if self._watchlist else None
                gap_pct = item.scan_result.metrics.gap_pct if item else 0.0
                has_catalyst = (
                    item.scan_result.catalyst is not None
                    if item else False
                )
                self._signal_tracker.record_signal(
                    signal_id=str(signal.signal_id),
                    symbol=symbol,
                    strength=signal.strength,
                    generated_at=signal.generated_at,
                    entry_price=signal.entry_price,
                    gap_pct=gap_pct,
                    has_catalyst=has_catalyst,
                )

                if self._on_signal:
                    self._on_signal(signal)

        except Exception as exc:
            logger.debug("Error evaluating {}: {}", symbol, exc)

    # ------------------------------------------------------------------
    # Exit evaluation
    # ------------------------------------------------------------------

    async def _evaluate_exits(self) -> None:
        open_positions = self._trade_manager.get_open_positions()
        if not open_positions:
            return

        for position in open_positions:
            try:
                current_quote = await self._provider.get_current_quote(position.symbol)
                current_price = current_quote.mid
                atr = await self._get_cached_atr(position.symbol)
                now = datetime.now(tz=timezone.utc)

                updated_position, actions = self._trade_manager.update_position(
                    position.position_id, current_price, now, atr
                )

                for action in actions:
                    logger.info(
                        "Position action: {} {} @ {:.4f} — {}",
                        position.symbol, action["action"],
                        action.get("price", 0), action["reason"],
                    )
            except Exception as exc:
                logger.warning("Exit eval error for {}: {}", position.symbol, exc)

    # ------------------------------------------------------------------
    # Signal outcome resolution (called by execution engine)
    # ------------------------------------------------------------------

    def resolve_signal_outcome(
        self,
        signal_id: str,
        exit_price: float,
        exit_reason: "ExitReason",
        net_pnl: float,
        r_multiple: float,
    ) -> None:
        """Called by execution engine when a trade closes. Updates signal tracker."""
        from core.enums import ExitReason
        self._signal_tracker.resolve_signal(
            signal_id=signal_id,
            exit_price=exit_price,
            exit_reason=exit_reason,
            net_pnl=net_pnl,
            r_multiple=r_multiple,
        )

    # ------------------------------------------------------------------
    # Real-time bar injection (WebSocket mode)
    # ------------------------------------------------------------------

    def on_new_bar(self, bar: Bar) -> None:
        symbol = bar.symbol
        if symbol not in self._bar_buffer:
            self._bar_buffer[symbol] = []
        self._bar_buffer[symbol].append(bar)
        if len(self._bar_buffer[symbol]) > 200:
            self._bar_buffer[symbol] = self._bar_buffer[symbol][-200:]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _get_cached_atr(self, symbol: str) -> float:
        cached = await self._cache.get_atr(symbol)
        if cached is not None:
            return cached
        try:
            atr = await self._provider.get_atr(symbol)
            await self._cache.set_atr(symbol, atr)
            return atr
        except Exception:
            return 0.0

    async def _get_sector(self, symbol: str) -> str:
        if symbol in self._sectors:
            return self._sectors[symbol]
        sector = await self._regime.get_sector_for_symbol(symbol)
        self._sectors[symbol] = sector
        # Push to risk engine for sector concentration checks
        if self._risk_engine is not None and sector:
            self._risk_engine.update_sector_map({symbol: sector})
        return sector


    def set_stop_settings(self, settings: StopLossSettings) -> None:
        """Inject StopLossSettings from risk_config so stop placement uses configured values."""
        self._stop_settings = settings

    def reset_daily(self) -> None:
        self._bar_buffer.clear()
        self._emitted_signals.clear()
        self._first_candles.clear()
        self._pm_bars.clear()
        self._sectors.clear()
        self._market_open = None
        self._first_candle_ready = False
        self._regime.reset_daily()
        logger.info("VWAPStrategy state reset for new day")
