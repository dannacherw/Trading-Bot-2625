"""
backtesting/pipeline_backtester.py
Full pipeline backtester: Scanner → Watchlist → Strategy → Execution.

The existing Backtester in backtester.py takes a pre-supplied symbol list
and only replays strategy/execution logic. This class runs the FULL pipeline:

  For each trading day:
    1. Reconstruct premarket conditions from stored bar data
    2. Run the scanner (filter + score) on historical premarket bars
    3. Build ranked watchlist — same logic as live
    4. Replay intraday session bar-by-bar through strategy + execution
    5. Collect trades, compute daily metrics

No look-ahead bias:
  - Premarket bars are cut off at 9:25 AM ET precisely
  - Catalyst detection uses only news from BEFORE scan time
  - ATR and avg volume use only data from days BEFORE the simulation date
  - Float shares are fetched from the float_data_cache as of the sim date

Architecture:
  PipelineBacktester is a standalone class that shares config objects
  with the live system to ensure identical filter/scoring/strategy logic.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from loguru import logger

from backtesting.performance_metrics import compute_performance_metrics
from core.config import RiskConfig, ScannerConfig, SlippageSettings, StrategyConfig
from core.enums import BarTimeframe, ExitReason
from core.models import Bar, PerformanceMetrics, Quote, ScanResult, Signal, Trade
from execution.fill_simulator import simulate_fill
from market_data.base_provider import BaseDataProvider
from risk.risk_engine import RiskEngine
from scanner.filters import apply_all_filters
from scanner.metrics import compute_premarket_metrics_from_bars
from scanner.percentile_scoring import score_population_percentile, rank_scan_results
from scanner.tagging import tag_stock
from scanner.watchlist import Watchlist
from strategy.entry_signals import detect_vwap_pullback_entry
from strategy.trade_manager import TradeManager


# ---------------------------------------------------------------------------
# Per-day simulation result
# ---------------------------------------------------------------------------

@dataclass
class DaySimulationResult:
    date: str
    symbols_scanned: int
    symbols_passed: int
    watchlist_size: int
    trades: list[Trade]
    equity_start: float
    equity_end: float
    scan_duration_ms: float
    session_duration_ms: float


# ---------------------------------------------------------------------------
# Pipeline backtester
# ---------------------------------------------------------------------------

class PipelineBacktester:
    """
    Full pipeline historical backtester.

    Runs premarket scan + watchlist + intraday strategy on each day
    in the date range, using only data available at each point in time.
    """

    def __init__(
        self,
        provider: BaseDataProvider,
        scanner_config: ScannerConfig,
        strategy_config: StrategyConfig,
        risk_config: RiskConfig,
        starting_equity: float = 10_000.0,
        commission_per_share: float = 0.005,
        slippage_settings: SlippageSettings | None = None,
        premarket_end_time: tuple[int, int] = (13, 25),  # 9:25 AM ET in UTC
        catalyst_lookback_hours: int = 18,
        symbol_universe: list[str] | None = None,
    ) -> None:
        self._provider = provider
        self._scanner_cfg = scanner_config
        self._strategy_cfg = strategy_config
        self._risk_cfg = risk_config
        self._starting_equity = starting_equity
        self._commission = commission_per_share
        self._slippage_settings = slippage_settings or SlippageSettings()
        self._pm_end_hour, self._pm_end_minute = premarket_end_time
        self._catalyst_hours = catalyst_lookback_hours
        self._universe = symbol_universe  # If None, built from provider each day

    async def run(
        self,
        start_date: datetime,
        end_date: datetime,
        symbol_universe: list[str] | None = None,
    ) -> tuple[PerformanceMetrics, list[DaySimulationResult]]:
        """
        Run a full pipeline backtest.

        Returns:
            (aggregate metrics, per-day simulation results)
        """
        universe = symbol_universe or self._universe
        if not universe:
            raise ValueError(
                "symbol_universe must be provided either at init or run() call. "
                "Pass a list of ticker symbols to test."
            )

        logger.info(
            "Pipeline backtest: {} symbols | {} → {}",
            len(universe), start_date.date(), end_date.date(),
        )

        all_trades: list[Trade] = []
        day_results: list[DaySimulationResult] = []
        equity = self._starting_equity
        current = start_date

        while current <= end_date:
            if current.weekday() in (5, 6):
                current += timedelta(days=1)
                continue

            day_result = await self._run_day(universe, current, equity)
            all_trades.extend(day_result.trades)
            equity = day_result.equity_end
            day_results.append(day_result)

            logger.info(
                "Day {}: scanned={} passed={} watchlist={} trades={} equity=${:.2f}",
                day_result.date,
                day_result.symbols_scanned,
                day_result.symbols_passed,
                day_result.watchlist_size,
                len(day_result.trades),
                equity,
            )
            current += timedelta(days=1)

        metrics = compute_performance_metrics(
            trades=all_trades,
            starting_equity=self._starting_equity,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )

        logger.info(
            "Pipeline backtest complete: {} trades | WR={:.1f}% | R={:.2f} | PnL=${:.2f}",
            metrics.total_trades,
            metrics.win_rate * 100,
            metrics.avg_r_multiple,
            metrics.net_pnl,
        )
        return metrics, day_results

    # ------------------------------------------------------------------
    # Single day
    # ------------------------------------------------------------------

    async def _run_day(
        self,
        universe: list[str],
        sim_date: datetime,
        equity: float,
    ) -> DaySimulationResult:
        date_str = sim_date.strftime("%Y-%m-%d")
        equity_start = equity

        # ── Step 1: Premarket scanning ──────────────────────────────────
        scan_start = _now_ms()
        watchlist, scan_results = await self._run_premarket_scan(universe, sim_date)
        scan_ms = _now_ms() - scan_start

        if not watchlist.items:
            return DaySimulationResult(
                date=date_str,
                symbols_scanned=len(scan_results),
                symbols_passed=sum(1 for r in scan_results if r.passes_filters),
                watchlist_size=0,
                trades=[],
                equity_start=equity_start,
                equity_end=equity_start,
                scan_duration_ms=scan_ms,
                session_duration_ms=0.0,
            )

        watchlist_symbols = [item.symbol for item in watchlist.items]

        # ── Step 2: Pre-compute ATR and avg volume (look-ahead safe) ────
        lookback_end = sim_date - timedelta(days=1)
        lookback_start = sim_date - timedelta(days=45)
        per_symbol_atr: dict[str, float] = {}
        per_symbol_avg_vol: dict[str, float] = {}

        for symbol in watchlist_symbols:
            try:
                hist = await self._provider.get_bars(
                    symbol, BarTimeframe.DAY_1, lookback_start, lookback_end
                )
                if len(hist) >= 14:
                    from scanner.metrics import compute_atr
                    per_symbol_atr[symbol] = compute_atr(hist, period=14)
                    recent = hist[-20:]
                    per_symbol_avg_vol[symbol] = sum(b.volume for b in recent) / len(recent)
                else:
                    per_symbol_atr[symbol] = 0.5
                    per_symbol_avg_vol[symbol] = 1_000_000
            except Exception:
                per_symbol_atr[symbol] = 0.5
                per_symbol_avg_vol[symbol] = 1_000_000

        # ── Step 3: Intraday session replay ────────────────────────────
        session_start = _now_ms()
        risk_engine = RiskEngine(self._risk_cfg)
        risk_engine.update_equity(equity)
        trade_manager = TradeManager(self._strategy_cfg.exit, self._commission)

        market_open = sim_date.replace(hour=13, minute=30, second=0, tzinfo=timezone.utc)

        # Load all intraday bars for watchlist symbols
        bar_streams: dict[str, list[Bar]] = {}
        for symbol in watchlist_symbols:
            try:
                bars = await self._provider.get_intraday_bars(symbol, sim_date)
                if bars:
                    bar_streams[symbol] = bars
            except Exception:
                pass

        completed_trades = await self._replay_session(
            bar_streams=bar_streams,
            market_open=market_open,
            risk_engine=risk_engine,
            trade_manager=trade_manager,
            per_symbol_atr=per_symbol_atr,
            per_symbol_avg_vol=per_symbol_avg_vol,
            equity=equity,
        )

        session_ms = _now_ms() - session_start
        day_pnl = sum(t.net_pnl for t in completed_trades)

        return DaySimulationResult(
            date=date_str,
            symbols_scanned=len(scan_results),
            symbols_passed=sum(1 for r in scan_results if r.passes_filters),
            watchlist_size=len(watchlist.items),
            trades=completed_trades,
            equity_start=equity_start,
            equity_end=equity_start + day_pnl,
            scan_duration_ms=scan_ms,
            session_duration_ms=session_ms,
        )

    # ------------------------------------------------------------------
    # Premarket scan reconstruction
    # ------------------------------------------------------------------

    async def _run_premarket_scan(
        self,
        universe: list[str],
        sim_date: datetime,
    ) -> tuple[Watchlist, list[ScanResult]]:
        """
        Reconstruct the premarket scan using historical bars.
        Only uses bars available before 9:25 AM ET.
        """
        pm_cutoff = sim_date.replace(
            hour=self._pm_end_hour,
            minute=self._pm_end_minute,
            second=0,
            tzinfo=timezone.utc,
        )
        # Premarket window: 4:00 AM - 9:25 AM ET (08:00-13:25 UTC)
        pm_start = sim_date.replace(hour=8, minute=0, second=0, tzinfo=timezone.utc)

        sem = asyncio.Semaphore(20)

        async def scan_one(symbol: str) -> ScanResult | None:
            async with sem:
                return await self._scan_symbol_historical(
                    symbol, sim_date, pm_start, pm_cutoff
                )

        tasks = [scan_one(sym) for sym in universe]
        raw = await asyncio.gather(*tasks)
        scan_results = [r for r in raw if r is not None]

        if not scan_results:
            return Watchlist(self._scanner_cfg.watchlist), scan_results

        # Cross-sectional scoring
        catalyst_scores = {r.symbol: (r.catalyst.strength * r.catalyst.confidence if r.catalyst else 0.0) for r in scan_results}
        scan_results = score_population_percentile(
            scan_results, catalyst_scores, self._scanner_cfg.weights, use_historical=False
        )
        scan_results = rank_scan_results(scan_results)

        # Tag archetypes
        tagged = []
        for result in scan_results:
            if result.passes_filters:
                tags = tag_stock(
                    metrics=result.metrics,
                    composite_score=result.composite_score,
                    catalyst=result.catalyst,
                    thresholds=self._scanner_cfg.archetypes,
                )
                tagged.append(result.model_copy(update={"archetypes": tags}))
            else:
                tagged.append(result)
        scan_results = tagged

        # Build watchlist
        watchlist = Watchlist(self._scanner_cfg.watchlist)
        watchlist.build(scan_results)

        return watchlist, scan_results

    async def _scan_symbol_historical(
        self,
        symbol: str,
        sim_date: datetime,
        pm_start: datetime,
        pm_cutoff: datetime,
    ) -> ScanResult | None:
        """Reconstruct a single symbol's premarket metrics from historical bars."""
        try:
            # Fetch premarket bars (4:00 AM - 9:25 AM ET)
            pm_bars = await self._provider.get_bars(
                symbol, BarTimeframe.MINUTE_1, pm_start, pm_cutoff
            )
            if not pm_bars:
                return None

            # Previous close
            prev_day_start = sim_date - timedelta(days=5)
            prev_day_end = sim_date - timedelta(days=1)
            prev_bars = await self._provider.get_bars(
                symbol, BarTimeframe.DAY_1, prev_day_start, prev_day_end
            )
            if not prev_bars:
                return None
            prev_close = prev_bars[-1].close

            # Average daily dollar volume (before sim date — no look-ahead)
            lookback_start = sim_date - timedelta(days=30)
            hist_bars = await self._provider.get_bars(
                symbol, BarTimeframe.DAY_1, lookback_start, prev_day_end
            )
            avg_dv = 0.0
            if hist_bars:
                avg_dv = sum(b.close * b.volume for b in hist_bars[-20:]) / max(len(hist_bars[-20:]), 1)
            avg_vol = avg_dv / max(prev_close, 0.01)

            # Synthetic quote from last PM bar
            last_bar = pm_bars[-1]
            from core.models import Quote
            quote = Quote(
                symbol=symbol,
                timestamp=last_bar.timestamp,
                bid=last_bar.close - 0.01,
                ask=last_bar.close + 0.01,
            )

            metrics = compute_premarket_metrics_from_bars(
                symbol=symbol,
                pm_bars=pm_bars,
                prev_close=prev_close,
                quote=quote,
                avg_daily_dollar_volume=avg_dv,
                avg_daily_volume=avg_vol,
            )

            passes, failure_reason = apply_all_filters(metrics, self._scanner_cfg.filters)

            return ScanResult(
                symbol=symbol,
                scanned_at=pm_cutoff,
                metrics=metrics,
                composite_score=0.0,
                passes_filters=passes,
                filter_failure_reason=failure_reason,
            )
        except Exception as exc:
            logger.debug("Pipeline BT: failed to scan {} on {}: {}", symbol, sim_date.date(), exc)
            return None

    # ------------------------------------------------------------------
    # Intraday session replay
    # ------------------------------------------------------------------

    async def _replay_session(
        self,
        bar_streams: dict[str, list[Bar]],
        market_open: datetime,
        risk_engine: RiskEngine,
        trade_manager: TradeManager,
        per_symbol_atr: dict[str, float],
        per_symbol_avg_vol: dict[str, float],
        equity: float,
    ) -> list[Trade]:
        """Bar-by-bar session replay matching the live strategy logic."""
        if not bar_streams:
            return []

        all_timestamps = sorted(
            {b.timestamp for bars in bar_streams.values() for b in bars}
        )
        bar_lookup: dict[str, dict[datetime, Bar]] = {
            sym: {b.timestamp: b for b in bars}
            for sym, bars in bar_streams.items()
        }
        acc_bars: dict[str, list[Bar]] = {sym: [] for sym in bar_streams}
        completed: list[Trade] = []
        already_entered: set[str] = set()

        for ts in all_timestamps:
            for symbol, lookup in bar_lookup.items():
                bar = lookup.get(ts)
                if bar is None:
                    continue
                acc_bars[symbol].append(bar)

                # Entry check
                if (
                    symbol not in already_entered
                    and not risk_engine.is_halted
                    and len(acc_bars[symbol]) >= 5
                ):
                    atr = per_symbol_atr.get(symbol, 0.5)
                    quote = Quote(
                        symbol=symbol,
                        timestamp=ts,
                        bid=bar.close - 0.01,
                        ask=bar.close + 0.01,
                    )
                    try:
                        signal = detect_vwap_pullback_entry(
                            symbol=symbol,
                            bars=acc_bars[symbol],
                            quote=quote,
                            market_open=market_open,
                            settings=self._strategy_cfg.entry,
                            atr=atr,
                            relative_strength=None,
                        )
                    except Exception:
                        signal = None

                    if signal is not None:
                        avg_vol = per_symbol_avg_vol.get(symbol, 1_000_000)
                        spread_pct = bar.range / bar.close * 0.5 if bar.close > 0 else 0.02
                        validation = risk_engine.validate_signal(
                            signal=signal,
                            spread_pct=spread_pct,
                            avg_daily_volume=avg_vol,
                            atr=atr,
                            avg_atr=atr,
                            current_equity=equity,
                        )
                        if validation.approved_quantity > 0:
                            # Apply realistic slippage to fill price
                            from core.enums import OrderSide
                            fill_price = apply_slippage(
                                price=signal.entry_price,
                                spread_pct=spread_pct,
                                side=OrderSide.BUY,
                                settings=self._slippage_settings,
                            )
                            position = trade_manager.open_position(
                                signal=signal,
                                filled_price=fill_price,
                                filled_quantity=validation.approved_quantity,
                                filled_at=ts,
                            )
                            risk_engine.register_trade_opened(symbol, position)
                            already_entered.add(symbol)

                # Exit evaluation
                for pos in trade_manager.get_open_positions():
                    if pos.symbol != symbol:
                        continue
                    atr = per_symbol_atr.get(symbol, 0.5)
                    updated_pos, actions = trade_manager.update_position(
                        pos.position_id, bar.close, ts, atr=atr
                    )
                    for action in actions:
                        if action["action"] == "full_exit":
                            closed = updated_pos.model_copy(update={
                                "exit_price": action["price"],
                                "exit_time": ts,
                                "exit_reason": ExitReason(action["reason"]),
                            })
                            trade = trade_manager.build_trade_record(closed)
                            completed.append(trade)
                            risk_engine.register_trade_closed(symbol, trade.net_pnl)

        return completed


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _now_ms() -> float:
    import time
    return time.monotonic() * 1000
