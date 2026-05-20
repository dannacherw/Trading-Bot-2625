"""
backtesting/backtester.py
Historical replay backtester for the VWAP pullback strategy.
Replays day-by-day using stored bar data.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from backtesting.performance_metrics import compute_performance_metrics
from core.config import StrategyConfig, RiskConfig
from core.enums import BarTimeframe
from core.models import Bar, PerformanceMetrics, Signal, Trade
from execution.fill_simulator import simulate_fill
from market_data.base_provider import BaseDataProvider
from risk.risk_engine import RiskEngine
from strategy.entry_signals import detect_vwap_pullback_entry
from strategy.exits import evaluate_all_exits
from strategy.trade_manager import TradeManager


class Backtester:
    """
    Event-driven bar-by-bar backtester.

    For each trading day:
    1. Load intraday 1-min bars for all watchlisted symbols
    2. Replay bars chronologically
    3. Check for entry signals each bar
    4. Manage open positions (exits, stops)
    5. Collect trades
    """

    def __init__(
        self,
        provider: BaseDataProvider,
        strategy_config: StrategyConfig,
        risk_config: RiskConfig,
        starting_equity: float = 10_000.0,
        commission_per_share: float = 0.005,
    ) -> None:
        self._provider = provider
        self._strategy_cfg = strategy_config
        self._risk_cfg = risk_config
        self._starting_equity = starting_equity
        self._commission = commission_per_share

    async def run(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> PerformanceMetrics:
        """Run a full backtest and return performance metrics."""
        logger.info(
            "Backtesting {} symbols from {} to {}",
            len(symbols), start_date.date(), end_date.date(),
        )

        all_trades: list[Trade] = []
        equity = self._starting_equity
        current_date = start_date

        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() in (5, 6):
                current_date += timedelta(days=1)
                continue

            day_trades = await self._run_day(symbols, current_date, equity)
            all_trades.extend(day_trades)
            equity += sum(t.net_pnl for t in day_trades)
            current_date += timedelta(days=1)

        metrics = compute_performance_metrics(
            trades=all_trades,
            starting_equity=self._starting_equity,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )
        logger.info(
            "Backtest complete: {} trades | WR={:.1f}% | R={:.2f} | PnL=${:.2f}",
            metrics.total_trades,
            metrics.win_rate * 100,
            metrics.avg_r_multiple,
            metrics.net_pnl,
        )
        return metrics

    async def _run_day(
        self, symbols: list[str], date: datetime, equity: float
    ) -> list[Trade]:
        """
        Run a single trading day simulation.

        LOOK-AHEAD BIAS FIX (v2):
        All historical lookbacks (ATR, avg daily volume) are computed using
        data strictly available BEFORE the simulation date. The provider
        is called with end_date = date - 1 day to enforce this boundary.
        """
        market_open = date.replace(hour=13, minute=30, second=0, tzinfo=timezone.utc)
        risk_engine = RiskEngine(self._risk_cfg)
        risk_engine.update_equity(equity)
        trade_manager = TradeManager(self._strategy_cfg.exit, self._commission)

        # Compute ATR and avg daily volume BEFORE the simulation date
        # to avoid using any information from the day being simulated
        lookback_end = date - timedelta(days=1)
        lookback_start = date - timedelta(days=40)  # Enough for 14-period ATR + buffer

        per_symbol_atr: dict[str, float] = {}
        per_symbol_avg_vol: dict[str, float] = {}

        for symbol in symbols:
            try:
                hist_bars = await self._provider.get_bars(
                    symbol, BarTimeframe.DAY_1, lookback_start, lookback_end
                )
                if len(hist_bars) >= 14:
                    from scanner.metrics import compute_atr
                    per_symbol_atr[symbol] = compute_atr(hist_bars, period=14)
                    # Average daily volume over the last 20 sessions before simulation date
                    recent_20 = hist_bars[-20:]
                    per_symbol_avg_vol[symbol] = (
                        sum(b.volume for b in recent_20) / len(recent_20)
                    )
                else:
                    per_symbol_atr[symbol] = 0.50   # Fallback
                    per_symbol_avg_vol[symbol] = 1_000_000
            except Exception:
                per_symbol_atr[symbol] = 0.50
                per_symbol_avg_vol[symbol] = 1_000_000

        # Collect all intraday bars for the simulation date
        all_bar_streams: dict[str, list[Bar]] = {}
        for symbol in symbols:
            try:
                bars = await self._provider.get_intraday_bars(symbol, date)
                all_bar_streams[symbol] = bars
            except Exception:
                pass

        if not all_bar_streams:
            return []

        all_timestamps: list[datetime] = sorted(
            {b.timestamp for bars in all_bar_streams.values() for b in bars}
        )
        bar_lookup: dict[str, dict[datetime, Bar]] = {
            sym: {b.timestamp: b for b in bars}
            for sym, bars in all_bar_streams.items()
        }
        acc_bars: dict[str, list[Bar]] = {sym: [] for sym in symbols}
        completed_trades: list[Trade] = []
        already_traded: set[str] = set()

        for ts in all_timestamps:
            for symbol, lookup in bar_lookup.items():
                bar = lookup.get(ts)
                if bar is None:
                    continue
                acc_bars[symbol].append(bar)

                # ---- Entry signals ----
                if (
                    symbol not in already_traded
                    and not risk_engine.is_halted
                    and len(acc_bars[symbol]) >= 5
                ):
                    from core.models import Quote
                    quote = Quote(
                        symbol=symbol, timestamp=ts,
                        bid=bar.close - 0.01, ask=bar.close + 0.01,
                    )
                    atr = per_symbol_atr.get(symbol, 0.50)
                    try:
                        signal = detect_vwap_pullback_entry(
                            symbol=symbol,
                            bars=acc_bars[symbol],
                            quote=quote,
                            market_open=market_open,
                            settings=self._strategy_cfg.entry,
                            atr=atr,
                            relative_strength=None,  # Not computed in backtest
                        )
                    except Exception:
                        signal = None

                    if signal is not None:
                        avg_vol = per_symbol_avg_vol.get(symbol, 1_000_000)
                        validation = risk_engine.validate_signal(
                            signal=signal,
                            spread_pct=0.02,
                            avg_daily_volume=avg_vol,
                            atr=atr,
                            avg_atr=atr,
                            current_equity=equity,
                        )
                        if validation.approved_quantity > 0:
                            position = trade_manager.open_position(
                                signal=signal,
                                filled_price=signal.entry_price,
                                filled_quantity=validation.approved_quantity,
                                filled_at=ts,
                            )
                            risk_engine.register_trade_opened(symbol, position)
                            already_traded.add(symbol)

                # ---- Exit evaluation ----
                for pos in trade_manager.get_open_positions():
                    if pos.symbol != symbol:
                        continue
                    atr = per_symbol_atr.get(symbol, 0.50)
                    updated_pos, actions = trade_manager.update_position(
                        pos.position_id, bar.close, ts, atr=atr
                    )
                    for action in actions:
                        if action["action"] == "full_exit":
                            from core.enums import ExitReason
                            closed = updated_pos.model_copy(update={
                                "exit_price": action["price"],
                                "exit_time": ts,
                                "exit_reason": ExitReason(action["reason"]),
                            })
                            trade = trade_manager.build_trade_record(closed)
                            completed_trades.append(trade)
                            risk_engine.register_trade_closed(symbol, trade.net_pnl)

        return completed_trades
