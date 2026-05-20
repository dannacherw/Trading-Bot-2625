"""
backtesting/performance_metrics.py
Compute performance statistics from a list of completed trades.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from core.models import PerformanceMetrics, Trade


def compute_performance_metrics(
    trades: list[Trade],
    starting_equity: float,
    start_date: str,
    end_date: str,
    trading_days: int = 252,
) -> PerformanceMetrics:
    """Full performance statistics from a trade list."""
    if not trades:
        return _empty_metrics(start_date, end_date)

    net_pnls = [t.net_pnl for t in trades]
    winners = [t for t in trades if t.is_winner]
    losers = [t for t in trades if not t.is_winner]

    total_net_pnl = sum(net_pnls)
    gross_profit = sum(t.net_pnl for t in winners)
    gross_loss = abs(sum(t.net_pnl for t in losers))

    win_rate = len(winners) / len(trades)
    total_return_pct = total_net_pnl / starting_equity * 100.0

    # Approximate annualised return
    calendar_days = max(1, _days_between(start_date, end_date))
    years = calendar_days / 365.0
    annualised = (((1 + total_return_pct / 100.0) ** (1 / years)) - 1) * 100.0 if years > 0 else 0.0

    # Equity curve for drawdown/Sharpe
    equity = starting_equity
    equity_curve = [equity]
    for t in sorted(trades, key=lambda x: x.exit_time):
        equity += t.net_pnl
        equity_curve.append(equity)

    max_dd_pct = _compute_max_drawdown(equity_curve)
    daily_returns = _compute_daily_returns(trades, starting_equity)
    sharpe = _compute_sharpe(daily_returns, trading_days)
    sortino = _compute_sortino(daily_returns, trading_days)

    return PerformanceMetrics(
        start_date=start_date,
        end_date=end_date,
        total_trades=len(trades),
        winning_trades=len(winners),
        losing_trades=len(losers),
        win_rate=round(win_rate, 4),
        total_return_pct=round(total_return_pct, 4),
        annualized_return_pct=round(annualised, 4),
        max_drawdown_pct=round(max_dd_pct, 4),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        profit_factor=round(gross_profit / gross_loss, 4) if gross_loss > 0 else float("inf"),
        avg_r_multiple=round(float(np.mean([t.r_multiple for t in trades])), 4),
        avg_winner_r=round(float(np.mean([t.r_multiple for t in winners])), 4) if winners else 0.0,
        avg_loser_r=round(float(np.mean([t.r_multiple for t in losers])), 4) if losers else 0.0,
        avg_hold_minutes=round(float(np.mean([t.hold_duration_seconds / 60 for t in trades])), 2),
        total_commission=round(sum(t.commission for t in trades), 2),
        net_pnl=round(total_net_pnl, 4),
    )


def _empty_metrics(start_date: str, end_date: str) -> PerformanceMetrics:
    return PerformanceMetrics(
        start_date=start_date, end_date=end_date,
        total_trades=0, winning_trades=0, losing_trades=0,
        win_rate=0.0, total_return_pct=0.0, annualized_return_pct=0.0,
        max_drawdown_pct=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
        profit_factor=0.0, avg_r_multiple=0.0, avg_winner_r=0.0,
        avg_loser_r=0.0, avg_hold_minutes=0.0, total_commission=0.0, net_pnl=0.0,
    )


def _compute_max_drawdown(equity_curve: list[float]) -> float:
    if len(equity_curve) < 2:
        return 0.0
    arr = np.array(equity_curve)
    rolling_max = np.maximum.accumulate(arr)
    drawdowns = (arr - rolling_max) / rolling_max * 100.0
    return float(np.min(drawdowns))


def _compute_daily_returns(trades: list[Trade], starting_equity: float) -> list[float]:
    """Compute daily PnL returns as a fraction of equity."""
    from collections import defaultdict
    daily_pnl: dict[str, float] = defaultdict(float)
    for t in trades:
        day = t.exit_time.strftime("%Y-%m-%d")
        daily_pnl[day] += t.net_pnl
    equity = starting_equity
    returns = []
    for day in sorted(daily_pnl):
        ret = daily_pnl[day] / equity
        returns.append(ret)
        equity += daily_pnl[day]
    return returns


def _compute_sharpe(daily_returns: list[float], trading_days: int = 252) -> float:
    if len(daily_returns) < 2:
        return 0.0
    arr = np.array(daily_returns)
    std = np.std(arr, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(arr) / std * math.sqrt(trading_days))


def _compute_sortino(daily_returns: list[float], trading_days: int = 252) -> float:
    if len(daily_returns) < 2:
        return 0.0
    arr = np.array(daily_returns)
    neg = arr[arr < 0]
    downside_std = np.std(neg, ddof=1) if len(neg) > 1 else 0.0
    if downside_std == 0:
        return 0.0
    return float(np.mean(arr) / downside_std * math.sqrt(trading_days))


def _days_between(start: str, end: str) -> int:
    try:
        d1 = datetime.strptime(start, "%Y-%m-%d")
        d2 = datetime.strptime(end, "%Y-%m-%d")
        return abs((d2 - d1).days)
    except ValueError:
        return 252
