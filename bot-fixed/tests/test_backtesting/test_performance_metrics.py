"""
tests/test_backtesting/test_performance_metrics.py
Tests for performance metric computation from trade lists.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from backtesting.performance_metrics import (
    _compute_max_drawdown,
    _compute_sharpe,
    compute_performance_metrics,
)
from core.enums import ExitReason, PositionSide
from core.models import Trade


def _make_trade(
    symbol: str = "AAPL",
    net_pnl: float = 100.0,
    r_multiple: float = 1.5,
    exit_reason: ExitReason = ExitReason.TARGET_1,
    hold_minutes: float = 30.0,
    commission: float = 0.25,
) -> Trade:
    entry = datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
    return Trade(
        trade_id=uuid4(),
        position_id=uuid4(),
        symbol=symbol,
        side=PositionSide.LONG,
        entry_price=100.0,
        exit_price=100.0 + (net_pnl + commission) / 50,  # rough
        quantity=50,
        entry_time=entry,
        exit_time=entry.replace(minute=int(hold_minutes)),
        exit_reason=exit_reason,
        gross_pnl=net_pnl + commission,
        commission=commission,
        net_pnl=net_pnl,
        r_multiple=r_multiple,
        hold_duration_seconds=hold_minutes * 60,
    )


class TestComputePerformanceMetrics:
    def test_empty_trade_list(self):
        m = compute_performance_metrics([], 10_000.0, "2024-01-01", "2024-03-31")
        assert m.total_trades == 0
        assert m.net_pnl == 0.0

    def test_all_winners(self):
        trades = [_make_trade(net_pnl=100.0, r_multiple=1.5) for _ in range(10)]
        m = compute_performance_metrics(trades, 10_000.0, "2024-01-01", "2024-03-31")
        assert m.win_rate == pytest.approx(1.0)
        assert m.net_pnl == pytest.approx(1000.0)
        assert m.profit_factor == float("inf")

    def test_all_losers(self):
        trades = [_make_trade(net_pnl=-50.0, r_multiple=-1.0, exit_reason=ExitReason.STOP_LOSS)
                  for _ in range(5)]
        m = compute_performance_metrics(trades, 10_000.0, "2024-01-01", "2024-03-31")
        assert m.win_rate == pytest.approx(0.0)
        assert m.net_pnl == pytest.approx(-250.0)

    def test_mixed_trades(self):
        winners = [_make_trade(net_pnl=80.0, r_multiple=1.5) for _ in range(6)]
        losers = [_make_trade(net_pnl=-40.0, r_multiple=-1.0, exit_reason=ExitReason.STOP_LOSS)
                  for _ in range(4)]
        m = compute_performance_metrics(winners + losers, 10_000.0, "2024-01-01", "2024-03-31")
        assert m.win_rate == pytest.approx(0.6)
        assert m.winning_trades == 6
        assert m.losing_trades == 4
        assert m.total_trades == 10
        assert m.net_pnl == pytest.approx(480.0 - 160.0)

    def test_profit_factor(self):
        winners = [_make_trade(net_pnl=100.0) for _ in range(3)]
        losers = [_make_trade(net_pnl=-50.0, exit_reason=ExitReason.STOP_LOSS) for _ in range(2)]
        m = compute_performance_metrics(winners + losers, 10_000.0, "2024-01-01", "2024-01-31")
        assert m.profit_factor == pytest.approx(3.0, rel=0.01)

    def test_avg_r_multiple(self):
        trades = [
            _make_trade(r_multiple=2.0),
            _make_trade(r_multiple=-1.0, exit_reason=ExitReason.STOP_LOSS),
        ]
        m = compute_performance_metrics(trades, 10_000.0, "2024-01-01", "2024-01-31")
        assert m.avg_r_multiple == pytest.approx(0.5)

    def test_max_drawdown_non_positive(self):
        trades = [_make_trade(net_pnl=-100.0, exit_reason=ExitReason.STOP_LOSS) for _ in range(5)]
        m = compute_performance_metrics(trades, 10_000.0, "2024-01-01", "2024-01-31")
        assert m.max_drawdown_pct <= 0.0

    def test_commission_counted(self):
        trade = _make_trade(net_pnl=100.0, commission=2.50)
        m = compute_performance_metrics([trade], 10_000.0, "2024-01-01", "2024-01-31")
        assert m.total_commission == pytest.approx(2.50)


class TestMaxDrawdown:
    def test_monotonic_growth_no_drawdown(self):
        equity = [10_000, 10_100, 10_200, 10_300]
        dd = _compute_max_drawdown(equity)
        assert dd == pytest.approx(0.0, abs=0.01)

    def test_drawdown_computed_correctly(self):
        # Peak 10_200, trough 9_900 → DD = (9_900 - 10_200) / 10_200 = -2.94%
        equity = [10_000, 10_200, 9_900, 10_100]
        dd = _compute_max_drawdown(equity)
        assert dd == pytest.approx(-2.941, abs=0.01)

    def test_empty_equity(self):
        assert _compute_max_drawdown([]) == 0.0
        assert _compute_max_drawdown([10_000]) == 0.0


class TestSharpeRatio:
    def test_zero_returns(self):
        returns = [0.0, 0.0, 0.0, 0.0, 0.0]
        assert _compute_sharpe(returns) == 0.0

    def test_positive_sharpe_for_consistent_gains(self):
        returns = [0.01] * 50
        sharpe = _compute_sharpe(returns)
        assert sharpe > 0

    def test_empty_returns(self):
        assert _compute_sharpe([]) == 0.0
