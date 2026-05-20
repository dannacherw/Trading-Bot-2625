"""
tests/test_backtesting_v2/test_pipeline_backtester.py
Tests for the full pipeline backtester and walk-forward validator.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backtesting.pipeline_backtester import (
    DaySimulationResult,
    PipelineBacktester,
    _now_ms,
)
from backtesting.walk_forward import (
    WalkForwardConfig,
    WalkForwardReport,
    WalkForwardValidator,
    _add_months,
    _days_in_month,
)
from core.config import RiskConfig, ScannerConfig, StrategyConfig, AccountSettings
from core.models import Bar, PerformanceMetrics
from core.enums import BarTimeframe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bar(
    symbol: str = "AAPL",
    dt: datetime | None = None,
    open_: float = 100.0,
    high: float = 101.0,
    low: float = 99.5,
    close: float = 100.5,
    volume: int = 10_000,
) -> Bar:
    dt = dt or datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
    return Bar(
        symbol=symbol, timestamp=dt,
        open=open_, high=high, low=low, close=close,
        volume=volume, timeframe=BarTimeframe.MINUTE_1,
    )


def _empty_metrics() -> PerformanceMetrics:
    return PerformanceMetrics(
        start_date="2024-01-01", end_date="2024-01-31",
        total_trades=0, winning_trades=0, losing_trades=0,
        win_rate=0.0, total_return_pct=0.0, annualized_return_pct=0.0,
        max_drawdown_pct=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
        profit_factor=0.0, avg_r_multiple=0.0, avg_winner_r=0.0,
        avg_loser_r=0.0, avg_hold_minutes=0.0, total_commission=0.0, net_pnl=0.0,
    )


def _metrics(
    trades: int = 10,
    win_rate: float = 0.6,
    avg_r: float = 0.5,
    sharpe: float = 1.2,
    net_pnl: float = 500.0,
) -> PerformanceMetrics:
    wins = int(trades * win_rate)
    return PerformanceMetrics(
        start_date="2024-01-01", end_date="2024-01-31",
        total_trades=trades, winning_trades=wins, losing_trades=trades - wins,
        win_rate=win_rate, total_return_pct=5.0, annualized_return_pct=60.0,
        max_drawdown_pct=-2.0, sharpe_ratio=sharpe, sortino_ratio=sharpe * 1.2,
        profit_factor=1.8, avg_r_multiple=avg_r, avg_winner_r=avg_r * 1.5,
        avg_loser_r=-1.0, avg_hold_minutes=25.0, total_commission=15.0, net_pnl=net_pnl,
    )


def _make_scanner_config() -> ScannerConfig:
    return ScannerConfig()


def _make_strategy_config() -> StrategyConfig:
    return StrategyConfig()


def _make_risk_config() -> RiskConfig:
    return RiskConfig(account=AccountSettings(starting_equity=10_000.0))


def _make_provider() -> AsyncMock:
    provider = AsyncMock()
    provider.get_bars = AsyncMock(return_value=[])
    provider.get_intraday_bars = AsyncMock(return_value=[])
    provider.get_premarket_bars = AsyncMock(return_value=[])
    return provider


# ---------------------------------------------------------------------------
# _now_ms
# ---------------------------------------------------------------------------

class TestNowMs:
    def test_returns_positive_float(self):
        t = _now_ms()
        assert isinstance(t, float)
        assert t > 0


# ---------------------------------------------------------------------------
# PipelineBacktester — basic construction
# ---------------------------------------------------------------------------

class TestPipelineBacktesterConstruction:
    def test_requires_universe_at_run(self):
        bt = PipelineBacktester(
            provider=_make_provider(),
            scanner_config=_make_scanner_config(),
            strategy_config=_make_strategy_config(),
            risk_config=_make_risk_config(),
        )
        with pytest.raises(ValueError, match="symbol_universe"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                bt.run(
                    datetime(2024, 1, 15, tzinfo=timezone.utc),
                    datetime(2024, 1, 15, tzinfo=timezone.utc),
                    symbol_universe=None,
                )
            )

    def test_init_with_universe(self):
        bt = PipelineBacktester(
            provider=_make_provider(),
            scanner_config=_make_scanner_config(),
            strategy_config=_make_strategy_config(),
            risk_config=_make_risk_config(),
            symbol_universe=["AAPL", "TSLA"],
        )
        assert bt is not None


# ---------------------------------------------------------------------------
# PipelineBacktester — run with no passing scans
# ---------------------------------------------------------------------------

class TestPipelineBacktesterRun:
    @pytest.mark.asyncio
    async def test_run_with_empty_bars_returns_zero_trades(self):
        provider = _make_provider()
        provider.get_bars = AsyncMock(return_value=[])
        provider.get_intraday_bars = AsyncMock(return_value=[])

        bt = PipelineBacktester(
            provider=provider,
            scanner_config=_make_scanner_config(),
            strategy_config=_make_strategy_config(),
            risk_config=_make_risk_config(),
        )

        start = datetime(2024, 1, 15, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, tzinfo=timezone.utc)

        metrics, day_results = await bt.run(start, end, symbol_universe=["AAPL"])
        assert metrics.total_trades == 0
        assert len(day_results) == 1
        assert day_results[0].trades == []

    @pytest.mark.asyncio
    async def test_weekends_skipped(self):
        provider = _make_provider()
        bt = PipelineBacktester(
            provider=provider,
            scanner_config=_make_scanner_config(),
            strategy_config=_make_strategy_config(),
            risk_config=_make_risk_config(),
        )
        # Jan 13-14 2024 is Sat-Sun
        start = datetime(2024, 1, 13, tzinfo=timezone.utc)
        end = datetime(2024, 1, 14, tzinfo=timezone.utc)
        metrics, day_results = await bt.run(start, end, symbol_universe=["AAPL"])
        assert len(day_results) == 0

    @pytest.mark.asyncio
    async def test_day_result_structure(self):
        provider = _make_provider()
        bt = PipelineBacktester(
            provider=provider,
            scanner_config=_make_scanner_config(),
            strategy_config=_make_strategy_config(),
            risk_config=_make_risk_config(),
        )
        start = datetime(2024, 1, 15, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, tzinfo=timezone.utc)
        _, day_results = await bt.run(start, end, symbol_universe=["AAPL"])

        assert len(day_results) == 1
        d = day_results[0]
        assert d.date == "2024-01-15"
        assert isinstance(d.equity_start, float)
        assert isinstance(d.equity_end, float)
        assert isinstance(d.scan_duration_ms, float)

    @pytest.mark.asyncio
    async def test_equity_unchanged_with_no_trades(self):
        provider = _make_provider()
        bt = PipelineBacktester(
            provider=provider,
            scanner_config=_make_scanner_config(),
            strategy_config=_make_strategy_config(),
            risk_config=_make_risk_config(),
            starting_equity=25_000.0,
        )
        start = datetime(2024, 1, 15, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, tzinfo=timezone.utc)
        _, day_results = await bt.run(start, end, symbol_universe=["AAPL"])
        assert day_results[0].equity_start == 25_000.0
        assert day_results[0].equity_end == 25_000.0


# ---------------------------------------------------------------------------
# _add_months helper
# ---------------------------------------------------------------------------

class TestAddMonths:
    def test_add_one_month(self):
        dt = datetime(2024, 1, 15, tzinfo=timezone.utc)
        result = _add_months(dt, 1)
        assert result.month == 2
        assert result.day == 15

    def test_add_across_year(self):
        dt = datetime(2024, 11, 1, tzinfo=timezone.utc)
        result = _add_months(dt, 2)
        assert result.year == 2025
        assert result.month == 1

    def test_handles_month_end(self):
        # Jan 31 + 1 month = Feb 28/29
        dt = datetime(2024, 1, 31, tzinfo=timezone.utc)
        result = _add_months(dt, 1)
        assert result.month == 2
        assert result.day <= 29  # Respects Feb length

    def test_add_six_months(self):
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        result = _add_months(dt, 6)
        assert result.month == 7
        assert result.year == 2024


# ---------------------------------------------------------------------------
# WalkForwardValidator — fold scheduling
# ---------------------------------------------------------------------------

class TestFoldScheduling:
    def test_correct_number_of_folds(self):
        validator = WalkForwardValidator(
            backtester=MagicMock(),
            config=WalkForwardConfig(train_months=6, test_months=1, step_months=1),
        )
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)
        folds = validator._build_fold_schedule(start, end)
        # With 6m train, 1m test, 1m step over 12 months → 6 folds
        assert len(folds) == 6

    def test_no_folds_if_range_too_short(self):
        validator = WalkForwardValidator(
            backtester=MagicMock(),
            config=WalkForwardConfig(train_months=6, test_months=1, step_months=1),
        )
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 3, 31, tzinfo=timezone.utc)  # Only 3 months
        folds = validator._build_fold_schedule(start, end)
        assert len(folds) == 0

    def test_fold_train_test_no_overlap(self):
        validator = WalkForwardValidator(
            backtester=MagicMock(),
            config=WalkForwardConfig(train_months=3, test_months=1, step_months=1),
        )
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)
        folds = validator._build_fold_schedule(start, end)
        for train_start, train_end, test_start, test_end in folds:
            assert test_start > train_end

    def test_fold_test_start_follows_train_end(self):
        validator = WalkForwardValidator(
            backtester=MagicMock(),
            config=WalkForwardConfig(train_months=3, test_months=1, step_months=1),
        )
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)
        folds = validator._build_fold_schedule(start, end)
        for train_start, train_end, test_start, test_end in folds:
            delta = (test_start - train_end).days
            assert delta == 1  # Test starts day after train ends


# ---------------------------------------------------------------------------
# WalkForwardValidator — report compilation
# ---------------------------------------------------------------------------

class TestReportCompilation:
    def _make_fold_result(
        self,
        fold_index: int = 1,
        is_trades: int = 30,
        oos_trades: int = 8,
        is_sharpe: float = 1.5,
        oos_sharpe: float = 0.8,
        is_wr: float = 0.62,
        oos_wr: float = 0.55,
        is_r: float = 0.6,
        oos_r: float = 0.4,
    ):
        from backtesting.walk_forward import FoldResult
        return FoldResult(
            fold_index=fold_index,
            train_start="2024-01-01", train_end="2024-06-30",
            test_start="2024-07-01", test_end="2024-07-31",
            train_metrics=_metrics(trades=is_trades, win_rate=is_wr, avg_r=is_r, sharpe=is_sharpe),
            test_metrics=_metrics(trades=oos_trades, win_rate=oos_wr, avg_r=oos_r, sharpe=oos_sharpe),
            is_statistically_thin=(is_trades < 30 or oos_trades < 5),
            walk_forward_efficiency=round(oos_sharpe / is_sharpe, 4) if is_sharpe > 0 else 0.0,
        )

    def test_avg_wfe_computed(self):
        validator = WalkForwardValidator(backtester=MagicMock())
        folds = [self._make_fold_result(is_sharpe=1.5, oos_sharpe=0.9) for _ in range(3)]
        report = validator._compile_report(folds)
        expected_wfe = 0.9 / 1.5
        assert abs(report.avg_wfe - expected_wfe) < 0.05

    def test_edge_decay_detected_declining_oos(self):
        validator = WalkForwardValidator(backtester=MagicMock())
        # OOS avg_r declining: 0.5, 0.3, 0.1
        folds = [
            self._make_fold_result(fold_index=i+1, oos_r=0.5 - i * 0.2)
            for i in range(3)
        ]
        report = validator._compile_report(folds)
        assert report.edge_decay_detected

    def test_no_edge_decay_stable_oos(self):
        validator = WalkForwardValidator(backtester=MagicMock())
        folds = [self._make_fold_result(fold_index=i+1, oos_r=0.5) for i in range(3)]
        report = validator._compile_report(folds)
        assert not report.edge_decay_detected

    def test_thin_folds_excluded_from_aggregates(self):
        validator = WalkForwardValidator(
            backtester=MagicMock(),
            config=WalkForwardConfig(min_train_trades=30, min_test_trades=5),
        )
        thin_fold = self._make_fold_result(is_trades=5, oos_trades=2)  # Thin
        good_fold = self._make_fold_result(is_trades=50, oos_trades=10)
        report = validator._compile_report([thin_fold, good_fold])
        assert report.valid_folds == 1  # Only the good fold counts

    def test_good_wfe_note_in_report(self):
        validator = WalkForwardValidator(backtester=MagicMock())
        folds = [self._make_fold_result(is_sharpe=1.0, oos_sharpe=0.8) for _ in range(3)]
        report = validator._compile_report(folds)
        assert "edge" in report.notes.lower() or "wfe" in report.notes.lower()

    @pytest.mark.asyncio
    async def test_run_raises_on_no_folds(self):
        bt = PipelineBacktester(
            provider=_make_provider(),
            scanner_config=_make_scanner_config(),
            strategy_config=_make_strategy_config(),
            risk_config=_make_risk_config(),
        )
        validator = WalkForwardValidator(
            backtester=bt,
            config=WalkForwardConfig(train_months=6, test_months=1),
        )
        with pytest.raises(ValueError, match="too short"):
            await validator.run(
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 3, 1, tzinfo=timezone.utc),  # Only 2 months
                symbol_universe=["AAPL"],
            )


# ---------------------------------------------------------------------------
# DaySimulationResult
# ---------------------------------------------------------------------------

class TestDaySimulationResult:
    def test_fields_accessible(self):
        result = DaySimulationResult(
            date="2024-01-15",
            symbols_scanned=100,
            symbols_passed=8,
            watchlist_size=5,
            trades=[],
            equity_start=10_000.0,
            equity_end=10_200.0,
            scan_duration_ms=1500.0,
            session_duration_ms=45000.0,
        )
        assert result.date == "2024-01-15"
        assert result.equity_end == 10_200.0
        assert result.symbols_passed == 8
