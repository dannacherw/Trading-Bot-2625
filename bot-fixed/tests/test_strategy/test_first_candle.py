"""
tests/test_strategy/test_first_candle.py
Tests for first candle analysis computation and validity gate.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.enums import BarTimeframe
from core.models import Bar
from strategy.first_candle import FirstCandleAnalysis, compute_first_candle, is_valid_setup


def _bar(
    open_: float, high: float, low: float, close: float,
    volume: int = 10_000,
    ts: datetime | None = None,
) -> Bar:
    if ts is None:
        ts = datetime(2024, 1, 15, 13, 30, tzinfo=timezone.utc)
    return Bar(
        symbol="AAPL", timestamp=ts,
        open=open_, high=high, low=low, close=close,
        volume=volume, timeframe=BarTimeframe.MINUTE_1,
    )


@pytest.fixture
def market_open() -> datetime:
    return datetime(2024, 1, 15, 13, 30, 0, tzinfo=timezone.utc)


def _make_opening_bars(market_open: datetime, count: int = 5) -> list[Bar]:
    """5 bullish bars opening at 9:30 ET."""
    bars = []
    for i in range(count):
        ts = market_open.replace(minute=market_open.minute + i)
        p = 100.0 + i * 0.10
        bars.append(_bar(p, p + 0.15, p - 0.02, p + 0.12, volume=20_000, ts=ts))
    return bars


def _make_pm_bars(count: int = 30, avg_vol: int = 8_000) -> list[Bar]:
    return [
        _bar(99.0, 99.5, 98.8, 99.2, volume=avg_vol,
             ts=datetime(2024, 1, 15, 9 + i // 60, i % 60, tzinfo=timezone.utc))
        for i in range(count)
    ]


class TestComputeFirstCandle:
    def test_aggregates_5_bars(self, market_open):
        bars = _make_opening_bars(market_open)
        pm = _make_pm_bars()
        result = compute_first_candle("AAPL", bars, pm, market_open)
        assert result is not None
        assert result.symbol == "AAPL"

    def test_open_equals_first_bar_open(self, market_open):
        bars = _make_opening_bars(market_open)
        result = compute_first_candle("AAPL", bars, _make_pm_bars(), market_open)
        assert result is not None
        assert result.open_price == bars[0].open

    def test_high_is_max_across_bars(self, market_open):
        bars = _make_opening_bars(market_open)
        result = compute_first_candle("AAPL", bars, _make_pm_bars(), market_open)
        assert result is not None
        expected_high = max(b.high for b in bars)
        assert result.high == pytest.approx(expected_high)

    def test_close_position_at_top(self, market_open):
        """Bars closing near their highs → close_position close to 1.0."""
        bars = _make_opening_bars(market_open)
        result = compute_first_candle("AAPL", bars, _make_pm_bars(), market_open)
        assert result is not None
        assert result.close_position >= 0.60

    def test_volume_vs_pm_avg_computed(self, market_open):
        bars = _make_opening_bars(market_open, count=5)
        pm_bars = _make_pm_bars(count=30, avg_vol=5_000)
        result = compute_first_candle("AAPL", bars, pm_bars, market_open)
        assert result is not None
        assert result.volume_vs_pm_avg == pytest.approx(20_000 / 5_000 * 5, rel=0.5)

    def test_returns_none_when_no_opening_bars(self, market_open):
        # Bars all before market open
        pre_bars = [
            _bar(99.0, 99.5, 98.5, 99.2, ts=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc))
        ]
        result = compute_first_candle("AAPL", pre_bars, [], market_open)
        assert result is None

    def test_bearish_candle_detected(self, market_open):
        ts = market_open
        bars = [_bar(101.0, 101.5, 99.5, 99.8, ts=ts)]  # Open > Close = bearish
        result = compute_first_candle("AAPL", bars, [], market_open)
        assert result is not None
        assert result.is_bullish is False


class TestIsValidSetup:
    def _make_strong_analysis(self) -> FirstCandleAnalysis:
        return FirstCandleAnalysis(
            symbol="AAPL",
            computed_at=datetime(2024, 1, 15, 13, 35, tzinfo=timezone.utc),
            open_price=100.0, high=100.8, low=99.8, close=100.6,
            volume=50_000,
            close_position=0.80,
            body_pct=0.60,
            upper_wick_pct=0.20,
            is_bullish=True,
            volume_vs_pm_avg=3.5,
            pm_bar_count=30,
            vwap_at_close=100.2,
            closed_above_vwap=True,
        )

    def test_valid_setup_passes(self):
        analysis = self._make_strong_analysis()
        ok, _ = is_valid_setup(analysis)
        assert ok is True

    def test_none_analysis_passes(self):
        """No first candle data → allow through."""
        ok, reason = is_valid_setup(None)
        assert ok is True
        assert "unavailable" in reason

    def test_bearish_fails(self):
        import dataclasses
        analysis = dataclasses.replace(
            self._make_strong_analysis(), is_bullish=False, close=99.5
        )
        ok, reason = is_valid_setup(analysis)
        assert ok is False
        assert "bearish" in reason.lower()

    def test_low_close_position_fails(self):
        import dataclasses
        analysis = dataclasses.replace(self._make_strong_analysis(), close_position=0.30)
        ok, reason = is_valid_setup(analysis, min_close_position=0.70)
        assert ok is False
        assert "close position" in reason.lower()

    def test_low_volume_fails(self):
        import dataclasses
        analysis = dataclasses.replace(self._make_strong_analysis(), volume_vs_pm_avg=1.2)
        ok, reason = is_valid_setup(analysis, min_volume_vs_pm=2.0)
        assert ok is False
        assert "volume" in reason.lower()

    def test_below_vwap_fails(self):
        import dataclasses
        analysis = dataclasses.replace(
            self._make_strong_analysis(),
            closed_above_vwap=False,
            close=99.8,
        )
        ok, reason = is_valid_setup(analysis)
        assert ok is False
        assert "vwap" in reason.lower()
