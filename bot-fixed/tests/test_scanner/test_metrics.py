"""
tests/test_scanner/test_metrics.py
Unit tests for all scanner metric computation functions.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.enums import BarTimeframe
from core.models import Bar, Quote
from scanner.metrics import (
    compute_atr,
    compute_gap_pct,
    compute_intraday_vwap,
    compute_premarket_metrics_from_bars,
    compute_range_pct,
    compute_range_position,
    compute_relative_volume,
    compute_trend_quality,
    compute_vwap_slope,
)
from tests.conftest import make_bar, make_quote


# ---------------------------------------------------------------------------
# gap_pct
# ---------------------------------------------------------------------------

class TestGapPct:
    def test_positive_gap(self):
        assert compute_gap_pct(prev_close=100.0, premarket_last=106.0) == pytest.approx(6.0)

    def test_negative_gap(self):
        assert compute_gap_pct(100.0, 95.0) == pytest.approx(-5.0)

    def test_zero_prev_close(self):
        assert compute_gap_pct(0.0, 100.0) == 0.0

    def test_no_change(self):
        assert compute_gap_pct(100.0, 100.0) == 0.0


# ---------------------------------------------------------------------------
# relative_volume
# ---------------------------------------------------------------------------

class TestRelativeVolume:
    def test_normal_relative_volume(self):
        # avg_daily_vol=1_000_000, session_bars=390 → avg/bar=2564
        # pm bars = 30 bars → expected pm vol = 76_923
        # actual pm vol = 230_769 → rel vol ≈ 3x
        rv = compute_relative_volume(
            premarket_volume=230_769,
            avg_daily_volume=1_000_000,
            premarket_bars_count=30,
            session_bars=390,
        )
        assert rv == pytest.approx(3.0, rel=0.05)

    def test_zero_avg_volume(self):
        rv = compute_relative_volume(100_000, 0, 30)
        assert rv == 1.0

    def test_zero_bars(self):
        rv = compute_relative_volume(100_000, 1_000_000, 0)
        assert rv == 1.0


# ---------------------------------------------------------------------------
# range_pct
# ---------------------------------------------------------------------------

class TestRangePct:
    def test_basic_range(self):
        assert compute_range_pct(110.0, 100.0) == pytest.approx(10.0)

    def test_zero_low(self):
        assert compute_range_pct(110.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# range_position
# ---------------------------------------------------------------------------

class TestRangePosition:
    def test_at_high(self):
        assert compute_range_position(110.0, 100.0, 110.0) == pytest.approx(1.0)

    def test_at_low(self):
        assert compute_range_position(100.0, 100.0, 110.0) == pytest.approx(0.0)

    def test_midpoint(self):
        assert compute_range_position(105.0, 100.0, 110.0) == pytest.approx(0.5)

    def test_flat_range(self):
        assert compute_range_position(100.0, 100.0, 100.0) == 0.5

    def test_clamping(self):
        # Out-of-range current should be clamped
        result = compute_range_position(115.0, 100.0, 110.0)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# trend_quality
# ---------------------------------------------------------------------------

class TestTrendQuality:
    def test_perfect_uptrend(self):
        bars = [make_bar(close_=100.0 + i) for i in range(10)]
        assert compute_trend_quality(bars) == pytest.approx(1.0)

    def test_perfect_downtrend(self):
        bars = [make_bar(close_=100.0 - i) for i in range(10)]
        assert compute_trend_quality(bars) == 0.0

    def test_single_bar(self):
        bars = [make_bar()]
        assert compute_trend_quality(bars) == 0.5

    def test_mixed(self):
        # 3 up, 2 down out of 5 gaps
        closes = [100, 101, 102, 101, 102, 103]
        bars = [make_bar(close_=c) for c in closes]
        result = compute_trend_quality(bars)
        assert 0.6 <= result <= 0.9


# Helper to make bars with custom close
def make_bar(close_: float = 100.0) -> Bar:
    return Bar(
        symbol="TEST",
        timestamp=datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc),
        open=close_ - 0.05,
        high=close_ + 0.05,
        low=close_ - 0.10,
        close=close_,
        volume=10_000,
        timeframe=BarTimeframe.MINUTE_1,
    )


# ---------------------------------------------------------------------------
# vwap
# ---------------------------------------------------------------------------

class TestIntraDayVWAP:
    def test_single_bar(self):
        bar = Bar(
            symbol="T", timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            open=100, high=102, low=99, close=101, volume=1000,
            timeframe=BarTimeframe.MINUTE_1,
        )
        vwap = compute_intraday_vwap([bar])
        expected = (102 + 99 + 101) / 3.0
        assert vwap == pytest.approx(expected)

    def test_zero_volume(self):
        bar = Bar(
            symbol="T", timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            open=100, high=102, low=99, close=101, volume=0,
            timeframe=BarTimeframe.MINUTE_1,
        )
        vwap = compute_intraday_vwap([bar])
        assert vwap == 101.0  # Falls back to close


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

class TestATR:
    def test_constant_bars(self):
        bars = [
            Bar(
                symbol="T",
                timestamp=datetime(2024, 1, i + 1, tzinfo=timezone.utc),
                open=100, high=101, low=99, close=100, volume=1000,
                timeframe=BarTimeframe.DAY_1,
            )
            for i in range(15)
        ]
        atr = compute_atr(bars, period=14)
        # Every bar's TR ≈ 2 (high - low)
        assert 1.5 <= atr <= 2.5

    def test_insufficient_bars(self):
        assert compute_atr([], 14) == 0.0
