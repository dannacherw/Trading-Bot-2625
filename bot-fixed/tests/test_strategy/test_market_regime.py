"""
tests/test_strategy/test_market_regime.py
Tests for the market regime filter and relative strength computation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from core.config import MarketRegimeSettings
from core.enums import BarTimeframe
from core.models import Bar
from strategy.market_regime import MarketRegimeFilter


def _spy_bars(move_pct: float = 0.0, count: int = 10) -> list[Bar]:
    """Build synthetic SPY bars with a given open-to-current move."""
    base = 450.0
    open_price = base
    final = base * (1 + move_pct / 100.0)
    bars = []
    for i in range(count):
        price = open_price + (final - open_price) * (i / max(count - 1, 1))
        bars.append(Bar(
            symbol="SPY",
            timestamp=datetime(2024, 1, 15, 13, 30 + i, tzinfo=timezone.utc),
            open=open_price if i == 0 else price - 0.05,
            high=price + 0.10,
            low=price - 0.05,
            close=price,
            volume=1_000_000,
            vwap=price,
            timeframe=BarTimeframe.MINUTE_1,
        ))
    return bars


@pytest.fixture
def settings():
    return MarketRegimeSettings(
        enabled=True,
        default_benchmark="SPY",
        max_benchmark_down_pct=-0.40,
        min_benchmark_vwap_slope=-0.005,
        vwap_slope_lookback_bars=5,
    )


@pytest.fixture
def mock_provider():
    return AsyncMock()


class TestMarketRegimeFilter:
    async def test_favorable_when_spy_flat(self, settings, mock_provider):
        mock_provider.get_intraday_bars = AsyncMock(return_value=_spy_bars(0.0))
        regime = MarketRegimeFilter(mock_provider, settings)
        ok, reason = await regime.is_favorable()
        assert ok is True

    async def test_favorable_when_spy_up(self, settings, mock_provider):
        mock_provider.get_intraday_bars = AsyncMock(return_value=_spy_bars(0.5))
        regime = MarketRegimeFilter(mock_provider, settings)
        ok, _ = await regime.is_favorable()
        assert ok is True

    async def test_unfavorable_when_spy_down_beyond_threshold(self, settings, mock_provider):
        mock_provider.get_intraday_bars = AsyncMock(return_value=_spy_bars(-0.8))
        regime = MarketRegimeFilter(mock_provider, settings)
        ok, reason = await regime.is_favorable()
        assert ok is False
        assert "SPY" in reason
        assert "down" in reason.lower()

    async def test_favorable_when_disabled(self, mock_provider):
        settings = MarketRegimeSettings(enabled=False)
        regime = MarketRegimeFilter(mock_provider, settings)
        ok, _ = await regime.is_favorable()
        assert ok is True  # Disabled = always favorable

    async def test_favorable_when_no_bars(self, settings, mock_provider):
        mock_provider.get_intraday_bars = AsyncMock(return_value=[])
        regime = MarketRegimeFilter(mock_provider, settings)
        ok, _ = await regime.is_favorable()
        # No data = allow through with warning
        assert ok is True

    async def test_favorable_on_spy_down_within_threshold(self, settings, mock_provider):
        mock_provider.get_intraday_bars = AsyncMock(return_value=_spy_bars(-0.30))
        regime = MarketRegimeFilter(mock_provider, settings)
        ok, _ = await regime.is_favorable()
        assert ok is True  # -0.30% < -0.40% threshold = still allowed


class TestBenchmarkResolution:
    def test_biotech_maps_to_xbi(self, settings, mock_provider):
        regime = MarketRegimeFilter(mock_provider, settings)
        assert regime._resolve_benchmark("Biotechnology") == "XBI"
        assert regime._resolve_benchmark("Biotech") == "XBI"

    def test_technology_maps_to_qqq(self, settings, mock_provider):
        regime = MarketRegimeFilter(mock_provider, settings)
        assert regime._resolve_benchmark("Technology") == "QQQ"

    def test_unknown_sector_maps_to_spy(self, settings, mock_provider):
        regime = MarketRegimeFilter(mock_provider, settings)
        assert regime._resolve_benchmark("Widgets") == "SPY"
        assert regime._resolve_benchmark("") == "SPY"

    def test_financials_maps_correctly(self, settings, mock_provider):
        regime = MarketRegimeFilter(mock_provider, settings)
        assert regime._resolve_benchmark("Financials") == "XLF"


class TestRelativeStrength:
    async def test_positive_rs_when_outperforming(self, settings, mock_provider):
        spy_bars = _spy_bars(0.5)   # SPY up 0.5%
        symbol_bars = _spy_bars(3.0)  # Stock up 3.0%
        for b in symbol_bars:
            b = b.model_copy(update={"symbol": "AAPL"})
        mock_provider.get_intraday_bars = AsyncMock(return_value=spy_bars)
        regime = MarketRegimeFilter(mock_provider, settings)
        rs = await regime.get_relative_strength("AAPL", symbol_bars)
        assert rs > 2.0  # 3.0 - 0.5 = 2.5% outperformance

    async def test_negative_rs_when_underperforming(self, settings, mock_provider):
        spy_bars = _spy_bars(1.0)
        symbol_bars = _spy_bars(0.2)
        mock_provider.get_intraday_bars = AsyncMock(return_value=spy_bars)
        regime = MarketRegimeFilter(mock_provider, settings)
        rs = await regime.get_relative_strength("AAPL", symbol_bars)
        assert rs < 0

    async def test_zero_rs_when_no_symbol_bars(self, settings, mock_provider):
        mock_provider.get_intraday_bars = AsyncMock(return_value=_spy_bars(0.5))
        regime = MarketRegimeFilter(mock_provider, settings)
        rs = await regime.get_relative_strength("AAPL", [])
        assert rs == 0.0
