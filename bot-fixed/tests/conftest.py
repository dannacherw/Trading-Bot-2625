"""
tests/conftest.py
Shared pytest fixtures — mocks, sample data, and async helpers.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from core.config import (
    ArchetypeThresholds, DailyLimitSettings, EntrySettings, ExitSettings,
    LiquiditySettings, PositionLimitSettings, PremarketFilterSettings,
    RiskConfig, ScoringWeights, SchwabSettings, BrokerSettings,
    StrategyConfig, VolatilitySettings, AccountSettings, StopLossSettings,
)
from core.enums import BarTimeframe, PositionSide
from core.models import Bar, Position, PremarketMetrics, Quote, Signal
from core.enums import SignalStrength, SignalType


# ---------------------------------------------------------------------------
# Shared datetime helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def market_open() -> datetime:
    return datetime(2024, 1, 15, 13, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def scan_date() -> datetime:
    return datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Sample market data
# ---------------------------------------------------------------------------

def make_bar(
    symbol: str = "AAPL",
    ts: datetime | None = None,
    open_: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.5,
    volume: int = 10_000,
    vwap: float | None = 100.3,
) -> Bar:
    if ts is None:
        ts = datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
    return Bar(
        symbol=symbol, timestamp=ts,
        open=open_, high=high, low=low, close=close,
        volume=volume, vwap=vwap, timeframe=BarTimeframe.MINUTE_1,
    )


@pytest.fixture
def sample_bar() -> Bar:
    return make_bar()


@pytest.fixture
def sample_bars(market_open: datetime) -> list[Bar]:
    """A sequence of 20 orderly bullish 1-min bars."""
    bars = []
    base_price = 100.0
    for i in range(20):
        ts = market_open.replace(minute=i)
        p = base_price + i * 0.05
        bars.append(Bar(
            symbol="AAPL", timestamp=ts,
            open=p, high=p + 0.10, low=p - 0.03, close=p + 0.07,
            volume=12_000 + i * 200,
            vwap=p + 0.03,
            timeframe=BarTimeframe.MINUTE_1,
        ))
    return bars


def make_quote(
    symbol: str = "AAPL",
    bid: float = 100.45,
    ask: float = 100.55,
) -> Quote:
    return Quote(
        symbol=symbol,
        timestamp=datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc),
        bid=bid, ask=ask, bid_size=100, ask_size=100,
    )


@pytest.fixture
def sample_quote() -> Quote:
    return make_quote()


@pytest.fixture
def sample_metrics() -> PremarketMetrics:
    return PremarketMetrics(
        symbol="AAPL",
        computed_at=datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc),
        prev_close=95.0,
        premarket_open=97.0,
        premarket_high=101.5,
        premarket_low=96.5,
        premarket_last=101.0,
        premarket_volume=500_000,
        premarket_dollar_volume=5_000_000.0,
        gap_pct=6.32,
        relative_volume=3.5,
        spread_pct=0.08,
        range_pct=5.18,
        range_position=0.90,
        trend_quality=0.75,
        avg_daily_dollar_volume=30_000_000,
    )


@pytest.fixture
def sample_signal() -> Signal:
    return Signal(
        symbol="AAPL",
        signal_type=SignalType.VWAP_PULLBACK_LONG,
        strength=SignalStrength.STRONG,
        generated_at=datetime(2024, 1, 15, 14, 15, 0, tzinfo=timezone.utc),
        entry_price=101.00,
        stop_price=100.20,
        target_1_price=102.20,
        target_2_price=103.00,
        vwap_at_signal=100.85,
        suggested_quantity=50,
    )


@pytest.fixture
def sample_position(sample_signal: Signal) -> Position:
    return Position(
        symbol="AAPL",
        side=PositionSide.LONG,
        entry_price=101.00,
        entry_time=datetime(2024, 1, 15, 14, 15, 0, tzinfo=timezone.utc),
        quantity=50,
        remaining_quantity=50,
        stop_price=100.20,
        target_1_price=102.20,
        target_2_price=103.00,
        signal_id=sample_signal.signal_id,
    )


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def premarket_filter_settings() -> PremarketFilterSettings:
    return PremarketFilterSettings()


@pytest.fixture
def scoring_weights() -> ScoringWeights:
    return ScoringWeights()


@pytest.fixture
def archetype_thresholds() -> ArchetypeThresholds:
    return ArchetypeThresholds()


@pytest.fixture
def exit_settings() -> ExitSettings:
    return ExitSettings()


@pytest.fixture
def entry_settings() -> EntrySettings:
    return EntrySettings()


@pytest.fixture
def risk_config() -> RiskConfig:
    return RiskConfig(
        account=AccountSettings(starting_equity=10_000.0),
        daily_limits=DailyLimitSettings(max_daily_loss_pct=2.0, max_trades_per_day=6),
        position_limits=PositionLimitSettings(
            max_open_positions=3,
            default_risk_per_trade_pct=0.35,
            min_trade_dollar_value=500.0,
        ),
        stop_loss=StopLossSettings(),
        volatility=VolatilitySettings(),
        liquidity=LiquiditySettings(),
    )


@pytest.fixture
def strategy_config() -> StrategyConfig:
    return StrategyConfig(entry=EntrySettings(), exit=ExitSettings())


@pytest.fixture
def mock_broker() -> MagicMock:
    broker = AsyncMock()
    broker.name = "mock"
    broker.is_paper = True
    broker.supports_fractional_shares = False
    broker.get_account_equity = AsyncMock(return_value=10_000.0)
    broker.get_buying_power = AsyncMock(return_value=10_000.0)
    broker.submit_order = AsyncMock(return_value="BROKER-ORDER-123")
    broker.cancel_order = AsyncMock(return_value=True)
    broker.get_order_status = AsyncMock(return_value={"status": "FILLED", "filledQuantity": 50})
    broker.get_open_orders = AsyncMock(return_value=[])
    broker.get_positions = AsyncMock(return_value=[])
    broker.is_paper = True
    return broker


@pytest.fixture
def mock_provider() -> MagicMock:
    provider = AsyncMock()
    provider.name = "mock"
    provider.supports_streaming = False
    provider.supports_news = False
    return provider
