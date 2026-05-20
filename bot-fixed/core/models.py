"""
core/models.py
Shared immutable data models (Pydantic v2) used across all system layers.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field

from core.enums import (
    ArchetypeTag,
    BarTimeframe,
    CatalystCategory,
    ExitReason,
    NewsTimingCategory,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    PositionStatus,
    SignalStrength,
    SignalType,
    TimeInForce,
    RiskCheckResult,
    RiskRejectionReason,
)


# ---------------------------------------------------------------------------
# Market Data Models
# ---------------------------------------------------------------------------

class Bar(BaseModel):
    """OHLCV bar for any timeframe."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    timeframe: BarTimeframe = BarTimeframe.MINUTE_1
    is_confirmed: bool = True  # False for the current (live) bar

    @computed_field  # type: ignore[misc]
    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3.0

    @computed_field  # type: ignore[misc]
    @property
    def range(self) -> float:
        return self.high - self.low

    @computed_field  # type: ignore[misc]
    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @computed_field  # type: ignore[misc]
    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open


class Quote(BaseModel):
    """Real-time bid/ask snapshot."""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    bid_size: int = 0
    ask_size: int = 0

    @computed_field  # type: ignore[misc]
    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @computed_field  # type: ignore[misc]
    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @computed_field  # type: ignore[misc]
    @property
    def spread_pct(self) -> float:
        return (self.spread / self.mid * 100.0) if self.mid > 0 else 0.0


class PremarketMetrics(BaseModel):
    """Computed premarket metrics for a single stock."""
    symbol: str
    computed_at: datetime
    prev_close: float
    premarket_open: float
    premarket_high: float
    premarket_low: float
    premarket_last: float
    premarket_volume: int
    premarket_dollar_volume: float
    gap_pct: float                    # (pm_last - prev_close) / prev_close * 100
    relative_volume: float            # pm_vol / avg_pm_vol
    spread_pct: float
    range_pct: float                  # (pm_high - pm_low) / pm_low * 100
    range_position: float             # (pm_last - pm_low) / (pm_high - pm_low)
    trend_quality: float              # 0–1 score of orderliness
    avg_daily_dollar_volume: float
    # Float-related fields — Optional: None when API data unavailable
    float_shares: Optional[int] = None
    pm_float_rotation_pct: Optional[float] = None  # pm_vol / float_shares * 100


# ---------------------------------------------------------------------------
# Catalyst Models
# ---------------------------------------------------------------------------

class Catalyst(BaseModel):
    """Detected catalyst for a stock move."""
    symbol: str
    detected_at: datetime
    category: CatalystCategory
    confidence: float = Field(ge=0.0, le=1.0)
    strength: float = Field(ge=0.0, le=1.0)
    summary: str = ""
    source: str = ""
    recency_hours: float = 0.0
    news_timing: NewsTimingCategory = NewsTimingCategory.UNKNOWN


# ---------------------------------------------------------------------------
# Scanner / Watchlist Models
# ---------------------------------------------------------------------------

class ScanResult(BaseModel):
    """Full scanner output for a single symbol."""
    symbol: str
    scanned_at: datetime
    metrics: PremarketMetrics
    catalyst: Optional[Catalyst] = None
    composite_score: float = Field(ge=0.0, le=1.0, default=0.0)
    archetypes: list[ArchetypeTag] = Field(default_factory=list)
    passes_filters: bool = False
    filter_failure_reason: Optional[str] = None


class WatchlistItem(BaseModel):
    """Entry in the ranked watchlist handed to the strategy engine."""
    symbol: str
    rank: int
    scan_result: ScanResult
    is_focus: bool = False            # True = top priority focus list
    notes: str = ""


# ---------------------------------------------------------------------------
# Order Models
# ---------------------------------------------------------------------------

class Order(BaseModel):
    """Represents an order submitted to the broker."""
    order_id: UUID = Field(default_factory=uuid4)
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    status: OrderStatus = OrderStatus.PENDING
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    filled_quantity: int = 0
    avg_fill_price: Optional[float] = None
    broker_order_id: Optional[str] = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Position / Trade Models
# ---------------------------------------------------------------------------

class Position(BaseModel):
    """Open or closed position."""
    position_id: UUID = Field(default_factory=uuid4)
    symbol: str
    side: PositionSide
    status: PositionStatus = PositionStatus.OPEN
    entry_price: float
    entry_time: datetime
    quantity: int
    remaining_quantity: int
    stop_price: float
    target_1_price: float
    target_2_price: float
    breakeven_price: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[ExitReason] = None
    realized_pnl: float = 0.0
    commission: float = 0.0
    signal_id: Optional[UUID] = None

    @computed_field  # type: ignore[misc]
    @property
    def dollar_risk(self) -> float:
        return abs(self.entry_price - self.stop_price) * self.quantity

    @computed_field  # type: ignore[misc]
    @property
    def open_value(self) -> float:
        return self.entry_price * self.remaining_quantity

    def unrealized_pnl(self, current_price: float) -> float:
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) * self.remaining_quantity
        return (self.entry_price - current_price) * self.remaining_quantity

    def r_multiple(self, current_price: float) -> float:
        risk_per_share = abs(self.entry_price - self.stop_price)
        if risk_per_share == 0:
            return 0.0
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) / risk_per_share
        return (self.entry_price - current_price) / risk_per_share


class Trade(BaseModel):
    """Completed (fully closed) trade record."""
    trade_id: UUID = Field(default_factory=uuid4)
    position_id: UUID
    symbol: str
    side: PositionSide
    entry_price: float
    exit_price: float
    quantity: int
    entry_time: datetime
    exit_time: datetime
    exit_reason: ExitReason
    gross_pnl: float
    commission: float
    net_pnl: float
    r_multiple: float
    hold_duration_seconds: float

    @computed_field  # type: ignore[misc]
    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0


# ---------------------------------------------------------------------------
# Signal Models
# ---------------------------------------------------------------------------

class Signal(BaseModel):
    """Trading signal generated by the strategy engine."""
    signal_id: UUID = Field(default_factory=uuid4)
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    generated_at: datetime
    entry_price: float
    stop_price: float
    target_1_price: float
    target_2_price: float
    suggested_quantity: int = 0
    vwap_at_signal: Optional[float] = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Risk Models
# ---------------------------------------------------------------------------

class RiskValidation(BaseModel):
    """Result of a risk check for a proposed trade."""
    result: RiskCheckResult
    rejection_reason: Optional[RiskRejectionReason] = None
    original_quantity: int
    approved_quantity: int
    message: str = ""


class DailyRiskState(BaseModel):
    """Snapshot of daily risk consumption."""
    date: str                         # YYYY-MM-DD
    starting_equity: float
    current_equity: float
    realized_pnl_today: float
    unrealized_pnl: float
    trades_today: int
    open_positions: int
    daily_loss_limit: float
    daily_loss_used: float
    is_halted: bool = False

    @computed_field  # type: ignore[misc]
    @property
    def loss_pct_used(self) -> float:
        if self.daily_loss_limit == 0:
            return 0.0
        return self.daily_loss_used / self.daily_loss_limit * 100.0

    @computed_field  # type: ignore[misc]
    @property
    def remaining_loss_budget(self) -> float:
        return max(0.0, self.daily_loss_limit - self.daily_loss_used)


# ---------------------------------------------------------------------------
# Performance / Analytics Models
# ---------------------------------------------------------------------------

class PerformanceMetrics(BaseModel):
    """Computed backtest or live performance statistics."""
    start_date: str
    end_date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    avg_r_multiple: float
    avg_winner_r: float
    avg_loser_r: float
    avg_hold_minutes: float
    total_commission: float
    net_pnl: float

    @computed_field  # type: ignore[misc]
    @property
    def loss_rate(self) -> float:
        return 1.0 - self.win_rate
