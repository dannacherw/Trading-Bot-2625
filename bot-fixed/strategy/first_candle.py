"""
strategy/first_candle.py
First 5-minute candle analysis (9:30–9:35 ET).

The opening candle is the most information-dense single bar of the day.
It reveals early institutional intent, sets the opening range, and gives
a volume baseline. A stock that opens strongly (close in top 30% of candle,
above-average volume, above VWAP) has demonstrated early buyer control.

This module computes a FirstCandleAnalysis from the first 5 one-minute bars,
and exposes an is_valid_setup() gate for the entry signal checker.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from loguru import logger

from core.models import Bar


@dataclass(frozen=True)
class FirstCandleAnalysis:
    """Immutable snapshot of the first 5-minute candle characteristics."""
    symbol: str
    computed_at: datetime

    # Price structure
    open_price: float
    high: float
    low: float
    close: float
    volume: int

    # Derived quality metrics
    close_position: float       # (close - low) / (high - low) — 0=at low, 1=at high
    body_pct: float             # |close - open| / open * 100
    upper_wick_pct: float       # (high - max(open,close)) / open * 100 — seller pressure
    is_bullish: bool            # close >= open

    # Volume context
    volume_vs_pm_avg: float     # candle vol / avg PM 1-min bar vol
    pm_bar_count: int           # how many PM bars were used for baseline

    # VWAP context
    vwap_at_close: float
    closed_above_vwap: bool

    @property
    def is_clean_open(self) -> bool:
        """Bullish first candle with close in top 30% of range and low upper wick."""
        return (
            self.is_bullish
            and self.close_position >= 0.70
            and self.upper_wick_pct <= 0.30
        )

    @property
    def has_strong_volume(self) -> bool:
        return self.volume_vs_pm_avg >= 2.0

    def __str__(self) -> str:
        return (
            f"FirstCandle({self.symbol}): "
            f"{'bull' if self.is_bullish else 'bear'} "
            f"close_pos={self.close_position:.2f} "
            f"vol_vs_pm={self.volume_vs_pm_avg:.1f}x "
            f"above_vwap={self.closed_above_vwap}"
        )


def compute_first_candle(
    symbol: str,
    minute_bars: list[Bar],
    pm_bars: list[Bar],
    market_open: datetime,
    candle_minutes: int = 5,
) -> FirstCandleAnalysis | None:
    """
    Aggregate the first `candle_minutes` bars after market open into a
    FirstCandleAnalysis. Returns None if insufficient bars.

    Args:
        minute_bars: Intraday 1-min bars (must include bars from market open)
        pm_bars: Premarket 1-min bars (for volume baseline)
        market_open: Market open time (e.g. 9:30 ET)
        candle_minutes: How many bars to aggregate (default: 5 = 9:30–9:35)
    """
    # Filter to bars within the first N minutes after open
    opening_bars = [
        b for b in minute_bars
        if 0 <= (b.timestamp - market_open).total_seconds() < candle_minutes * 60
    ]

    if len(opening_bars) < 1:
        logger.debug("{}: no opening bars found for first candle analysis", symbol)
        return None

    # Aggregate into a single synthetic candle
    open_price = opening_bars[0].open
    high = max(b.high for b in opening_bars)
    low = min(b.low for b in opening_bars)
    close = opening_bars[-1].close
    volume = sum(b.volume for b in opening_bars)

    # Compute VWAP over the opening period
    total_tp_vol = sum(
        ((b.high + b.low + b.close) / 3.0) * b.volume for b in opening_bars
    )
    total_vol = sum(b.volume for b in opening_bars)
    vwap_at_close = total_tp_vol / total_vol if total_vol > 0 else close

    # Derived metrics
    candle_range = high - low
    close_position = (close - low) / candle_range if candle_range > 0 else 0.5
    body_pct = abs(close - open_price) / open_price * 100.0 if open_price > 0 else 0.0
    upper_wick = high - max(open_price, close)
    upper_wick_pct = upper_wick / open_price * 100.0 if open_price > 0 else 0.0

    # Volume vs. average premarket bar volume
    pm_bar_count = len(pm_bars)
    avg_pm_bar_vol = (
        sum(b.volume for b in pm_bars) / pm_bar_count if pm_bar_count > 0 else 1
    )
    volume_vs_pm_avg = volume / avg_pm_bar_vol if avg_pm_bar_vol > 0 else 1.0

    analysis = FirstCandleAnalysis(
        symbol=symbol,
        computed_at=datetime.now(tz=timezone.utc),
        open_price=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        close_position=round(close_position, 3),
        body_pct=round(body_pct, 3),
        upper_wick_pct=round(upper_wick_pct, 3),
        is_bullish=close >= open_price,
        volume_vs_pm_avg=round(volume_vs_pm_avg, 2),
        pm_bar_count=pm_bar_count,
        vwap_at_close=round(vwap_at_close, 4),
        closed_above_vwap=close >= vwap_at_close,
    )

    logger.debug("{}", str(analysis))
    return analysis


def is_valid_setup(
    analysis: FirstCandleAnalysis | None,
    min_close_position: float = 0.70,
    min_volume_vs_pm: float = 2.0,
) -> tuple[bool, str]:
    """
    Gate check: does the first candle confirm a high-probability setup?

    Args:
        analysis: Computed FirstCandleAnalysis, or None if unavailable
        min_close_position: Minimum close position in range [0, 1]
        min_volume_vs_pm: Minimum volume vs. avg PM bar volume

    Returns:
        (True, "") if setup is valid, (False, reason) if not.
        Returns (True, "") when analysis is None (graceful degradation).
    """
    if analysis is None:
        # Cannot compute — allow through with no gate
        return True, "first candle data unavailable"

    if not analysis.is_bullish:
        return False, (
            f"first candle bearish (open={analysis.open_price:.2f} close={analysis.close:.2f})"
        )

    if analysis.close_position < min_close_position:
        return False, (
            f"first candle weak close position {analysis.close_position:.2f} "
            f"< min {min_close_position:.2f} (sellers in control)"
        )

    if analysis.volume_vs_pm_avg < min_volume_vs_pm:
        return False, (
            f"first candle volume {analysis.volume_vs_pm_avg:.1f}x PM avg "
            f"< min {min_volume_vs_pm:.1f}x (low institutional participation)"
        )

    if not analysis.closed_above_vwap:
        return False, (
            f"first candle closed below VWAP ({analysis.close:.2f} < {analysis.vwap_at_close:.2f})"
        )

    return True, ""
