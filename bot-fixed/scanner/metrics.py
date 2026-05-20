"""
scanner/metrics.py
Pure functions for computing premarket metrics from raw bar/quote data.
All functions are synchronous and stateless for easy testing.
"""
from __future__ import annotations

import statistics
from datetime import datetime

import numpy as np

from core.models import Bar, PremarketMetrics, Quote


def compute_gap_pct(prev_close: float, premarket_last: float) -> float:
    """Gap percentage from previous close to current premarket price."""
    if prev_close <= 0:
        return 0.0
    return (premarket_last - prev_close) / prev_close * 100.0


def compute_relative_volume(
    premarket_volume: int,
    avg_daily_volume: float,
    premarket_bars_count: int,
    session_bars: int = 390,  # Regular session bars (1-min)
) -> float:
    """
    Relative volume: actual PM vol vs expected PM vol given time elapsed.
    avg_daily_volume is in shares.
    """
    if avg_daily_volume <= 0 or premarket_bars_count <= 0:
        return 1.0
    expected_per_bar = avg_daily_volume / session_bars
    expected_pm_vol = expected_per_bar * premarket_bars_count
    return premarket_volume / expected_pm_vol if expected_pm_vol > 0 else 1.0


def compute_range_pct(high: float, low: float) -> float:
    """Percentage range = (high - low) / low * 100."""
    if low <= 0:
        return 0.0
    return (high - low) / low * 100.0


def compute_range_position(current: float, low: float, high: float) -> float:
    """
    Position of current price within the premarket range [0, 1].
    0 = at the low, 1 = at the high.
    """
    span = high - low
    if span <= 0:
        return 0.5
    return max(0.0, min(1.0, (current - low) / span))


def compute_trend_quality(bars: list[Bar]) -> float:
    """
    Measure orderliness of the premarket move — volume-weighted.

    Returns a score in [0, 1]:
      1.0 = perfectly orderly (all bars close higher, large bars drive the move)
      0.0 = perfectly disorderly (large bars are all down)

    Bug fixed from v1: pure up/down bar count ignores bar size.
    A stock with 10 tiny up bars and 1 massive down bar scored 0.90 — wrong.
    This version weights each bar by its volume, so large bars count more.
    Also penalises excessive upper wicks (selling pressure at highs).
    """
    if len(bars) < 2:
        return 0.5

    # Use only bars[1:] for weighting — bars[0] is the anchor for direction comparison
    # but contributes no up/down signal itself (no previous bar to compare against).
    # Including it in total_vol would deflate all weights slightly.
    comparison_bars = bars[1:]   # The bars that actually have a direction
    closes = [b.close for b in bars]
    total_vol = sum(b.volume for b in comparison_bars) or 1

    weighted_up = 0.0
    total_weight = 0.0

    for i in range(1, len(bars)):
        bar = bars[i]
        weight = bar.volume / total_vol
        total_weight += weight

        if closes[i] >= closes[i - 1]:
            # Up bar — discount for upper wick (selling pressure at highs)
            bar_range = bar.high - bar.low
            if bar_range > 0:
                upper_wick = bar.high - max(bar.open, bar.close)
                wick_penalty = min(0.3, upper_wick / bar_range * 0.5)
                weighted_up += weight * (1.0 - wick_penalty)
            else:
                weighted_up += weight
        # Down bars contribute 0 weighted_up

    if total_weight <= 0:
        return 0.5
    return max(0.0, min(1.0, weighted_up / total_weight))


def compute_vwap_slope(bars: list[Bar], lookback: int = 3) -> float:
    """
    Estimate VWAP slope over the last `lookback` bars.
    Returns slope in price-per-bar units.
    """
    vwap_bars = [b for b in bars[-lookback:] if b.vwap is not None]
    if len(vwap_bars) < 2:
        return 0.0
    vwaps = [b.vwap for b in vwap_bars]  # type: ignore[misc]
    x = np.arange(len(vwaps), dtype=float)
    slope = float(np.polyfit(x, vwaps, 1)[0])
    return slope


def compute_intraday_vwap(bars: list[Bar]) -> float:
    """
    Compute cumulative VWAP from bar data.
    VWAP = sum(typical_price * volume) / sum(volume)
    """
    total_tp_vol = sum(
        ((b.high + b.low + b.close) / 3.0) * b.volume for b in bars
    )
    total_vol = sum(b.volume for b in bars)
    if total_vol == 0:
        return bars[-1].close if bars else 0.0
    return total_tp_vol / total_vol


def compute_atr(bars: list[Bar], period: int = 14) -> float:
    """Compute ATR from a list of daily or intraday bars."""
    if len(bars) < 2:
        return 0.0
    trs = []
    for i in range(1, len(bars)):
        prev_close = bars[i - 1].close
        tr = max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - prev_close),
            abs(bars[i].low - prev_close),
        )
        trs.append(tr)
    return float(np.mean(trs[-period:])) if trs else 0.0


def compute_float_rotation_pct(
    premarket_volume: int,
    float_shares: int | None,
) -> float | None:
    """
    Premarket float rotation = (PM volume / float shares) * 100.
    Returns None if float data is unavailable.
    A stock turning over ≥2% of its float in premarket shows genuine institutional interest.
    """
    if float_shares is None or float_shares <= 0:
        return None
    return (premarket_volume / float_shares) * 100.0


def compute_premarket_metrics_from_bars(
    symbol: str,
    pm_bars: list[Bar],
    prev_close: float,
    quote: Quote,
    avg_daily_dollar_volume: float,
    avg_daily_volume: float,
    float_shares: int | None = None,
) -> PremarketMetrics:
    """
    Compute all premarket metrics from bar/quote data.
    This is the canonical computation path — used by scanner + tests.
    """
    if not pm_bars:
        raise ValueError(f"No premarket bars provided for {symbol}")

    pm_open = pm_bars[0].open
    pm_high = max(b.high for b in pm_bars)
    pm_low = min(b.low for b in pm_bars)
    pm_last = pm_bars[-1].close
    pm_vol = sum(b.volume for b in pm_bars)
    pm_dv = sum(((b.high + b.low + b.close) / 3.0) * b.volume for b in pm_bars)

    gap_pct = compute_gap_pct(prev_close, pm_last)
    rel_vol = compute_relative_volume(pm_vol, avg_daily_volume, len(pm_bars))
    range_pct = compute_range_pct(pm_high, pm_low)
    range_pos = compute_range_position(pm_last, pm_low, pm_high)
    trend_q = compute_trend_quality(pm_bars)
    float_rotation = compute_float_rotation_pct(pm_vol, float_shares)

    return PremarketMetrics(
        symbol=symbol,
        computed_at=datetime.now(tz=timezone.utc),
        prev_close=prev_close,
        premarket_open=pm_open,
        premarket_high=pm_high,
        premarket_low=pm_low,
        premarket_last=pm_last,
        premarket_volume=pm_vol,
        premarket_dollar_volume=pm_dv,
        gap_pct=gap_pct,
        relative_volume=rel_vol,
        spread_pct=quote.spread_pct,
        range_pct=range_pct,
        range_position=range_pos,
        trend_quality=trend_q,
        avg_daily_dollar_volume=avg_daily_dollar_volume,
        float_shares=float_shares,
        pm_float_rotation_pct=float_rotation,
    )
