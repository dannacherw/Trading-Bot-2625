"""
execution/slippage_model.py — v2
Realistic slippage estimation with adverse selection and market impact.

v1 used 5bps flat. v2 models:
  - Adverse selection: limit orders fill when market is moving against you
    (approximately 60% of the spread as additional cost)
  - Market impact: larger orders relative to bar volume cost more
  - Base slippage: irreducible execution friction (routing, latency)
  - High-volatility multiplier: wider uncertainty in fast markets
"""
from __future__ import annotations

from core.config import SlippageSettings
from core.enums import OrderSide


def estimate_slippage(
    price: float,
    spread_pct: float,
    side: OrderSide,
    settings: SlippageSettings,
    is_high_vol: bool = False,
    order_size_shares: int = 0,
    avg_bar_volume: float = 0.0,
) -> float:
    """
    Estimate total slippage cost in price units (always positive).

    Components:
    1. Base slippage: fixed 5bps irreducible friction
    2. Adverse selection: 60% of spread — limit orders fill when price moves away
    3. Market impact: proportional to order size vs. average bar volume
    4. High-vol multiplier applied to base + adverse selection

    Args:
        price: Reference price (mid or last)
        spread_pct: Current bid-ask spread as percentage of mid
        side: BUY or SELL
        settings: SlippageSettings from config
        is_high_vol: True during first 15min, news events, or high ATR
        order_size_shares: Number of shares in the order
        avg_bar_volume: Average 1-min bar volume for the stock

    Returns:
        Slippage in price units (add to buy price, subtract from sell price)
    """
    # 1. Base slippage
    base = price * (settings.base_slippage_pct / 100.0)

    # 2. Adverse selection: 60% of spread
    spread_dollars = price * (spread_pct / 100.0)
    adverse_selection = spread_dollars * settings.spread_slippage_factor

    # 3. Market impact (only meaningful when order is large relative to volume)
    market_impact = 0.0
    if order_size_shares > 0 and avg_bar_volume > 0:
        participation_rate = order_size_shares / avg_bar_volume
        # Impact scales with sqrt of participation rate (square-root market impact model)
        import math
        market_impact = price * 0.001 * math.sqrt(participation_rate)

    total = base + adverse_selection + market_impact

    # 4. High-vol multiplier on base + adverse selection (not market impact)
    if is_high_vol:
        total = (base + adverse_selection) * settings.high_vol_slippage_multiplier + market_impact

    return round(max(0.0, total), 4)


def apply_slippage(
    price: float,
    spread_pct: float,
    side: OrderSide,
    settings: SlippageSettings,
    is_high_vol: bool = False,
    order_size_shares: int = 0,
    avg_bar_volume: float = 0.0,
) -> float:
    """Return fill price after applying realistic slippage."""
    slip = estimate_slippage(
        price, spread_pct, side, settings, is_high_vol,
        order_size_shares, avg_bar_volume,
    )
    if side in (OrderSide.BUY, OrderSide.BUY_TO_COVER):
        return price + slip   # Buys fill higher
    return price - slip       # Sells fill lower
