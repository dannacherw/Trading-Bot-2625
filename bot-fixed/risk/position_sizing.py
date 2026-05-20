"""
risk/position_sizing.py
Position sizing based on dollar risk per trade.
Applies adjustments for volatility, liquidity, spread, and account equity.
"""
from __future__ import annotations

import math

from loguru import logger

from core.config import (
    LiquiditySettings,
    PositionLimitSettings,
    SlippageSettings,
    VolatilitySettings,
)
from core.exceptions import PositionSizingError


def compute_base_shares(
    entry_price: float,
    stop_price: float,
    account_equity: float,
    risk_pct: float,  # e.g. 0.50 for 0.50%
) -> int:
    """
    Base position size using dollar-risk formula.
    shares = (equity * risk_pct/100) / |entry - stop|

    At $10k with 0.50% risk:
      dollar_risk = $50
      On a $20 stock with $0.50 stop: 100 shares
      On a $50 stock with $1.00 stop: 50 shares
    These are workable sizes. Minimum 1 share returned if math is valid.
    """
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share <= 0.01:
        raise PositionSizingError(
            f"Invalid risk per share: entry={entry_price:.4f} stop={stop_price:.4f} "
            f"(risk={risk_per_share:.4f} — stop too close to entry)"
        )
    if account_equity <= 0:
        raise PositionSizingError(f"Invalid equity: {account_equity}")

    dollar_risk = account_equity * (risk_pct / 100.0)
    raw_shares = dollar_risk / risk_per_share

    # At $10k, minimum viable trade is 1 share if the math works out
    return max(0, int(raw_shares))


def apply_volatility_adjustment(
    shares: int,
    atr: float,
    avg_atr: float,
    settings: VolatilitySettings,
) -> int:
    """
    Scale down size if current ATR is elevated relative to average.
    """
    if avg_atr <= 0 or atr <= 0:
        return shares
    atr_ratio = atr / avg_atr
    if atr_ratio > settings.atr_adjustment_threshold:
        adjusted = int(shares * settings.atr_size_reduction_factor)
        logger.debug(
            "Volatility adjustment: ATR ratio={:.2f} — size {} → {}",
            atr_ratio, shares, adjusted,
        )
        return adjusted
    return shares


def apply_liquidity_adjustment(
    shares: int,
    entry_price: float,
    avg_daily_volume: float,
    settings: LiquiditySettings,
) -> int:
    """
    Cap size to a fraction of average daily volume.
    Also reduce for low dollar-volume stocks.
    """
    max_shares_adv = int(avg_daily_volume * settings.max_pct_of_adv / 100.0)
    shares = min(shares, max_shares_adv)

    # Reduce if dollar volume is below threshold for full size
    dollar_vol = entry_price * avg_daily_volume
    if dollar_vol < settings.min_dollar_volume_for_full_size:
        scale = dollar_vol / settings.min_dollar_volume_for_full_size
        shares = int(shares * scale)
        logger.debug(
            "Liquidity adjustment: $vol={:.1f}M → size scaled to {}",
            dollar_vol / 1e6, shares,
        )
    return shares


def apply_spread_adjustment(
    shares: int,
    spread_pct: float,
    max_spread_pct: float,
    penalty_factor: float,
) -> int:
    """Reduce size if the spread is wide."""
    if spread_pct > max_spread_pct:
        adjusted = int(shares * penalty_factor)
        logger.debug(
            "Spread adjustment: spread={:.3f}% > max={:.3f}% — size {} → {}",
            spread_pct, max_spread_pct, shares, adjusted,
        )
        return adjusted
    return shares


def apply_capital_cap(
    shares: int,
    entry_price: float,
    account_equity: float,
    max_capital_pct: float,
) -> int:
    """Never put more than max_capital_pct% of equity in one trade."""
    max_value = account_equity * (max_capital_pct / 100.0)
    max_shares_by_capital = int(max_value / entry_price)
    return min(shares, max_shares_by_capital)


def compute_final_position_size(
    entry_price: float,
    stop_price: float,
    account_equity: float,
    spread_pct: float,
    avg_daily_volume: float,
    atr: float,
    avg_atr: float,
    position_settings: PositionLimitSettings,
    volatility_settings: VolatilitySettings,
    liquidity_settings: LiquiditySettings,
    risk_pct: float | None = None,
    max_spread_pct: float = 0.10,
    spread_penalty: float = 0.5,
) -> int:
    """
    Full position sizing pipeline.
    Returns final approved share count (may be 0 if all adjustments kill the size).
    """
    if risk_pct is None:
        risk_pct = position_settings.default_risk_per_trade_pct

    # Clamp risk_pct to configured bounds
    risk_pct = max(
        position_settings.min_risk_per_trade_pct,
        min(risk_pct, position_settings.max_risk_per_trade_pct),
    )

    # Base size
    shares = compute_base_shares(entry_price, stop_price, account_equity, risk_pct)

    # Adjustments (applied in order)
    shares = apply_volatility_adjustment(shares, atr, avg_atr, volatility_settings)
    shares = apply_liquidity_adjustment(shares, entry_price, avg_daily_volume, liquidity_settings)
    shares = apply_spread_adjustment(shares, spread_pct, max_spread_pct, spread_penalty)
    shares = apply_capital_cap(
        shares, entry_price, account_equity, position_settings.max_capital_per_position_pct
    )

    # Minimum dollar value check
    if shares * entry_price < position_settings.min_trade_dollar_value:
        logger.debug(
            "Position size too small: {} shares @ {:.2f} = ${:.0f} < min ${}",
            shares, entry_price, shares * entry_price, position_settings.min_trade_dollar_value,
        )
        return 0

    return shares
