"""
risk/stop_loss_engine.py
Dynamic stop loss computation. Computes initial stops from ATR and VWAP.
Also computes hard stop limits from risk configuration.
"""
from __future__ import annotations

from core.config import StopLossSettings
from core.enums import PositionSide


def compute_atr_stop(
    entry_price: float,
    atr: float,
    side: PositionSide,
    multiplier: float = 1.5,
) -> float:
    """ATR-based stop: entry ± (ATR * multiplier)."""
    offset = atr * multiplier
    if side == PositionSide.LONG:
        return entry_price - offset
    return entry_price + offset


def compute_vwap_stop(
    entry_price: float,
    vwap: float,
    side: PositionSide,
    buffer_pct: float = 0.10,
) -> float:
    """
    VWAP-based stop with a small buffer.
    Long: stop just below VWAP.
    Short: stop just above VWAP.
    """
    buffer = vwap * (buffer_pct / 100.0)
    if side == PositionSide.LONG:
        return vwap - buffer
    return vwap + buffer


def compute_max_allowed_stop(
    entry_price: float,
    side: PositionSide,
    max_stop_pct: float = 3.0,
) -> float:
    """
    Hard cap: the stop can never be farther than max_stop_pct% from entry.
    """
    offset = entry_price * (max_stop_pct / 100.0)
    if side == PositionSide.LONG:
        return entry_price - offset
    return entry_price + offset


def compute_optimal_stop(
    entry_price: float,
    vwap: float,
    atr: float,
    side: PositionSide,
    settings: StopLossSettings,
) -> float:
    """
    Select the tightest valid stop from:
    - ATR stop
    - VWAP stop (if enabled)
    Bounded by the hard cap (max_stop_pct).
    """
    atr_stop = compute_atr_stop(entry_price, atr, side, settings.atr_multiplier)

    candidates = [atr_stop]
    if settings.use_vwap_as_stop and vwap > 0:
        vwap_stop = compute_vwap_stop(
            entry_price, vwap, side, settings.vwap_stop_buffer_pct
        )
        candidates.append(vwap_stop)

    hard_cap = compute_max_allowed_stop(entry_price, side, settings.max_stop_pct)

    if side == PositionSide.LONG:
        # Take the higher (tighter) stop, but don't go below the hard cap
        best = max(candidates)
        return max(best, hard_cap)
    else:
        # Take the lower (tighter) stop for shorts
        best = min(candidates)
        return min(best, hard_cap)


def validate_stop(
    entry_price: float,
    stop_price: float,
    side: PositionSide,
    max_stop_pct: float = 3.0,
) -> tuple[bool, str]:
    """Validate that a stop price is within acceptable bounds."""
    if side == PositionSide.LONG:
        if stop_price >= entry_price:
            return False, f"Stop {stop_price:.4f} >= entry {entry_price:.4f} for long"
        pct = (entry_price - stop_price) / entry_price * 100.0
    else:
        if stop_price <= entry_price:
            return False, f"Stop {stop_price:.4f} <= entry {entry_price:.4f} for short"
        pct = (stop_price - entry_price) / entry_price * 100.0

    if pct > max_stop_pct:
        return False, f"Stop distance {pct:.2f}% > max {max_stop_pct:.2f}%"

    return True, ""
