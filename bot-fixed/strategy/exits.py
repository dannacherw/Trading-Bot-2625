"""
strategy/exits.py
Exit signal detection for open positions.
Handles target exits, stop-loss, trailing stops, time exits, and EOD.
"""
from __future__ import annotations

from datetime import datetime, time

from loguru import logger

from core.config import ExitSettings
from core.enums import ExitReason, PositionSide, SignalType
from core.models import Bar, Position, Signal


def check_stop_loss(position: Position, current_price: float) -> ExitReason | None:
    """Check if position has hit its stop loss."""
    if position.side == PositionSide.LONG and current_price <= position.stop_price:
        return ExitReason.STOP_LOSS
    if position.side == PositionSide.SHORT and current_price >= position.stop_price:
        return ExitReason.STOP_LOSS
    return None


def check_trailing_stop(
    position: Position, current_price: float
) -> ExitReason | None:
    """Check if trailing stop has been triggered."""
    if position.trailing_stop_price is None:
        return None
    if position.side == PositionSide.LONG and current_price <= position.trailing_stop_price:
        return ExitReason.TRAILING_STOP
    if position.side == PositionSide.SHORT and current_price >= position.trailing_stop_price:
        return ExitReason.TRAILING_STOP
    return None


def check_target_1(position: Position, current_price: float) -> bool:
    """Check if first profit target has been hit."""
    if position.side == PositionSide.LONG:
        return current_price >= position.target_1_price
    return current_price <= position.target_1_price


def check_target_2(position: Position, current_price: float) -> bool:
    """Check if second profit target has been hit."""
    if position.side == PositionSide.LONG:
        return current_price >= position.target_2_price
    return current_price <= position.target_2_price


def compute_updated_trailing_stop(
    position: Position,
    current_price: float,
    atr: float,
    settings: ExitSettings,
) -> float | None:
    """
    Compute new trailing stop price.
    Trail is activated after the position reaches `trail_activation_r` R-multiple.
    Returns new stop price, or None if trail not yet activated.
    """
    r = position.r_multiple(current_price)
    if r < settings.trail_activation_r:
        return None

    trail_offset = atr * settings.trail_offset_atr_multiple
    if position.side == PositionSide.LONG:
        new_trail = current_price - trail_offset
        # Never move trail backwards
        current_trail = position.trailing_stop_price or position.stop_price
        return max(new_trail, current_trail)
    else:
        new_trail = current_price + trail_offset
        current_trail = position.trailing_stop_price or position.stop_price
        return min(new_trail, current_trail)


def compute_breakeven_stop(position: Position) -> float:
    """Return entry price + small buffer as break-even stop."""
    buffer = position.entry_price * 0.001  # 10bps buffer
    if position.side == PositionSide.LONG:
        return position.entry_price + buffer
    return position.entry_price - buffer


def should_move_to_breakeven(
    position: Position,
    current_price: float,
    settings: ExitSettings,
) -> bool:
    """Check if position should be moved to break-even."""
    if position.breakeven_price is not None:
        return False  # Already at break-even
    r = position.r_multiple(current_price)
    return r >= settings.move_to_breakeven_r


def check_time_exit(
    position: Position,
    current_time: datetime,
    settings: ExitSettings,
) -> ExitReason | None:
    """Check all time-based exit conditions."""
    # Max hold time
    hold_minutes = (current_time - position.entry_time).total_seconds() / 60.0
    if hold_minutes >= settings.max_hold_minutes:
        return ExitReason.TIME_EXIT

    # EOD exit
    eod_h, eod_m = (int(x) for x in settings.eod_exit_time.split(":"))
    eod = time(eod_h, eod_m)
    current_t = current_time.time()
    if current_t >= eod:
        return ExitReason.EOD_EXIT

    return None


def evaluate_all_exits(
    position: Position,
    current_price: float,
    current_time: datetime,
    atr: float,
    settings: ExitSettings,
) -> tuple[ExitReason | None, float | None]:
    """
    Full exit evaluation for an open position.
    Returns (exit_reason, new_stop_or_trail) or (None, updated_stop).

    Priority:
    1. Stop loss / trailing stop (hard exits)
    2. Time exits
    3. Target exits (handled separately as partial fills)
    """
    # 1. Hard stops (highest priority)
    trailing_reason = check_trailing_stop(position, current_price)
    if trailing_reason:
        return trailing_reason, None

    stop_reason = check_stop_loss(position, current_price)
    if stop_reason:
        return stop_reason, None

    # 2. Time exits
    time_reason = check_time_exit(position, current_time, settings)
    if time_reason:
        return time_reason, None

    # 3. Compute updated trailing stop (no exit yet)
    if settings.use_trailing_stop:
        new_trail = compute_updated_trailing_stop(position, current_price, atr, settings)
        return None, new_trail

    return None, None
