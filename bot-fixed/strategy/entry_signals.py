"""
strategy/entry_signals.py — v2
VWAP pullback entry signal detection.

Key changes from v1:
  1. Stop placement: structural swing low during pullback, NOT fixed 10bps below VWAP
  2. VWAP reclaim: requires declining volume on touch + meaningful close above VWAP
  3. Relative strength gate: symbol must outperform benchmark by min threshold
  4. All logic remains pure/stateless — takes bar data, returns Signal or None
"""
from __future__ import annotations

from datetime import datetime, time, timezone

from loguru import logger

from core.config import EntrySettings, StopLossSettings
from core.enums import SignalStrength, SignalType
from core.models import Bar, Quote, Signal
from scanner.metrics import compute_intraday_vwap, compute_vwap_slope


def _et_time(dt: datetime) -> time:
    """Extract approximate time-of-day in ET (UTC-4 during EDT)."""
    return dt.time()


# ---------------------------------------------------------------------------
# Time gate
# ---------------------------------------------------------------------------

def is_within_trading_window(
    current_time: datetime,
    settings: EntrySettings,
    market_open: datetime,
) -> tuple[bool, str]:
    """Check all time-gate conditions for entry."""
    minutes_since_open = (current_time - market_open).total_seconds() / 60.0

    if minutes_since_open < settings.earliest_entry_minutes_after_open:
        return False, f"Too early ({minutes_since_open:.0f}min after open)"

    session_minutes = 390.0
    minutes_remaining = session_minutes - minutes_since_open
    if minutes_remaining < settings.latest_entry_minutes_before_close:
        return False, f"Too close to close ({minutes_remaining:.0f}min remaining)"

    t = _et_time(current_time)
    lo_h, lo_m = (int(x) for x in settings.blackout_lunch_start.split(":"))
    hi_h, hi_m = (int(x) for x in settings.blackout_lunch_end.split(":"))
    blackout_start = time(lo_h, lo_m)
    blackout_end = time(hi_h, hi_m)
    if blackout_start <= t <= blackout_end:
        return False, f"Lunch blackout ({settings.blackout_lunch_start}–{settings.blackout_lunch_end})"

    return True, ""


# ---------------------------------------------------------------------------
# Stop placement: structural swing low (v2 — replaces fixed formula)
# ---------------------------------------------------------------------------

def find_pullback_swing_low(
    bars: list[Bar],
    vwap: float,
    lookback: int = 8,
    vwap_proximity_pct: float = 0.20,
) -> float:
    """
    Find the lowest low during the recent pullback to VWAP.

    Identifies bars where price was within `vwap_proximity_pct`% of VWAP
    (i.e., the stock touched or approached VWAP), then returns the minimum
    low of those bars as the structural anchor for the stop.

    This stop is anchored to price structure the market has already tested
    and defended — not to a round-number formula that stops get swept to.

    Args:
        bars: Recent intraday bars
        vwap: Current cumulative VWAP
        lookback: How many bars to look back for the pullback
        vwap_proximity_pct: How close to VWAP counts as "touching" it

    Returns:
        Structural low price to place stop just below
    """
    proximity_band = vwap * (vwap_proximity_pct / 100.0)
    recent = bars[-lookback:]

    # Bars where price touched or dipped near/below VWAP
    pullback_bars = [b for b in recent if b.low <= vwap + proximity_band]

    if not pullback_bars:
        # No clear pullback bars — fall back to a tight VWAP buffer
        return vwap * (1.0 - 0.0015)  # 15bps below VWAP as fallback

    swing_low = min(b.low for b in pullback_bars)

    # Add a 5-cent buffer below the swing low to avoid being stopped
    # by the exact same tick that printed the low
    buffer = max(0.05, swing_low * 0.001)  # $0.05 or 10bps, whichever is larger
    return swing_low - buffer


# ---------------------------------------------------------------------------
# VWAP reclaim check (v2 — replaces simplistic 5-bar check)
# ---------------------------------------------------------------------------

def is_clean_vwap_reclaim(
    bars: list[Bar],
    vwap: float,
    settings: EntrySettings,
    lookback: int = 8,
) -> tuple[bool, str]:
    """
    Validate that a genuine VWAP reclaim has occurred.

    Conditions (all must be true):
    1. At least one of the recent bars touched VWAP (low ≤ VWAP + proximity band)
    2. The touch bar has declining volume vs. the bar before it
       (seller exhaustion on the dip — buyers are absorbing)
    3. The current bar closes meaningfully above VWAP (≥ min_close_above_pct %)
    4. The current bar's close is in the top portion of its range (≥ close_position_min)
       (close near high = buyers still in control)

    Returns (True, "") if all conditions met, (False, reason) otherwise.
    """
    if len(bars) < 4:
        return False, "insufficient bars for reclaim check"

    recent = bars[-lookback:]
    current_bar = recent[-1]

    # --- Condition 1: price touched VWAP in recent bars ---
    proximity_band = vwap * 0.0020  # Within 0.20% of VWAP counts as a touch
    touch_bars = [
        (i, b) for i, b in enumerate(recent[:-1])
        if b.low <= vwap + proximity_band
    ]
    if not touch_bars:
        return False, f"no VWAP touch found in last {lookback} bars"

    # --- Condition 2: touch bar shows declining volume (exhaustion) ---
    if settings.vwap_reclaim_touch_vol_decay:
        touch_idx, touch_bar = touch_bars[-1]
        if touch_idx > 0:
            prev_bar = recent[touch_idx - 1]
            if touch_bar.volume > prev_bar.volume * 0.90:
                return False, (
                    f"touch bar volume not declining "
                    f"({touch_bar.volume:,} vs prev {prev_bar.volume:,}) — not exhaustion"
                )

    # --- Condition 3: current bar closes meaningfully above VWAP ---
    min_above_pct = settings.vwap_reclaim_min_close_above_pct / 100.0
    if current_bar.close < vwap * (1.0 + min_above_pct):
        return False, (
            f"close {current_bar.close:.4f} not sufficiently above VWAP {vwap:.4f} "
            f"(need {min_above_pct*100:.2f}%)"
        )

    # --- Condition 4: close in upper portion of bar range ---
    bar_range = current_bar.high - current_bar.low
    if bar_range > 0:
        close_pos = (current_bar.close - current_bar.low) / bar_range
        if close_pos < settings.vwap_reclaim_close_position_min:
            return False, (
                f"close position {close_pos:.2f} < min {settings.vwap_reclaim_close_position_min:.2f} "
                "(sellers still present in the bar)"
            )

    return True, ""


# ---------------------------------------------------------------------------
# Main entry signal detector
# ---------------------------------------------------------------------------

def detect_vwap_pullback_entry(
    symbol: str,
    bars: list[Bar],
    quote: Quote,
    market_open: datetime,
    settings: EntrySettings,
    atr: float = 0.0,
    relative_strength: float | None = None,
    stop_settings: StopLossSettings | None = None,
) -> Signal | None:
    """
    Detect a VWAP pullback continuation entry signal — v2.

    Conditions checked in order (cheapest first):
    1.  Time gate (not too early, not too late, not in lunch blackout)
    2.  Minimum opening momentum (stock must have established direction)
    3.  Direction: long only (price above open)
    4.  VWAP proximity (price near VWAP — the pullback has completed)
    5.  VWAP reclaim — v2: volume-confirmed, close-position-validated reclaim
    6.  Relative strength vs. benchmark ≥ min_relative_strength (new gate)
    7.  Entry bar quality: bullish, meaningful body
    8.  Volume confirmation on entry bar
    9.  Stop computed from structural swing low (not formula)
    10. Signal strength classification
    """
    if len(bars) < 5:
        return None

    now = bars[-1].timestamp
    time_ok, time_reason = is_within_trading_window(now, settings, market_open)
    if not time_ok:
        logger.debug("{}: time gate — {}", symbol, time_reason)
        return None

    # ---- Compute VWAP ----
    vwap = compute_intraday_vwap(bars)
    if vwap <= 0:
        return None

    # ---- Opening momentum ----
    open_price = bars[0].open
    current_price = bars[-1].close
    opening_move_pct = abs(current_price - open_price) / open_price * 100.0

    if opening_move_pct < settings.min_opening_move_pct:
        return None

    # ---- Direction: long only ----
    if current_price < open_price:
        return None

    # ---- Opening range preservation (Gap 3 fix) ----
    # The single most common trap: stock gaps up, runs 20min, then reverses
    # hard through VWAP. By the time it "reclaims" VWAP it has given back
    # 80%+ of the opening move and institutional buyers are gone.
    # Rule: current price must still be above 40% of the initial opening move.
    # If the stock has retraced more than 60% of its opening move, the thesis
    # is broken regardless of VWAP proximity.
    opening_move_dollars = abs(bars[0].high - open_price)   # High of first bar vs open
    if opening_move_dollars > 0:
        # Use the session high as the reference peak (not just bar[0].high)
        session_high = max(b.high for b in bars)
        move_from_open = session_high - open_price
        current_retrace = session_high - current_price
        retrace_fraction = current_retrace / move_from_open if move_from_open > 0 else 0.0

        # Allow up to 60% retracement — more than that and the move is exhausted
        max_retrace_fraction = getattr(settings, "max_opening_range_retrace_pct", 0.60)
        if retrace_fraction > max_retrace_fraction:
            logger.debug(
                "{}: opening range failed — retraced {:.1%} of opening move "
                "(max {:.1%}). Session high={:.2f} current={:.2f}",
                symbol, retrace_fraction, max_retrace_fraction, session_high, current_price,
            )
            return None

    # ---- VWAP proximity ----
    dist_from_vwap_pct = abs(current_price - vwap) / vwap * 100.0
    if dist_from_vwap_pct > settings.max_entry_distance_from_vwap_pct:
        return None

    # ---- VWAP reclaim — v2 ----
    if settings.require_vwap_reclaim:
        reclaim_ok, reclaim_reason = is_clean_vwap_reclaim(bars, vwap, settings)
        if not reclaim_ok:
            logger.debug("{}: VWAP reclaim failed — {}", symbol, reclaim_reason)
            return None

    # ---- Relative strength gate (new in v2) ----
    if settings.require_relative_strength and relative_strength is not None:
        if relative_strength < settings.min_relative_strength_vs_benchmark:
            logger.debug(
                "{}: RS gate failed — RS={:.2f}% < min {:.2f}%",
                symbol, relative_strength, settings.min_relative_strength_vs_benchmark,
            )
            return None

    # ---- Entry bar quality ----
    entry_bar = bars[-1]
    if not entry_bar.is_bullish:
        return None

    entry_bar_size_pct = entry_bar.body / entry_bar.open * 100.0 if entry_bar.open > 0 else 0.0
    if entry_bar_size_pct < settings.min_entry_bar_size_pct:
        return None

    # ---- Volume confirmation ----
    if len(bars) >= 10:
        avg_bar_vol = sum(b.volume for b in bars[-10:-1]) / 9.0
        if avg_bar_vol > 0 and entry_bar.volume < avg_bar_vol * settings.min_entry_volume_ratio:
            return None

    # ---- Coordinated stop placement (v3) ----
    # Uses StopLossSettings for all thresholds — nothing hardcoded.
    # Takes the TIGHTEST (highest for longs) of three structural methods,
    # bounded by the configured hard cap. Result: most protective valid stop.
    _stop = stop_settings or StopLossSettings()
    entry_price = quote.ask

    # Method 1: Structural swing low — anchored to price structure the
    # market has already tested. Best method when a clean pullback exists.
    structural_stop = find_pullback_swing_low(bars, vwap)

    # Method 2: ATR-based stop — uses configured multiplier (default 1.2x)
    # Provides a volatility-adjusted floor independent of price structure.
    atr_stop = (entry_price - atr * _stop.atr_multiplier) if atr > 0 else structural_stop

    # Method 3: VWAP stop — just below VWAP using configured buffer.
    # Natural support level; if price closes below VWAP the thesis is broken.
    vwap_buffer = vwap * (_stop.vwap_stop_buffer_pct / 100.0)
    vwap_stop = vwap - vwap_buffer

    # Method 4: Hard cap — never risk more than max_stop_pct from entry
    hard_cap_stop = entry_price * (1.0 - _stop.max_stop_pct / 100.0)

    # Tightest valid stop = highest price that is still below entry
    stop_candidates = [structural_stop, atr_stop, vwap_stop]
    stop_candidates = [s for s in stop_candidates if 0 < s < entry_price]
    stop_price = max(stop_candidates) if stop_candidates else hard_cap_stop

    # Hard cap enforcement — stop can never be farther than max_stop_pct
    stop_price = max(stop_price, hard_cap_stop)

    risk = entry_price - stop_price
    if risk <= 0.01:  # Degenerate — stop too close to entry
        return None

    # ---- Targets: R-multiples ----
    target_1 = entry_price + risk * 1.5
    target_2 = entry_price + risk * 2.5

    # ---- Signal strength ----
    slope = compute_vwap_slope(bars)
    rs_score = relative_strength if relative_strength is not None else 0.0

    # Signal strength thresholds calibrated to new RS baseline (min = 0.8%)
    # STRONG:   steep VWAP slope + strong opening move + RS significantly above min
    # MODERATE: flat-to-positive slope + modest move + RS at or above min
    # WEAK:     passed gates but lower conviction — smaller size appropriate
    if slope > 0.02 and opening_move_pct >= 2.5 and rs_score >= 2.0:
        strength = SignalStrength.STRONG
    elif slope >= -0.001 and opening_move_pct >= 1.0 and rs_score >= 0.8:
        strength = SignalStrength.MODERATE
    else:
        strength = SignalStrength.WEAK

    rs_note = f"rs={rs_score:.2f}%" if relative_strength is not None else "rs=N/A"

    return Signal(
        symbol=symbol,
        signal_type=SignalType.VWAP_PULLBACK_LONG,
        strength=strength,
        generated_at=now,
        entry_price=round(entry_price, 4),
        stop_price=round(stop_price, 4),
        target_1_price=round(target_1, 4),
        target_2_price=round(target_2, 4),
        vwap_at_signal=round(vwap, 4),
        notes=(
            f"opening_move={opening_move_pct:.1f}% "
            f"dist_vwap={dist_from_vwap_pct:.3f}% "
            f"slope={slope:.4f} {rs_note} "
            f"stop=swing_low"
        ),
    )
