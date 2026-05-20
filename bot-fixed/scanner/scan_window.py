"""
scanner/scan_window.py
Premarket scan window enforcement.

The legal scan window is 8:00–9:25 AM ET (05:00–13:25 UTC during EDT,
06:00–14:25 UTC during EST). Scans outside this window are rejected
with a clear reason.

Key behaviours:
  - Hard start: no scans before 8:00 AM ET (premarket data too thin)
  - Hard stop: no new scans after 9:25 AM ET (session about to open)
  - Warning zone: 9:15–9:25 AM ET — scans allowed but flagged
  - Weekend / holiday aware: immediately rejected

Config is driven by ScanWindowSettings injected from scanner_config.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone, timedelta
from typing import NamedTuple

from loguru import logger

# ---------------------------------------------------------------------------
# US market holidays (static list for current + next year)
# Extend annually or replace with a trading-calendar library
# ---------------------------------------------------------------------------

_MARKET_HOLIDAYS_2024_2025: frozenset[date] = frozenset({
    # 2024
    date(2024, 1, 1),   # New Year's Day
    date(2024, 1, 15),  # MLK Day
    date(2024, 2, 19),  # Presidents' Day
    date(2024, 3, 29),  # Good Friday
    date(2024, 5, 27),  # Memorial Day
    date(2024, 6, 19),  # Juneteenth
    date(2024, 7, 4),   # Independence Day
    date(2024, 9, 2),   # Labor Day
    date(2024, 11, 28), # Thanksgiving
    date(2024, 12, 25), # Christmas
    # 2025
    date(2025, 1, 1),   # New Year's Day
    date(2025, 1, 20),  # MLK Day
    date(2025, 2, 17),  # Presidents' Day
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 26),  # Memorial Day
    date(2025, 6, 19),  # Juneteenth
    date(2025, 7, 4),   # Independence Day
    date(2025, 9, 1),   # Labor Day
    date(2025, 11, 27), # Thanksgiving
    date(2025, 12, 25), # Christmas
    # 2026
    date(2026, 1, 1),   # New Year's Day
    date(2026, 1, 19),  # MLK Day
    date(2026, 2, 16),  # Presidents' Day
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 6, 19),  # Juneteenth
    date(2026, 7, 3),   # Independence Day observed
    date(2026, 9, 7),   # Labor Day
    date(2026, 11, 26), # Thanksgiving
    date(2026, 12, 25), # Christmas
})


# ---------------------------------------------------------------------------
# Settings dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScanWindowSettings:
    """Injected via ScannerConfig.scan_window."""
    # 8:00 AM ET in UTC offset (EDT = UTC-4, EST = UTC-5)
    # We handle DST automatically via _et_now()
    window_open_et_hour: int = 8
    window_open_et_minute: int = 0
    window_close_et_hour: int = 9
    window_close_et_minute: int = 25
    warning_zone_et_hour: int = 9
    warning_zone_et_minute: int = 15
    enforce_window: bool = True   # Set False for testing/backtesting
    allow_weekends: bool = False  # Set True only for testing


# ---------------------------------------------------------------------------
# Window check result
# ---------------------------------------------------------------------------

class WindowCheckResult(NamedTuple):
    allowed: bool
    reason: str
    in_warning_zone: bool = False
    minutes_to_open: float | None = None   # Positive = time until window opens
    minutes_to_close: float | None = None  # Positive = time until window closes


# ---------------------------------------------------------------------------
# ET conversion helpers
# ---------------------------------------------------------------------------

def _is_edt(dt: datetime) -> bool:
    """
    Approximate EDT detection: second Sunday in March → first Sunday in November.
    This avoids a pytz/zoneinfo dependency while being accurate for US markets.
    """
    year = dt.year
    # Second Sunday in March
    march1 = date(year, 3, 1)
    march_sun2 = march1 + timedelta(days=(6 - march1.weekday()) % 7 + 7)
    edt_start = datetime(year, 3, march_sun2.day, 2, 0, 0, tzinfo=timezone.utc)
    # First Sunday in November
    nov1 = date(year, 11, 1)
    nov_sun1 = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)
    edt_end = datetime(year, 11, nov_sun1.day, 2, 0, 0, tzinfo=timezone.utc)
    # EDT start and end are stored in local time; we work in UTC
    edt_start_utc = edt_start + timedelta(hours=5)  # UTC = ET + 5 during EST
    edt_end_utc = edt_end + timedelta(hours=4)       # UTC = ET + 4 during EDT
    return edt_start_utc <= dt < edt_end_utc


def _utc_to_et(dt: datetime) -> datetime:
    """Convert UTC datetime to ET (handles EDT/EST)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    offset_hours = -4 if _is_edt(dt) else -5
    return dt + timedelta(hours=offset_hours)


def _et_time_now(now_utc: datetime | None = None) -> time:
    """Return current ET time."""
    utc = now_utc or datetime.now(tz=timezone.utc)
    return _utc_to_et(utc).time()


# ---------------------------------------------------------------------------
# Core window check
# ---------------------------------------------------------------------------

def check_scan_window(
    settings: ScanWindowSettings,
    now_utc: datetime | None = None,
) -> WindowCheckResult:
    """
    Check whether a scan is permitted at the given time.

    Returns a WindowCheckResult with .allowed indicating whether the scan
    should proceed. Callers should log or record the .reason field.
    """
    if not settings.enforce_window:
        return WindowCheckResult(allowed=True, reason="Window enforcement disabled")

    utc = now_utc or datetime.now(tz=timezone.utc)
    today = _utc_to_et(utc).date()

    # Weekend check
    if today.weekday() >= 5 and not settings.allow_weekends:
        return WindowCheckResult(
            allowed=False,
            reason=f"Market closed (weekend: {today.strftime('%A')})",
        )

    # Holiday check
    if today in _MARKET_HOLIDAYS_2024_2025:
        return WindowCheckResult(
            allowed=False,
            reason=f"Market closed (holiday: {today.isoformat()})",
        )

    et_time = _et_time_now(utc)
    window_open  = time(settings.window_open_et_hour,  settings.window_open_et_minute)
    window_close = time(settings.window_close_et_hour, settings.window_close_et_minute)
    warn_zone    = time(settings.warning_zone_et_hour, settings.warning_zone_et_minute)

    # Convert to minutes since midnight for arithmetic
    def _mins(t: time) -> float:
        return t.hour * 60 + t.minute + t.second / 60

    now_mins   = _mins(et_time)
    open_mins  = _mins(window_open)
    close_mins = _mins(window_close)
    warn_mins  = _mins(warn_zone)

    if now_mins < open_mins:
        minutes_to_open = open_mins - now_mins
        return WindowCheckResult(
            allowed=False,
            reason=f"Scan window opens at {window_open.strftime('%H:%M')} ET ({minutes_to_open:.0f}min away)",
            minutes_to_open=minutes_to_open,
        )

    if now_mins >= close_mins:
        return WindowCheckResult(
            allowed=False,
            reason=f"Scan window closed at {window_close.strftime('%H:%M')} ET — session starting soon",
            minutes_to_close=0.0,
        )

    minutes_to_close = close_mins - now_mins
    in_warning = now_mins >= warn_mins

    if in_warning:
        logger.warning(
            "⚠ Scan in warning zone ({:.0f}min to session open) — last scan cycle",
            minutes_to_close,
        )
        return WindowCheckResult(
            allowed=True,
            reason=f"Warning zone: {minutes_to_close:.0f}min until window closes",
            in_warning_zone=True,
            minutes_to_close=minutes_to_close,
        )

    return WindowCheckResult(
        allowed=True,
        reason=f"Window open — {minutes_to_close:.0f}min remaining",
        minutes_to_close=minutes_to_close,
    )


def next_window_open_seconds(
    settings: ScanWindowSettings,
    now_utc: datetime | None = None,
) -> float:
    """
    Return seconds until the next scan window opens.
    Returns 0 if currently inside the window.
    Used by PremarketScanner to sleep until the window opens.
    """
    result = check_scan_window(settings, now_utc)
    if result.allowed:
        return 0.0
    if result.minutes_to_open is not None:
        return result.minutes_to_open * 60
    # Window is closed for today — return seconds to next trading day 8:00 AM ET
    utc = now_utc or datetime.now(tz=timezone.utc)
    et_now = _utc_to_et(utc)
    # Find next weekday
    next_day = et_now.date() + timedelta(days=1)
    while next_day.weekday() >= 5 or next_day in _MARKET_HOLIDAYS_2024_2025:
        next_day += timedelta(days=1)
    offset_hours = 4 if _is_edt(utc) else 5
    next_open_et = datetime(
        next_day.year, next_day.month, next_day.day,
        settings.window_open_et_hour, settings.window_open_et_minute, 0,
    )
    next_open_utc = next_open_et + timedelta(hours=offset_hours)
    next_open_utc = next_open_utc.replace(tzinfo=timezone.utc)
    return max(0.0, (next_open_utc - utc).total_seconds())


def is_market_holiday(check_date: date) -> bool:
    """Return True if *check_date* is a known US market holiday."""
    return check_date in _MARKET_HOLIDAYS_2024_2025


# ---------------------------------------------------------------------------
# Public helper re-exports for other modules
# ---------------------------------------------------------------------------

def seconds_until_time_et(hour: int, minute: int, now_utc: datetime | None = None) -> float:
    """
    Return seconds until the next occurrence of hour:minute ET today.
    Returns 0.0 if that time has already passed today.
    Public wrapper used by analytics modules.
    """
    utc = now_utc or datetime.now(tz=timezone.utc)
    offset_hours = -4 if _is_edt(utc) else -5
    et_now = utc + timedelta(hours=offset_hours)
    target = et_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if et_now >= target:
        return 0.0
    return (target - et_now).total_seconds()


def is_edt(now_utc: datetime | None = None) -> bool:
    """Public wrapper: return True if currently in EDT (US daylight saving time)."""
    utc = now_utc or datetime.now(tz=timezone.utc)
    return _is_edt(utc)
