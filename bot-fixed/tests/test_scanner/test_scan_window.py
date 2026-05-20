"""
tests/test_scanner/test_scan_window.py
Tests for scan window enforcement — 8:00–9:25 AM ET gate.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from scanner.scan_window import (
    ScanWindowSettings,
    WindowCheckResult,
    _is_edt,
    _utc_to_et,
    check_scan_window,
    is_market_holiday,
    next_window_open_seconds,
)


def _utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# EDT detection
# ---------------------------------------------------------------------------

class TestEdtDetection:
    def test_july_is_edt(self):
        assert _is_edt(_utc(2024, 7, 15, 12))

    def test_january_is_not_edt(self):
        assert not _is_edt(_utc(2024, 1, 15, 12))

    def test_december_is_not_edt(self):
        assert not _is_edt(_utc(2024, 12, 1, 12))

    def test_march_before_change_is_not_edt(self):
        # DST starts second Sunday in March 2024 = March 10
        assert not _is_edt(_utc(2024, 3, 9, 12))

    def test_march_after_change_is_edt(self):
        assert _is_edt(_utc(2024, 3, 11, 12))


# ---------------------------------------------------------------------------
# UTC to ET conversion
# ---------------------------------------------------------------------------

class TestUtcToEt:
    def test_summer_utc_to_edt(self):
        # 14:00 UTC in July = 10:00 EDT (UTC-4)
        utc = _utc(2024, 7, 15, 14, 0)
        et = _utc_to_et(utc)
        assert et.hour == 10
        assert et.minute == 0

    def test_winter_utc_to_est(self):
        # 14:00 UTC in January = 09:00 EST (UTC-5)
        utc = _utc(2024, 1, 15, 14, 0)
        et = _utc_to_et(utc)
        assert et.hour == 9
        assert et.minute == 0


# ---------------------------------------------------------------------------
# Holiday detection
# ---------------------------------------------------------------------------

class TestHolidayDetection:
    def test_christmas_2024_is_holiday(self):
        from datetime import date
        assert is_market_holiday(date(2024, 12, 25))

    def test_regular_day_is_not_holiday(self):
        from datetime import date
        assert not is_market_holiday(date(2024, 7, 16))

    def test_thanksgiving_2025(self):
        from datetime import date
        assert is_market_holiday(date(2025, 11, 27))


# ---------------------------------------------------------------------------
# check_scan_window — window enforcement OFF
# ---------------------------------------------------------------------------

class TestScanWindowEnforcementOff:
    def test_always_allowed_when_disabled(self):
        settings = ScanWindowSettings(enforce_window=False)
        # Any time, including weekend
        result = check_scan_window(settings, _utc(2024, 1, 13, 10))  # Saturday
        assert result.allowed
        assert "disabled" in result.reason.lower()


# ---------------------------------------------------------------------------
# check_scan_window — weekends
# ---------------------------------------------------------------------------

class TestScanWindowWeekend:
    def test_saturday_rejected(self):
        settings = ScanWindowSettings()
        result = check_scan_window(settings, _utc(2024, 1, 13, 13, 0))  # Saturday
        assert not result.allowed
        assert "weekend" in result.reason.lower()

    def test_sunday_rejected(self):
        settings = ScanWindowSettings()
        result = check_scan_window(settings, _utc(2024, 1, 14, 13, 0))  # Sunday
        assert not result.allowed


# ---------------------------------------------------------------------------
# check_scan_window — inside/outside window (summer/EDT)
# ---------------------------------------------------------------------------

class TestScanWindowTiming:
    """
    In EDT (UTC-4):
      8:00 AM ET = 12:00 UTC
      9:15 AM ET = 13:15 UTC
      9:25 AM ET = 13:25 UTC
    """

    def test_before_window_rejected(self):
        settings = ScanWindowSettings()
        # 7:00 AM EDT = 11:00 UTC (July weekday)
        result = check_scan_window(settings, _utc(2024, 7, 15, 11, 0))
        assert not result.allowed
        assert result.minutes_to_open is not None
        assert result.minutes_to_open > 0

    def test_inside_window_allowed(self):
        settings = ScanWindowSettings()
        # 8:30 AM EDT = 12:30 UTC
        result = check_scan_window(settings, _utc(2024, 7, 15, 12, 30))
        assert result.allowed
        assert not result.in_warning_zone

    def test_warning_zone_allowed_with_flag(self):
        settings = ScanWindowSettings()
        # 9:20 AM EDT = 13:20 UTC
        result = check_scan_window(settings, _utc(2024, 7, 15, 13, 20))
        assert result.allowed
        assert result.in_warning_zone

    def test_after_window_rejected(self):
        settings = ScanWindowSettings()
        # 9:30 AM EDT = 13:30 UTC (market open)
        result = check_scan_window(settings, _utc(2024, 7, 15, 13, 30))
        assert not result.allowed

    def test_at_exactly_window_open(self):
        settings = ScanWindowSettings()
        # 8:00 AM EDT = 12:00 UTC
        result = check_scan_window(settings, _utc(2024, 7, 15, 12, 0))
        assert result.allowed

    def test_at_exactly_window_close(self):
        settings = ScanWindowSettings()
        # 9:25 AM EDT = 13:25 UTC
        result = check_scan_window(settings, _utc(2024, 7, 15, 13, 25))
        assert not result.allowed

    def test_holiday_rejected(self):
        settings = ScanWindowSettings()
        # Christmas 2024 (Wednesday) at 9:00 AM ET
        result = check_scan_window(settings, _utc(2024, 12, 25, 14, 0))
        assert not result.allowed
        assert "holiday" in result.reason.lower()

    def test_minutes_to_close_positive_inside_window(self):
        settings = ScanWindowSettings()
        result = check_scan_window(settings, _utc(2024, 7, 15, 12, 30))
        assert result.minutes_to_close is not None
        assert result.minutes_to_close > 0

    def test_winter_window(self):
        """Test window enforcement in EST (UTC-5)."""
        settings = ScanWindowSettings()
        # 8:30 AM EST = 13:30 UTC (January weekday)
        result = check_scan_window(settings, _utc(2024, 1, 15, 13, 30))
        assert result.allowed


# ---------------------------------------------------------------------------
# next_window_open_seconds
# ---------------------------------------------------------------------------

class TestNextWindowOpenSeconds:
    def test_inside_window_returns_zero(self):
        settings = ScanWindowSettings()
        # 8:30 AM EDT = 12:30 UTC
        secs = next_window_open_seconds(settings, _utc(2024, 7, 15, 12, 30))
        assert secs == 0.0

    def test_before_window_returns_positive(self):
        settings = ScanWindowSettings()
        # 7:00 AM EDT = 11:00 UTC
        secs = next_window_open_seconds(settings, _utc(2024, 7, 15, 11, 0))
        assert secs > 0

    def test_before_window_correct_duration(self):
        settings = ScanWindowSettings()
        # 7:00 AM EDT = 11:00 UTC → window opens at 12:00 UTC → 60 min = 3600s
        secs = next_window_open_seconds(settings, _utc(2024, 7, 15, 11, 0))
        assert abs(secs - 3600) < 60  # Allow 1-minute tolerance

    def test_after_window_returns_next_day(self):
        settings = ScanWindowSettings()
        # 10:00 AM EDT Monday = 14:00 UTC → next window is Tuesday 8:00 AM
        secs = next_window_open_seconds(settings, _utc(2024, 7, 15, 14, 0))  # Monday
        # Should be ~22 hours (window opens at 12:00 UTC next day)
        assert secs > 20 * 3600
        assert secs < 26 * 3600
