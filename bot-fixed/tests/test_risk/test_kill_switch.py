"""
tests/test_risk/test_kill_switch.py
Tests for the KillSwitchMonitor soft-halt system.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from risk.kill_switch import (
    KillSwitchEvent,
    KillSwitchMonitor,
    KillSwitchSettings,
    KillSwitchTrigger,
)


def _monitor(
    max_losses: int = 3,
    max_stale_mins: float = 10.0,
    max_spread: float = 0.40,
    max_spy_down: float = -1.5,
    max_vix: float = 35.0,
) -> KillSwitchMonitor:
    settings = KillSwitchSettings(
        max_consecutive_losses=max_losses,
        data_staleness_halt_enabled=True,
        max_stale_scan_minutes=max_stale_mins,
        spread_expansion_halt_enabled=True,
        max_avg_spread_pct=max_spread,
        broker_disconnect_halt_enabled=True,
        volatility_halt_enabled=True,
        max_spy_intraday_down_pct=max_spy_down,
        max_vix_level=max_vix,
    )
    return KillSwitchMonitor(settings)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_not_halted_initially(self):
        m = _monitor()
        assert not m.is_halted

    def test_entry_allowed_initially(self):
        m = _monitor()
        allowed, reason = m.check_entry_allowed()
        assert allowed
        assert reason == ""

    def test_no_active_triggers(self):
        m = _monitor()
        assert m.active_triggers == []

    def test_zero_consecutive_losses(self):
        m = _monitor()
        assert m.consecutive_losses == 0


# ---------------------------------------------------------------------------
# Consecutive loss halt
# ---------------------------------------------------------------------------

class TestConsecutiveLossHalt:
    def test_halt_after_max_consecutive_losses(self):
        m = _monitor(max_losses=3)
        m.record_trade_result(-100.0)
        m.record_trade_result(-200.0)
        assert not m.is_halted
        m.record_trade_result(-50.0)
        assert m.is_halted
        assert KillSwitchTrigger.CONSECUTIVE_LOSSES in m.active_triggers

    def test_win_resets_counter(self):
        m = _monitor(max_losses=3)
        m.record_trade_result(-100.0)
        m.record_trade_result(-200.0)
        m.record_trade_result(50.0)   # Win resets
        assert m.consecutive_losses == 0
        assert not m.is_halted

    def test_win_after_halt_does_not_auto_resolve(self):
        """A win does NOT automatically clear a consecutive loss halt."""
        m = _monitor(max_losses=2)
        m.record_trade_result(-100.0)
        m.record_trade_result(-100.0)
        assert m.is_halted
        m.record_trade_result(500.0)  # Big win
        # Still halted — requires human restart
        assert m.is_halted

    def test_no_halt_below_threshold(self):
        m = _monitor(max_losses=3)
        m.record_trade_result(-100.0)
        m.record_trade_result(-100.0)
        assert not m.is_halted

    def test_entry_blocked_when_halted(self):
        m = _monitor(max_losses=1)
        m.record_trade_result(-100.0)
        allowed, reason = m.check_entry_allowed()
        assert not allowed
        assert "CONSECUTIVE_LOSSES" in reason

    def test_halt_recorded_in_event_log(self):
        m = _monitor(max_losses=1)
        m.record_trade_result(-50.0)
        assert len(m.event_log) == 1
        assert m.event_log[0].trigger == KillSwitchTrigger.CONSECUTIVE_LOSSES


# ---------------------------------------------------------------------------
# Data staleness halt
# ---------------------------------------------------------------------------

class TestDataStalenessHalt:
    def test_no_halt_if_no_scans_yet(self):
        """Don't penalise early startup before first scan."""
        m = _monitor(max_stale_mins=5.0)
        now = datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
        m.check_data_staleness(now)
        assert not m.is_halted

    def test_halt_on_stale_scan(self):
        m = _monitor(max_stale_mins=5.0)
        old_time = datetime(2024, 1, 15, 13, 50, 0, tzinfo=timezone.utc)
        now = datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)  # 10 min later
        m.update_last_scan_time(old_time)
        m.check_data_staleness(now)
        assert m.is_halted
        assert KillSwitchTrigger.DATA_STALENESS in m.active_triggers

    def test_fresh_scan_auto_resolves_staleness(self):
        m = _monitor(max_stale_mins=5.0)
        old_time = datetime(2024, 1, 15, 13, 50, 0, tzinfo=timezone.utc)
        now = datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
        m.update_last_scan_time(old_time)
        m.check_data_staleness(now)
        assert m.is_halted
        # Fresh scan clears it automatically
        m.update_last_scan_time(now)
        assert not m.is_halted

    def test_no_halt_within_threshold(self):
        m = _monitor(max_stale_mins=10.0)
        scan_time = datetime(2024, 1, 15, 13, 55, 0, tzinfo=timezone.utc)
        now = datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)  # 5 min later
        m.update_last_scan_time(scan_time)
        m.check_data_staleness(now)
        assert not m.is_halted


# ---------------------------------------------------------------------------
# Spread expansion halt
# ---------------------------------------------------------------------------

class TestSpreadExpansionHalt:
    def test_halt_on_wide_spreads(self):
        m = _monitor(max_spread=0.30)
        m.update_spreads([0.35, 0.40, 0.38])  # All above threshold
        assert m.is_halted
        assert KillSwitchTrigger.SPREAD_EXPANSION in m.active_triggers

    def test_no_halt_on_normal_spreads(self):
        m = _monitor(max_spread=0.30)
        m.update_spreads([0.10, 0.12, 0.08])
        assert not m.is_halted

    def test_auto_resolve_when_spreads_normalise(self):
        m = _monitor(max_spread=0.30)
        m.update_spreads([0.40, 0.45, 0.42])
        assert m.is_halted
        m.update_spreads([0.10, 0.12, 0.08])
        assert not m.is_halted

    def test_too_few_symbols_no_halt(self):
        """Need at least spread_check_min_symbols to trigger."""
        m = _monitor(max_spread=0.10)
        m.update_spreads([0.50, 0.60])  # Only 2 symbols (min is 3)
        assert not m.is_halted


# ---------------------------------------------------------------------------
# Broker disconnect halt
# ---------------------------------------------------------------------------

class TestBrokerDisconnectHalt:
    def test_halt_on_disconnect(self):
        m = _monitor()
        m.update_broker_status(connected=False)
        assert m.is_halted
        assert KillSwitchTrigger.BROKER_DISCONNECT in m.active_triggers

    def test_auto_resolve_on_reconnect(self):
        m = _monitor()
        m.update_broker_status(connected=False)
        assert m.is_halted
        m.update_broker_status(connected=True)
        assert not m.is_halted

    def test_no_halt_when_connected(self):
        m = _monitor()
        m.update_broker_status(connected=True)
        assert not m.is_halted


# ---------------------------------------------------------------------------
# Volatility circuit breaker
# ---------------------------------------------------------------------------

class TestVolatilityCircuit:
    def test_halt_on_spy_down(self):
        m = _monitor(max_spy_down=-1.5)
        m.update_market_conditions(spy_price=98.5, spy_open=100.0, vix_level=20.0)
        assert m.is_halted
        assert KillSwitchTrigger.VOLATILITY_CIRCUIT in m.active_triggers

    def test_no_halt_on_small_spy_move(self):
        m = _monitor(max_spy_down=-1.5)
        m.update_market_conditions(spy_price=99.5, spy_open=100.0, vix_level=20.0)
        assert not m.is_halted

    def test_halt_on_high_vix(self):
        m = _monitor(max_vix=35.0)
        m.update_market_conditions(spy_price=100.0, spy_open=100.0, vix_level=40.0)
        assert m.is_halted

    def test_no_halt_on_normal_vix(self):
        m = _monitor(max_vix=35.0)
        m.update_market_conditions(spy_price=100.0, spy_open=100.0, vix_level=20.0)
        assert not m.is_halted

    def test_auto_resolve_when_conditions_normalise(self):
        m = _monitor(max_spy_down=-1.5)
        m.update_market_conditions(spy_price=98.0, spy_open=100.0, vix_level=20.0)
        assert m.is_halted
        m.update_market_conditions(spy_price=99.5, spy_open=100.0, vix_level=20.0)
        assert not m.is_halted


# ---------------------------------------------------------------------------
# Manual halt
# ---------------------------------------------------------------------------

class TestManualHalt:
    def test_manual_trigger(self):
        m = _monitor()
        m.trigger_manual_halt("Operator halted for maintenance")
        assert m.is_halted
        assert KillSwitchTrigger.MANUAL in m.active_triggers

    def test_manual_halt_message_in_status(self):
        m = _monitor()
        m.trigger_manual_halt("Test halt")
        allowed, reason = m.check_entry_allowed()
        assert not allowed
        assert "MANUAL" in reason


# ---------------------------------------------------------------------------
# Resume / resolution
# ---------------------------------------------------------------------------

class TestResume:
    def test_resume_specific_trigger(self):
        m = _monitor(max_losses=1, max_spread=0.10)
        m.record_trade_result(-100.0)   # Triggers consecutive loss halt
        m.update_spreads([0.50, 0.60, 0.70])  # Triggers spread halt
        assert len(m.active_triggers) == 2
        m.resume(KillSwitchTrigger.CONSECUTIVE_LOSSES)
        assert KillSwitchTrigger.CONSECUTIVE_LOSSES not in m.active_triggers
        assert KillSwitchTrigger.SPREAD_EXPANSION in m.active_triggers
        assert m.is_halted  # Still halted due to spread

    def test_resume_all(self):
        m = _monitor(max_losses=1)
        m.record_trade_result(-100.0)
        m.trigger_manual_halt("Test")
        assert len(m.active_triggers) == 2
        m.resume()  # Clear all
        assert not m.is_halted
        assert m.active_triggers == []

    def test_resume_nonexistent_trigger_no_error(self):
        m = _monitor()
        m.resume(KillSwitchTrigger.MANUAL)  # Not active — should not raise


# ---------------------------------------------------------------------------
# Multiple concurrent halts
# ---------------------------------------------------------------------------

class TestMultipleConcurrentHalts:
    def test_multiple_triggers_all_active(self):
        m = _monitor(max_losses=1, max_spread=0.10)
        m.record_trade_result(-100.0)
        m.update_spreads([0.50, 0.60, 0.70])
        assert len(m.active_triggers) == 2
        assert KillSwitchTrigger.CONSECUTIVE_LOSSES in m.active_triggers
        assert KillSwitchTrigger.SPREAD_EXPANSION in m.active_triggers

    def test_duplicate_trigger_not_duplicated(self):
        m = _monitor(max_losses=1)
        m.record_trade_result(-100.0)
        m.record_trade_result(-100.0)  # Already halted
        assert len(m.event_log) == 1  # Only one event logged


# ---------------------------------------------------------------------------
# Daily reset
# ---------------------------------------------------------------------------

class TestDailyReset:
    def test_daily_reset_clears_counters(self):
        m = _monitor(max_losses=5)
        m.record_trade_result(-100.0)
        m.record_trade_result(-100.0)
        assert m.consecutive_losses == 2
        m.reset_daily()
        assert m.consecutive_losses == 0

    def test_daily_reset_does_not_clear_active_halts(self):
        """Halts survive daily reset — require human resolution."""
        m = _monitor(max_losses=1)
        m.record_trade_result(-100.0)
        assert m.is_halted
        m.reset_daily()
        assert m.is_halted  # Still halted — requires human


# ---------------------------------------------------------------------------
# Status summary
# ---------------------------------------------------------------------------

class TestStatusSummary:
    def test_status_summary_not_halted(self):
        m = _monitor()
        summary = m.status_summary()
        assert summary["halted"] is False
        assert summary["consecutive_losses"] == 0

    def test_status_summary_when_halted(self):
        m = _monitor(max_losses=1)
        m.record_trade_result(-50.0)
        summary = m.status_summary()
        assert summary["halted"] is True
        assert "CONSECUTIVE_LOSSES" in summary["active_triggers"]

    def test_status_summary_spy_move(self):
        m = _monitor()
        m.update_market_conditions(spy_price=98.0, spy_open=100.0)
        summary = m.status_summary()
        assert summary["spy_move_pct"] is not None
        assert abs(summary["spy_move_pct"] - (-2.0)) < 0.01

    def test_status_summary_recent_events(self):
        m = _monitor(max_losses=1)
        m.record_trade_result(-100.0)
        summary = m.status_summary()
        assert len(summary["recent_events"]) == 1
        assert summary["recent_events"][0]["trigger"] == "CONSECUTIVE_LOSSES"
