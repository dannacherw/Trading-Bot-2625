"""
tests/test_strategy/test_signal_tracker.py
Tests for the rolling signal outcome tracker.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from core.enums import ExitReason, SignalStrength
from strategy.signal_tracker import SignalOutcomeTracker


def _ts(hour: int = 14, minute: int = 0) -> datetime:
    return datetime(2024, 1, 15, hour, minute, tzinfo=timezone.utc)


def _fill(
    tracker: SignalOutcomeTracker,
    signal_id: str,
    winner: bool,
    strength: SignalStrength = SignalStrength.STRONG,
    gap_pct: float = 5.0,
    hour: int = 14,
) -> None:
    tracker.record_signal(
        signal_id=signal_id,
        symbol="AAPL",
        strength=strength,
        generated_at=_ts(hour=hour),
        entry_price=100.0,
        gap_pct=gap_pct,
        has_catalyst=True,
    )
    tracker.resolve_signal(
        signal_id=signal_id,
        exit_price=101.5 if winner else 99.0,
        exit_reason=ExitReason.TARGET_1 if winner else ExitReason.STOP_LOSS,
        net_pnl=75.0 if winner else -50.0,
        r_multiple=1.5 if winner else -1.0,
    )


class TestRecordAndResolve:
    def test_pending_signal_not_in_resolved(self):
        tracker = SignalOutcomeTracker(persist_path=None)
        tracker.record_signal("s1", "AAPL", SignalStrength.STRONG, _ts(), 100.0)
        assert tracker.get_rolling_win_rate() is None  # No resolved signals

    def test_resolved_signal_counted(self):
        tracker = SignalOutcomeTracker(persist_path=None)
        for i in range(10):
            _fill(tracker, f"s{i}", winner=True)
        wr = tracker.get_rolling_win_rate()
        assert wr == pytest.approx(1.0)

    def test_mixed_win_rate(self):
        tracker = SignalOutcomeTracker(persist_path=None)
        for i in range(6):
            _fill(tracker, f"win{i}", winner=True)
        for i in range(4):
            _fill(tracker, f"lose{i}", winner=False)
        wr = tracker.get_rolling_win_rate()
        assert wr == pytest.approx(0.6, abs=0.01)

    def test_unknown_signal_id_ignored(self):
        tracker = SignalOutcomeTracker(persist_path=None)
        # Should not raise
        tracker.resolve_signal("nonexistent", 101.0, ExitReason.TARGET_1, 50.0, 1.5)


class TestRollingWindowByStrength:
    def test_rolling_filtered_by_strength(self):
        tracker = SignalOutcomeTracker(persist_path=None)
        for i in range(8):
            _fill(tracker, f"strong{i}", winner=(i % 2 == 0), strength=SignalStrength.STRONG)
        for i in range(8):
            _fill(tracker, f"mod{i}", winner=True, strength=SignalStrength.MODERATE)

        strong_wr = tracker.get_rolling_win_rate(strength=SignalStrength.STRONG)
        mod_wr = tracker.get_rolling_win_rate(strength=SignalStrength.MODERATE)
        assert strong_wr is not None
        assert mod_wr is not None
        assert mod_wr > strong_wr

    def test_returns_none_below_min_samples(self):
        tracker = SignalOutcomeTracker(persist_path=None)
        _fill(tracker, "s1", winner=True)
        _fill(tracker, "s2", winner=False)
        # Only 2 resolved — below min 5
        result = tracker.get_rolling_win_rate()
        assert result is None


class TestPerformanceSlices:
    def test_all_dimensions_populated(self):
        tracker = SignalOutcomeTracker(persist_path=None)
        for i in range(12):
            _fill(tracker, f"s{i}", winner=(i % 3 != 0), gap_pct=5.0 + i * 0.5, hour=14)
        slices = tracker.get_all_slices()
        keys = {s.dimension for s in slices}
        assert "overall" in keys
        assert any("strength=" in k for k in keys)
        assert any("time=" in k for k in keys)
        assert any("gap=" in k for k in keys)

    def test_overall_accuracy_correct(self):
        tracker = SignalOutcomeTracker(persist_path=None)
        for i in range(20):
            _fill(tracker, f"s{i}", winner=(i < 13))  # 13 wins, 7 losses
        slices = {s.dimension: s for s in tracker.get_all_slices()}
        overall = slices.get("overall")
        assert overall is not None
        assert overall.win_rate == pytest.approx(13 / 20, abs=0.01)


class TestWarnings:
    def test_warning_logged_on_overall_degradation(self):
        tracker = SignalOutcomeTracker(persist_path=None)
        with patch("strategy.signal_tracker.logger") as mock_log:
            for i in range(20):
                _fill(tracker, f"s{i}", winner=(i < 8))  # 40% WR — below 45% threshold
            calls = [str(c) for c in mock_log.warning.call_args_list]
            assert any("SIGNAL QUALITY" in c or "DEGRADATION" in c for c in calls)

    def test_no_warning_on_good_performance(self):
        tracker = SignalOutcomeTracker(persist_path=None)
        with patch("strategy.signal_tracker.logger") as mock_log:
            for i in range(20):
                _fill(tracker, f"s{i}", winner=(i < 16))  # 80% WR
            # No degradation warnings should be logged
            warning_calls = [str(c) for c in mock_log.warning.call_args_list]
            degradation_warnings = [c for c in warning_calls if "DEGRADATION" in c]
            assert len(degradation_warnings) == 0


class TestPersistence:
    def test_saves_and_loads_from_disk(self, tmp_path):
        path = str(tmp_path / "tracker.json")
        tracker1 = SignalOutcomeTracker(persist_path=path)
        for i in range(6):
            _fill(tracker1, f"s{i}", winner=(i % 2 == 0))

        # Load in new instance
        tracker2 = SignalOutcomeTracker(persist_path=path)
        wr = tracker2.get_rolling_win_rate()
        assert wr is not None
        assert 0.0 <= wr <= 1.0
