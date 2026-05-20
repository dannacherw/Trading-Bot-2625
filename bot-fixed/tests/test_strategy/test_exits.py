"""
tests/test_strategy/test_exits.py
Tests for all exit condition checks.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.enums import ExitReason, PositionSide
from strategy.exits import (
    check_stop_loss,
    check_target_1,
    check_target_2,
    check_time_exit,
    check_trailing_stop,
    compute_breakeven_stop,
    compute_updated_trailing_stop,
    evaluate_all_exits,
    should_move_to_breakeven,
)


class TestStopLoss:
    def test_long_stop_triggered(self, sample_position):
        reason = check_stop_loss(sample_position, current_price=99.0)
        assert reason == ExitReason.STOP_LOSS

    def test_long_stop_not_triggered(self, sample_position):
        reason = check_stop_loss(sample_position, current_price=101.50)
        assert reason is None

    def test_long_at_exactly_stop(self, sample_position):
        reason = check_stop_loss(sample_position, current_price=100.20)
        assert reason == ExitReason.STOP_LOSS


class TestTrailingStop:
    def test_no_trailing_stop_set(self, sample_position):
        reason = check_trailing_stop(sample_position, 100.0)
        assert reason is None

    def test_trailing_stop_triggered(self, sample_position):
        pos = sample_position.model_copy(update={"trailing_stop_price": 101.50})
        reason = check_trailing_stop(pos, current_price=101.40)
        assert reason == ExitReason.TRAILING_STOP

    def test_trailing_stop_not_triggered(self, sample_position):
        pos = sample_position.model_copy(update={"trailing_stop_price": 101.50})
        reason = check_trailing_stop(pos, current_price=102.00)
        assert reason is None


class TestTargets:
    def test_target_1_hit(self, sample_position):
        # target_1 = 102.20
        assert check_target_1(sample_position, 102.30) is True

    def test_target_1_not_hit(self, sample_position):
        assert check_target_1(sample_position, 101.50) is False

    def test_target_2_hit(self, sample_position):
        # target_2 = 103.00
        assert check_target_2(sample_position, 103.10) is True

    def test_target_2_not_hit(self, sample_position):
        assert check_target_2(sample_position, 102.50) is False


class TestBreakeven:
    def test_should_move_to_breakeven_at_1r(self, sample_position, exit_settings):
        # risk = 101.00 - 100.20 = 0.80 per share
        # 1R = 101.00 + 0.80 = 101.80
        assert should_move_to_breakeven(sample_position, 101.85, exit_settings) is True

    def test_should_not_move_before_1r(self, sample_position, exit_settings):
        assert should_move_to_breakeven(sample_position, 101.40, exit_settings) is False

    def test_already_at_breakeven(self, sample_position, exit_settings):
        pos = sample_position.model_copy(update={"breakeven_price": 101.10})
        assert should_move_to_breakeven(pos, 102.00, exit_settings) is False

    def test_breakeven_stop_above_entry(self, sample_position):
        be = compute_breakeven_stop(sample_position)
        assert be > sample_position.entry_price


class TestTimeExit:
    def test_max_hold_exceeded(self, sample_position, exit_settings):
        late_time = sample_position.entry_time + timedelta(minutes=130)
        reason = check_time_exit(sample_position, late_time, exit_settings)
        assert reason == ExitReason.TIME_EXIT

    def test_eod_exit(self, sample_position, exit_settings):
        # EOD at 15:45 ET ≈ 19:45 UTC
        eod_time = sample_position.entry_time.replace(hour=19, minute=46)
        reason = check_time_exit(sample_position, eod_time, exit_settings)
        assert reason == ExitReason.EOD_EXIT

    def test_normal_time_no_exit(self, sample_position, exit_settings):
        normal_time = sample_position.entry_time + timedelta(minutes=30)
        reason = check_time_exit(sample_position, normal_time, exit_settings)
        assert reason is None


class TestEvaluateAllExits:
    def test_stop_loss_priority(self, sample_position, exit_settings):
        normal_time = sample_position.entry_time + timedelta(minutes=10)
        reason, _ = evaluate_all_exits(
            sample_position, current_price=99.0,
            current_time=normal_time, atr=0.50, settings=exit_settings,
        )
        assert reason == ExitReason.STOP_LOSS

    def test_no_exit_on_healthy_position(self, sample_position, exit_settings):
        normal_time = sample_position.entry_time + timedelta(minutes=10)
        reason, _ = evaluate_all_exits(
            sample_position, current_price=101.50,
            current_time=normal_time, atr=0.50, settings=exit_settings,
        )
        assert reason is None
