"""
tests/test_strategy/test_trade_manager.py
Tests for the TradeManager — position lifecycle, partial exits, stops.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.enums import ExitReason, PositionStatus
from strategy.trade_manager import TradeManager


class TestTradeManager:
    @pytest.fixture
    def manager(self, exit_settings):
        return TradeManager(exit_settings, commission_per_share=0.005)

    def test_open_position_registers(self, manager, sample_signal):
        pos = manager.open_position(
            signal=sample_signal,
            filled_price=101.00,
            filled_quantity=50,
            filled_at=datetime(2024, 1, 15, 14, 15, tzinfo=timezone.utc),
        )
        assert manager.open_position_count == 1
        assert pos.symbol == "AAPL"
        assert pos.remaining_quantity == 50

    def test_stop_loss_triggers_close(self, manager, sample_signal, exit_settings):
        pos = manager.open_position(
            signal=sample_signal,
            filled_price=101.00,
            filled_quantity=50,
            filled_at=datetime(2024, 1, 15, 14, 15, tzinfo=timezone.utc),
        )
        # Price at stop
        now = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        updated, actions = manager.update_position(pos.position_id, 100.10, now, atr=0.50)
        exit_actions = [a for a in actions if a["action"] == "full_exit"]
        assert len(exit_actions) > 0
        assert exit_actions[0]["reason"] == ExitReason.STOP_LOSS.value

    def test_target_1_triggers_partial_exit(self, manager, sample_signal):
        pos = manager.open_position(
            signal=sample_signal,
            filled_price=101.00,
            filled_quantity=50,
            filled_at=datetime(2024, 1, 15, 14, 15, tzinfo=timezone.utc),
        )
        # Price at target 1 (102.20)
        now = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        updated, actions = manager.update_position(pos.position_id, 102.25, now, atr=0.50)
        partial_actions = [a for a in actions if a["action"] == "partial_exit"]
        assert len(partial_actions) > 0
        assert partial_actions[0]["reason"] == ExitReason.TARGET_1.value
        # Remaining quantity should be reduced
        assert updated.remaining_quantity < pos.remaining_quantity

    def test_build_trade_record_from_closed_position(self, manager, sample_signal):
        pos = manager.open_position(
            signal=sample_signal,
            filled_price=101.00,
            filled_quantity=50,
            filled_at=datetime(2024, 1, 15, 14, 15, tzinfo=timezone.utc),
        )
        now = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        # Force a stop exit
        updated, actions = manager.update_position(pos.position_id, 100.10, now, atr=0.50)

        # After stop triggers, position should be closed
        if updated.status == PositionStatus.CLOSED:
            trade = manager.build_trade_record(updated)
            assert trade.symbol == "AAPL"
            assert trade.hold_duration_seconds > 0

    def test_breakeven_stop_moves_correctly(self, manager, sample_signal, exit_settings):
        pos = manager.open_position(
            sample_signal, filled_price=101.00, filled_quantity=50,
            filled_at=datetime(2024, 1, 15, 14, 15, tzinfo=timezone.utc),
        )
        # 1R = 0.80 per share, so 1R gain = 101.80
        now = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        updated, actions = manager.update_position(pos.position_id, 101.85, now, atr=0.50)
        stop_actions = [a for a in actions if a["action"] == "stop_update" and a["reason"] == "breakeven"]
        assert len(stop_actions) > 0
        assert stop_actions[0]["price"] >= pos.entry_price

    def test_get_open_positions_after_close(self, manager, sample_signal):
        pos = manager.open_position(
            sample_signal, filled_price=101.00, filled_quantity=50,
            filled_at=datetime(2024, 1, 15, 14, 15, tzinfo=timezone.utc),
        )
        assert manager.open_position_count == 1
        now = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        manager.update_position(pos.position_id, 99.50, now, atr=0.50)
        # After stop, count should drop to 0
        assert manager.open_position_count == 0
