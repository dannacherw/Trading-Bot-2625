"""
tests/test_risk/test_risk_engine.py
Tests for the risk engine — trade validation, daily limits, halt.
"""
from __future__ import annotations

import pytest

from core.enums import RiskCheckResult, RiskRejectionReason
from risk.risk_engine import RiskEngine


class TestRiskEngineValidation:
    def test_approves_valid_signal(self, risk_config, sample_signal):
        engine = RiskEngine(risk_config)
        result = engine.validate_signal(
            signal=sample_signal,
            spread_pct=0.05,
            avg_daily_volume=1_000_000,
            atr=0.50,
            avg_atr=0.50,
            current_equity=10_000.0,
        )
        assert result.result == RiskCheckResult.APPROVED
        assert result.approved_quantity > 0

    def test_rejects_when_halted(self, risk_config, sample_signal):
        engine = RiskEngine(risk_config)
        engine._is_halted = True
        result = engine.validate_signal(
            sample_signal, spread_pct=0.05,
            avg_daily_volume=1_000_000, atr=0.50, avg_atr=0.50,
        )
        assert result.result == RiskCheckResult.REJECTED
        assert result.rejection_reason == RiskRejectionReason.DAILY_LOSS_LIMIT

    def test_rejects_when_max_positions_reached(self, risk_config, sample_signal, sample_position):
        engine = RiskEngine(risk_config)
        # Fill up all position slots
        for i in range(risk_config.position_limits.max_open_positions):
            engine.register_trade_opened(f"SYM{i}", sample_position)
        result = engine.validate_signal(
            sample_signal, spread_pct=0.05,
            avg_daily_volume=1_000_000, atr=0.50, avg_atr=0.50,
        )
        assert result.result == RiskCheckResult.REJECTED
        assert result.rejection_reason == RiskRejectionReason.MAX_POSITIONS

    def test_rejects_when_max_trades_reached(self, risk_config, sample_signal):
        engine = RiskEngine(risk_config)
        engine._trades_today = risk_config.daily_limits.max_trades_per_day
        result = engine.validate_signal(
            sample_signal, spread_pct=0.05,
            avg_daily_volume=1_000_000, atr=0.50, avg_atr=0.50,
        )
        assert result.result == RiskCheckResult.REJECTED
        assert result.rejection_reason == RiskRejectionReason.MAX_TRADES

    def test_rejects_wide_spread(self, risk_config, sample_signal):
        engine = RiskEngine(risk_config)
        result = engine.validate_signal(
            sample_signal, spread_pct=0.50,
            avg_daily_volume=1_000_000, atr=0.50, avg_atr=0.50,
        )
        assert result.result == RiskCheckResult.REJECTED
        assert result.rejection_reason == RiskRejectionReason.SPREAD_TOO_WIDE


class TestRiskEngineState:
    def test_daily_loss_tracking(self, risk_config, sample_signal, sample_position):
        engine = RiskEngine(risk_config)
        engine.register_trade_opened("AAPL", sample_position)
        engine.register_trade_closed("AAPL", pnl=-100.0)
        state = engine.get_state()
        assert state.daily_loss_used == pytest.approx(100.0)
        assert state.trades_today == 1

    def test_halt_on_daily_loss_limit(self, risk_config):
        engine = RiskEngine(risk_config)
        # 2% of $10,000 = $200 loss limit
        engine.register_trade_opened("AAPL", engine._open_positions.get("AAPL"))  # no-op
        engine.register_trade_closed("AAPL", pnl=-250.0)
        assert engine.is_halted is True

    def test_daily_reset_clears_state(self, risk_config, sample_signal, sample_position):
        engine = RiskEngine(risk_config)
        engine.register_trade_opened("AAPL", sample_position)
        engine.register_trade_closed("AAPL", pnl=-50.0)
        engine.reset_daily(new_equity=9_950.0)
        assert engine.trades_today == 0
        assert engine.is_halted is False
        state = engine.get_state()
        assert state.daily_loss_used == 0.0


class TestPositionSizing:
    def test_approved_quantity_is_positive(self, risk_config, sample_signal):
        engine = RiskEngine(risk_config)
        result = engine.validate_signal(
            sample_signal, spread_pct=0.05,
            avg_daily_volume=5_000_000, atr=0.50, avg_atr=0.50,
        )
        if result.result == RiskCheckResult.APPROVED:
            assert result.approved_quantity > 0

    def test_size_within_capital_cap(self, risk_config, sample_signal):
        engine = RiskEngine(risk_config)
        result = engine.validate_signal(
            sample_signal, spread_pct=0.05,
            avg_daily_volume=5_000_000, atr=0.50, avg_atr=0.50,
            current_equity=10_000.0,
        )
        if result.result == RiskCheckResult.APPROVED:
            max_value = 10_000.0 * risk_config.position_limits.max_capital_per_position_pct / 100
            assert result.approved_quantity * sample_signal.entry_price <= max_value + 1
