"""
tests/test_risk/test_position_sizing.py
Unit tests for all position sizing functions.
"""
from __future__ import annotations

import pytest

from core.config import LiquiditySettings, PositionLimitSettings, VolatilitySettings
from core.exceptions import PositionSizingError
from risk.position_sizing import (
    apply_capital_cap,
    apply_liquidity_adjustment,
    apply_spread_adjustment,
    apply_volatility_adjustment,
    compute_base_shares,
    compute_final_position_size,
)


class TestBaseShares:
    def test_normal_calculation(self):
        # $10k equity, 0.35% risk, entry=101, stop=100.20 → risk/share=0.80
        # dollar_risk = $35 → shares = 43
        shares = compute_base_shares(101.0, 100.20, 10_000.0, 0.35)
        assert shares == pytest.approx(43, abs=2)

    def test_zero_risk_raises(self):
        with pytest.raises(PositionSizingError):
            compute_base_shares(100.0, 100.0, 10_000.0, 0.35)

    def test_negative_risk_raises(self):
        with pytest.raises(PositionSizingError):
            compute_base_shares(100.0, 101.0, 10_000.0, 0.35)  # stop above entry (long)

    def test_scales_with_equity(self):
        s1 = compute_base_shares(100.0, 99.0, 10_000.0, 0.35)
        s2 = compute_base_shares(100.0, 99.0, 20_000.0, 0.35)
        assert s2 == pytest.approx(s1 * 2, abs=2)

    def test_scales_with_risk_pct(self):
        s1 = compute_base_shares(100.0, 99.0, 10_000.0, 0.25)
        s2 = compute_base_shares(100.0, 99.0, 10_000.0, 0.50)
        assert s2 == pytest.approx(s1 * 2, abs=2)


class TestVolatilityAdjustment:
    def test_no_adjustment_when_atr_normal(self):
        settings = VolatilitySettings(atr_adjustment_threshold=1.5, atr_size_reduction_factor=0.7)
        shares = apply_volatility_adjustment(100, atr=0.50, avg_atr=0.50, settings=settings)
        assert shares == 100

    def test_reduces_size_on_high_vol(self):
        settings = VolatilitySettings(atr_adjustment_threshold=1.5, atr_size_reduction_factor=0.7)
        shares = apply_volatility_adjustment(100, atr=1.0, avg_atr=0.50, settings=settings)
        assert shares == 70  # 0.7 reduction

    def test_zero_avg_atr_returns_unchanged(self):
        settings = VolatilitySettings()
        shares = apply_volatility_adjustment(100, atr=1.0, avg_atr=0.0, settings=settings)
        assert shares == 100


class TestLiquidityAdjustment:
    def test_caps_by_adv(self):
        settings = LiquiditySettings(max_pct_of_adv=5.0, min_dollar_volume_for_full_size=5_000_000)
        # avg_daily_vol=10_000 → max 5% = 500 shares
        shares = apply_liquidity_adjustment(1000, 100.0, 10_000, settings)
        assert shares <= 500

    def test_no_cap_needed_for_liquid_stock(self):
        settings = LiquiditySettings(max_pct_of_adv=5.0, min_dollar_volume_for_full_size=5_000_000)
        # avg_daily_vol=10_000_000 → max = 500_000 shares (well above input)
        shares = apply_liquidity_adjustment(100, 100.0, 10_000_000, settings)
        assert shares == 100


class TestSpreadAdjustment:
    def test_no_adjustment_for_tight_spread(self):
        shares = apply_spread_adjustment(100, spread_pct=0.05, max_spread_pct=0.10, penalty_factor=0.5)
        assert shares == 100

    def test_reduces_for_wide_spread(self):
        shares = apply_spread_adjustment(100, spread_pct=0.20, max_spread_pct=0.10, penalty_factor=0.5)
        assert shares == 50


class TestCapitalCap:
    def test_caps_at_max_capital_pct(self):
        # 30% of $10k = $3000 max → at $100/share = 30 shares
        shares = apply_capital_cap(100, entry_price=100.0, account_equity=10_000.0, max_capital_pct=30.0)
        assert shares == 30

    def test_no_cap_when_under_limit(self):
        shares = apply_capital_cap(10, entry_price=100.0, account_equity=10_000.0, max_capital_pct=30.0)
        assert shares == 10


class TestFinalPositionSize:
    def test_full_pipeline_returns_positive(self):
        ps = PositionLimitSettings()
        vs = VolatilitySettings()
        ls = LiquiditySettings()
        size = compute_final_position_size(
            entry_price=101.0,
            stop_price=100.20,
            account_equity=10_000.0,
            spread_pct=0.05,
            avg_daily_volume=5_000_000,
            atr=0.50,
            avg_atr=0.50,
            position_settings=ps,
            volatility_settings=vs,
            liquidity_settings=ls,
        )
        assert size > 0

    def test_returns_zero_for_tiny_trade(self):
        ps = PositionLimitSettings(min_trade_dollar_value=500.0)
        vs = VolatilitySettings()
        ls = LiquiditySettings()
        # Very tight stop, very low equity → size too small
        size = compute_final_position_size(
            entry_price=100.0,
            stop_price=99.99,  # 1-cent stop → huge shares but capital capped
            account_equity=100.0,  # tiny equity
            spread_pct=0.05,
            avg_daily_volume=1_000,
            atr=0.50,
            avg_atr=0.50,
            position_settings=ps,
            volatility_settings=vs,
            liquidity_settings=ls,
        )
        assert size == 0
