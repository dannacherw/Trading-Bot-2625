"""
tests/test_execution/test_execution_engine.py
Tests for the execution engine and order router using broker mocks.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from core.enums import OrderStatus, RiskCheckResult
from core.models import Order, RiskValidation
from execution.order_router import OrderRouter
from execution.slippage_model import apply_slippage, estimate_slippage
from core.enums import OrderSide
from core.config import SlippageSettings, OrderSettings


# ---------------------------------------------------------------------------
# Slippage model tests
# ---------------------------------------------------------------------------

class TestSlippageModel:
    @pytest.fixture
    def slip_settings(self):
        return SlippageSettings(
            base_slippage_pct=0.05,
            high_vol_slippage_multiplier=2.0,
            spread_slippage_factor=0.5,
        )

    def test_buy_fills_higher(self, slip_settings):
        price = 100.0
        filled = apply_slippage(price, 0.10, OrderSide.BUY, slip_settings)
        assert filled > price

    def test_sell_fills_lower(self, slip_settings):
        price = 100.0
        filled = apply_slippage(price, 0.10, OrderSide.SELL, slip_settings)
        assert filled < price

    def test_high_vol_increases_slippage(self, slip_settings):
        normal = estimate_slippage(100.0, 0.05, OrderSide.BUY, slip_settings, is_high_vol=False)
        high_vol = estimate_slippage(100.0, 0.05, OrderSide.BUY, slip_settings, is_high_vol=True)
        assert high_vol == pytest.approx(normal * 2.0, rel=0.01)

    def test_wider_spread_increases_slippage(self, slip_settings):
        tight = estimate_slippage(100.0, 0.05, OrderSide.BUY, slip_settings)
        wide = estimate_slippage(100.0, 0.50, OrderSide.BUY, slip_settings)
        assert wide > tight


# ---------------------------------------------------------------------------
# Order router tests
# ---------------------------------------------------------------------------

class TestOrderRouter:
    @pytest.fixture
    def order_settings(self):
        return OrderSettings(
            default_order_type="LIMIT",
            limit_slippage_buffer_pct=0.05,
            max_order_value=3000.0,
            min_order_value=500.0,
            time_in_force="DAY",
        )

    @pytest.fixture
    def router(self, mock_broker, order_settings):
        return OrderRouter(mock_broker, order_settings)

    async def test_submit_entry_creates_order(self, router, sample_signal, mock_broker):
        mock_broker.is_paper = True
        order = await router.submit_entry(sample_signal, quantity=50)
        assert order.symbol == sample_signal.symbol
        assert order.quantity == 50
        assert order.status == OrderStatus.SUBMITTED

    async def test_paper_fill_resolves_immediately(self, router, sample_signal, mock_broker):
        mock_broker.is_paper = True
        order = await router.submit_entry(sample_signal, quantity=50)
        filled = await router.await_fill(order.order_id, timeout=5.0)
        assert filled.status == OrderStatus.FILLED
        assert filled.filled_quantity == 50

    async def test_submit_exit(self, router, mock_broker):
        from core.enums import OrderType
        mock_broker.is_paper = True
        order = await router.submit_exit("AAPL", quantity=50, price=102.0)
        assert order.symbol == "AAPL"

    async def test_cancel_all(self, router, sample_signal, mock_broker):
        mock_broker.is_paper = True
        await router.submit_entry(sample_signal, quantity=50)
        cancelled = await router.cancel_all()
        assert isinstance(cancelled, int)


# ---------------------------------------------------------------------------
# Fill simulator tests
# ---------------------------------------------------------------------------

class TestFillSimulator:
    def test_market_order_always_fills(self, sample_bar):
        from core.enums import OrderSide, OrderType
        from core.models import Order
        from execution.fill_simulator import simulate_fill
        from core.config import FillSimulationSettings

        settings = FillSimulationSettings(fill_probability=1.0, partial_fill_probability=0.0)
        order = Order(
            symbol="AAPL", side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=100,
        )
        filled = simulate_fill(order, sample_bar, settings)
        assert filled.status == OrderStatus.FILLED
        assert filled.filled_quantity == 100

    def test_limit_buy_fills_when_bar_hits(self, sample_bar):
        from core.enums import OrderSide, OrderType
        from core.models import Order
        from execution.fill_simulator import simulate_fill
        from core.config import FillSimulationSettings

        settings = FillSimulationSettings(fill_probability=1.0, partial_fill_probability=0.0)
        # bar.low = 99.0, limit at 99.50 → should fill
        order = Order(
            symbol="AAPL", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, quantity=50,
            limit_price=99.50,
        )
        filled = simulate_fill(order, sample_bar, settings)
        assert filled.status == OrderStatus.FILLED

    def test_limit_buy_no_fill_when_bar_too_high(self, sample_bar):
        from core.enums import OrderSide, OrderType
        from core.models import Order
        from execution.fill_simulator import simulate_fill
        from core.config import FillSimulationSettings

        settings = FillSimulationSettings(fill_probability=1.0, partial_fill_probability=0.0)
        # bar.low = 99.0, limit at 98.0 → should NOT fill (bar never got that low)
        order = Order(
            symbol="AAPL", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, quantity=50,
            limit_price=98.0,
        )
        filled = simulate_fill(order, sample_bar, settings)
        assert filled.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED)
        assert filled.filled_quantity == 0
