"""
execution/fill_simulator.py
Simulates order fills for backtesting and paper trading.
Uses bar data to determine if a limit order would have filled.
"""
from __future__ import annotations

import random
from datetime import datetime

from core.config import FillSimulationSettings
from core.enums import OrderSide, OrderStatus, OrderType
from core.models import Bar, Order


def simulate_fill(
    order: Order,
    bar: Bar,
    settings: FillSimulationSettings,
    slippage_pct: float = 0.05,
) -> Order:
    """
    Simulate an order fill against a bar.
    Returns the order with updated fill status.
    """
    if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
        return order

    fill_price: float | None = None
    fill_qty = order.quantity

    if order.order_type == OrderType.MARKET:
        # Market orders always fill at open + slippage
        slip = bar.open * slippage_pct / 100.0
        if order.side in (OrderSide.BUY, OrderSide.BUY_TO_COVER):
            fill_price = bar.open + slip
        else:
            fill_price = bar.open - slip

    elif order.order_type == OrderType.LIMIT:
        # Limit order fills if the bar crosses the limit price
        if order.limit_price is None:
            return order
        if order.side in (OrderSide.BUY, OrderSide.BUY_TO_COVER):
            if bar.low <= order.limit_price:
                # Random fill probability
                if random.random() < settings.fill_probability:
                    fill_price = order.limit_price
        else:
            if bar.high >= order.limit_price:
                if random.random() < settings.fill_probability:
                    fill_price = order.limit_price

    elif order.order_type == OrderType.STOP:
        if order.stop_price is None:
            return order
        if order.side in (OrderSide.BUY, OrderSide.BUY_TO_COVER):
            if bar.high >= order.stop_price:
                fill_price = max(bar.open, order.stop_price)
        else:
            if bar.low <= order.stop_price:
                fill_price = min(bar.open, order.stop_price)

    if fill_price is None:
        return order

    # Partial fill simulation
    if random.random() < settings.partial_fill_probability:
        fill_qty = max(1, int(fill_qty * random.uniform(0.5, 0.95)))
        new_status = OrderStatus.PARTIAL if fill_qty < order.quantity else OrderStatus.FILLED
    else:
        new_status = OrderStatus.FILLED

    return order.model_copy(update={
        "status": new_status,
        "filled_quantity": fill_qty,
        "avg_fill_price": fill_price,
        "filled_at": bar.timestamp,
    })
