"""
execution/order_router.py
Routes orders to the broker and tracks their lifecycle.
Handles retries, fill polling, and status updates.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import UUID

from loguru import logger

from core.config import OrderSettings
from core.enums import OrderSide, OrderStatus, OrderType, TimeInForce
from core.exceptions import OrderRejectedError, OrderTimeoutError
from core.models import Order, Signal
from execution.base_broker import BaseBroker


class OrderRouter:
    """
    Submits orders to the broker and polls for fill confirmations.
    Maintains a local registry of pending/open orders.
    """

    def __init__(
        self,
        broker: BaseBroker,
        settings: OrderSettings,
        poll_interval: float = 1.0,
        fill_timeout: float = 30.0,
    ) -> None:
        self._broker = broker
        self._settings = settings
        self._poll_interval = poll_interval
        self._fill_timeout = fill_timeout
        self._pending: dict[UUID, Order] = {}

    async def submit_entry(
        self,
        signal: Signal,
        quantity: int,
        spread_pct: float = 0.05,
    ) -> Order:
        """Build and submit a limit entry order from a signal."""
        # Limit price = entry + small buffer (to improve fill probability)
        buffer = signal.entry_price * (self._settings.limit_slippage_buffer_pct / 100.0)
        limit_price = round(signal.entry_price + buffer, 2)

        order = Order(
            symbol=signal.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            limit_price=limit_price,
            time_in_force=TimeInForce.DAY,
            submitted_at=datetime.now(tz=timezone.utc),
            notes=f"Entry signal {signal.signal_id}",
        )
        return await self._submit(order)

    async def submit_exit(
        self,
        symbol: str,
        quantity: int,
        price: float,
        order_type: OrderType = OrderType.LIMIT,
        stop_price: float | None = None,
    ) -> Order:
        """Submit an exit (sell) order."""
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=order_type,
            quantity=quantity,
            limit_price=price if order_type == OrderType.LIMIT else None,
            stop_price=stop_price,
            time_in_force=TimeInForce.DAY,
            submitted_at=datetime.now(tz=timezone.utc),
        )
        return await self._submit(order)

    async def submit_stop(self, symbol: str, quantity: int, stop_price: float) -> Order:
        """Submit a protective stop order."""
        return await self.submit_exit(
            symbol, quantity, price=stop_price,
            order_type=OrderType.STOP, stop_price=stop_price,
        )

    async def _submit(self, order: Order) -> Order:
        try:
            broker_id = await self._broker.submit_order(order)
            order = order.model_copy(update={
                "broker_order_id": broker_id,
                "status": OrderStatus.SUBMITTED,
            })
            self._pending[order.order_id] = order
            logger.info(
                "Order submitted: {} {} {} | broker_id={}",
                order.symbol, order.side.value, order.quantity, broker_id,
            )
            return order
        except Exception as exc:
            raise OrderRejectedError(f"Order submission failed: {exc}") from exc

    async def await_fill(
        self, order_id: UUID, timeout: float | None = None
    ) -> Order:
        """
        Poll broker until the order is filled or times out.
        Returns the filled order.
        """
        deadline = asyncio.get_event_loop().time() + (timeout or self._fill_timeout)
        order = self._pending.get(order_id)
        if order is None:
            raise ValueError(f"Order {order_id} not found in pending orders")

        if self._broker.is_paper:
            # Paper orders fill immediately (simulation)
            filled_order = order.model_copy(update={
                "status": OrderStatus.FILLED,
                "filled_quantity": order.quantity,
                "avg_fill_price": order.limit_price or 0.0,
                "filled_at": datetime.now(tz=timezone.utc),
            })
            self._pending.pop(order_id, None)
            return filled_order

        while asyncio.get_event_loop().time() < deadline:
            try:
                status_data = await self._broker.get_order_status(
                    order.broker_order_id or str(order_id)
                )
                order = self._update_from_broker_status(order, status_data)
                if order.status in (OrderStatus.FILLED, OrderStatus.PARTIAL):
                    self._pending.pop(order_id, None)
                    return order
                if order.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                    raise OrderRejectedError(
                        f"Order {order_id} {order.status.value}", str(order_id)
                    )
            except (OrderRejectedError, OrderTimeoutError):
                raise
            except Exception as exc:
                logger.warning("Poll error for {}: {}", order_id, exc)
            await asyncio.sleep(self._poll_interval)

        await self._broker.cancel_order(order.broker_order_id or str(order_id))
        raise OrderTimeoutError(
            f"Order {order_id} did not fill within {timeout or self._fill_timeout}s"
        )

    async def cancel_all(self) -> int:
        """Cancel all pending orders. Returns count of successful cancellations."""
        cancelled = 0
        for order in list(self._pending.values()):
            if order.broker_order_id:
                ok = await self._broker.cancel_order(order.broker_order_id)
                if ok:
                    cancelled += 1
        self._pending.clear()
        return cancelled

    @staticmethod
    def _update_from_broker_status(order: Order, data: dict) -> Order:
        """Map broker status response to local Order model."""
        status_map = {
            "WORKING": OrderStatus.SUBMITTED,
            "PENDING_ACTIVATION": OrderStatus.PENDING,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }
        raw_status = data.get("status", "WORKING")
        status = status_map.get(raw_status, OrderStatus.SUBMITTED)

        fill_qty = int(data.get("filledQuantity", 0))
        avg_price = float(data.get("price", order.limit_price or 0.0))

        return order.model_copy(update={
            "status": status,
            "filled_quantity": fill_qty,
            "avg_fill_price": avg_price if fill_qty > 0 else None,
        })
