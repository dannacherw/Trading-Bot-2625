"""
execution/base_broker.py
Abstract broker interface. All broker implementations must subclass this.
Keeps execution engine and strategy code completely broker-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from core.models import Order, Position


class BaseBroker(ABC):
    """
    Contract that every broker adapter must implement.
    Methods are async to support REST and WebSocket-based brokers.
    """

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """Authenticate and open connections."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connections and clean up."""

    @abstractmethod
    async def is_connected(self) -> bool:
        """Return True if the broker connection is active and authenticated."""

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_account_equity(self) -> float:
        """Return current account equity (liquidation value)."""

    @abstractmethod
    async def get_buying_power(self) -> float:
        """Return current available buying power."""

    @abstractmethod
    async def get_positions(self) -> list[dict[str, Any]]:
        """Return all open positions from the broker."""

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    @abstractmethod
    async def submit_order(self, order: Order) -> str:
        """
        Submit an order to the broker.
        Returns the broker-assigned order ID.
        """

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order. Returns True if successfully cancelled."""

    @abstractmethod
    async def get_order_status(self, broker_order_id: str) -> dict[str, Any]:
        """Return raw order status from broker."""

    @abstractmethod
    async def get_open_orders(self) -> list[dict[str, Any]]:
        """Return all open (unfilled) orders."""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable broker name."""

    @property
    @abstractmethod
    def is_paper(self) -> bool:
        """True if this is a paper trading / simulated broker."""

    @property
    def supports_fractional_shares(self) -> bool:
        return False
