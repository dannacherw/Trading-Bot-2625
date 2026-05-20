"""
strategy/trade_manager.py
Manages the full lifecycle of an open trade: entry tracking,
stop updates, partial exits, final close, and PnL computation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from loguru import logger

from core.enums import ExitReason, PositionSide, PositionStatus
from core.models import Position, Signal, Trade
from core.config import ExitSettings
from strategy.exits import (
    check_target_1,
    check_target_2,
    compute_breakeven_stop,
    evaluate_all_exits,
    should_move_to_breakeven,
)


class TradeManager:
    """
    Manages open positions and their lifecycle.
    Does not interact with the broker — that's the execution engine's job.
    Emits exit signals / stop updates to be processed by execution.
    """

    def __init__(self, exit_settings: ExitSettings, commission_per_share: float = 0.005) -> None:
        self._exit_settings = exit_settings
        self._commission = commission_per_share
        self._positions: dict[UUID, Position] = {}

    # ------------------------------------------------------------------
    # Position entry
    # ------------------------------------------------------------------

    def open_position(
        self,
        signal: Signal,
        filled_price: float,
        filled_quantity: int,
        filled_at: datetime,
        commission: float = 0.0,
    ) -> Position:
        risk = abs(filled_price - signal.stop_price)
        t1 = filled_price + risk * self._exit_settings.target_1_risk_reward
        t2 = filled_price + risk * self._exit_settings.target_2_risk_reward

        position = Position(
            symbol=signal.symbol,
            side=PositionSide.LONG,
            entry_price=filled_price,
            entry_time=filled_at,
            quantity=filled_quantity,
            remaining_quantity=filled_quantity,
            stop_price=signal.stop_price,
            target_1_price=t1,
            target_2_price=t2,
            commission=commission,
            signal_id=signal.signal_id,
        )
        self._positions[position.position_id] = position
        logger.info(
            "Position opened: {} {} @ {:.4f} | stop={:.4f} t1={:.4f} t2={:.4f}",
            signal.symbol, filled_quantity, filled_price,
            signal.stop_price, t1, t2,
        )
        return position

    # ------------------------------------------------------------------
    # Position updates on each new bar
    # ------------------------------------------------------------------

    def update_position(
        self,
        position_id: UUID,
        current_price: float,
        current_time: datetime,
        atr: float,
    ) -> tuple[Position, list[dict]]:
        """
        Update a position given the current price.
        Returns (updated_position, list_of_actions).

        Actions are dicts with keys: action, reason, price, quantity.
        """
        position = self._positions.get(position_id)
        if position is None:
            raise ValueError(f"Position {position_id} not found")

        actions: list[dict] = []
        p = position  # working reference

        # ---- Move to break-even ----
        if should_move_to_breakeven(p, current_price, self._exit_settings):
            be_stop = compute_breakeven_stop(p)
            p = p.model_copy(update={
                "breakeven_price": be_stop,
                "stop_price": max(p.stop_price, be_stop),
            })
            logger.debug("{}: moved to breakeven @ {:.4f}", p.symbol, be_stop)
            actions.append({"action": "stop_update", "price": be_stop, "reason": "breakeven"})

        # ---- Target 1 partial exit ----
        if (
            p.status == PositionStatus.OPEN
            and check_target_1(p, current_price)
        ):
            partial_qty = max(1, int(p.remaining_quantity * self._exit_settings.target_1_size_pct / 100))
            actions.append({
                "action": "partial_exit",
                "reason": ExitReason.TARGET_1.value,
                "price": p.target_1_price,
                "quantity": partial_qty,
            })
            p = p.model_copy(update={
                "remaining_quantity": p.remaining_quantity - partial_qty,
                "status": PositionStatus.PARTIALLY_CLOSED,
                "realized_pnl": p.realized_pnl + partial_qty * (p.target_1_price - p.entry_price),
            })

        # ---- Target 2 full exit ----
        if check_target_2(p, current_price) and p.remaining_quantity > 0:
            actions.append({
                "action": "full_exit",
                "reason": ExitReason.TARGET_2.value,
                "price": p.target_2_price,
                "quantity": p.remaining_quantity,
            })
            p = self._close_position(p, p.target_2_price, current_time, ExitReason.TARGET_2)

        # ---- Hard stop / trailing / time exits ----
        if p.remaining_quantity > 0:
            exit_reason, new_stop = evaluate_all_exits(
                p, current_price, current_time, atr, self._exit_settings
            )
            if exit_reason:
                actions.append({
                    "action": "full_exit",
                    "reason": exit_reason.value,
                    "price": current_price,
                    "quantity": p.remaining_quantity,
                })
                p = self._close_position(p, current_price, current_time, exit_reason)
            elif new_stop is not None:
                p = p.model_copy(update={"trailing_stop_price": new_stop})
                actions.append({"action": "stop_update", "price": new_stop, "reason": "trail"})

        self._positions[position_id] = p
        return p, actions

    # ------------------------------------------------------------------
    # Position close
    # ------------------------------------------------------------------

    def _close_position(
        self,
        position: Position,
        exit_price: float,
        exit_time: datetime,
        reason: ExitReason,
    ) -> Position:
        return position.model_copy(update={
            "status": PositionStatus.CLOSED,
            "exit_price": exit_price,
            "exit_time": exit_time,
            "exit_reason": reason,
            "remaining_quantity": 0,
        })

    def build_trade_record(self, position: Position) -> Trade:
        """Build a Trade record from a closed Position."""
        assert position.status == PositionStatus.CLOSED
        assert position.exit_price is not None
        assert position.exit_time is not None
        assert position.exit_reason is not None

        gross_pnl = (position.exit_price - position.entry_price) * position.quantity
        commission = position.commission + position.quantity * self._commission
        hold_secs = (position.exit_time - position.entry_time).total_seconds()
        risk_per_share = abs(position.entry_price - position.stop_price)
        r_mult = (
            (position.exit_price - position.entry_price) / risk_per_share
            if risk_per_share > 0 else 0.0
        )

        return Trade(
            position_id=position.position_id,
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=position.exit_price,
            quantity=position.quantity,
            entry_time=position.entry_time,
            exit_time=position.exit_time,
            exit_reason=position.exit_reason,
            gross_pnl=round(gross_pnl, 4),
            commission=round(commission, 4),
            net_pnl=round(gross_pnl - commission, 4),
            r_multiple=round(r_mult, 3),
            hold_duration_seconds=hold_secs,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_open_positions(self) -> list[Position]:
        return [
            p for p in self._positions.values()
            if p.status in (PositionStatus.OPEN, PositionStatus.PARTIALLY_CLOSED)
        ]

    def get_position(self, position_id: UUID) -> Position | None:
        return self._positions.get(position_id)

    @property
    def open_position_count(self) -> int:
        return len(self.get_open_positions())
