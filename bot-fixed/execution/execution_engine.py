"""
execution/execution_engine.py
Main execution orchestrator. Receives signals from the strategy,
validates through risk, sizes positions, and routes to the broker.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from loguru import logger

from core.enums import RiskCheckResult
from core.exceptions import OrderRejectedError, OrderTimeoutError
from core.models import Order, Position, Signal, Trade
from database.repository import OrderRepository, PositionRepository, SignalRepository, TradeRepository
from execution.base_broker import BaseBroker
from execution.order_router import OrderRouter
from market_data.base_provider import BaseDataProvider
from risk.kill_switch import KillSwitchMonitor
from market_data.market_cache import MarketCache
from risk.risk_engine import RiskEngine
from strategy.trade_manager import TradeManager


class ExecutionEngine:
    """
    Ties together:
      - Risk validation (RiskEngine)
      - Order routing (OrderRouter)
      - Position tracking (TradeManager)
      - Persistence (repositories)
    """

    def __init__(
        self,
        broker: BaseBroker,
        risk_engine: RiskEngine,
        trade_manager: TradeManager,
        data_provider: BaseDataProvider,
        order_router: OrderRouter,
        signal_repo: SignalRepository,
        order_repo: OrderRepository,
        position_repo: PositionRepository,
        trade_repo: TradeRepository,
        cache: MarketCache | None = None,
        on_trade_closed: Callable[[Trade], None] | None = None,
        kill_switch: KillSwitchMonitor | None = None,
    ) -> None:
        self._broker = broker
        self._risk = risk_engine
        self._trade_mgr = trade_manager
        self._provider = data_provider
        self._router = order_router
        self._signal_repo = signal_repo
        self._order_repo = order_repo
        self._position_repo = position_repo
        self._trade_repo = trade_repo
        self._cache = cache or MarketCache()
        self._on_trade_closed = on_trade_closed
        self._kill_switch = kill_switch

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    async def handle_signal(self, signal: Signal) -> None:
        """
        Process an inbound trade signal end-to-end.
        Validates risk → sizes → submits → confirms fill → tracks.
        """
        logger.info(
            "ExecutionEngine received signal: {} {} strength={}",
            signal.symbol, signal.signal_type.value, signal.strength.value,
        )

        # Kill switch gate — blocks new entries when halted
        if self._kill_switch is not None:
            allowed, reason = self._kill_switch.check_entry_allowed()
            if not allowed:
                logger.warning(
                    "Signal blocked by kill switch for {}: {}", signal.symbol, reason
                )
                return

        # ---- Fetch market context for risk sizing ----
        try:
            quote = await self._provider.get_current_quote(signal.symbol)
            atr = await self._get_atr(signal.symbol)
            avg_atr = atr  # Simplified — use same ATR as avg for now
            avg_daily_vol = await self._provider.get_avg_daily_dollar_volume(signal.symbol)
            avg_daily_vol = avg_daily_vol / max(quote.mid, 0.01)
        except Exception as exc:
            logger.error("Failed to fetch market context for {}: {}", signal.symbol, exc)
            return

        # ---- Risk validation ----
        equity = await self._broker.get_account_equity()
        validation = self._risk.validate_signal(
            signal=signal,
            spread_pct=quote.spread_pct,
            avg_daily_volume=avg_daily_vol,
            atr=atr,
            avg_atr=avg_atr,
            current_equity=equity,
        )

        if validation.result == RiskCheckResult.REJECTED:
            logger.warning(
                "Signal rejected for {}: {} — {}",
                signal.symbol, validation.rejection_reason, validation.message,
            )
            await self._signal_repo.save(signal)
            return

        quantity = validation.approved_quantity
        logger.info(
            "Risk approved {}: {} shares | {}",
            signal.symbol, quantity, validation.message,
        )

        # ---- Update signal with approved quantity ----
        signal = signal.model_copy(update={"suggested_quantity": quantity})
        await self._signal_repo.save(signal)

        # ---- Submit entry order ----
        try:
            order = await self._router.submit_entry(signal, quantity, quote.spread_pct)
            await self._order_repo.save(order)
        except OrderRejectedError as exc:
            logger.error("Entry order rejected for {}: {}", signal.symbol, exc)
            return

        # ---- Await fill ----
        try:
            filled_order = await self._router.await_fill(order.order_id)
            await self._order_repo.save(filled_order)
        except (OrderRejectedError, OrderTimeoutError) as exc:
            logger.warning("Entry fill failed for {}: {}", signal.symbol, exc)
            return

        if filled_order.avg_fill_price is None:
            logger.error("Fill price missing for {}", signal.symbol)
            return

        # ---- Open position ----
        commission = filled_order.filled_quantity * 0.005  # $0.005/share
        position = self._trade_mgr.open_position(
            signal=signal,
            filled_price=filled_order.avg_fill_price,
            filled_quantity=filled_order.filled_quantity,
            filled_at=filled_order.filled_at or datetime.now(tz=timezone.utc),
            commission=commission,
        )
        self._risk.register_trade_opened(signal.symbol, position)
        await self._position_repo.save(position)
        await self._signal_repo.mark_acted_on(signal.signal_id)

        logger.info(
            "✅ Position opened: {} {} @ {:.4f}",
            signal.symbol, filled_order.filled_quantity, filled_order.avg_fill_price,
        )

    # ------------------------------------------------------------------
    # Exit handling
    # ------------------------------------------------------------------

    async def execute_exit(
        self,
        position: Position,
        exit_price: float,
        exit_quantity: int,
        exit_reason: str,
    ) -> None:
        """Execute an exit for an open position."""
        from core.enums import OrderType
        try:
            order = await self._router.submit_exit(
                symbol=position.symbol,
                quantity=exit_quantity,
                price=exit_price,
                order_type=OrderType.LIMIT,
            )
            await self._order_repo.save(order)
            filled = await self._router.await_fill(order.order_id, timeout=15.0)
            await self._order_repo.save(filled)

            if filled.avg_fill_price:
                await self._handle_position_closed(
                    position, filled.avg_fill_price, exit_quantity
                )
        except Exception as exc:
            logger.error("Exit execution failed for {}: {}", position.symbol, exc)

    async def _handle_position_closed(
        self,
        position: Position,
        fill_price: float,
        fill_qty: int,
    ) -> None:
        now = datetime.now(tz=timezone.utc)
        gross_pnl = (fill_price - position.entry_price) * fill_qty

        self._risk.register_trade_closed(position.symbol, gross_pnl)

        if fill_qty >= position.remaining_quantity:
            trade = self._trade_mgr.build_trade_record(
                position.model_copy(update={
                    "exit_price": fill_price,
                    "exit_time": now,
                    "remaining_quantity": 0,
                })
            )
            await self._trade_repo.save(trade)
            await self._position_repo.save(
                position.model_copy(update={"exit_price": fill_price, "exit_time": now})
            )
            # Notify kill switch of trade result (consecutive loss tracking)
            if self._kill_switch is not None:
                self._kill_switch.record_trade_result(trade.net_pnl)
            if self._on_trade_closed:
                self._on_trade_closed(trade)
            logger.info(
                "✅ Trade closed: {} | PnL=${:.2f} | R={:.2f}",
                position.symbol, trade.net_pnl, trade.r_multiple,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _get_atr(self, symbol: str) -> float:
        cached = await self._cache.get_atr(symbol)
        if cached is not None:
            return cached
        try:
            atr = await self._provider.get_atr(symbol)
            await self._cache.set_atr(symbol, atr)
            return atr
        except Exception:
            return 0.0
