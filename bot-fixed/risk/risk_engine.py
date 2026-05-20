"""
risk/risk_engine.py
Central risk engine. Validates all proposed trades against:
  - Daily loss limits
  - Max open positions
  - Max trades per day
  - Capital limits
  - Minimum size
Tracks daily PnL and emits halt signal if limits are breached.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from loguru import logger

from core.config import RiskConfig
from core.enums import RiskCheckResult, RiskRejectionReason
from core.exceptions import DailyLossLimitBreached, PositionLimitError
from core.models import DailyRiskState, Position, RiskValidation, Signal
from risk.portfolio_constraints import (
    check_sector_concentration,
    check_correlated_positions,
    check_duplicate_position,
)
from risk.position_sizing import compute_final_position_size


class RiskEngine:
    """
    Stateful risk manager. Must be reset each trading day.
    Thread-safe: all state mutations are sequential within the async event loop.
    """

    def __init__(self, config: RiskConfig) -> None:
        self._config = config
        self._equity = config.account.starting_equity
        self._daily_loss = 0.0
        self._trades_today = 0
        self._open_positions: dict[str, Position] = {}  # symbol -> position
        self._is_halted = False
        self._date_str: str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        self._on_halt: Callable[[], None] | None = None
        # Sector map for concentration checks — populated by strategy layer
        self._sector_map: dict[str, str] = {}
        # Known correlated pairs for correlation checks
        self._correlated_pairs: set[frozenset[str]] = set()

    # ------------------------------------------------------------------
    # Trade validation
    # ------------------------------------------------------------------

    def validate_signal(
        self,
        signal: Signal,
        spread_pct: float,
        avg_daily_volume: float,
        atr: float,
        avg_atr: float,
        current_equity: float | None = None,
    ) -> RiskValidation:
        """
        Full risk validation for a proposed trade signal.
        Returns RiskValidation with approved quantity.
        """
        equity = current_equity if current_equity is not None else self._equity

        # ---- Daily loss halt ----
        if self._is_halted:
            return RiskValidation(
                result=RiskCheckResult.REJECTED,
                rejection_reason=RiskRejectionReason.DAILY_LOSS_LIMIT,
                original_quantity=0,
                approved_quantity=0,
                message="Trading halted: daily loss limit breached",
            )

        # ---- Daily loss remaining check ----
        daily_limit = equity * (self._config.daily_limits.max_daily_loss_pct / 100.0)
        if self._daily_loss >= daily_limit:
            self._trigger_halt(f"Daily loss limit reached: ${self._daily_loss:.2f}")
            return RiskValidation(
                result=RiskCheckResult.REJECTED,
                rejection_reason=RiskRejectionReason.DAILY_LOSS_LIMIT,
                original_quantity=0,
                approved_quantity=0,
                message=f"Daily loss limit ${daily_limit:.2f} reached",
            )

        # ---- Max open positions ----
        if len(self._open_positions) >= self._config.position_limits.max_open_positions:
            return RiskValidation(
                result=RiskCheckResult.REJECTED,
                rejection_reason=RiskRejectionReason.MAX_POSITIONS,
                original_quantity=0,
                approved_quantity=0,
                message=(
                    f"Max open positions "
                    f"({self._config.position_limits.max_open_positions}) reached"
                ),
            )

        # ---- Max trades per day ----
        if self._trades_today >= self._config.daily_limits.max_trades_per_day:
            return RiskValidation(
                result=RiskCheckResult.REJECTED,
                rejection_reason=RiskRejectionReason.MAX_TRADES,
                original_quantity=0,
                approved_quantity=0,
                message=f"Max trades per day ({self._config.daily_limits.max_trades_per_day}) reached",
            )

        # ---- Portfolio concentration checks ----
        open_list = list(self._open_positions.values())

        dup_ok, dup_reason = check_duplicate_position(signal.symbol, open_list)
        if not dup_ok:
            return RiskValidation(
                result=RiskCheckResult.REJECTED,
                rejection_reason=RiskRejectionReason.MAX_POSITIONS,
                original_quantity=0, approved_quantity=0,
                message=dup_reason,
            )

        if self._sector_map:
            sector_ok, sector_reason = check_sector_concentration(
                proposed_symbol=signal.symbol,
                open_positions=open_list,
                sector_map=self._sector_map,
                max_sector_pct=self._config.position_limits.max_capital_per_position_pct * 2,
                total_equity=equity,
            )
            if not sector_ok:
                return RiskValidation(
                    result=RiskCheckResult.REJECTED,
                    rejection_reason=RiskRejectionReason.MAX_POSITIONS,
                    original_quantity=0, approved_quantity=0,
                    message=f"Sector concentration: {sector_reason}",
                )

        if self._correlated_pairs:
            corr_ok, corr_reason = check_correlated_positions(
                proposed_symbol=signal.symbol,
                open_positions=open_list,
                correlated_pairs=self._correlated_pairs,
                max_correlated=self._config.position_limits.max_open_positions - 1,
            )
            if not corr_ok:
                return RiskValidation(
                    result=RiskCheckResult.REJECTED,
                    rejection_reason=RiskRejectionReason.MAX_POSITIONS,
                    original_quantity=0, approved_quantity=0,
                    message=f"Correlation limit: {corr_reason}",
                )

        # ---- Spread check — configurable, aligned with scanner filter ----
        max_spread = self._config.max_entry_spread_pct
        if spread_pct > max_spread:
            return RiskValidation(
                result=RiskCheckResult.REJECTED,
                rejection_reason=RiskRejectionReason.SPREAD_TOO_WIDE,
                original_quantity=0,
                approved_quantity=0,
                message=f"Spread {spread_pct:.3f}% > max {max_spread:.3f}%",
            )

        # ---- Position sizing ----
        shares = compute_final_position_size(
            entry_price=signal.entry_price,
            stop_price=signal.stop_price,
            account_equity=equity,
            spread_pct=spread_pct,
            avg_daily_volume=avg_daily_volume,
            atr=atr,
            avg_atr=avg_atr,
            position_settings=self._config.position_limits,
            volatility_settings=self._config.volatility,
            liquidity_settings=self._config.liquidity,
        )

        if shares <= 0:
            return RiskValidation(
                result=RiskCheckResult.REJECTED,
                rejection_reason=RiskRejectionReason.SIZE_TOO_SMALL,
                original_quantity=0,
                approved_quantity=0,
                message="Position size too small after adjustments",
            )

        dollar_risk = abs(signal.entry_price - signal.stop_price) * shares
        logger.info(
            "Risk approved: {} {} shares | ${:.2f} risk | equity={:.2f}",
            signal.symbol, shares, dollar_risk, equity,
        )

        return RiskValidation(
            result=RiskCheckResult.APPROVED,
            original_quantity=signal.suggested_quantity,
            approved_quantity=shares,
            message=f"Approved: {shares} shares, ${dollar_risk:.2f} risk",
        )

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def register_trade_opened(self, symbol: str, position: Position) -> None:
        self._open_positions[symbol] = position

    def register_trade_closed(
        self, symbol: str, pnl: float
    ) -> None:
        self._open_positions.pop(symbol, None)
        self._trades_today += 1
        if pnl < 0:
            self._daily_loss += abs(pnl)
            logger.debug(
                "Daily loss updated: +${:.2f} (total: ${:.2f})", abs(pnl), self._daily_loss
            )
            # Check if halt should be triggered
            limit = self._equity * (self._config.daily_limits.max_daily_loss_pct / 100.0)
            if self._daily_loss >= limit:
                self._trigger_halt(f"Post-close daily loss limit reached: ${self._daily_loss:.2f}")

    def update_equity(self, equity: float) -> None:
        self._equity = equity

    # ------------------------------------------------------------------
    # Halt
    # ------------------------------------------------------------------

    def _trigger_halt(self, reason: str) -> None:
        if not self._is_halted:
            self._is_halted = True
            logger.warning("⛔ TRADING HALTED: {}", reason)
            if self._on_halt:
                self._on_halt()

    def set_halt_callback(self, callback: Callable[[], None]) -> None:
        self._on_halt = callback

    def update_sector_map(self, sector_map: dict[str, str]) -> None:
        """
        Update the sector map used for concentration checks.
        Called by VWAPStrategy when sectors are resolved for watchlist symbols.
        """
        self._sector_map.update(sector_map)

    def register_correlated_pair(self, symbol_a: str, symbol_b: str) -> None:
        """Register two symbols as correlated for portfolio constraint checking."""
        self._correlated_pairs.add(frozenset([symbol_a, symbol_b]))

    def clear_halt(self) -> None:
        """Manual halt override (requires explicit user action)."""
        self._is_halted = False
        logger.warning("Trading halt manually cleared")

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def reset_daily(self, new_equity: float | None = None) -> None:
        self._daily_loss = 0.0
        self._trades_today = 0
        self._open_positions.clear()
        self._is_halted = False
        if new_equity is not None:
            self._equity = new_equity
        self._date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        logger.info("Risk engine reset for new day — equity=${:.2f}", self._equity)

    # ------------------------------------------------------------------
    # State snapshot
    # ------------------------------------------------------------------

    def get_state(self) -> DailyRiskState:
        daily_limit = self._equity * (self._config.daily_limits.max_daily_loss_pct / 100.0)
        return DailyRiskState(
            date=self._date_str,
            starting_equity=self._config.account.starting_equity,
            current_equity=self._equity,
            realized_pnl_today=-self._daily_loss,
            unrealized_pnl=0.0,
            trades_today=self._trades_today,
            open_positions=len(self._open_positions),
            daily_loss_limit=daily_limit,
            daily_loss_used=self._daily_loss,
            is_halted=self._is_halted,
        )

    @property
    def is_halted(self) -> bool:
        return self._is_halted

    @property
    def trades_today(self) -> int:
        return self._trades_today
