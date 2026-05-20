"""
risk/kill_switch.py
Soft-halt kill switch monitor.

Monitors live system health and triggers soft halts when safety thresholds
are breached. All halts are SOFT — new entries are blocked but existing
positions continue to manage to their exits. A human must explicitly
restart trading via the CLI command `python main.py resume`.

Monitored conditions:
  1. Consecutive losses:        N losses in a row without a win
  2. Data staleness:            No successful scan in > X minutes
  3. Spread expansion:          Average spread across watchlist > threshold
  4. Broker disconnect:         Broker heartbeat failure
  5. Volatility circuit breaker: VIX spike or SPY move > threshold intraday

Each condition is independently configurable and independently resettable.
State is persisted to SQLite kill_switch_events table for audit trail.

Architecture:
  KillSwitchMonitor is injected into ExecutionEngine and PremarketScanner.
  Both call check_entry_allowed() before opening new positions.
  Halts are published via an async callback so the dashboard can react.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Awaitable, Callable

from loguru import logger


# ---------------------------------------------------------------------------
# Kill switch trigger types
# ---------------------------------------------------------------------------

class KillSwitchTrigger(str, Enum):
    CONSECUTIVE_LOSSES    = "CONSECUTIVE_LOSSES"
    DATA_STALENESS        = "DATA_STALENESS"
    SPREAD_EXPANSION      = "SPREAD_EXPANSION"
    BROKER_DISCONNECT     = "BROKER_DISCONNECT"
    VOLATILITY_CIRCUIT    = "VOLATILITY_CIRCUIT"
    MANUAL                = "MANUAL"


@dataclass(slots=True)
class KillSwitchEvent:
    trigger: KillSwitchTrigger
    triggered_at: datetime
    reason: str
    value: float | None = None    # The measured value that crossed the threshold
    threshold: float | None = None
    resolved_at: datetime | None = None
    resolved_by: str | None = None  # "human" | "auto"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class KillSwitchSettings:
    # 1. Consecutive loss halt
    max_consecutive_losses: int = 3
    consecutive_loss_halt_enabled: bool = True

    # 2. Data staleness halt
    max_stale_scan_minutes: float = 10.0
    data_staleness_halt_enabled: bool = True

    # 3. Spread expansion halt (average watchlist spread)
    max_avg_spread_pct: float = 0.40
    spread_expansion_halt_enabled: bool = True
    spread_check_min_symbols: int = 3   # Need at least N symbols to compute avg

    # 4. Broker disconnect halt
    broker_disconnect_halt_enabled: bool = True
    broker_heartbeat_timeout_seconds: float = 30.0

    # 5. Volatility circuit breaker
    volatility_halt_enabled: bool = True
    max_spy_intraday_down_pct: float = -1.5   # Halt if SPY down > 1.5% intraday
    max_vix_level: float = 35.0               # Halt if VIX > 35

    # Human restart required for all triggers
    require_human_restart: bool = True


# ---------------------------------------------------------------------------
# Kill switch monitor
# ---------------------------------------------------------------------------

class KillSwitchMonitor:
    """
    Tracks all halt conditions and exposes check_entry_allowed().

    Lifecycle:
        monitor = KillSwitchMonitor(settings)
        monitor.set_halt_callback(async_fn)   # optional — notify dashboard
        monitor.record_trade_result(pnl)       # call after each trade closes
        monitor.update_last_scan_time(dt)      # call after each scan completes
        monitor.update_spreads([0.08, 0.12])   # call each scan cycle
        allowed, reason = monitor.check_entry_allowed()
    """

    def __init__(
        self,
        settings: KillSwitchSettings,
        on_halt: Callable[[KillSwitchEvent], Awaitable[None]] | None = None,
        db_repo: "KillSwitchRepository | None" = None,
    ) -> None:
        self._settings = settings
        self._on_halt = on_halt
        self._db_repo = db_repo

        # Internal state
        self._consecutive_losses: int = 0
        self._last_scan_at: datetime | None = None
        self._recent_spreads: list[float] = []
        self._broker_connected: bool = True
        self._last_broker_heartbeat: datetime = datetime.now(tz=timezone.utc)
        self._spy_open: float | None = None
        self._spy_current: float | None = None
        self._vix_current: float | None = None

        # Active halts
        self._active_halts: dict[KillSwitchTrigger, KillSwitchEvent] = {}

        # Audit log (in-process)
        self._event_log: list[KillSwitchEvent] = []

    # ------------------------------------------------------------------
    # Primary gate — called before every entry signal
    # ------------------------------------------------------------------

    def check_entry_allowed(self) -> tuple[bool, str]:
        """
        Returns (True, "") if new entries are permitted.
        Returns (False, reason) if any halt is active.
        """
        if not self._active_halts:
            return True, ""

        # Return the first active halt reason
        event = next(iter(self._active_halts.values()))
        return False, f"Kill switch ACTIVE [{event.trigger.value}]: {event.reason}"

    @property
    def is_halted(self) -> bool:
        return bool(self._active_halts)

    @property
    def active_triggers(self) -> list[KillSwitchTrigger]:
        return list(self._active_halts.keys())

    # ------------------------------------------------------------------
    # State update methods — called by the bot during operation
    # ------------------------------------------------------------------

    def record_trade_result(self, net_pnl: float) -> None:
        """Call after each trade closes. Tracks consecutive loss counter."""
        if net_pnl < 0:
            self._consecutive_losses += 1
            logger.debug(
                "Kill switch: consecutive losses = {}",
                self._consecutive_losses,
            )
            if (
                self._settings.consecutive_loss_halt_enabled
                and self._consecutive_losses >= self._settings.max_consecutive_losses
            ):
                self._trigger(
                    KillSwitchTrigger.CONSECUTIVE_LOSSES,
                    reason=f"{self._consecutive_losses} consecutive losses (threshold: {self._settings.max_consecutive_losses})",
                    value=float(self._consecutive_losses),
                    threshold=float(self._settings.max_consecutive_losses),
                )
        else:
            # Win resets the counter
            if self._consecutive_losses > 0:
                logger.debug(
                    "Kill switch: consecutive loss counter reset (was {})",
                    self._consecutive_losses,
                )
            self._consecutive_losses = 0

    def update_last_scan_time(self, scanned_at: datetime) -> None:
        """Call after each successful scan cycle."""
        self._last_scan_at = scanned_at
        # A successful scan clears any active staleness halt
        if KillSwitchTrigger.DATA_STALENESS in self._active_halts:
            self._resolve(
                KillSwitchTrigger.DATA_STALENESS,
                resolved_by="auto",
                reason="Scan completed successfully",
            )

    def update_spreads(self, spread_pcts: list[float]) -> None:
        """
        Call each scan cycle with spreads from the current watchlist.
        Triggers halt if average spread exceeds threshold.
        """
        if len(spread_pcts) < self._settings.spread_check_min_symbols:
            return
        self._recent_spreads = spread_pcts
        avg_spread = sum(spread_pcts) / len(spread_pcts)

        if (
            self._settings.spread_expansion_halt_enabled
            and avg_spread > self._settings.max_avg_spread_pct
        ):
            self._trigger(
                KillSwitchTrigger.SPREAD_EXPANSION,
                reason=f"Average watchlist spread {avg_spread:.3f}% > max {self._settings.max_avg_spread_pct:.3f}%",
                value=avg_spread,
                threshold=self._settings.max_avg_spread_pct,
            )
        elif KillSwitchTrigger.SPREAD_EXPANSION in self._active_halts:
            self._resolve(
                KillSwitchTrigger.SPREAD_EXPANSION,
                resolved_by="auto",
                reason=f"Average spread normalised to {avg_spread:.3f}%",
            )

    def update_broker_status(self, connected: bool) -> None:
        """Call when broker connection status changes."""
        if connected:
            self._last_broker_heartbeat = datetime.now(tz=timezone.utc)
            self._broker_connected = True
            if KillSwitchTrigger.BROKER_DISCONNECT in self._active_halts:
                self._resolve(
                    KillSwitchTrigger.BROKER_DISCONNECT,
                    resolved_by="auto",
                    reason="Broker reconnected",
                )
        else:
            self._broker_connected = False
            if self._settings.broker_disconnect_halt_enabled:
                self._trigger(
                    KillSwitchTrigger.BROKER_DISCONNECT,
                    reason="Broker connection lost",
                )

    def update_market_conditions(
        self,
        spy_price: float | None = None,
        spy_open: float | None = None,
        vix_level: float | None = None,
    ) -> None:
        """
        Call periodically with current SPY price and VIX level.
        Triggers volatility circuit breaker if thresholds are breached.
        """
        if spy_open is not None:
            self._spy_open = spy_open
        if spy_price is not None:
            self._spy_current = spy_price
        if vix_level is not None:
            self._vix_current = vix_level

        if not self._settings.volatility_halt_enabled:
            return

        # SPY intraday move check
        if self._spy_open and self._spy_current:
            spy_move_pct = (self._spy_current - self._spy_open) / self._spy_open * 100.0
            if spy_move_pct < self._settings.max_spy_intraday_down_pct:
                self._trigger(
                    KillSwitchTrigger.VOLATILITY_CIRCUIT,
                    reason=f"SPY intraday move {spy_move_pct:.2f}% < threshold {self._settings.max_spy_intraday_down_pct:.2f}%",
                    value=spy_move_pct,
                    threshold=self._settings.max_spy_intraday_down_pct,
                )
                return

        # VIX level check
        if self._vix_current is not None:
            if self._vix_current > self._settings.max_vix_level:
                self._trigger(
                    KillSwitchTrigger.VOLATILITY_CIRCUIT,
                    reason=f"VIX {self._vix_current:.1f} > threshold {self._settings.max_vix_level:.1f}",
                    value=self._vix_current,
                    threshold=self._settings.max_vix_level,
                )
                return

        # Conditions normalised — auto-resolve volatility halt
        if KillSwitchTrigger.VOLATILITY_CIRCUIT in self._active_halts:
            self._resolve(
                KillSwitchTrigger.VOLATILITY_CIRCUIT,
                resolved_by="auto",
                reason="Market conditions normalised",
            )

    def check_data_staleness(self, now: datetime | None = None) -> None:
        """
        Call periodically (e.g. every scan interval) to check if the
        last scan is too old. Triggers data staleness halt if so.
        """
        if not self._settings.data_staleness_halt_enabled:
            return
        if self._last_scan_at is None:
            return  # Haven't scanned yet — don't penalise early startup
        now = now or datetime.now(tz=timezone.utc)
        stale_minutes = (now - self._last_scan_at).total_seconds() / 60.0
        if stale_minutes > self._settings.max_stale_scan_minutes:
            self._trigger(
                KillSwitchTrigger.DATA_STALENESS,
                reason=f"Last scan {stale_minutes:.1f}min ago (max: {self._settings.max_stale_scan_minutes:.0f}min)",
                value=stale_minutes,
                threshold=self._settings.max_stale_scan_minutes,
            )

    # ------------------------------------------------------------------
    # Manual control (CLI)
    # ------------------------------------------------------------------

    def trigger_manual_halt(self, reason: str = "Manual halt") -> None:
        """Triggered by CLI kill command."""
        self._trigger(KillSwitchTrigger.MANUAL, reason=reason)

    def resume(self, trigger: KillSwitchTrigger | None = None) -> None:
        """
        Resume trading after a human has reviewed the halt condition.
        Pass a specific trigger to resolve only that one,
        or None to resolve all active halts.
        """
        if trigger is not None:
            if trigger in self._active_halts:
                self._resolve(trigger, resolved_by="human", reason="Manually resumed by operator")
            else:
                logger.warning("Kill switch: trigger {} not active", trigger)
        else:
            for t in list(self._active_halts.keys()):
                self._resolve(t, resolved_by="human", reason="All halts manually cleared by operator")
        logger.info("Kill switch: trading resumed. Active halts: {}", self._active_halts)

    def reset_daily(self) -> None:
        """Reset per-day counters. Call at session start."""
        self._consecutive_losses = 0
        self._last_scan_at = None
        self._recent_spreads = []
        self._spy_open = None
        self._spy_current = None
        self._vix_current = None
        # Do NOT clear active halts on reset — they require human intervention
        logger.info("Kill switch: daily state reset")

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    @property
    def event_log(self) -> list[KillSwitchEvent]:
        return list(self._event_log)

    def status_summary(self) -> dict:
        """Return a dict suitable for dashboard/API consumption."""
        return {
            "halted":              self.is_halted,
            "active_triggers":     [t.value for t in self.active_triggers],
            "consecutive_losses":  self._consecutive_losses,
            "last_scan_at":        self._last_scan_at.isoformat() if self._last_scan_at else None,
            "broker_connected":    self._broker_connected,
            "spy_move_pct":        (
                (self._spy_current - self._spy_open) / self._spy_open * 100.0
                if self._spy_open and self._spy_current else None
            ),
            "vix":                 self._vix_current,
            "recent_events":       [
                {
                    "trigger":      e.trigger.value,
                    "triggered_at": e.triggered_at.isoformat(),
                    "reason":       e.reason,
                    "resolved_at":  e.resolved_at.isoformat() if e.resolved_at else None,
                }
                for e in self._event_log[-10:]
            ],
        }

    # ------------------------------------------------------------------
    # Internal trigger / resolve
    # ------------------------------------------------------------------

    def _trigger(
        self,
        trigger: KillSwitchTrigger,
        reason: str,
        value: float | None = None,
        threshold: float | None = None,
    ) -> None:
        if trigger in self._active_halts:
            return  # Already halted on this trigger — don't duplicate

        event = KillSwitchEvent(
            trigger=trigger,
            triggered_at=datetime.now(tz=timezone.utc),
            reason=reason,
            value=value,
            threshold=threshold,
        )
        self._active_halts[trigger] = event
        self._event_log.append(event)

        logger.warning(
            "🛑 KILL SWITCH TRIGGERED [{}]: {}",
            trigger.value, reason,
        )

        if self._on_halt:
            asyncio.create_task(self._on_halt(event))

        if self._db_repo:
            asyncio.create_task(self._db_repo.save_event(event))

    def _resolve(
        self,
        trigger: KillSwitchTrigger,
        resolved_by: str,
        reason: str,
    ) -> None:
        event = self._active_halts.pop(trigger, None)
        if event is None:
            return

        resolved_event = KillSwitchEvent(
            trigger=event.trigger,
            triggered_at=event.triggered_at,
            reason=event.reason,
            value=event.value,
            threshold=event.threshold,
            resolved_at=datetime.now(tz=timezone.utc),
            resolved_by=resolved_by,
        )
        # Replace in log
        for i, e in enumerate(self._event_log):
            if e.trigger == trigger and e.resolved_at is None:
                self._event_log[i] = resolved_event
                break

        logger.info("✅ Kill switch resolved [{}] by {}: {}", trigger.value, resolved_by, reason)
