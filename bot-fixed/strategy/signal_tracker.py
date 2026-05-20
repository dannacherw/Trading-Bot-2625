"""
strategy/signal_tracker.py
Rolling signal quality tracker — monitors win rate across multiple dimensions
and logs warnings when performance degrades.

Per user decision: this module logs warnings only; humans decide whether to pause.
No automatic trading halt is triggered.

Tracks win rate by:
  - Signal strength (STRONG / MODERATE / WEAK)
  - Time of day bucket (9:30–10:00, 10:00–11:30, 11:30–13:00, 13:00–15:45)
  - Gap bucket (3–6%, 6–10%, 10%+)
  - Catalyst present vs. absent
  - Overall rolling window (last N signals)
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from core.enums import ExitReason, SignalStrength


@dataclass
class SignalOutcome:
    """Records the outcome of a single acted-on signal."""
    signal_id: str
    symbol: str
    strength: SignalStrength
    generated_at: datetime
    entry_price: float
    exit_price: float | None
    exit_reason: ExitReason | None
    net_pnl: float | None
    r_multiple: float | None
    gap_pct: float = 0.0
    has_catalyst: bool = False

    @property
    def is_resolved(self) -> bool:
        return self.exit_price is not None

    @property
    def is_winner(self) -> bool:
        return self.net_pnl is not None and self.net_pnl > 0

    @property
    def time_bucket(self) -> str:
        h = self.generated_at.hour
        m = self.generated_at.minute
        # Approximate ET: subtract 4 hours from UTC
        et_hour = (h - 4) % 24
        if et_hour == 9 and m < 30:
            return "pre_open"
        elif et_hour == 9:
            return "09:30-10:00"
        elif et_hour == 10:
            return "10:00-11:00"
        elif et_hour == 11:
            return "11:00-12:00"
        elif et_hour == 12:
            return "12:00-13:00"
        else:
            return "13:00-close"

    @property
    def gap_bucket(self) -> str:
        if self.gap_pct < 6:
            return "3-6%"
        elif self.gap_pct < 10:
            return "6-10%"
        else:
            return "10%+"


@dataclass
class PerformanceSlice:
    """Win rate and avg R for a given filter dimension."""
    dimension: str
    total: int = 0
    winners: int = 0
    sum_r: float = 0.0
    min_sample_size: int = 10

    @property
    def win_rate(self) -> float:
        return self.winners / self.total if self.total > 0 else 0.0

    @property
    def avg_r(self) -> float:
        return self.sum_r / self.total if self.total > 0 else 0.0

    @property
    def sample_adequate(self) -> bool:
        return self.total >= self.min_sample_size

    def __str__(self) -> str:
        if not self.sample_adequate:
            return f"{self.dimension}: {self.total} trades (need {self.min_sample_size} for significance)"
        return (
            f"{self.dimension}: WR={self.win_rate:.1%} "
            f"avgR={self.avg_r:+.2f} n={self.total}"
        )


class SignalOutcomeTracker:
    """
    Records signal outcomes and computes rolling performance statistics.
    Logs structured warnings when win rates drop below thresholds.

    Data is optionally persisted to a JSON file for cross-session continuity.
    """

    WARN_WIN_RATE_STRONG = 0.50     # Warn if STRONG signals below 50% WR
    WARN_WIN_RATE_OVERALL = 0.45    # Warn if overall rolling below 45% WR
    ROLLING_WINDOW = 20             # Rolling window size for recent performance

    def __init__(
        self,
        persist_path: str | None = "data/signal_tracker.json",
    ) -> None:
        self._outcomes: deque[SignalOutcome] = deque(maxlen=500)
        self._pending: dict[str, SignalOutcome] = {}  # signal_id → unresolved
        self._persist_path = Path(persist_path) if persist_path else None
        self._load_from_disk()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_signal(
        self,
        signal_id: str,
        symbol: str,
        strength: SignalStrength,
        generated_at: datetime,
        entry_price: float,
        gap_pct: float = 0.0,
        has_catalyst: bool = False,
    ) -> None:
        """Register a signal that was acted on (entry taken)."""
        outcome = SignalOutcome(
            signal_id=signal_id,
            symbol=symbol,
            strength=strength,
            generated_at=generated_at,
            entry_price=entry_price,
            exit_price=None,
            exit_reason=None,
            net_pnl=None,
            r_multiple=None,
            gap_pct=gap_pct,
            has_catalyst=has_catalyst,
        )
        self._pending[signal_id] = outcome

    def resolve_signal(
        self,
        signal_id: str,
        exit_price: float,
        exit_reason: ExitReason,
        net_pnl: float,
        r_multiple: float,
    ) -> None:
        """Mark a signal as resolved with its trade outcome."""
        pending = self._pending.pop(signal_id, None)
        if pending is None:
            logger.debug("SignalTracker: unknown signal_id {}", signal_id)
            return

        import dataclasses
        resolved = dataclasses.replace(
            pending,
            exit_price=exit_price,
            exit_reason=exit_reason,
            net_pnl=net_pnl,
            r_multiple=r_multiple,
        )
        self._outcomes.append(resolved)
        self._check_warnings()
        self._save_to_disk()

    # ------------------------------------------------------------------
    # Performance analysis
    # ------------------------------------------------------------------

    def get_all_slices(self) -> list[PerformanceSlice]:
        """Compute win rates across all tracked dimensions."""
        resolved = [o for o in self._outcomes if o.is_resolved]
        if not resolved:
            return []

        slices: dict[str, PerformanceSlice] = {}

        def _update(key: str, outcome: SignalOutcome) -> None:
            if key not in slices:
                slices[key] = PerformanceSlice(dimension=key)
            s = slices[key]
            s.total += 1
            if outcome.is_winner:
                s.winners += 1
            if outcome.r_multiple is not None:
                s.sum_r += outcome.r_multiple

        for o in resolved:
            _update("overall", o)
            _update(f"strength={o.strength.value}", o)
            _update(f"time={o.time_bucket}", o)
            _update(f"gap={o.gap_bucket}", o)
            _update(f"catalyst={'yes' if o.has_catalyst else 'no'}", o)

        return list(slices.values())

    def get_rolling_win_rate(
        self, n: int = ROLLING_WINDOW, strength: SignalStrength | None = None
    ) -> float | None:
        """Rolling win rate over the last n resolved outcomes."""
        resolved = [o for o in self._outcomes if o.is_resolved]
        if strength:
            resolved = [o for o in resolved if o.strength == strength]
        recent = resolved[-n:]
        if len(recent) < 5:
            return None  # Insufficient data
        return sum(1 for o in recent if o.is_winner) / len(recent)

    # ------------------------------------------------------------------
    # Warning system
    # ------------------------------------------------------------------

    def _check_warnings(self) -> None:
        """Log structured warnings when performance degrades."""
        # Overall rolling window
        overall_wr = self.get_rolling_win_rate(self.ROLLING_WINDOW)
        if overall_wr is not None and overall_wr < self.WARN_WIN_RATE_OVERALL:
            logger.warning(
                "⚠️  SIGNAL QUALITY DEGRADATION | "
                "Overall rolling WR={:.1%} < threshold {:.1%} over last {} signals. "
                "Consider pausing or reviewing market conditions.",
                overall_wr, self.WARN_WIN_RATE_OVERALL, self.ROLLING_WINDOW,
            )

        # STRONG signal rolling window (most actionable)
        strong_wr = self.get_rolling_win_rate(15, SignalStrength.STRONG)
        if strong_wr is not None and strong_wr < self.WARN_WIN_RATE_STRONG:
            logger.warning(
                "⚠️  STRONG SIGNAL DEGRADATION | "
                "STRONG signal rolling WR={:.1%} < threshold {:.1%} over last 15 signals. "
                "Human review recommended before next entry.",
                strong_wr, self.WARN_WIN_RATE_STRONG,
            )

    def log_daily_report(self) -> None:
        """Log a full performance breakdown at session end."""
        slices = self.get_all_slices()
        if not slices:
            logger.info("SignalTracker: no resolved signals to report")
            return

        logger.info("=== Signal Quality Report ===")
        for s in sorted(slices, key=lambda x: x.dimension):
            logger.info("  {}", str(s))
        logger.info("=== End Report ===")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_to_disk(self) -> None:
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(exist_ok=True, parents=True)
            data = [
                {
                    "signal_id": o.signal_id,
                    "symbol": o.symbol,
                    "strength": o.strength.value,
                    "generated_at": o.generated_at.isoformat(),
                    "entry_price": o.entry_price,
                    "exit_price": o.exit_price,
                    "exit_reason": o.exit_reason.value if o.exit_reason else None,
                    "net_pnl": o.net_pnl,
                    "r_multiple": o.r_multiple,
                    "gap_pct": o.gap_pct,
                    "has_catalyst": o.has_catalyst,
                }
                for o in self._outcomes
                if o.is_resolved
            ]
            self._persist_path.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.debug("SignalTracker: failed to persist: {}", exc)

    def _load_from_disk(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text())
            for d in data:
                o = SignalOutcome(
                    signal_id=d["signal_id"],
                    symbol=d["symbol"],
                    strength=SignalStrength(d["strength"]),
                    generated_at=datetime.fromisoformat(d["generated_at"]),
                    entry_price=d["entry_price"],
                    exit_price=d.get("exit_price"),
                    exit_reason=ExitReason(d["exit_reason"]) if d.get("exit_reason") else None,
                    net_pnl=d.get("net_pnl"),
                    r_multiple=d.get("r_multiple"),
                    gap_pct=d.get("gap_pct", 0.0),
                    has_catalyst=d.get("has_catalyst", False),
                )
                self._outcomes.append(o)
            logger.info("SignalTracker: loaded {} historical outcomes", len(self._outcomes))
        except Exception as exc:
            logger.debug("SignalTracker: failed to load history: {}", exc)
