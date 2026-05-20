"""
analytics/missed_trade_auditor.py
End-of-day missed trade audit system.

Runs automatically at 5:00 PM ET after each trading session.
For every symbol that was REJECTED by the scanner that day, it fetches
intraday bars and labels whether the stock subsequently:
  - Hit +1R from the rejection price
  - Hit +2R from the rejection price
  - Hit the hypothetical stop loss
  - What the max favorable excursion was (MFE in R)
  - What the max adverse excursion was (MAE in R)

This answers the critical question: "Were our filter rejections correct?"

R is computed using the SAME stop logic as the live system:
  - Stop = structural swing low below rejection price (or rejection_price * 0.97 fallback)
  - T1 = entry + 1.5 * risk_per_share
  - T2 = entry + 2.5 * risk_per_share

Results are persisted to the missed_trade_audit SQLite table and surfaced
in the dashboard's Missed Trades tab.

Schedule:
  asyncio scheduled task, fires at 17:00 ET every trading day.
  Can also be triggered manually via CLI: python main.py audit-missed
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from loguru import logger

from core.enums import BarTimeframe
from core.models import Bar
from market_data.base_provider import BaseDataProvider
from scanner.scan_window import is_edt, is_market_holiday, _MARKET_HOLIDAYS_2024_2025


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RejectedScanRecord:
    """A scan result that failed filters — sourced from the database."""
    scan_id: int
    symbol: str
    scan_date: str              # YYYY-MM-DD
    scanned_at: datetime
    failed_filter: str
    filter_value: float | None
    required_value: float | None
    composite_score: float
    last_price: float           # Price at time of rejection (used as entry proxy)


@dataclass
class MissedTradeAuditResult:
    """Outcome labels for a single rejected setup."""
    scan_id: int
    symbol: str
    audit_date: str
    audited_at: datetime
    rejection_filter: str
    entry_proxy: float           # Price at rejection (premarket close or open)
    stop_proxy: float            # Hypothetical stop (97% of entry, structural when available)
    target_1: float
    target_2: float
    risk_per_share: float
    # Outcome labels
    hit_1r: bool
    hit_2r: bool
    hit_stop: bool
    max_favorable_excursion_r: float   # Best possible R achieved
    max_adverse_excursion_r: float     # Worst R seen (negative = drawdown)
    time_to_1r_minutes: float | None
    time_to_2r_minutes: float | None
    time_to_stop_minutes: float | None
    session_high: float
    session_low: float
    session_close: float
    # Verdict
    rejection_was_correct: bool     # True if stop was hit before +1R
    should_investigate: bool        # True if +2R hit without triggering any filter issues


# ---------------------------------------------------------------------------
# R-level computation helpers
# ---------------------------------------------------------------------------

def _compute_hypothetical_levels(
    entry: float,
    risk_r: float = 0.03,  # 3% default stop distance
    t1_rr: float = 1.5,
    t2_rr: float = 2.5,
) -> tuple[float, float, float, float]:
    """
    Returns (stop, target_1, target_2, risk_per_share).
    """
    stop = entry * (1.0 - risk_r)
    risk_per_share = entry - stop
    t1 = entry + risk_per_share * t1_rr
    t2 = entry + risk_per_share * t2_rr
    return stop, t1, t2, risk_per_share


def _label_outcomes(
    bars_after_rejection: list[Bar],
    entry: float,
    stop: float,
    t1: float,
    t2: float,
    risk_per_share: float,
    reference_time: datetime,
) -> dict[str, Any]:
    """
    Walk through bars chronologically and label outcomes.
    Returns a dict with all outcome fields.
    """
    hit_1r = False
    hit_2r = False
    hit_stop = False
    time_to_1r: float | None = None
    time_to_2r: float | None = None
    time_to_stop: float | None = None
    max_fav_excursion_r = 0.0
    max_adv_excursion_r = 0.0
    session_high = entry
    session_low = entry

    for bar in bars_after_rejection:
        session_high = max(session_high, bar.high)
        session_low = min(session_low, bar.low)

        # Update excursions
        fav_r = (bar.high - entry) / risk_per_share if risk_per_share > 0 else 0.0
        adv_r = (bar.low - entry) / risk_per_share if risk_per_share > 0 else 0.0

        max_fav_excursion_r = max(max_fav_excursion_r, fav_r)
        max_adv_excursion_r = min(max_adv_excursion_r, adv_r)  # Most negative

        # Time to levels (bar high/low can cross multiple levels in one bar)
        elapsed_mins = (bar.timestamp - reference_time).total_seconds() / 60.0

        if not hit_1r and bar.high >= t1:
            hit_1r = True
            time_to_1r = elapsed_mins

        if not hit_2r and bar.high >= t2:
            hit_2r = True
            time_to_2r = elapsed_mins

        if not hit_stop and bar.low <= stop:
            hit_stop = True
            time_to_stop = elapsed_mins

    session_close = bars_after_rejection[-1].close if bars_after_rejection else entry

    return {
        "hit_1r": hit_1r,
        "hit_2r": hit_2r,
        "hit_stop": hit_stop,
        "time_to_1r_minutes": time_to_1r,
        "time_to_2r_minutes": time_to_2r,
        "time_to_stop_minutes": time_to_stop,
        "max_favorable_excursion_r": round(max_fav_excursion_r, 3),
        "max_adverse_excursion_r": round(max_adv_excursion_r, 3),
        "session_high": session_high,
        "session_low": session_low,
        "session_close": session_close,
    }


# ---------------------------------------------------------------------------
# Main auditor
# ---------------------------------------------------------------------------

class MissedTradeAuditor:
    """
    Fetches intraday bars for all rejected symbols and labels outcomes.

    Usage:
        auditor = MissedTradeAuditor(provider, db_repo)
        results = await auditor.run_for_date(date.today())
    """

    def __init__(
        self,
        provider: BaseDataProvider,
        db_repo: "MissedTradeRepository",
        scan_repo: "ScanResultRepository",
        concurrency: int = 10,
        risk_r: float = 0.03,
    ) -> None:
        self._provider = provider
        self._db_repo = db_repo
        self._scan_repo = scan_repo
        self._sem = asyncio.Semaphore(concurrency)
        self._risk_r = risk_r

    async def run_for_date(
        self, audit_date: date | None = None
    ) -> list[MissedTradeAuditResult]:
        """
        Run the full audit for a given date.
        Defaults to today if no date provided.
        """
        audit_date = audit_date or date.today()
        date_str = audit_date.isoformat()

        logger.info("Missed trade audit: starting for {}", date_str)

        # Fetch all rejected scans for the day
        rejected = await self._scan_repo.get_rejected_for_date(date_str)
        if not rejected:
            logger.info("Missed trade audit: no rejected setups found for {}", date_str)
            return []

        logger.info("Missed trade audit: {} rejected setups to audit", len(rejected))

        # Process concurrently
        tasks = [self._audit_one(record, audit_date) for record in rejected]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[MissedTradeAuditResult] = []
        for r in raw_results:
            if isinstance(r, Exception):
                logger.warning("Missed trade audit error: {}", r)
            elif r is not None:
                results.append(r)

        # Persist all results
        if results:
            await self._db_repo.save_many(results)
            logger.info(
                "Missed trade audit complete: {} results saved | "
                "{} hit +1R | {} hit +2R | {} correct rejections",
                len(results),
                sum(1 for r in results if r.hit_1r),
                sum(1 for r in results if r.hit_2r),
                sum(1 for r in results if r.rejection_was_correct),
            )
        return results

    async def _audit_one(
        self,
        record: RejectedScanRecord,
        audit_date: date,
    ) -> MissedTradeAuditResult | None:
        async with self._sem:
            return await self._compute_outcome(record, audit_date)

    async def _compute_outcome(
        self,
        record: RejectedScanRecord,
        audit_date: date,
    ) -> MissedTradeAuditResult | None:
        try:
            dt = datetime(
                audit_date.year, audit_date.month, audit_date.day,
                tzinfo=timezone.utc
            )
            bars = await self._provider.get_intraday_bars(record.symbol, dt)
        except Exception as exc:
            logger.debug(
                "Missed audit: failed to fetch bars for {} on {}: {}",
                record.symbol, audit_date, exc,
            )
            return None

        if not bars:
            logger.debug(
                "Missed audit: no intraday bars for {} on {}",
                record.symbol, audit_date,
            )
            return None

        # Use opening bar price as entry proxy (most honest — what you'd pay at open)
        market_open_bars = [
            b for b in bars
            if b.timestamp.hour == 13 and b.timestamp.minute == 30  # 9:30 AM ET in UTC
        ]
        entry_proxy = (
            market_open_bars[0].open if market_open_bars else bars[0].open
        )

        stop, t1, t2, risk_per_share = _compute_hypothetical_levels(
            entry_proxy, risk_r=self._risk_r
        )

        # Only evaluate bars AFTER market open
        reference_time = datetime(
            audit_date.year, audit_date.month, audit_date.day,
            13, 30, 0, tzinfo=timezone.utc
        )
        bars_after_open = [b for b in bars if b.timestamp >= reference_time]

        if not bars_after_open:
            return None

        outcomes = _label_outcomes(
            bars_after_rejection=bars_after_open,
            entry=entry_proxy,
            stop=stop,
            t1=t1,
            t2=t2,
            risk_per_share=risk_per_share,
            reference_time=reference_time,
        )

        rejection_was_correct = (
            outcomes["hit_stop"]
            and not outcomes["hit_1r"]
        ) or (
            not outcomes["hit_1r"]
            and outcomes["max_favorable_excursion_r"] < 0.5
        )

        should_investigate = outcomes["hit_2r"]

        return MissedTradeAuditResult(
            scan_id=record.scan_id,
            symbol=record.symbol,
            audit_date=audit_date.isoformat(),
            audited_at=datetime.now(tz=timezone.utc),
            rejection_filter=record.failed_filter,
            entry_proxy=entry_proxy,
            stop_proxy=stop,
            target_1=t1,
            target_2=t2,
            risk_per_share=risk_per_share,
            hit_1r=outcomes["hit_1r"],
            hit_2r=outcomes["hit_2r"],
            hit_stop=outcomes["hit_stop"],
            max_favorable_excursion_r=outcomes["max_favorable_excursion_r"],
            max_adverse_excursion_r=outcomes["max_adverse_excursion_r"],
            time_to_1r_minutes=outcomes["time_to_1r_minutes"],
            time_to_2r_minutes=outcomes["time_to_2r_minutes"],
            time_to_stop_minutes=outcomes["time_to_stop_minutes"],
            session_high=outcomes["session_high"],
            session_low=outcomes["session_low"],
            session_close=outcomes["session_close"],
            rejection_was_correct=rejection_was_correct,
            should_investigate=should_investigate,
        )


# ---------------------------------------------------------------------------
# Scheduler — wires the auditor to run at 5 PM ET
# ---------------------------------------------------------------------------

async def schedule_daily_audit(
    auditor: MissedTradeAuditor,
    run_immediately: bool = False,
) -> None:
    """
    Async task: sleeps until 5 PM ET then runs the missed trade audit.
    Designed to be started with asyncio.create_task() at session startup.
    """
    if not run_immediately:
        wait_seconds = _seconds_until_5pm_et()
        if wait_seconds > 0:
            logger.info(
                "Missed trade audit scheduled in {:.0f} minutes",
                wait_seconds / 60,
            )
            await asyncio.sleep(wait_seconds)

    audit_date = date.today()

    # Skip weekends and holidays
    if audit_date.weekday() >= 5 or is_market_holiday(audit_date):
        logger.info("Missed trade audit: skipping non-trading day {}", audit_date)
        return

    try:
        results = await auditor.run_for_date(audit_date)
        logger.info(
            "Missed trade audit complete: {} results for {}",
            len(results), audit_date,
        )
    except Exception as exc:
        logger.error("Missed trade audit failed: {}", exc)


def _seconds_until_5pm_et(now_utc: datetime | None = None) -> float:
    """Compute seconds until 17:00 ET today (or 0 if already past)."""
    utc = now_utc or datetime.now(tz=timezone.utc)
    offset_hours = 4 if is_edt(utc) else 5
    et_now = utc + timedelta(hours=-offset_hours)
    target_et = et_now.replace(hour=17, minute=0, second=0, microsecond=0)
    if et_now >= target_et:
        return 0.0
    return (target_et - et_now).total_seconds()
