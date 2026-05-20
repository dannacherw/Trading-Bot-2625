"""
analytics/eod_runner.py
End-of-day task orchestrator.

Fires automatically at 17:00 ET as an asyncio task inside the live trading
session. Sequentially runs:

  1. Missed trade audit     — label what each rejected setup did post-open
  2. Percentile rebuild     — update 30-day quantile distributions for scoring
  3. Daily performance row  — write today's summary to Daily_Performance sheet
  4. Filter quality report  — print filter false-negative analysis to terminal
  5. Scan batch sync        — write today's scan_results to Sheets Scan_Log
  6. Bot health log         — write final health snapshot to Bot_Health_Log

Architecture:
  - Runs as a single asyncio.create_task() inside _run_session().
  - Each task is wrapped in try/except so a failure in one does not abort the rest.
  - All wall-clock timing uses the scan_window helpers for correct EDT/EST handling.
  - The runner logs start/finish times for each step so the operator can see
    exactly where time is spent in the session logs.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

from loguru import logger

from scanner.scan_window import seconds_until_time_et, is_market_holiday

if TYPE_CHECKING:
    from analytics.filter_quality import MissedTradeRepository
    from analytics.missed_trade_auditor import MissedTradeAuditor
    from database.db_manager import DatabaseManager
    from database.repository import ScanResultRepository, TradeRepository
    from database.v4_repositories import (
        MissedTradeRepository,
        PercentileDistributionRepository,
    )
    from integrations.sheets_writer import SheetsWriter
    from market_data.vix_provider import VIXMonitor


# ---------------------------------------------------------------------------
# EOD runner
# ---------------------------------------------------------------------------

class EODRunner:
    """
    Orchestrates all end-of-day tasks.

    Inject all dependencies at construction; call schedule() from inside the
    trading session's asyncio.gather() to start the background waiter.

    Example:
        eod = EODRunner(
            db=db,
            trade_repo=trade_repo,
            scan_repo=scan_repo,
            missed_auditor=auditor,
            percentile_repo=pct_repo,
            missed_repo=missed_repo,
            sheets_writer=writer,   # may be None if Sheets not configured
            vix_monitor=vix,        # may be None
        )
        asyncio.create_task(eod.schedule())
    """

    def __init__(
        self,
        db: "DatabaseManager",
        trade_repo: "TradeRepository",
        scan_repo: "ScanResultRepository",
        missed_auditor: "MissedTradeAuditor",
        percentile_repo: "PercentileDistributionRepository",
        missed_repo: "MissedTradeRepository",
        sheets_writer: "SheetsWriter | None" = None,
        vix_monitor: "VIXMonitor | None" = None,
        run_immediately: bool = False,   # True for testing / manual runs
    ) -> None:
        self._db              = db
        self._trade_repo      = trade_repo
        self._scan_repo       = scan_repo
        self._missed_auditor  = missed_auditor
        self._pct_repo        = percentile_repo
        self._missed_repo     = missed_repo
        self._sheets          = sheets_writer
        self._vix             = vix_monitor
        self._run_immediately = run_immediately

    async def schedule(self) -> None:
        """
        Sleep until 17:00 ET then run all EOD tasks.
        Designed to be started with asyncio.create_task().
        """
        today = date.today()

        if today.weekday() >= 5 or is_market_holiday(today):
            logger.info("EOD runner: skipping non-trading day {}", today)
            return

        if not self._run_immediately:
            wait_secs = seconds_until_time_et(17, 0)
            if wait_secs > 0:
                logger.info(
                    "EOD runner: scheduled in {:.0f} minutes ({:.0f}s)",
                    wait_secs / 60, wait_secs,
                )
                try:
                    await asyncio.sleep(wait_secs)
                except asyncio.CancelledError:
                    logger.info("EOD runner: cancelled before firing")
                    return

        logger.info("EOD runner: starting — {}", datetime.now(tz=timezone.utc).isoformat())
        await self._run_all(today)
        logger.info("EOD runner: complete")

    # ------------------------------------------------------------------
    # Task orchestration
    # ------------------------------------------------------------------

    async def _run_all(self, today: date) -> None:
        date_str = today.isoformat()

        # Step 1: Missed trade audit
        await self._step("missed_trade_audit", self._run_missed_audit(today))

        # Step 2: Percentile distributions rebuild
        await self._step("percentile_rebuild", self._run_percentile_rebuild())

        # Step 3: Sheets — daily performance row
        if self._sheets is not None:
            await self._step("daily_performance_sheet", self._run_daily_perf(date_str))

        # Step 4: Filter quality report (terminal only)
        await self._step("filter_quality_report", self._run_filter_report(date_str))

        # Step 5: Sheets — batch scan log sync
        if self._sheets is not None:
            await self._step("scan_log_batch_sync", self._run_scan_batch(date_str))

        # Step 6: Bot health log
        if self._sheets is not None:
            await self._step("bot_health_log", self._run_health_log())

    @staticmethod
    async def _step(name: str, coro: "asyncio.Coroutine") -> None:  # type: ignore[type-arg]
        """Run a coroutine, catching and logging any exception without aborting."""
        t0 = asyncio.get_event_loop().time()
        try:
            await coro
            elapsed = asyncio.get_event_loop().time() - t0
            logger.info("EOD [{}]: done in {:.1f}s", name, elapsed)
        except Exception as exc:
            logger.error("EOD [{}]: FAILED — {}", name, exc)

    # ------------------------------------------------------------------
    # Individual EOD tasks
    # ------------------------------------------------------------------

    async def _run_missed_audit(self, today: date) -> None:
        results = await self._missed_auditor.run_for_date(today)
        logger.info(
            "Missed audit: {} results | {} hit +1R | {} hit +2R | {} correct rejections",
            len(results),
            sum(1 for r in results if r.hit_1r),
            sum(1 for r in results if r.hit_2r),
            sum(1 for r in results if r.rejection_was_correct),
        )

    async def _run_percentile_rebuild(self) -> None:
        """
        Recompute 30-day rolling percentile distributions from scan history.
        Stores results in percentile_distributions table and loads them into
        the global QuantileStore so tomorrow's session uses fresh distributions.
        """
        from scanner.percentile_scoring import load_quantile_store

        quantiles = await self._pct_repo.compute_and_save_from_scans(
            self._db, lookback_days=30
        )
        load_quantile_store(quantiles)
        logger.info(
            "Percentile rebuild: {} features updated (samples: {})",
            len(quantiles),
            {q.feature: q.sample_count for q in quantiles},
        )

    async def _run_daily_perf(self, date_str: str) -> None:
        assert self._sheets is not None
        from core.enums import PositionSide
        trades = await self._trade_repo.get_trades_for_date(date_str)
        if not trades:
            logger.info("Daily performance: no trades today")
            return
        regime = (
            self._vix._last_vix.__class__.__name__
            if self._vix and self._vix.last_vix else ""
        )
        await self._sheets.append_daily_performance(
            date_str=date_str,
            trades=trades,
            market_regime=regime,
        )

    async def _run_filter_report(self, date_str: str) -> None:
        from analytics.filter_quality import build_filter_stats, print_filter_quality_report

        # Report on last 30 days to give meaningful sample sizes
        from datetime import timedelta
        start_30d = (date.fromisoformat(date_str) - timedelta(days=30)).isoformat()

        stats = await build_filter_stats(
            missed_repo=self._missed_repo,
            date_range_start=start_30d,
            date_range_end=date_str,
        )
        print_filter_quality_report(stats, start_30d, date_str)

    async def _run_scan_batch(self, date_str: str) -> None:
        assert self._sheets is not None
        # Fetch ALL scan results for today (pass + fail) for Sheets batch
        rows = await self._db.fetch_all(
            """
            SELECT raw_json FROM scan_results
            WHERE date(scanned_at) = ?
            ORDER BY scanned_at ASC
            """,
            (date_str,),
        )
        if not rows:
            logger.info("Scan batch: no scan results for {}", date_str)
            return

        import json
        from core.models import ScanResult
        results = []
        for row in rows:
            try:
                results.append(ScanResult.model_validate_json(row["raw_json"]))
            except Exception:
                continue

        await self._sheets.batch_append_scans(results)
        logger.info("Scan batch: {} rows written to Sheets", len(results))

    async def _run_health_log(self) -> None:
        assert self._sheets is not None
        health = {
            "timestamp":            datetime.now(tz=timezone.utc),
            "bot_status":           "STOPPED",
            "scanner_status":       "STOPPED",
            "broker_status":        "DISCONNECTED",
            "data_feed_status":     "DISCONNECTED",
            "api_latency_ms":       "",
            "order_latency_ms":     "",
            "errors_count":         0,
            "last_successful_scan": None,
            "last_successful_order": None,
            "kill_switch_triggered": False,
            "error_message":        "EOD — session complete",
        }
        await self._sheets.append_bot_health(health)
