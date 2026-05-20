"""
database/v4_repositories.py
Typed async repositories for all v4 schema tables.

Follows the exact same pattern as repository.py — async methods,
DatabaseManager injection, minimal serialisation overhead.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any

import numpy as np

from analytics.missed_trade_auditor import MissedTradeAuditResult, RejectedScanRecord
from database.db_manager import DatabaseManager
from risk.kill_switch import KillSwitchEvent, KillSwitchTrigger
from scanner.percentile_scoring import FeatureQuantiles


# ---------------------------------------------------------------------------
# Float data cache repository
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass


@_dataclass
class _FloatLookupResult:
    """Result of a DB float lookup. Distinguishes miss from cached-None."""
    was_found: bool   # True = row exists in DB (even if float_shares is None)
    value: int | None # float_shares value (None = float genuinely unknown)


class FloatCacheRepository:
    """
    Persist float share data to SQLite to avoid re-fetching each session.

    Key design: was_fetched column distinguishes:
      - Row absent: never fetched (MISS)
      - Row present, float_shares=NULL: fetched but unavailable from all sources
      - Row present, float_shares=N: known float
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    async def get_with_hit_flag(
        self, symbol: str, as_of_date: str | None = None
    ) -> _FloatLookupResult:
        """
        Return a lookup result distinguishing cache miss from cached-None.
        Used by CompositeFloatProvider to avoid false "not found" on unknown floats.
        """
        today = as_of_date or date.today().isoformat()
        row = await self._db.fetch_one(
            "SELECT float_shares, was_fetched FROM float_data_cache WHERE symbol=? AND fetched_date=?",
            (symbol, today),
        )
        if row is None or not row["was_fetched"]:
            return _FloatLookupResult(was_found=False, value=None)
        return _FloatLookupResult(was_found=True, value=row["float_shares"])

    async def get(self, symbol: str, as_of_date: str | None = None) -> int | None:
        """Simple get — returns None for both miss and cached-None. Use get_with_hit_flag for accuracy."""
        result = await self.get_with_hit_flag(symbol, as_of_date)
        return result.value

    async def set(
        self,
        symbol: str,
        float_shares: int | None,
        source: str = "unknown",
    ) -> None:
        today = date.today().isoformat()
        now = datetime.now(tz=timezone.utc).isoformat()
        await self._db.execute(
            """
            INSERT INTO float_data_cache
                (symbol, float_shares, was_fetched, source, fetched_date, fetched_at)
            VALUES (?, ?, 1, ?, ?, ?)
            ON CONFLICT(symbol, fetched_date) DO UPDATE SET
                float_shares=excluded.float_shares,
                was_fetched=1,
                source=excluded.source,
                fetched_at=excluded.fetched_at
            """,
            (symbol, float_shares, source, today, now),
        )

    async def get_batch(
        self, symbols: list[str], as_of_date: str | None = None
    ) -> dict[str, int | None]:
        """Return {symbol: float_shares} for multiple symbols. None = unknown or not fetched."""
        today = as_of_date or date.today().isoformat()
        placeholders = ",".join("?" for _ in symbols)
        rows = await self._db.fetch_all(
            f"SELECT symbol, float_shares, was_fetched FROM float_data_cache "
            f"WHERE symbol IN ({placeholders}) AND fetched_date=?",
            (*symbols, today),
        )
        result: dict[str, int | None] = {sym: None for sym in symbols}
        for row in rows:
            if row["was_fetched"]:
                result[row["symbol"]] = row["float_shares"]
        return result


# ---------------------------------------------------------------------------
# Kill switch events repository
# ---------------------------------------------------------------------------

class KillSwitchRepository:
    """Persist kill switch events for audit trail and dashboard."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    async def save_event(self, event: KillSwitchEvent) -> None:
        await self._db.execute(
            """
            INSERT INTO kill_switch_events
                (trigger, triggered_at, reason, value, threshold, resolved_at, resolved_by)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.trigger.value,
                event.triggered_at.isoformat(),
                event.reason,
                event.value,
                event.threshold,
                event.resolved_at.isoformat() if event.resolved_at else None,
                event.resolved_by,
            ),
        )

    async def update_resolved(
        self,
        trigger: KillSwitchTrigger,
        triggered_at: datetime,
        resolved_at: datetime,
        resolved_by: str,
    ) -> None:
        await self._db.execute(
            """
            UPDATE kill_switch_events
            SET resolved_at=?, resolved_by=?
            WHERE trigger=? AND triggered_at=? AND resolved_at IS NULL
            """,
            (
                resolved_at.isoformat(),
                resolved_by,
                trigger.value,
                triggered_at.isoformat(),
            ),
        )

    async def get_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = await self._db.fetch_all(
            "SELECT * FROM kill_switch_events ORDER BY triggered_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]

    async def get_unresolved(self) -> list[dict[str, Any]]:
        rows = await self._db.fetch_all(
            "SELECT * FROM kill_switch_events WHERE resolved_at IS NULL ORDER BY triggered_at ASC",
        )
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Missed trade audit repository
# ---------------------------------------------------------------------------

class MissedTradeRepository:
    """Persist and query missed trade audit results."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    async def save_many(self, results: list[MissedTradeAuditResult]) -> None:
        params = [
            (
                r.scan_id, r.symbol, r.audit_date, r.audited_at.isoformat(),
                r.rejection_filter, r.entry_proxy, r.stop_proxy, r.target_1, r.target_2,
                r.risk_per_share,
                int(r.hit_1r), int(r.hit_2r), int(r.hit_stop),
                r.max_favorable_excursion_r, r.max_adverse_excursion_r,
                r.time_to_1r_minutes, r.time_to_2r_minutes, r.time_to_stop_minutes,
                r.session_high, r.session_low, r.session_close,
                int(r.rejection_was_correct), int(r.should_investigate),
            )
            for r in results
        ]
        await self._db.execute_many(
            """
            INSERT OR REPLACE INTO missed_trade_audit (
                scan_id, symbol, audit_date, audited_at,
                rejection_filter, entry_proxy, stop_proxy, target_1, target_2,
                risk_per_share,
                hit_1r, hit_2r, hit_stop,
                max_favorable_excursion_r, max_adverse_excursion_r,
                time_to_1r_minutes, time_to_2r_minutes, time_to_stop_minutes,
                session_high, session_low, session_close,
                rejection_was_correct, should_investigate
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            params,
        )

    async def get_for_date(self, audit_date: str) -> list[dict[str, Any]]:
        rows = await self._db.fetch_all(
            "SELECT * FROM missed_trade_audit WHERE audit_date=? ORDER BY symbol",
            (audit_date,),
        )
        return [dict(r) for r in rows]

    async def get_filter_stats(
        self, date_range_start: str, date_range_end: str
    ) -> list[dict[str, Any]]:
        """
        Aggregate stats per rejection filter — key input for filter quality analysis.
        Returns: filter, total_rejected, hit_1r_count, hit_2r_count, correct_rejection_count
        """
        rows = await self._db.fetch_all(
            """
            SELECT
                rejection_filter,
                COUNT(*)                            AS total_rejected,
                SUM(hit_1r)                         AS hit_1r_count,
                SUM(hit_2r)                         AS hit_2r_count,
                SUM(hit_stop)                       AS hit_stop_count,
                SUM(rejection_was_correct)          AS correct_rejections,
                SUM(should_investigate)             AS should_investigate_count,
                AVG(max_favorable_excursion_r)      AS avg_mfe_r,
                AVG(max_adverse_excursion_r)        AS avg_mae_r
            FROM missed_trade_audit
            WHERE audit_date >= ? AND audit_date <= ?
            GROUP BY rejection_filter
            ORDER BY hit_1r_count DESC
            """,
            (date_range_start, date_range_end),
        )
        return [dict(r) for r in rows]

    async def get_should_investigate(
        self, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Return all missed trades flagged for investigation (hit +2R)."""
        rows = await self._db.fetch_all(
            """
            SELECT * FROM missed_trade_audit
            WHERE should_investigate=1
            ORDER BY audit_date DESC, max_favorable_excursion_r DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Scan result repository extension — adds get_rejected_for_date
# ---------------------------------------------------------------------------

class ExtendedScanResultRepository:
    """
    Extends the base ScanResultRepository with methods needed by the auditor.
    Wraps the existing repo rather than replacing it.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    async def get_rejected_for_date(self, date_str: str) -> list[RejectedScanRecord]:
        """
        Fetch all scan results that failed filters for a given date.
        Used by MissedTradeAuditor to know what to audit.
        """
        rows = await self._db.fetch_all(
            """
            SELECT
                id, symbol, scanned_at, raw_json,
                composite_score
            FROM scan_results
            WHERE date(scanned_at) = ?
              AND passes_filters = 0
            ORDER BY composite_score DESC
            """,
            (date_str,),
        )
        results = []
        for row in rows:
            try:
                raw = json.loads(row["raw_json"])
                metrics = raw.get("metrics", {})
                failed_filter = raw.get("filter_failure_reason", "unknown")
                last_price = metrics.get("premarket_last", 0.0) or metrics.get("last_price", 0.0)
                results.append(
                    RejectedScanRecord(
                        scan_id=row["id"],
                        symbol=row["symbol"],
                        scan_date=date_str,
                        scanned_at=datetime.fromisoformat(row["scanned_at"]),
                        failed_filter=failed_filter or "unknown",
                        filter_value=None,
                        required_value=None,
                        composite_score=row["composite_score"],
                        last_price=last_price,
                    )
                )
            except Exception:
                continue
        return results


# ---------------------------------------------------------------------------
# Percentile distributions repository
# ---------------------------------------------------------------------------

class PercentileDistributionRepository:
    """Persist and load rolling percentile quantile arrays for scoring."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    async def save(
        self,
        feature: str,
        quantiles: FeatureQuantiles,
        date_range_start: str,
        date_range_end: str,
    ) -> None:
        quantile_list = quantiles.quantiles.tolist()
        await self._db.execute(
            """
            INSERT INTO percentile_distributions
                (feature, computed_at, sample_count, quantiles_json, date_range_start, date_range_end)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                feature,
                quantiles.computed_at,
                quantiles.sample_count,
                json.dumps(quantile_list),
                date_range_start,
                date_range_end,
            ),
        )

    async def load_latest(self, features: list[str]) -> list[FeatureQuantiles]:
        """Load the most recent quantile distribution for each feature."""
        results = []
        for feature in features:
            row = await self._db.fetch_one(
                """
                SELECT * FROM percentile_distributions
                WHERE feature=?
                ORDER BY computed_at DESC
                LIMIT 1
                """,
                (feature,),
            )
            if row is None:
                continue
            try:
                quantile_arr = np.array(json.loads(row["quantiles_json"]), dtype=float)
                results.append(
                    FeatureQuantiles(
                        feature=row["feature"],
                        quantiles=quantile_arr,
                        sample_count=row["sample_count"],
                        computed_at=row["computed_at"],
                    )
                )
            except Exception:
                continue
        return results

    async def compute_and_save_from_scans(
        self,
        db: DatabaseManager,
        lookback_days: int = 30,
    ) -> list[FeatureQuantiles]:
        """
        Compute quantiles from the last N days of scan data and persist.
        Called EOD to keep the percentile distributions fresh.
        """
        from scanner.percentile_scoring import build_quantiles_from_samples, FEATURE_NAMES
        from datetime import date, timedelta

        end_date = date.today().isoformat()
        start_date = (date.today() - timedelta(days=lookback_days)).isoformat()

        rows = await db.fetch_all(
            """
            SELECT raw_json FROM scan_results
            WHERE date(scanned_at) >= ? AND date(scanned_at) <= ?
              AND passes_filters = 1
            """,
            (start_date, end_date),
        )

        # Extract feature values from raw JSON
        feature_values: dict[str, list[float]] = {f: [] for f in FEATURE_NAMES}

        for row in rows:
            try:
                raw = json.loads(row["raw_json"])
                m = raw.get("metrics", {})
                float_rot = m.get("pm_float_rotation_pct") or 0.0
                spread_inv = max(0.0, 1.0 - (m.get("spread_pct", 0.0) or 0.0) / 0.5)
                feature_values["trend_quality"].append(m.get("trend_quality", 0.0) or 0.0)
                feature_values["relative_volume"].append(m.get("relative_volume", 0.0) or 0.0)
                feature_values["float_rotation_pct"].append(float_rot)
                feature_values["gap_pct"].append(m.get("gap_pct", 0.0) or 0.0)
                feature_values["premarket_dollar_volume"].append(m.get("premarket_dollar_volume", 0.0) or 0.0)
                feature_values["spread_inverse"].append(spread_inv)
            except Exception:
                continue

        quantiles = []
        for feature in FEATURE_NAMES:
            q = build_quantiles_from_samples(feature, feature_values[feature])
            await self.save(q.feature, q, start_date, end_date)
            quantiles.append(q)

        return quantiles
