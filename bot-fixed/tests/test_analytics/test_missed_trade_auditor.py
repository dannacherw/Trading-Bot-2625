"""
tests/test_analytics/test_missed_trade_auditor.py
Tests for the MissedTradeAuditor EOD labeling system.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from analytics.missed_trade_auditor import (
    MissedTradeAuditor,
    MissedTradeAuditResult,
    RejectedScanRecord,
    _compute_hypothetical_levels,
    _label_outcomes,
    _seconds_until_5pm_et,
)
from core.enums import BarTimeframe
from core.models import Bar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bar(
    symbol: str = "GME",
    hour: int = 14,
    minute: int = 0,
    high: float = 102.0,
    low: float = 99.0,
    close: float = 101.0,
    open_: float = 100.0,
) -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=datetime(2024, 1, 15, hour, minute, 0, tzinfo=timezone.utc),
        open=open_, high=high, low=low, close=close,
        volume=50_000,
        timeframe=BarTimeframe.MINUTE_1,
    )


def _rejected_record(
    symbol: str = "GME",
    last_price: float = 100.0,
    failed_filter: str = "spread > max",
) -> RejectedScanRecord:
    return RejectedScanRecord(
        scan_id=1,
        symbol=symbol,
        scan_date="2024-01-15",
        scanned_at=datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc),
        failed_filter=failed_filter,
        filter_value=0.38,
        required_value=0.25,
        composite_score=0.42,
        last_price=last_price,
    )


def _ref_time() -> datetime:
    return datetime(2024, 1, 15, 13, 30, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# _compute_hypothetical_levels
# ---------------------------------------------------------------------------

class TestHypotheticalLevels:
    def test_stop_below_entry(self):
        stop, t1, t2, risk = _compute_hypothetical_levels(100.0, risk_r=0.03)
        assert stop < 100.0
        assert abs(stop - 97.0) < 0.01

    def test_targets_above_entry(self):
        stop, t1, t2, risk = _compute_hypothetical_levels(100.0, risk_r=0.03)
        assert t1 > 100.0
        assert t2 > t1

    def test_t1_is_1_5r(self):
        stop, t1, t2, risk = _compute_hypothetical_levels(100.0, risk_r=0.03, t1_rr=1.5)
        expected_t1 = 100.0 + risk * 1.5
        assert abs(t1 - expected_t1) < 0.001

    def test_t2_is_2_5r(self):
        stop, t1, t2, risk = _compute_hypothetical_levels(100.0, risk_r=0.03, t2_rr=2.5)
        expected_t2 = 100.0 + risk * 2.5
        assert abs(t2 - expected_t2) < 0.001

    def test_risk_per_share_correct(self):
        stop, t1, t2, risk = _compute_hypothetical_levels(100.0, risk_r=0.03)
        assert abs(risk - 3.0) < 0.01


# ---------------------------------------------------------------------------
# _label_outcomes
# ---------------------------------------------------------------------------

class TestLabelOutcomes:
    """Use entry=100, stop=97, t1=104.5, t2=107.5, risk=3."""

    def _params(self):
        entry, risk = 100.0, 3.0
        stop = entry - risk
        t1 = entry + risk * 1.5
        t2 = entry + risk * 2.5
        return entry, stop, t1, t2, risk

    def test_hit_1r(self):
        entry, stop, t1, t2, risk = self._params()
        bars = [_make_bar(high=105.0, low=99.0)]  # High crosses t1=104.5
        outcomes = _label_outcomes(bars, entry, stop, t1, t2, risk, _ref_time())
        assert outcomes["hit_1r"] is True
        assert outcomes["hit_2r"] is False

    def test_hit_2r(self):
        entry, stop, t1, t2, risk = self._params()
        bars = [_make_bar(high=108.0, low=99.0)]  # Crosses both t1 and t2
        outcomes = _label_outcomes(bars, entry, stop, t1, t2, risk, _ref_time())
        assert outcomes["hit_1r"] is True
        assert outcomes["hit_2r"] is True

    def test_hit_stop(self):
        entry, stop, t1, t2, risk = self._params()
        bars = [_make_bar(high=101.0, low=96.0)]  # Low crosses stop=97
        outcomes = _label_outcomes(bars, entry, stop, t1, t2, risk, _ref_time())
        assert outcomes["hit_stop"] is True
        assert outcomes["hit_1r"] is False

    def test_no_hits_flat_session(self):
        entry, stop, t1, t2, risk = self._params()
        bars = [_make_bar(high=101.0, low=98.5)]  # Between stop and t1
        outcomes = _label_outcomes(bars, entry, stop, t1, t2, risk, _ref_time())
        assert not outcomes["hit_1r"]
        assert not outcomes["hit_2r"]
        assert not outcomes["hit_stop"]

    def test_mfe_positive_on_gain(self):
        entry, stop, t1, t2, risk = self._params()
        bars = [_make_bar(high=103.0, low=99.5)]
        outcomes = _label_outcomes(bars, entry, stop, t1, t2, risk, _ref_time())
        assert outcomes["max_favorable_excursion_r"] > 0

    def test_mae_negative_on_drawdown(self):
        entry, stop, t1, t2, risk = self._params()
        bars = [_make_bar(high=100.5, low=98.0)]  # Partial drawdown
        outcomes = _label_outcomes(bars, entry, stop, t1, t2, risk, _ref_time())
        assert outcomes["max_adverse_excursion_r"] < 0

    def test_time_to_1r_recorded(self):
        entry, stop, t1, t2, risk = self._params()
        bar1 = _make_bar(hour=13, minute=30, high=101.0, low=99.5)
        bar2 = _make_bar(hour=13, minute=45, high=105.0, low=99.5)  # Hits t1
        ref = datetime(2024, 1, 15, 13, 30, 0, tzinfo=timezone.utc)
        outcomes = _label_outcomes([bar1, bar2], entry, stop, t1, t2, risk, ref)
        assert outcomes["time_to_1r_minutes"] is not None
        assert outcomes["time_to_1r_minutes"] > 0

    def test_session_close_is_last_bar_close(self):
        entry, stop, t1, t2, risk = self._params()
        bars = [
            _make_bar(hour=13, minute=30, close=101.0),
            _make_bar(hour=13, minute=31, close=102.5),
        ]
        outcomes = _label_outcomes(bars, entry, stop, t1, t2, risk, _ref_time())
        assert outcomes["session_close"] == 102.5


# ---------------------------------------------------------------------------
# MissedTradeAuditor
# ---------------------------------------------------------------------------

class TestMissedTradeAuditor:
    def _make_auditor(self, bars: list[Bar]) -> MissedTradeAuditor:
        provider = AsyncMock()
        provider.get_intraday_bars = AsyncMock(return_value=bars)

        db_repo = AsyncMock()
        db_repo.save_many = AsyncMock()

        scan_repo = AsyncMock()

        return MissedTradeAuditor(
            provider=provider,
            db_repo=db_repo,
            scan_repo=scan_repo,
        )

    @pytest.mark.asyncio
    async def test_run_for_date_empty_rejected(self):
        auditor = self._make_auditor([])
        auditor._scan_repo.get_rejected_for_date = AsyncMock(return_value=[])
        results = await auditor.run_for_date(date(2024, 1, 15))
        assert results == []

    @pytest.mark.asyncio
    async def test_hit_2r_flagged_for_investigation(self):
        # Build bars where stock clearly hits +2R
        entry = 100.0
        risk = 3.0
        bars = [
            _make_bar(hour=13, minute=30, open_=100.0, high=101.0, low=99.5),
            _make_bar(hour=13, minute=31, open_=101.0, high=108.0, low=100.5),  # Hits 2R
        ]
        auditor = self._make_auditor(bars)
        record = _rejected_record(last_price=100.0)

        result = await auditor._compute_outcome(record, date(2024, 1, 15))

        assert result is not None
        assert result.hit_2r is True
        assert result.should_investigate is True

    @pytest.mark.asyncio
    async def test_hit_stop_marked_correct_rejection(self):
        bars = [
            _make_bar(hour=13, minute=30, open_=100.0, high=100.5, low=96.0),  # Hits stop
        ]
        auditor = self._make_auditor(bars)
        record = _rejected_record(last_price=100.0)
        result = await auditor._compute_outcome(record, date(2024, 1, 15))
        assert result is not None
        assert result.hit_stop is True
        assert result.rejection_was_correct is True

    @pytest.mark.asyncio
    async def test_no_bars_returns_none(self):
        auditor = self._make_auditor([])
        record = _rejected_record()
        result = await auditor._compute_outcome(record, date(2024, 1, 15))
        assert result is None

    @pytest.mark.asyncio
    async def test_run_saves_to_repo(self):
        bars = [_make_bar(hour=13, minute=30, high=108.0, low=99.5)]
        auditor = self._make_auditor(bars)
        record = _rejected_record()
        auditor._scan_repo.get_rejected_for_date = AsyncMock(return_value=[record])

        await auditor.run_for_date(date(2024, 1, 15))

        auditor._db_repo.save_many.assert_called_once()

    @pytest.mark.asyncio
    async def test_provider_exception_returns_none(self):
        provider = AsyncMock()
        provider.get_intraday_bars = AsyncMock(side_effect=Exception("API error"))
        auditor = MissedTradeAuditor(
            provider=provider,
            db_repo=AsyncMock(),
            scan_repo=AsyncMock(),
        )
        result = await auditor._compute_outcome(_rejected_record(), date(2024, 1, 15))
        assert result is None


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------

class TestScheduleHelpers:
    def test_5pm_wait_positive_before_5pm(self):
        """Before 5 PM ET, wait time should be positive."""
        before_5pm = datetime(2024, 7, 15, 20, 0, 0, tzinfo=timezone.utc)  # 4 PM EDT
        secs = _seconds_until_5pm_et(before_5pm)
        assert secs > 0

    def test_5pm_wait_zero_after_5pm(self):
        """After 5 PM ET, wait time should be 0."""
        after_5pm = datetime(2024, 7, 15, 22, 0, 0, tzinfo=timezone.utc)  # 6 PM EDT
        secs = _seconds_until_5pm_et(after_5pm)
        assert secs == 0.0

    def test_5pm_wait_approximately_correct(self):
        """At 3 PM EDT (19:00 UTC), should be ~2 hours until 5 PM."""
        three_pm_edt = datetime(2024, 7, 15, 19, 0, 0, tzinfo=timezone.utc)
        secs = _seconds_until_5pm_et(three_pm_edt)
        assert abs(secs - 7200) < 120  # Within 2 minutes of 2 hours
