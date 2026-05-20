"""
tests/test_scanner/test_watchlist.py
Tests for the Watchlist builder.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.config import WatchlistSettings
from core.models import ScanResult
from scanner.watchlist import Watchlist


def _make_scan_result(
    symbol: str,
    score: float,
    passes: bool,
    sample_metrics,
    scan_date: datetime,
) -> ScanResult:
    return ScanResult(
        symbol=symbol,
        scanned_at=scan_date,
        metrics=sample_metrics,
        composite_score=score,
        passes_filters=passes,
    )


class TestWatchlist:
    @pytest.fixture
    def settings(self) -> WatchlistSettings:
        return WatchlistSettings(
            max_focus_list_size=3,
            max_watchlist_size=10,
            min_score_threshold=0.50,
        )

    @pytest.fixture
    def scan_results(self, sample_metrics, scan_date) -> list[ScanResult]:
        return [
            _make_scan_result("AAA", 0.90, True, sample_metrics, scan_date),
            _make_scan_result("BBB", 0.75, True, sample_metrics, scan_date),
            _make_scan_result("CCC", 0.60, True, sample_metrics, scan_date),
            _make_scan_result("DDD", 0.40, True, sample_metrics, scan_date),  # Below threshold
            _make_scan_result("EEE", 0.20, False, sample_metrics, scan_date),  # Filtered out
        ]

    def test_builds_watchlist(self, settings, scan_results):
        wl = Watchlist(settings)
        # Sort by score desc before building
        sorted_results = sorted(scan_results, key=lambda r: r.composite_score, reverse=True)
        wl.build(sorted_results)
        assert len(wl) == 4  # EEE fails filter, not included
        assert wl.symbols[0] == "AAA"  # Highest score first

    def test_focus_list_respects_threshold(self, settings, scan_results):
        wl = Watchlist(settings)
        sorted_results = sorted(scan_results, key=lambda r: r.composite_score, reverse=True)
        wl.build(sorted_results)
        focus = wl.focus_list
        # Focus list should only include symbols above 0.50 threshold
        for item in focus:
            assert item.scan_result.composite_score >= settings.min_score_threshold

    def test_focus_list_capped_at_max(self, settings, scan_results):
        wl = Watchlist(settings)
        sorted_results = sorted(scan_results, key=lambda r: r.composite_score, reverse=True)
        wl.build(sorted_results)
        assert len(wl.focus_list) <= settings.max_focus_list_size

    def test_is_focus_correct(self, settings, scan_results):
        wl = Watchlist(settings)
        sorted_results = sorted(scan_results, key=lambda r: r.composite_score, reverse=True)
        wl.build(sorted_results)
        # "AAA" at 0.90 should be in focus
        assert wl.is_focus("AAA") is True
        # "DDD" at 0.40 should not be in focus (below threshold)
        assert wl.is_focus("DDD") is False

    def test_filtered_symbols_not_included(self, settings, scan_results):
        wl = Watchlist(settings)
        sorted_results = sorted(scan_results, key=lambda r: r.composite_score, reverse=True)
        wl.build(sorted_results)
        assert "EEE" not in wl.symbols

    def test_empty_scan_results(self, settings):
        wl = Watchlist(settings)
        wl.build([])
        assert len(wl) == 0
        assert len(wl.focus_list) == 0

    def test_to_table_returns_list_of_dicts(self, settings, scan_results):
        wl = Watchlist(settings)
        sorted_results = sorted(scan_results, key=lambda r: r.composite_score, reverse=True)
        wl.build(sorted_results)
        table = wl.to_table()
        assert isinstance(table, list)
        if table:
            assert isinstance(table[0], dict)
            assert "symbol" in table[0]
            assert "score" in table[0]

    def test_get_returns_correct_item(self, settings, scan_results):
        wl = Watchlist(settings)
        sorted_results = sorted(scan_results, key=lambda r: r.composite_score, reverse=True)
        wl.build(sorted_results)
        item = wl.get("BBB")
        assert item is not None
        assert item.symbol == "BBB"

    def test_get_returns_none_for_unknown(self, settings, scan_results):
        wl = Watchlist(settings)
        wl.build(scan_results)
        assert wl.get("ZZZZZ") is None

    def test_watchlist_max_size_respected(self, settings, sample_metrics, scan_date):
        # Create more results than the max
        results = [
            _make_scan_result(f"SYM{i}", 0.90 - i * 0.05, True, sample_metrics, scan_date)
            for i in range(15)
        ]
        wl = Watchlist(WatchlistSettings(max_watchlist_size=5, max_focus_list_size=3))
        wl.build(results)
        assert len(wl) <= 5
