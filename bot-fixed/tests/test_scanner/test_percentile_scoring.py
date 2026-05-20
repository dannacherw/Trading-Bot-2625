"""
tests/test_scanner/test_percentile_scoring.py
Tests for the percentile-rank scoring system.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pytest

from core.config import ScoringWeights
from core.models import PremarketMetrics, ScanResult
from scanner.percentile_scoring import (
    FeatureQuantiles,
    NormalisedMetrics,
    QuantileStore,
    _cross_sectional_rank,
    _fallback_norm,
    _percentile_rank,
    build_quantiles_from_samples,
    compute_composite_score,
    load_quantile_store,
    normalise_percentile_cross_sectional,
    normalise_percentile_historical,
    rank_scan_results,
    score_population_percentile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metrics(
    symbol: str = "AAPL",
    gap_pct: float = 7.0,
    rel_vol: float = 8.0,
    spread_pct: float = 0.08,
    trend_quality: float = 0.80,
    float_rot: float | None = 3.0,
    pm_dollar_vol: float = 5_000_000.0,
) -> PremarketMetrics:
    return PremarketMetrics(
        symbol=symbol,
        computed_at=datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc),
        prev_close=95.0,
        premarket_open=97.0,
        premarket_high=102.0,
        premarket_low=96.5,
        premarket_last=101.5,
        premarket_volume=500_000,
        premarket_dollar_volume=pm_dollar_vol,
        gap_pct=gap_pct,
        relative_volume=rel_vol,
        spread_pct=spread_pct,
        range_pct=5.0,
        range_position=0.90,
        trend_quality=trend_quality,
        avg_daily_dollar_volume=30_000_000,
        pm_float_rotation_pct=float_rot,
    )


def _make_scan_result(
    symbol: str = "AAPL",
    passes: bool = True,
    score: float = 0.75,
    **kwargs,
) -> ScanResult:
    return ScanResult(
        symbol=symbol,
        scanned_at=datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc),
        metrics=_make_metrics(symbol=symbol, **kwargs),
        composite_score=score,
        passes_filters=passes,
        filter_failure_reason=None if passes else "test filter",
    )


# ---------------------------------------------------------------------------
# _percentile_rank tests
# ---------------------------------------------------------------------------

class TestPercentileRank:
    def test_value_at_median(self):
        quantiles = np.linspace(0, 100, 101)  # 0, 1, 2, ..., 100
        rank = _percentile_rank(50.0, quantiles)
        assert abs(rank - 0.5) < 0.02

    def test_value_below_min_returns_zero(self):
        quantiles = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        assert _percentile_rank(5.0, quantiles) == 0.0

    def test_value_above_max_returns_one(self):
        quantiles = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        assert _percentile_rank(100.0, quantiles) == 1.0

    def test_interpolation(self):
        quantiles = np.array([0.0, 10.0, 20.0])
        rank = _percentile_rank(5.0, quantiles)
        assert 0.2 < rank < 0.6

    def test_empty_quantiles(self):
        assert _percentile_rank(5.0, np.array([])) == 0.5

    def test_result_bounded_0_1(self):
        quantiles = np.linspace(0, 50, 101)
        for v in [-100, 0, 25, 50, 200]:
            r = _percentile_rank(float(v), quantiles)
            assert 0.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# _cross_sectional_rank tests
# ---------------------------------------------------------------------------

class TestCrossSectionalRank:
    def test_highest_value_ranks_near_one(self):
        pop = [1.0, 2.0, 3.0, 4.0, 5.0]
        rank = _cross_sectional_rank(5.0, pop)
        assert rank > 0.8

    def test_lowest_value_ranks_near_zero(self):
        pop = [1.0, 2.0, 3.0, 4.0, 5.0]
        rank = _cross_sectional_rank(1.0, pop)
        assert rank < 0.2

    def test_middle_value_near_half(self):
        pop = [1.0, 2.0, 3.0, 4.0, 5.0]
        rank = _cross_sectional_rank(3.0, pop)
        assert 0.4 <= rank <= 0.6

    def test_empty_population_returns_half(self):
        assert _cross_sectional_rank(5.0, []) == 0.5

    def test_all_same_values(self):
        pop = [5.0, 5.0, 5.0]
        rank = _cross_sectional_rank(5.0, pop)
        assert 0.0 <= rank <= 1.0


# ---------------------------------------------------------------------------
# build_quantiles_from_samples tests
# ---------------------------------------------------------------------------

class TestBuildQuantiles:
    def test_returns_101_quantiles(self):
        values = list(range(100))
        q = build_quantiles_from_samples("gap_pct", values)
        assert len(q.quantiles) == 101

    def test_sample_count(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        q = build_quantiles_from_samples("rv", values)
        assert q.sample_count == 5

    def test_empty_values_returns_linear_fallback(self):
        q = build_quantiles_from_samples("feature", [])
        assert q.sample_count == 0
        assert len(q.quantiles) == 101

    def test_quantiles_are_monotonic(self):
        values = list(range(1000))
        q = build_quantiles_from_samples("feature", values)
        diffs = np.diff(q.quantiles)
        assert np.all(diffs >= 0)

    def test_feature_name_stored(self):
        q = build_quantiles_from_samples("trend_quality", [0.5, 0.7, 0.9])
        assert q.feature == "trend_quality"


# ---------------------------------------------------------------------------
# QuantileStore tests
# ---------------------------------------------------------------------------

class TestQuantileStore:
    def test_initially_not_fresh(self):
        store = QuantileStore()
        assert not store.is_fresh()

    def test_fresh_after_load(self):
        store = QuantileStore()
        q = build_quantiles_from_samples("gap_pct", [1.0, 2.0, 3.0])
        store.load([q])
        assert store.is_fresh()

    def test_get_loaded_feature(self):
        store = QuantileStore()
        q = build_quantiles_from_samples("trend_quality", [0.5, 0.6, 0.7])
        store.load([q])
        result = store.get("trend_quality")
        assert result is not None
        assert result.feature == "trend_quality"

    def test_get_unknown_feature_returns_none(self):
        store = QuantileStore()
        assert store.get("nonexistent") is None

    def test_has_all_features(self):
        store = QuantileStore()
        q1 = build_quantiles_from_samples("gap_pct", [1.0])
        q2 = build_quantiles_from_samples("trend_quality", [1.0])
        store.load([q1, q2])
        assert store.has_all_features(["gap_pct", "trend_quality"])
        assert not store.has_all_features(["gap_pct", "missing"])


# ---------------------------------------------------------------------------
# normalise_percentile_cross_sectional tests
# ---------------------------------------------------------------------------

class TestNormaliseCrossSectional:
    def test_highest_rv_gets_high_rank(self):
        best = _make_metrics(rel_vol=20.0)
        others = [_make_metrics(rel_vol=v) for v in [2.0, 5.0, 8.0]]
        population = others + [best]
        result = normalise_percentile_cross_sectional(best, 0.5, population)
        assert result.relative_volume > 0.7

    def test_lowest_spread_inverse_gets_low_rank(self):
        worst = _make_metrics(spread_pct=0.45)  # High spread → low spread_inverse
        others = [_make_metrics(spread_pct=v) for v in [0.05, 0.08, 0.10]]
        population = others + [worst]
        result = normalise_percentile_cross_sectional(worst, 0.0, population)
        assert result.spread_inverse < 0.3

    def test_all_scores_in_0_1(self):
        metrics = _make_metrics()
        population = [_make_metrics(rel_vol=float(v)) for v in range(2, 22)]
        result = normalise_percentile_cross_sectional(metrics, 0.7, population)
        for field in ["trend_quality", "relative_volume", "float_rotation_pct",
                      "gap_pct", "catalyst_score", "premarket_dollar_volume", "spread_inverse"]:
            val = getattr(result, field)
            assert 0.0 <= val <= 1.0, f"{field}={val} out of range"

    def test_none_float_rotation_treated_as_zero(self):
        metrics = _make_metrics(float_rot=None)
        population = [_make_metrics(float_rot=3.0), _make_metrics(float_rot=5.0)]
        result = normalise_percentile_cross_sectional(metrics, 0.0, population)
        assert 0.0 <= result.float_rotation_pct <= 1.0


# ---------------------------------------------------------------------------
# compute_composite_score tests
# ---------------------------------------------------------------------------

class TestCompositeScore:
    def test_perfect_metrics_near_one(self):
        normed = NormalisedMetrics(
            trend_quality=1.0, relative_volume=1.0, float_rotation_pct=1.0,
            gap_pct=1.0, catalyst_score=1.0, premarket_dollar_volume=1.0, spread_inverse=1.0,
        )
        score = compute_composite_score(normed, ScoringWeights())
        assert score > 0.95

    def test_zero_metrics_near_zero(self):
        normed = NormalisedMetrics(
            trend_quality=0.0, relative_volume=0.0, float_rotation_pct=0.0,
            gap_pct=0.0, catalyst_score=0.0, premarket_dollar_volume=0.0, spread_inverse=0.0,
        )
        score = compute_composite_score(normed, ScoringWeights())
        assert score < 0.05

    def test_no_float_redistributes_weight(self):
        with_float = NormalisedMetrics(
            trend_quality=0.8, relative_volume=0.8, float_rotation_pct=0.8,
            gap_pct=0.8, catalyst_score=0.8, premarket_dollar_volume=0.8, spread_inverse=0.8,
        )
        without_float = NormalisedMetrics(
            trend_quality=0.8, relative_volume=0.8, float_rotation_pct=0.0,
            gap_pct=0.8, catalyst_score=0.8, premarket_dollar_volume=0.8, spread_inverse=0.8,
        )
        w = ScoringWeights()
        s1 = compute_composite_score(with_float, w)
        s2 = compute_composite_score(without_float, w)
        # Both should be near 0.8 — weight redistribution keeps score stable
        assert abs(s1 - 0.8) < 0.10
        assert abs(s2 - 0.8) < 0.10

    def test_result_always_bounded(self):
        import random
        w = ScoringWeights()
        for _ in range(50):
            normed = NormalisedMetrics(
                trend_quality=random.random(),
                relative_volume=random.random(),
                float_rotation_pct=random.random(),
                gap_pct=random.random(),
                catalyst_score=random.random(),
                premarket_dollar_volume=random.random(),
                spread_inverse=random.random(),
            )
            score = compute_composite_score(normed, w)
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# score_population_percentile tests
# ---------------------------------------------------------------------------

class TestScorePopulation:
    def test_failing_results_unchanged(self):
        failing = _make_scan_result("GME", passes=False, score=0.0)
        passing = _make_scan_result("AAPL", passes=True, score=0.0)
        results = score_population_percentile(
            [failing, passing],
            {"AAPL": 0.8, "GME": 0.0},
            ScoringWeights(),
            use_historical=False,
        )
        gme = next(r for r in results if r.symbol == "GME")
        assert gme.composite_score == 0.0

    def test_passing_results_get_scores(self):
        results = [_make_scan_result(s, passes=True, score=0.0) for s in ["AAPL", "NVDA", "TSLA"]]
        catalyst_scores = {"AAPL": 0.9, "NVDA": 0.5, "TSLA": 0.1}
        scored = score_population_percentile(
            results, catalyst_scores, ScoringWeights(), use_historical=False
        )
        for r in scored:
            if r.passes_filters:
                assert r.composite_score > 0.0

    def test_empty_population_returns_unchanged(self):
        results = [_make_scan_result("GME", passes=False, score=0.3)]
        scored = score_population_percentile(results, {}, ScoringWeights(), use_historical=False)
        assert scored[0].composite_score == 0.3


# ---------------------------------------------------------------------------
# rank_scan_results tests
# ---------------------------------------------------------------------------

class TestRankScanResults:
    def test_passing_sorted_by_score_descending(self):
        results = [
            _make_scan_result("A", passes=True, score=0.5),
            _make_scan_result("B", passes=True, score=0.9),
            _make_scan_result("C", passes=True, score=0.1),
        ]
        ranked = rank_scan_results(results)
        scores = [r.composite_score for r in ranked if r.passes_filters]
        assert scores == sorted(scores, reverse=True)

    def test_passing_before_failing(self):
        results = [
            _make_scan_result("FAIL", passes=False, score=0.99),
            _make_scan_result("PASS", passes=True, score=0.01),
        ]
        ranked = rank_scan_results(results)
        assert ranked[0].symbol == "PASS"
