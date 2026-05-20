"""
tests/test_scanner/test_scoring.py — v2
Tests for the updated composite score formula with float rotation and new weights.
"""
from __future__ import annotations

import pytest

from core.config import ScoringWeights
from scanner.scoring import (
    NormalisedMetrics,
    compute_composite_score,
    normalise_metrics,
)


def _normed(**kwargs) -> NormalisedMetrics:
    defaults = dict(
        trend_quality=0.8,
        relative_volume=0.7,
        float_rotation_pct=0.6,
        gap_pct=0.5,
        catalyst_score=0.9,
        premarket_dollar_volume=0.6,
        spread_inverse=0.8,
    )
    defaults.update(kwargs)
    return NormalisedMetrics(**defaults)


class TestNormalisedMetricsHasFloatRotation:
    def test_float_rotation_field_exists(self):
        n = _normed()
        assert hasattr(n, "float_rotation_pct")

    def test_no_range_position_field(self):
        """range_position was removed in v2 — verify it's gone."""
        n = _normed()
        assert not hasattr(n, "range_position")


class TestCompositeScore:
    def test_score_in_range(self):
        n = _normed()
        w = ScoringWeights()
        score = compute_composite_score(n, w)
        assert 0.0 <= score <= 1.0

    def test_all_zeros_scores_zero(self):
        n = _normed(**{k: 0.0 for k in NormalisedMetrics.__dataclass_fields__})
        score = compute_composite_score(n, ScoringWeights())
        assert score == pytest.approx(0.0, abs=0.01)

    def test_all_ones_scores_one(self):
        n = _normed(**{k: 1.0 for k in NormalisedMetrics.__dataclass_fields__})
        score = compute_composite_score(n, ScoringWeights())
        assert score == pytest.approx(1.0, abs=0.01)

    def test_float_data_unknown_redistributes_weight(self):
        """With float_rotation_pct=0.0, score should still be valid and comparable."""
        with_float = _normed(float_rotation_pct=0.8)
        no_float = _normed(float_rotation_pct=0.0)
        w = ScoringWeights()
        score_with = compute_composite_score(with_float, w)
        score_without = compute_composite_score(no_float, w)
        # Both should be in [0, 1] — no NaN or out-of-range
        assert 0.0 <= score_with <= 1.0
        assert 0.0 <= score_without <= 1.0
        # Float adds value — with float should score higher
        assert score_with > score_without

    def test_trend_quality_has_highest_weight(self):
        """trend_quality is now the highest-weighted feature (0.22)."""
        w = ScoringWeights()
        assert w.trend_quality == 0.22
        assert w.trend_quality > w.gap_pct   # Trend quality > gap pct

    def test_gap_pct_lower_than_v1(self):
        """gap_pct reduced from 0.24 to 0.14 in v2."""
        w = ScoringWeights()
        assert w.gap_pct == pytest.approx(0.14, abs=0.01)

    def test_weights_sum_to_one(self):
        w = ScoringWeights()
        total = (
            w.trend_quality + w.relative_volume + w.float_rotation_pct
            + w.gap_pct + w.catalyst_score + w.premarket_dollar_volume
            + w.spread_inverse
        )
        assert total == pytest.approx(1.0, abs=0.01)


class TestNormaliseMetrics:
    def test_fixed_range_normalise_outputs_valid_range(self, sample_metrics):
        normed = normalise_metrics(sample_metrics, catalyst_score=0.80)
        for field_name in NormalisedMetrics.__dataclass_fields__:
            val = getattr(normed, field_name)
            assert 0.0 <= val <= 1.0, f"{field_name}={val} out of range"

    def test_high_float_rotation_boosts_score(self, sample_metrics):
        low_rot = sample_metrics.model_copy(update={"pm_float_rotation_pct": 0.5})
        high_rot = sample_metrics.model_copy(update={"pm_float_rotation_pct": 8.0})
        n_low = normalise_metrics(low_rot, 0.7)
        n_high = normalise_metrics(high_rot, 0.7)
        assert n_high.float_rotation_pct > n_low.float_rotation_pct

    def test_unknown_float_normalises_to_zero(self, sample_metrics):
        no_float = sample_metrics.model_copy(update={"pm_float_rotation_pct": None})
        normed = normalise_metrics(no_float, 0.7)
        assert normed.float_rotation_pct == 0.0
