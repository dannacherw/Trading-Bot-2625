"""
tests/test_scanner/test_filters.py — v2
Tests for updated premarket filter predicates (tiered gap, float rotation).
"""
from __future__ import annotations

import pytest

from core.config import PremarketFilterSettings
from core.enums import CatalystCategory
from core.models import Catalyst
from scanner.filters import (
    apply_all_filters,
    check_float_rotation,
    check_min_gap,
    check_min_premarket_volume,
    check_premarket_range,
    check_range_position,
    check_relative_volume,
    check_spread,
    get_filter_summary,
)
from datetime import datetime, timezone


def _make_catalyst(
    category: CatalystCategory = CatalystCategory.EARNINGS,
    confidence: float = 0.90,
) -> Catalyst:
    return Catalyst(
        symbol="AAPL",
        detected_at=datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc),
        category=category,
        confidence=confidence,
        strength=0.80,
    )


@pytest.fixture
def settings() -> PremarketFilterSettings:
    return PremarketFilterSettings()


class TestTieredGapFilter:
    def test_passes_with_strong_catalyst(self, settings, sample_metrics):
        """A 4% gap with a high-confidence earnings catalyst should pass."""
        low_gap = sample_metrics.model_copy(update={"gap_pct": 4.0})
        catalyst = _make_catalyst(CatalystCategory.EARNINGS, confidence=0.90)
        ok, reason = check_min_gap(low_gap, settings, catalyst)
        assert ok is True, f"Expected pass, got: {reason}"

    def test_fails_without_catalyst_at_low_gap(self, settings, sample_metrics):
        """A 4% gap without a catalyst should fail (no_catalyst floor = 6%)."""
        low_gap = sample_metrics.model_copy(update={"gap_pct": 4.0})
        ok, reason = check_min_gap(low_gap, settings, catalyst=None)
        assert ok is False
        assert "no catalyst" in reason.lower() or "6" in reason

    def test_fails_with_low_confidence_catalyst(self, settings, sample_metrics):
        """A 4% gap with low-confidence catalyst (below 0.80) should fail."""
        low_gap = sample_metrics.model_copy(update={"gap_pct": 4.0})
        weak_cat = _make_catalyst(CatalystCategory.EARNINGS, confidence=0.50)
        ok, reason = check_min_gap(low_gap, settings, weak_cat)
        assert ok is False

    def test_fails_with_press_release_catalyst(self, settings, sample_metrics):
        """Press release is not in HIGH_QUALITY_CATALYST_CATEGORIES."""
        low_gap = sample_metrics.model_copy(update={"gap_pct": 4.0})
        cat = _make_catalyst(CatalystCategory.PRESS_RELEASE, confidence=0.95)
        ok, reason = check_min_gap(low_gap, settings, cat)
        assert ok is False

    def test_passes_high_gap_no_catalyst(self, settings, sample_metrics):
        """A 7% gap without catalyst passes the 6% no-catalyst floor."""
        high_gap = sample_metrics.model_copy(update={"gap_pct": 7.5})
        ok, _ = check_min_gap(high_gap, settings, catalyst=None)
        assert ok is True

    def test_fda_catalyst_qualifies(self, settings, sample_metrics):
        low_gap = sample_metrics.model_copy(update={"gap_pct": 3.5})
        cat = _make_catalyst(CatalystCategory.FDA_OR_BIOTECH_NEWS, 0.92)
        ok, _ = check_min_gap(low_gap, settings, cat)
        assert ok is True

    def test_ma_catalyst_qualifies(self, settings, sample_metrics):
        low_gap = sample_metrics.model_copy(update={"gap_pct": 3.5})
        cat = _make_catalyst(CatalystCategory.M_AND_A, 0.88)
        ok, _ = check_min_gap(low_gap, settings, cat)
        assert ok is True


class TestFloatRotationFilter:
    def test_passes_when_filter_disabled(self, sample_metrics):
        s = PremarketFilterSettings(use_float_rotation_filter=False)
        ok, _ = check_float_rotation(sample_metrics, s)
        assert ok is True

    def test_passes_when_float_unknown(self, settings, sample_metrics):
        """Float data unavailable → allow through per user decision."""
        no_float = sample_metrics.model_copy(update={"pm_float_rotation_pct": None})
        ok, _ = check_float_rotation(no_float, settings)
        assert ok is True

    def test_fails_when_rotation_too_low(self, settings, sample_metrics):
        low_rot = sample_metrics.model_copy(update={"pm_float_rotation_pct": 0.5})
        ok, reason = check_float_rotation(low_rot, settings)
        assert ok is False
        assert "float rotation" in reason.lower()

    def test_passes_when_rotation_adequate(self, settings, sample_metrics):
        good_rot = sample_metrics.model_copy(update={"pm_float_rotation_pct": 3.5})
        ok, _ = check_float_rotation(good_rot, settings)
        assert ok is True


class TestApplyAllFilters:
    def test_all_pass_with_catalyst(self, settings, sample_metrics):
        cat = _make_catalyst(CatalystCategory.EARNINGS, 0.90)
        ok, reason = apply_all_filters(sample_metrics, settings, cat)
        assert ok is True

    def test_first_failure_short_circuits(self, settings, sample_metrics):
        failing = sample_metrics.model_copy(update={"gap_pct": 0.5})
        ok, reason = apply_all_filters(failing, settings)
        assert ok is False
        assert reason is not None

    def test_filter_summary_has_all_keys(self, settings, sample_metrics):
        summary = get_filter_summary(sample_metrics, settings)
        assert "check_min_gap" in summary
        assert "check_float_rotation" in summary

    def test_backward_compatible_no_catalyst(self, settings, sample_metrics):
        """apply_all_filters with catalyst=None should not raise."""
        high_gap = sample_metrics.model_copy(update={"gap_pct": 8.0})
        ok, _ = apply_all_filters(high_gap, settings, catalyst=None)
        assert isinstance(ok, bool)
