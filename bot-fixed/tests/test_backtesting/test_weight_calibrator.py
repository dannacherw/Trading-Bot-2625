"""
tests/test_backtesting/test_weight_calibrator.py
Tests for the logistic regression scoring weight calibrator.

Covers:
  - Calibration with valid data produces weights that sum to 1.0
  - All calibrated weights are non-negative (constraint enforced)
  - Calibration fails gracefully below minimum sample threshold
  - Feature importances match the feature names
  - Export to ScoringWeights dataclass produces valid output
  - Edge cases: all winners, all losers, borderline sample size
"""
from __future__ import annotations

import pytest

from backtesting.weight_calibrator import (
    CalibrationDataPoint,
    CalibrationResult,
    ScoringWeightCalibrator,
)
from core.config import ScoringWeights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_point(
    followed_through: bool,
    trend_quality: float = 0.7,
    relative_volume: float = 0.6,
    float_rotation_pct: float = 0.5,
    gap_pct: float = 0.4,
    catalyst_score: float = 0.8,
    premarket_dollar_volume: float = 0.5,
    spread_inverse: float = 0.9,
    symbol: str = "AAPL",
) -> CalibrationDataPoint:
    return CalibrationDataPoint(
        symbol=symbol,
        date="2024-01-15",
        trend_quality=trend_quality,
        relative_volume=relative_volume,
        float_rotation_pct=float_rotation_pct,
        gap_pct=gap_pct,
        catalyst_score=catalyst_score,
        premarket_dollar_volume=premarket_dollar_volume,
        spread_inverse=spread_inverse,
        followed_through=followed_through,
    )


def _make_dataset(
    n_winners: int, n_losers: int
) -> list[CalibrationDataPoint]:
    """Build a simple balanced dataset."""
    data = []
    for i in range(n_winners):
        data.append(_make_point(
            followed_through=True,
            trend_quality=0.8 + (i % 3) * 0.05,
            gap_pct=0.6 + (i % 4) * 0.05,
            symbol=f"WIN{i:03d}",
        ))
    for i in range(n_losers):
        data.append(_make_point(
            followed_through=False,
            trend_quality=0.3 + (i % 3) * 0.05,
            gap_pct=0.2 + (i % 4) * 0.05,
            symbol=f"LOSE{i:03d}",
        ))
    return data


# ---------------------------------------------------------------------------
# Calibration core
# ---------------------------------------------------------------------------

class TestCalibrationWithSufficientData:
    def test_returns_calibration_result(self):
        cal = ScoringWeightCalibrator(min_samples=50)
        data = _make_dataset(40, 30)
        result = cal.calibrate(data)
        assert isinstance(result, CalibrationResult)

    def test_weights_sum_to_one(self):
        cal = ScoringWeightCalibrator(min_samples=50)
        data = _make_dataset(40, 30)
        result = cal.calibrate(data)
        w = result.weights
        total = (
            w.trend_quality + w.relative_volume + w.float_rotation_pct
            + w.gap_pct + w.catalyst_score + w.premarket_dollar_volume
            + w.spread_inverse
        )
        assert total == pytest.approx(1.0, abs=0.02)

    def test_all_weights_non_negative(self):
        cal = ScoringWeightCalibrator(min_samples=50)
        data = _make_dataset(40, 30)
        result = cal.calibrate(data)
        w = result.weights
        assert w.trend_quality >= 0.0
        assert w.relative_volume >= 0.0
        assert w.float_rotation_pct >= 0.0
        assert w.gap_pct >= 0.0
        assert w.catalyst_score >= 0.0
        assert w.premarket_dollar_volume >= 0.0
        assert w.spread_inverse >= 0.0

    def test_n_samples_reported_correctly(self):
        cal = ScoringWeightCalibrator(min_samples=50)
        data = _make_dataset(35, 30)
        result = cal.calibrate(data)
        assert result.n_samples == 65

    def test_train_accuracy_in_valid_range(self):
        cal = ScoringWeightCalibrator(min_samples=50)
        data = _make_dataset(40, 30)
        result = cal.calibrate(data)
        assert 0.0 <= result.train_accuracy <= 1.0

    def test_log_loss_positive(self):
        cal = ScoringWeightCalibrator(min_samples=50)
        data = _make_dataset(40, 30)
        result = cal.calibrate(data)
        assert result.log_loss >= 0.0

    def test_feature_importances_has_all_features(self):
        cal = ScoringWeightCalibrator(min_samples=50)
        data = _make_dataset(40, 30)
        result = cal.calibrate(data)
        expected_features = {
            "trend_quality", "relative_volume", "float_rotation_pct",
            "gap_pct", "catalyst_score", "premarket_dollar_volume", "spread_inverse",
        }
        assert set(result.feature_importances.keys()) == expected_features

    def test_result_weights_are_scoring_weights_instance(self):
        cal = ScoringWeightCalibrator(min_samples=50)
        data = _make_dataset(40, 30)
        result = cal.calibrate(data)
        assert isinstance(result.weights, ScoringWeights)


class TestCalibrationBelowMinimumSamples:
    def test_raises_when_below_min_samples(self):
        cal = ScoringWeightCalibrator(min_samples=50)
        data = _make_dataset(20, 20)  # Only 40 samples, below 50
        with pytest.raises(Exception):
            cal.calibrate(data)

    def test_passes_at_exact_minimum(self):
        cal = ScoringWeightCalibrator(min_samples=50)
        data = _make_dataset(30, 20)  # Exactly 50 samples
        result = cal.calibrate(data)
        assert result.n_samples == 50


class TestCalibrationEdgeCases:
    def test_weights_remain_valid_with_float_missing(self):
        """float_rotation_pct=0.0 for all points (data unavailable scenario)."""
        cal = ScoringWeightCalibrator(min_samples=50)
        data = [
            _make_point(followed_through=(i % 2 == 0), float_rotation_pct=0.0)
            for i in range(60)
        ]
        result = cal.calibrate(data)
        total = (
            result.weights.trend_quality + result.weights.relative_volume
            + result.weights.float_rotation_pct + result.weights.gap_pct
            + result.weights.catalyst_score + result.weights.premarket_dollar_volume
            + result.weights.spread_inverse
        )
        assert total == pytest.approx(1.0, abs=0.02)

    def test_high_win_rate_dataset(self):
        """Calibration with >80% winners still produces valid weights."""
        cal = ScoringWeightCalibrator(min_samples=50)
        data = _make_dataset(n_winners=55, n_losers=10)
        result = cal.calibrate(data)
        assert result.n_samples == 65
        assert 0.0 <= result.train_accuracy <= 1.0

    def test_calibrated_weights_differ_from_heuristic_defaults(self):
        """
        After fitting on intentionally skewed data (trend_quality always high
        for winners), the fitted weight for trend_quality should be larger than
        the uniform 1/7 ≈ 0.143 baseline.
        This validates the optimiser is actually learning.
        """
        cal = ScoringWeightCalibrator(min_samples=50)
        # Winners always have high trend_quality, losers have low
        data = []
        for i in range(45):
            data.append(_make_point(followed_through=True, trend_quality=0.95,
                                    relative_volume=0.5, gap_pct=0.5,
                                    catalyst_score=0.5, float_rotation_pct=0.5,
                                    premarket_dollar_volume=0.5, spread_inverse=0.5))
        for i in range(25):
            data.append(_make_point(followed_through=False, trend_quality=0.05,
                                    relative_volume=0.5, gap_pct=0.5,
                                    catalyst_score=0.5, float_rotation_pct=0.5,
                                    premarket_dollar_volume=0.5, spread_inverse=0.5))
        result = cal.calibrate(data)
        # After fitting, trend_quality weight should be above baseline (1/7 ≈ 0.143)
        assert result.weights.trend_quality > 0.143


class TestWeightExport:
    def test_export_returns_scoring_weights(self):
        cal = ScoringWeightCalibrator(min_samples=50)
        data = _make_dataset(35, 25)
        result = cal.calibrate(data)
        assert isinstance(result.weights, ScoringWeights)

    def test_export_to_yaml_writes_file(self, tmp_path):
        cal = ScoringWeightCalibrator(min_samples=50)
        data = _make_dataset(35, 25)
        result = cal.calibrate(data)
        yaml_path = str(tmp_path / "scanner_config.yaml")

        # Create minimal YAML so export has something to update
        import yaml
        (tmp_path / "scanner_config.yaml").write_text(
            yaml.dump({"scoring_weights": {
                "trend_quality": 0.22,
                "relative_volume": 0.20,
                "float_rotation_pct": 0.14,
                "gap_pct": 0.14,
                "catalyst_score": 0.12,
                "premarket_dollar_volume": 0.12,
                "spread_inverse": 0.06,
            }})
        )
        ScoringWeightCalibrator.export_to_yaml(result, yaml_path)

        updated = yaml.safe_load(open(yaml_path).read())
        weights_section = updated.get("scoring_weights", {})
        total = sum(weights_section.values())
        assert total == pytest.approx(1.0, abs=0.02)
