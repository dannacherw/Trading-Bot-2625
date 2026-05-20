"""
scanner/scoring.py
Composite premarket score computation.
Normalises each metric to [0, 1] using population stats, then applies weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core.config import ScoringWeights
from core.exceptions import ScoringError
from core.models import PremarketMetrics, ScanResult


@dataclass
class NormalisedMetrics:
    trend_quality: float
    relative_volume: float
    float_rotation_pct: float       # 0.0 when float data unavailable
    gap_pct: float
    catalyst_score: float
    premarket_dollar_volume: float
    spread_inverse: float


def _minmax_norm(value: float, values: list[float]) -> float:
    """Min-max normalise a single value against a population."""
    if len(values) < 2:
        return 0.5
    lo, hi = min(values), max(values)
    if hi == lo:
        return 0.5
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _clip_norm(value: float, low: float, high: float) -> float:
    """Clip a value to [low, high] then normalise."""
    return max(0.0, min(1.0, (value - low) / (high - low))) if high != low else 0.5


def normalise_metrics(
    metrics: PremarketMetrics,
    catalyst_score: float,
    population: list[PremarketMetrics] | None = None,
) -> NormalisedMetrics:
    """
    Normalise a single stock's metrics.
    If `population` is provided, normalisation is relative (cross-sectional).
    Otherwise, fixed reference ranges are used.
    """
    if population and len(population) > 1:
        return _cross_sectional_normalise(metrics, catalyst_score, population)
    return _fixed_range_normalise(metrics, catalyst_score)


def _fixed_range_normalise(
    m: PremarketMetrics, catalyst_score: float
) -> NormalisedMetrics:
    """Normalise using domain-knowledge fixed ranges."""
    # Float rotation: 0–10% range; None → treated as 0 (no boost, no penalty)
    float_rot_raw = m.pm_float_rotation_pct if m.pm_float_rotation_pct is not None else 0.0
    return NormalisedMetrics(
        trend_quality=max(0.0, min(1.0, m.trend_quality)),
        relative_volume=_clip_norm(m.relative_volume, 2.0, 20.0),
        float_rotation_pct=_clip_norm(float_rot_raw, 0.0, 10.0),
        gap_pct=_clip_norm(m.gap_pct, 3.0, 30.0),
        catalyst_score=max(0.0, min(1.0, catalyst_score)),
        premarket_dollar_volume=_clip_norm(m.premarket_dollar_volume, 1_000_000, 50_000_000),
        spread_inverse=_clip_norm(1.0 - m.spread_pct / 0.5, 0.0, 1.0),
    )


def _cross_sectional_normalise(
    m: PremarketMetrics,
    catalyst_score: float,
    population: list[PremarketMetrics],
) -> NormalisedMetrics:
    """Normalise relative to the current scan population (rank-based)."""
    float_rot_raw = m.pm_float_rotation_pct if m.pm_float_rotation_pct is not None else 0.0
    pop_float_rots = [
        (p.pm_float_rotation_pct if p.pm_float_rotation_pct is not None else 0.0)
        for p in population
    ]
    return NormalisedMetrics(
        trend_quality=_minmax_norm(m.trend_quality, [p.trend_quality for p in population]),
        relative_volume=_minmax_norm(m.relative_volume, [p.relative_volume for p in population]),
        float_rotation_pct=_minmax_norm(float_rot_raw, pop_float_rots),
        gap_pct=_minmax_norm(m.gap_pct, [p.gap_pct for p in population]),
        catalyst_score=max(0.0, min(1.0, catalyst_score)),
        premarket_dollar_volume=_minmax_norm(
            m.premarket_dollar_volume, [p.premarket_dollar_volume for p in population]
        ),
        spread_inverse=_minmax_norm(
            1.0 - m.spread_pct / 0.5,
            [1.0 - p.spread_pct / 0.5 for p in population],
        ),
    )


def compute_composite_score(
    normalised: NormalisedMetrics,
    weights: ScoringWeights,
) -> float:
    """
    Weighted composite score in [0, 1].

    When float rotation data is unavailable (score=0.0), its weight is
    redistributed proportionally across the other factors so the total
    always equals 1.0.
    """
    has_float = normalised.float_rotation_pct > 0.0

    if has_float:
        score = (
            weights.trend_quality * normalised.trend_quality
            + weights.relative_volume * normalised.relative_volume
            + weights.float_rotation_pct * normalised.float_rotation_pct
            + weights.gap_pct * normalised.gap_pct
            + weights.catalyst_score * normalised.catalyst_score
            + weights.premarket_dollar_volume * normalised.premarket_dollar_volume
            + weights.spread_inverse * normalised.spread_inverse
        )
        total_weight = (
            weights.trend_quality + weights.relative_volume + weights.float_rotation_pct
            + weights.gap_pct + weights.catalyst_score + weights.premarket_dollar_volume
            + weights.spread_inverse
        )
    else:
        # Redistribute float_rotation weight proportionally across other factors
        other_weight = (
            weights.trend_quality + weights.relative_volume + weights.gap_pct
            + weights.catalyst_score + weights.premarket_dollar_volume + weights.spread_inverse
        )
        scale = 1.0 / other_weight if other_weight > 0 else 1.0
        score = (
            weights.trend_quality * normalised.trend_quality * scale
            + weights.relative_volume * normalised.relative_volume * scale
            + weights.gap_pct * normalised.gap_pct * scale
            + weights.catalyst_score * normalised.catalyst_score * scale
            + weights.premarket_dollar_volume * normalised.premarket_dollar_volume * scale
            + weights.spread_inverse * normalised.spread_inverse * scale
        )
        total_weight = 1.0

    if abs(total_weight - 1.0) > 0.02 and has_float:
        raise ScoringError(f"Scoring weights do not sum to 1.0 (sum={total_weight:.3f})")

    return max(0.0, min(1.0, score))


def rank_scan_results(results: list[ScanResult]) -> list[ScanResult]:
    """Sort scan results by composite score descending."""
    return sorted(results, key=lambda r: r.composite_score, reverse=True)


def score_population(
    scan_results: list[ScanResult],
    catalyst_scores: dict[str, float],
    weights: ScoringWeights,
) -> list[ScanResult]:
    """
    Re-score a list of ScanResults cross-sectionally.
    Modifies composite_score in-place (returns new list with updated scores).
    """
    passing = [r for r in scan_results if r.passes_filters]
    if not passing:
        return scan_results

    population_metrics = [r.metrics for r in passing]
    updated = []
    for result in scan_results:
        if not result.passes_filters:
            updated.append(result)
            continue
        cat_score = catalyst_scores.get(result.symbol, 0.0)
        normed = normalise_metrics(result.metrics, cat_score, population_metrics)
        score = compute_composite_score(normed, weights)
        updated.append(result.model_copy(update={"composite_score": score}))
    return updated
