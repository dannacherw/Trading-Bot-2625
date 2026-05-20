"""
scanner/percentile_scoring.py
Percentile-rank based composite scoring.

Replaces the min-max normalisation in scoring.py with rank-based
percentile scoring, which is:
  - Outlier-resistant: a 100x relative volume doesn't compress all others
  - Distribution-aware: scores reflect where a stock stands vs history
  - Consistent: a score of 0.80 means the stock is in the 80th percentile

Two normalisation modes:
  1. Cross-sectional (today's population only):
     - Fast, uses only today's scan results
     - Fragile on low-count days (5 stocks = coarse percentile grid)
  2. Rolling historical (recommended):
     - Uses 30-day rolling distribution from SQLite
     - Stable scores regardless of today's population size
     - Falls back to cross-sectional if history is unavailable

Architecture:
  PercentileScorer is stateless per-call but loads historical quantiles
  lazily from the database on first use each day, then caches in-process.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import numpy as np
from loguru import logger

from core.config import ScoringWeights
from core.models import PremarketMetrics, ScanResult


# ---------------------------------------------------------------------------
# Normalised metric container (same shape as the old NormalisedMetrics)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class NormalisedMetrics:
    trend_quality: float
    relative_volume: float
    float_rotation_pct: float
    gap_pct: float
    catalyst_score: float
    premarket_dollar_volume: float
    spread_inverse: float


# ---------------------------------------------------------------------------
# Historical quantile store
# ---------------------------------------------------------------------------

@dataclass
class FeatureQuantiles:
    """
    Pre-computed percentile breakpoints for a feature, loaded from history.
    quantiles[i] = value at the i-th percentile (0..100).
    """
    feature: str
    quantiles: np.ndarray  # shape (101,) — values at pct 0, 1, …, 100
    sample_count: int
    computed_at: str


class QuantileStore:
    """
    In-process daily cache of per-feature quantile arrays.
    Loaded from the SQLite percentile_distributions table once per day.
    """

    def __init__(self) -> None:
        self._store: dict[str, FeatureQuantiles] = {}
        self._loaded_date: str = ""

    def is_fresh(self) -> bool:
        return self._loaded_date == date.today().isoformat()

    def load(self, quantiles: list[FeatureQuantiles]) -> None:
        self._store = {q.feature: q for q in quantiles}
        self._loaded_date = date.today().isoformat()

    def get(self, feature: str) -> FeatureQuantiles | None:
        return self._store.get(feature)

    def has_all_features(self, features: list[str]) -> bool:
        return all(f in self._store for f in features)


_QUANTILE_STORE = QuantileStore()

FEATURE_NAMES = [
    "trend_quality",
    "relative_volume",
    "float_rotation_pct",
    "gap_pct",
    "premarket_dollar_volume",
    "spread_inverse",  # computed as 1 - spread_pct/0.5
]


# ---------------------------------------------------------------------------
# Core percentile normalisation
# ---------------------------------------------------------------------------

def _percentile_rank(value: float, quantile_arr: np.ndarray) -> float:
    """
    Given a sorted array of quantile breakpoints (pct 0..100),
    return the percentile rank of value in [0, 1].

    Uses linear interpolation between the two nearest breakpoints.
    """
    if len(quantile_arr) == 0:
        return 0.5
    # np.searchsorted: index where value fits in sorted array
    idx = np.searchsorted(quantile_arr, value, side="right")
    n = len(quantile_arr)  # = 101 for pct 0..100

    if idx == 0:
        return 0.0
    if idx >= n:
        return 1.0

    # Linear interpolation between breakpoints
    lo_pct = (idx - 1) / (n - 1)
    hi_pct = idx / (n - 1)
    lo_val = quantile_arr[idx - 1]
    hi_val = quantile_arr[idx]

    if hi_val == lo_val:
        return lo_pct

    frac = (value - lo_val) / (hi_val - lo_val)
    return float(np.clip(lo_pct + frac * (hi_pct - lo_pct), 0.0, 1.0))


def _cross_sectional_rank(value: float, population: list[float]) -> float:
    """
    Simple cross-sectional percentile rank of value in population.
    Returns fraction of population values strictly below value.
    """
    if not population:
        return 0.5
    n = len(population)
    below = sum(1 for v in population if v < value)
    # +0.5 adjustment avoids 0.0 and 1.0 for min/max (Hazen formula)
    return float(np.clip((below + 0.5) / n, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _extract_features(m: PremarketMetrics, catalyst_score: float) -> dict[str, float]:
    """Extract raw feature values from metrics for scoring."""
    float_rot = m.pm_float_rotation_pct if m.pm_float_rotation_pct is not None else 0.0
    spread_inv = max(0.0, 1.0 - m.spread_pct / 0.5)
    return {
        "trend_quality":          m.trend_quality,
        "relative_volume":        m.relative_volume,
        "float_rotation_pct":     float_rot,
        "gap_pct":                m.gap_pct,
        "catalyst_score":         catalyst_score,
        "premarket_dollar_volume": m.premarket_dollar_volume,
        "spread_inverse":         spread_inv,
    }


# ---------------------------------------------------------------------------
# Public normalisation functions
# ---------------------------------------------------------------------------

def normalise_percentile_historical(
    metrics: PremarketMetrics,
    catalyst_score: float,
) -> NormalisedMetrics:
    """
    Normalise using pre-loaded historical quantile distributions.
    Falls back to 0.5 for features with no history.
    """
    feats = _extract_features(metrics, catalyst_score)
    normed: dict[str, float] = {}

    for name in FEATURE_NAMES:
        q = _QUANTILE_STORE.get(name)
        raw = feats.get(name, 0.0)
        if q is not None and q.sample_count >= 30:
            normed[name] = _percentile_rank(raw, q.quantiles)
        else:
            # No history — fall back to fixed-range clip normalisation
            normed[name] = _fallback_norm(name, raw)

    # Catalyst score is always bounded [0, 1] directly
    normed["catalyst_score"] = float(np.clip(catalyst_score, 0.0, 1.0))

    return NormalisedMetrics(
        trend_quality=normed["trend_quality"],
        relative_volume=normed["relative_volume"],
        float_rotation_pct=normed["float_rotation_pct"],
        gap_pct=normed["gap_pct"],
        catalyst_score=normed["catalyst_score"],
        premarket_dollar_volume=normed["premarket_dollar_volume"],
        spread_inverse=normed["spread_inverse"],
    )


def normalise_percentile_cross_sectional(
    metrics: PremarketMetrics,
    catalyst_score: float,
    population: list[PremarketMetrics],
) -> NormalisedMetrics:
    """
    Normalise against today's scan population using rank-based percentiles.
    More robust than min-max for small populations.
    """
    feats = _extract_features(metrics, catalyst_score)

    pop_feats: dict[str, list[float]] = {name: [] for name in FEATURE_NAMES}
    for m in population:
        pf = _extract_features(m, 0.0)  # catalyst varies — use 0 for population rank
        for name in FEATURE_NAMES:
            pop_feats[name].append(pf[name])

    normed: dict[str, float] = {}
    for name in FEATURE_NAMES:
        normed[name] = _cross_sectional_rank(feats[name], pop_feats[name])
    normed["catalyst_score"] = float(np.clip(catalyst_score, 0.0, 1.0))

    return NormalisedMetrics(
        trend_quality=normed["trend_quality"],
        relative_volume=normed["relative_volume"],
        float_rotation_pct=normed["float_rotation_pct"],
        gap_pct=normed["gap_pct"],
        catalyst_score=normed["catalyst_score"],
        premarket_dollar_volume=normed["premarket_dollar_volume"],
        spread_inverse=normed["spread_inverse"],
    )


def _fallback_norm(feature: str, value: float) -> float:
    """Fixed-range normalisation used when no historical quantiles exist."""
    RANGES: dict[str, tuple[float, float]] = {
        "trend_quality":           (0.0, 1.0),
        "relative_volume":         (2.0, 20.0),
        "float_rotation_pct":      (0.0, 10.0),
        "gap_pct":                 (3.0, 30.0),
        "premarket_dollar_volume": (1_000_000, 50_000_000),
        "spread_inverse":          (0.0, 1.0),
    }
    lo, hi = RANGES.get(feature, (0.0, 1.0))
    if hi == lo:
        return 0.5
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Composite score computation (weight application — unchanged interface)
# ---------------------------------------------------------------------------

def compute_composite_score(
    normalised: NormalisedMetrics,
    weights: ScoringWeights,
) -> float:
    """
    Weighted composite score in [0, 1].

    When float rotation is 0.0 (data unavailable), its weight is
    redistributed proportionally across the remaining factors.
    """
    has_float = normalised.float_rotation_pct > 0.0

    w = weights
    if has_float:
        total_w = (
            w.trend_quality + w.relative_volume + w.float_rotation_pct
            + w.gap_pct + w.catalyst_score + w.premarket_dollar_volume + w.spread_inverse
        )
        score = (
            w.trend_quality          * normalised.trend_quality
            + w.relative_volume      * normalised.relative_volume
            + w.float_rotation_pct   * normalised.float_rotation_pct
            + w.gap_pct              * normalised.gap_pct
            + w.catalyst_score       * normalised.catalyst_score
            + w.premarket_dollar_volume * normalised.premarket_dollar_volume
            + w.spread_inverse       * normalised.spread_inverse
        )
    else:
        other_w = (
            w.trend_quality + w.relative_volume + w.gap_pct
            + w.catalyst_score + w.premarket_dollar_volume + w.spread_inverse
        )
        scale = 1.0 / other_w if other_w > 0 else 1.0
        score = (
            w.trend_quality          * normalised.trend_quality          * scale
            + w.relative_volume      * normalised.relative_volume        * scale
            + w.gap_pct              * normalised.gap_pct                * scale
            + w.catalyst_score       * normalised.catalyst_score         * scale
            + w.premarket_dollar_volume * normalised.premarket_dollar_volume * scale
            + w.spread_inverse       * normalised.spread_inverse         * scale
        )
        total_w = 1.0

    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Score population (replaces score_population in scoring.py)
# ---------------------------------------------------------------------------

def score_population_percentile(
    scan_results: list[ScanResult],
    catalyst_scores: dict[str, float],
    weights: ScoringWeights,
    use_historical: bool = True,
) -> list[ScanResult]:
    """
    Re-score all passing scan results using percentile normalisation.

    Args:
        scan_results: Full list of scan results (pass + fail)
        catalyst_scores: symbol -> catalyst_score mapping
        weights: Scoring weights
        use_historical: If True and history is loaded, use historical percentiles.
                        Falls back to cross-sectional if history unavailable.
    """
    passing = [r for r in scan_results if r.passes_filters]
    if not passing:
        return scan_results

    use_hist = use_historical and _QUANTILE_STORE.is_fresh() and _QUANTILE_STORE.has_all_features(FEATURE_NAMES)

    if not use_hist:
        logger.debug("Percentile scoring: using cross-sectional (no history loaded)")
        population_metrics = [r.metrics for r in passing]
        updated = []
        for result in scan_results:
            if not result.passes_filters:
                updated.append(result)
                continue
            cat_score = catalyst_scores.get(result.symbol, 0.0)
            normed = normalise_percentile_cross_sectional(
                result.metrics, cat_score, population_metrics
            )
            score = compute_composite_score(normed, weights)
            updated.append(result.model_copy(update={"composite_score": score}))
        return updated
    else:
        logger.debug("Percentile scoring: using 30-day historical quantiles")
        updated = []
        for result in scan_results:
            if not result.passes_filters:
                updated.append(result)
                continue
            cat_score = catalyst_scores.get(result.symbol, 0.0)
            normed = normalise_percentile_historical(result.metrics, cat_score)
            score = compute_composite_score(normed, weights)
            updated.append(result.model_copy(update={"composite_score": score}))
        return updated


def rank_scan_results(results: list[ScanResult]) -> list[ScanResult]:
    """Sort passing results first by composite score descending."""
    return sorted(results, key=lambda r: (r.passes_filters, r.composite_score), reverse=True)


# ---------------------------------------------------------------------------
# Quantile builder — run EOD to update the historical distribution table
# ---------------------------------------------------------------------------

def build_quantiles_from_samples(
    feature: str,
    values: list[float],
    n_quantiles: int = 100,
) -> FeatureQuantiles:
    """
    Build a FeatureQuantiles object from a list of raw feature values.
    Call this on the last 30 days of scan data from the database.
    """
    if not values:
        return FeatureQuantiles(
            feature=feature,
            quantiles=np.linspace(0.0, 1.0, n_quantiles + 1),
            sample_count=0,
            computed_at=datetime.now(tz=timezone.utc).isoformat(),
        )
    arr = np.array(values, dtype=float)
    # Remove NaN/inf
    arr = arr[np.isfinite(arr)]
    percentiles = np.percentile(arr, np.linspace(0, 100, n_quantiles + 1))
    return FeatureQuantiles(
        feature=feature,
        quantiles=percentiles,
        sample_count=len(arr),
        computed_at=datetime.now(tz=timezone.utc).isoformat(),
    )


def load_quantile_store(quantiles: list[FeatureQuantiles]) -> None:
    """Load quantiles into the global store. Called at session start."""
    _QUANTILE_STORE.load(quantiles)
    logger.info(
        "Percentile scoring: loaded {} feature distributions (sample counts: {})",
        len(quantiles),
        {q.feature: q.sample_count for q in quantiles},
    )
