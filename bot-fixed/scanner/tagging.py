"""
scanner/tagging.py
Applies archetype tags to scan results based on metric thresholds.
Multiple tags can apply to a single stock simultaneously.
"""
from __future__ import annotations

from core.config import ArchetypeThresholds
from core.enums import ArchetypeTag
from core.models import Catalyst, PremarketMetrics, ScanResult


def tag_stock(
    metrics: PremarketMetrics,
    composite_score: float,
    catalyst: Catalyst | None,
    thresholds: ArchetypeThresholds,
) -> list[ArchetypeTag]:
    """
    Determine which archetype tags apply to a stock.
    Returns a non-empty list (always at least one tag).
    """
    tags: list[ArchetypeTag] = []

    # STRONG_GAPPER: large premarket gap
    if metrics.gap_pct >= thresholds.strong_gapper_gap_pct:
        tags.append(ArchetypeTag.STRONG_GAPPER)

    # ORDERLY_TREND: clean, consistent premarket move
    if metrics.trend_quality >= thresholds.orderly_trend_min_trend_quality:
        tags.append(ArchetypeTag.ORDERLY_TREND)

    # VOLATILE_CANDIDATE: wide premarket range (could be whippy)
    if metrics.range_pct >= thresholds.volatile_candidate_range_pct:
        tags.append(ArchetypeTag.VOLATILE_CANDIDATE)

    # SPREAD_RISK: spread too wide for clean execution
    if metrics.spread_pct >= thresholds.spread_risk_spread_pct:
        tags.append(ArchetypeTag.SPREAD_RISK)

    # LIKELY_LEADER: high composite score (top-tier setup)
    if composite_score >= thresholds.likely_leader_min_score:
        tags.append(ArchetypeTag.LIKELY_LEADER)

    # CATALYST_BACKED / NO_CONFIRMED_CATALYST
    if catalyst and catalyst.confidence >= thresholds.catalyst_backed_min_confidence:
        tags.append(ArchetypeTag.CATALYST_BACKED)
    else:
        tags.append(ArchetypeTag.NO_CONFIRMED_CATALYST)

    # EXTENDED_PREMARKET: extreme range — may be exhausted by open
    if metrics.range_pct >= thresholds.extended_premarket_range_pct:
        tags.append(ArchetypeTag.EXTENDED_PREMARKET)

    return tags or [ArchetypeTag.NO_CONFIRMED_CATALYST]


def describe_archetypes(tags: list[ArchetypeTag]) -> str:
    """Human-readable string of archetypes for logging."""
    return " | ".join(t.value for t in tags)


def is_high_quality_setup(tags: list[ArchetypeTag]) -> bool:
    """
    Heuristic: a setup is 'high quality' if it has at least one positive
    archetype and no disqualifying ones.
    """
    positive = {ArchetypeTag.STRONG_GAPPER, ArchetypeTag.ORDERLY_TREND,
                ArchetypeTag.LIKELY_LEADER, ArchetypeTag.CATALYST_BACKED}
    disqualifying = {ArchetypeTag.SPREAD_RISK}

    has_positive = bool(set(tags) & positive)
    has_disqualifier = bool(set(tags) & disqualifying)
    return has_positive and not has_disqualifier
