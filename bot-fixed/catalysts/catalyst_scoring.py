"""
catalysts/catalyst_scoring.py
Converts catalyst classification results into a strength score [0, 1].
v2: Adds news timing adjustments — late/near-open news gets a penalty
    because it causes overreaction and higher reversal risk intraday.
"""
from __future__ import annotations

from datetime import datetime, timezone

from core.enums import CatalystCategory, NewsTimingCategory

# ---------------------------------------------------------------------------
# Category strength weights
# ---------------------------------------------------------------------------

CATEGORY_STRENGTH: dict[CatalystCategory, float] = {
    CatalystCategory.FDA_OR_BIOTECH_NEWS:     1.0,
    CatalystCategory.EARNINGS:                0.95,
    CatalystCategory.M_AND_A:                 0.90,
    CatalystCategory.EARNINGS_GUIDANCE:       0.80,
    CatalystCategory.ANALYST_UPGRADE:         0.65,
    CatalystCategory.ANALYST_DOWNGRADE:       0.60,
    CatalystCategory.PARTNERSHIP_OR_CONTRACT: 0.70,
    CatalystCategory.PRODUCT_LAUNCH:          0.60,
    CatalystCategory.LEGAL_OR_REGULATORY:     0.55,
    CatalystCategory.PRESS_RELEASE:           0.35,
    CatalystCategory.MACRO_RELATED:           0.30,
    CatalystCategory.UNKNOWN:                 0.10,
}

# ---------------------------------------------------------------------------
# News timing multipliers
# Overnight/early premarket → institutions have time to position cleanly → higher quality
# Late premarket / near open → overreaction, crowded, higher reversal risk → penalty
# ---------------------------------------------------------------------------

TIMING_MULTIPLIER: dict[NewsTimingCategory, float] = {
    NewsTimingCategory.OVERNIGHT:       1.10,   # Premium: clean institutional positioning
    NewsTimingCategory.EARLY_PREMARKET: 1.00,   # Baseline
    NewsTimingCategory.LATE_PREMARKET:  0.80,   # Penalty: less time to digest, crowded
    NewsTimingCategory.NEAR_OPEN:       0.65,   # Larger penalty: chaotic, high reversal risk
    NewsTimingCategory.INTRADAY:        0.75,   # Mid-session news — unpredictable
    NewsTimingCategory.UNKNOWN:         0.90,   # Slight discount for uncertainty
}


def classify_news_timing(published_at: datetime | None) -> NewsTimingCategory:
    """
    Determine the news timing category based on when it was published
    relative to NYSE market open (9:30 ET = 13:30 UTC).
    """
    if published_at is None:
        return NewsTimingCategory.UNKNOWN

    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)

    # Extract hour/minute in UTC (EST = UTC-5, EDT = UTC-4)
    # We approximate ET as UTC-4 (EDT, market hours)
    et_hour = (published_at.hour - 4) % 24   # rough ET conversion

    if et_hour < 6:           # Before 6:00 AM ET
        return NewsTimingCategory.OVERNIGHT
    elif et_hour < 8:          # 6:00–8:00 AM ET
        return NewsTimingCategory.EARLY_PREMARKET
    elif et_hour < 9 or (et_hour == 9 and published_at.minute < 15):
        return NewsTimingCategory.LATE_PREMARKET    # 8:00–9:15 AM ET
    elif et_hour == 9 and published_at.minute < 30:
        return NewsTimingCategory.NEAR_OPEN         # 9:15–9:30 AM ET
    else:
        return NewsTimingCategory.INTRADAY          # 9:30 AM ET and after


def compute_catalyst_strength(
    category: CatalystCategory,
    classifier_confidence: float,
) -> float:
    """
    Compute catalyst strength score [0, 1].
    Combines category importance with classifier confidence.
    """
    base_strength = CATEGORY_STRENGTH.get(category, 0.1)
    return min(1.0, base_strength * classifier_confidence)


def compute_recency_decay(published_at: datetime, now: datetime | None = None) -> float:
    """
    Recency factor [0, 1]. News within 2h = 1.0, decays over 24h.
    Note: recency decay is secondary to timing category.
    """
    if now is None:
        now = datetime.now(tz=timezone.utc)
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    hours_ago = (now - published_at).total_seconds() / 3600.0
    if hours_ago <= 2.0:
        return 1.0
    if hours_ago >= 24.0:
        return 0.1
    return 1.0 - (hours_ago - 2.0) / 22.0 * 0.9


def compute_final_catalyst_score(
    category: CatalystCategory,
    confidence: float,
    published_at: datetime | None = None,
    timing: NewsTimingCategory = NewsTimingCategory.UNKNOWN,
) -> float:
    """
    Final catalyst score used in the composite scanner formula.
    = strength * timing_multiplier * recency_decay (if time known).

    The timing multiplier is the key v2 addition: late premarket news
    is penalised because it creates crowded, reversal-prone intraday setups.
    """
    strength = compute_catalyst_strength(category, confidence)
    timing_mult = TIMING_MULTIPLIER.get(timing, 0.90)

    if published_at is not None:
        recency = compute_recency_decay(published_at)
        return min(1.0, strength * timing_mult * recency)
    return min(1.0, strength * timing_mult)
