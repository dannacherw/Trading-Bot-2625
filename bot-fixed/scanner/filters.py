"""
scanner/filters.py
Premarket filter predicates — v2.
Changes from v1:
  - Tiered gap filter: lower floor allowed only with confirmed high-quality catalyst
  - Float rotation filter: requires ≥2% of float traded premarket (skipped if no float data)
  - All other filters unchanged
"""
from __future__ import annotations

from core.config import PremarketFilterSettings
from core.enums import CatalystCategory
from core.models import Catalyst, PremarketMetrics

FilterResult = tuple[bool, str | None]

# Categories considered "high quality" for the tiered gap floor
HIGH_QUALITY_CATALYST_CATEGORIES = {
    CatalystCategory.EARNINGS,
    CatalystCategory.FDA_OR_BIOTECH_NEWS,
    CatalystCategory.M_AND_A,
    CatalystCategory.EARNINGS_GUIDANCE,
}


def check_min_gap(
    metrics: PremarketMetrics,
    settings: PremarketFilterSettings,
    catalyst: Catalyst | None = None,
) -> FilterResult:
    """
    Tiered gap filter.
    - With a confirmed high-quality catalyst: min gap = settings.min_gap_pct (default 3%)
    - Without: min gap = settings.min_gap_pct_no_catalyst (default 6%)
    Prevents taking weak, uncatalyzed gap stocks that reverse at the open.
    """
    catalyst_qualifies = (
        catalyst is not None
        and catalyst.category in HIGH_QUALITY_CATALYST_CATEGORIES
        and catalyst.confidence >= settings.high_quality_catalyst_min_confidence
    )
    effective_min = settings.min_gap_pct if catalyst_qualifies else settings.min_gap_pct_no_catalyst

    if metrics.gap_pct < effective_min:
        catalyst_str = f"catalyst={catalyst.category.value}" if catalyst else "no catalyst"
        return False, (
            f"gap {metrics.gap_pct:.1f}% < min {effective_min:.1f}% ({catalyst_str})"
        )
    return True, None


def check_min_premarket_volume(
    metrics: PremarketMetrics,
    settings: PremarketFilterSettings,
    catalyst: Catalyst | None = None,
) -> FilterResult:
    if metrics.premarket_volume < settings.min_premarket_volume:
        return False, (
            f"PM vol {metrics.premarket_volume:,} < min {settings.min_premarket_volume:,}"
        )
    return True, None


def check_relative_volume(
    metrics: PremarketMetrics,
    settings: PremarketFilterSettings,
    catalyst: Catalyst | None = None,
) -> FilterResult:
    if metrics.relative_volume < settings.min_relative_volume:
        return False, (
            f"rel vol {metrics.relative_volume:.1f}x < min {settings.min_relative_volume:.1f}x"
        )
    return True, None


def check_min_dollar_volume(
    metrics: PremarketMetrics,
    settings: PremarketFilterSettings,
    catalyst: Catalyst | None = None,
) -> FilterResult:
    if metrics.premarket_dollar_volume < settings.min_premarket_dollar_volume:
        return False, (
            f"PM $vol ${metrics.premarket_dollar_volume:,.0f} "
            f"< min ${settings.min_premarket_dollar_volume:,.0f}"
        )
    return True, None


def check_spread(
    metrics: PremarketMetrics,
    settings: PremarketFilterSettings,
    catalyst: Catalyst | None = None,
) -> FilterResult:
    if metrics.spread_pct > settings.max_spread_pct:
        return False, f"spread {metrics.spread_pct:.3f}% > max {settings.max_spread_pct:.3f}%"
    return True, None


def check_premarket_range(
    metrics: PremarketMetrics,
    settings: PremarketFilterSettings,
    catalyst: Catalyst | None = None,
) -> FilterResult:
    if metrics.range_pct > settings.max_premarket_range_pct:
        return False, (
            f"PM range {metrics.range_pct:.1f}% > max {settings.max_premarket_range_pct:.1f}%"
        )
    return True, None


def check_range_position(
    metrics: PremarketMetrics,
    settings: PremarketFilterSettings,
    catalyst: Catalyst | None = None,
) -> FilterResult:
    if metrics.range_position < settings.min_range_position:
        return False, (
            f"range pos {metrics.range_position:.2f} < min {settings.min_range_position:.2f} "
            "(not trading in upper half of PM range)"
        )
    return True, None


def check_float_rotation(
    metrics: PremarketMetrics,
    settings: PremarketFilterSettings,
    catalyst: Catalyst | None = None,
) -> FilterResult:
    """
    Float rotation filter.
    Skipped entirely (passes) when:
      - Float data is unavailable (metrics.pm_float_rotation_pct is None)
      - Filter is disabled in settings
    Active when float data IS available and filter is enabled.
    """
    if not settings.use_float_rotation_filter:
        return True, None
    if metrics.pm_float_rotation_pct is None:
        # Float unknown — allow through per user decision
        return True, None
    if metrics.pm_float_rotation_pct < settings.min_pm_float_rotation_pct:
        return False, (
            f"float rotation {metrics.pm_float_rotation_pct:.2f}% "
            f"< min {settings.min_pm_float_rotation_pct:.2f}%"
        )
    return True, None


# ---------------------------------------------------------------------------
# Composite filter — ordered by cheapest/most eliminating first
# ---------------------------------------------------------------------------

_ALL_FILTERS = [
    check_min_gap,             # Cheapest check — run first
    check_min_premarket_volume,
    check_relative_volume,
    check_min_dollar_volume,
    check_spread,
    check_premarket_range,
    check_range_position,
    check_float_rotation,      # Last — requires float data lookup
]


def apply_all_filters(
    metrics: PremarketMetrics,
    settings: PremarketFilterSettings,
    catalyst: Catalyst | None = None,
) -> FilterResult:
    """
    Run all premarket filters in order.
    Returns (True, None) if all pass, or (False, reason) on first failure.
    Pass `catalyst` to enable tiered gap logic.
    """
    for filter_fn in _ALL_FILTERS:
        passes, reason = filter_fn(metrics, settings, catalyst)
        if not passes:
            return False, reason
    return True, None


def get_filter_summary(
    metrics: PremarketMetrics,
    settings: PremarketFilterSettings,
    catalyst: Catalyst | None = None,
) -> dict[str, FilterResult]:
    """Run all filters and return results for all (for debugging/logging)."""
    return {fn.__name__: fn(metrics, settings, catalyst) for fn in _ALL_FILTERS}
