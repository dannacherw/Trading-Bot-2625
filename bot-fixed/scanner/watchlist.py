"""
scanner/watchlist.py
Builds and manages the ranked watchlist and priority focus list.
The watchlist is the interface between the scanner and the strategy engine.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator

from loguru import logger

from core.config import WatchlistSettings
from core.enums import ArchetypeTag
from core.models import ScanResult, WatchlistItem
from scanner.tagging import is_high_quality_setup


class Watchlist:
    """
    Ranked list of trading candidates built from scanner output.

    Focus list = top N highest-scoring, high-quality setups.
    Full watchlist = all stocks that passed filters, sorted by score.
    """

    def __init__(self, settings: WatchlistSettings) -> None:
        self._settings = settings
        self._items: list[WatchlistItem] = []
        self._built_at: datetime | None = None

    def build(self, scan_results: list[ScanResult]) -> None:
        """
        Build watchlist from scan results.
        Input must be pre-sorted by composite_score descending.
        """
        passing = [r for r in scan_results if r.passes_filters]

        # Cap at max watchlist size
        passing = passing[: self._settings.max_watchlist_size]

        # Determine focus list: top N that also meet score threshold
        focus_candidates = [
            r for r in passing
            if r.composite_score >= self._settings.min_score_threshold
        ]

        # Prefer high-quality setups for focus
        hq = [r for r in focus_candidates if is_high_quality_setup(r.archetypes)]
        non_hq = [r for r in focus_candidates if r not in hq]
        focus_ordered = (hq + non_hq)[: self._settings.max_focus_list_size]
        focus_symbols = {r.symbol for r in focus_ordered}

        self._items = [
            WatchlistItem(
                symbol=r.symbol,
                rank=idx + 1,
                scan_result=r,
                is_focus=r.symbol in focus_symbols,
            )
            for idx, r in enumerate(passing)
        ]
        self._built_at = datetime.now(tz=timezone.utc)

        logger.info(
            "Watchlist built: {} total | {} focus",
            len(self._items),
            len(focus_symbols),
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def items(self) -> list[WatchlistItem]:
        return self._items

    @property
    def focus_list(self) -> list[WatchlistItem]:
        return [i for i in self._items if i.is_focus]

    @property
    def symbols(self) -> list[str]:
        return [i.symbol for i in self._items]

    @property
    def focus_symbols(self) -> list[str]:
        return [i.symbol for i in self.focus_list]

    @property
    def built_at(self) -> datetime | None:
        return self._built_at

    def get(self, symbol: str) -> WatchlistItem | None:
        return next((i for i in self._items if i.symbol == symbol), None)

    def is_watchlisted(self, symbol: str) -> bool:
        return any(i.symbol == symbol for i in self._items)

    def is_focus(self, symbol: str) -> bool:
        return any(i.symbol == symbol and i.is_focus for i in self._items)

    def __iter__(self) -> Iterator[WatchlistItem]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def to_table(self) -> list[dict]:
        """Return watchlist as a list of dicts for rich table display."""
        rows = []
        for item in self._items:
            m = item.scan_result.metrics
            rows.append({
                "rank": item.rank,
                "symbol": item.symbol,
                "focus": "★" if item.is_focus else "",
                "score": f"{item.scan_result.composite_score:.3f}",
                "gap%": f"{m.gap_pct:+.1f}%",
                "rel_vol": f"{m.relative_volume:.1f}x",
                "pm_$vol": f"${m.premarket_dollar_volume / 1e6:.1f}M",
                "spread%": f"{m.spread_pct:.3f}%",
                "archetypes": " ".join(t.value for t in item.scan_result.archetypes[:2]),
                "catalyst": (
                    item.scan_result.catalyst.category.value
                    if item.scan_result.catalyst
                    else "—"
                ),
            })
        return rows
