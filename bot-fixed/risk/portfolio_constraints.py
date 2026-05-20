"""
risk/portfolio_constraints.py
Portfolio-level constraints to prevent over-concentration
and correlated position buildup.
"""
from __future__ import annotations

from core.models import Position


def check_sector_concentration(
    proposed_symbol: str,
    open_positions: list[Position],
    sector_map: dict[str, str],
    max_sector_pct: float = 60.0,
    total_equity: float = 10_000.0,
) -> tuple[bool, str]:
    """
    Ensure that adding a new position doesn't put too much capital
    in a single sector.
    """
    proposed_sector = sector_map.get(proposed_symbol, "UNKNOWN")
    sector_value = sum(
        p.open_value
        for p in open_positions
        if sector_map.get(p.symbol, "UNKNOWN") == proposed_sector
    )
    pct = sector_value / total_equity * 100.0 if total_equity > 0 else 0.0
    if pct >= max_sector_pct:
        return False, (
            f"Sector '{proposed_sector}' already at {pct:.1f}% "
            f"(max {max_sector_pct:.1f}%)"
        )
    return True, ""


def check_correlated_positions(
    proposed_symbol: str,
    open_positions: list[Position],
    correlated_pairs: set[frozenset[str]],
    max_correlated: int = 2,
) -> tuple[bool, str]:
    """
    Reject if adding this symbol would exceed max_correlated positions
    that are known to be correlated.
    """
    open_symbols = {p.symbol for p in open_positions}
    correlated_count = sum(
        1 for pair in correlated_pairs
        if proposed_symbol in pair and bool(pair & open_symbols)
    )
    if correlated_count >= max_correlated:
        return False, (
            f"{proposed_symbol} has {correlated_count} correlated "
            f"open positions (max {max_correlated})"
        )
    return True, ""


def check_duplicate_position(
    symbol: str, open_positions: list[Position]
) -> tuple[bool, str]:
    """Reject if we already have an open position in this symbol."""
    if any(p.symbol == symbol for p in open_positions):
        return False, f"Already have an open position in {symbol}"
    return True, ""
