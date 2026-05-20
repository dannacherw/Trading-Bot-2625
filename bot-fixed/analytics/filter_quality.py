"""
analytics/filter_quality.py
Filter quality analysis report.

Reads from missed_trade_audit and aggregates per-filter statistics:
  - How many trades each filter blocked
  - What fraction of those later hit +1R or +2R (false negative rate)
  - What fraction correctly stopped losers (true rejection rate)
  - Average MFE in R for blocked trades (what we left on the table)

Output: rich terminal table. Run via:
    python main.py filter-report --days 30

Interpretation guide:
  - High hit_1r_rate + high MFE  → filter is too strict, creating false negatives
  - Low hit_1r_rate + low MFE    → filter correctly rejected bad setups
  - should_investigate_rate > 20% → strongly consider loosening the filter

A quant's rule of thumb: if a filter has a false negative rate above 30%
and the average MFE of blocked trades is above +1.0R, that filter is costing
more in missed opportunity than it saves in avoided losses.
"""
from __future__ import annotations

from dataclasses import dataclass

from loguru import logger


@dataclass
class FilterStats:
    filter_name: str
    total_rejected: int
    hit_1r_count: int
    hit_2r_count: int
    hit_stop_count: int
    correct_rejections: int
    should_investigate_count: int
    avg_mfe_r: float
    avg_mae_r: float

    @property
    def hit_1r_rate(self) -> float:
        return self.hit_1r_count / self.total_rejected if self.total_rejected else 0.0

    @property
    def hit_2r_rate(self) -> float:
        return self.hit_2r_count / self.total_rejected if self.total_rejected else 0.0

    @property
    def correct_rejection_rate(self) -> float:
        return self.correct_rejections / self.total_rejected if self.total_rejected else 0.0

    @property
    def should_investigate_rate(self) -> float:
        return self.should_investigate_count / self.total_rejected if self.total_rejected else 0.0

    @property
    def verdict(self) -> str:
        """Plain-English verdict for this filter's performance."""
        if self.total_rejected < 10:
            return "INSUFFICIENT DATA"
        if self.hit_1r_rate > 0.35 and self.avg_mfe_r > 1.0:
            return "⚠ TOO STRICT — HIGH FALSE NEGATIVES"
        if self.hit_1r_rate > 0.25:
            return "WATCH — MODERATE FALSE NEGATIVES"
        if self.correct_rejection_rate > 0.70:
            return "✓ GOOD — CORRECTLY REJECTING LOSERS"
        return "NEUTRAL"


def _row_to_stats(row: dict) -> FilterStats:
    return FilterStats(
        filter_name=row.get("rejection_filter") or row.get("failed_filter", "unknown"),
        total_rejected=int(row.get("total_rejected", 0)),
        hit_1r_count=int(row.get("hit_1r_count", 0)),
        hit_2r_count=int(row.get("hit_2r_count", 0)),
        hit_stop_count=int(row.get("hit_stop_count", 0)),
        correct_rejections=int(row.get("correct_rejections", 0)),
        should_investigate_count=int(row.get("should_investigate_count", 0)),
        avg_mfe_r=float(row.get("avg_mfe_r", 0.0) or 0.0),
        avg_mae_r=float(row.get("avg_mae_r", 0.0) or 0.0),
    )


def print_filter_quality_report(
    stats: list[FilterStats],
    date_range_start: str,
    date_range_end: str,
) -> None:
    """
    Print a formatted filter quality report to the terminal.
    Uses rich if available, falls back to plain text.
    """
    if not stats:
        print(f"\nNo missed trade data found for {date_range_start} → {date_range_end}.\n"
              "The EOD audit must run for at least one day before this report is meaningful.")
        return

    try:
        _print_rich(stats, date_range_start, date_range_end)
    except ImportError:
        _print_plain(stats, date_range_start, date_range_end)


def _print_rich(
    stats: list[FilterStats],
    date_range_start: str,
    date_range_end: str,
) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    console.print(f"\n[bold cyan]Filter Quality Report[/bold cyan] "
                  f"[dim]{date_range_start} → {date_range_end}[/dim]\n")

    table = Table(box=box.SIMPLE_HEAD, show_footer=False)
    table.add_column("Filter",              style="cyan",  no_wrap=True)
    table.add_column("Rejected",            justify="right")
    table.add_column("Hit +1R",             justify="right")
    table.add_column("Hit +2R",             justify="right")
    table.add_column("Hit Stop",            justify="right")
    table.add_column("+1R Rate",            justify="right")
    table.add_column("Correct %",           justify="right")
    table.add_column("Avg MFE (R)",         justify="right")
    table.add_column("Avg MAE (R)",         justify="right")
    table.add_column("Verdict",             no_wrap=True)

    # Sort: highest false negative rate first — most important to review
    stats_sorted = sorted(stats, key=lambda s: s.hit_1r_rate, reverse=True)

    for s in stats_sorted:
        verdict_style = (
            "bold red"    if "TOO STRICT" in s.verdict else
            "yellow"      if "WATCH"      in s.verdict else
            "bold green"  if "GOOD"       in s.verdict else
            "dim"
        )
        table.add_row(
            s.filter_name,
            str(s.total_rejected),
            str(s.hit_1r_count),
            str(s.hit_2r_count),
            str(s.hit_stop_count),
            f"{s.hit_1r_rate:.1%}",
            f"{s.correct_rejection_rate:.1%}",
            f"{s.avg_mfe_r:+.2f}R",
            f"{s.avg_mae_r:+.2f}R",
            f"[{verdict_style}]{s.verdict}[/{verdict_style}]",
        )

    console.print(table)

    # Summary callouts
    warnings = [s for s in stats if "TOO STRICT" in s.verdict]
    if warnings:
        console.print("\n[bold red]Filters flagged for review:[/bold red]")
        for s in warnings:
            console.print(
                f"  [red]•[/red] [cyan]{s.filter_name}[/cyan]: "
                f"{s.hit_1r_rate:.1%} of rejections later hit +1R, "
                f"avg MFE = {s.avg_mfe_r:+.2f}R"
            )
        console.print(
            "\n[dim]These filters are blocking profitable setups more often than "
            "protecting against losses. Consider loosening their thresholds by 10–15% "
            "and validating with the walk-forward backtester before going live.[/dim]\n"
        )
    else:
        console.print(
            "[dim]No filters are flagged. All filters are within acceptable "
            "false-negative thresholds.[/dim]\n"
        )


def _print_plain(
    stats: list[FilterStats],
    date_range_start: str,
    date_range_end: str,
) -> None:
    print(f"\n{'='*90}")
    print(f"Filter Quality Report  {date_range_start} → {date_range_end}")
    print(f"{'='*90}")
    header = (
        f"{'Filter':<35} {'Rej':>5} {'1R':>5} {'2R':>5} "
        f"{'1R%':>7} {'Correct%':>9} {'MFE':>8} {'Verdict'}"
    )
    print(header)
    print("-" * 90)
    for s in sorted(stats, key=lambda s: s.hit_1r_rate, reverse=True):
        print(
            f"{s.filter_name:<35} {s.total_rejected:>5} {s.hit_1r_count:>5} "
            f"{s.hit_2r_count:>5} {s.hit_1r_rate:>7.1%} "
            f"{s.correct_rejection_rate:>9.1%} {s.avg_mfe_r:>+8.2f}  "
            f"{s.verdict}"
        )
    print(f"{'='*90}\n")


async def build_filter_stats(
    missed_repo: "MissedTradeRepository",  # type: ignore[name-defined]
    date_range_start: str,
    date_range_end: str,
) -> list[FilterStats]:
    """
    Query the missed_trade_audit table and return per-filter statistics.
    Delegates to MissedTradeRepository.get_filter_stats() which runs the
    GROUP BY SQL query.
    """
    raw_rows = await missed_repo.get_filter_stats(date_range_start, date_range_end)
    return [_row_to_stats(row) for row in raw_rows]
