"""
analytics/reporting.py
Generate text, table, and rich console reports from performance data.
"""
from __future__ import annotations

from core.models import PerformanceMetrics, Trade

try:
    from rich.console import Console
    from rich.table import Table
    _RICH = True
except ImportError:
    _RICH = False


def print_performance_report(metrics: PerformanceMetrics) -> None:
    """Print a formatted performance summary to stdout."""
    if _RICH:
        _print_rich_report(metrics)
    else:
        _print_plain_report(metrics)


def _print_rich_report(metrics: PerformanceMetrics) -> None:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    table = Table(title=f"Performance: {metrics.start_date} → {metrics.end_date}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    rows = [
        ("Total Trades", str(metrics.total_trades)),
        ("Win Rate", f"{metrics.win_rate * 100:.1f}%"),
        ("Net PnL", f"${metrics.net_pnl:+.2f}"),
        ("Total Return", f"{metrics.total_return_pct:+.2f}%"),
        ("Annualised Return", f"{metrics.annualized_return_pct:+.2f}%"),
        ("Max Drawdown", f"{metrics.max_drawdown_pct:.2f}%"),
        ("Sharpe Ratio", f"{metrics.sharpe_ratio:.3f}"),
        ("Sortino Ratio", f"{metrics.sortino_ratio:.3f}"),
        ("Profit Factor", f"{metrics.profit_factor:.3f}"),
        ("Avg R Multiple", f"{metrics.avg_r_multiple:.3f}"),
        ("Avg Winner R", f"{metrics.avg_winner_r:.3f}"),
        ("Avg Loser R", f"{metrics.avg_loser_r:.3f}"),
        ("Avg Hold (min)", f"{metrics.avg_hold_minutes:.1f}"),
        ("Total Commission", f"${metrics.total_commission:.2f}"),
    ]
    for name, val in rows:
        table.add_row(name, val)
    console.print(table)


def _print_plain_report(metrics: PerformanceMetrics) -> None:
    print(f"\n{'='*50}")
    print(f"Performance: {metrics.start_date} → {metrics.end_date}")
    print(f"{'='*50}")
    print(f"Total Trades:     {metrics.total_trades}")
    print(f"Win Rate:         {metrics.win_rate * 100:.1f}%")
    print(f"Net PnL:          ${metrics.net_pnl:+.2f}")
    print(f"Total Return:     {metrics.total_return_pct:+.2f}%")
    print(f"Max Drawdown:     {metrics.max_drawdown_pct:.2f}%")
    print(f"Sharpe Ratio:     {metrics.sharpe_ratio:.3f}")
    print(f"Profit Factor:    {metrics.profit_factor:.3f}")
    print(f"Avg R Multiple:   {metrics.avg_r_multiple:.3f}")
    print(f"{'='*50}\n")


def print_trade_log(trades: list[Trade]) -> None:
    """Print a chronological trade log."""
    if not trades:
        print("No trades to display.")
        return

    if _RICH:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(title=f"Trade Log ({len(trades)} trades)")
        for col in ["Symbol", "Entry", "Exit", "Qty", "PnL", "R", "Exit Reason", "Hold"]:
            table.add_column(col)
        for t in sorted(trades, key=lambda x: x.exit_time):
            hold_min = f"{t.hold_duration_seconds / 60:.0f}m"
            pnl_style = "green" if t.net_pnl >= 0 else "red"
            table.add_row(
                t.symbol,
                f"{t.entry_price:.2f}",
                f"{t.exit_price:.2f}",
                str(t.quantity),
                f"[{pnl_style}]${t.net_pnl:+.2f}[/{pnl_style}]",
                f"{t.r_multiple:+.2f}",
                t.exit_reason.value,
                hold_min,
            )
        console.print(table)
    else:
        print(f"\n{'Symbol':8} {'Entry':8} {'Exit':8} {'PnL':10} {'R':6} {'Reason':20}")
        print("-" * 65)
        for t in sorted(trades, key=lambda x: x.exit_time):
            print(
                f"{t.symbol:8} {t.entry_price:8.2f} {t.exit_price:8.2f} "
                f"${t.net_pnl:+9.2f} {t.r_multiple:+5.2f} {t.exit_reason.value:20}"
            )
