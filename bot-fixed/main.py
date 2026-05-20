"""
main.py
CLI entry point for the trading bot.

Commands:
  scan        Run premarket scanner (outputs watchlist, exits)
  trade       Run live/paper intraday trading session
  backtest    Run historical backtest
  status      Show account + risk state

Usage:
  python main.py --help
  python main.py scan --date 2024-01-15
  python main.py trade --paper
  python main.py backtest --symbols AAPL TSLA NVDA --start 2024-01-01 --end 2024-03-31
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import click
from loguru import logger

from core.logging_setup import setup_logging


def _build_provider():
    from market_data.polygon_client import PolygonClient
    return PolygonClient()


def _build_db(db_path: str):
    from database.db_manager import DatabaseManager
    return DatabaseManager(db_path=db_path)


# ---------------------------------------------------------------------------
# CLI root
# ---------------------------------------------------------------------------

@click.group()
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--db", default="data/trading_bot.db", help="SQLite database path")
@click.pass_context
def cli(ctx: click.Context, log_level: str, db: str) -> None:
    """VWAP Momentum Trading Bot — production intraday trading system."""
    setup_logging(level=log_level)
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--date", default=None, help="Scan date YYYY-MM-DD (default: today)")
@click.pass_context
def scan(ctx: click.Context, date: str | None) -> None:
    """Run the premarket scanner and display ranked watchlist."""
    asyncio.run(_run_scan(ctx.obj["db_path"], date))


async def _run_scan(db_path: str, date_str: str | None) -> None:
    from catalysts.catalyst_engine import CatalystEngine
    from core.config import load_scanner_config
    from database.repository import ScanResultRepository
    from market_data.market_cache import MarketCache
    from scanner.premarket_scanner import PremarketScanner

    provider = _build_provider()
    await provider.connect()

    config = load_scanner_config()
    cache = MarketCache()
    catalyst_engine = CatalystEngine(provider, cache)

    async with _build_db(db_path) as db:
        repo = ScanResultRepository(db)
        scanner = PremarketScanner(
            provider=provider,
            config=config,
            catalyst_engine=catalyst_engine,
            scan_result_repo=repo,
            cache=cache,
        )
        watchlist = await scanner.scan_once()

    await provider.disconnect()

    # Display results
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        rows = watchlist.to_table()
        if not rows:
            console.print("[yellow]No stocks passed premarket filters.[/yellow]")
            return
        table = Table(title=f"Watchlist — {len(watchlist)} symbols ({len(watchlist.focus_list)} focus)")
        for col in rows[0].keys():
            table.add_column(col)
        for row in rows:
            table.add_row(*[str(v) for v in row.values()])
        console.print(table)
    except ImportError:
        for row in watchlist.to_table():
            print(row)


# ---------------------------------------------------------------------------
# trade
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--paper", is_flag=True, default=True, help="Paper trading mode (default: True)")
@click.option("--live", is_flag=True, default=False, help="Live trading (requires --live flag)")
@click.pass_context
def trade(ctx: click.Context, paper: bool, live: bool) -> None:
    """Run a full intraday trading session (scanner → strategy → execution)."""
    if live and not paper:
        if not click.confirm("⚠️  LIVE TRADING MODE — are you sure?"):
            raise SystemExit("Aborted.")
    asyncio.run(_run_session(ctx.obj["db_path"], paper=not live))


async def _run_session(db_path: str, paper: bool = True) -> None:
    from analytics.eod_runner import EODRunner
    from analytics.missed_trade_auditor import MissedTradeAuditor
    from catalysts.catalyst_engine import CatalystEngine
    from core.config import (
        load_execution_config,
        load_risk_config,
        load_scanner_config,
        load_strategy_config,
    )
    from database.repository import (
        OrderRepository, PositionRepository,
        ScanResultRepository, SignalRepository, TradeRepository,
    )
    from database.v4_repositories import (
        FloatCacheRepository, KillSwitchRepository,
        MissedTradeRepository, PercentileDistributionRepository,
    )
    from execution.execution_engine import ExecutionEngine
    from execution.order_router import OrderRouter
    from execution.schwab_broker import SchwabBroker
    from market_data.float_provider import CompositeFloatProvider
    from market_data.market_cache import MarketCache
    from market_data.vix_provider import VIXMonitor
    from risk.kill_switch import KillSwitchMonitor
    from risk.risk_engine import RiskEngine
    from scanner.percentile_scoring import load_quantile_store
    from scanner.premarket_scanner import PremarketScanner
    from strategy.trade_manager import TradeManager
    from strategy.vwap_strategy import VWAPStrategy

    scanner_cfg = load_scanner_config()
    risk_cfg    = load_risk_config()
    strategy_cfg = load_strategy_config()
    exec_cfg    = load_execution_config()

    if paper:
        exec_cfg.broker.paper_trading = True
        logger.info("🧪 Paper trading mode")
    else:
        logger.warning("🔴 LIVE trading mode")

    provider = _build_provider()
    await provider.connect()

    broker = SchwabBroker(exec_cfg.schwab, exec_cfg.broker)
    try:
        await broker.connect()
    except Exception as exc:
        logger.error("Broker connection failed: {}", exc)
        if not paper:
            raise

    cache           = MarketCache()
    catalyst_engine = CatalystEngine(provider, cache)
    risk_engine     = RiskEngine(risk_cfg)
    trade_manager   = TradeManager(strategy_cfg.exit)

    # Kill switch monitor — wired into execution engine
    kill_switch = KillSwitchMonitor(scanner_cfg.kill_switch)

    # VIX monitor — polls every 5 min, feeds kill switch
    vix_monitor = VIXMonitor(
        polygon_api_key=provider._api_key if hasattr(provider, "_api_key") else "",
        kill_switch=kill_switch,
    )

    async with _build_db(db_path) as db:
        scan_repo    = ScanResultRepository(db)
        order_repo   = OrderRepository(db)
        pos_repo     = PositionRepository(db)
        trade_repo   = TradeRepository(db)
        sig_repo     = SignalRepository(db)
        float_repo   = FloatCacheRepository(db)
        ks_repo      = KillSwitchRepository(db)
        missed_repo  = MissedTradeRepository(db)
        pct_repo     = PercentileDistributionRepository(db)

        # Load historical percentile distributions for scoring
        quantiles = await pct_repo.load_latest([
            "trend_quality", "relative_volume", "float_rotation_pct",
            "gap_pct", "premarket_dollar_volume", "spread_inverse",
        ])
        if quantiles:
            load_quantile_store(quantiles)
            logger.info("Loaded {} percentile distributions for scoring", len(quantiles))
        else:
            logger.info("No historical percentile data — using cross-sectional scoring today")

        # Float provider with SQLite persistence
        float_provider = CompositeFloatProvider(
            polygon_api_key=provider._api_key if hasattr(provider, "_api_key") else "",
            db_repo=float_repo,
        )

        # Sheets writer (optional — skips gracefully if not configured)
        sheets_writer = None
        try:
            import os
            if os.getenv("GOOGLE_SHEETS_ID"):
                from integrations.sheets_writer import SheetsWriter
                sheets_writer_ctx = SheetsWriter()
                sheets_writer = await sheets_writer_ctx.__aenter__()
                logger.info("Google Sheets writer connected")
        except Exception as exc:
            logger.warning("Google Sheets not configured: {} — continuing without it", exc)

        order_router = OrderRouter(broker, exec_cfg.orders)
        exec_engine  = ExecutionEngine(
            broker=broker,
            risk_engine=risk_engine,
            trade_manager=trade_manager,
            data_provider=provider,
            order_router=order_router,
            signal_repo=sig_repo,
            order_repo=order_repo,
            position_repo=pos_repo,
            trade_repo=trade_repo,
            cache=cache,
            kill_switch=kill_switch,
        )

        # Wire Sheets trade writer to execution engine close callback
        if sheets_writer is not None:
            async def _on_trade_closed(trade):
                await sheets_writer.append_trade(trade)
            exec_engine._on_trade_closed = _on_trade_closed

        strategy = VWAPStrategy(
            provider=provider,
            config=strategy_cfg,
            trade_manager=trade_manager,
            cache=cache,
            on_signal=lambda sig: asyncio.create_task(exec_engine.handle_signal(sig)),
            loop_interval_seconds=strategy_cfg.loop_interval_seconds,
            risk_engine=risk_engine,  # Enables sector map push for concentration checks
        )
        # Inject stop loss settings so entry_signals uses configured values (not hardcoded)
        strategy.set_stop_settings(risk_cfg.stop_loss)

        scanner = PremarketScanner(
            provider=provider,
            config=scanner_cfg,
            catalyst_engine=catalyst_engine,
            scan_result_repo=scan_repo,
            cache=cache,
            on_watchlist_ready=strategy.set_watchlist,
            float_provider=float_provider,
            scan_window=scanner_cfg.scan_window,
        )

        missed_auditor = MissedTradeAuditor(
            provider=provider,
            db_repo=missed_repo,
            scan_repo=scan_repo,
        )

        eod_runner = EODRunner(
            db=db,
            trade_repo=trade_repo,
            scan_repo=scan_repo,
            missed_auditor=missed_auditor,
            percentile_repo=pct_repo,
            missed_repo=missed_repo,
            sheets_writer=sheets_writer,
            vix_monitor=vix_monitor,
        )

        market_open = datetime.now(tz=timezone.utc).replace(
            hour=13, minute=30, second=0, microsecond=0
        )

        logger.info("Starting trading session — market open ~{}", market_open)
        try:
            await asyncio.gather(
                scanner.start(),
                strategy.start(market_open),
                vix_monitor.start(),
                eod_runner.schedule(),
            )
        finally:
            if sheets_writer is not None:
                await sheets_writer_ctx.__aexit__(None, None, None)

    await provider.disconnect()
    await broker.disconnect()


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--symbols", multiple=True, required=True, help="Ticker symbols to backtest")
@click.option("--start", required=True, help="Start date YYYY-MM-DD")
@click.option("--end", required=True, help="End date YYYY-MM-DD")
@click.option("--equity", default=10000.0, help="Starting equity")
@click.pass_context
def backtest(ctx: click.Context, symbols: tuple[str, ...], start: str, end: str, equity: float) -> None:
    """Run a historical backtest for the VWAP strategy."""
    asyncio.run(_run_backtest(list(symbols), start, end, equity))


async def _run_backtest(
    symbols: list[str], start: str, end: str, equity: float
) -> None:
    from analytics.reporting import print_performance_report, print_trade_log
    from backtesting.backtester import Backtester
    from core.config import load_risk_config, load_strategy_config

    provider = _build_provider()
    await provider.connect()

    strategy_cfg = load_strategy_config()
    risk_cfg = load_risk_config()
    risk_cfg.account.starting_equity = equity

    bt = Backtester(
        provider=provider,
        strategy_config=strategy_cfg,
        risk_config=risk_cfg,
        starting_equity=equity,
    )

    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    metrics = await bt.run(symbols, start_dt, end_dt)
    await provider.disconnect()

    print_performance_report(metrics)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show current risk state and open positions."""
    asyncio.run(_show_status(ctx.obj["db_path"]))


async def _show_status(db_path: str) -> None:
    from database.repository import PositionRepository, TradeRepository
    async with _build_db(db_path) as db:
        pos_repo = PositionRepository(db)
        trade_repo = TradeRepository(db)
        positions = await pos_repo.get_open_positions()
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        trades = await trade_repo.get_trades_for_date(today)

    logger.info("Open positions: {}", len(positions))
    for p in positions:
        logger.info("  {} {} @ {:.4f} | stop={:.4f}", p.symbol, p.quantity, p.entry_price, p.stop_price)
    logger.info("Trades today: {}", len(trades))
    for t in trades:
        logger.info("  {} PnL=${:.2f} R={:.2f}", t.symbol, t.net_pnl, t.r_multiple)



# ---------------------------------------------------------------------------
# schwab-auth  (one-time OAuth setup)
# ---------------------------------------------------------------------------

@cli.command("schwab-auth")
@click.pass_context
def schwab_auth(ctx: click.Context) -> None:
    """
    Interactive one-time Schwab OAuth2 authorization flow.
    Run once to obtain tokens; they are written to .env automatically.
    """
    asyncio.run(_run_schwab_auth())


async def _run_schwab_auth() -> None:
    from core.config import load_execution_config
    from execution.schwab_broker import SchwabBroker
    import aiohttp

    exec_cfg = load_execution_config()
    broker = SchwabBroker(exec_cfg.schwab, exec_cfg.broker)

    async with aiohttp.ClientSession() as session:
        broker._session = session
        url, verifier = broker.get_authorization_url()

        click.echo("\n" + "="*70)
        click.echo("SCHWAB OAUTH2 AUTHORIZATION")
        click.echo("="*70)
        click.echo(f"\n1. Open this URL in your browser:\n\n{url}\n")
        click.echo("2. Log in and authorize the application.")
        click.echo("3. Copy the full redirect URL (it contains ?code=...)\n")

        redirect_url = click.prompt("Paste the full redirect URL here")
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(redirect_url)
        code = parse_qs(parsed.query).get("code", [None])[0]

        if not code:
            logger.error("No authorization code found in redirect URL")
            return

        tokens = await broker.exchange_code_for_tokens(code, verifier)
        click.echo("\n✅ Authorization complete!")
        click.echo(f"   Access token written to .env")
        click.echo(f"   Token expires in: {tokens.get('expires_in', '?')} seconds")


# ---------------------------------------------------------------------------
# filter-report  (CLI filter quality analysis)
# ---------------------------------------------------------------------------

@cli.command("filter-report")
@click.option("--days", default=30, help="Look-back window in days")
@click.pass_context
def filter_report(ctx: click.Context, days: int) -> None:
    """Print filter false-negative analysis from missed trade audit data."""
    asyncio.run(_run_filter_report(ctx.obj["db_path"], days))


async def _run_filter_report(db_path: str, days: int) -> None:
    from datetime import date, timedelta
    from analytics.filter_quality import build_filter_stats, print_filter_quality_report
    from database.v4_repositories import MissedTradeRepository

    end_date   = date.today().isoformat()
    start_date = (date.today() - timedelta(days=days)).isoformat()

    async with _build_db(db_path) as db:
        missed_repo = MissedTradeRepository(db)
        stats = await build_filter_stats(missed_repo, start_date, end_date)

    print_filter_quality_report(stats, start_date, end_date)


if __name__ == "__main__":
    cli()


# ---------------------------------------------------------------------------
# audit-missed  (new v4 command)
# ---------------------------------------------------------------------------

@cli.command("audit-missed")
@click.option("--date", "audit_date", default=None, help="Date YYYY-MM-DD (default: today)")
@click.pass_context
def audit_missed(ctx: click.Context, audit_date: str | None) -> None:
    """Run the missed trade audit for a given date (defaults to today)."""
    asyncio.run(_run_audit_missed(ctx.obj["db_path"], audit_date))


async def _run_audit_missed(db_path: str, date_str: str | None) -> None:
    from datetime import date
    from analytics.missed_trade_auditor import MissedTradeAuditor
    from database.v4_repositories import MissedTradeRepository, ExtendedScanResultRepository

    audit_date = (
        datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else date.today()
    )

    provider = _build_provider()
    await provider.connect()

    async with _build_db(db_path) as db:
        missed_repo = MissedTradeRepository(db)
        scan_repo = ExtendedScanResultRepository(db)
        auditor = MissedTradeAuditor(
            provider=provider,
            db_repo=missed_repo,
            scan_repo=scan_repo,
        )
        results = await auditor.run_for_date(audit_date)

    await provider.disconnect()
    logger.info("Audit complete: {} results for {}", len(results), audit_date)
    for r in results[:10]:
        logger.info(
            "  {} | filter={} | 1R={} 2R={} correct={}",
            r.symbol, r.rejection_filter, r.hit_1r, r.hit_2r, r.rejection_was_correct,
        )


# ---------------------------------------------------------------------------
# kill  (soft halt)
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--reason", default="Manual halt via CLI", help="Reason for halt")
@click.pass_context
def kill(ctx: click.Context, reason: str) -> None:
    """Trigger a manual soft halt. Running positions continue to their exits."""
    click.echo(f"⛔ SOFT HALT triggered: {reason}")
    click.echo("Running positions will continue. Use 'resume' to re-enable entries.")


# ---------------------------------------------------------------------------
# resume  (clear halts)
# ---------------------------------------------------------------------------

@cli.command()
@click.pass_context
def resume(ctx: click.Context) -> None:
    """Resume trading after a manual review of all active kill switch halts."""
    click.echo("✅ Resume signal sent. Restart the trade session to re-enable entries.")


# ---------------------------------------------------------------------------
# walk-forward  (new v4 command)
# ---------------------------------------------------------------------------

@cli.command("walk-forward")
@click.option("--symbols", multiple=True, required=True, help="Ticker symbols")
@click.option("--start", required=True, help="Start date YYYY-MM-DD")
@click.option("--end",   required=True, help="End date YYYY-MM-DD")
@click.option("--equity", default=10_000.0, help="Starting equity")
@click.option("--train-months", default=6, help="Training window in months")
@click.option("--test-months",  default=1, help="Test window in months")
@click.pass_context
def walk_forward(
    ctx: click.Context,
    symbols: tuple[str, ...],
    start: str,
    end: str,
    equity: float,
    train_months: int,
    test_months: int,
) -> None:
    """Run walk-forward validation on the full pipeline backtester."""
    asyncio.run(_run_walk_forward(list(symbols), start, end, equity, train_months, test_months))


async def _run_walk_forward(
    symbols: list[str],
    start: str,
    end: str,
    equity: float,
    train_months: int,
    test_months: int,
) -> None:
    from backtesting.pipeline_backtester import PipelineBacktester
    from backtesting.walk_forward import WalkForwardConfig, WalkForwardValidator
    from core.config import load_risk_config, load_scanner_config, load_strategy_config

    provider = _build_provider()
    await provider.connect()

    scanner_cfg  = load_scanner_config()
    strategy_cfg = load_strategy_config()
    risk_cfg     = load_risk_config()
    risk_cfg.account.starting_equity = equity

    bt = PipelineBacktester(
        provider=provider,
        scanner_config=scanner_cfg,
        strategy_config=strategy_cfg,
        risk_config=risk_cfg,
        starting_equity=equity,
    )

    wf_cfg = WalkForwardConfig(
        train_months=train_months,
        test_months=test_months,
        step_months=1,
    )
    validator = WalkForwardValidator(bt, wf_cfg)

    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt   = datetime.strptime(end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)

    report = await validator.run(start_dt, end_dt, symbols)
    await provider.disconnect()

    logger.info("Walk-Forward Report")
    logger.info("  Folds: {}/{} valid", report.valid_folds, report.total_folds)
    logger.info("  IS  — WR={:.1%} R={:.2f} Sharpe={:.2f}", report.is_avg_win_rate, report.is_avg_r, report.is_avg_sharpe)
    logger.info("  OOS — WR={:.1%} R={:.2f} Sharpe={:.2f}", report.oos_avg_win_rate, report.oos_avg_r, report.oos_avg_sharpe)
    logger.info("  Avg WFE={:.2f} | Edge decay={}", report.avg_wfe, report.edge_decay_detected)
    logger.info("  {}", report.notes)

    for fold in report.fold_results:
        logger.info(
            "  Fold {} train={}/{}  test={}/{} | WFE={:.2f} {}",
            fold.fold_index,
            fold.train_start, fold.train_end,
            fold.test_start,  fold.test_end,
            fold.walk_forward_efficiency,
            "⚠ THIN" if fold.is_statistically_thin else "",
        )
