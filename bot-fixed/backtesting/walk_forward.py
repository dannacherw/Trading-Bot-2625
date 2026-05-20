"""
backtesting/walk_forward.py
Walk-forward validation framework.

Splits the historical period into overlapping train/test windows and runs
the full pipeline backtester on each fold. This prevents overfitting to
a specific historical period by validating on unseen forward data.

Default configuration (proven robust for daily momentum):
  - Train window:  6 months
  - Test window:   1 month
  - Step size:     1 month (rolling)
  - Min train trades: 30 (folds with fewer are flagged as statistically thin)

Output:
  - Per-fold metrics (train + test)
  - Aggregate in-sample vs out-of-sample comparison
  - Edge decay detection (is performance degrading fold-over-fold?)
  - Walk-forward efficiency ratio (OOS Sharpe / IS Sharpe)

Example fold structure for 12-month period:
  Fold 1: train Jan-Jun, test Jul
  Fold 2: train Feb-Jul, test Aug
  ...
  Fold 6: train Jun-Nov, test Dec
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from loguru import logger

from backtesting.performance_metrics import compute_performance_metrics, PerformanceMetrics
from backtesting.pipeline_backtester import PipelineBacktester
from core.config import RiskConfig, ScannerConfig, StrategyConfig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardConfig:
    train_months: int = 6
    test_months: int = 1
    step_months: int = 1
    min_train_trades: int = 30   # Flag folds below this as statistically thin
    min_test_trades: int = 5     # Minimum trades to include fold in aggregates


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_metrics: PerformanceMetrics
    test_metrics: PerformanceMetrics
    is_statistically_thin: bool
    walk_forward_efficiency: float  # OOS Sharpe / IS Sharpe (> 0.5 is good)
    notes: str = ""


@dataclass
class WalkForwardReport:
    config: WalkForwardConfig
    total_folds: int
    valid_folds: int                      # Folds with enough trades to be meaningful
    fold_results: list[FoldResult]

    # Aggregate in-sample stats
    is_avg_win_rate: float
    is_avg_r: float
    is_avg_sharpe: float

    # Aggregate out-of-sample stats
    oos_avg_win_rate: float
    oos_avg_r: float
    oos_avg_sharpe: float

    # Degradation metrics
    avg_wfe: float                        # Walk-forward efficiency (>0.5 = strategy has edge)
    edge_decay_detected: bool             # True if OOS metrics declining fold-over-fold
    oos_profit_factor: float
    notes: str = ""


# ---------------------------------------------------------------------------
# Walk-forward validator
# ---------------------------------------------------------------------------

class WalkForwardValidator:
    """
    Runs walk-forward validation on the full pipeline backtester.

    Usage:
        validator = WalkForwardValidator(pipeline_bt, wf_config)
        report = await validator.run(
            start_date=datetime(2024, 1, 1, ...),
            end_date=datetime(2024, 12, 31, ...),
            symbol_universe=["AAPL", "NVDA", ...],
        )
    """

    def __init__(
        self,
        backtester: PipelineBacktester,
        config: WalkForwardConfig | None = None,
    ) -> None:
        self._bt = backtester
        self._cfg = config or WalkForwardConfig()

    async def run(
        self,
        start_date: datetime,
        end_date: datetime,
        symbol_universe: list[str],
    ) -> WalkForwardReport:
        """Run all folds and compile the walk-forward report."""
        folds = self._build_fold_schedule(start_date, end_date)

        if not folds:
            raise ValueError(
                f"Date range {start_date.date()} – {end_date.date()} is too short "
                f"for walk-forward with {self._cfg.train_months}m train + {self._cfg.test_months}m test."
            )

        logger.info(
            "Walk-forward: {} folds | {}m train / {}m test | {} symbols",
            len(folds), self._cfg.train_months, self._cfg.test_months, len(symbol_universe),
        )

        fold_results: list[FoldResult] = []
        for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
            logger.info(
                "Fold {}/{}: train {} → {} | test {} → {}",
                i + 1, len(folds),
                train_start.date(), train_end.date(),
                test_start.date(), test_end.date(),
            )

            fold = await self._run_fold(
                fold_index=i + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                symbol_universe=symbol_universe,
            )
            fold_results.append(fold)

        return self._compile_report(fold_results)

    # ------------------------------------------------------------------
    # Fold scheduling
    # ------------------------------------------------------------------

    def _build_fold_schedule(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate (train_start, train_end, test_start, test_end) tuples.
        Rolls forward by step_months each iteration.
        """
        folds = []
        train_start = start_date

        while True:
            train_end = _add_months(train_start, self._cfg.train_months) - timedelta(days=1)
            test_start = train_end + timedelta(days=1)
            test_end = _add_months(test_start, self._cfg.test_months) - timedelta(days=1)

            if test_end > end_date:
                break

            folds.append((train_start, train_end, test_start, test_end))
            train_start = _add_months(train_start, self._cfg.step_months)

        return folds

    # ------------------------------------------------------------------
    # Single fold execution
    # ------------------------------------------------------------------

    async def _run_fold(
        self,
        fold_index: int,
        train_start: datetime,
        train_end: datetime,
        test_start: datetime,
        test_end: datetime,
        symbol_universe: list[str],
    ) -> FoldResult:
        # In-sample (train)
        is_metrics, _ = await self._bt.run(train_start, train_end, symbol_universe)

        # Out-of-sample (test)
        oos_metrics, _ = await self._bt.run(test_start, test_end, symbol_universe)

        is_thin = (
            is_metrics.total_trades < self._cfg.min_train_trades
            or oos_metrics.total_trades < self._cfg.min_test_trades
        )

        # Walk-forward efficiency = OOS Sharpe / IS Sharpe
        wfe = 0.0
        if is_metrics.sharpe_ratio > 0:
            wfe = oos_metrics.sharpe_ratio / is_metrics.sharpe_ratio
        elif oos_metrics.sharpe_ratio > 0:
            wfe = 1.0  # IS was flat but OOS was positive — unusual but okay

        notes = []
        if is_thin:
            notes.append(f"Thin: IS={is_metrics.total_trades} OOS={oos_metrics.total_trades} trades")
        if wfe < 0.3:
            notes.append(f"Low WFE={wfe:.2f} — possible overfit")
        if oos_metrics.win_rate < 0.35:
            notes.append(f"Low OOS win rate {oos_metrics.win_rate:.1%}")

        logger.info(
            "Fold {} result: IS WR={:.1%} R={:.2f} Sharpe={:.2f} | OOS WR={:.1%} R={:.2f} Sharpe={:.2f} | WFE={:.2f}",
            fold_index,
            is_metrics.win_rate, is_metrics.avg_r_multiple, is_metrics.sharpe_ratio,
            oos_metrics.win_rate, oos_metrics.avg_r_multiple, oos_metrics.sharpe_ratio,
            wfe,
        )

        return FoldResult(
            fold_index=fold_index,
            train_start=train_start.strftime("%Y-%m-%d"),
            train_end=train_end.strftime("%Y-%m-%d"),
            test_start=test_start.strftime("%Y-%m-%d"),
            test_end=test_end.strftime("%Y-%m-%d"),
            train_metrics=is_metrics,
            test_metrics=oos_metrics,
            is_statistically_thin=is_thin,
            walk_forward_efficiency=round(wfe, 4),
            notes=" | ".join(notes),
        )

    # ------------------------------------------------------------------
    # Report compilation
    # ------------------------------------------------------------------

    def _compile_report(self, folds: list[FoldResult]) -> WalkForwardReport:
        valid = [f for f in folds if not f.is_statistically_thin]

        def _avg(vals: list[float]) -> float:
            return float(np.mean(vals)) if vals else 0.0

        is_win_rates  = [f.train_metrics.win_rate      for f in valid]
        is_avg_rs     = [f.train_metrics.avg_r_multiple for f in valid]
        is_sharpes    = [f.train_metrics.sharpe_ratio   for f in valid]
        oos_win_rates = [f.test_metrics.win_rate        for f in valid]
        oos_avg_rs    = [f.test_metrics.avg_r_multiple  for f in valid]
        oos_sharpes   = [f.test_metrics.sharpe_ratio    for f in valid]
        wfes          = [f.walk_forward_efficiency       for f in valid]

        # Edge decay: is OOS avg_r declining across folds?
        edge_decay = False
        if len(oos_avg_rs) >= 3:
            # Simple linear regression slope on OOS R
            x = np.arange(len(oos_avg_rs), dtype=float)
            slope = float(np.polyfit(x, oos_avg_rs, 1)[0])
            edge_decay = slope < -0.05  # Declining by more than 0.05R per fold

        # OOS profit factor: use net_pnl-weighted approach across folds
        # avg_winner_r * winning_trades approximates gross profit in R-units
        # We normalise to per-trade R so cross-fold aggregation is valid
        total_oos_gross_profit_r = sum(
            f.test_metrics.avg_winner_r * f.test_metrics.winning_trades
            for f in valid if f.test_metrics.winning_trades > 0
        )
        total_oos_gross_loss_r = sum(
            abs(f.test_metrics.avg_loser_r) * f.test_metrics.losing_trades
            for f in valid if f.test_metrics.losing_trades > 0
        )
        oos_pf = (
            total_oos_gross_profit_r / total_oos_gross_loss_r
            if total_oos_gross_loss_r > 0 else 0.0
        )

        notes = []
        avg_wfe = _avg(wfes)
        if avg_wfe >= 0.5:
            notes.append(f"✓ Good WFE={avg_wfe:.2f} — strategy has genuine OOS edge")
        elif avg_wfe >= 0.3:
            notes.append(f"⚠ Marginal WFE={avg_wfe:.2f} — strategy may be overfit")
        else:
            notes.append(f"✗ Poor WFE={avg_wfe:.2f} — strategy likely overfit to history")

        if edge_decay:
            notes.append("⚠ Edge decay detected — OOS performance declining over time")

        return WalkForwardReport(
            config=self._cfg,
            total_folds=len(folds),
            valid_folds=len(valid),
            fold_results=folds,
            is_avg_win_rate=_avg(is_win_rates),
            is_avg_r=_avg(is_avg_rs),
            is_avg_sharpe=_avg(is_sharpes),
            oos_avg_win_rate=_avg(oos_win_rates),
            oos_avg_r=_avg(oos_avg_rs),
            oos_avg_sharpe=_avg(oos_sharpes),
            avg_wfe=avg_wfe,
            edge_decay_detected=edge_decay,
            oos_profit_factor=round(oos_pf, 4),
            notes=" | ".join(notes),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_months(dt: datetime, months: int) -> datetime:
    """Add calendar months to a datetime."""
    month = dt.month + months
    year = dt.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    day = min(dt.day, _days_in_month(year, month))
    return dt.replace(year=year, month=month, day=day)


def _days_in_month(year: int, month: int) -> int:
    import calendar
    return calendar.monthrange(year, month)[1]
