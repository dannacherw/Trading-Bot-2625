"""
backtesting/weight_calibrator.py
Logistic regression weight calibrator for scanner scoring.

Uses historical scan outcomes (did this stock follow through? yes/no)
to fit a logistic regression model, replacing heuristic scoring weights
with empirically-derived coefficients.

Workflow:
    1. Accumulate scan_results + trade outcomes in the database over time
    2. Run calibrate() to fit the model on the historical data
    3. Export weights as a ScoringWeights dataclass
    4. Optionally write directly to scanner_config.yaml

Requires scipy (available) and numpy (available).
Minimum recommended sample size: 100 resolved scan results.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # Logistic sigmoid

from core.config import ScoringWeights


@dataclass
class CalibrationDataPoint:
    """Feature vector + label for a single scan result."""
    symbol: str
    date: str

    # Normalised features (same dimensions as scoring formula)
    trend_quality: float
    relative_volume: float
    float_rotation_pct: float    # 0.0 if unavailable
    gap_pct: float
    catalyst_score: float
    premarket_dollar_volume: float
    spread_inverse: float

    # Label: did the stock make a profitable follow-through?
    # True = stock moved +1.5% or more from open within 2 hours
    followed_through: bool


@dataclass
class CalibrationResult:
    """Output of a calibration run."""
    weights: ScoringWeights
    n_samples: int
    train_accuracy: float
    log_loss: float
    feature_importances: dict[str, float]
    converged: bool
    notes: str = ""


class ScoringWeightCalibrator:
    """
    Fits logistic regression on historical scan outcomes to produce
    empirically-calibrated scoring weights.

    The scoring formula is treated as a linear model:
        P(follow_through) = sigmoid(w · x)
    where x is the normalised feature vector and w are the weights.

    Constraints:
        - All weights must be positive (each feature contributes positively)
        - Weights must sum to 1.0 (to maintain the [0,1] composite score range)

    These constraints are enforced via projected gradient + softmax.
    """

    FEATURE_NAMES = [
        "trend_quality",
        "relative_volume",
        "float_rotation_pct",
        "gap_pct",
        "catalyst_score",
        "premarket_dollar_volume",
        "spread_inverse",
    ]
    N_FEATURES = len(FEATURE_NAMES)
    MIN_SAMPLES = 50  # Hard minimum; 150+ recommended

    def __init__(self, min_samples: int = MIN_SAMPLES) -> None:
        self._min_samples = min_samples

    # ------------------------------------------------------------------
    # Main calibration entry point
    # ------------------------------------------------------------------

    def calibrate(self, data: list[CalibrationDataPoint]) -> CalibrationResult:
        """
        Fit logistic regression on the provided data points.

        Args:
            data: List of CalibrationDataPoint with normalised features and labels

        Returns:
            CalibrationResult with fitted weights and diagnostics
        """
        if len(data) < self._min_samples:
            return CalibrationResult(
                weights=ScoringWeights(),  # Return defaults
                n_samples=len(data),
                train_accuracy=0.0,
                log_loss=float("inf"),
                feature_importances={},
                converged=False,
                notes=(
                    f"Insufficient data ({len(data)} samples, need {self._min_samples}). "
                    "Using heuristic defaults."
                ),
            )

        X, y = self._build_matrices(data)
        weights_raw, result = self._fit(X, y)

        # Normalise to sum=1 with all-positive constraint
        weights_positive = np.maximum(weights_raw, 0.01)  # Floor at 1%
        weights_normalised = weights_positive / weights_positive.sum()

        # Evaluate on training set
        probs = expit(X @ weights_normalised)
        predictions = (probs >= 0.5).astype(int)
        accuracy = float(np.mean(predictions == y))
        ll = self._log_loss(y, probs)

        # Feature importances (normalised weight magnitudes)
        importances = {
            name: float(w)
            for name, w in zip(self.FEATURE_NAMES, weights_normalised)
        }

        fitted_weights = ScoringWeights(
            trend_quality=round(float(weights_normalised[0]), 4),
            relative_volume=round(float(weights_normalised[1]), 4),
            float_rotation_pct=round(float(weights_normalised[2]), 4),
            gap_pct=round(float(weights_normalised[3]), 4),
            catalyst_score=round(float(weights_normalised[4]), 4),
            premarket_dollar_volume=round(float(weights_normalised[5]), 4),
            spread_inverse=round(float(weights_normalised[6]), 4),
        )

        return CalibrationResult(
            weights=fitted_weights,
            n_samples=len(data),
            train_accuracy=round(accuracy, 4),
            log_loss=round(ll, 4),
            feature_importances=importances,
            converged=result.success,
            notes=f"scipy.optimize status: {result.message}",
        )

    # ------------------------------------------------------------------
    # Internal fitting
    # ------------------------------------------------------------------

    def _build_matrices(
        self, data: list[CalibrationDataPoint]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build feature matrix X (n, 7) and label vector y (n,)."""
        rows = []
        labels = []
        for d in data:
            row = [
                d.trend_quality,
                d.relative_volume,
                d.float_rotation_pct,
                d.gap_pct,
                d.catalyst_score,
                d.premarket_dollar_volume,
                d.spread_inverse,
            ]
            rows.append(row)
            labels.append(1 if d.followed_through else 0)
        return np.array(rows, dtype=float), np.array(labels, dtype=float)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, Any]:
        """
        Minimise negative log-likelihood with L2 regularisation.
        Start from heuristic weights for a warm start.
        """
        w0 = np.array([0.22, 0.20, 0.14, 0.14, 0.12, 0.12, 0.06])  # Heuristic init

        def neg_log_likelihood(w: np.ndarray) -> float:
            w_pos = np.maximum(w, 1e-6)
            w_norm = w_pos / w_pos.sum()
            probs = expit(X @ w_norm)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            ll = y * np.log(probs) + (1 - y) * np.log(1 - probs)
            l2 = 0.01 * np.sum(w_norm ** 2)  # Light regularisation
            return float(-ll.mean() + l2)

        result = minimize(
            neg_log_likelihood,
            w0,
            method="L-BFGS-B",
            bounds=[(0.01, 1.0)] * self.N_FEATURES,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        return result.x, result

    @staticmethod
    def _log_loss(y: np.ndarray, probs: np.ndarray) -> float:
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return float(-(y * np.log(probs) + (1 - y) * np.log(1 - probs)).mean())

    # ------------------------------------------------------------------
    # Data preparation from database
    # ------------------------------------------------------------------

    @staticmethod
    async def load_from_database(
        db_manager: "DatabaseManager",  # type: ignore[name-defined]
        follow_through_pct: float = 1.5,
        follow_through_minutes: int = 120,
    ) -> list[CalibrationDataPoint]:
        """
        Load historical scan results + trade outcomes from the database.
        Labels each scan result as positive (followed through) or negative.

        A scan result is positive if the stock moved ≥ follow_through_pct %
        from its open price within follow_through_minutes minutes of market open.

        This requires both scan_results and trades tables to have data.
        """
        # Load scan results
        scan_rows = await db_manager.fetch_all(
            """
            SELECT symbol, scanned_at, composite_score, gap_pct,
                   relative_volume, pm_dollar_vol, spread_pct, raw_json
            FROM scan_results
            WHERE passes_filters = 1
            ORDER BY scanned_at ASC
            """
        )

        # Load trades for outcome labelling
        trade_rows = await db_manager.fetch_all(
            """
            SELECT symbol, entry_time, exit_price, entry_price, r_multiple
            FROM trades
            """
        )

        # Build a trade outcome lookup: (symbol, date) → r_multiple
        trade_outcomes: dict[tuple[str, str], float] = {}
        for t in trade_rows:
            date_str = t["entry_time"][:10]
            key = (t["symbol"], date_str)
            trade_outcomes[key] = t["r_multiple"]

        points = []
        for row in scan_rows:
            try:
                raw = json.loads(row["raw_json"])
                metrics = raw.get("metrics", {})
                date_str = row["scanned_at"][:10]
                symbol = row["symbol"]

                # Label: did we have a profitable trade on this day?
                r = trade_outcomes.get((symbol, date_str))
                followed_through = r is not None and r >= 1.0

                # Compute normalised features (using fixed-range normalisation)
                from scanner.scoring import _fixed_range_normalise
                from core.models import PremarketMetrics
                from datetime import datetime

                pm = PremarketMetrics(
                    symbol=symbol,
                    computed_at=datetime.now(tz=timezone.utc),
                    prev_close=metrics.get("prev_close", 0),
                    premarket_open=metrics.get("premarket_open", 0),
                    premarket_high=metrics.get("premarket_high", 0),
                    premarket_low=metrics.get("premarket_low", 0),
                    premarket_last=metrics.get("premarket_last", 0),
                    premarket_volume=metrics.get("premarket_volume", 0),
                    premarket_dollar_volume=metrics.get("premarket_dollar_volume", 0),
                    gap_pct=metrics.get("gap_pct", 0),
                    relative_volume=metrics.get("relative_volume", 1),
                    spread_pct=metrics.get("spread_pct", 0),
                    range_pct=metrics.get("range_pct", 0),
                    range_position=metrics.get("range_position", 0.5),
                    trend_quality=metrics.get("trend_quality", 0.5),
                    avg_daily_dollar_volume=metrics.get("avg_daily_dollar_volume", 0),
                    pm_float_rotation_pct=metrics.get("pm_float_rotation_pct"),
                )

                catalyst_score = raw.get("composite_score", 0.0) * 0.1  # Rough proxy
                normed = _fixed_range_normalise(pm, catalyst_score)

                points.append(CalibrationDataPoint(
                    symbol=symbol,
                    date=date_str,
                    trend_quality=normed.trend_quality,
                    relative_volume=normed.relative_volume,
                    float_rotation_pct=normed.float_rotation_pct,
                    gap_pct=normed.gap_pct,
                    catalyst_score=normed.catalyst_score,
                    premarket_dollar_volume=normed.premarket_dollar_volume,
                    spread_inverse=normed.spread_inverse,
                    followed_through=followed_through,
                ))
            except Exception:
                continue

        return points

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @staticmethod
    def export_to_yaml(result: CalibrationResult, path: str = "config/scanner_config.yaml") -> None:
        """
        Write calibrated weights back into scanner_config.yaml.
        Preserves all other settings.
        """
        import yaml
        config_path = Path(path)
        if not config_path.exists():
            return

        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        w = result.weights
        config["scoring_weights"] = {
            "trend_quality": w.trend_quality,
            "relative_volume": w.relative_volume,
            "float_rotation_pct": w.float_rotation_pct,
            "gap_pct": w.gap_pct,
            "catalyst_score": w.catalyst_score,
            "premarket_dollar_volume": w.premarket_dollar_volume,
            "spread_inverse": w.spread_inverse,
        }
        config["_calibration_metadata"] = {
            "n_samples": result.n_samples,
            "train_accuracy": result.train_accuracy,
            "log_loss": result.log_loss,
            "converged": result.converged,
        }

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"Calibrated weights written to {config_path}")
        print(f"  Samples: {result.n_samples} | Accuracy: {result.train_accuracy:.1%} | LogLoss: {result.log_loss:.4f}")
        print(f"  Feature importances: {result.feature_importances}")
