"""
core/exceptions.py
Custom exception hierarchy. All system exceptions inherit from TradingBotError
to enable clean catch-all handling at the orchestrator level.
"""


class TradingBotError(Exception):
    """Base exception for all trading bot errors."""


# ---------------------------------------------------------------------------
# Data / Market Errors
# ---------------------------------------------------------------------------

class MarketDataError(TradingBotError):
    """Raised when market data cannot be fetched or is malformed."""


class DataProviderError(MarketDataError):
    """Raised by a data provider client (e.g. Polygon)."""


class InsufficientDataError(MarketDataError):
    """Raised when there are not enough bars/quotes for computation."""


class WebSocketError(MarketDataError):
    """Raised on WebSocket connection or subscription failures."""


class UniverseBuildError(MarketDataError):
    """Raised when the tradable universe cannot be constructed."""


# ---------------------------------------------------------------------------
# Scanner Errors
# ---------------------------------------------------------------------------

class ScannerError(TradingBotError):
    """Raised by the premarket scanner."""


class ScoringError(ScannerError):
    """Raised when composite score cannot be computed."""


# ---------------------------------------------------------------------------
# Strategy Errors
# ---------------------------------------------------------------------------

class StrategyError(TradingBotError):
    """Raised by strategy logic."""


class SignalGenerationError(StrategyError):
    """Raised when a signal cannot be generated."""


class VWAPComputationError(StrategyError):
    """Raised when VWAP cannot be computed."""


# ---------------------------------------------------------------------------
# Risk Errors
# ---------------------------------------------------------------------------

class RiskError(TradingBotError):
    """Raised by the risk engine."""


class DailyLossLimitBreached(RiskError):
    """Raised when the daily loss limit is hit — halts trading."""


class PositionLimitError(RiskError):
    """Raised when max open positions would be exceeded."""


class PositionSizingError(RiskError):
    """Raised when a valid position size cannot be calculated."""


# ---------------------------------------------------------------------------
# Execution / Broker Errors
# ---------------------------------------------------------------------------

class ExecutionError(TradingBotError):
    """Raised by the execution engine."""


class BrokerError(ExecutionError):
    """Raised when a broker API call fails."""


class OrderRejectedError(ExecutionError):
    """Raised when the broker rejects an order."""

    def __init__(self, message: str, order_id: str | None = None) -> None:
        super().__init__(message)
        self.order_id = order_id


class OrderTimeoutError(ExecutionError):
    """Raised when an order fill confirmation times out."""


class AuthenticationError(BrokerError):
    """Raised when broker OAuth authentication fails."""


class RateLimitError(BrokerError):
    """Raised when broker API rate limits are exceeded."""


# ---------------------------------------------------------------------------
# Database Errors
# ---------------------------------------------------------------------------

class DatabaseError(TradingBotError):
    """Raised when a database operation fails."""


class MigrationError(DatabaseError):
    """Raised when a schema migration fails."""


# ---------------------------------------------------------------------------
# Configuration Errors
# ---------------------------------------------------------------------------

class ConfigurationError(TradingBotError):
    """Raised when configuration is missing or invalid."""
