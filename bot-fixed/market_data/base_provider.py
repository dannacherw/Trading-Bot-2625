"""
market_data/base_provider.py
Abstract base class that all market data providers must implement.
This keeps strategy, scanner, and backtesting code provider-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import AsyncGenerator, Callable

from core.models import Bar, Quote, PremarketMetrics
from core.enums import BarTimeframe


class BaseDataProvider(ABC):
    """
    Contract that every data provider must fulfil.
    Methods are async to support both REST polling and streaming.
    """

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """Initialise HTTP session / authenticate."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Tear down connections gracefully."""

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_tradable_universe(
        self,
        min_price: float = 5.0,
        min_avg_dollar_volume: float = 20_000_000,
        max_results: int = 1500,
    ) -> list[str]:
        """Return list of eligible ticker symbols sorted by dollar volume desc."""

    @abstractmethod
    async def get_ticker_details(self, symbol: str) -> dict:
        """Return raw ticker metadata (type, exchange, name, etc.)."""

    async def get_float_shares(self, symbol: str) -> int | None:
        """
        Return shares outstanding (float) for a symbol.
        Returns None if data is unavailable — callers must handle gracefully.
        Default implementation returns None; override in providers that support it.
        """
        return None

    # ------------------------------------------------------------------
    # Historical data
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_bars(
        self,
        symbol: str,
        timeframe: BarTimeframe,
        start: datetime,
        end: datetime,
    ) -> list[Bar]:
        """Fetch OHLCV bars for a symbol over a date range."""

    @abstractmethod
    async def get_prev_close(self, symbol: str) -> float:
        """Return the previous regular-session closing price."""

    @abstractmethod
    async def get_atr(self, symbol: str, period: int = 14) -> float:
        """Return the ATR over the last `period` daily bars."""

    @abstractmethod
    async def get_avg_daily_dollar_volume(
        self, symbol: str, lookback_days: int = 20
    ) -> float:
        """Return average daily dollar volume over lookback period."""

    # ------------------------------------------------------------------
    # Premarket data
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_premarket_bars(self, symbol: str, date: datetime) -> list[Bar]:
        """Fetch 1-min premarket bars for a given date."""

    @abstractmethod
    async def get_premarket_quote(self, symbol: str) -> Quote:
        """Return current premarket bid/ask quote."""

    @abstractmethod
    async def get_premarket_metrics(
        self, symbol: str, date: datetime
    ) -> PremarketMetrics:
        """Compute and return all premarket metrics in one call."""

    # ------------------------------------------------------------------
    # Intraday / real-time
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_latest_bar(
        self, symbol: str, timeframe: BarTimeframe = BarTimeframe.MINUTE_1
    ) -> Bar:
        """Return the most recently closed bar."""

    @abstractmethod
    async def get_current_quote(self, symbol: str) -> Quote:
        """Return the current NBBO quote."""

    @abstractmethod
    async def get_intraday_bars(
        self, symbol: str, date: datetime, timeframe: BarTimeframe = BarTimeframe.MINUTE_1
    ) -> list[Bar]:
        """Return all intraday bars for a given date."""

    # ------------------------------------------------------------------
    # Streaming (optional — providers that support it)
    # ------------------------------------------------------------------

    async def subscribe_trades(
        self,
        symbols: list[str],
        callback: Callable[[dict], None],
    ) -> None:
        """Subscribe to real-time trade stream. Override in streaming providers."""
        raise NotImplementedError("This provider does not support streaming.")

    async def subscribe_quotes(
        self,
        symbols: list[str],
        callback: Callable[[dict], None],
    ) -> None:
        """Subscribe to real-time quote stream."""
        raise NotImplementedError("This provider does not support streaming.")

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from real-time streams."""

    # ------------------------------------------------------------------
    # News / catalysts (optional)
    # ------------------------------------------------------------------

    async def get_news(
        self,
        symbol: str,
        published_after: datetime | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Return recent news articles for a symbol."""
        return []

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name, e.g. 'polygon'."""

    @property
    def supports_streaming(self) -> bool:
        return False

    @property
    def supports_news(self) -> bool:
        return False
