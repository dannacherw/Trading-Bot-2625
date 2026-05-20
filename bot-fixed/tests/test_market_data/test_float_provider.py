"""
tests/test_market_data/test_float_provider.py
Tests for the CompositeFloatProvider (Polygon + Yahoo fallback).
"""
from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from market_data.float_provider import (
    CompositeFloatProvider,
    _FloatCache,
    MISSING,
    _fetch_float_polygon,
    _fetch_float_yahoo,
)


# ---------------------------------------------------------------------------
# _FloatCache tests
# ---------------------------------------------------------------------------

class TestFloatCache:
    def test_miss_on_empty(self):
        cache = _FloatCache()
        result = cache.get("AAPL")
        assert result is MISSING

    def test_hit_same_day(self):
        cache = _FloatCache()
        cache.set("AAPL", 15_000_000)
        result = cache.get("AAPL")
        assert result == 15_000_000

    def test_hit_with_none_value(self):
        """None is a valid cached value (float unknown)."""
        cache = _FloatCache()
        cache.set("AAPL", None)
        result = cache.get("AAPL")
        assert result is None  # Not a MISSING — None was explicitly cached

    def test_miss_on_stale_date(self):
        cache = _FloatCache()
        cache._store["AAPL"] = (15_000_000, "2020-01-01")  # Old date
        result = cache.get("AAPL")
        assert result is MISSING

    def test_overwrite(self):
        cache = _FloatCache()
        cache.set("AAPL", 10_000_000)
        cache.set("AAPL", 20_000_000)
        assert cache.get("AAPL") == 20_000_000


# ---------------------------------------------------------------------------
# _fetch_float_polygon tests
# ---------------------------------------------------------------------------

class TestFetchFloatPolygon:
    @pytest.mark.asyncio
    async def test_returns_share_class_outstanding(self):
        mock_session = MagicMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "results": {
                "share_class_shares_outstanding": 50_000_000,
                "weighted_shares_outstanding": 100_000_000,
            }
        })
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=False),
        ))
        result = await _fetch_float_polygon("AAPL", mock_session, "test_key")
        assert result == 50_000_000

    @pytest.mark.asyncio
    async def test_falls_back_to_weighted_outstanding(self):
        mock_session = MagicMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "results": {
                "share_class_shares_outstanding": None,
                "weighted_shares_outstanding": 75_000_000,
            }
        })
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=False),
        ))
        result = await _fetch_float_polygon("AAPL", mock_session, "test_key")
        assert result == 75_000_000

    @pytest.mark.asyncio
    async def test_returns_none_on_404(self):
        mock_session = MagicMock()
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=False),
        ))
        result = await _fetch_float_polygon("FAKE", mock_session, "test_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=Exception("Network error"))
        result = await _fetch_float_polygon("AAPL", mock_session, "key")
        assert result is None


# ---------------------------------------------------------------------------
# CompositeFloatProvider tests
# ---------------------------------------------------------------------------

class TestCompositeFloatProvider:
    @pytest.mark.asyncio
    async def test_polygon_success_no_yahoo_call(self):
        """When Polygon returns a value, Yahoo should not be called."""
        provider = CompositeFloatProvider(polygon_api_key="test")

        with (
            patch("market_data.float_provider._fetch_float_polygon", new_callable=AsyncMock) as mock_poly,
            patch("market_data.float_provider._fetch_float_yahoo", new_callable=AsyncMock) as mock_yahoo,
            patch("market_data.float_provider._cache", new=_FloatCache()),
        ):
            mock_poly.return_value = 25_000_000
            mock_yahoo.return_value = 30_000_000

            async with provider:
                result = await provider.get_float_shares("AAPL")

        assert result == 25_000_000
        mock_yahoo.assert_not_called()

    @pytest.mark.asyncio
    async def test_yahoo_fallback_when_polygon_fails(self):
        """When Polygon returns None, Yahoo should be called."""
        provider = CompositeFloatProvider(polygon_api_key="test")

        with (
            patch("market_data.float_provider._fetch_float_polygon", new_callable=AsyncMock) as mock_poly,
            patch("market_data.float_provider._fetch_float_yahoo", new_callable=AsyncMock) as mock_yahoo,
            patch("market_data.float_provider._cache", new=_FloatCache()),
        ):
            mock_poly.return_value = None
            mock_yahoo.return_value = 18_000_000

            async with provider:
                result = await provider.get_float_shares("TSLA")

        assert result == 18_000_000
        mock_yahoo.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_hit_skips_api_calls(self):
        """Second call for same symbol should use cache, not hit APIs."""
        fresh_cache = _FloatCache()
        fresh_cache.set("NVDA", 12_000_000)
        provider = CompositeFloatProvider(polygon_api_key="test")

        with (
            patch("market_data.float_provider._fetch_float_polygon", new_callable=AsyncMock) as mock_poly,
            patch("market_data.float_provider._cache", fresh_cache),
        ):
            async with provider:
                result = await provider.get_float_shares("NVDA")

        assert result == 12_000_000
        mock_poly.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_fetch(self):
        provider = CompositeFloatProvider(polygon_api_key="test")
        symbols = ["AAPL", "TSLA", "NVDA"]

        with (
            patch("market_data.float_provider._fetch_float_polygon", new_callable=AsyncMock) as mock_poly,
            patch("market_data.float_provider._fetch_float_yahoo", new_callable=AsyncMock) as mock_yahoo,
            patch("market_data.float_provider._cache", new=_FloatCache()),
        ):
            mock_poly.side_effect = [10_000_000, None, 8_000_000]
            mock_yahoo.return_value = 20_000_000

            async with provider:
                result = await provider.get_float_shares_batch(symbols)

        assert result["AAPL"] == 10_000_000
        assert result["TSLA"] == 20_000_000  # Yahoo fallback
        assert result["NVDA"] == 8_000_000

    @pytest.mark.asyncio
    async def test_both_sources_fail_returns_none(self):
        provider = CompositeFloatProvider(polygon_api_key="test")

        with (
            patch("market_data.float_provider._fetch_float_polygon", new_callable=AsyncMock, return_value=None),
            patch("market_data.float_provider._fetch_float_yahoo", new_callable=AsyncMock, return_value=None),
            patch("market_data.float_provider._cache", new=_FloatCache()),
        ):
            async with provider:
                result = await provider.get_float_shares("UNKNOWN")

        assert result is None
