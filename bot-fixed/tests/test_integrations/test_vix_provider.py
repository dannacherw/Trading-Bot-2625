"""
tests/test_integrations/test_vix_provider.py
Tests for VIX fetcher and monitor.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from market_data.vix_provider import (
    VIXMonitor,
    _fetch_vix_polygon,
    _fetch_vix_yahoo,
)


class TestFetchVixPolygon:
    @pytest.mark.asyncio
    async def test_returns_value_on_success(self):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "tickers": [{"value": 18.5}]
        })
        session = MagicMock()
        session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))
        result = await _fetch_vix_polygon(session, "api_key")
        assert result == 18.5

    @pytest.mark.asyncio
    async def test_returns_none_on_403(self):
        mock_resp = AsyncMock()
        mock_resp.status = 403
        session = MagicMock()
        session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))
        result = await _fetch_vix_polygon(session, "api_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_tickers(self):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"tickers": []})
        session = MagicMock()
        session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))
        result = await _fetch_vix_polygon(session, "api_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        session = MagicMock()
        session.get = MagicMock(side_effect=Exception("Network error"))
        result = await _fetch_vix_polygon(session, "api_key")
        assert result is None


class TestFetchVixYahoo:
    @pytest.mark.asyncio
    async def test_returns_close_from_json(self):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "chart": {"result": [{
                "indicators": {
                    "quote": [{"close": [None, 17.2, 18.1, 19.0]}]
                }
            }]}
        })
        session = MagicMock()
        session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))
        result = await _fetch_vix_yahoo(session)
        assert result == 19.0

    @pytest.mark.asyncio
    async def test_filters_none_closes(self):
        """None values in close list (pre-market gaps) should be ignored."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "chart": {"result": [{
                "indicators": {
                    "quote": [{"close": [16.5, None, None]}]
                }
            }]}
        })
        session = MagicMock()
        session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))
        result = await _fetch_vix_yahoo(session)
        assert result == 16.5

    @pytest.mark.asyncio
    async def test_returns_none_on_http_error(self):
        mock_resp = AsyncMock()
        mock_resp.status = 429
        session = MagicMock()
        session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))
        result = await _fetch_vix_yahoo(session)
        assert result is None


class TestVIXMonitor:
    def test_initial_state(self):
        monitor = VIXMonitor(polygon_api_key="test")
        assert monitor.last_vix is None
        assert monitor.last_spy_move_pct is None
        assert monitor.last_fetched is None

    @pytest.mark.asyncio
    async def test_poll_updates_kill_switch(self):
        kill_switch = MagicMock()
        kill_switch.update_market_conditions = MagicMock()
        monitor = VIXMonitor(polygon_api_key="test", kill_switch=kill_switch)

        with (
            patch(
                "market_data.vix_provider._fetch_vix_polygon",
                new_callable=AsyncMock,
                return_value=22.5,
            ),
            patch(
                "market_data.vix_provider._fetch_vix_yahoo",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            monitor._session = AsyncMock()
            monitor._session.get = MagicMock(return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=AsyncMock(
                    status=200,
                    json=AsyncMock(return_value={
                        "chart": {"result": [{
                            "meta": {"regularMarketPrice": 450.0},
                            "indicators": {"quote": [{"open": [448.0]}]},
                        }]}
                    })
                )),
                __aexit__=AsyncMock(return_value=False),
            ))
            await monitor._poll()

        assert monitor.last_vix == 22.5
        kill_switch.update_market_conditions.assert_called_once()
        call_kwargs = kill_switch.update_market_conditions.call_args[1]
        assert call_kwargs["vix_level"] == 22.5

    @pytest.mark.asyncio
    async def test_polygon_failure_falls_back_to_yahoo(self):
        monitor = VIXMonitor(polygon_api_key="test")
        monitor._session = AsyncMock()

        with (
            patch(
                "market_data.vix_provider._fetch_vix_polygon",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "market_data.vix_provider._fetch_vix_yahoo",
                new_callable=AsyncMock,
                return_value=21.3,
            ),
            patch.object(monitor, "_fetch_spy", new_callable=AsyncMock, return_value=(None, None)),
        ):
            await monitor._poll()

        assert monitor.last_vix == 21.3

    @pytest.mark.asyncio
    async def test_stale_value_preserved_on_fetch_failure(self):
        monitor = VIXMonitor(polygon_api_key="test")
        monitor._last_vix = 19.0  # Previously known value
        monitor._session = AsyncMock()

        with (
            patch("market_data.vix_provider._fetch_vix_polygon", new_callable=AsyncMock, return_value=None),
            patch("market_data.vix_provider._fetch_vix_yahoo", new_callable=AsyncMock, return_value=None),
            patch.object(monitor, "_fetch_spy", new_callable=AsyncMock, return_value=(None, None)),
        ):
            await monitor._poll()

        # Stale value preserved
        assert monitor.last_vix == 19.0
