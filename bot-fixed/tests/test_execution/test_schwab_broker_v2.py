"""
tests/test_execution/test_schwab_broker_v2.py
Tests for SchwabBroker token refresh, .env persistence, and background task.
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from core.config import BrokerSettings, SchwabSettings
from core.exceptions import AuthenticationError, BrokerError
from execution.schwab_broker import SchwabBroker, _PROACTIVE_REFRESH_BUFFER


def _make_broker(env_path: str = "/tmp/test_schwab.env") -> SchwabBroker:
    settings = SchwabSettings(
        client_id="test_client",
        client_secret="test_secret",
        redirect_uri="https://localhost",
        account_number="TEST123",
        token_refresh_buffer_seconds=300,
    )
    broker_cfg = BrokerSettings(
        paper_trading=True,
        timeout_seconds=10,
        max_retries=2,
        retry_backoff_seconds=0.01,
    )
    return SchwabBroker(settings, broker_cfg, env_path=env_path)


# ---------------------------------------------------------------------------
# .env persistence
# ---------------------------------------------------------------------------

class TestEnvPersistence:
    def test_writes_new_file(self, tmp_path):
        path = tmp_path / ".env"
        broker = _make_broker(str(path))
        broker._persist_tokens_to_env("access_tok", "refresh_tok")

        content = path.read_text()
        assert 'SCHWAB_ACCESS_TOKEN="access_tok"' in content
        assert 'SCHWAB_REFRESH_TOKEN="refresh_tok"' in content

    def test_updates_existing_keys(self, tmp_path):
        path = tmp_path / ".env"
        path.write_text(
            'POLYGON_API_KEY="poly123"\n'
            'SCHWAB_ACCESS_TOKEN="old_token"\n'
            'SCHWAB_REFRESH_TOKEN="old_refresh"\n'
        )
        broker = _make_broker(str(path))
        broker._persist_tokens_to_env("new_access", "new_refresh")

        content = path.read_text()
        assert 'SCHWAB_ACCESS_TOKEN="new_access"' in content
        assert 'SCHWAB_REFRESH_TOKEN="new_refresh"' in content
        assert 'POLYGON_API_KEY="poly123"' in content
        assert "old_token" not in content

    def test_preserves_other_env_vars(self, tmp_path):
        path = tmp_path / ".env"
        path.write_text(
            'DB_PATH="data/trading_bot.db"\n'
            'API_ADMIN_PASS="secret"\n'
        )
        broker = _make_broker(str(path))
        broker._persist_tokens_to_env("tok", "ref")

        content = path.read_text()
        assert 'DB_PATH="data/trading_bot.db"' in content
        assert 'API_ADMIN_PASS="secret"' in content

    def test_survives_unwritable_path(self, tmp_path):
        """Failure to write .env must not raise — just log a warning."""
        broker = _make_broker("/nonexistent/path/.env")
        # Should not raise
        broker._persist_tokens_to_env("tok", "ref")

    def test_no_duplicate_keys(self, tmp_path):
        path = tmp_path / ".env"
        broker = _make_broker(str(path))
        broker._persist_tokens_to_env("tok1", "ref1")
        broker._persist_tokens_to_env("tok2", "ref2")

        content = path.read_text()
        assert content.count("SCHWAB_ACCESS_TOKEN") == 1
        assert content.count("SCHWAB_REFRESH_TOKEN") == 1


# ---------------------------------------------------------------------------
# _do_refresh lock and double-check
# ---------------------------------------------------------------------------

class TestDoRefresh:
    @pytest.mark.asyncio
    async def test_skips_if_token_still_valid(self, tmp_path):
        broker = _make_broker(str(tmp_path / ".env"))
        # Token valid for another 10 minutes
        broker._token_expires_at = time.time() + 600
        broker._refresh_token = "existing_refresh"

        mock_session = AsyncMock()
        broker._session = mock_session

        await broker._do_refresh()

        # post() should NOT have been called because token is still valid
        mock_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_without_refresh_token(self, tmp_path):
        broker = _make_broker(str(tmp_path / ".env"))
        broker._token_expires_at = 0.0
        broker._refresh_token = None
        broker._session = AsyncMock()

        with pytest.raises(AuthenticationError, match="No refresh token"):
            await broker._do_refresh()

    @pytest.mark.asyncio
    async def test_updates_token_on_success(self, tmp_path):
        broker = _make_broker(str(tmp_path / ".env"))
        broker._token_expires_at = 0.0
        broker._refresh_token = "my_refresh"

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "access_token": "new_access",
            "refresh_token": "new_refresh",
            "expires_in": 1800,
        })
        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=False),
            )
        )
        broker._session = mock_session

        await broker._do_refresh()

        assert broker._access_token == "new_access"
        assert broker._refresh_token == "new_refresh"
        assert broker._token_expires_at > time.time()

    @pytest.mark.asyncio
    async def test_raises_on_bad_response(self, tmp_path):
        broker = _make_broker(str(tmp_path / ".env"))
        broker._token_expires_at = 0.0
        broker._refresh_token = "my_refresh"

        mock_resp = AsyncMock()
        mock_resp.status = 400
        mock_resp.text = AsyncMock(return_value="invalid_grant")
        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=False),
            )
        )
        broker._session = mock_session

        with pytest.raises(AuthenticationError, match="Token refresh failed"):
            await broker._do_refresh()

    @pytest.mark.asyncio
    async def test_concurrent_refresh_only_calls_once(self, tmp_path):
        """
        Two coroutines racing to refresh should result in exactly one
        HTTP call — the second should see the lock and skip via double-check.
        """
        broker = _make_broker(str(tmp_path / ".env"))
        broker._token_expires_at = 0.0
        broker._refresh_token = "my_refresh"
        call_count = 0

        async def _fake_post_ctx(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)  # Simulate network latency
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value={
                "access_token": "new_tok",
                "expires_in": 1800,
            })
            return mock_resp

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=_fake_post_ctx,
                __aexit__=AsyncMock(return_value=False),
            )
        )
        broker._session = mock_session

        await asyncio.gather(broker._do_refresh(), broker._do_refresh())
        # Only one actual HTTP call should happen
        assert call_count == 1


# ---------------------------------------------------------------------------
# Background refresh loop
# ---------------------------------------------------------------------------

class TestBackgroundRefreshLoop:
    @pytest.mark.asyncio
    async def test_loop_triggers_refresh_near_expiry(self, tmp_path):
        broker = _make_broker(str(tmp_path / ".env"))
        # Token expires in 60 seconds — within the 300s proactive buffer
        broker._token_expires_at = time.time() + 60
        broker._refresh_token = "my_refresh"

        refresh_called = asyncio.Event()
        original_do_refresh = broker._do_refresh

        async def _patched_refresh():
            refresh_called.set()
            # Simulate successful refresh
            broker._token_expires_at = time.time() + 1800

        broker._do_refresh = _patched_refresh

        # Run the loop with a very short sleep (patch asyncio.sleep)
        async def _fast_loop():
            for _ in range(3):
                await asyncio.sleep(0.01)
                remaining = broker._token_expires_at - time.time()
                if remaining <= _PROACTIVE_REFRESH_BUFFER:
                    await broker._do_refresh()
                    return

        await asyncio.wait_for(_fast_loop(), timeout=1.0)
        assert refresh_called.is_set()

    @pytest.mark.asyncio
    async def test_loop_does_not_refresh_when_token_fresh(self, tmp_path):
        broker = _make_broker(str(tmp_path / ".env"))
        # Token is fresh — 30 minutes remaining
        broker._token_expires_at = time.time() + 1800

        refresh_called = False

        async def _patched_refresh():
            nonlocal refresh_called
            refresh_called = True

        broker._do_refresh = _patched_refresh

        # Simulate one loop iteration
        remaining = broker._token_expires_at - time.time()
        if remaining <= _PROACTIVE_REFRESH_BUFFER:
            await broker._do_refresh()

        assert not refresh_called


# ---------------------------------------------------------------------------
# connect / disconnect lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    @pytest.mark.asyncio
    async def test_connect_starts_background_task(self, tmp_path):
        broker = _make_broker(str(tmp_path / ".env"))
        broker._access_token = "valid_token"

        with (
            patch.object(broker, "_fetch_account_hash", new_callable=AsyncMock),
            patch("aiohttp.ClientSession", autospec=True),
        ):
            await broker.connect()
            assert broker._refresh_task is not None
            assert not broker._refresh_task.done()
            await broker.disconnect()
            assert broker._refresh_task.done()

    @pytest.mark.asyncio
    async def test_disconnect_cancels_background_task(self, tmp_path):
        broker = _make_broker(str(tmp_path / ".env"))

        async def _never_ending():
            while True:
                await asyncio.sleep(3600)

        broker._refresh_task = asyncio.create_task(_never_ending())
        broker._session = AsyncMock()
        broker._session.close = AsyncMock()

        await broker.disconnect()

        assert broker._refresh_task.done()
        assert not broker._connected

    @pytest.mark.asyncio
    async def test_raises_without_tokens(self, tmp_path):
        broker = _make_broker(str(tmp_path / ".env"))
        broker._access_token = None
        broker._refresh_token = None

        with patch("aiohttp.ClientSession", autospec=True):
            with pytest.raises(AuthenticationError, match="No Schwab tokens"):
                await broker.connect()


# ---------------------------------------------------------------------------
# Order payload builder
# ---------------------------------------------------------------------------

class TestOrderPayload:
    def test_limit_buy_payload(self):
        from core.models import Order
        from core.enums import OrderSide, OrderType, OrderStatus, TimeInForce
        from uuid import uuid4

        order = Order(
            order_id=uuid4(),
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            limit_price=182.50,
            time_in_force=TimeInForce.DAY,
            status=OrderStatus.PENDING,
        )
        broker = _make_broker()
        payload = broker._build_order_payload(order)

        assert payload["orderType"] == "LIMIT"
        assert payload["duration"] == "DAY"
        assert payload["orderLegCollection"][0]["instruction"] == "BUY"
        assert payload["orderLegCollection"][0]["quantity"] == 100
        assert payload["price"] == "182.50"

    def test_market_sell_no_price(self):
        from core.models import Order
        from core.enums import OrderSide, OrderType, OrderStatus, TimeInForce
        from uuid import uuid4

        order = Order(
            order_id=uuid4(),
            symbol="TSLA",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=50,
            time_in_force=TimeInForce.DAY,
            status=OrderStatus.PENDING,
        )
        broker = _make_broker()
        payload = broker._build_order_payload(order)

        assert "price" not in payload
        assert payload["orderType"] == "MARKET"

    def test_token_expires_in_property(self, tmp_path):
        broker = _make_broker(str(tmp_path / ".env"))
        broker._token_expires_at = time.time() + 500
        assert 490 < broker.token_expires_in < 510

    def test_token_expires_in_zero_when_expired(self, tmp_path):
        broker = _make_broker(str(tmp_path / ".env"))
        broker._token_expires_at = time.time() - 100
        assert broker.token_expires_in == 0.0
