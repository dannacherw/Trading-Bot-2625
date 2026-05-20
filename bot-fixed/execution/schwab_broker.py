"""
execution/schwab_broker.py
Charles Schwab API broker — v2.

Changes from v1:
  - Background asyncio task proactively refreshes the token 5 min before expiry.
    This prevents mid-session failures when a token expires during order flow.
  - _refresh_lock prevents concurrent token refresh races across coroutines.
  - Refreshed tokens are written back to .env immediately so the next session
    starts with valid credentials without re-running the OAuth flow.
  - 401 responses trigger a single proactive refresh then retry, rather than
    surfacing the error to callers.
  - disconnect() cancels the background task cleanly before closing the session.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import re
import secrets
import time
import urllib.parse
from pathlib import Path
from typing import Any

import aiohttp
from loguru import logger

from core.config import BrokerSettings, SchwabSettings
from core.enums import OrderSide, OrderType, TimeInForce
from core.exceptions import AuthenticationError, BrokerError, RateLimitError
from core.models import Order
from execution.base_broker import BaseBroker

_TRADER_URL = "https://api.schwabapi.com/trader/v1"
_AUTH_URL   = "https://api.schwabapi.com/v1/oauth/token"

# Refresh proactively when fewer than this many seconds remain
_PROACTIVE_REFRESH_BUFFER = 300   # 5 minutes
# Per-request safety buffer — refresh if under 30 s
_REQUEST_REFRESH_BUFFER   = 30


class SchwabBroker(BaseBroker):
    """
    Charles Schwab API broker.

    One-time setup:
        1. url, verifier = broker.get_authorization_url()
        2. Direct user to url; capture the redirected ?code= param.
        3. await broker.exchange_code_for_tokens(code, verifier)
           → tokens written to .env automatically.

    Subsequent sessions:
        Tokens are read from env vars (SCHWAB_ACCESS_TOKEN / SCHWAB_REFRESH_TOKEN)
        at construction. Background task keeps them fresh throughout the session.
    """

    def __init__(
        self,
        schwab_settings: SchwabSettings,
        broker_settings: BrokerSettings,
        env_path: str | Path = ".env",
    ) -> None:
        self._schwab   = schwab_settings
        self._broker   = broker_settings
        self._env_path = Path(env_path)

        self._session: aiohttp.ClientSession | None = None
        self._access_token:  str | None = os.getenv("SCHWAB_ACCESS_TOKEN")
        self._refresh_token: str | None = os.getenv("SCHWAB_REFRESH_TOKEN")
        self._token_expires_at: float   = 0.0
        self._account_hash: str | None  = None
        self._connected                 = False

        self._refresh_lock          = asyncio.Lock()
        self._refresh_task: asyncio.Task | None = None  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._broker.timeout_seconds)
        )

        if self._access_token:
            try:
                await self._fetch_account_hash()
                self._connected = True
                logger.info("SchwabBroker: connected with existing token")
            except Exception:
                logger.info("SchwabBroker: existing token invalid, refreshing")
                await self._do_refresh()
                await self._fetch_account_hash()
                self._connected = True
                logger.info("SchwabBroker: connected after token refresh")
        elif self._refresh_token:
            await self._do_refresh()
            await self._fetch_account_hash()
            self._connected = True
            logger.info("SchwabBroker: connected via refresh token")
        else:
            raise AuthenticationError(
                "No Schwab tokens found. Run: python main.py schwab-auth"
            )

        # Proactive background refresh so we never hit expiry mid-trade
        self._refresh_task = asyncio.create_task(
            self._token_refresh_loop(), name="schwab-token-refresh"
        )

    async def disconnect(self) -> None:
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        logger.info("SchwabBroker: disconnected")

    async def is_connected(self) -> bool:
        return self._connected and self._session is not None

    # ------------------------------------------------------------------
    # Background token refresh loop
    # ------------------------------------------------------------------

    async def _token_refresh_loop(self) -> None:
        """
        Long-lived task: sleeps 60 s between checks, refreshes when the
        token is within _PROACTIVE_REFRESH_BUFFER seconds of expiry.
        Failures are logged but do not crash the task.
        """
        while True:
            try:
                await asyncio.sleep(60)
                remaining = self._token_expires_at - time.time()
                if remaining <= _PROACTIVE_REFRESH_BUFFER:
                    logger.info(
                        "SchwabBroker: token expiring in {:.0f}s — proactive refresh",
                        remaining,
                    )
                    await self._do_refresh()
            except asyncio.CancelledError:
                logger.debug("SchwabBroker: token refresh loop cancelled")
                return
            except Exception as exc:
                logger.error("SchwabBroker: background refresh error: {}", exc)

    # ------------------------------------------------------------------
    # Token refresh (locked to prevent concurrent races)
    # ------------------------------------------------------------------

    async def _do_refresh(self) -> None:
        """
        Refresh the access token under a lock.
        Any coroutine that arrives while a refresh is in progress will
        wait and then skip its own refresh (double-check pattern).
        """
        async with self._refresh_lock:
            # Another coroutine refreshed while we waited — skip
            if time.time() < self._token_expires_at - _REQUEST_REFRESH_BUFFER:
                return

            if not self._refresh_token:
                raise AuthenticationError(
                    "No refresh token. Re-run: python main.py schwab-auth"
                )

            session = self._get_session()
            payload = {
                "grant_type":    "refresh_token",
                "refresh_token": self._refresh_token,
                "client_id":     self._schwab.client_id,
            }
            async with session.post(_AUTH_URL, data=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise AuthenticationError(
                        f"Token refresh failed ({resp.status}): {body[:200]}"
                    )
                tokens = await resp.json()

            self._access_token = tokens["access_token"]
            if "refresh_token" in tokens:
                self._refresh_token = tokens["refresh_token"]
            expires_in = int(tokens.get("expires_in", 1800))
            self._token_expires_at = time.time() + expires_in

            self._persist_tokens_to_env(self._access_token, self._refresh_token or "")
            logger.debug(
                "SchwabBroker: token refreshed — valid {}s, written to {}",
                expires_in, self._env_path,
            )

    async def _ensure_token_valid(self) -> None:
        """Per-request safety gate."""
        if time.time() >= self._token_expires_at - _REQUEST_REFRESH_BUFFER:
            await self._do_refresh()

    # ------------------------------------------------------------------
    # .env persistence
    # ------------------------------------------------------------------

    def _persist_tokens_to_env(self, access_token: str, refresh_token: str) -> None:
        """
        Update SCHWAB_ACCESS_TOKEN and SCHWAB_REFRESH_TOKEN in the .env
        file in-place, preserving every other line. Creates file if absent.
        """
        try:
            path = self._env_path
            existing = path.read_text().splitlines() if path.exists() else []

            def _upsert(key: str, value: str, lines: list[str]) -> list[str]:
                pat = re.compile(rf"^{re.escape(key)}\s*=")
                new = f'{key}="{value}"'
                for i, ln in enumerate(lines):
                    if pat.match(ln):
                        lines[i] = new
                        return lines
                lines.append(new)
                return lines

            existing = _upsert("SCHWAB_ACCESS_TOKEN",  access_token,  existing)
            existing = _upsert("SCHWAB_REFRESH_TOKEN", refresh_token, existing)
            path.write_text("\n".join(existing) + "\n")
        except Exception as exc:
            logger.warning("SchwabBroker: could not write tokens to {}: {}", self._env_path, exc)

    # ------------------------------------------------------------------
    # OAuth2 — one-time setup
    # ------------------------------------------------------------------

    def get_authorization_url(self) -> tuple[str, str]:
        """Return (authorization_url, code_verifier). Store verifier for exchange."""
        verifier = secrets.token_urlsafe(96)
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        params = urllib.parse.urlencode({
            "response_type":         "code",
            "client_id":             self._schwab.client_id,
            "redirect_uri":          self._schwab.redirect_uri,
            "code_challenge":        challenge,
            "code_challenge_method": "S256",
            "scope":                 "readonly trade",
        })
        return f"https://api.schwabapi.com/v1/oauth/authorize?{params}", verifier

    async def exchange_code_for_tokens(
        self, auth_code: str, code_verifier: str
    ) -> dict[str, str]:
        """Exchange authorization code for initial access + refresh tokens."""
        session = self._get_session()
        payload = {
            "grant_type":    "authorization_code",
            "code":          auth_code,
            "client_id":     self._schwab.client_id,
            "redirect_uri":  self._schwab.redirect_uri,
            "code_verifier": code_verifier,
        }
        async with session.post(_AUTH_URL, data=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise AuthenticationError(f"Token exchange failed: {body}")
            tokens = await resp.json()

        self._access_token     = tokens["access_token"]
        self._refresh_token    = tokens.get("refresh_token", "")
        expires_in             = int(tokens.get("expires_in", 1800))
        self._token_expires_at = time.time() + expires_in

        self._persist_tokens_to_env(self._access_token, self._refresh_token)
        logger.info(
            "Schwab OAuth complete — tokens valid {}s, persisted to {}",
            expires_in, self._env_path,
        )
        return tokens

    # ------------------------------------------------------------------
    # Internal HTTP helper
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        json_body: dict | None = None,
        params: dict | None = None,
    ) -> Any:
        await self._ensure_token_valid()
        session = self._get_session()
        url     = f"{_TRADER_URL}{path}"

        for attempt in range(self._broker.max_retries):
            headers = {"Authorization": f"Bearer {self._access_token}"}
            try:
                async with session.request(
                    method, url, headers=headers, json=json_body, params=params
                ) as resp:
                    if resp.status == 429:
                        raise RateLimitError("Schwab rate limit")
                    if resp.status == 401:
                        # Token rejected — do one refresh then retry inline
                        logger.warning("Schwab 401 — refreshing token inline")
                        await self._do_refresh()
                        continue
                    if resp.status in (200, 201, 204):
                        return {} if (resp.content_length == 0 or resp.status == 204) \
                               else await resp.json()
                    body = await resp.text()
                    raise BrokerError(f"Schwab {resp.status}: {body[:300]}")
            except (RateLimitError, AuthenticationError):
                raise
            except BrokerError:
                if attempt == self._broker.max_retries - 1:
                    raise
                backoff = self._broker.retry_backoff_seconds * (2 ** attempt)
                logger.warning(
                    "Schwab request failed (attempt {}/{}), retrying in {:.1f}s",
                    attempt + 1, self._broker.max_retries, backoff,
                )
                await asyncio.sleep(backoff)

        raise BrokerError("Max retries exceeded")

    # ------------------------------------------------------------------
    # Account queries
    # ------------------------------------------------------------------

    async def _fetch_account_hash(self) -> str:
        data     = await self._request("GET", "/accounts/accountNumbers")
        accounts = data if isinstance(data, list) else []
        if not accounts:
            raise BrokerError("No accounts on Schwab profile")

        acct_num = self._schwab.account_number
        match = (
            next((a for a in accounts if a.get("accountNumber") == acct_num), None)
            if acct_num else accounts[0]
        )
        if match is None:
            raise BrokerError(f"Account {acct_num!r} not found")

        self._account_hash = match["hashValue"]
        logger.debug("Account hash: {}...", self._account_hash[:8])
        return self._account_hash

    async def get_account_equity(self) -> float:
        data = await self._request("GET", f"/accounts/{self._account_hash}")
        return float(
            data.get("securitiesAccount", {})
                .get("currentBalances", {})
                .get("liquidationValue", 0.0)
        )

    async def get_buying_power(self) -> float:
        data = await self._request("GET", f"/accounts/{self._account_hash}")
        return float(
            data.get("securitiesAccount", {})
                .get("currentBalances", {})
                .get("buyingPower", 0.0)
        )

    async def get_positions(self) -> list[dict[str, Any]]:
        data = await self._request(
            "GET", f"/accounts/{self._account_hash}", params={"fields": "positions"}
        )
        return data.get("securitiesAccount", {}).get("positions", [])

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def submit_order(self, order: Order) -> str:
        payload = self._build_order_payload(order)
        logger.info(
            "Order: {} {} {} @ {} [{}]",
            order.symbol, order.side.value, order.quantity,
            order.limit_price or "MKT", order.order_type.value,
        )
        if self._broker.paper_trading:
            logger.info("[PAPER] Not sent to Schwab")
            return f"PAPER-{order.order_id}"
        await self._request(
            "POST", f"/accounts/{self._account_hash}/orders", json_body=payload
        )
        return str(order.order_id)

    async def cancel_order(self, broker_order_id: str) -> bool:
        if self._broker.paper_trading:
            logger.info("[PAPER] Cancel: {}", broker_order_id)
            return True
        try:
            await self._request(
                "DELETE", f"/accounts/{self._account_hash}/orders/{broker_order_id}"
            )
            return True
        except BrokerError:
            return False

    async def get_order_status(self, broker_order_id: str) -> dict[str, Any]:
        return await self._request(
            "GET", f"/accounts/{self._account_hash}/orders/{broker_order_id}"
        )

    async def get_open_orders(self) -> list[dict[str, Any]]:
        data = await self._request(
            "GET",
            f"/accounts/{self._account_hash}/orders",
            params={"status": "WORKING"},
        )
        return data if isinstance(data, list) else []

    # ------------------------------------------------------------------
    # Payload builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_order_payload(order: Order) -> dict[str, Any]:
        side_map = {
            OrderSide.BUY:          "BUY",
            OrderSide.SELL:         "SELL",
            OrderSide.SELL_SHORT:   "SELL_SHORT",
            OrderSide.BUY_TO_COVER: "BUY_TO_COVER",
        }
        type_map = {
            OrderType.MARKET:     "MARKET",
            OrderType.LIMIT:      "LIMIT",
            OrderType.STOP:       "STOP",
            OrderType.STOP_LIMIT: "STOP_LIMIT",
        }
        tif_map = {
            TimeInForce.DAY: "DAY",
            TimeInForce.GTC: "GOOD_TILL_CANCEL",
            TimeInForce.IOC: "FILL_OR_KILL",
            TimeInForce.FOK: "FILL_OR_KILL",
        }
        payload: dict[str, Any] = {
            "orderType":          type_map[order.order_type],
            "session":            "NORMAL",
            "duration":           tif_map.get(order.time_in_force, "DAY"),
            "orderStrategyType":  "SINGLE",
            "orderLegCollection": [{
                "orderLegType": "EQUITY",
                "instruction":  side_map[order.side],
                "quantity":     order.quantity,
                "instrument":   {"symbol": order.symbol, "assetType": "EQUITY"},
            }],
        }
        if order.limit_price is not None:
            payload["price"] = str(round(order.limit_price, 2))
        if order.stop_price is not None:
            payload["stopPrice"] = str(round(order.stop_price, 2))
        return payload

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise BrokerError("SchwabBroker not connected — call connect() first")
        return self._session

    @property
    def name(self) -> str:
        return "schwab"

    @property
    def is_paper(self) -> bool:
        return self._broker.paper_trading

    @property
    def token_expires_in(self) -> float:
        """Seconds until current access token expires. 0 if already expired."""
        return max(0.0, self._token_expires_at - time.time())
