"""
integrations/sheets_writer.py
Google Sheets API writer for the trading bot data warehouse.

Design decisions (answered by the operator):
  - Real-time: Trade_Log, Decision_Log, Kill_Switch_Events written immediately
    after each event. These are low-volume (< 20 rows/session) so rate limits
    are not a concern.
  - Batch EOD: Scan_Log written once at end of day. Scanning 1500 symbols
    generates ~500 rows — writing them in real-time would exhaust the Sheets
    API write quota (300 writes/min) within the first scan cycle.
  - All other sheets (Daily_Performance, Execution_Log, Watchlist_Log) are
    written at EOD as part of the scheduled EOD runner.

Authentication:
  Uses a Google service account JSON key file.
  Path: GOOGLE_SERVICE_ACCOUNT_JSON env var or ./credentials/service_account.json
  The service account must have Editor access to the target spreadsheet.
  Spreadsheet ID: GOOGLE_SHEETS_ID env var.

Sheet name → tab name mapping must exactly match what the Apps Script created.

Rate limit handling:
  - Exponential backoff with jitter on 429/503.
  - Batch appends use values().append() with a single HTTP call per sheet.
  - A semaphore limits concurrent writes to 3 to avoid self-rate-limiting.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import aiohttp
from loguru import logger

# ---------------------------------------------------------------------------
# Google Sheets API constants
# ---------------------------------------------------------------------------

_SHEETS_BASE = "https://sheets.googleapis.com/v4/spreadsheets"
_AUTH_URL    = "https://oauth2.googleapis.com/token"
_SCOPE       = "https://www.googleapis.com/auth/spreadsheets"

# Default credential paths
_DEFAULT_SA_PATH = Path("credentials/service_account.json")

# Sheets tab names (must match what the Apps Script created)
SHEET_TRADE_LOG       = "Trade_Log"
SHEET_DECISION_LOG    = "Decision_Log"
SHEET_KILL_SWITCH     = "Kill_Switch_Events"       # Maps to kill_switch_events sheet
SHEET_SCAN_LOG        = "Scan_Log"
SHEET_WATCHLIST_LOG   = "Watchlist_Log"
SHEET_EXECUTION_LOG   = "Execution_Log"
SHEET_DAILY_PERF      = "Daily_Performance"
SHEET_STRATEGY_PERF   = "Strategy_Performance"
SHEET_BOT_HEALTH      = "Bot_Health_Log"


# ---------------------------------------------------------------------------
# JWT / OAuth2 helpers for service account auth
# ---------------------------------------------------------------------------

def _load_service_account() -> dict[str, Any]:
    """Load service account JSON from env var path or default location."""
    sa_path_env = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if sa_path_env:
        path = Path(sa_path_env)
    elif _DEFAULT_SA_PATH.exists():
        path = _DEFAULT_SA_PATH
    else:
        raise RuntimeError(
            "Google service account JSON not found. "
            "Set GOOGLE_SERVICE_ACCOUNT_JSON env var or place file at "
            f"{_DEFAULT_SA_PATH}"
        )
    with path.open() as f:
        return json.load(f)


def _build_jwt(sa: dict[str, Any]) -> str:
    """
    Build a signed JWT for service account OAuth2.
    Uses only stdlib — no google-auth dependency required.
    """
    import base64
    import hashlib
    import hmac
    import json as _json

    now = int(time.time())
    header  = {"alg": "RS256", "typ": "JWT"}
    payload = {
        "iss":   sa["client_email"],
        "scope": _SCOPE,
        "aud":   _AUTH_URL,
        "iat":   now,
        "exp":   now + 3600,
    }

    def _b64(data: dict) -> str:
        return (
            base64.urlsafe_b64encode(_json.dumps(data).encode())
            .rstrip(b"=")
            .decode()
        )

    unsigned = f"{_b64(header)}.{_b64(payload)}"

    # Sign with RSA-SHA256 using the private key from the service account
    # We use the cryptography library if available, otherwise fallback to
    # subprocess openssl (avoids hard dependency on cryptography package)
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        private_key = serialization.load_pem_private_key(
            sa["private_key"].encode(), password=None
        )
        signature = private_key.sign(unsigned.encode(), padding.PKCS1v15(), hashes.SHA256())
        sig_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
    except ImportError:
        # Fallback: use subprocess openssl (always available on Linux)
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pem", delete=False, mode="w") as f:
            f.write(sa["private_key"])
            key_path = f.name
        try:
            result = subprocess.run(
                ["openssl", "dgst", "-sha256", "-sign", key_path],
                input=unsigned.encode(),
                capture_output=True,
                check=True,
            )
            sig_b64 = base64.urlsafe_b64encode(result.stdout).rstrip(b"=").decode()
        finally:
            Path(key_path).unlink(missing_ok=True)

    return f"{unsigned}.{sig_b64}"


async def _get_access_token(session: aiohttp.ClientSession, sa: dict[str, Any]) -> str:
    """Exchange a service account JWT for a Google OAuth2 access token."""
    jwt = _build_jwt(sa)
    async with session.post(
        _AUTH_URL,
        data={
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion":  jwt,
        },
        timeout=aiohttp.ClientTimeout(total=10),
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"Google auth failed ({resp.status}): {body[:200]}")
        data = await resp.json()
        return data["access_token"]


# ---------------------------------------------------------------------------
# Core writer
# ---------------------------------------------------------------------------

class SheetsWriter:
    """
    Async Google Sheets writer.

    Usage:
        async with SheetsWriter() as writer:
            await writer.append_trade(trade)
            await writer.append_scan_batch(results)

    All public methods are safe to call from multiple coroutines simultaneously.
    The internal semaphore ensures we never blast more than 3 concurrent
    HTTP writes to the Sheets API.
    """

    def __init__(
        self,
        spreadsheet_id: str | None = None,
        sa_path: str | Path | None = None,
    ) -> None:
        self._spreadsheet_id = spreadsheet_id or os.getenv("GOOGLE_SHEETS_ID", "")
        if not self._spreadsheet_id:
            raise RuntimeError(
                "Spreadsheet ID not set. "
                "Pass spreadsheet_id= or set GOOGLE_SHEETS_ID env var."
            )

        self._sa_path = Path(sa_path) if sa_path else None
        self._sa: dict[str, Any] = {}
        self._access_token: str  = ""
        self._token_expires_at: float = 0.0

        self._session: aiohttp.ClientSession | None = None
        self._write_sem = asyncio.Semaphore(3)  # Max 3 concurrent Sheets writes

    async def __aenter__(self) -> "SheetsWriter":
        self._session = aiohttp.ClientSession()
        self._sa = _load_service_account()
        await self._refresh_token()
        logger.info("SheetsWriter: authenticated to spreadsheet {}", self._spreadsheet_id[:8] + "...")
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Public write methods
    # ------------------------------------------------------------------

    async def append_trade(self, trade: "Trade") -> None:  # type: ignore[name-defined]
        """
        Write one completed trade to Trade_Log immediately.
        Called by ExecutionEngine after each trade close.
        """
        row = [
            str(trade.trade_id),
            str(trade.position_id),
            trade.entry_time.strftime("%Y-%m-%d"),
            trade.symbol,
            "",                                    # strategy (not on Trade model directly)
            trade.side.value,
            trade.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
            trade.entry_price,
            trade.exit_time.strftime("%Y-%m-%d %H:%M:%S"),
            trade.exit_price,
            trade.quantity,
            round(trade.entry_price * trade.quantity, 2),
            "",                                    # stop_price (not on closed Trade)
            "",                                    # target_1_price
            "",                                    # target_2_price
            "",                                    # risk_per_share
            "",                                    # planned_risk_dollars
            round(trade.gross_pnl, 4),
            round(trade.commission, 4),
            round(trade.net_pnl, 4),
            round(trade.r_multiple, 4),
            round(trade.hold_duration_seconds, 1),
            trade.exit_reason.value,
            "WIN" if trade.net_pnl > 0 else ("LOSS" if trade.net_pnl < 0 else "BREAKEVEN"),
            "",                                    # notes
        ]
        await self._append_rows(SHEET_TRADE_LOG, [row])
        logger.debug("Sheets: trade written — {} {:.2f}", trade.symbol, trade.net_pnl)

    async def append_decision(self, decision: dict[str, Any]) -> None:
        """
        Write a bot decision to Decision_Log immediately.
        Pass a dict with keys matching the Decision_Log schema.
        """
        row = [
            decision.get("decision_id", ""),
            _fmt_dt(decision.get("timestamp")),
            decision.get("symbol", ""),
            decision.get("decision_type", ""),
            decision.get("decision_result", ""),
            decision.get("reason", ""),
            decision.get("input_score", ""),
            decision.get("threshold", ""),
            decision.get("actual_value", ""),
            decision.get("required_value", ""),
            decision.get("bot_version", ""),
            decision.get("config_version", ""),
        ]
        await self._append_rows(SHEET_DECISION_LOG, [row])

    async def append_kill_switch_event(self, event: "KillSwitchEvent") -> None:  # type: ignore[name-defined]
        """Write a kill switch event to Kill_Switch_Events immediately."""
        row = [
            event.trigger.value,
            _fmt_dt(event.triggered_at),
            event.reason,
            event.value if event.value is not None else "",
            event.threshold if event.threshold is not None else "",
            _fmt_dt(event.resolved_at) if event.resolved_at else "",
            event.resolved_by or "",
        ]
        await self._append_rows(SHEET_KILL_SWITCH, [row])
        logger.debug("Sheets: kill switch event written — {}", event.trigger.value)

    async def batch_append_scans(self, results: "list[ScanResult]") -> None:  # type: ignore[name-defined]
        """
        Write all scan results for the day to Scan_Log in one batch.
        Called by the EOD runner, not during live scanning.
        Each ScanResult maps directly to the Scan_Log schema.
        """
        if not results:
            return
        rows = [_scan_result_to_row(r) for r in results]
        await self._append_rows(SHEET_SCAN_LOG, rows)
        logger.info("Sheets: {} scan results written to Scan_Log", len(rows))

    async def append_daily_performance(
        self,
        date_str: str,
        trades: "list[Trade]",                    # type: ignore[name-defined]
        market_regime: str = "",
        notes: str = "",
    ) -> None:
        """Write one Daily_Performance row for the given date."""
        if not trades:
            return
        winners    = [t for t in trades if t.net_pnl > 0]
        losers     = [t for t in trades if t.net_pnl <= 0]
        gross_pnl  = sum(t.gross_pnl for t in trades)
        net_pnl    = sum(t.net_pnl for t in trades)
        total_r    = sum(t.r_multiple for t in trades)
        avg_r      = total_r / len(trades) if trades else 0.0
        win_rate   = len(winners) / len(trades) if trades else 0.0
        max_dd     = min((t.net_pnl for t in trades), default=0.0)
        largest_w  = max((t.net_pnl for t in winners), default=0.0)
        largest_l  = min((t.net_pnl for t in losers), default=0.0)

        row = [
            date_str,
            len(trades),
            len(winners),
            len(losers),
            round(win_rate * 100, 2),
            round(gross_pnl, 2),
            round(net_pnl, 2),
            round(total_r, 4),
            round(avg_r, 4),
            round(max_dd, 2),
            round(largest_w, 2),
            round(largest_l, 2),
            "",          # avg_slippage_pct — populated from execution log
            "FALSE",     # daily_loss_limit_hit — set by risk engine
            market_regime,
            notes,
        ]
        await self._append_rows(SHEET_DAILY_PERF, [row])
        logger.info("Sheets: daily performance row written for {}", date_str)

    async def append_bot_health(self, health: dict[str, Any]) -> None:
        """Write a health snapshot to Bot_Health_Log."""
        row = [
            _fmt_dt(health.get("timestamp")),
            health.get("bot_status", ""),
            health.get("scanner_status", ""),
            health.get("broker_status", ""),
            health.get("data_feed_status", ""),
            health.get("api_latency_ms", ""),
            health.get("order_latency_ms", ""),
            health.get("errors_count", 0),
            _fmt_dt(health.get("last_successful_scan")),
            _fmt_dt(health.get("last_successful_order")),
            "TRUE" if health.get("kill_switch_triggered") else "FALSE",
            health.get("error_message", ""),
        ]
        await self._append_rows(SHEET_BOT_HEALTH, [row])

    # ------------------------------------------------------------------
    # Internal HTTP
    # ------------------------------------------------------------------

    async def _append_rows(
        self,
        sheet_name: str,
        rows: list[list[Any]],
    ) -> None:
        """
        Append rows to a named sheet tab using the values().append() API.
        Retries on transient errors with exponential backoff + jitter.
        """
        if not rows:
            return

        url = (
            f"{_SHEETS_BASE}/{self._spreadsheet_id}/values/"
            f"{sheet_name}!A1:append"
        )
        body = {"values": [[str(v) if v is not None else "" for v in row] for row in rows]}
        params = {
            "valueInputOption":      "RAW",
            "insertDataOption":      "INSERT_ROWS",
            "includeValuesInResponse": "false",
        }

        async with self._write_sem:
            await self._ensure_token()
            headers = {
                "Authorization": f"Bearer {self._access_token}",
                "Content-Type":  "application/json",
            }
            for attempt in range(5):
                try:
                    assert self._session is not None
                    async with self._session.post(
                        url,
                        headers=headers,
                        json=body,
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=15),
                    ) as resp:
                        if resp.status in (200, 201):
                            return
                        if resp.status in (429, 503):
                            # Rate limited — back off with jitter
                            import random
                            backoff = (2 ** attempt) + random.uniform(0, 1)
                            logger.warning(
                                "Sheets rate limited ({}), retrying in {:.1f}s",
                                resp.status, backoff,
                            )
                            await asyncio.sleep(backoff)
                            continue
                        if resp.status == 401:
                            await self._refresh_token()
                            headers["Authorization"] = f"Bearer {self._access_token}"
                            continue
                        body_text = await resp.text()
                        logger.error(
                            "Sheets write failed ({}) for {}: {}",
                            resp.status, sheet_name, body_text[:200],
                        )
                        return  # Non-retriable — log and move on
                except Exception as exc:
                    if attempt == 4:
                        logger.error("Sheets write error after 5 attempts: {}", exc)
                        return
                    await asyncio.sleep(2 ** attempt)

    async def _ensure_token(self) -> None:
        """Refresh the access token if it's within 60 seconds of expiry."""
        if time.time() >= self._token_expires_at - 60:
            await self._refresh_token()

    async def _refresh_token(self) -> None:
        assert self._session is not None
        self._access_token = await _get_access_token(self._session, self._sa)
        self._token_expires_at = time.time() + 3500  # Tokens are valid for 3600s


# ---------------------------------------------------------------------------
# Row serialisation helpers
# ---------------------------------------------------------------------------

def _fmt_dt(dt: datetime | str | None) -> str:
    if dt is None:
        return ""
    if isinstance(dt, str):
        return dt
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _scan_result_to_row(r: Any) -> list[Any]:
    """
    Convert a ScanResult to a Scan_Log row.
    Column order must exactly match the Scan_Log schema in the Apps Script.
    """
    m = r.metrics
    return [
        "",                                              # scan_id (DB assigns)
        _fmt_dt(r.scanned_at),                          # timestamp
        r.scanned_at.strftime("%Y-%m-%d"),              # date
        r.symbol,                                        # symbol
        round(m.premarket_last, 4),                     # last_price
        round(m.prev_close, 4),                         # previous_close
        round(m.gap_pct, 4),                            # gap_pct
        m.premarket_volume,                              # premarket_volume
        round(m.relative_volume, 4),                    # relative_premarket_volume
        round(m.premarket_dollar_volume, 2),             # premarket_dollar_volume
        round(m.spread_pct, 4),                         # spread_pct
        round(m.range_pct, 4),                          # range_pct
        round(m.range_position, 4),                     # range_position
        round(m.trend_quality, 4),                      # trend_quality
        round(r.catalyst.strength * r.catalyst.confidence if r.catalyst else 0.0, 4),  # catalyst_score
        m.float_shares or "",                            # float_shares
        round(m.pm_float_rotation_pct, 4) if m.pm_float_rotation_pct is not None else "",  # float_rotation_pct
        round(r.composite_score, 4),                    # score
        "TRUE" if r.passes_filters else "FALSE",        # passed
        r.filter_failure_reason or "",                  # filter_reasons
        ",".join(t.value for t in r.archetypes),        # tags
        "",                                              # market_regime (from VIXMonitor)
    ]
