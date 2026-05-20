"""
database/repository.py
Typed async repositories for all domain entities.
Each repository wraps DatabaseManager and handles serialisation.
"""
from __future__ import annotations

import json
from datetime import datetime, date
from typing import Any
from uuid import UUID

from core.models import Bar, Order, Position, ScanResult, Signal, Trade
from core.enums import BarTimeframe, OrderStatus, PositionStatus
from database.db_manager import DatabaseManager


def _dt(s: str | None) -> datetime | None:
    if s is None:
        return None
    return datetime.fromisoformat(s)


class BarRepository:
    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    async def upsert(self, bar: Bar) -> None:
        await self._db.execute(
            """
            INSERT INTO bars (symbol, timestamp, timeframe, open, high, low, close, volume, vwap)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, timestamp, timeframe) DO UPDATE SET
                open=excluded.open, high=excluded.high, low=excluded.low,
                close=excluded.close, volume=excluded.volume, vwap=excluded.vwap
            """,
            (
                bar.symbol,
                bar.timestamp.isoformat(),
                bar.timeframe.value,
                bar.open, bar.high, bar.low, bar.close, bar.volume, bar.vwap,
            ),
        )

    async def upsert_many(self, bars: list[Bar]) -> None:
        params = [
            (
                b.symbol, b.timestamp.isoformat(), b.timeframe.value,
                b.open, b.high, b.low, b.close, b.volume, b.vwap,
            )
            for b in bars
        ]
        await self._db.execute_many(
            """
            INSERT INTO bars (symbol, timestamp, timeframe, open, high, low, close, volume, vwap)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, timestamp, timeframe) DO UPDATE SET
                open=excluded.open, high=excluded.high, low=excluded.low,
                close=excluded.close, volume=excluded.volume, vwap=excluded.vwap
            """,
            params,
        )

    async def get_bars(
        self,
        symbol: str,
        timeframe: BarTimeframe,
        start: datetime,
        end: datetime,
    ) -> list[Bar]:
        rows = await self._db.fetch_all(
            """
            SELECT * FROM bars
            WHERE symbol=? AND timeframe=? AND timestamp>=? AND timestamp<=?
            ORDER BY timestamp ASC
            """,
            (symbol, timeframe.value, start.isoformat(), end.isoformat()),
        )
        return [self._row_to_bar(r) for r in rows]

    async def get_latest_bar(self, symbol: str, timeframe: BarTimeframe) -> Bar | None:
        row = await self._db.fetch_one(
            "SELECT * FROM bars WHERE symbol=? AND timeframe=? ORDER BY timestamp DESC LIMIT 1",
            (symbol, timeframe.value),
        )
        return self._row_to_bar(row) if row else None

    @staticmethod
    def _row_to_bar(row: dict[str, Any]) -> Bar:
        return Bar(
            symbol=row["symbol"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            timeframe=BarTimeframe(row["timeframe"]),
            open=row["open"], high=row["high"],
            low=row["low"], close=row["close"],
            volume=row["volume"], vwap=row.get("vwap"),
        )


class ScanResultRepository:
    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    async def save(self, result: ScanResult) -> None:
        await self._db.execute(
            """
            INSERT INTO scan_results
                (symbol, scanned_at, composite_score, passes_filters,
                 gap_pct, relative_volume, pm_dollar_vol, spread_pct,
                 archetypes, catalyst_cat, catalyst_conf, raw_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                result.symbol,
                result.scanned_at.isoformat(),
                result.composite_score,
                int(result.passes_filters),
                result.metrics.gap_pct,
                result.metrics.relative_volume,
                result.metrics.premarket_dollar_volume,
                result.metrics.spread_pct,
                json.dumps([t.value for t in result.archetypes]),
                result.catalyst.category.value if result.catalyst else None,
                result.catalyst.confidence if result.catalyst else None,
                result.model_dump_json(),
            ),
        )

    async def get_latest_scan(self, symbol: str) -> ScanResult | None:
        row = await self._db.fetch_one(
            "SELECT raw_json FROM scan_results WHERE symbol=? ORDER BY scanned_at DESC LIMIT 1",
            (symbol,),
        )
        if row is None:
            return None
        return ScanResult.model_validate_json(row["raw_json"])

    async def get_passing_scans(self, date_str: str) -> list[ScanResult]:
        rows = await self._db.fetch_all(
            """
            SELECT raw_json FROM scan_results
            WHERE passes_filters=1 AND scanned_at LIKE ?
            ORDER BY composite_score DESC
            """,
            (f"{date_str}%",),
        )
        return [ScanResult.model_validate_json(r["raw_json"]) for r in rows]


class OrderRepository:
    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    async def save(self, order: Order) -> None:
        await self._db.execute(
            """
            INSERT OR REPLACE INTO orders
                (order_id, symbol, side, order_type, quantity, limit_price, stop_price,
                 time_in_force, status, submitted_at, filled_at, filled_quantity,
                 avg_fill_price, broker_order_id, notes, raw_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                str(order.order_id), order.symbol, order.side.value,
                order.order_type.value, order.quantity, order.limit_price,
                order.stop_price, order.time_in_force.value, order.status.value,
                order.submitted_at.isoformat() if order.submitted_at else None,
                order.filled_at.isoformat() if order.filled_at else None,
                order.filled_quantity, order.avg_fill_price,
                order.broker_order_id, order.notes, order.model_dump_json(),
            ),
        )

    async def get(self, order_id: UUID) -> Order | None:
        row = await self._db.fetch_one(
            "SELECT raw_json FROM orders WHERE order_id=?", (str(order_id),)
        )
        return Order.model_validate_json(row["raw_json"]) if row else None

    async def get_open_orders(self) -> list[Order]:
        rows = await self._db.fetch_all(
            "SELECT raw_json FROM orders WHERE status IN ('PENDING','SUBMITTED','PARTIAL')"
        )
        return [Order.model_validate_json(r["raw_json"]) for r in rows]


class PositionRepository:
    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    async def save(self, position: Position) -> None:
        await self._db.execute(
            """
            INSERT OR REPLACE INTO positions
                (position_id, symbol, side, status, entry_price, entry_time,
                 quantity, remaining_quantity, stop_price, target_1_price,
                 target_2_price, breakeven_price, trailing_stop_price,
                 exit_price, exit_time, exit_reason, realized_pnl, commission,
                 signal_id, raw_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                str(position.position_id), position.symbol, position.side.value,
                position.status.value, position.entry_price,
                position.entry_time.isoformat(), position.quantity,
                position.remaining_quantity, position.stop_price,
                position.target_1_price, position.target_2_price,
                position.breakeven_price, position.trailing_stop_price,
                position.exit_price,
                position.exit_time.isoformat() if position.exit_time else None,
                position.exit_reason.value if position.exit_reason else None,
                position.realized_pnl, position.commission,
                str(position.signal_id) if position.signal_id else None,
                position.model_dump_json(),
            ),
        )

    async def get_open_positions(self) -> list[Position]:
        rows = await self._db.fetch_all(
            "SELECT raw_json FROM positions WHERE status IN ('OPEN','PARTIALLY_CLOSED')"
        )
        return [Position.model_validate_json(r["raw_json"]) for r in rows]

    async def get(self, position_id: UUID) -> Position | None:
        row = await self._db.fetch_one(
            "SELECT raw_json FROM positions WHERE position_id=?", (str(position_id),)
        )
        return Position.model_validate_json(row["raw_json"]) if row else None

    async def get_by_symbol(self, symbol: str) -> list[Position]:
        rows = await self._db.fetch_all(
            "SELECT raw_json FROM positions WHERE symbol=? ORDER BY entry_time DESC",
            (symbol,),
        )
        return [Position.model_validate_json(r["raw_json"]) for r in rows]


class TradeRepository:
    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    async def save(self, trade: Trade) -> None:
        await self._db.execute(
            """
            INSERT OR REPLACE INTO trades
                (trade_id, position_id, symbol, side, entry_price, exit_price,
                 quantity, entry_time, exit_time, exit_reason, gross_pnl,
                 commission, net_pnl, r_multiple, hold_duration_secs)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                str(trade.trade_id), str(trade.position_id), trade.symbol,
                trade.side.value, trade.entry_price, trade.exit_price,
                trade.quantity, trade.entry_time.isoformat(),
                trade.exit_time.isoformat(), trade.exit_reason.value,
                trade.gross_pnl, trade.commission, trade.net_pnl,
                trade.r_multiple, trade.hold_duration_seconds,
            ),
        )

    async def get_trades_for_date(self, date_str: str) -> list[Trade]:
        rows = await self._db.fetch_all(
            "SELECT * FROM trades WHERE exit_time LIKE ? ORDER BY exit_time ASC",
            (f"{date_str}%",),
        )
        return [self._row_to_trade(r) for r in rows]

    async def get_all_trades(self) -> list[Trade]:
        rows = await self._db.fetch_all("SELECT * FROM trades ORDER BY exit_time ASC")
        return [self._row_to_trade(r) for r in rows]

    @staticmethod
    def _row_to_trade(row: dict[str, Any]) -> Trade:
        from core.enums import ExitReason, PositionSide
        from uuid import UUID
        return Trade(
            trade_id=UUID(row["trade_id"]),
            position_id=UUID(row["position_id"]),
            symbol=row["symbol"],
            side=PositionSide(row["side"]),
            entry_price=row["entry_price"],
            exit_price=row["exit_price"],
            quantity=row["quantity"],
            entry_time=datetime.fromisoformat(row["entry_time"]),
            exit_time=datetime.fromisoformat(row["exit_time"]),
            exit_reason=ExitReason(row["exit_reason"]),
            gross_pnl=row["gross_pnl"],
            commission=row["commission"],
            net_pnl=row["net_pnl"],
            r_multiple=row["r_multiple"],
            hold_duration_seconds=row["hold_duration_secs"],
        )


class SignalRepository:
    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    async def save(self, signal: Signal) -> None:
        await self._db.execute(
            """
            INSERT OR REPLACE INTO signals
                (signal_id, symbol, signal_type, strength, generated_at,
                 entry_price, stop_price, target_1_price, target_2_price,
                 suggested_qty, vwap_at_signal, notes, acted_on)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                str(signal.signal_id), signal.symbol, signal.signal_type.value,
                signal.strength.value, signal.generated_at.isoformat(),
                signal.entry_price, signal.stop_price,
                signal.target_1_price, signal.target_2_price,
                signal.suggested_quantity, signal.vwap_at_signal,
                signal.notes, 0,
            ),
        )

    async def mark_acted_on(self, signal_id: UUID) -> None:
        await self._db.execute(
            "UPDATE signals SET acted_on=1 WHERE signal_id=?", (str(signal_id),)
        )
