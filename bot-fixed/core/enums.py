"""
core/enums.py
All shared enumerations for the trading system.
"""
from enum import Enum, auto


# ---------------------------------------------------------------------------
# Market / Data
# ---------------------------------------------------------------------------

class MarketSession(str, Enum):
    PREMARKET = "premarket"
    REGULAR = "regular"
    POSTMARKET = "postmarket"
    CLOSED = "closed"


class BarTimeframe(str, Enum):
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    DAY_1 = "1d"


# ---------------------------------------------------------------------------
# Scanner / Catalyst
# ---------------------------------------------------------------------------

class ArchetypeTag(str, Enum):
    STRONG_GAPPER = "STRONG_GAPPER"
    ORDERLY_TREND = "ORDERLY_TREND"
    VOLATILE_CANDIDATE = "VOLATILE_CANDIDATE"
    SPREAD_RISK = "SPREAD_RISK"
    LIKELY_LEADER = "LIKELY_LEADER"
    CATALYST_BACKED = "CATALYST_BACKED"
    NO_CONFIRMED_CATALYST = "NO_CONFIRMED_CATALYST"
    EXTENDED_PREMARKET = "EXTENDED_PREMARKET"


class CatalystCategory(str, Enum):
    EARNINGS = "EARNINGS"
    EARNINGS_GUIDANCE = "EARNINGS_GUIDANCE"
    ANALYST_UPGRADE = "ANALYST_UPGRADE"
    ANALYST_DOWNGRADE = "ANALYST_DOWNGRADE"
    FDA_OR_BIOTECH_NEWS = "FDA_OR_BIOTECH_NEWS"
    M_AND_A = "M_AND_A"
    PARTNERSHIP_OR_CONTRACT = "PARTNERSHIP_OR_CONTRACT"
    PRODUCT_LAUNCH = "PRODUCT_LAUNCH"
    LEGAL_OR_REGULATORY = "LEGAL_OR_REGULATORY"
    MACRO_RELATED = "MACRO_RELATED"
    PRESS_RELEASE = "PRESS_RELEASE"
    UNKNOWN = "UNKNOWN"


class NewsTimingCategory(str, Enum):
    """
    When the news was published relative to market open.
    Overnight/early premarket news allows institutions time to position cleanly.
    Late/near-open news causes overreaction and higher reversal risk.
    """
    OVERNIGHT = "OVERNIGHT"          # Before 6:00 AM ET — 3.5+ hrs to digest
    EARLY_PREMARKET = "EARLY_PM"     # 6:00–8:00 AM ET — 1.5–3.5 hrs
    LATE_PREMARKET = "LATE_PM"       # 8:00–9:15 AM ET — under 1.5 hrs, higher reversal risk
    NEAR_OPEN = "NEAR_OPEN"          # After 9:15 AM ET — chaotic, avoid
    INTRADAY = "INTRADAY"            # Published during session
    UNKNOWN = "UNKNOWN"              # Cannot determine


# ---------------------------------------------------------------------------
# Orders / Execution
# ---------------------------------------------------------------------------

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    SELL_SHORT = "SELL_SHORT"
    BUY_TO_COVER = "BUY_TO_COVER"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class TimeInForce(str, Enum):
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


# ---------------------------------------------------------------------------
# Positions / Trades
# ---------------------------------------------------------------------------

class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class PositionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"


class ExitReason(str, Enum):
    TARGET_1 = "TARGET_1"
    TARGET_2 = "TARGET_2"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    BREAKEVEN_STOP = "BREAKEVEN_STOP"
    TIME_EXIT = "TIME_EXIT"
    EOD_EXIT = "EOD_EXIT"
    MANUAL = "MANUAL"
    RISK_LIMIT = "RISK_LIMIT"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"


# ---------------------------------------------------------------------------
# Strategy / Signals
# ---------------------------------------------------------------------------

class SignalType(str, Enum):
    VWAP_PULLBACK_LONG = "VWAP_PULLBACK_LONG"
    VWAP_BREAKDOWN_SHORT = "VWAP_BREAKDOWN_SHORT"
    EXIT_TARGET = "EXIT_TARGET"
    EXIT_STOP = "EXIT_STOP"
    EXIT_TIME = "EXIT_TIME"


class SignalStrength(str, Enum):
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

class RiskCheckResult(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    REDUCED = "REDUCED"


class RiskRejectionReason(str, Enum):
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    MAX_POSITIONS = "MAX_POSITIONS"
    MAX_TRADES = "MAX_TRADES"
    CAPITAL_LIMIT = "CAPITAL_LIMIT"
    LIQUIDITY = "LIQUIDITY"
    SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
    SIZE_TOO_SMALL = "SIZE_TOO_SMALL"
    MARKET_CONDITIONS = "MARKET_CONDITIONS"


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

class SystemMode(str, Enum):
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"


class BrokerName(str, Enum):
    SCHWAB = "schwab"
    ALPACA = "alpaca"
    PAPER = "paper"
