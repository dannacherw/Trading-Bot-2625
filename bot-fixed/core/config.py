"""
core/config.py
Centralised configuration loader. Reads YAML config files and .env,
exposes strongly-typed settings objects used throughout the system.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from core.exceptions import ConfigurationError

load_dotenv()

_CONFIG_DIR = Path(__file__).parent.parent / "config"


def _load_yaml(filename: str) -> dict[str, Any]:
    path = _CONFIG_DIR / filename
    if not path.exists():
        raise ConfigurationError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Scanner Settings
# ---------------------------------------------------------------------------

@dataclass
class UniverseSettings:
    min_price: float = 5.0
    max_universe_size: int = 1500
    min_avg_daily_dollar_volume: float = 20_000_000
    exclude_etfs: bool = True
    exclude_preferred: bool = True
    exclude_warrants: bool = True
    exclude_adrs: bool = True
    exclude_spacs: bool = True



@dataclass
class ScanWindowSettings:
    """Controls the 8:00–9:25 AM ET premarket scan window."""
    window_open_et_hour: int = 8
    window_open_et_minute: int = 0
    window_close_et_hour: int = 9
    window_close_et_minute: int = 25
    warning_zone_et_hour: int = 9
    warning_zone_et_minute: int = 15
    enforce_window: bool = True
    allow_weekends: bool = False  # Set True only for testing


@dataclass
class KillSwitchSettings:
    """Controls soft-halt kill switch triggers."""
    max_consecutive_losses: int = 3      # Halt after 3 losses in a row
    consecutive_loss_halt_enabled: bool = True
    max_stale_scan_minutes: float = 10.0
    data_staleness_halt_enabled: bool = True
    max_avg_spread_pct: float = 0.35     # Aligned with scanner max_spread (0.30) + buffer
    spread_expansion_halt_enabled: bool = True
    spread_check_min_symbols: int = 3
    broker_disconnect_halt_enabled: bool = True
    broker_heartbeat_timeout_seconds: float = 30.0
    volatility_halt_enabled: bool = True
    max_spy_intraday_down_pct: float = -1.5
    max_vix_level: float = 40.0          # Was 35 — aligned with strategy_config halt_on_vix_above
    require_human_restart: bool = True


@dataclass
class PremarketFilterSettings:
    min_gap_pct: float = 3.0
    min_gap_pct_no_catalyst: float = 5.0              # Was 6.0 — loosened
    high_quality_catalyst_min_confidence: float = 0.75  # Was 0.80 — slightly more inclusive
    min_premarket_volume: int = 50_000                 # Was 100k — loosened
    min_relative_volume: float = 1.5                  # Was 2.0 — loosened
    min_premarket_dollar_volume: float = 500_000       # Was 1M — loosened
    max_spread_pct: float = 0.30                      # Was 0.25 — aligned with risk engine
    max_premarket_range_pct: float = 20.0             # Was 15.0 — allow biotech volatility
    min_range_position: float = 0.40                  # Was 0.50 — slight loosening
    use_float_rotation_filter: bool = True
    min_pm_float_rotation_pct: float = 1.5            # Was 2.0 — loosened


@dataclass
class ScoringWeights:
    trend_quality: float = 0.22         # Orderly premarket move → best predictor of follow-through
    relative_volume: float = 0.20
    float_rotation_pct: float = 0.14   # Low float + high rotation = institutional interest
    gap_pct: float = 0.14
    catalyst_score: float = 0.12
    premarket_dollar_volume: float = 0.12
    spread_inverse: float = 0.06


@dataclass
class WatchlistSettings:
    max_focus_list_size: int = 5
    max_watchlist_size: int = 20
    min_score_threshold: float = 0.35


@dataclass
class ArchetypeThresholds:
    strong_gapper_gap_pct: float = 8.0
    orderly_trend_min_trend_quality: float = 0.65
    volatile_candidate_range_pct: float = 10.0
    spread_risk_spread_pct: float = 0.15
    likely_leader_min_score: float = 0.75
    catalyst_backed_min_confidence: float = 0.6
    extended_premarket_range_pct: float = 12.0


@dataclass
class ScannerConfig:
    universe: UniverseSettings = field(default_factory=UniverseSettings)
    filters: PremarketFilterSettings = field(default_factory=PremarketFilterSettings)
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    watchlist: WatchlistSettings = field(default_factory=WatchlistSettings)
    archetypes: ArchetypeThresholds = field(default_factory=ArchetypeThresholds)
    scan_interval_seconds: int = 60
    scan_window: ScanWindowSettings = field(default_factory=ScanWindowSettings)
    kill_switch: KillSwitchSettings = field(default_factory=KillSwitchSettings)


# ---------------------------------------------------------------------------
# Risk Settings
# ---------------------------------------------------------------------------

@dataclass
class AccountSettings:
    starting_equity: float = 10_000.0
    currency: str = "USD"


@dataclass
class DailyLimitSettings:
    max_daily_loss_pct: float = 1.5     # Was 2.0 — tighter at $10k ($150 max daily loss)
    max_daily_loss_hard_stop: bool = True
    max_trades_per_day: int = 5         # Was 6 — quality over quantity


@dataclass
class PositionLimitSettings:
    max_open_positions: int = 3
    min_risk_per_trade_pct: float = 0.25
    max_risk_per_trade_pct: float = 0.75        # Was 0.50 — allows larger on A+ setups
    default_risk_per_trade_pct: float = 0.50   # Was 0.35 — $50 risk at $10k is workable
    max_capital_per_position_pct: float = 25.0  # Was 30 — tighter concentration
    min_trade_dollar_value: float = 300.0       # Was 500 — allows $5-10 stocks


@dataclass
class StopLossSettings:
    atr_multiplier: float = 1.2          # Was 1.5 — tighter stops improve R/R
    max_stop_pct: float = 2.5            # Was 3.0 — hard cap tighter for $10k
    use_vwap_as_stop: bool = True
    vwap_stop_buffer_pct: float = 0.08   # Was 0.10 — slightly tighter


@dataclass
class VolatilitySettings:
    atr_period: int = 14
    atr_adjustment_threshold: float = 1.5
    atr_size_reduction_factor: float = 0.7


@dataclass
class LiquiditySettings:
    max_pct_of_adv: float = 5.0
    min_dollar_volume_for_full_size: float = 5_000_000


@dataclass
class RiskConfig:
    account: AccountSettings = field(default_factory=AccountSettings)
    daily_limits: DailyLimitSettings = field(default_factory=DailyLimitSettings)
    position_limits: PositionLimitSettings = field(default_factory=PositionLimitSettings)
    stop_loss: StopLossSettings = field(default_factory=StopLossSettings)
    volatility: VolatilitySettings = field(default_factory=VolatilitySettings)
    liquidity: LiquiditySettings = field(default_factory=LiquiditySettings)
    max_entry_spread_pct: float = 0.30   # Entry gate — aligned with scanner max_spread_pct


# ---------------------------------------------------------------------------
# Strategy Settings
# ---------------------------------------------------------------------------

@dataclass
class EntrySettings:
    momentum_window_minutes: int = 15
    min_opening_move_pct: float = 1.0              # Was 1.5 — allow tight opens with strong catalyst
    max_entry_distance_from_vwap_pct: float = 0.60  # Was 0.50 — slightly more allowance
    require_vwap_reclaim: bool = True
    # VWAP reclaim quality conditions
    vwap_reclaim_min_close_above_pct: float = 0.08    # Was 0.10 — slight loosening
    vwap_reclaim_close_position_min: float = 0.55     # Was 0.60 — still upper half of bar
    vwap_reclaim_touch_vol_decay: bool = True          # Keep — exhaustion dip is key signal
    min_entry_volume_ratio: float = 1.2               # Was 1.3 — slight loosening
    earliest_entry_minutes_after_open: int = 5
    latest_entry_minutes_before_close: int = 90       # Was 60 — allows entries until 2 PM ET
    blackout_lunch_start: str = "11:45"
    blackout_lunch_end: str = "12:45"                 # Was 13:00 — shorter lunch blackout
    min_entry_bar_size_pct: float = 0.15              # Was 0.20 — catches smaller-bodied bars
    # Relative strength gate
    require_relative_strength: bool = True
    min_relative_strength_vs_benchmark: float = 0.8  # Was 1.5 — critical fix, see comments
    # First candle gate
    require_first_candle_analysis: bool = True
    first_candle_min_close_position: float = 0.65    # Was 0.70 — slight loosening
    first_candle_min_volume_vs_pm: float = 1.5        # Was 2.0 — realistic for small universe
    # Opening range preservation filter (Gap 3 fix)
    # Reject if price has retraced more than this fraction of the opening move.
    # 0.60 = allow up to 60% retrace; more than that = thesis is broken.
    max_opening_range_retrace_pct: float = 0.60


@dataclass
class MarketRegimeSettings:
    """Controls the intraday market regime gate."""
    enabled: bool = True
    default_benchmark: str = "SPY"
    max_benchmark_down_pct: float = -0.40   # Halt new longs if benchmark down >0.4% from open
    min_benchmark_vwap_slope: float = -0.005  # Halt if benchmark VWAP slope is clearly negative
    vwap_slope_lookback_bars: int = 5
    # Sector → benchmark ETF mapping (override per-symbol based on ticker sector)
    sector_benchmark_map: dict = field(default_factory=lambda: {
        "Healthcare": "XBI",
        "Biotechnology": "XBI",
        "Biotech": "XBI",
        "Technology": "QQQ",
        "Communication Services": "QQQ",
        "Financial Services": "XLF",
        "Financials": "XLF",
        "Energy": "XLE",
        "Industrials": "XLI",
        "Consumer Cyclical": "XLY",
        "Consumer Defensive": "XLP",
        "Basic Materials": "XLB",
        "Utilities": "XLU",
        "Real Estate": "XLRE",
    })


@dataclass
class ExitSettings:
    target_1_risk_reward: float = 1.5
    target_2_risk_reward: float = 2.5
    target_1_size_pct: float = 50.0
    use_trailing_stop: bool = True
    trail_activation_r: float = 1.0      # Was 1.5 — protect capital sooner at $10k
    trail_offset_atr_multiple: float = 0.5
    max_hold_minutes: int = 90           # Was 120 — tighter time discipline
    eod_exit_time: str = "15:45"
    move_to_breakeven_r: float = 0.8    # Was 1.0 — protect capital sooner


@dataclass
class StrategyConfig:
    entry: EntrySettings = field(default_factory=EntrySettings)
    exit: ExitSettings = field(default_factory=ExitSettings)
    regime: MarketRegimeSettings = field(default_factory=MarketRegimeSettings)
    vwap_slope_lookback_bars: int = 3
    min_vwap_slope: float = -0.001      # Allow very slight negative slope (updated default)
    loop_interval_seconds: int = 10     # Strategy evaluation loop interval (was hardcoded 30s)


# ---------------------------------------------------------------------------
# Execution Settings
# ---------------------------------------------------------------------------

@dataclass
class BrokerSettings:
    name: str = "schwab"
    base_url: str = "https://api.schwabapi.com"
    auth_url: str = "https://api.schwabapi.com/v1/oauth/token"
    paper_trading: bool = True
    timeout_seconds: int = 10
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0


@dataclass
class SchwabSettings:
    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str = "https://127.0.0.1"
    token_refresh_buffer_seconds: int = 300
    account_number: str = ""


@dataclass
class OrderSettings:
    default_order_type: str = "LIMIT"
    limit_slippage_buffer_pct: float = 0.05
    max_order_value: float = 3000.0
    min_order_value: float = 500.0
    time_in_force: str = "DAY"


@dataclass
class SlippageSettings:
    model: str = "percentage"
    base_slippage_pct: float = 0.05
    high_vol_slippage_multiplier: float = 2.0
    spread_slippage_factor: float = 0.5


@dataclass
class FillSimulationSettings:
    fill_probability: float = 0.95
    partial_fill_probability: float = 0.10
    fill_delay_seconds: float = 0.5


@dataclass
class ExecutionConfig:
    broker: BrokerSettings = field(default_factory=BrokerSettings)
    schwab: SchwabSettings = field(default_factory=SchwabSettings)
    orders: OrderSettings = field(default_factory=OrderSettings)
    slippage: SlippageSettings = field(default_factory=SlippageSettings)
    fill_simulation: FillSimulationSettings = field(default_factory=FillSimulationSettings)


# ---------------------------------------------------------------------------
# Master settings loader
# ---------------------------------------------------------------------------

def _nested_update(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge source into target (source wins)."""
    for k, v in source.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            _nested_update(target[k], v)
        else:
            target[k] = v
    return target


def _dataclass_from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Shallow-populate a dataclass from a dict, ignoring unknown keys."""
    import dataclasses
    fields = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in fields})


def load_scanner_config() -> ScannerConfig:
    raw = _load_yaml("scanner_config.yaml")
    cfg = ScannerConfig()
    if "universe" in raw:
        cfg.universe = _dataclass_from_dict(UniverseSettings, raw["universe"])
    if "premarket_filters" in raw:
        cfg.filters = _dataclass_from_dict(PremarketFilterSettings, raw["premarket_filters"])
    if "scoring_weights" in raw:
        cfg.weights = _dataclass_from_dict(ScoringWeights, raw["scoring_weights"])
    if "watchlist" in raw:
        cfg.watchlist = _dataclass_from_dict(WatchlistSettings, raw["watchlist"])
    if "archetype_thresholds" in raw:
        cfg.archetypes = _dataclass_from_dict(ArchetypeThresholds, raw["archetype_thresholds"])
    if "scan_window" in raw:
        sw_raw = raw["scan_window"]
        cfg.scan_interval_seconds = sw_raw.get("interval_seconds", 60)
        cfg.scan_window = _dataclass_from_dict(ScanWindowSettings, sw_raw)
    if "kill_switch" in raw:
        cfg.kill_switch = _dataclass_from_dict(KillSwitchSettings, raw["kill_switch"])
    return cfg


def load_risk_config() -> RiskConfig:
    raw = _load_yaml("risk_config.yaml")
    cfg = RiskConfig()
    if "account" in raw:
        cfg.account = _dataclass_from_dict(AccountSettings, raw["account"])
    if "daily_limits" in raw:
        cfg.daily_limits = _dataclass_from_dict(DailyLimitSettings, raw["daily_limits"])
    if "position_limits" in raw:
        cfg.position_limits = _dataclass_from_dict(PositionLimitSettings, raw["position_limits"])
    if "stop_loss" in raw:
        cfg.stop_loss = _dataclass_from_dict(StopLossSettings, raw["stop_loss"])
    if "volatility" in raw:
        cfg.volatility = _dataclass_from_dict(VolatilitySettings, raw["volatility"])
    if "liquidity" in raw:
        cfg.liquidity = _dataclass_from_dict(LiquiditySettings, raw["liquidity"])
    if "max_entry_spread_pct" in raw:
        cfg.max_entry_spread_pct = float(raw["max_entry_spread_pct"])
    return cfg


def load_strategy_config() -> StrategyConfig:
    raw = _load_yaml("strategy_config.yaml")
    cfg = StrategyConfig()
    if "entry" in raw:
        cfg.entry = _dataclass_from_dict(EntrySettings, raw["entry"])
    if "exit" in raw:
        cfg.exit = _dataclass_from_dict(ExitSettings, raw["exit"])
    if "market_regime" in raw:
        regime_raw = raw["market_regime"]
        # sector_benchmark_map needs special handling (nested dict)
        base = _dataclass_from_dict(MarketRegimeSettings, regime_raw)
        if "sector_benchmark_map" in regime_raw:
            base.sector_benchmark_map = regime_raw["sector_benchmark_map"]
        cfg.regime = base
    cfg.vwap_slope_lookback_bars = raw.get("signals", {}).get("vwap_slope_lookback_bars", 3)
    cfg.min_vwap_slope = raw.get("signals", {}).get("min_vwap_slope", -0.001)
    cfg.loop_interval_seconds = raw.get("strategy_loop_interval_seconds", 10)
    return cfg


def load_execution_config() -> ExecutionConfig:
    raw = _load_yaml("execution_config.yaml")
    cfg = ExecutionConfig()

    # Overlay env vars for secrets
    if "schwab" in raw:
        raw["schwab"]["client_id"] = os.getenv("SCHWAB_CLIENT_ID", raw["schwab"].get("client_id", ""))
        raw["schwab"]["client_secret"] = os.getenv(
            "SCHWAB_CLIENT_SECRET", raw["schwab"].get("client_secret", "")
        )
        raw["schwab"]["account_number"] = os.getenv(
            "SCHWAB_ACCOUNT_NUMBER", raw["schwab"].get("account_number", "")
        )
        cfg.schwab = _dataclass_from_dict(SchwabSettings, raw["schwab"])

    if "broker" in raw:
        cfg.broker = _dataclass_from_dict(BrokerSettings, raw["broker"])
    if "orders" in raw:
        cfg.orders = _dataclass_from_dict(OrderSettings, raw["orders"])
    if "slippage" in raw:
        cfg.slippage = _dataclass_from_dict(SlippageSettings, raw["slippage"])
    if "fill_simulation" in raw:
        cfg.fill_simulation = _dataclass_from_dict(FillSimulationSettings, raw["fill_simulation"])
    return cfg
