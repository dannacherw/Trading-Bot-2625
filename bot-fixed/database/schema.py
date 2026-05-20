"""
database/schema.py
All SQLite table definitions. Schema versioned via migrations table.
"""

SCHEMA_VERSION = 4

CREATE_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

MIGRATIONS: dict[int, str] = {
    1: """
-- v1: core tables
CREATE TABLE IF NOT EXISTS bars (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL,
    timeframe   TEXT    NOT NULL DEFAULT '1m',
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    volume      INTEGER NOT NULL,
    vwap        REAL,
    UNIQUE(symbol, timestamp, timeframe)
);
CREATE INDEX IF NOT EXISTS idx_bars_symbol_ts ON bars(symbol, timestamp);

CREATE TABLE IF NOT EXISTS quotes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL,
    bid         REAL    NOT NULL,
    ask         REAL    NOT NULL,
    bid_size    INTEGER DEFAULT 0,
    ask_size    INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_quotes_symbol_ts ON quotes(symbol, timestamp);

CREATE TABLE IF NOT EXISTS scan_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    scanned_at      TEXT    NOT NULL,
    composite_score REAL    NOT NULL,
    passes_filters  INTEGER NOT NULL DEFAULT 0,
    gap_pct         REAL,
    relative_volume REAL,
    pm_dollar_vol   REAL,
    spread_pct      REAL,
    archetypes      TEXT,
    catalyst_cat    TEXT,
    catalyst_conf   REAL,
    raw_json        TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scan_symbol_ts ON scan_results(symbol, scanned_at);
""",

    2: """
-- v2: orders, positions, trades
CREATE TABLE IF NOT EXISTS orders (
    order_id        TEXT    PRIMARY KEY,
    symbol          TEXT    NOT NULL,
    side            TEXT    NOT NULL,
    order_type      TEXT    NOT NULL,
    quantity        INTEGER NOT NULL,
    limit_price     REAL,
    stop_price      REAL,
    time_in_force   TEXT    NOT NULL DEFAULT 'DAY',
    status          TEXT    NOT NULL DEFAULT 'PENDING',
    submitted_at    TEXT,
    filled_at       TEXT,
    filled_quantity INTEGER DEFAULT 0,
    avg_fill_price  REAL,
    broker_order_id TEXT,
    notes           TEXT,
    raw_json        TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);

CREATE TABLE IF NOT EXISTS positions (
    position_id         TEXT    PRIMARY KEY,
    symbol              TEXT    NOT NULL,
    side                TEXT    NOT NULL,
    status              TEXT    NOT NULL DEFAULT 'OPEN',
    entry_price         REAL    NOT NULL,
    entry_time          TEXT    NOT NULL,
    quantity            INTEGER NOT NULL,
    remaining_quantity  INTEGER NOT NULL,
    stop_price          REAL    NOT NULL,
    target_1_price      REAL    NOT NULL,
    target_2_price      REAL    NOT NULL,
    breakeven_price     REAL,
    trailing_stop_price REAL,
    exit_price          REAL,
    exit_time           TEXT,
    exit_reason         TEXT,
    realized_pnl        REAL    DEFAULT 0.0,
    commission          REAL    DEFAULT 0.0,
    signal_id           TEXT,
    raw_json            TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);

CREATE TABLE IF NOT EXISTS trades (
    trade_id            TEXT    PRIMARY KEY,
    position_id         TEXT    NOT NULL,
    symbol              TEXT    NOT NULL,
    side                TEXT    NOT NULL,
    entry_price         REAL    NOT NULL,
    exit_price          REAL    NOT NULL,
    quantity            INTEGER NOT NULL,
    entry_time          TEXT    NOT NULL,
    exit_time           TEXT    NOT NULL,
    exit_reason         TEXT    NOT NULL,
    gross_pnl           REAL    NOT NULL,
    commission          REAL    NOT NULL,
    net_pnl             REAL    NOT NULL,
    r_multiple          REAL    NOT NULL,
    hold_duration_secs  REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);
""",

    3: """
-- v3: daily risk state, signals
CREATE TABLE IF NOT EXISTS daily_risk_state (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT    NOT NULL UNIQUE,
    starting_equity REAL    NOT NULL,
    current_equity  REAL    NOT NULL,
    realized_pnl    REAL    NOT NULL DEFAULT 0.0,
    unrealized_pnl  REAL    NOT NULL DEFAULT 0.0,
    trades_today    INTEGER NOT NULL DEFAULT 0,
    open_positions  INTEGER NOT NULL DEFAULT 0,
    daily_loss_used REAL    NOT NULL DEFAULT 0.0,
    is_halted       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS signals (
    signal_id       TEXT    PRIMARY KEY,
    symbol          TEXT    NOT NULL,
    signal_type     TEXT    NOT NULL,
    strength        TEXT    NOT NULL,
    generated_at    TEXT    NOT NULL,
    entry_price     REAL    NOT NULL,
    stop_price      REAL    NOT NULL,
    target_1_price  REAL    NOT NULL,
    target_2_price  REAL    NOT NULL,
    suggested_qty   INTEGER NOT NULL DEFAULT 0,
    vwap_at_signal  REAL,
    notes           TEXT,
    acted_on        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts ON signals(symbol, generated_at);
""",

    4: """
-- v4: missed_trade_audit, float_data_cache, kill_switch_events,
--     percentile_distributions, scan_window_log

CREATE TABLE IF NOT EXISTS missed_trade_audit (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id                     INTEGER NOT NULL,
    symbol                      TEXT    NOT NULL,
    audit_date                  TEXT    NOT NULL,
    audited_at                  TEXT    NOT NULL,
    rejection_filter            TEXT    NOT NULL,
    entry_proxy                 REAL    NOT NULL,
    stop_proxy                  REAL    NOT NULL,
    target_1                    REAL    NOT NULL,
    target_2                    REAL    NOT NULL,
    risk_per_share              REAL    NOT NULL,
    hit_1r                      INTEGER NOT NULL DEFAULT 0,
    hit_2r                      INTEGER NOT NULL DEFAULT 0,
    hit_stop                    INTEGER NOT NULL DEFAULT 0,
    max_favorable_excursion_r   REAL    NOT NULL DEFAULT 0.0,
    max_adverse_excursion_r     REAL    NOT NULL DEFAULT 0.0,
    time_to_1r_minutes          REAL,
    time_to_2r_minutes          REAL,
    time_to_stop_minutes        REAL,
    session_high                REAL,
    session_low                 REAL,
    session_close               REAL,
    rejection_was_correct       INTEGER NOT NULL DEFAULT 0,
    should_investigate          INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_missed_audit_date   ON missed_trade_audit(audit_date);
CREATE INDEX IF NOT EXISTS idx_missed_audit_symbol ON missed_trade_audit(symbol);
CREATE INDEX IF NOT EXISTS idx_missed_audit_filter ON missed_trade_audit(rejection_filter);

CREATE TABLE IF NOT EXISTS float_data_cache (
    symbol          TEXT    NOT NULL,
    float_shares    INTEGER,
    was_fetched     INTEGER NOT NULL DEFAULT 1,
    source          TEXT    NOT NULL DEFAULT 'unknown',
    fetched_date    TEXT    NOT NULL,
    fetched_at      TEXT    NOT NULL,
    PRIMARY KEY (symbol, fetched_date)
);
CREATE INDEX IF NOT EXISTS idx_float_symbol ON float_data_cache(symbol);

CREATE TABLE IF NOT EXISTS kill_switch_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger         TEXT    NOT NULL,
    triggered_at    TEXT    NOT NULL,
    reason          TEXT    NOT NULL,
    value           REAL,
    threshold       REAL,
    resolved_at     TEXT,
    resolved_by     TEXT
);
CREATE INDEX IF NOT EXISTS idx_kill_switch_ts ON kill_switch_events(triggered_at);

CREATE TABLE IF NOT EXISTS percentile_distributions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    feature          TEXT    NOT NULL,
    computed_at      TEXT    NOT NULL,
    sample_count     INTEGER NOT NULL,
    quantiles_json   TEXT    NOT NULL,
    date_range_start TEXT    NOT NULL,
    date_range_end   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_percentile_feature ON percentile_distributions(feature, computed_at);

CREATE TABLE IF NOT EXISTS scan_window_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at        TEXT    NOT NULL,
    allowed          INTEGER NOT NULL,
    reason           TEXT    NOT NULL,
    in_warning_zone  INTEGER NOT NULL DEFAULT 0,
    minutes_to_close REAL
);
""",
}
