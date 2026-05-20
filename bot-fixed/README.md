# Trading Bot

A production-oriented US equities algorithmic trading system focused on **intraday momentum continuation strategies using VWAP pullback entries**. This project is designed as a modular quantitative trading framework that supports:

* Premarket scanning
* Catalyst detection
* VWAP pullback execution logic
* Risk management
* Dynamic position sizing
* Backtesting and strategy evaluation
* Live execution integration

The architecture prioritizes:

* Robustness over speed
* Scalability
* Realistic execution assumptions
* Professional risk management
* Expandability into a full systematic trading platform

The system is designed around **liquid US equities** and uses **Polygon.io (Massive)** as the primary market data provider. 

---

# Core Strategy

The trading system focuses on identifying stocks with:

* Strong premarket momentum
* High relative volume
* Tight spreads
* Clean premarket structure
* Confirmed catalysts

The strategy scans for stocks likely to produce:

> High-quality VWAP pullback continuation setups after the market open.

The bot is optimized for:

* Intraday trading
* Momentum continuation
* Liquidity
* Controlled risk
* Scalability with account growth

---

# Primary Components

## 1. Premarket Scanner

The premarket scanner continuously evaluates the US equities universe between:

* **8:00 AM ET**
* **9:25 AM ET**

It filters and ranks stocks based on:

* Gap %
* Relative volume
* Premarket dollar volume
* Spread quality
* Range structure
* Trend quality
* Catalyst strength

### Scanner Features

* Universe filtering
* Premarket metric computation
* Catalyst detection
* Composite scoring
* Archetype tagging
* Ranked watchlist generation
* Priority focus list generation

### Universe Constraints

Eligible stocks must meet:

* Price >= $5
* Top 1,500 US stocks by average daily dollar volume
* Average daily dollar volume >= $20M

The scanner excludes:

* ETFs
* Preferred shares
* Warrants
* Rights
* ADRs (when possible)
* SPAC units
* Closed-end funds
* Non-common-equity instruments

### Premarket Filters

Stocks must meet all configurable thresholds:

* Premarket gap >= 3%
* Premarket volume >= 100k shares
* Relative premarket volume >= 2.0
* Premarket dollar volume >= $1M
* Bid-ask spread <= 0.25%
* Premarket range <= 15%
* Trading in upper 50% of premarket range

### Composite Premarket Score

```text
Premarket Score =
0.24 * normalized_gap_pct
+ 0.22 * normalized_relative_volume
+ 0.18 * normalized_premarket_dollar_volume
+ 0.12 * normalized_spread_inverse
+ 0.10 * normalized_premarket_range_position
+ 0.08 * normalized_trend_quality
+ 0.06 * normalized_catalyst_score
```

### Archetype Tags

The scanner classifies stocks into categories such as:

* STRONG_GAPPER
* ORDERLY_TREND
* VOLATILE_CANDIDATE
* SPREAD_RISK
* LIKELY_LEADER
* CATALYST_BACKED
* NO_CONFIRMED_CATALYST
* EXTENDED_PREMARKET

---

# 2. Catalyst Detection Layer

The catalyst system attempts to classify the reason behind a stock’s move.

Supported catalyst categories include:

* EARNINGS
* EARNINGS_GUIDANCE
* ANALYST_UPGRADE
* ANALYST_DOWNGRADE
* FDA_OR_BIOTECH_NEWS
* M_AND_A
* PARTNERSHIP_OR_CONTRACT
* PRODUCT_LAUNCH
* LEGAL_OR_REGULATORY
* MACRO_RELATED
* PRESS_RELEASE
* UNKNOWN

Each catalyst includes:

* Category
* Confidence score
* Strength score
* Text summary
* Recency

The architecture is extensible for future NLP and news API integration.

---

# 3. VWAP Pullback Strategy

The execution strategy looks for:

* Strong opening momentum
* Clean pullbacks toward VWAP
* Momentum resumption after pullback
* Healthy liquidity and spread conditions

The scanner hands off structured watchlist data to the live strategy engine after the open.

---

# 4. Position Sizing Engine

The position sizing engine ensures:

* Consistent dollar risk
* Dynamic scaling with account growth
* Risk-adjusted trade sizing
* Capital preservation

### Starting Account Parameters

* Starting equity: $10,000
* Maximum daily loss: 2%
* Risk per trade: 0.25%–0.5%
* Max open positions: 3
* Max trades/day: 6
* Max capital allocation/position: 25%–30%

### Core Formula

```text
position_size = risk_per_trade / stop_distance
```

Where:

```text
risk_per_trade = account_equity × risk_percentage
```

The sizing engine also applies:

* Liquidity constraints
* Spread adjustments
* ATR-based volatility adjustments
* Capital allocation limits
* Minimum viable trade size checks

---

# 5. Risk Management System

The risk engine is designed to prevent catastrophic losses and maintain consistent portfolio exposure.

Features include:

* Daily loss limits
* Position-level risk controls
* Portfolio-level exposure checks
* Maximum simultaneous positions
* Dynamic stop-loss integration
* Volatility-aware controls

---

# 6. Backtesting & Strategy Evaluation Engine

The backtesting framework is designed to simulate real-world trading conditions as closely as possible.

### Features

* Sequential historical replay
* Realistic execution simulation
* Spread modeling
* Slippage simulation
* Liquidity constraints
* Dynamic equity tracking
* Walk-forward testing
* Parameter sensitivity analysis
* Out-of-sample validation

### Supported Data

Using Polygon/Massive historical market data:

* OHLCV bars
* 1-minute bars
* Optional 5-minute bars
* Trade volume
* Historical VWAP
* Optional quote data

### Performance Metrics

The engine computes:

* Total return
* Annualized return
* Maximum drawdown
* Sharpe ratio
* Sortino ratio
* Win rate
* Profit factor
* Equity curve
* Monthly returns
* Trade duration statistics

---

# System Architecture

```text
                +----------------------+
                | Polygon/Massive API |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Market Data Layer    |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Premarket Scanner    |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Catalyst Detection   |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Scoring & Ranking    |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Watchlist Engine     |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | VWAP Strategy Engine |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Risk Management      |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Position Sizing      |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Execution Engine     |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Logging & Analytics  |
                +----------------------+
```

---

# Proposed Project Structure

```text
trading-bot/
│
├── README.md
├── requirements.txt
├── config/
│   ├── scanner_config.yaml
│   ├── risk_config.yaml
│   ├── strategy_config.yaml
│   └── execution_config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── historical/
│
├── market_data/
│   ├── polygon_client.py
│   ├── websocket_handler.py
│   ├── universe_builder.py
│   └── market_cache.py
│
├── scanner/
│   ├── premarket_scanner.py
│   ├── metrics.py
│   ├── filters.py
│   ├── scoring.py
│   ├── tagging.py
│   └── watchlist.py
│
├── catalysts/
│   ├── catalyst_engine.py
│   ├── news_classifier.py
│   └── catalyst_scoring.py
│
├── strategy/
│   ├── vwap_strategy.py
│   ├── entry_signals.py
│   ├── exits.py
│   └── trade_manager.py
│
├── risk/
│   ├── risk_engine.py
│   ├── stop_loss_engine.py
│   ├── position_sizing.py
│   └── portfolio_constraints.py
│
├── execution/
│   ├── execution_engine.py
│   ├── slippage_model.py
│   ├── order_router.py
│   └── fill_simulator.py
│
├── backtesting/
│   ├── backtester.py
│   ├── historical_replay.py
│   ├── performance_metrics.py
│   ├── walk_forward.py
│   └── parameter_optimization.py
│
├── analytics/
│   ├── equity_curve.py
│   ├── trade_analysis.py
│   └── reporting.py
│
├── logs/
│
└── tests/
```

---

# Engineering Principles

The system is designed with the following engineering standards:

* Modular architecture
* Strong separation of concerns
* Type hints throughout
* Parameterized configuration
* Production-oriented design
* Reusable components
* Backtest/live parity
* Extensible interfaces
* Comprehensive logging

---

# Future Roadmap

Potential future expansions include:

* Multi-strategy support
* Machine learning ranking models
* NLP-driven catalyst analysis
* Multi-timeframe confirmation
* Portfolio optimization
* Reinforcement learning execution logic
* Broker integration
* Cloud deployment
* Distributed scanning architecture
* Real-time monitoring dashboards

---

# Data Provider

The system uses:

* [Polygon.io (Massive)](https://polygon.io/?utm_source=chatgpt.com)

Polygon was selected for:

* Real-time equities data
* Historical market data
* REST + WebSocket APIs
* Tick-level support
* Strong developer tooling
* Scalability for algorithmic trading systems

