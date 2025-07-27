# Meta Quant - Algorithmic Trading Backtesting Framework

![Language](https://img.shields.io/badge/Python-3.9%2B-blue.svg) ![Libraries](https://img.shields.io/badge/Libraries-Pandas%20%7C%20NumPy%20%7C%20PyTZ-orange.svg) ![Status](https://img.shields.io/badge/Status-Development-green.svg)

Meta Quant is a sophisticated, event-driven backtesting framework designed to test and validate algorithmic options trading strategies on Indian market indices. This project moves beyond simple signal generation to build a complete, institutional-grade simulation environment that prioritizes **adaptive risk management** and **realistic execution**.

The system is engineered to be modular and extensible, serving as a robust platform for quantitative strategy research. It simulates trading on a minute-by-minute basis, from data ingestion and caching to dynamic position sizing and performance attribution.

---

### ► Core Philosophy: Risk-First, Adaptive Trading

The central thesis of this framework is that long-term profitability is a function of superior risk management, not just superior signals. The `PortfolioManager` is the brain of the system, enforcing a "risk-first" approach inspired by professional trading desks.

Unlike static backtesters, Meta Quant is built on a dynamic core that adjusts to changing market conditions in real-time.

### ► Key Features

*   **Multi-Strategy Portfolio:** Implements a portfolio of three distinct trading models to capitalize on different market conditions:
    1.  **`AlphaPredator` (Trend-Following):** Utilizes EMA crossovers and ADX for trading in strong, directional markets.
    2.  **`SMCPredator` (Smart-Money Concepts):** Identifies high-probability reversals from institutional order blocks.
    3.  **`MeanReversion` (Range-Trading):** Uses Bollinger Bands to trade pullbacks in choppy, non-trending markets.

*   **Adaptive Regime Engine:**
    *   Dynamically classifies the market into **Trend (Bullish/Bearish) or Chop** regimes.
    *   Classifies market volatility into **High, Normal, or Low** regimes.
    *   These regimes are used to intelligently activate/deactivate strategies and modify risk parameters.

*   **Sophisticated Risk & Position Sizing:**
    *   **Per-Trade Risk Control:** Position size is calculated based on a fixed percentage of capital at risk.
    *   **Dynamic Sizing Modifiers:** Position size is adaptively adjusted based on **strategy allocation, signal confidence score, and market volatility**, allowing the system to take larger risks on high-conviction setups in stable environments.
    *   **Drawdown Circuit Breaker:** Automatically halts new trading for the day if a maximum daily drawdown limit is breached.

*   **Realistic Execution Simulation:**
    *   **Slippage & Brokerage:** Models real-world trading costs by incorporating configurable slippage (in basis points) and per-trade brokerage fees.
    *   **Option Delta Hedging:** Translates stop-losses from the underlying futures price to the traded option's price using an estimated delta, ensuring risk is managed on the correct instrument.
    *   **Conservative Fill Logic:** Assumes stop-losses are hit before take-profits in ambiguous candles, preventing over-optimistic results.

*   **Efficient Data Handling:**
    *   **`kiteconnect` Integration:** Designed to work with the Zerodha Kite Connect API for fetching historical data and instrument lists. Includes a mock API for offline development.
    *   **Performance Caching:** Caches instrument lists and OHLC data to the highly efficient **Parquet** format, dramatically speeding up subsequent runs and reducing API calls.
    *   **Timezone-Aware:** All operations are strictly handled in the `Asia/Kolkata` timezone using `pytz` to prevent critical timing errors.

### ► System Architecture

The framework is architected using Object-Oriented principles to ensure modularity and extensibility.

1.  **Data Layer (`fetch_*`)**: Handles all interaction with the data source (real or mock), including caching.
2.  **Strategy Layer (`StrategyBase` subclasses)**: Encapsulates the signal generation logic for each trading model. A new strategy can be added simply by creating a new class.
3.  **Risk Layer (`PortfolioManager`)**: The central control unit. It maintains state (capital, PnL), enforces risk rules, and has final veto power on all trading decisions.
4.  **Orchestration Layer (`run_apex_backtest`)**: The main event loop that simulates the passage of time, feeding data to the strategies and passing signals to the portfolio manager.
