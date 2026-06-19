# BTC VWAP Backtesting Framework

[Project write up PDF](systematic-trading-research-vwap.pdf)

Custom Python research framework for testing systematic BTCUSDT futures strategies using Binance USD-M Futures market data.

This project focuses on a VWAP mean-reversion strategy with trend, volume, and holding-period constraints. It includes data loading, indicator generation, event-driven signal logic, backtesting, grid search, walk-forward validation, and result reporting.

## Strategy Overview

The strategy uses BTCUSDT daily candles and 5-minute candles.

- Daily candles are used to calculate rolling VWAP, distance standard deviation, moving averages, and average quote volume.
- 5-minute candles are used for signal checks, rolling 24-hour quote volume, trade execution timing, and equity curve construction.
- Z-score is calculated as:

```text
(5-minute close - rolling daily VWAP) / rolling daily distance standard deviation
```

The strategy can enter long or short positions when price is far from VWAP, exit when z-score mean-reverts, filter entries by moving-average trend, force exits after max holding periods, and block trading during high-volume conditions.

## Project Structure

```text
data/
notebooks/
results/
src/
  data/
    load_data.py
  indicators/
    vwap.py
  backtest.py
  strategy.py
download_binance_history.py
grid_search.py
main.py
volume_report.py
walk_forward_validation.py
```

## Main Components

- `main.py` runs a single backtest using the active parameter settings.
- `grid_search.py` runs large parameter sweeps using multiprocessing.
- `walk_forward_validation.py` validates selected parameter sets over later out-of-sample windows.
- `volume_report.py` summarizes monthly quote volume behavior.
- `download_binance_history.py` downloads Binance monthly futures kline data.
- `src/strategy.py` contains the event-driven position logic.
- `src/backtest.py` calculates returns, fees, equity curve, performance summaries, and trade summaries.
- `src/indicators/vwap.py` calculates rolling VWAP and distance statistics.

## Research Features

- Vectorized pandas calculations for indicators and data preparation.
- Event-driven strategy logic for stateful position management.
- Multiprocessing grid search using `ProcessPoolExecutor`.
- Per-worker signal-data caching for faster large sweeps.
- Array-based grid-search metric computation to reduce repeated DataFrame overhead.
- Walk-forward validation across sequential out-of-sample windows.
- Equity curve chart, trade summary chart, and performance summary output.

## Data

Historical data is expected in `data/raw/`.

The downloader can be run with:

```powershell
python download_binance_history.py
```

Raw market data is intentionally ignored by git. Re-download it locally when reproducing results.

## Running A Backtest

Activate the virtual environment, then run:

```powershell
.\venv\Scripts\python.exe main.py
```

Key parameters are near the top of `main.py`.

## Running A Grid Search

Edit the parameter lists near the top of `grid_search.py`, then run:

```powershell
.\venv\Scripts\python.exe grid_search.py
```

The grid search writes leaderboards and candidate parameter sets into `results/`.

## Walk-Forward Validation

Set `VALIDATION_STAGE` near the top of `walk_forward_validation.py`, update the backtest window in `main.py`, then run:

```powershell
.\venv\Scripts\python.exe walk_forward_validation.py
```

Each stage reads the prior stage's survivor list and writes a new validation CSV and survivor CSV.

## Selected Outputs

The public repo keeps only the most useful lightweight outputs:

- `results/equity_curve.png`
- `results/performance_summary.txt`
- `results/trade_summary.png`

Large CSV outputs are generated locally and ignored by git.

## Disclaimer

This project is for research and educational purposes only. It is not financial advice. Backtest results are hypothetical and may not fully account for real-world execution issues such as slippage, funding, liquidity constraints, exchange outages, or liquidation risk.
