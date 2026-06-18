from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
import os

import numpy as np
import pandas as pd

from main import (
    BACKTEST_END,
    BACKTEST_START,
    DAILY_FILES,
    DAILY_WARMUP_START,
    FIVE_MIN_FILES,
    MAX_HOLD_HOURS,
    filter_date_range,
)
from src.data.load_data import load_binance_klines
from src.indicators.vwap import add_daily_rolling_vwap


# Parameter grid. Edit these lists to define the candidate-discovery sweep.
# This is a candidate-discovery sweep, not final parameter optimization.
# Keep the grid broad enough to find promising regions, then forward test later.
MAX_HOLD_BARS_VALUES = [ 500, 750, 1000 ]
MOVING_AVERAGE_LENGTH_VALUES = [ 100, 150]
Z_SCORE_ENTRY_VALUES = [ 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, ]
Z_SCORE_EXIT_VALUES = [0.0, -0.05, -0.1, -0.125, -0.149]
VWAP_LENGTH_VALUES = [3, 4, 5, 6, 7, 8, 9, 10]
HIGH_QUOTE_VOLUME_MULTIPLIER_VALUES = [ 0.5, 0.75, 1.0, 1.66, 2.0, 2.5, 50.0]
HIGH_VOLUME_EXIT_Z_VALUES = [0.1, 0.05, 0.0, -0.05, -0.1, -0.125, -0.149]

MIN_TRADE_COUNT = 30
MAX_ACCEPTABLE_DRAWDOWN = -0.60
PERIODS_PER_YEAR = 365 * 24 * 12
PROGRESS_EVERY = 25
TOP_LEADERBOARD_COUNT = 40
SIGNAL_CACHE_SIZE = 2
MAX_WORKERS = min(8, os.cpu_count() or 1)
MAP_CHUNKSIZE = 5

RESULTS_DIR = Path("results")
FULL_RESULTS_PATH = RESULTS_DIR / "grid_search_results.csv"
TOP_SHARPE_PATH = RESULTS_DIR / "top_40_by_sharpe.csv"
TOP_CALMAR_PATH = RESULTS_DIR / "top_40_by_calmar.csv"
CANDIDATES_PATH = RESULTS_DIR / "candidate_parameter_sets.csv"
LOW_TRADE_COUNT_PATH = RESULTS_DIR / "rejected_low_trade_count.csv"

_DAILY_DF = None
_FIVE_MIN_DF = None
_SIGNAL_CACHE = None


def init_worker():
    """Load and filter historical data once per worker process."""
    global _DAILY_DF, _FIVE_MIN_DF, _SIGNAL_CACHE

    daily_df = load_binance_klines(DAILY_FILES)
    five_min_df = load_binance_klines(FIVE_MIN_FILES)

    _DAILY_DF = filter_date_range(daily_df, DAILY_WARMUP_START, BACKTEST_END)
    _FIVE_MIN_DF = filter_date_range(five_min_df, BACKTEST_START, BACKTEST_END)
    _SIGNAL_CACHE = OrderedDict()


def generate_parameter_grid():
    """Build every parameter combination in the grid."""
    parameter_grid = []

    # Group by the expensive prepared-data inputs so each worker can reuse
    # prepared signal data across many strategy parameter combinations.
    for moving_average_length in MOVING_AVERAGE_LENGTH_VALUES:
        for vwap_length in VWAP_LENGTH_VALUES:
            for (
                max_hold_bars,
                z_score_entry,
                z_score_exit,
                high_quote_volume_multiplier,
                high_volume_exit_z,
            ) in product(
                MAX_HOLD_BARS_VALUES,
                Z_SCORE_ENTRY_VALUES,
                Z_SCORE_EXIT_VALUES,
                HIGH_QUOTE_VOLUME_MULTIPLIER_VALUES,
                HIGH_VOLUME_EXIT_Z_VALUES,
            ):
                parameter_grid.append(
                    (
                        max_hold_bars,
                        moving_average_length,
                        z_score_entry,
                        z_score_exit,
                        vwap_length,
                        high_quote_volume_multiplier,
                        high_volume_exit_z,
                    )
                )

    return parameter_grid


def prepare_signal_data_for_grid(daily_df, five_min_df, vwap_length, moving_average_length):
    """Prepare signal data using the same strategy inputs with variable windows."""
    daily = add_daily_rolling_vwap(daily_df, window=vwap_length)
    five_min = five_min_df.copy()

    daily["trend_ma"] = (
        daily["close"]
        .rolling(window=moving_average_length, min_periods=moving_average_length)
        .mean()
    )
    daily["avg_quote_volume_30d"] = (
        daily["quote_volume"].rolling(window=30, min_periods=30).mean()
    )
    daily["date"] = daily["open_time"].dt.date
    five_min["date"] = five_min["open_time"].dt.date
    five_min["quote_volume_24h"] = (
        five_min["quote_volume"].rolling(window=288, min_periods=288).sum()
    )

    daily_signals = daily[
        ["date", "vwap", "distance_std", "trend_ma", "avg_quote_volume_30d"]
    ].copy()

    # Shift all daily fields so each 5-minute candle only sees completed days.
    daily_signals[
        ["vwap", "distance_std", "trend_ma", "avg_quote_volume_30d"]
    ] = daily_signals[
        ["vwap", "distance_std", "trend_ma", "avg_quote_volume_30d"]
    ].shift(1)

    merged = five_min.merge(daily_signals, on="date", how="left")
    merged = merged.dropna(
        subset=[
            "vwap",
            "distance_std",
            "trend_ma",
            "avg_quote_volume_30d",
            "quote_volume_24h",
        ]
    ).copy()
    merged = merged[merged["distance_std"] != 0].copy()

    merged["z_score"] = (merged["close"] - merged["vwap"]) / merged["distance_std"]

    return merged[
        [
            "open_time",
            "close_time",
            "close",
            "trend_ma",
            "quote_volume_24h",
            "avg_quote_volume_30d",
            "z_score",
        ]
    ].reset_index(drop=True)


def get_prepared_signal_data(vwap_length, moving_average_length):
    """Reuse prepared signal data within each worker process."""
    cache_key = (vwap_length, moving_average_length)

    if cache_key in _SIGNAL_CACHE:
        _SIGNAL_CACHE.move_to_end(cache_key)
        return _SIGNAL_CACHE[cache_key]

    signal_df = prepare_signal_data_for_grid(
        _DAILY_DF,
        _FIVE_MIN_DF,
        vwap_length=vwap_length,
        moving_average_length=moving_average_length,
    )
    _SIGNAL_CACHE[cache_key] = signal_df

    if len(_SIGNAL_CACHE) > SIGNAL_CACHE_SIZE:
        _SIGNAL_CACHE.popitem(last=False)

    return signal_df


def generate_positions_for_grid(
    signal_df,
    entry_z,
    exit_z,
    high_quote_volume_multiplier,
    high_volume_exit_z,
    max_hold_hours,
    max_hold_bars,
):
    """Generate positions with the same rules as src.strategy.generate_signals."""
    row_count = len(signal_df)
    positions = np.zeros(row_count, dtype=np.int8)

    position = 0
    entry_time = None
    entry_bar_index = None
    max_hold_delta = None
    high_volume_block_active = False
    high_volume_block_side = 0
    high_volume_event_released = False
    post_exit_wait_side = 0

    if max_hold_hours is not None:
        max_hold_delta = pd.Timedelta(hours=max_hold_hours).to_timedelta64()

    z_scores = signal_df["z_score"].to_numpy()
    open_times = signal_df["open_time"].to_numpy()
    closes = signal_df["close"].to_numpy()
    trend_ma = signal_df["trend_ma"].to_numpy()
    high_quote_volume_events = (
        signal_df["quote_volume_24h"].to_numpy()
        > signal_df["avg_quote_volume_30d"].to_numpy() * high_quote_volume_multiplier
    )
    long_allowed_values = closes > trend_ma
    short_allowed_values = closes < trend_ma

    for bar_index in range(row_count):
        z_score = z_scores[bar_index]

        if np.isnan(z_score):
            positions[bar_index] = position
            continue

        high_quote_volume_event = high_quote_volume_events[bar_index]

        if not high_quote_volume_event:
            high_volume_event_released = False

        if (
            high_quote_volume_event
            and not high_volume_event_released
            and not high_volume_block_active
        ):
            position = 0
            entry_time = None
            entry_bar_index = None
            high_volume_block_active = True
            high_volume_block_side = np.sign(z_score)

        if high_volume_block_active:
            position = 0
            entry_time = None
            entry_bar_index = None

            if (
                high_volume_block_side == 0
                or (high_volume_block_side > 0 and z_score <= high_volume_exit_z)
                or (high_volume_block_side < 0 and z_score >= -high_volume_exit_z)
            ):
                high_volume_block_active = False
                high_volume_event_released = True

            positions[bar_index] = position
            continue

        if post_exit_wait_side != 0:
            position = 0

            if (
                (post_exit_wait_side > 0 and z_score >= -exit_z)
                or (post_exit_wait_side < 0 and z_score <= exit_z)
            ):
                post_exit_wait_side = 0

            positions[bar_index] = position
            continue

        if (
            position != 0
            and max_hold_delta is not None
            and entry_time is not None
            and open_times[bar_index] - entry_time > max_hold_delta
        ):
            post_exit_wait_side = position
            position = 0
            entry_time = None
            entry_bar_index = None
            positions[bar_index] = position
            continue

        if (
            position != 0
            and max_hold_bars is not None
            and entry_bar_index is not None
            and bar_index - entry_bar_index > max_hold_bars
        ):
            post_exit_wait_side = position
            position = 0
            entry_time = None
            entry_bar_index = None
            positions[bar_index] = position
            continue

        if position == 0:
            if long_allowed_values[bar_index] and z_score <= -entry_z:
                position = 1
                entry_time = open_times[bar_index]
                entry_bar_index = bar_index
            elif short_allowed_values[bar_index] and z_score >= entry_z:
                position = -1
                entry_time = open_times[bar_index]
                entry_bar_index = bar_index
        elif position == 1:
            if z_score >= -exit_z:
                position = 0
                entry_time = None
                entry_bar_index = None
        elif position == -1:
            if z_score <= exit_z:
                position = 0
                entry_time = None
                entry_bar_index = None

        positions[bar_index] = position

    return positions


def calculate_grid_metrics_fast(signal_df, positions, fee_rate=0.0004):
    """Calculate grid metrics without building full backtest/trade DataFrames."""
    if len(signal_df) == 0:
        return empty_metrics()

    closes = signal_df["close"].to_numpy(dtype=float)

    returns = np.zeros(len(closes), dtype=float)
    returns[1:] = (closes[1:] / closes[:-1]) - 1

    position_lagged = np.zeros(len(positions), dtype=float)
    position_lagged[1:] = positions[:-1]

    turnover = np.zeros(len(position_lagged), dtype=float)
    turnover[0] = abs(position_lagged[0])
    turnover[1:] = np.abs(np.diff(position_lagged))

    fees = turnover * fee_rate
    strategy_returns = (position_lagged * returns) - fees
    equity_curve = np.cumprod(1 + strategy_returns)

    total_return = equity_curve[-1] - 1
    max_drawdown = calculate_max_drawdown_array(equity_curve)
    sharpe = annualized_sharpe_array(strategy_returns)
    sortino = annualized_sortino_array(strategy_returns)
    calmar = calculate_calmar(total_return, max_drawdown)

    trade_metrics = calculate_trade_metrics_fast(position_lagged, strategy_returns)

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "trade_count": trade_metrics["trade_count"],
        "win_rate": trade_metrics["win_rate"],
        "avg_win": trade_metrics["avg_win"],
        "avg_loss": trade_metrics["avg_loss"],
        "profit_factor": trade_metrics["profit_factor"],
        "avg_bars_held": trade_metrics["avg_bars_held"],
        "median_bars_held": trade_metrics["median_bars_held"],
        "exposure_pct": (np.abs(position_lagged) > 0).mean(),
        "total_fees_paid": fees.sum(),
        "largest_win_pct": trade_metrics["largest_win_pct"],
        "largest_loss_pct": trade_metrics["largest_loss_pct"],
    }


def calculate_trade_metrics_fast(position_lagged, strategy_returns):
    """Replicate summarize_trades metrics from position changes."""
    sides = np.sign(position_lagged)
    change_indices = np.flatnonzero(
        np.r_[True, sides[1:] != sides[:-1]]
    )
    cumulative_growth = np.cumprod(1 + strategy_returns)

    trade_returns = []
    bars_held = []
    open_trade = None
    previous_side = 0

    def interval_return(start_index, end_index):
        if end_index < start_index:
            return 0.0

        start_growth = cumulative_growth[start_index - 1] if start_index > 0 else 1.0
        return (cumulative_growth[end_index] / start_growth) - 1

    for index in change_indices:
        current_side = int(sides[index])

        if previous_side == 0 and current_side != 0:
            open_trade = {
                "entry_index": max(index - 1, 0),
                "entry_return_index": index,
            }

        elif previous_side != 0 and current_side == 0 and open_trade is not None:
            exit_index = max(index - 1, 0)
            trade_returns.append(
                interval_return(open_trade["entry_return_index"], index)
            )
            bars_held.append(exit_index - open_trade["entry_index"])
            open_trade = None

        elif previous_side != 0 and current_side != 0 and previous_side != current_side:
            exit_index = max(index - 1, 0)

            if open_trade is not None:
                trade_returns.append(
                    interval_return(open_trade["entry_return_index"], exit_index)
                )
                bars_held.append(exit_index - open_trade["entry_index"])

            open_trade = {
                "entry_index": max(index - 1, 0),
                "entry_return_index": index,
            }

        previous_side = current_side

    if open_trade is not None:
        exit_index = len(position_lagged) - 1
        trade_returns.append(
            interval_return(open_trade["entry_return_index"], exit_index)
        )
        bars_held.append(exit_index - open_trade["entry_index"])

    if not trade_returns:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "avg_bars_held": 0.0,
            "median_bars_held": 0.0,
            "largest_win_pct": 0.0,
            "largest_loss_pct": 0.0,
        }

    trade_returns = np.asarray(trade_returns, dtype=float)
    bars_held = np.asarray(bars_held, dtype=float)
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    gross_profit = wins.sum()
    gross_loss = losses.sum()

    if gross_loss < 0:
        profit_factor = gross_profit / abs(gross_loss)
    elif gross_profit > 0:
        profit_factor = np.inf
    else:
        profit_factor = 0.0

    return {
        "trade_count": len(trade_returns),
        "win_rate": len(wins) / len(trade_returns),
        "avg_win": wins.mean() if len(wins) else 0.0,
        "avg_loss": losses.mean() if len(losses) else 0.0,
        "profit_factor": profit_factor,
        "avg_bars_held": bars_held.mean(),
        "median_bars_held": np.median(bars_held),
        "largest_win_pct": wins.max() if len(wins) else 0.0,
        "largest_loss_pct": losses.min() if len(losses) else 0.0,
    }


def annualized_sharpe_array(returns):
    """Calculate annualized Sharpe from a NumPy return array."""
    std = returns.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0

    return (returns.mean() / std) * (PERIODS_PER_YEAR ** 0.5)


def annualized_sortino_array(returns):
    """Calculate annualized Sortino from a NumPy return array."""
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std(ddof=1)
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0

    return (returns.mean() / downside_std) * (PERIODS_PER_YEAR ** 0.5)


def calculate_max_drawdown_array(equity_curve):
    """Calculate max drawdown from a NumPy equity curve."""
    drawdown = equity_curve / np.maximum.accumulate(equity_curve) - 1
    return drawdown.min()


def run_parameter_combination(params):
    """Run one full backtest for a single parameter combination."""
    (
        max_hold_bars,
        moving_average_length,
        z_score_entry,
        z_score_exit,
        vwap_length,
        high_quote_volume_multiplier,
        high_volume_exit_z,
    ) = params

    try:
        signal_df = get_prepared_signal_data(
            vwap_length=vwap_length,
            moving_average_length=moving_average_length,
        )

        positions = generate_positions_for_grid(
            signal_df,
            entry_z=z_score_entry,
            exit_z=z_score_exit,
            high_quote_volume_multiplier=high_quote_volume_multiplier,
            high_volume_exit_z=high_volume_exit_z,
            max_hold_hours=MAX_HOLD_HOURS,
            max_hold_bars=max_hold_bars,
        )

        metrics = calculate_grid_metrics_fast(signal_df, positions)
        metrics.update(
            {
                "max_hold_bars": max_hold_bars,
                "moving_average_length": moving_average_length,
                "z_score_entry": z_score_entry,
                "z_score_exit": z_score_exit,
                "vwap_length": vwap_length,
                "high_quote_volume_multiplier": high_quote_volume_multiplier,
                "high_volume_exit_z": high_volume_exit_z,
            }
        )

        metrics["passes_trade_count_filter"] = (
            metrics["trade_count"] >= MIN_TRADE_COUNT
        )
        metrics["candidate_for_walk_forward"] = (
            metrics["passes_trade_count_filter"]
            and metrics["sharpe"] > 0
            and metrics["total_return"] > 0
            and metrics["max_drawdown"] > MAX_ACCEPTABLE_DRAWDOWN
        )

        return metrics

    except Exception as error:
        return {
            "max_hold_bars": max_hold_bars,
            "moving_average_length": moving_average_length,
            "z_score_entry": z_score_entry,
            "z_score_exit": z_score_exit,
            "vwap_length": vwap_length,
            "high_quote_volume_multiplier": high_quote_volume_multiplier,
            "high_volume_exit_z": high_volume_exit_z,
            "sharpe": np.nan,
            "sortino": np.nan,
            "total_return": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
            "trade_count": 0,
            "win_rate": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "profit_factor": np.nan,
            "avg_bars_held": np.nan,
            "median_bars_held": np.nan,
            "exposure_pct": np.nan,
            "total_fees_paid": np.nan,
            "largest_win_pct": np.nan,
            "largest_loss_pct": np.nan,
            "passes_trade_count_filter": False,
            "candidate_for_walk_forward": False,
            "error": str(error),
        }


def calculate_grid_metrics(backtest_df, trade_summary_df):
    """Calculate performance, trade, risk, and execution metrics for one run."""
    if backtest_df.empty:
        return empty_metrics()

    returns = backtest_df["strategy_returns"]
    equity_curve = backtest_df["equity_curve"]

    total_return = equity_curve.iloc[-1] - 1
    max_drawdown = calculate_max_drawdown(equity_curve)
    sharpe = annualized_sharpe(returns)
    sortino = annualized_sortino(returns)
    calmar = calculate_calmar(total_return, max_drawdown)

    exposure_pct = (backtest_df["position_lagged"].abs() > 0).mean()
    total_fees_paid = backtest_df["fees"].sum()

    trade_metrics = calculate_trade_metrics(trade_summary_df)

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "trade_count": trade_metrics["trade_count"],
        "win_rate": trade_metrics["win_rate"],
        "avg_win": trade_metrics["avg_win"],
        "avg_loss": trade_metrics["avg_loss"],
        "profit_factor": trade_metrics["profit_factor"],
        "avg_bars_held": trade_metrics["avg_bars_held"],
        "median_bars_held": trade_metrics["median_bars_held"],
        "exposure_pct": exposure_pct,
        "total_fees_paid": total_fees_paid,
        "largest_win_pct": trade_metrics["largest_win_pct"],
        "largest_loss_pct": trade_metrics["largest_loss_pct"],
    }


def empty_metrics():
    """Return default metrics for a run with no usable backtest rows."""
    return {
        "sharpe": 0.0,
        "sortino": 0.0,
        "total_return": 0.0,
        "max_drawdown": 0.0,
        "calmar": 0.0,
        "trade_count": 0,
        "win_rate": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "profit_factor": 0.0,
        "avg_bars_held": 0.0,
        "median_bars_held": 0.0,
        "exposure_pct": 0.0,
        "total_fees_paid": 0.0,
        "largest_win_pct": 0.0,
        "largest_loss_pct": 0.0,
    }


def annualized_sharpe(returns):
    """Calculate annualized Sharpe from 5-minute strategy returns."""
    std = returns.std()
    if std == 0 or pd.isna(std):
        return 0.0

    return (returns.mean() / std) * (PERIODS_PER_YEAR ** 0.5)


def annualized_sortino(returns):
    """Calculate annualized Sortino using downside volatility only."""
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    if downside_std == 0 or pd.isna(downside_std):
        return 0.0

    return (returns.mean() / downside_std) * (PERIODS_PER_YEAR ** 0.5)


def calculate_max_drawdown(equity_curve):
    """Calculate max peak-to-trough equity drawdown."""
    drawdown = equity_curve / equity_curve.cummax() - 1
    return drawdown.min()


def calculate_calmar(total_return, max_drawdown):
    """Calculate Calmar-like return divided by absolute max drawdown."""
    if max_drawdown == 0 or pd.isna(max_drawdown):
        return 0.0

    return total_return / abs(max_drawdown)


def calculate_trade_metrics(trade_summary_df):
    """Calculate trade-level win/loss and holding-period statistics."""
    if trade_summary_df.empty:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "avg_bars_held": 0.0,
            "median_bars_held": 0.0,
            "largest_win_pct": 0.0,
            "largest_loss_pct": 0.0,
        }

    returns = trade_summary_df["net_return"]
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    gross_profit = wins.sum()
    gross_loss = losses.sum()

    if gross_loss < 0:
        profit_factor = gross_profit / abs(gross_loss)
    elif gross_profit > 0:
        profit_factor = np.inf
    else:
        profit_factor = 0.0

    return {
        "trade_count": len(trade_summary_df),
        "win_rate": len(wins) / len(trade_summary_df),
        "avg_win": wins.mean() if len(wins) else 0.0,
        "avg_loss": losses.mean() if len(losses) else 0.0,
        "profit_factor": profit_factor,
        "avg_bars_held": trade_summary_df["bars_held"].mean(),
        "median_bars_held": trade_summary_df["bars_held"].median(),
        "largest_win_pct": wins.max() if len(wins) else 0.0,
        "largest_loss_pct": losses.min() if len(losses) else 0.0,
    }


def build_candidate_parameter_sets(results_df):
    """Select a diverse candidate set for later walk-forward testing."""
    eligible = results_df[results_df["candidate_for_walk_forward"]].copy()
    if eligible.empty:
        return eligible

    candidate_parts = [
        eligible.sort_values("sharpe", ascending=False).head(TOP_LEADERBOARD_COUNT),
        eligible.sort_values("calmar", ascending=False).head(TOP_LEADERBOARD_COUNT),
        eligible[
            (eligible["trade_count"] >= MIN_TRADE_COUNT * 2)
            & (eligible["sharpe"] > 0)
            & (eligible["total_return"] > 0)
        ]
        .sort_values(["trade_count", "sharpe"], ascending=[False, False])
        .head(20),
    ]

    region_scores = (
        eligible.groupby(["moving_average_length", "vwap_length"], as_index=False)
        .agg(
            region_count=("sharpe", "count"),
            median_sharpe=("sharpe", "median"),
            median_calmar=("calmar", "median"),
        )
        .query("region_count >= 3")
        .sort_values(["median_sharpe", "median_calmar"], ascending=False)
        .head(10)
    )

    for _, region in region_scores.iterrows():
        region_rows = eligible[
            (eligible["moving_average_length"] == region["moving_average_length"])
            & (eligible["vwap_length"] == region["vwap_length"])
        ]
        candidate_parts.append(
            region_rows.sort_values(["sharpe", "calmar"], ascending=False).head(3)
        )

    candidates = pd.concat(candidate_parts, ignore_index=True)
    candidates = candidates.drop_duplicates(
        subset=[
            "max_hold_bars",
            "moving_average_length",
            "z_score_entry",
            "z_score_exit",
            "vwap_length",
            "high_quote_volume_multiplier",
            "high_volume_exit_z",
        ]
    )
    return candidates.sort_values(["sharpe", "calmar"], ascending=False)


def save_outputs(results_df):
    """Save the full results, leaderboards, candidates, and rejection files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sorted_results = results_df.sort_values("sharpe", ascending=False)
    sorted_results.to_csv(FULL_RESULTS_PATH, index=False)
    sorted_results.head(TOP_LEADERBOARD_COUNT).to_csv(TOP_SHARPE_PATH, index=False)
    sorted_results.sort_values("calmar", ascending=False).head(
        TOP_LEADERBOARD_COUNT
    ).to_csv(
        TOP_CALMAR_PATH,
        index=False,
    )
    sorted_results[~sorted_results["passes_trade_count_filter"]].to_csv(
        LOW_TRADE_COUNT_PATH,
        index=False,
    )

    candidates = build_candidate_parameter_sets(sorted_results)
    candidates.to_csv(CANDIDATES_PATH, index=False)

    return candidates


def print_leaderboards(results_df):
    """Print useful summary leaderboards for quick terminal inspection."""
    print("\nTop 10 by Sharpe")
    print(results_df.sort_values("sharpe", ascending=False).head(10).to_string(index=False))

    print("\nTop 10 by Calmar")
    print(results_df.sort_values("calmar", ascending=False).head(10).to_string(index=False))

    profitable = results_df[results_df["total_return"] > 0]
    print("\nTop 10 by trade count among profitable strategies")
    if profitable.empty:
        print("No profitable strategies found.")
    else:
        print(
            profitable.sort_values(["trade_count", "sharpe"], ascending=[False, False])
            .head(10)
            .to_string(index=False)
        )


def main():
    """Run the full multiprocessing grid search and save result files."""
    parameter_grid = generate_parameter_grid()
    total_combinations = len(parameter_grid)
    completed = 0
    results = []

    print("Starting grid search", flush=True)
    print(f"Total combinations: {total_combinations}", flush=True)
    print(f"CPU workers: {MAX_WORKERS}", flush=True)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker) as executor:
        for result in executor.map(
            run_parameter_combination,
            parameter_grid,
            chunksize=MAP_CHUNKSIZE,
        ):
            results.append(result)
            completed += 1

            if completed % PROGRESS_EVERY == 0 or completed == total_combinations:
                remaining = total_combinations - completed
                pct_complete = (completed / total_combinations) * 100
                print(
                    f"Completed {completed}/{total_combinations} | "
                    f"Remaining {remaining} | {pct_complete:.1f}%",
                    flush=True,
                )

    results_df = pd.DataFrame(results)
    candidates = save_outputs(results_df)

    passing_trade_count = results_df["passes_trade_count_filter"].sum()
    failing_trade_count = len(results_df) - passing_trade_count

    print("\nGrid Search Summary", flush=True)
    print("===================", flush=True)
    print(f"Total combinations tested: {len(results_df)}", flush=True)
    print(f"Combinations passing trade count filter: {passing_trade_count}", flush=True)
    print(f"Combinations failing trade count filter: {failing_trade_count}", flush=True)
    print(f"Candidate parameter sets saved: {len(candidates)}", flush=True)

    print_leaderboards(results_df)

    print("\nSaved files", flush=True)
    print(f"Full results: {FULL_RESULTS_PATH}", flush=True)
    print(f"Top {TOP_LEADERBOARD_COUNT} by Sharpe: {TOP_SHARPE_PATH}", flush=True)
    print(f"Top {TOP_LEADERBOARD_COUNT} by Calmar: {TOP_CALMAR_PATH}", flush=True)
    print(f"Candidate parameter sets: {CANDIDATES_PATH}", flush=True)
    print(f"Rejected low trade count: {LOW_TRADE_COUNT_PATH}", flush=True)


if __name__ == "__main__":
    main()
