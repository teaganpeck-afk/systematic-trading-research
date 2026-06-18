from pathlib import Path

import numpy as np
import pandas as pd

from grid_search import calculate_grid_metrics, prepare_signal_data_for_grid
from main import (
    BACKTEST_END,
    BACKTEST_START,
    DAILY_FILES,
    DAILY_WARMUP_START,
    FIVE_MIN_FILES,
    MAX_HOLD_HOURS,
    filter_date_range,
)
from src.backtest import run_backtest, summarize_trades
from src.data.load_data import load_binance_klines
from src.strategy import generate_signals


# Validation controls. Edit VALIDATION_STAGE and the date window in main.py
# before running the next out-of-sample validation.
# Walk-forward validation script.
# This is not an optimizer. It only retests previously selected parameter
# sets on the current date window configured in main.py.
MIN_TRADE_COUNT = 30
MAX_ACCEPTABLE_DRAWDOWN = -0.60
VALIDATION_STAGE = 4

RESULTS_DIR = Path("results")
TOP_SHARPE_PATH = RESULTS_DIR / "top_40_by_sharpe.csv"
TOP_CALMAR_PATH = RESULTS_DIR / "top_40_by_calmar.csv"
PART1_FULL_OUTPUT_PATH = RESULTS_DIR / "walk_forward_validation.csv"
PART1_SURVIVORS_OUTPUT_PATH = RESULTS_DIR / "walk_forward_survivors.csv"
PART2_FULL_OUTPUT_PATH = RESULTS_DIR / "walk_forward_part2_validation.csv"
PART2_SURVIVORS_OUTPUT_PATH = RESULTS_DIR / "walk_forward_part2_survivors.csv"
PART3_FULL_OUTPUT_PATH = RESULTS_DIR / "walk_forward_part3_validation.csv"
PART3_SURVIVORS_OUTPUT_PATH = RESULTS_DIR / "walk_forward_part3_survivors.csv"
PART3_TRADE_BARS_OUTPUT_PATH = RESULTS_DIR / "walk_forward_part3_trade_bars.csv"
PART3_HOLD_BUCKET_OUTPUT_PATH = RESULTS_DIR / "walk_forward_part3_hold_bucket_summary.csv"
PART4_FULL_OUTPUT_PATH = RESULTS_DIR / "walk_forward_part4_validation.csv"
PART4_SURVIVORS_OUTPUT_PATH = RESULTS_DIR / "walk_forward_part4_survivors.csv"
PART5_FULL_OUTPUT_PATH = RESULTS_DIR / "walk_forward_part5_validation.csv"
PART5_SURVIVORS_OUTPUT_PATH = RESULTS_DIR / "walk_forward_part5_survivors.csv"
PART6_FULL_OUTPUT_PATH = RESULTS_DIR / "walk_forward_part6_validation.csv"
PART6_SURVIVORS_OUTPUT_PATH = RESULTS_DIR / "walk_forward_part6_survivors.csv"

PARAMETER_COLUMNS = [
    "max_hold_bars",
    "moving_average_length",
    "z_score_entry",
    "z_score_exit",
    "vwap_length",
    "high_quote_volume_multiplier",
    "high_volume_exit_z",
]

HOLD_BUCKETS = [
    (0, 49, "0-49"),
    (50, 99, "50-99"),
    (100, 249, "100-249"),
    (250, 499, "250-499"),
    (500, 749, "500-749"),
    (750, 999, "750-999"),
    (1000, None, "1000+"),
]


def normalize_parameter_columns(candidates):
    """Keep parameter columns clean after loading from CSV files."""
    if "high_quote_volume_multiplier" not in candidates.columns:
        candidates["high_quote_volume_multiplier"] = 2.0
    if "high_volume_exit_z" not in candidates.columns:
        candidates["high_volume_exit_z"] = candidates["z_score_exit"]

    candidates = candidates[PARAMETER_COLUMNS].drop_duplicates()
    candidates["max_hold_bars"] = candidates["max_hold_bars"].astype(int)
    candidates["moving_average_length"] = candidates["moving_average_length"].astype(int)
    candidates["vwap_length"] = candidates["vwap_length"].astype(int)
    candidates["high_quote_volume_multiplier"] = candidates[
        "high_quote_volume_multiplier"
    ].astype(float)
    candidates["high_volume_exit_z"] = candidates["high_volume_exit_z"].astype(float)

    return candidates.sort_values(PARAMETER_COLUMNS).reset_index(drop=True)


def load_part1_candidate_parameters():
    """Load top-40 Sharpe/Calmar rows and remove duplicate parameter sets."""
    top_sharpe = pd.read_csv(TOP_SHARPE_PATH)
    top_calmar = pd.read_csv(TOP_CALMAR_PATH)

    candidates = pd.concat([top_sharpe, top_calmar], ignore_index=True)
    return normalize_parameter_columns(candidates)


def load_part2_candidate_parameters():
    """Load the survivors from the prior walk-forward validation step."""
    candidates = pd.read_csv(PART1_SURVIVORS_OUTPUT_PATH)
    return normalize_parameter_columns(candidates)


def load_part3_candidate_parameters():
    """Load the survivors from the second walk-forward validation step."""
    candidates = pd.read_csv(PART2_SURVIVORS_OUTPUT_PATH)
    return normalize_parameter_columns(candidates)


def load_part4_candidate_parameters():
    """Load the survivors from the third walk-forward validation step."""
    candidates = pd.read_csv(PART3_SURVIVORS_OUTPUT_PATH)
    return normalize_parameter_columns(candidates)


def load_part5_candidate_parameters():
    """Load the survivors from the fourth walk-forward validation step."""
    candidates = pd.read_csv(PART4_SURVIVORS_OUTPUT_PATH)
    return normalize_parameter_columns(candidates)


def load_part6_candidate_parameters():
    """Load the survivors from the fifth walk-forward validation step."""
    candidates = pd.read_csv(PART5_SURVIVORS_OUTPUT_PATH)
    return normalize_parameter_columns(candidates)


def get_validation_config():
    """Select which walk-forward stage to run."""
    if VALIDATION_STAGE == 1:
        return {
            "stage_name": "Part 1",
            "candidate_loader": load_part1_candidate_parameters,
            "full_output_path": PART1_FULL_OUTPUT_PATH,
            "survivors_output_path": PART1_SURVIVORS_OUTPUT_PATH,
        }

    if VALIDATION_STAGE == 2:
        return {
            "stage_name": "Part 2",
            "candidate_loader": load_part2_candidate_parameters,
            "full_output_path": PART2_FULL_OUTPUT_PATH,
            "survivors_output_path": PART2_SURVIVORS_OUTPUT_PATH,
        }

    if VALIDATION_STAGE == 3:
        return {
            "stage_name": "Part 3",
            "candidate_loader": load_part3_candidate_parameters,
            "full_output_path": PART3_FULL_OUTPUT_PATH,
            "survivors_output_path": PART3_SURVIVORS_OUTPUT_PATH,
            "trade_bars_output_path": PART3_TRADE_BARS_OUTPUT_PATH,
            "hold_bucket_output_path": PART3_HOLD_BUCKET_OUTPUT_PATH,
        }

    if VALIDATION_STAGE == 4:
        return {
            "stage_name": "Part 4",
            "candidate_loader": load_part4_candidate_parameters,
            "full_output_path": PART4_FULL_OUTPUT_PATH,
            "survivors_output_path": PART4_SURVIVORS_OUTPUT_PATH,
        }

    if VALIDATION_STAGE == 5:
        return {
            "stage_name": "Part 5",
            "candidate_loader": load_part5_candidate_parameters,
            "full_output_path": PART5_FULL_OUTPUT_PATH,
            "survivors_output_path": PART5_SURVIVORS_OUTPUT_PATH,
        }

    if VALIDATION_STAGE == 6:
        return {
            "stage_name": "Part 6",
            "candidate_loader": load_part6_candidate_parameters,
            "full_output_path": PART6_FULL_OUTPUT_PATH,
            "survivors_output_path": PART6_SURVIVORS_OUTPUT_PATH,
        }

    raise ValueError("VALIDATION_STAGE must be 1, 2, 3, 4, 5, or 6")


def load_market_data():
    """Load and filter market data using the same date settings as main.py."""
    daily_df = load_binance_klines(DAILY_FILES)
    five_min_df = load_binance_klines(FIVE_MIN_FILES)

    daily_df = filter_date_range(daily_df, DAILY_WARMUP_START, BACKTEST_END)
    five_min_df = filter_date_range(five_min_df, BACKTEST_START, BACKTEST_END)

    return daily_df, five_min_df


def run_parameter_set_on_window(daily_df, five_min_df, parameter_row):
    """Run one selected parameter set on the current main.py backtest window."""
    signal_df = prepare_signal_data_for_grid(
        daily_df,
        five_min_df,
        vwap_length=int(parameter_row["vwap_length"]),
        moving_average_length=int(parameter_row["moving_average_length"]),
    )

    signal_df = generate_signals(
        signal_df,
        entry_z=float(parameter_row["z_score_entry"]),
        exit_z=float(parameter_row["z_score_exit"]),
        high_quote_volume_multiplier=float(
            parameter_row["high_quote_volume_multiplier"]
        ),
        high_volume_exit_z=float(parameter_row["high_volume_exit_z"]),
        max_hold_hours=MAX_HOLD_HOURS,
        max_hold_bars=int(parameter_row["max_hold_bars"]),
    )

    backtest_df = run_backtest(signal_df)
    trade_summary_df = summarize_trades(backtest_df)

    return calculate_grid_metrics(backtest_df, trade_summary_df), trade_summary_df


def validate_parameter_set(daily_df, five_min_df, parameter_row):
    """Run one candidate and add pass/fail flags for this validation window."""
    metrics, trade_summary_df = run_parameter_set_on_window(
        daily_df,
        five_min_df,
        parameter_row,
    )

    result = {column: parameter_row[column] for column in PARAMETER_COLUMNS}
    result.update(metrics)
    result["passes_trade_count_filter"] = result["trade_count"] >= MIN_TRADE_COUNT
    result["passed_walk_forward_test"] = (
        result["sharpe"] > 0
        and result["total_return"] > 0
        and result["passes_trade_count_filter"]
        and result["max_drawdown"] > MAX_ACCEPTABLE_DRAWDOWN
    )

    return result, trade_summary_df


def get_hold_bucket(bars_held):
    """Assign a trade to a holding-time bucket."""
    for lower_bound, upper_bound, label in HOLD_BUCKETS:
        if upper_bound is None and bars_held >= lower_bound:
            return label
        if upper_bound is not None and lower_bound <= bars_held <= upper_bound:
            return label

    return np.nan


def add_parameter_columns(df, parameter_row):
    """Attach parameter values to trade-level or bucket-level output rows."""
    output = df.copy()
    for column in reversed(PARAMETER_COLUMNS):
        output.insert(0, column, parameter_row[column])

    return output


def build_trade_bars_output(parameter_row, trade_summary_df):
    """Create trade-level hold-time rows for Stage 3 investigation."""
    if trade_summary_df.empty:
        return pd.DataFrame(columns=PARAMETER_COLUMNS)

    trade_details = trade_summary_df.copy()
    trade_details["hold_bucket"] = trade_details["bars_held"].apply(get_hold_bucket)

    columns = [
        "trade_id",
        "side",
        "status",
        "entry_time",
        "exit_time",
        "bars_held",
        "hold_bucket",
        "gross_return",
        "fees_paid",
        "net_return",
    ]

    return add_parameter_columns(trade_details[columns], parameter_row)


def calculate_profit_factor(net_returns):
    """Calculate profit factor for a group of trade returns."""
    wins = net_returns[net_returns > 0].sum()
    losses = net_returns[net_returns < 0].sum()

    if losses == 0:
        return np.inf if wins > 0 else np.nan

    return wins / abs(losses)


def build_hold_bucket_output(parameter_row, trade_summary_df):
    """Summarize trade counts and profitability by hold-time bucket."""
    rows = []

    for lower_bound, upper_bound, label in HOLD_BUCKETS:
        if trade_summary_df.empty:
            bucket_trades = trade_summary_df
        elif upper_bound is None:
            bucket_trades = trade_summary_df[trade_summary_df["bars_held"] >= lower_bound]
        else:
            bucket_trades = trade_summary_df[
                (trade_summary_df["bars_held"] >= lower_bound)
                & (trade_summary_df["bars_held"] <= upper_bound)
            ]

        if bucket_trades.empty:
            rows.append(
                {
                    "hold_bucket": label,
                    "trade_count": 0,
                    "total_net_return": 0.0,
                    "avg_net_return": np.nan,
                    "win_rate": np.nan,
                    "profit_factor": np.nan,
                    "avg_bars_held": np.nan,
                    "total_gross_return": 0.0,
                }
            )
            continue

        net_returns = bucket_trades["net_return"]
        rows.append(
            {
                "hold_bucket": label,
                "trade_count": len(bucket_trades),
                "total_net_return": net_returns.sum(),
                "avg_net_return": net_returns.mean(),
                "win_rate": (net_returns > 0).mean(),
                "profit_factor": calculate_profit_factor(net_returns),
                "avg_bars_held": bucket_trades["bars_held"].mean(),
                "total_gross_return": bucket_trades["gross_return"].sum(),
            }
        )

    return add_parameter_columns(pd.DataFrame(rows), parameter_row)


def save_outputs(results_df, full_output_path, survivors_output_path):
    """Save all validation rows and the survivor subset."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if results_df.empty:
        results_df.to_csv(full_output_path, index=False)
        results_df.to_csv(survivors_output_path, index=False)
        return results_df, results_df.copy()

    sorted_results = results_df.sort_values(
        ["passed_walk_forward_test", "sharpe", "calmar"],
        ascending=[False, False, False],
    )
    survivors = sorted_results[sorted_results["passed_walk_forward_test"]].copy()

    sorted_results.to_csv(full_output_path, index=False)
    survivors.to_csv(survivors_output_path, index=False)

    return sorted_results, survivors


def save_stage3_investigation_outputs(
    trade_bars_rows,
    hold_bucket_rows,
    trade_bars_output_path,
    hold_bucket_output_path,
):
    """Save Stage 3 trade hold-time detail and bucket summaries."""
    trade_bars_df = pd.concat(trade_bars_rows, ignore_index=True)
    hold_bucket_df = pd.concat(hold_bucket_rows, ignore_index=True)

    trade_bars_df.to_csv(trade_bars_output_path, index=False)
    hold_bucket_df.to_csv(hold_bucket_output_path, index=False)

    return trade_bars_df, hold_bucket_df


def print_top_10(title, df, sort_columns, ascending):
    """Print a compact top-10 leaderboard for terminal inspection."""
    print(f"\n{title}")
    if df.empty:
        print("No rows to display.")
        return

    display_columns = PARAMETER_COLUMNS + [
        "sharpe",
        "calmar",
        "total_return",
        "max_drawdown",
        "trade_count",
        "win_rate",
        "profit_factor",
    ]
    print(
        df.sort_values(sort_columns, ascending=ascending)
        .head(10)[display_columns]
        .to_string(index=False)
    )


def main():
    """Validate prior top parameter sets on the current main.py date window."""
    config = get_validation_config()
    candidates = config["candidate_loader"]()
    daily_df, five_min_df = load_market_data()

    total_candidates = len(candidates)
    results = []
    trade_bars_rows = []
    hold_bucket_rows = []
    collect_stage3_investigation = VALIDATION_STAGE == 3

    print(f"Starting walk-forward validation {config['stage_name']}")
    print("This validates prior candidates; it does not run a new optimization.")
    print(f"Window start from main.py: {BACKTEST_START}")
    print(f"Window end from main.py: {BACKTEST_END}")
    print(f"Unique candidate parameter sets: {total_candidates}")

    for current, parameter_row in enumerate(candidates.to_dict("records"), start=1):
        params_for_print = {column: parameter_row[column] for column in PARAMETER_COLUMNS}
        print(f"Candidate {current}/{total_candidates}: {params_for_print}")

        try:
            result, trade_summary_df = validate_parameter_set(
                daily_df,
                five_min_df,
                parameter_row,
            )
            results.append(result)

            if collect_stage3_investigation:
                trade_bars_rows.append(
                    build_trade_bars_output(parameter_row, trade_summary_df)
                )
                hold_bucket_rows.append(
                    build_hold_bucket_output(parameter_row, trade_summary_df)
                )
        except Exception as error:
            failed_result = {column: parameter_row[column] for column in PARAMETER_COLUMNS}
            failed_result.update(
                {
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
                    "passed_walk_forward_test": False,
                    "error": str(error),
                }
            )
            results.append(failed_result)

            if collect_stage3_investigation:
                empty_trades = pd.DataFrame()
                trade_bars_rows.append(build_trade_bars_output(parameter_row, empty_trades))
                hold_bucket_rows.append(build_hold_bucket_output(parameter_row, empty_trades))

    results_df = pd.DataFrame(results)
    sorted_results, survivors = save_outputs(
        results_df,
        config["full_output_path"],
        config["survivors_output_path"],
    )

    if collect_stage3_investigation:
        save_stage3_investigation_outputs(
            trade_bars_rows,
            hold_bucket_rows,
            config["trade_bars_output_path"],
            config["hold_bucket_output_path"],
        )

    passed_trade_filter = int(sorted_results["passes_trade_count_filter"].sum())
    passed_validation = int(sorted_results["passed_walk_forward_test"].sum())

    print(f"\nWalk-Forward Validation {config['stage_name']} Summary")
    print("======================================")
    print(f"Unique candidate parameter sets tested: {len(sorted_results)}")
    print(f"Passed trade count filter: {passed_trade_filter}")
    print(f"Passed walk-forward test: {passed_validation}")

    print_top_10("Top 10 by Sharpe", sorted_results, ["sharpe"], [False])
    print_top_10("Top 10 by Calmar", sorted_results, ["calmar"], [False])
    print_top_10(
        "Top 10 by Trade Count Among Profitable Strategies",
        sorted_results[sorted_results["total_return"] > 0],
        ["trade_count", "sharpe"],
        [False, False],
    )

    print("\nSaved files")
    print(f"Full validation: {config['full_output_path']}")
    print(f"Walk-forward survivors: {config['survivors_output_path']}")
    if collect_stage3_investigation:
        print(f"Stage 3 trade bars: {config['trade_bars_output_path']}")
        print(f"Stage 3 hold buckets: {config['hold_bucket_output_path']}")


if __name__ == "__main__":
    main()
