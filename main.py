from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.backtest import calculate_performance_summary, run_backtest, summarize_trades
from src.data.load_data import load_binance_klines
from src.strategy import generate_signals, prepare_signal_data


# Data files loaded for the active research run.
DAILY_FILES = [
    "BTCUSDT-1d-2020-01.csv",
    "BTCUSDT-1d-2020-02.csv",
    "BTCUSDT-1d-2020-03.csv",
    "BTCUSDT-1d-2020-04.csv",
    "BTCUSDT-1d-2020-05.csv",
    "BTCUSDT-1d-2020-06.csv",
    "BTCUSDT-1d-2020-07.csv",
    "BTCUSDT-1d-2020-08.csv",
    "BTCUSDT-1d-2020-09.csv",
    "BTCUSDT-1d-2020-10.csv",
    "BTCUSDT-1d-2020-11.csv",
    "BTCUSDT-1d-2020-12.csv",
    "BTCUSDT-1d-2021-01.csv",
    "BTCUSDT-1d-2021-02.csv",
    "BTCUSDT-1d-2021-03.csv",
    "BTCUSDT-1d-2021-04.csv",
    "BTCUSDT-1d-2021-05.csv",
    "BTCUSDT-1d-2021-06.csv",
    "BTCUSDT-1d-2021-07.csv",
    "BTCUSDT-1d-2021-08.csv",
    "BTCUSDT-1d-2021-09.csv",
    "BTCUSDT-1d-2021-10.csv",
    "BTCUSDT-1d-2021-11.csv",
    "BTCUSDT-1d-2021-12.csv",
    "BTCUSDT-1d-2022-01.csv",
    "BTCUSDT-1d-2022-02.csv",
    "BTCUSDT-1d-2022-03.csv",
    "BTCUSDT-1d-2022-04.csv",
    "BTCUSDT-1d-2022-05.csv",
    "BTCUSDT-1d-2022-06.csv",
    "BTCUSDT-1d-2022-07.csv",
    "BTCUSDT-1d-2022-08.csv",
    "BTCUSDT-1d-2022-09.csv",
    "BTCUSDT-1d-2022-10.csv",
    "BTCUSDT-1d-2022-11.csv",
    "BTCUSDT-1d-2022-12.csv",
    "BTCUSDT-1d-2023-01.csv",
    "BTCUSDT-1d-2023-02.csv",
    "BTCUSDT-1d-2023-03.csv",
    "BTCUSDT-1d-2023-04.csv",
    "BTCUSDT-1d-2023-05.csv",
    "BTCUSDT-1d-2023-06.csv",
    "BTCUSDT-1d-2023-07.csv",
    "BTCUSDT-1d-2023-08.csv",
    "BTCUSDT-1d-2023-09.csv",
    "BTCUSDT-1d-2023-10.csv",
    "BTCUSDT-1d-2023-11.csv",
    "BTCUSDT-1d-2023-12.csv",
    "BTCUSDT-1d-2024-01.csv",
    "BTCUSDT-1d-2024-02.csv",
    "BTCUSDT-1d-2024-03.csv",
    "BTCUSDT-1d-2024-04.csv",
    "BTCUSDT-1d-2024-05.csv",
    "BTCUSDT-1d-2024-06.csv",
    "BTCUSDT-1d-2024-07.csv",
    "BTCUSDT-1d-2024-08.csv",
    "BTCUSDT-1d-2024-09.csv",
    "BTCUSDT-1d-2024-10.csv",
    "BTCUSDT-1d-2024-11.csv",
    "BTCUSDT-1d-2024-12.csv",
    "BTCUSDT-1d-2025-01.csv",
    "BTCUSDT-1d-2025-02.csv",
    "BTCUSDT-1d-2025-03.csv",
    "BTCUSDT-1d-2025-04.csv",
    "BTCUSDT-1d-2025-05.csv",
    "BTCUSDT-1d-2025-06.csv",
    "BTCUSDT-1d-2025-07.csv",
    "BTCUSDT-1d-2025-08.csv",
    "BTCUSDT-1d-2025-09.csv",
    "BTCUSDT-1d-2025-10.csv",
    "BTCUSDT-1d-2025-11.csv",
    "BTCUSDT-1d-2025-12.csv",
    "BTCUSDT-1d-2026-01.csv",
    "BTCUSDT-1d-2026-02.csv",
    "BTCUSDT-1d-2026-03.csv",
    "BTCUSDT-1d-2026-04.csv",
    "BTCUSDT-1d-2026-05.csv",
]

FIVE_MIN_FILES = [
    "BTCUSDT-5m-2020-01.csv",
    "BTCUSDT-5m-2020-02.csv",
    "BTCUSDT-5m-2020-03.csv",
    "BTCUSDT-5m-2020-04.csv",
    "BTCUSDT-5m-2020-05.csv",
    "BTCUSDT-5m-2020-06.csv",
    "BTCUSDT-5m-2020-07.csv",
    "BTCUSDT-5m-2020-08.csv",
    "BTCUSDT-5m-2020-09.csv",
    "BTCUSDT-5m-2020-10.csv",
    "BTCUSDT-5m-2020-11.csv",
    "BTCUSDT-5m-2020-12.csv",
    "BTCUSDT-5m-2021-01.csv",
    "BTCUSDT-5m-2021-02.csv",
    "BTCUSDT-5m-2021-03.csv",
    "BTCUSDT-5m-2021-04.csv",
    "BTCUSDT-5m-2021-05.csv",
    "BTCUSDT-5m-2021-06.csv",
    "BTCUSDT-5m-2021-07.csv",
    "BTCUSDT-5m-2021-08.csv",
    "BTCUSDT-5m-2021-09.csv",
    "BTCUSDT-5m-2021-10.csv",
    "BTCUSDT-5m-2021-11.csv",
    "BTCUSDT-5m-2021-12.csv",
    "BTCUSDT-5m-2022-01.csv",
    "BTCUSDT-5m-2022-02.csv",
    "BTCUSDT-5m-2022-03.csv",
    "BTCUSDT-5m-2022-04.csv",
    "BTCUSDT-5m-2022-05.csv",
    "BTCUSDT-5m-2022-06.csv",
    "BTCUSDT-5m-2022-07.csv",
    "BTCUSDT-5m-2022-08.csv",
    "BTCUSDT-5m-2022-09.csv",
    "BTCUSDT-5m-2022-10.csv",
    "BTCUSDT-5m-2022-11.csv",
    "BTCUSDT-5m-2022-12.csv",
    "BTCUSDT-5m-2023-01.csv",
    "BTCUSDT-5m-2023-02.csv",
    "BTCUSDT-5m-2023-03.csv",
    "BTCUSDT-5m-2023-04.csv",
    "BTCUSDT-5m-2023-05.csv",
    "BTCUSDT-5m-2023-06.csv",
    "BTCUSDT-5m-2023-07.csv",
    "BTCUSDT-5m-2023-08.csv",
    "BTCUSDT-5m-2023-09.csv",
    "BTCUSDT-5m-2023-10.csv",
    "BTCUSDT-5m-2023-11.csv",
    "BTCUSDT-5m-2023-12.csv",
    "BTCUSDT-5m-2024-01.csv",
    "BTCUSDT-5m-2024-02.csv",
    "BTCUSDT-5m-2024-03.csv",
    "BTCUSDT-5m-2024-04.csv",
    "BTCUSDT-5m-2024-05.csv",
    "BTCUSDT-5m-2024-06.csv",
    "BTCUSDT-5m-2024-07.csv",
    "BTCUSDT-5m-2024-08.csv",
    "BTCUSDT-5m-2024-09.csv",
    "BTCUSDT-5m-2024-10.csv",
    "BTCUSDT-5m-2024-11.csv",
    "BTCUSDT-5m-2024-12.csv",
    "BTCUSDT-5m-2025-01.csv",
    "BTCUSDT-5m-2025-02.csv",
    "BTCUSDT-5m-2025-03.csv",
    "BTCUSDT-5m-2025-04.csv",
    "BTCUSDT-5m-2025-05.csv",
    "BTCUSDT-5m-2025-06.csv",
    "BTCUSDT-5m-2025-07.csv",
    "BTCUSDT-5m-2025-08.csv",
    "BTCUSDT-5m-2025-09.csv",
    "BTCUSDT-5m-2025-10.csv",
    "BTCUSDT-5m-2025-11.csv",
    "BTCUSDT-5m-2025-12.csv",
    "BTCUSDT-5m-2026-01.csv",
    "BTCUSDT-5m-2026-02.csv",
    "BTCUSDT-5m-2026-03.csv",
    "BTCUSDT-5m-2026-04.csv",
    "BTCUSDT-5m-2026-05.csv",
]

# Backtest window and manual strategy parameters.
BACKTEST_START = "2025-08-01"
BACKTEST_END = "2026-05-31"  # Exclusive.
DAILY_WARMUP_START = "2020-01-01"
MAX_HOLD_HOURS = None
MAX_HOLD_BARS = 1000
HIGH_QUOTE_VOLUME_MULTIPLIER = 1.0
HIGH_VOLUME_EXIT_Z = -0.0
RESULTS_DIR = Path("results")


def filter_date_range(df, start, end):
    start_time = pd.Timestamp(start, tz="UTC")
    end_time = pd.Timestamp(end, tz="UTC")

    return df[
        (df["open_time"] >= start_time)
        & (df["open_time"] < end_time)
    ].copy()


def count_trades(backtest_df):
    if backtest_df.empty:
        return 0

    entries = (
        (backtest_df["position"].shift(1).fillna(0) == 0)
        & (backtest_df["position"] != 0)
    )
    return int(entries.sum())


def save_equity_curve_chart(backtest_df, chart_path):
    plt.figure(figsize=(12, 6))

    if backtest_df.empty:
        plt.title("VWAP Strategy Equity Curve")
        plt.text(0.5, 0.5, "No backtest rows for this date range", ha="center")
        plt.axis("off")
    else:
        plt.plot(backtest_df["open_time"], backtest_df["equity_curve"])
        plt.title("VWAP Strategy Equity Curve")
        plt.xlabel("Time")
        plt.ylabel("Equity")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    plt.savefig(chart_path)
    plt.close()


def save_trade_summary_chart(trade_summary_df, chart_path):
    long_trades = trade_summary_df[trade_summary_df["side"] == "long"].copy()
    short_trades = trade_summary_df[trade_summary_df["side"] == "short"].copy()

    long_rows = min(len(long_trades), 30) + 2
    short_rows = min(len(short_trades), 30) + 2
    figure_height = max(6, 0.38 * (long_rows + short_rows) + 2)

    fig, axes = plt.subplots(2, 1, figsize=(18, figure_height))
    fig.suptitle("Trade Summary by Side", fontsize=14, fontweight="bold")

    _draw_trade_table(axes[0], long_trades, "Long Trades")
    _draw_trade_table(axes[1], short_trades, "Short Trades")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()


def _draw_trade_table(axis, trades, title):
    axis.axis("off")

    if trades.empty:
        axis.set_title(title, fontweight="bold")
        axis.text(0.5, 0.5, "No trades", ha="center", va="center")
        return

    display_df, note = _format_trade_table(trades)
    chart_columns = [
        "trade_id",
        "status",
        "entry_time",
        "entry_price",
        "entry_z_score",
        "exit_time",
        "exit_price",
        "exit_z_score",
        "bars_held",
        "entry_position",
        "max_position",
        "gross_return",
        "fees_paid",
        "net_return",
    ]

    table = axis.table(
        cellText=display_df[chart_columns].values,
        colLabels=chart_columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.25)

    total_row = len(display_df)
    for column in range(len(chart_columns)):
        table[(total_row, column)].set_text_props(fontweight="bold")

    axis_title = f"{title} ({len(trades)} trades)"
    if note:
        axis_title = f"{axis_title}\n{note}"
    axis.set_title(axis_title, fontweight="bold")


def _format_trade_table(trades):
    max_rows = 30
    note = ""

    if len(trades) > max_rows:
        chart_trades = pd.concat([trades.head(15), trades.tail(15)])
        note = f"Showing first 15 and last 15 of {len(trades)} trades"
    else:
        chart_trades = trades.copy()

    display_df = pd.DataFrame(
        {
            "trade_id": chart_trades["trade_id"].astype(str),
            "status": chart_trades["status"].astype(str),
            "entry_time": chart_trades["entry_time"].astype(str),
            "entry_price": chart_trades["entry_price"].map("{:.2f}".format),
            "entry_z_score": chart_trades["entry_z_score"].map("{:.2f}".format),
            "exit_time": "",
            "exit_price": "",
            "exit_z_score": "",
            "bars_held": chart_trades["bars_held"].astype(int).astype(str),
            "entry_position": chart_trades["entry_position"].map("{:.1f}".format),
            "max_position": chart_trades["max_position"].map("{:.1f}".format),
            "gross_return": chart_trades["gross_return"].map("{:.2%}".format),
            "fees_paid": chart_trades["fees_paid"].map("{:.2%}".format),
            "net_return": chart_trades["net_return"].map("{:.2%}".format),
        }
    )

    closed_trades = chart_trades["status"] == "closed"
    display_df.loc[closed_trades, "exit_time"] = chart_trades.loc[
        closed_trades,
        "exit_time",
    ].astype(str)
    display_df.loc[closed_trades, "exit_price"] = chart_trades.loc[
        closed_trades,
        "exit_price",
    ].map("{:.2f}".format)
    display_df.loc[closed_trades, "exit_z_score"] = chart_trades.loc[
        closed_trades,
        "exit_z_score",
    ].map("{:.2f}".format)

    total_row = pd.DataFrame([_trade_total_row(trades)])
    display_df = pd.concat([display_df, total_row], ignore_index=True)

    return display_df, note


def _trade_total_row(trades):
    return {
        "trade_id": "TOTAL",
        "status": f"{len(trades)} trades",
        "entry_time": "",
        "entry_price": "",
        "entry_z_score": "",
        "exit_time": "",
        "exit_price": "",
        "exit_z_score": "",
        "bars_held": f"{int(trades['bars_held'].sum())}",
        "entry_position": f"{trades['entry_position'].sum():.1f}",
        "max_position": f"{trades['max_position'].sum():.1f}",
        "gross_return": f"{trades['gross_return'].sum():.2%}",
        "fees_paid": f"{trades['fees_paid'].sum():.2%}",
        "net_return": f"{trades['net_return'].sum():.2%}",
    }


def format_performance_summary(performance_summary_df):
    if performance_summary_df.empty:
        return "Performance summary: n/a"

    lines = ["Performance Summary", "==================="]

    for _, row in performance_summary_df.iterrows():
        metric = row["metric"]
        strategy_value = row["strategy"]
        buy_hold_value = row["buy_and_hold"]

        if metric == "sharpe":
            lines.append(
                f"Sharpe: strategy {strategy_value:.2f}, "
                f"buy and hold {buy_hold_value:.2f}"
            )
        elif metric in {"max_drawdown", "total_return"}:
            label = metric.replace("_", " ").title()
            lines.append(
                f"{label}: strategy {strategy_value:.2%}, "
                f"buy and hold {buy_hold_value:.2%}"
            )

    return "\n".join(lines)


def main():
    daily_df = load_binance_klines(DAILY_FILES)
    five_min_df = load_binance_klines(FIVE_MIN_FILES)

    daily_df = filter_date_range(daily_df, DAILY_WARMUP_START, BACKTEST_END)
    five_min_df = filter_date_range(five_min_df, BACKTEST_START, BACKTEST_END)

    signal_df = prepare_signal_data(daily_df, five_min_df)
    signal_df = generate_signals(
        signal_df,
        high_quote_volume_multiplier=HIGH_QUOTE_VOLUME_MULTIPLIER,
        high_volume_exit_z=HIGH_VOLUME_EXIT_Z,
        max_hold_hours=MAX_HOLD_HOURS,
        max_hold_bars=MAX_HOLD_BARS,
    )
    backtest_df = run_backtest(signal_df)
    trade_summary_df = summarize_trades(backtest_df)
    performance_summary_df = calculate_performance_summary(backtest_df)

    trade_count = len(trade_summary_df)

    print("Backtest period start: ")
    print(BACKTEST_START)
    print("Backtest period end: ")
    print(BACKTEST_END)
    print("Max hold hours: ")
    print(MAX_HOLD_HOURS)
    print("Max hold bars: ")
    print(MAX_HOLD_BARS)
    print("High quote volume multiplier: ")
    print(HIGH_QUOTE_VOLUME_MULTIPLIER)
    print("High volume exit z: ")
    print(HIGH_VOLUME_EXIT_Z)
    print(f"Rows: {len(backtest_df)}")
    print(f"Trades: {trade_count}")

    if backtest_df.empty:
        print("Final equity: n/a")
        print("Total return: n/a")
        print("No trades can be generated in this window with a 30-day VWAP warmup.")
    else:
        final_equity = backtest_df["equity_curve"].iloc[-1]
        total_return = final_equity - 1

        print(f"Final equity: {final_equity:.4f}")
        print(f"Total return: {total_return:.2%}")
        print("")
        print(format_performance_summary(performance_summary_df))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    chart_path = RESULTS_DIR / "equity_curve.png"
    trade_summary_path = RESULTS_DIR / "trade_summary.csv"
    trade_summary_chart_path = RESULTS_DIR / "trade_summary.png"
    performance_summary_path = RESULTS_DIR / "performance_summary.csv"
    performance_summary_text_path = RESULTS_DIR / "performance_summary.txt"

    trade_summary_df.to_csv(trade_summary_path, index=False)
    performance_summary_df.to_csv(performance_summary_path, index=False)
    performance_summary_text_path.write_text(
        format_performance_summary(performance_summary_df),
        encoding="utf-8",
    )
    save_equity_curve_chart(backtest_df, chart_path)
    save_trade_summary_chart(trade_summary_df, trade_summary_chart_path)

    print(f"Saved equity chart to: {chart_path}")
    print(f"Saved trade summary to: {trade_summary_path}")
    print(f"Saved trade summary chart to: {trade_summary_chart_path}")
    print(f"Saved performance summary to: {performance_summary_path}")
    print(f"Saved performance summary text to: {performance_summary_text_path}")


if __name__ == "__main__":
    main()
