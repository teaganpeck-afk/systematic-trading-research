from pathlib import Path

import pandas as pd

from src.data.load_data import load_binance_klines


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_PATH = RESULTS_DIR / "monthly_volume_report.txt"


def get_daily_files():
    return sorted(path.name for path in RAW_DATA_DIR.glob("BTCUSDT-1d-*.csv"))


def build_volume_report(daily_df):
    data = daily_df.copy()
    data["month"] = data["open_time"].dt.strftime("%Y-%m")
    data["date"] = data["open_time"].dt.strftime("%Y-%m-%d")

    monthly_average = (
        data.groupby("month", as_index=False)
        .agg(
            days=("date", "count"),
            average_daily_quote_volume=("quote_volume", "mean"),
        )
        .sort_values("month")
    )

    top_days = (
        data.sort_values(["month", "quote_volume"], ascending=[True, False])
        .groupby("month", as_index=False)
        .head(3)
        .copy()
    )
    top_days["rank"] = top_days.groupby("month")["quote_volume"].rank(
        method="first",
        ascending=False,
    )
    top_days = top_days.sort_values(["month", "rank"])

    return monthly_average, top_days


def format_number(value):
    return f"{value:,.2f}"


def format_report(monthly_average, top_days):
    lines = []
    lines.append("BTCUSDT Monthly Daily Quote Volume Report")
    lines.append("=" * 43)
    lines.append("")

    for _, month_row in monthly_average.iterrows():
        month = month_row["month"]
        average_quote_volume = format_number(
            month_row["average_daily_quote_volume"]
        )
        days = int(month_row["days"])

        lines.append(f"{month}")
        lines.append(
            f"  Average daily quote volume: {average_quote_volume} USDT "
            f"over {days} days"
        )
        lines.append("  Top 3 quote volume days:")

        month_top_days = top_days[top_days["month"] == month]
        for _, day_row in month_top_days.iterrows():
            rank = int(day_row["rank"])
            date = day_row["date"]
            quote_volume = format_number(day_row["quote_volume"])
            lines.append(f"    {rank}. {date} - {quote_volume} USDT")

        lines.append("")

    return "\n".join(lines)


def main():
    daily_files = get_daily_files()
    if not daily_files:
        raise FileNotFoundError(f"No daily BTCUSDT files found in {RAW_DATA_DIR}")

    daily_df = load_binance_klines(daily_files)
    monthly_average, top_days = build_volume_report(daily_df)
    report = format_report(monthly_average, top_days)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(report)
    print(f"Saved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
