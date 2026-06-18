import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
raw_data_dir = project_root / "data" / "raw"

BINANCE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]

def load_binance_klines(file_names):
    frames = []

    for file_name in file_names:
        file_path = raw_data_dir / file_name

        df = pd.read_csv(file_path)
        if "open_time" not in df.columns:
            df = pd.read_csv(file_path, header=None, names=BINANCE_COLUMNS)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    data["open_time"] = pd.to_datetime(
        data["open_time"],
        unit="ms",
        utc=True,
    )

    data["close_time"] = pd.to_datetime(
        data["close_time"],
        unit="ms",
        utc=True,
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]

    data[numeric_columns] = data[numeric_columns].astype(float)
    data = data.sort_values("open_time").reset_index(drop=True)

    return data
