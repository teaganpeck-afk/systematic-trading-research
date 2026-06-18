import numpy as np
import pandas as pd

from src.indicators.vwap import add_daily_rolling_vwap


def prepare_signal_data(daily_df, five_min_df):
    daily = add_daily_rolling_vwap(daily_df)
    five_min = five_min_df.copy()

    daily["trend_ma"] = daily["close"].rolling(window=100, min_periods=100).mean()
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

    # A daily candle is complete only after that UTC day closes, so intraday
    # candles use the previous completed day's VWAP, MA, distance std, and 30-day
    # average quote volume.
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

    merged["z_score"] = (merged["close"] - merged["vwap"]) / merged[
        "distance_std"
    ]

    return merged.reset_index(drop=True)


def generate_signals(
    df,
    entry_z=1.5,
    exit_z=-0.0,
    high_quote_volume_multiplier=1.0,
    high_volume_exit_z=-0.0,
    max_hold_hours=None,
    max_hold_bars=1000,
):
    data = df.copy()
    positions = []
    position = 0
    entry_time = None
    entry_bar_index = None
    max_hold_delta = None
    high_volume_block_active = False
    high_volume_block_side = 0
    high_volume_event_released = False
    post_exit_wait_side = 0

    if max_hold_hours is not None:
        max_hold_delta = pd.Timedelta(hours=max_hold_hours)

    data["high_quote_volume_event"] = (
        data["quote_volume_24h"]
        > data["avg_quote_volume_30d"] * high_quote_volume_multiplier
    )
    data["above_trend_ma"] = data["close"] > data["trend_ma"]
    data["below_trend_ma"] = data["close"] < data["trend_ma"]

    z_scores = data["z_score"].to_numpy()
    open_times = data["open_time"].to_numpy()
    high_quote_volume_events = data["high_quote_volume_event"].to_numpy()
    long_allowed_values = data["above_trend_ma"].to_numpy()
    short_allowed_values = data["below_trend_ma"].to_numpy()

    for bar_index in range(len(data)):
        z_score = z_scores[bar_index]
        current_time = open_times[bar_index]
        high_quote_volume_event = high_quote_volume_events[bar_index]
        long_allowed = long_allowed_values[bar_index]
        short_allowed = short_allowed_values[bar_index]

        if np.isnan(z_score):
            positions.append(position)
            continue

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

            positions.append(position)
            continue

        if post_exit_wait_side != 0:
            position = 0

            if (
                (post_exit_wait_side > 0 and z_score >= -exit_z)
                or (post_exit_wait_side < 0 and z_score <= exit_z)
            ):
                post_exit_wait_side = 0

            positions.append(position)
            continue

        if (
            position != 0
            and max_hold_delta is not None
            and entry_time is not None
            and current_time - entry_time > max_hold_delta
        ):
            post_exit_wait_side = position
            position = 0
            entry_time = None
            entry_bar_index = None
            positions.append(position)
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
            positions.append(position)
            continue

        if position == 0:
            if long_allowed and z_score <= -entry_z:
                position = 1
                entry_time = current_time
                entry_bar_index = bar_index
            elif short_allowed and z_score >= entry_z:
                position = -1
                entry_time = current_time
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

        positions.append(position)

    data["position"] = positions
    return data
