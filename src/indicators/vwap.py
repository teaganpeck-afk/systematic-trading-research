def add_daily_rolling_vwap(daily_df, window=4):
    df = daily_df.copy()

    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["pv"] = df["typical_price"] * df["volume"]

    rolling_pv = df["pv"].rolling(window=window, min_periods=window).sum()
    rolling_volume = df["volume"].rolling(window=window, min_periods=window).sum()

    df["vwap"] = rolling_pv / rolling_volume
    df["distance"] = df["close"] - df["vwap"]
    df["distance_std"] = df["distance"].rolling(
        window=window,
        min_periods=window,
    ).std()

    return df
