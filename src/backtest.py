import pandas as pd


def run_backtest(signal_df, fee_rate=0.0004):
    data = signal_df.copy()

    data["returns"] = data["close"].pct_change().fillna(0.0)
    data["buy_hold_equity_curve"] = (1 + data["returns"]).cumprod()
    data["position_lagged"] = data["position"].shift(1).fillna(0.0)

    data["turnover"] = data["position_lagged"].diff().abs().fillna(
        data["position_lagged"].abs()
    )
    data["fees"] = data["turnover"] * fee_rate

    data["strategy_returns"] = (data["position_lagged"] * data["returns"]) - data[
        "fees"
    ]
    data["equity_curve"] = (1 + data["strategy_returns"]).cumprod()

    return data


def calculate_performance_summary(backtest_df, periods_per_year=365 * 24 * 12):
    columns = [
        "metric",
        "strategy",
        "buy_and_hold",
    ]

    if backtest_df.empty:
        return pd.DataFrame(columns=columns)

    strategy_returns = backtest_df["strategy_returns"]
    buy_hold_returns = backtest_df["returns"]

    strategy_equity = backtest_df["equity_curve"]
    buy_hold_equity = backtest_df["buy_hold_equity_curve"]

    rows = [
        {
            "metric": "sharpe",
            "strategy": _annualized_sharpe(strategy_returns, periods_per_year),
            "buy_and_hold": _annualized_sharpe(buy_hold_returns, periods_per_year),
        },
        {
            "metric": "max_drawdown",
            "strategy": _max_drawdown(strategy_equity),
            "buy_and_hold": _max_drawdown(buy_hold_equity),
        },
        {
            "metric": "total_return",
            "strategy": strategy_equity.iloc[-1] - 1,
            "buy_and_hold": buy_hold_equity.iloc[-1] - 1,
        },
    ]

    return pd.DataFrame(rows, columns=columns)


def _annualized_sharpe(returns, periods_per_year):
    std = returns.std()
    if std == 0 or pd.isna(std):
        return 0.0

    return (returns.mean() / std) * (periods_per_year ** 0.5)


def _max_drawdown(equity_curve):
    drawdown = equity_curve / equity_curve.cummax() - 1
    return drawdown.min()


def summarize_trades(backtest_df):
    columns = [
        "trade_id",
        "side",
        "status",
        "entry_time",
        "entry_price",
        "entry_position",
        "entry_z_score",
        "exit_time",
        "exit_price",
        "exit_z_score",
        "bars_held",
        "max_position",
        "gross_return",
        "fees_paid",
        "net_return",
    ]

    if backtest_df.empty:
        return pd.DataFrame(columns=columns)

    data = backtest_df.reset_index(drop=True)
    trades = []
    open_trade = None
    previous_position = 0.0
    positions = data["position_lagged"].to_numpy()

    for index, position in enumerate(positions):
        position = float(position)
        previous_side = _position_side(previous_position)
        current_side = _position_side(position)

        if previous_side == 0 and current_side != 0:
            open_trade = _start_trade(data, trades, index, position)

        elif previous_side != 0 and current_side == 0 and open_trade is not None:
            exit_index = max(index - 1, 0)
            exit_row = data.loc[exit_index]

            trades.append(
                _finish_trade(
                    data=data,
                    trade=open_trade,
                    status="closed",
                    exit_index=exit_index,
                    exit_return_index=index,
                    exit_time=exit_row["close_time"],
                    exit_price=exit_row["close"],
                    exit_z_score=exit_row["z_score"],
                )
            )
            open_trade = None

        elif (
            previous_side != 0
            and current_side != 0
            and previous_side != current_side
        ):
            exit_index = max(index - 1, 0)
            exit_row = data.loc[exit_index]

            if open_trade is not None:
                trades.append(
                    _finish_trade(
                        data=data,
                        trade=open_trade,
                        status="closed",
                        exit_index=exit_index,
                        exit_return_index=exit_index,
                        exit_time=exit_row["close_time"],
                        exit_price=exit_row["close"],
                        exit_z_score=exit_row["z_score"],
                    )
                )

            open_trade = _start_trade(data, trades, index, position)

        previous_position = position

    if open_trade is not None:
        trades.append(
            _finish_trade(
                data=data,
                trade=open_trade,
                status="open",
                exit_index=len(data) - 1,
                exit_return_index=len(data) - 1,
                exit_time=pd.NaT,
                exit_price=pd.NA,
                exit_z_score=pd.NA,
            )
        )

    return pd.DataFrame(trades, columns=columns)


def _position_side(position):
    if position > 0:
        return 1
    if position < 0:
        return -1
    return 0


def _start_trade(data, trades, index, position):
    entry_index = max(index - 1, 0)
    entry_row = data.loc[entry_index]

    return {
        "trade_id": len(trades) + 1,
        "side": "long" if position > 0 else "short",
        "direction": 1 if position > 0 else -1,
        "entry_index": entry_index,
        "entry_return_index": index,
        "entry_time": entry_row["close_time"],
        "entry_price": entry_row["close"],
        "entry_position": abs(position),
        "entry_z_score": entry_row["z_score"],
    }


def _finish_trade(
    data,
    trade,
    status,
    exit_index,
    exit_return_index,
    exit_time,
    exit_price,
    exit_z_score,
):
    if status == "open":
        mark_price = data.loc[exit_index, "close"]
    else:
        mark_price = exit_price

    gross_return = trade["direction"] * (
        (mark_price / trade["entry_price"]) - 1
    )

    trade_returns = data.loc[
        trade["entry_return_index"]:exit_return_index,
        "strategy_returns",
    ]
    net_return = (1 + trade_returns).prod() - 1

    max_position = data.loc[
        trade["entry_return_index"]:exit_return_index,
        "position_lagged",
    ].abs().max()

    fees_paid = data.loc[
        trade["entry_return_index"]:exit_return_index,
        "fees",
    ].sum()

    return {
        "trade_id": trade["trade_id"],
        "side": trade["side"],
        "status": status,
        "entry_time": trade["entry_time"],
        "entry_price": trade["entry_price"],
        "entry_position": trade["entry_position"],
        "entry_z_score": trade["entry_z_score"],
        "exit_time": exit_time,
        "exit_price": exit_price,
        "exit_z_score": exit_z_score,
        "bars_held": exit_index - trade["entry_index"],
        "max_position": max_position,
        "gross_return": gross_return,
        "fees_paid": fees_paid,
        "net_return": net_return,
    }
