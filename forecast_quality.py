"""Time-available forecast evidence; never score an outcome before its horizon ends."""
from __future__ import annotations
import pandas as pd


def matured_history(backtest: pd.DataFrame, as_of_date, horizon: int, window_days: int) -> pd.DataFrame:
    """Only use forecasts with outcomes observable by as_of_date.

    A prediction issued on t for h days only becomes scorable on t + h.
    The most recent h days must not influence historical model weights.
    """
    if horizon < 1 or window_days < 1:
        raise ValueError("horizon and window_days must be positive")
    if backtest.empty:
        return backtest
    as_of = pd.Timestamp(as_of_date)
    cutoff = as_of - pd.Timedelta(days=horizon)
    window_start = as_of - pd.Timedelta(days=window_days)
    return backtest.loc[(backtest.index <= cutoff) & (backtest.index >= window_start)]
