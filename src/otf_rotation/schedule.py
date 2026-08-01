"""Shared month-end schedule builder per planning.md §11 P1-1.

Rules:
- Each calendar month has at most one signal date (last trading day of that month).
- No pseudo month-end at arbitrary backtest start date.
- First signal is the first natural month-end after start date.
"""

from __future__ import annotations

import pandas as pd


def build_month_end_schedule(
    trading_dates: pd.DatetimeIndex,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> list[pd.Timestamp]:
    """Build month-end signal schedule.

    Each calendar month has exactly one signal date: the last trading day
    on or before that calendar month's end, if it falls within [start, end].

    Args:
        trading_dates: sorted DatetimeIndex of valid trading days.
        start: backtest start date (inclusive).
        end: backtest end date (inclusive).

    Returns:
        List of signal dates (last trading day of each month), sorted ascending.
    """
    t_start = pd.Timestamp(start)
    t_end = pd.Timestamp(end)

    # Generate calendar month ends between start and end
    months = pd.date_range(
        start=t_start,
        end=t_end,
        freq="ME",  # month end (calendar)
    )

    signal_dates: list[pd.Timestamp] = []

    for me in months:
        # Last trading day on or before this calendar month end
        valid = trading_dates[(trading_dates >= t_start) & (trading_dates <= me)]
        if len(valid) == 0:
            continue
        signal_date = valid[-1]

        # Avoid duplicates if two calendar months share the same last trading day
        if not signal_dates or signal_date != signal_dates[-1]:
            signal_dates.append(signal_date)

    return signal_dates


def build_quarter_end_schedule(
    trading_dates: pd.DatetimeIndex,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> list[pd.Timestamp]:
    """Build one signal date at the end of each calendar quarter.

    The signal is the last available trading day on or before the calendar
    quarter end.  This is deliberately separate from the month-end builder so
    a quarterly strategy cannot accidentally observe on an intermediate
    month-end.
    """
    t_start = pd.Timestamp(start)
    t_end = pd.Timestamp(end)
    trading_dates = pd.DatetimeIndex(pd.to_datetime(trading_dates)).sort_values()
    periods = pd.period_range(
        start=t_start.to_period("Q"),
        end=t_end.to_period("Q"),
        freq="Q",
    )
    signal_dates: list[pd.Timestamp] = []
    for period in periods:
        quarter_end = period.end_time.normalize()
        valid = trading_dates[
            (trading_dates >= t_start) & (trading_dates <= quarter_end)
        ]
        if len(valid) == 0:
            continue
        signal_date = pd.Timestamp(valid[-1])
        if not signal_dates or signal_date != signal_dates[-1]:
            signal_dates.append(signal_date)
    return signal_dates


def build_signal_submit_map(
    trading_dates: pd.DatetimeIndex,
    signal_dates: list[pd.Timestamp],
    end: str | pd.Timestamp,
) -> dict[str, str]:
    """Map each signal date to its submit date (next trading day).

    Returns:
        dict[submit_date_str -> signal_date_str]
        If multiple signals map to same submit date, keep the earliest signal.
    """
    t_end = pd.Timestamp(end)
    signal_map: dict[str, str] = {}

    for sd in signal_dates:
        candidates = trading_dates[trading_dates > sd]
        if len(candidates) == 0:
            continue
        submit_date = candidates[0]
        if submit_date > t_end:
            continue

        sd_str = sd.strftime("%Y-%m-%d")
        sub_str = submit_date.strftime("%Y-%m-%d")

        if sub_str in signal_map:
            # Multiple signals map to same submit date; keep earliest signal
            existing = pd.Timestamp(signal_map[sub_str])
            if sd < existing:
                signal_map[sub_str] = sd_str
        else:
            signal_map[sub_str] = sd_str

    return signal_map
