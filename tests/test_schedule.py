"""Tests for month-end schedule builder per P0-A."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import pandas as pd
from otf_rotation.schedule import build_month_end_schedule, build_signal_submit_map


class TestBuildMonthEndSchedule:
    def test_one_signal_per_calendar_month(self):
        trading_dates = pd.bdate_range("2021-01-04", "2021-12-31")
        signals = build_month_end_schedule(trading_dates, "2021-01-01", "2021-12-31")
        months_seen = {s.month for s in signals}
        assert len(months_seen) == 12

    def test_no_pseudo_signal_at_start(self):
        trading_dates = pd.bdate_range("2021-01-04", "2021-12-31")
        signals = build_month_end_schedule(trading_dates, "2021-01-15", "2021-12-31")
        assert signals[0].month == 1
        assert signals[0] >= pd.Timestamp("2021-01-15")

    def test_last_trading_day_used(self):
        trading_dates = pd.bdate_range("2021-01-04", "2021-12-31")
        signals = build_month_end_schedule(trading_dates, "2021-01-01", "2021-01-31")
        assert len(signals) == 1
        assert signals[0] <= pd.Timestamp("2021-01-31")

    def test_holiday_gap_skipped(self):
        trading_dates = pd.bdate_range("2021-01-04", "2021-12-31")
        holiday = trading_dates[5]
        trimmed = trading_dates[trading_dates != holiday]
        signals_full = build_month_end_schedule(trading_dates, "2021-01-01", "2021-12-31")
        signals_trimmed = build_month_end_schedule(trimmed, "2021-01-01", "2021-12-31")
        assert len(signals_full) == len(signals_trimmed)

    def test_empty_range(self):
        trading_dates = pd.bdate_range("2021-01-04", "2021-01-31")
        signals = build_month_end_schedule(trading_dates, "2025-01-01", "2025-12-31")
        assert len(signals) == 0


class TestBuildSignalSubmitMap:
    def test_submit_is_next_trading_day(self):
        trading_dates = pd.bdate_range("2021-01-04", "2021-12-31")
        signals = build_month_end_schedule(trading_dates, "2021-01-01", "2021-03-31")
        mapping = build_signal_submit_map(trading_dates, signals, "2021-12-31")
        for submit_str, signal_str in mapping.items():
            submit = pd.Timestamp(submit_str)
            signal = pd.Timestamp(signal_str)
            assert submit > signal

    def test_no_future_leak(self):
        trading_dates = pd.bdate_range("2021-01-04", "2021-06-30")
        signals = build_month_end_schedule(trading_dates, "2021-01-01", "2021-03-31")
        mapping = build_signal_submit_map(trading_dates, signals, "2021-12-31")
        for submit_str in mapping:
            assert pd.Timestamp(submit_str) <= trading_dates[-1]

    def test_empty_mapping_at_end(self):
        trading_dates = pd.bdate_range("2021-01-04", "2021-01-31")
        signals = build_month_end_schedule(trading_dates, "2021-01-01", "2021-01-31")
        mapping = build_signal_submit_map(trading_dates, signals, "2021-01-31")
        assert len(mapping) == 0
