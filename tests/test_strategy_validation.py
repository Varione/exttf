"""Tests for retrospective rolling strategy validation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from strategy_validation import evaluate_stability, period_metrics, rolling_windows


def _daily_returns(start: str, end: str, value: float) -> pd.DataFrame:
    dates = pd.bdate_range(start, end)
    return pd.DataFrame({"date": dates, "daily_return": value})


def test_period_metrics_positive_constant_return():
    result = period_metrics(_daily_returns("2020-01-01", "2020-12-31", 0.0002))
    assert result["cagr"] > 0
    assert result["max_drawdown"] == 0.0


def test_rolling_windows_are_time_ordered_and_complete():
    daily = _daily_returns("2018-01-01", "2024-12-31", 0.0001)
    windows = rolling_windows(daily, years=2)
    assert len(windows) == 6
    assert windows["start"].is_monotonic_increasing
    assert (windows["end"] <= daily["date"].max()).all()


def test_stable_windows_pass_gate():
    windows = pd.DataFrame(
        {
            "cagr": [0.05, 0.04, 0.06, 0.03],
            "sharpe": [0.7, 0.5, 0.8, 0.6],
            "max_drawdown": [-0.10, -0.12, -0.08, -0.15],
        }
    )
    assert evaluate_stability(windows)["stability_gate_passed"]


def test_unstable_windows_fail_gate():
    windows = pd.DataFrame(
        {
            "cagr": [0.10, -0.20, 0.02, -0.10],
            "sharpe": [0.8, -0.5, 0.2, -0.3],
            "max_drawdown": [-0.10, -0.35, -0.15, -0.30],
        }
    )
    result = evaluate_stability(windows)
    assert not result["stability_gate_passed"]
    assert not result["checks"]["worst_max_drawdown"]
