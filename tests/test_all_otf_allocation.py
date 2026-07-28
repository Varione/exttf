from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from all_otf_allocation import add_defensive_residual


def test_defensive_residual_is_normalized_and_does_not_use_signal_day_nav():
    submit = pd.Timestamp("2024-04-02")
    signal = pd.Timestamp("2024-04-01")
    core = pd.DataFrame({"equity": [0.4]}, index=[submit])
    # Only two observations are strictly before the signal; the observation on
    # signal day must not make this fund pass a three-observation gate.
    dates = {"money": pd.DatetimeIndex(["2024-03-28", "2024-03-29", "2024-04-01"])}
    result, audit = add_defensive_residual(
        core, {submit: signal}, dates, {"money": 1.0}, minimum_history=3
    )
    assert "money" not in result.columns
    assert result.sum(axis=1).iloc[0] == pytest.approx(0.4)
    assert bool(audit.loc[0, "used_nav_strictly_before_signal"])


def test_defensive_residual_fills_to_one_when_eligible():
    submit = pd.Timestamp("2024-04-02")
    signal = pd.Timestamp("2024-04-01")
    core = pd.DataFrame({"equity": [0.4]}, index=[submit])
    dates = {
        "money": pd.DatetimeIndex(["2024-03-28", "2024-03-29"]),
        "bond": pd.DatetimeIndex(["2024-03-28", "2024-03-29"]),
    }
    result, _ = add_defensive_residual(
        core,
        {submit: signal},
        dates,
        {"money": 1.0, "bond": 3.0},
        minimum_history=2,
    )
    assert result.sum(axis=1).iloc[0] == pytest.approx(1.0)
    assert result.loc[submit, "money"] == pytest.approx(0.15)
    assert result.loc[submit, "bond"] == pytest.approx(0.45)
