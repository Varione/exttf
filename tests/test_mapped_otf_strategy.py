"""Unit tests for robust mapped OTC portfolio controls."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mapped_otf_strategy import MappedETFSignalBuilder


def test_exposure_group_collapses_duplicate_gold_funds():
    first = pd.Series(
        {"underlying_name": "黄金", "etf_name": "黄金ETF华安"}, name="518880"
    )
    second = pd.Series(
        {"underlying_name": "黄金", "etf_name": "黄金ETF博时"}, name="159937"
    )
    assert MappedETFSignalBuilder._exposure_group(first) == "COMMODITY_GOLD"
    assert MappedETFSignalBuilder._exposure_group(second) == "COMMODITY_GOLD"


def test_portfolio_volatility_uses_covariance_and_annualizes():
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.01] * 40,
            "B": [0.005, -0.005] * 40,
        }
    )
    weights = pd.Series({"A": 0.5, "B": 0.5})
    result = MappedETFSignalBuilder._portfolio_volatility(returns, weights)
    combined = returns.mul(weights, axis=1).sum(axis=1)
    assert result == pytest.approx(combined.std() * np.sqrt(252))


def test_exposure_group_keeps_distinct_broad_indices_separate():
    csi300 = pd.Series({"underlying_name": "沪深300"}, name="510300")
    csi500 = pd.Series({"underlying_name": "中证500"}, name="510500")
    assert MappedETFSignalBuilder._exposure_group(csi300) != (
        MappedETFSignalBuilder._exposure_group(csi500)
    )
