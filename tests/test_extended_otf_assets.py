from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fetch_extended_otf_assets import money_rows_to_nav


def test_money_income_compounds_weekend_before_sampling():
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"]),
            "per_10000_income": [1.0, 1.0, 1.0, 1.0],
        }
    )
    result = money_rows_to_nav(raw, pd.DatetimeIndex(["2024-01-05", "2024-01-08"]))
    expected = (1.0001 ** 3) - 1.0
    assert result.iloc[1]["daily_growth_pct"] / 100.0 == pytest.approx(expected)


def test_money_income_rejects_impossible_loss():
    raw = pd.DataFrame({"date": ["2024-01-01"], "per_10000_income": [-10000]})
    with pytest.raises(ValueError, match="INVALID_MONEY_FUND_INCOME"):
        money_rows_to_nav(raw, pd.DatetimeIndex(["2024-01-01"]))


def test_configuration_and_catalog_data_models_are_explicitly_distinct():
    config = pd.DataFrame(
        {"fund_code": ["000001"], "fund_name": ["示例"], "data_model": ["money_yield"]}
    )
    catalog = pd.DataFrame(
        {"fund_code": ["000001"], "fund_name": ["示例"], "data_model": ["unit_nav"]}
    )
    merged = config.merge(
        catalog,
        on=["fund_code", "fund_name"],
        suffixes=("_configured", "_catalog"),
    )
    assert merged.loc[0, "data_model_configured"] == "money_yield"
    assert merged.loc[0, "data_model_catalog"] == "unit_nav"
