"""Tests for full OTC catalog normalization."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_full_otf_catalog import normalize_catalog, split_generic_share_class


def test_share_class_split():
    assert split_generic_share_class("示例指数基金A") == ("示例指数基金", "A")
    assert split_generic_share_class("无份额标识") == ("无份额标识", "")


def test_catalog_classifies_money_and_fixed_income():
    raw = pd.DataFrame(
        [
            ["1", "a", "货币示例A", "货币型-普通货币", "a"],
            ["2", "b", "债券指数示例C", "指数型-固收", "b"],
            ["3", "c", "股票指数示例A", "指数型-股票", "c"],
        ]
    )
    result = normalize_catalog(raw)
    assert result["fund_code"].tolist() == ["000001", "000002", "000003"]
    assert result["research_scope"].tolist() == [
        "money", "passive_fixed_income", "passive_equity"
    ]
    assert result.loc[0, "data_model"] == "money_yield"
    assert result.loc[1, "data_model"] == "unit_nav"
