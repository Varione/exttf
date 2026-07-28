from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_etf_otf_mapping import build_mapping, normalize_text, split_share_class


def test_split_share_class():
    assert split_share_class("易方达沪深300ETF联接A") == (
        "易方达沪深300ETF联接",
        "A",
    )
    assert split_share_class("某基金ETF联接C类") == ("某基金ETF联接", "C")
    assert split_share_class("某基金ETF联接A(人民币)") == ("某基金ETF联接", "A")
    assert split_share_class("某基金ETF联接C美元现汇") == ("某基金ETF联接", "C")


def test_normalize_text_removes_execution_tokens():
    assert normalize_text("沪深300ETF联接(QDII)") == "沪深300"


def test_exact_underlying_and_issuer_mapping_is_high_confidence():
    feeders = pd.DataFrame(
        [
            {
                "fund_code": "110020",
                "fund_name": "易方达沪深300ETF联接A",
                "fund_family": "易方达沪深300ETF联接",
                "share_class": "A",
                "fund_type": "指数型-股票",
                "selected_share": True,
            }
        ]
    )
    etfs = pd.DataFrame(
        [
            {
                "symbol": "510310",
                "name": "沪深300ETF易方达",
                "underlying_name": "沪深300",
                "underlying_norm": "沪深300",
                "issuer_norm": "易方达",
                "name_norm": "沪深300易方达",
                "asset_class": "宽基",
                "rows_valid": 1000,
                "median_amount_60d": 1e8,
            },
            {
                "symbol": "510300",
                "name": "沪深300ETF华泰柏瑞",
                "underlying_name": "沪深300",
                "underlying_norm": "沪深300",
                "issuer_norm": "华泰柏瑞",
                "name_norm": "沪深300华泰柏瑞",
                "asset_class": "宽基",
                "rows_valid": 2000,
                "median_amount_60d": 2e8,
            },
        ]
    )
    result = build_mapping(feeders, etfs).iloc[0]
    assert result["etf_symbol"] == "510310"
    assert result["mapping_confidence"] == "HIGH"
    assert bool(result["executable"]) is True
