from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from otf_trading_rules import (
    FundTradingRule,
    ProductRuleBook,
    RedemptionFeeTier,
    SubscriptionFeeTier,
    TradingRestrictionEvent,
)


def test_holding_period_fee_tiers_are_half_open():
    book = ProductRuleBook(
        fee_tiers={
            "F": [
                RedemptionFeeTier(0, 7, 0.015),
                RedemptionFeeTier(7, None, 0.0),
            ]
        }
    )
    assert book.fee_rate("F", 6, 0.1) == pytest.approx(0.015)
    assert book.fee_rate("F", 7, 0.1) == 0.0


def test_trading_event_overrides_static_rule():
    rule = FundTradingRule("000001", 0.0, 1, 1, 1)
    event = TradingRestrictionEvent(
        "000001", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-10"),
        False, True, 1000.0, "temporary limit", "official notice"
    )
    book = ProductRuleBook({"000001": rule}, events={"000001": [event]})
    sub_open, red_open, limit, reason = book.order_constraints(
        "000001", pd.Timestamp("2024-01-05")
    )
    assert not sub_open
    assert red_open
    assert limit == 1000.0
    assert reason == "temporary limit"


def test_project_rule_files_load_and_cover_direct_assets():
    book = ProductRuleBook.from_csv()
    coverage = book.coverage(
        ["260102", "040003", "217004", "050003", "202301",
         "006663", "001512", "005839", "000148", "014430"]
    )
    assert coverage["rule_covered_count"] == 10
    assert book.rule_for("006663").redemption_settlement_days == 7
    assert book.fee_rate("006663", 6, 0.0) == pytest.approx(0.015)
    assert book.fee_rate("006663", 7, 0.1) == 0.0
    traded = book.coverage(
        ["000008", "000071", "000218", "007466", "016633", "021778",
         "050021", "050025", "160706"]
    )
    assert traded["externally_verified_count"] == 9


def test_fee_stress_scales_product_rules_and_tiers():
    book = ProductRuleBook(
        {"F": FundTradingRule("F", 0.01, 1, 1, 1)},
        {"F": [RedemptionFeeTier(0, None, 0.02)]},
        {"F": [SubscriptionFeeTier(0, None, 0.01, None)]},
    ).scaled_fees(2.0)
    assert book.rule_for("F").subscription_fee_rate == pytest.approx(0.02)
    assert book.fee_rate("F", 10, 0.0) == pytest.approx(0.04)
    assert book.subscription_fee_amount("F", 100.0, 0.0) == pytest.approx(2.0)


def test_official_amount_tier_supports_fixed_large_order_fee():
    book = ProductRuleBook.from_csv()
    assert book.subscription_fee_amount("001512", 500_000, 0.0) == 4_000
    assert book.subscription_fee_amount("001512", 6_000_000, 0.0) == 1_000


def test_channel_discount_does_not_discount_redemption_or_fixed_fee():
    book = ProductRuleBook(
        {"F": FundTradingRule("F", 0.01, 1, 1, 1)},
        {"F": [RedemptionFeeTier(0, None, 0.02)]},
        {
            "F": [
                SubscriptionFeeTier(0, 1000, 0.01, None),
                SubscriptionFeeTier(1000, None, None, 10.0),
            ]
        },
    ).with_subscription_discount(0.1)
    assert book.subscription_fee_amount("F", 100.0, 0.0) == pytest.approx(0.1)
    assert book.subscription_fee_amount("F", 2000.0, 0.0) == 10.0
    assert book.fee_rate("F", 10, 0.0) == pytest.approx(0.02)
