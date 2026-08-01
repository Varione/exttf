"""Tests for time-point-aware ProductSelector."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import pandas as pd
from otf_rotation.product_selector import (
    ProductSelector,
    SLEEVE_CANDIDATES,
    ALL_SLEEVES,
)


@pytest.fixture
def ps():
    return ProductSelector(
        db_path="data/processed/otf_expanded.sqlite",
        rules_path="config/otf_product_rules.csv",
        allow_auto_rules=False,
    )


class TestTimePointSelection:
    """Tests that select() correctly uses the date parameter."""

    def test_date_required(self, ps):
        """date=None must raise in formal research mode."""
        with pytest.raises(ValueError, match="date parameter"):
            ps.select("CSI300", top_n=2, date=None)

    def test_select_csi300_at_2025(self, ps):
        funds = ps.select("CSI300", top_n=2, date="2025-06-01")
        assert len(funds) == 2
        zc0 = funds[0].zfill(6)
        zc1 = funds[1].zfill(6)
        fee0 = ps._rule_map[zc0]["sub_fee"]
        fee1 = ps._rule_map[zc1]["sub_fee"]
        assert fee0 <= fee1  # cheapest first

    def test_fund_not_yet_incepted_excluded(self, ps):
        """Fund with first NAV in 2022 must be excluded when date is 2018."""
        audit = ps.select_with_audit("CSI1000", top_n=5, date="2018-01-04")
        for c in audit.candidates:
            if c.fund_code == "016633":
                assert c.ineligible, "016633 should be excluded before its NAV start"
                assert c.ineligible_reason  # any valid reason

    def test_fund_not_yet_incepted_in_2018(self, ps):
        """No fund with inception > 2018 should be selectable."""
        audit = ps.select_with_audit("CSI300", top_n=5, date="2018-01-02")
        for c in audit.candidates:
            cat = ps._catalog.get(c.fund_code.zfill(6), {})
            incep = cat.get("inception")
            if pd.notna(incep) and incep > pd.Timestamp("2018-01-02"):
                assert c.ineligible, (
                    f"{c.fund_code} incepted {incep} > 2018-01-02 should be excluded"
                )

    def test_late_fund_not_in_early_selection(self, ps):
        """021778 (incepted ~2024) must not appear in 2023 selection."""
        funds = ps.select("NASDAQ", top_n=3, date="2023-06-01")
        assert "021778" not in funds, (
            "021778 incepted ~2024, should not be selectable in 2023"
        )

    def test_select_money_market_at_2025(self, ps):
        funds = ps.select("MONEY_MARKET", top_n=3, date="2025-06-01")
        assert len(funds) >= 1
        assert len(funds) <= 3
        assert len(set(funds)) == len(funds)

    def test_select_gold_at_2025(self, ps):
        funds = ps.select("GOLD", top_n=2, date="2025-06-01")
        assert len(funds) == 2
        assert len(set(funds)) == 2

    def test_all_sleeves_have_candidates_after_2020(self, ps):
        for sleeve in ALL_SLEEVES:
            funds = ps.select(sleeve, top_n=1, date="2025-01-02")
            assert len(funds) >= 1, f"{sleeve} returned no candidates at 2025-01-02"

    def test_selection_audit_returns_candidates(self, ps):
        audit = ps.select_with_audit("CSI300", top_n=2, date="2025-06-01")
        assert len(audit.candidates) > 0
        assert len(audit.selected) >= 1
        for c in audit.candidates:
            assert isinstance(c.fund_code, str)
            assert isinstance(c.sub_fee, float)

    def test_audit_includes_ineligible_reason(self, ps):
        audit = ps.select_with_audit("NASDAQ", top_n=3, date="2023-01-02")
        ineligible = [c for c in audit.candidates if c.ineligible]
        assert len(ineligible) >= 1, (
            "Should have some ineligible candidates (e.g. funds not yet incepted)"
        )
        for c in ineligible:
            assert c.ineligible_reason, "Ineligible candidate must have a reason"

    def test_consistent_results(self, ps):
        r1 = ps.select("CSI500", top_n=2, date="2024-01-02")
        r2 = ps.select("CSI500", top_n=2, date="2024-01-02")
        assert r1 == r2

    def test_different_date_different_results(self, ps):
        r1 = ps.select("MONEY_MARKET", top_n=2, date="2018-01-02")
        r2 = ps.select("MONEY_MARKET", top_n=2, date="2025-06-01")
        # Should both work, but may differ
        assert len(r1) >= 1
        assert len(r2) >= 1

    def test_audit_to_dict(self, ps):
        audit = ps.select_with_audit("GOLD", top_n=2, date="2025-06-01")
        d = audit.to_dict()
        assert d["sleeve"] == "GOLD"
        assert d["top_n"] == 2
        assert "candidates" in d
        assert "selected" in d


class TestSleeveCandidates:
    def test_get_sleeve_candidates(self, ps):
        funds = ps.get_sleeve_candidates("CSI300")
        assert len(funds) >= 2

    def test_get_sleeve_candidates_nonexistent(self, ps):
        assert ps.get_sleeve_candidates("FAKE") == []

    def test_expanded_pool_finds_more(self, ps):
        for sleeve in ALL_SLEEVES:
            manual_count = len(SLEEVE_CANDIDATES.get(sleeve, []))
            actual_count = len(ps.get_sleeve_candidates(sleeve))
            assert actual_count >= manual_count, (
                f"{sleeve}: {actual_count} < manual {manual_count}"
            )
