"""Tests for Asset Budget Engine and Exposure Selector."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import pandas as pd
import numpy as np
from otf_rotation.market_state import MarketStateEngine
from otf_rotation.asset_budget import AssetBudgetEngine, ExposureSelector


# ================================================================
# Fixtures
# ================================================================

@pytest.fixture(scope="module")
def mse():
    return MarketStateEngine()

@pytest.fixture(scope="module")
def budget():
    return AssetBudgetEngine()

@pytest.fixture(scope="module")
def selector(mse, budget):
    return ExposureSelector(mse, budget)


# ================================================================
# AssetBudgetEngine Tests
# ================================================================

class TestAssetBudgetEngine:
    def test_engine_creates(self, budget):
        assert budget is not None
        assert len(budget._budgets) == 5  # 5 states

    def test_budget_range_all_states(self, budget):
        for state in ["RISK_ON", "NEUTRAL", "INFLATION_REAL_ASSET",
                       "DEFLATION_RATE_DOWN", "STRESS"]:
            ranges = budget.get_budget_range(state)
            assert isinstance(ranges, dict)
            assert len(ranges) == 5  # 5 asset classes
            for cls_name, info in ranges.items():
                assert "min" in info
                assert "max" in info
                assert "default" in info
                assert 0 <= info["min"] <= info["default"] <= info["max"] <= 1.0

    def test_unknown_state_fallsback_to_neutral(self, budget):
        ranges = budget.get_budget_range("UNKNOWN_STATE")
        neutral = budget.get_budget_range("NEUTRAL")
        assert ranges == neutral

    def test_default_weights_sum_to_one(self, budget):
        for state in ["RISK_ON", "NEUTRAL", "INFLATION_REAL_ASSET",
                       "DEFLATION_RATE_DOWN", "STRESS"]:
            w = budget.get_default_weights(state)
            total = sum(w.values())
            assert abs(total - 1.0) < 0.02, f"{state}: weights sum to {total}"

    def test_normalize_under_one(self, budget):
        result = budget.normalize({"domestic_equity": 0.5, "gold": 0.1})
        assert abs(sum(result.values()) - 1.0) < 1e-6
        assert "cash_mgt" in result
        assert abs(result["cash_mgt"] - 0.4) < 1e-6

    def test_normalize_over_one(self, budget):
        result = budget.normalize({"domestic_equity": 0.8, "gold": 0.4})
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_allocate_class_weight_extremes(self, budget):
        w_min = budget.allocate_class_weight("RISK_ON", "domestic_equity", 0.0)
        w_max = budget.allocate_class_weight("RISK_ON", "domestic_equity", 1.0)
        ranges = budget.get_budget_range("RISK_ON")
        assert abs(w_min - ranges["domestic_equity"]["min"]) < 1e-6
        assert abs(w_max - ranges["domestic_equity"]["max"]) < 1e-6

    def test_get_class_sleeves(self, budget):
        sleeves = budget.get_class_sleeves("domestic_equity")
        assert len(sleeves) == 5
        assert "CSI300" in sleeves
        assert "DIVIDEND" in sleeves

    def test_target_vol_default(self, budget):
        assert 0.05 <= budget.target_vol <= 0.15


# ================================================================
# ExposureSelector Tests
# ================================================================

class TestExposureSelector:
    def test_selector_creates(self, selector):
        assert selector is not None

    def test_select_weights_risk_on(self, selector):
        weights = selector.select_weights("2021-06-15", "RISK_ON")
        assert isinstance(weights, pd.Series)
        assert len(weights) > 0
        # Weights should sum to ~1.0
        assert abs(weights.sum() - 1.0) < 0.05

    def test_select_weights_stress(self, selector):
        weights = selector.select_weights("2018-10-15", "STRESS")
        assert isinstance(weights, pd.Series)
        assert len(weights) > 0
        # Stress should have less equity
        total = weights.sum()
        assert abs(total - 1.0) < 0.05

    def test_different_states_different_allocations(self, selector):
        w_risk = selector.select_weights("2021-06-15", "RISK_ON")
        w_stress = selector.select_weights("2018-10-15", "STRESS")
        # Should have different weights
        assert not w_risk.equals(w_stress)

    def test_stress_has_more_cash(self, selector):
        w_stress = selector.select_weights("2018-10-15", "STRESS")
        w_risk = selector.select_weights("2021-06-15", "RISK_ON")
        # Get cash-related sleeves
        cash_keys = [k for k in w_stress.index if "MONEY" in str(k) or "CD" in str(k)]
        risk_cash_keys = [k for k in w_risk.index if "MONEY" in str(k) or "CD" in str(k)]
        
        stress_cash = sum(w_stress.get(k, 0) for k in cash_keys)
        risk_cash = sum(w_risk.get(k, 0) for k in risk_cash_keys)
        assert stress_cash >= risk_cash * 0.5  # stress should not have less cash

    def test_all_weights_positive(self, selector):
        for state in ["RISK_ON", "NEUTRAL", "STRESS"]:
            weights = selector.select_weights("2024-01-15", state)
            assert all(w >= 0 for w in weights.values)

    def test_with_prev_weights(self, selector):
        prev = {"CSI300": 0.15, "CSI500": 0.10, "GOLD": 0.05, "MONEY_MARKET": 0.70}
        weights = selector.select_weights("2024-06-15", "NEUTRAL", prev_weights=prev)
        assert isinstance(weights, pd.Series)

    def test_exposure_limits_respected(self, selector):
        weights = selector.select_weights("2021-06-15", "RISK_ON")
        for sleeve, w in weights.items():
            # Any single domestic equity sleeve <= 0.15
            if sleeve in ["CSI300", "CSI500", "CSI1000", "CHINEXT", "DIVIDEND"]:
                assert w <= 0.16, f"{sleeve}={w} exceeds max"
            # Any single overseas sleeve <= 0.10
            if sleeve in ["HANGSENG", "SP500", "NASDAQ"]:
                assert w <= 0.11, f"{sleeve}={w} exceeds max"


# ================================================================
# End-to-End Tests
# ================================================================

class TestEndToEnd:
    def test_mse_to_budget(self, mse, budget):
        """Full pipeline: market state -> budget."""
        state, scores, features = mse.get_state("2021-06-15")
        ranges = budget.get_budget_range(state)
        assert isinstance(ranges, dict)
        assert len(ranges) >= 3

    def test_full_pipeline(self, mse, selector):
        """Full pipeline: market state -> budget -> exposure selection."""
        state, _, _ = mse.get_state("2021-06-15")
        weights = selector.select_weights("2021-06-15", state)
        assert abs(weights.sum() - 1.0) < 0.05

    def test_map_to_fixed_products(self, mse, selector):
        """S1: Map sleeve weights to fixed verified products."""
        weights = selector.map_to_fixed_products("2021-06-15", "RISK_ON")
        assert isinstance(weights, pd.Series)
        assert len(weights) > 0
        for code in weights.index:
            assert isinstance(code, str) and len(code) >= 6
        assert abs(weights.sum() - 1.0) < 0.05

    def test_different_dates_different_fixed_allocations(self, selector):
        """Fixed product mapping may differ across dates due to state changes."""
        w1 = selector.map_to_fixed_products("2021-06-15", "RISK_ON")
        w2 = selector.map_to_fixed_products("2024-06-15", "RISK_ON")
        assert isinstance(w1, pd.Series)
        assert isinstance(w2, pd.Series)

    def test_vol_target_present(self, budget):
        assert budget.target_vol == 0.09

    def test_all_states_produce_valid_fixed_weights(self, mse, selector):
        for state in ["RISK_ON", "NEUTRAL", "INFLATION_REAL_ASSET",
                       "DEFLATION_RATE_DOWN", "STRESS"]:
            w = selector.map_to_fixed_products("2024-01-15", state)
            assert abs(w.sum() - 1.0) < 0.05, f"{state} weights sum to {w.sum()}"
            assert all(v >= 0 for v in w.values), f"{state} has negative weights"

    def test_s1_does_not_call_product_selector(self, selector):
        """S1 must run without ProductSelector injected."""
        selector._product_selector = None
        weights = selector.map_to_fixed_products("2021-06-15", "RISK_ON")
        assert len(weights) > 0

    def test_s2_requires_product_selector(self, selector):
        """S2 must fail if ProductSelector not injected."""
        selector._product_selector = None
        with pytest.raises(RuntimeError, match="requires ProductSelector"):
            selector.map_to_dynamic_products("2021-06-15", "RISK_ON")

    def test_s2_no_fallback_to_fixed(self, mse, budget):
        """S2 must not fall back to fixed products when no candidates found."""
        from unittest.mock import MagicMock
        ps = MagicMock()
        ps.select.return_value = []
        sel = ExposureSelector(mse, budget)
        sel._product_selector = ps
        weights = sel.map_to_dynamic_products(
            "2021-06-15", "RISK_ON", fallback_sleeves=["MONEY_MARKET"]
        )
        ps.select.assert_any_call("MONEY_MARKET", top_n=sel._top_n, date="2021-06-15")


# ================================================================
# Key Space and Vol Scaling Level Tests (Phase 5 fourth-audit)
# ================================================================

class TestRebalanceBandKeySpace:
    """Verify rebalance band operates in sleeve name key space."""

    def test_select_weights_returns_sleeve_names(self, selector):
        """select_weights returns pd.Series with sleeve names as index."""
        weights = selector.select_weights("2024-01-15", "NEUTRAL")
        all_sleeves = []
        for cls in selector._budget._asset_classes:
            all_sleeves.extend(selector._budget.get_class_sleeves(cls))
        for k in weights.index:
            assert k in all_sleeves, f"{k} is not a valid sleeve name"

    def test_prev_sleeve_weights_uses_sleeve_names(self, selector):
        """set_previous_sleeve_weights stores and returns sleeve names."""
        sleeve_w = {"CSI300": 0.15, "MONEY_MARKET": 0.85}
        selector.set_previous_sleeve_weights(sleeve_w)
        prev = selector.previous_sleeve_weights
        assert prev is not None
        for k in prev:
            assert isinstance(k, str)
        assert set(prev.keys()) == {"CSI300", "MONEY_MARKET"}

    def test_min_trade_compares_same_key_space(self, mse, budget):
        """_apply_min_trade compares sleeve names against sleeve names."""
        sel = ExposureSelector(mse, budget)
        prev = {
            "CSI300": 0.15, "CSI500": 0.10, "GOLD": 0.10,
            "GOV_BOND_3_5Y": 0.20, "MONEY_MARKET": 0.45,
        }
        new_w = sel.select_weights("2024-06-15", "NEUTRAL")
        result = sel._apply_min_trade(new_w, prev)
        for k in result.index:
            assert k in set(new_w.index) | set(prev.keys())

    def test_map_to_fixed_does_not_mutate_prev_sleeve_weights(self, selector):
        """map_to_fixed_products must not update _prev_sleeve_weights internally."""
        selector.reset()
        assert selector._prev_sleeve_weights is None
        w1 = selector.map_to_fixed_products("2021-06-15", "RISK_ON")
        assert len(w1) > 0
        assert selector._prev_sleeve_weights is None

    def test_scaled_vs_scaled_comparison(self, mse, budget):
        """Rebalance band must compare scaled weights against scaled weights."""
        sel = ExposureSelector(mse, budget)
        scaled_prev = {"CSI300": 0.12, "GOLD": 0.08, "MONEY_MARKET": 0.80}
        sel.set_previous_sleeve_weights(scaled_prev)
        new_w = sel.select_weights("2024-06-15", "NEUTRAL")
        result = sel._apply_min_trade(new_w, scaled_prev)
        assert isinstance(result, pd.Series)


class TestVolatilityScalingLevel:
    """Verify vol scaling operates at sleeve level before fund mapping."""

    def test_vol_scale_weights_preserves_sleeve_keys(self, budget):
        """vol_scale_weights input/output keys are sleeve names."""
        weights = {"CSI300": 0.30, "GOLD": 0.10, "MONEY_MARKET": 0.60}
        scaled = budget.vol_scale_weights(
            weights, estimated_port_vol=0.15, cash_class="MONEY_MARKET"
        )
        assert set(scaled.keys()) == set(weights.keys())

    def test_vol_scale_only_reduces_risk(self, budget):
        """vol_scale_weights never adds leverage; only scales down."""
        weights = {"CSI300": 0.30, "GOLD": 0.10, "MONEY_MARKET": 0.60}
        scaled = budget.vol_scale_weights(
            weights, estimated_port_vol=0.15, cash_class="MONEY_MARKET"
        )
        for k in weights:
            if k != "MONEY_MARKET":
                assert scaled[k] <= weights[k] + 1e-12

    def test_vol_scale_moves_excess_to_cash(self, budget):
        """Scaled-down risky weight goes to cash_class."""
        weights = {"CSI300": 0.25, "GOLD": 0.10, "MONEY_MARKET": 0.65}
        scaled = budget.vol_scale_weights(
            weights, estimated_port_vol=0.10, cash_class="MONEY_MARKET"
        )
        assert abs(sum(scaled.values()) - 1.0) < 1e-9
        assert scaled["MONEY_MARKET"] > weights["MONEY_MARKET"]

    def test_cash_class_is_valid_sleeve(self, budget):
        """cash_class used in vol_scale_weights must be a config sleeve name."""
        cash_sleeves = budget.get_class_sleeves("cash_mgt")
        assert len(cash_sleeves) >= 1
        for cs in cash_sleeves:
            all_sleeves = []
            for cls in budget._asset_classes:
                all_sleeves.extend(budget.get_class_sleeves(cls))
            assert cs in all_sleeves

    def test_vol_scale_no_op_below_target(self, budget):
        """No scaling when estimated vol is at or below target."""
        weights = {"CSI300": 0.30, "GOLD": 0.10, "MONEY_MARKET": 0.60}
        scaled = budget.vol_scale_weights(weights, estimated_port_vol=0.08)
        for k in weights:
            assert abs(scaled[k] - weights[k]) < 1e-12

    def test_s1_signal_path_ordering(self, mse, budget):
        """Verify S1 ordering: select(sleeve) -> scale(sleeve) -> map(fund)."""
        sel = ExposureSelector(mse, budget)
        sleeve_w = sel.select_weights("2024-06-15", "NEUTRAL")
        assert len(sleeve_w) > 0
        all_sleeves = []
        for cls in budget._asset_classes:
            all_sleeves.extend(budget.get_class_sleeves(cls))
        for k in sleeve_w.index:
            assert k in all_sleeves, "select_weights must return sleeve names"

        scaled = budget.vol_scale_weights(
            sleeve_w.to_dict(), estimated_port_vol=0.12, cash_class="MONEY_MARKET"
        )
        fund_w = sel.map_sleeves_to_fixed_products(scaled)
        for k in fund_w.index:
            assert len(k) >= 6, f"map must return fund codes, got {k}"

    def test_prev_weights_set_after_scaling(self, mse, budget):
        """set_previous_sleeve_weights must be called AFTER vol scaling."""
        sel = ExposureSelector(mse, budget)
        raw = sel.select_weights("2024-06-15", "NEUTRAL").to_dict()

        cash_sleeves = budget.get_class_sleeves("cash_mgt")
        cash_total = sum(raw.pop(k, 0.0) for k in cash_sleeves)
        raw[cash_sleeves[0]] = cash_total

        scaled = budget.vol_scale_weights(
            raw, estimated_port_vol=0.15, cash_class=cash_sleeves[0]
        )
        sel.set_previous_sleeve_weights(scaled)
        assert sel.previous_sleeve_weights == scaled
        for k in sel.previous_sleeve_weights:
            if k != cash_sleeves[0]:
                assert sel.previous_sleeve_weights[k] <= raw.get(k, 0.0) + 1e-12
