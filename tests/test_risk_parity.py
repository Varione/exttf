"""Tests for Rolling Risk Parity Engine per P0-A."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import pandas as pd
import numpy as np
from otf_rotation.risk_parity import RollingRiskParityEngine


@pytest.fixture
def sample_nav():
    np.random.seed(42)
    dates = pd.bdate_range("2018-01-01", periods=600)
    navs = {}
    for asset in ["STOCK", "GOLD", "BOND", "CASH"]:
        mu = {"STOCK": 0.0005, "GOLD": 0.0002, "BOND": 0.0001, "CASH": 0.00003}[asset]
        sigma = {"STOCK": 0.015, "GOLD": 0.012, "BOND": 0.006, "CASH": 0.001}[asset]
        returns = np.random.normal(mu, sigma, len(dates))
        navs[asset] = (1 + returns).cumprod()
    return pd.DataFrame(navs, index=dates)


class TestRollingRiskParityEngine:
    def test_engine_creates(self, sample_nav):
        engine = RollingRiskParityEngine(
            sample_nav, asset_columns=["STOCK", "GOLD", "BOND", "CASH"]
        )
        assert engine.rolling_window == 252
        assert engine.asset_cap == 0.40

    def test_fallback_when_insufficient_history(self, sample_nav):
        engine = RollingRiskParityEngine(
            sample_nav, asset_columns=["STOCK", "GOLD", "BOND", "CASH"], min_window=60
        )
        early_date = sample_nav.index[10]
        weights = engine.get_weights(early_date)
        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_weights_sum_to_one(self, sample_nav):
        engine = RollingRiskParityEngine(
            sample_nav, asset_columns=["STOCK", "GOLD", "BOND", "CASH"]
        )
        mid_date = sample_nav.index[300]
        weights = engine.get_weights(mid_date)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_non_negative_weights(self, sample_nav):
        engine = RollingRiskParityEngine(
            sample_nav, asset_columns=["STOCK", "GOLD", "BOND", "CASH"]
        )
        mid_date = sample_nav.index[300]
        weights = engine.get_weights(mid_date)
        assert all(v >= 0 for v in weights.values())

    def test_asset_cap_respected(self, sample_nav):
        engine = RollingRiskParityEngine(
            sample_nav, asset_columns=["STOCK", "GOLD", "BOND", "CASH"], asset_cap=0.35
        )
        mid_date = sample_nav.index[300]
        weights = engine.get_weights(mid_date)
        assert all(v <= 0.35 + 1e-9 for v in weights.values())

    def test_risk_contributions_near_equal(self, sample_nav):
        engine = RollingRiskParityEngine(
            sample_nav, asset_columns=["STOCK", "GOLD", "BOND", "CASH"]
        )
        mid_date = sample_nav.index[300]
        weights, audit = engine.get_weights_with_audit(mid_date)
        rc = audit["risk_contributions"]
        target = 1.0 / len(weights)
        max_deviation = max(abs(v - target) for v in rc.values())
        assert max_deviation < 0.45

    def test_no_future_leak(self, sample_nav):
        engine = RollingRiskParityEngine(
            sample_nav, asset_columns=["STOCK", "GOLD", "BOND", "CASH"]
        )
        date_a = sample_nav.index[250]
        weights_a = engine.get_weights(date_a)
        corrupted = sample_nav.copy()
        corrupted.loc[date_a + pd.Timedelta(days=1):, "STOCK"] *= 2.0
        engine_b = RollingRiskParityEngine(
            corrupted, asset_columns=["STOCK", "GOLD", "BOND", "CASH"]
        )
        weights_b = engine_b.get_weights(date_a)
        assert np.allclose(list(weights_a.values()), list(weights_b.values()), atol=1e-9)

    def test_singular_covariance_fallback(self):
        dates = pd.bdate_range("2021-01-01", periods=100)
        nav_df = pd.DataFrame({"A": np.ones(100), "B": np.ones(100)}, index=dates)
        engine = RollingRiskParityEngine(nav_df, asset_columns=["A", "B"], min_window=30)
        weights = engine.get_weights(dates[-1])
        assert len(weights) == 2
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_audit_method_is_rolling(self, sample_nav):
        engine = RollingRiskParityEngine(
            sample_nav, asset_columns=["STOCK", "GOLD", "BOND", "CASH"]
        )
        mid_date = sample_nav.index[300]
        _, audit = engine.get_weights_with_audit(mid_date)
        assert audit["method"] == "rolling_risk_parity"

    def test_audit_fallback_on_error(self, sample_nav):
        dates = pd.bdate_range("2021-01-01", periods=300)
        nav_df = pd.DataFrame({"A": np.zeros(300)}, index=dates)
        engine = RollingRiskParityEngine(nav_df, asset_columns=["A"], min_window=60)
        weights, audit = engine.get_weights_with_audit(dates[-1])
        assert "fallback" in audit["method"]

    def test_asset_cap_strictly_enforced_after_projection(self):
        dates = pd.bdate_range("2021-01-01", periods=400)
        nav_df = pd.DataFrame({
            "A": (1 + np.random.normal(0.001, 0.005, 400)).cumprod(),
            "B": np.ones(400),
            "C": np.ones(400),
        }, index=dates)
        engine = RollingRiskParityEngine(nav_df, asset_columns=["A", "B", "C"], asset_cap=0.35, min_window=60)
        weights = engine.get_weights(dates[-1])
        assert all(v <= 0.35 + 1e-9 for v in weights.values()), f"Cap violated: {weights}"
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_negative_correlation_handling(self):
        dates = pd.bdate_range("2021-01-01", periods=400)
        r = np.random.normal(0, 0.01, 400)
        nav_df = pd.DataFrame({
            "A": (1 + r).cumprod(),
            "B": (1 - r).cumprod(),
        }, index=dates)
        engine = RollingRiskParityEngine(nav_df, asset_columns=["A", "B"], min_window=60)
        weights = engine.get_weights(dates[-1])
        assert all(v >= 0 for v in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_missing_data_handling(self):
        dates = pd.bdate_range("2021-01-01", periods=400)
        nav_df = pd.DataFrame({
            "A": (1 + np.random.normal(0.001, 0.01, 400)).cumprod(),
            "B": (1 + np.random.normal(0.001, 0.01, 400)).cumprod(),
        }, index=dates)
        nav_df.iloc[250:350, nav_df.columns.get_loc("B")] = np.nan
        engine = RollingRiskParityEngine(nav_df, asset_columns=["A", "B"], min_window=60)
        weights = engine.get_weights(dates[-1])
        assert len(weights) == 2

    def test_audit_includes_covariance_condition_number(self, sample_nav):
        engine = RollingRiskParityEngine(
            sample_nav, asset_columns=["STOCK", "GOLD", "BOND", "CASH"]
        )
        mid_date = sample_nav.index[300]
        _, audit = engine.get_weights_with_audit(mid_date)
        assert "covariance_condition_number" in audit
        assert isinstance(audit["covariance_condition_number"], float)

    def test_audit_includes_optimization_status(self, sample_nav):
        engine = RollingRiskParityEngine(
            sample_nav, asset_columns=["STOCK", "GOLD", "BOND", "CASH"]
        )
        mid_date = sample_nav.index[300]
        _, audit = engine.get_weights_with_audit(mid_date)
        assert "optimization_status" in audit

    def test_audit_fallback_reason_when_insufficient(self, sample_nav):
        engine = RollingRiskParityEngine(
            sample_nav, asset_columns=["STOCK", "GOLD", "BOND", "CASH"], min_window=60
        )
        early_date = sample_nav.index[10]
        _, audit = engine.get_weights_with_audit(early_date)
        assert audit["fallback_reason"] == "insufficient_window"

    def test_monetary_constant_nav_fallback(self):
        dates = pd.bdate_range("2021-01-01", periods=300)
        nav_df = pd.DataFrame({
            "A": np.ones(300),
            "B": (1 + np.random.normal(0.001, 0.01, 300)).cumprod(),
        }, index=dates)
        engine = RollingRiskParityEngine(nav_df, asset_columns=["A", "B"], min_window=60)
        weights = engine.get_weights(dates[-1])
        assert len(weights) == 2
        assert abs(sum(weights.values()) - 1.0) < 1e-6
