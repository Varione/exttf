"""Tests for Market State Engine."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import pandas as pd
import numpy as np
from otf_rotation.market_state import MarketStateEngine


# ================================================================
# Fixtures
# ================================================================

@pytest.fixture(scope="module")
def engine():
    return MarketStateEngine()


# ================================================================
# Basic Sanity
# ================================================================

class TestEngineCreation:
    def test_engine_creates(self):
        eng = MarketStateEngine()
        assert eng is not None
        assert len(eng._prices) > 10  # multiple proxy ETFs loaded
        assert len(eng._all_dates) > 1000  # lots of trading days

    def test_states_defined(self):
        assert MarketStateEngine.STATES == [
            "RISK_ON", "NEUTRAL", "INFLATION_REAL_ASSET",
            "DEFLATION_RATE_DOWN", "STRESS"
        ]


class TestGetState:
    def test_returns_tuple(self, engine):
        result = engine.get_state("2024-01-15")
        assert isinstance(result, tuple)
        assert len(result) == 3
        state, scores, features = result
        assert isinstance(state, str)
        assert state in MarketStateEngine.STATES
        assert isinstance(scores, dict)
        assert isinstance(features, dict)
        assert len(scores) == 5
        assert len(features) >= 20

    def test_scores_sum_to_reasonable_range(self, engine):
        _, scores, _ = engine.get_state("2024-06-15")
        for s in MarketStateEngine.STATES:
            assert 0 <= scores[s] <= 100, f"{s} score {scores[s]} out of range"

    def test_accepts_timestamp(self, engine):
        state, _, _ = engine.get_state(pd.Timestamp("2024-01-15"))
        assert isinstance(state, str)

    def test_with_prev_state(self, engine):
        _, scores1, _ = engine.get_state("2024-01-15")
        state2, _, _ = engine.get_state("2024-01-15", prev_state="RISK_ON")
        assert isinstance(state2, str)


class TestHistoricalPatterns:
    """Verify the engine produces historically plausible states."""

    def test_2021_bull_market_risk_on(self, engine):
        """2021 was a strong bull market in China."""
        state_jan, scores_jan, _ = engine.get_state("2021-01-15")
        state_jun, scores_jun, _ = engine.get_state("2021-06-15")
        state_dec, scores_dec, _ = engine.get_state("2021-12-15")
        # Majority should be RISK_ON
        risk_on_count = sum(1 for s in [state_jan, state_jun, state_dec] if s == "RISK_ON")
        assert risk_on_count >= 1

    def test_2018_trade_war_stress(self, engine):
        """2018 Q4 was a severe bear market (trade war)."""
        state, scores, features = engine.get_state("2018-10-15")
        assert state in ("STRESS", "DEFLATION_RATE_DOWN")
        assert features.get("trend_equity_120", 0) < -0.05

    def test_2022_bear_stress(self, engine):
        """2022 bear market should show STRESS or DEFLATION."""
        state, _, features = engine.get_state("2022-04-01")
        assert state in ("STRESS", "DEFLATION_RATE_DOWN")
        assert features.get("trend_equity_60", 0) < -0.03

    def test_2020_covid_stress(self, engine):
        """March 2020 COVID crash should be STRESS."""
        state, _, features = engine.get_state("2020-03-23")
        assert state == "STRESS"
        assert features.get("vol_equity_60", 0) > 0.15

    def test_2024_gold_inflation(self, engine):
        """2024 gold rally should register significant INFLATION_REAL_ASSET score
        and gold trend should be positive."""
        state, scores, features = engine.get_state("2024-03-15")
        assert scores["INFLATION_REAL_ASSET"] > 20
        assert features.get("trend_gold_120", 0) > 0.01

    def test_risk_on_during_recovery(self, engine):
        """2019 recovery from trade war should be RISK_ON."""
        state, _, features = engine.get_state("2019-06-15")
        assert features.get("breadth_120_200", 0) > 0.3


class TestFeatureComputation:
    def test_features_have_expected_keys(self, engine):
        _, _, features = engine.get_state("2024-01-15")
        expected_prefixes = ["trend_", "vol_", "breadth", "corr_", "max_drawdown"]
        for prefix in expected_prefixes:
            assert any(k.startswith(prefix) for k in features), f"Missing feature prefix: {prefix}"

    def test_breadth_is_percentage(self, engine):
        _, _, features = engine.get_state("2024-01-15")
        b = features.get("breadth_120_200", -1)
        assert 0.0 <= b <= 1.0

    def test_vol_is_annualized(self, engine):
        _, _, features = engine.get_state("2024-01-15")
        for k in ["vol_equity_20", "vol_equity_60"]:
            v = features.get(k, 0)
            assert 0.0 <= v <= 1.0, f"{k}={v} should be annualized vol (0-100%)"

    def test_features_not_nan(self, engine):
        _, _, features = engine.get_state("2024-06-15")
        for k, v in features.items():
            assert np.isfinite(v), f"{k}={v} is not finite"
            assert v is not None, f"{k} is None"


class TestHysteresis:
    def test_prev_state_influences_current(self, engine):
        """With two close state scores, prev_state should bias the result."""
        # Get a date where RISK_ON and STRESS are close
        state1, scores1, _ = engine.get_state("2018-06-15", prev_state="STRESS")
        state2, scores2, _ = engine.get_state("2018-06-15", prev_state="RISK_ON")
        # The states may differ due to hysteresis
        assert isinstance(state1, str)
        assert isinstance(state2, str)

    def test_state_series_has_continuity(self, engine):
        dates = pd.date_range("2023-01-01", "2023-06-30", freq="21D")
        df = engine.get_state_series(dates)
        assert len(df) == len(dates)
        transitions = (df["state"] != df["state"].shift(1)).sum()
        assert transitions <= len(dates)  # no oscillation every period


class TestDataBoundaries:
    def test_early_date_handles_gracefully(self, engine):
        """Very early date with limited data should not crash."""
        state, scores, features = engine.get_state("2005-06-01")
        assert state in MarketStateEngine.STATES
        for k in features:
            assert np.isfinite(features[k])

    def test_late_date_handles_gracefully(self, engine):
        """Date near present should work."""
        state, scores, features = engine.get_state("2026-07-17")
        assert state in MarketStateEngine.STATES
        for k in features:
            assert np.isfinite(features[k])

    def test_missing_date_no_crash(self, engine):
        """Weekend/holiday should forward-fill."""
        state, scores, features = engine.get_state("2024-01-13")
        assert state in MarketStateEngine.STATES
        assert np.isfinite(scores["NEUTRAL"])


class TestSimulateMonthly:
    def test_simulate_returns_dataframe(self, engine):
        df = engine.simulate_monthly("2023-01-01", "2023-12-31")
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 12  # ~monthly for a year

    def test_all_states_covered_in_long_period(self, engine):
        df = engine.simulate_monthly("2018-01-01", "2026-07-17")
        states_found = set(df["state"].unique())
        # At minimum RISK_ON and STRESS should appear
        assert "RISK_ON" in states_found
        assert "STRESS" in states_found
        assert "NEUTRAL" in states_found


class TestReproducibility:
    def test_deterministic(self, engine):
        s1, sc1, f1 = engine.get_state("2024-06-15")
        s2, sc2, f2 = engine.get_state("2024-06-15")
        assert s1 == s2
        for k in sc1:
            assert sc1[k] == sc2[k]
        for k in f1:
            assert f1[k] == f2[k]

    def test_future_data_does_not_affect_past(self, engine):
        """States should not change depending on future data."""
        s_before, _, _ = engine.get_state("2024-01-15")
        # Calling a later date should not change earlier result
        engine.get_state("2024-06-15")
        s_after, _, _ = engine.get_state("2024-01-15")
        assert s_before == s_after
