"""Tests for C3 low-turnover momentum signal."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from otf_rotation.c3_low_turnover_momentum import (
    C3LowTurnoverMomentumSignal,
    CORE_WEIGHTS,
    SATELLITE_POOL,
    SATELLITE_BUDGET,
    MAX_SATELLITES,
    SATELLITE_SLOT_WEIGHT,
    MIN_HOLDING_QUARTERS,
    DRIFT_THRESHOLD,
    FALLBACK_PRIMARY,
    FALLBACK_SECONDARY,
    _quarters_held,
)


def _make_growth_df_with_trends(
    codes: list[str],
    start_date: str = "2018-01-04",
    end_date: str = "2026-07-30",
    trends: dict[str, float] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    if trends is None:
        trends = {c: 0.05 for c in codes}
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start_date, end_date, freq="B")
    rows = []
    for code in codes:
        mean = trends.get(code, 0.05)
        growths = rng.normal(mean, 1.0, size=len(dates))
        for d, g in zip(dates, growths):
            rows.append({"fund_code": code, "nav_date": d, "daily_growth_pct": g})
    return pd.DataFrame(rows)


def _make_vshape_df(pool: list[str], victim_idx: int = 0) -> pd.DataFrame:
    """Build data where victim has trend_passing=True but momentum_score<=0.

    Pattern: uptrend, decline for ~160 days, recovery for ~40 days.
    """
    dates = pd.bdate_range("2018-01-04", "2026-07-30", freq="B")
    n = len(dates)
    rows = []
    for idx, code in enumerate(pool):
        if idx == victim_idx:
            # V-shape that yields trend_pass=True, momentum<=0
            growths = [0.10] * n
            for i in range(n - 200, n - 40):
                growths[i] = -0.08
            for i in range(n - 40, n):
                growths[i] = 0.15
        else:
            growths = [0.08] * n
        for d, g in zip(dates, growths):
            rows.append({"fund_code": code, "nav_date": d, "daily_growth_pct": g})
    return pd.DataFrame(rows)


class TestQuartersHeld:
    def test_same_quarter(self):
        assert _quarters_held(pd.Timestamp("2024-03-15"), pd.Timestamp("2024-03-31")) == 0

    def test_one_quarter_apart(self):
        assert _quarters_held(pd.Timestamp("2024-03-31"), pd.Timestamp("2024-06-30")) == 1

    def test_two_quarters_apart(self):
        assert _quarters_held(pd.Timestamp("2024-03-31"), pd.Timestamp("2024-09-30")) == 2

    def test_cross_year(self):
        assert _quarters_held(pd.Timestamp("2023-12-31"), pd.Timestamp("2024-06-30")) == 2


class TestCrossSectionalRanking:
    def test_rankings_are_computed(self):
        df = _make_growth_df_with_trends(list(SATELLITE_POOL), trends={c: 0.08 for c in SATELLITE_POOL})
        sig = C3LowTurnoverMomentumSignal(df)
        date = pd.Timestamp("2024-06-30")
        evs = sig._cross_sectional_rankings(date)

        ranked_codes = [c for c, e in evs.items() if e["eligible"]]
        ranks = sorted([evs[c]["cross_sectional_rank"] for c in ranked_codes])
        assert len(ranks) == len(set(ranks)), "Ranks must be unique"
        assert ranks == list(range(1, len(ranks) + 1)), "Ranks must be contiguous starting from 1"

    def test_rank_order_by_momentum_score(self):
        df = _make_growth_df_with_trends(list(SATELLITE_POOL), trends={c: 0.10 for c in SATELLITE_POOL})
        sig = C3LowTurnoverMomentumSignal(df)
        date = pd.Timestamp("2024-06-30")
        evs = sig._cross_sectional_rankings(date)

        eligible = [(c, evs[c]["momentum_score"]) for c in SATELLITE_POOL if evs[c]["eligible"]]
        for i in range(len(eligible) - 1):
            c1, s1 = eligible[i]
            c2, s2 = eligible[i + 1]
            r1 = evs[c1]["cross_sectional_rank"]
            r2 = evs[c2]["cross_sectional_rank"]
            if r1 < r2:
                assert s1 >= s2 - 1e-10


class TestRetentionRules:
    def test_retained_when_all_rules_pass_after_min_holding(self):
        df = _make_growth_df_with_trends(list(SATELLITE_POOL), trends={c: 0.10 for c in SATELLITE_POOL})
        sig = C3LowTurnoverMomentumSignal(df)

        q1 = pd.Timestamp("2024-06-30")
        _ = sig.generate_signal(q1)
        sel1 = list(sig.last_selected_satellites or [])
        assert len(sel1) >= 1

        q3 = pd.Timestamp("2024-12-31")
        _ = sig.generate_signal(q3)
        audit = sig.last_signal_audit
        for code in sel1:
            entry = audit["selection_audit"].get(code)
            if entry and entry["action"] == "RETAINED":
                assert "ALL_RULES_PASSED" in entry["reason"]

    def test_emergency_exit_below_ma200(self):
        pool = list(SATELLITE_POOL)
        df = _make_growth_df_with_trends(pool, trends={c: 0.10 for c in pool})
        sig = C3LowTurnoverMomentumSignal(df)

        q1 = pd.Timestamp("2024-06-30")
        _ = sig.generate_signal(q1)
        sel1 = list(sig.last_selected_satellites or [])
        assert len(sel1) >= 1

        victim = sel1[0]
        mask = (df["fund_code"] == victim) & (df["nav_date"] > pd.Timestamp("2024-06-01"))
        df.loc[mask, "daily_growth_pct"] = -2.0

        sig2 = C3LowTurnoverMomentumSignal(df)
        q2 = pd.Timestamp("2024-09-30")
        _ = sig2.generate_signal(q2)
        sig2._selection_dates[victim] = q1

        evs = sig2._cross_sectional_rankings(q2)
        keep, reason = sig2._should_retain(victim, evs[victim], q2)
        assert not keep
        assert "EMERGENCY_EXIT_BELOW_MA200" in reason

    def test_emergency_exit_momentum_non_positive_during_min_holding(self):
        """Audit fix #1: momentum<=0 triggers immediate exit even within min holding,
        when trend still passes (above MA200)."""
        pool = list(SATELLITE_POOL)
        df = _make_vshape_df(pool, victim_idx=0)
        sig = C3LowTurnoverMomentumSignal(df)

        signal_date = pd.Timestamp("2026-07-30")
        evs = sig._cross_sectional_rankings(signal_date)
        victim = pool[0]
        victim_ev = evs[victim]

        # Verify constructed scenario: trend passes, momentum fails
        assert victim_ev["trend_condition"], "Test requires trend to pass (above MA200)"
        assert not victim_ev["momentum_condition"], "Test requires momentum <= 0"

        # Only 1 quarter held — should still exit on momentum failure
        sig._selection_dates[victim] = pd.Timestamp("2026-06-30")
        keep, reason = sig._should_retain(victim, victim_ev, signal_date)
        assert not keep, "Should emergency exit on momentum<=0 even during min holding"
        assert "EMERGENCY_EXIT_MOMENTUM_NON_POSITIVE" in reason


class TestMinHoldingProtection:
    def test_protected_from_rank_only_during_min_holding(self):
        df = _make_growth_df_with_trends(list(SATELLITE_POOL), trends={c: 0.10 for c in SATELLITE_POOL})
        sig = C3LowTurnoverMomentumSignal(df)

        q1 = pd.Timestamp("2024-06-30")
        _ = sig.generate_signal(q1)
        sel1 = list(sig.last_selected_satellites or [])
        assert len(sel1) >= 1

        victim = sel1[0]
        mask = (df["fund_code"] == victim) & (df["nav_date"] > pd.Timestamp("2024-06-01"))
        df.loc[mask, "daily_growth_pct"] -= 0.1

        sig2 = C3LowTurnoverMomentumSignal(df)
        q2 = pd.Timestamp("2024-09-30")
        evs = sig2._cross_sectional_rankings(q2)

        sig2._selection_dates[victim] = q1
        keep, reason = sig2._should_retain(victim, evs[victim], q2)
        if evs[victim]["trend_condition"] and evs[victim]["momentum_condition"]:
            assert keep, "Should be protected from rank-only removal during min holding"


class TestSelectionDateCleanup:
    def test_removed_asset_selection_date_cleared_on_reselect(self):
        """Audit fix #2: re-selected asset gets fresh holding period."""
        pool = list(SATELLITE_POOL)
        df = _make_growth_df_with_trends(pool, trends={c: 0.10 for c in pool})
        sig = C3LowTurnoverMomentumSignal(df)

        q1 = pd.Timestamp("2024-06-30")
        _ = sig.generate_signal(q1)
        sel1 = list(sig.last_selected_satellites or [])
        assert len(sel1) >= 1

        victim = sel1[0]
        mask = (df["fund_code"] == victim) & (df["nav_date"] > pd.Timestamp("2024-09-01"))
        df.loc[mask, "daily_growth_pct"] = -2.5

        sig2 = C3LowTurnoverMomentumSignal(df)
        _ = sig2.generate_signal(q1)
        sig2._selection_dates = dict(sig._selection_dates)
        sig2.last_selected_satellites = list(sel1)
        sig2.last_accepted_target = dict(sig.last_accepted_target)

        q2 = pd.Timestamp("2024-09-30")
        _ = sig2.generate_signal(q2)

        assert victim not in sig2._selection_dates, "Exited asset selection date must be cleared"

        mask2 = (df["fund_code"] == victim) & (df["nav_date"] > pd.Timestamp("2024-12-01"))
        df.loc[mask2, "daily_growth_pct"] = 1.0

        sig3 = C3LowTurnoverMomentumSignal(df)
        _ = sig3.generate_signal(q1)
        sig3._selection_dates = dict(sig2._selection_dates)
        sig3.last_selected_satellites = list(sig2.last_selected_satellites or [])
        sig3.last_accepted_target = dict(sig2.last_accepted_target)

        q3 = pd.Timestamp("2024-12-31")
        _ = sig3.generate_signal(q3)

        if victim in (sig3.last_selected_satellites or []):
            assert sig3._selection_dates[victim] == q3, "Re-selected asset must get current selection date"


class TestMaxSatellites:
    def test_at_most_two_satellites(self):
        df = _make_growth_df_with_trends(list(SATELLITE_POOL), trends={c: 0.12 for c in SATELLITE_POOL})
        sig = C3LowTurnoverMomentumSignal(df)
        date = pd.Timestamp("2024-06-30")
        target = sig.generate_signal(date)

        sat_weights = {k: v for k, v in target.items() if k in SATELLITE_POOL and v > 0}
        assert len(sat_weights) <= MAX_SATELLITES


class TestSatelliteSlotWeight:
    def test_each_selected_at_15_percent(self):
        df = _make_growth_df_with_trends(list(SATELLITE_POOL), trends={c: 0.12 for c in SATELLITE_POOL})
        sig = C3LowTurnoverMomentumSignal(df)
        date = pd.Timestamp("2024-06-30")
        target = sig.generate_signal(date)

        for code in SATELLITE_POOL:
            if target.get(code, 0) > 0:
                assert abs(target[code] - SATELLITE_SLOT_WEIGHT) < 1e-9


class TestBudgetFallback:
    def test_unused_budget_goes_to_001512_and_260102_ratio_2_1(self):
        df = _make_growth_df_with_trends(list(SATELLITE_POOL), trends={c: -0.05 for c in SATELLITE_POOL})
        sig = C3LowTurnoverMomentumSignal(df)
        date = pd.Timestamp("2024-06-30")
        target = sig.generate_signal(date)

        sat_count = sum(1 for c in SATELLITE_POOL if target.get(c, 0) > 0)
        unused = SATELLITE_BUDGET - sat_count * SATELLITE_SLOT_WEIGHT

        if unused > 1e-9:
            expected_001512_extra = unused * 2.0 / 3.0
            expected_260102_extra = unused * 1.0 / 3.0
            assert abs(target[FALLBACK_PRIMARY] - CORE_WEIGHTS[FALLBACK_PRIMARY] - expected_001512_extra) < 1e-9
            assert abs(target[FALLBACK_SECONDARY] - CORE_WEIGHTS[FALLBACK_SECONDARY] - expected_260102_extra) < 1e-9

    def test_fully_used_no_fallback(self):
        df = _make_growth_df_with_trends(list(SATELLITE_POOL), trends={c: 0.12 for c in SATELLITE_POOL})
        sig = C3LowTurnoverMomentumSignal(df)
        date = pd.Timestamp("2024-06-30")
        target = sig.generate_signal(date)

        sat_count = sum(1 for c in SATELLITE_POOL if target.get(c, 0) > 0)
        if sat_count == MAX_SATELLITES:
            assert abs(target[FALLBACK_PRIMARY] - CORE_WEIGHTS[FALLBACK_PRIMARY]) < 1e-9
            assert abs(target[FALLBACK_SECONDARY] - CORE_WEIGHTS[FALLBACK_SECONDARY]) < 1e-9


class TestDriftBasedNoTrade:
    def test_small_drift_no_trade_when_selection_unchanged(self):
        pool = list(SATELLITE_POOL)
        dates = pd.bdate_range("2018-01-04", "2026-07-30", freq="B")
        rows = []
        for code in list(CORE_WEIGHTS.keys()) + pool:
            growths = [0.04] * len(dates)
            for d, g in zip(dates, growths):
                rows.append({"fund_code": code, "nav_date": d, "daily_growth_pct": g})
        df = pd.DataFrame(rows)

        sig = C3LowTurnoverMomentumSignal(df)

        q1 = pd.Timestamp("2024-06-30")
        t1 = sig.generate_signal(q1)
        sel1 = set(sig.last_selected_satellites or [])

        q2 = pd.Timestamp("2024-09-30")
        t2 = sig.generate_signal(q2)
        audit = sig.last_signal_audit

        if sel1 == set(sig.last_selected_satellites or []):
            assert audit["action"] == "NO_TRADE"
            assert t2 == t1
            assert audit["max_drift_deviation_pct_points"] < 8.0

    def test_large_drift_triggers_rebalance(self):
        pool = list(SATELLITE_POOL)
        dates = pd.bdate_range("2018-01-04", "2026-07-30", freq="B")
        rows = []

        for code in list(CORE_WEIGHTS.keys()) + pool:
            growths = [0.04] * len(dates)
            if code == "001512":
                mask_q2 = (dates > pd.Timestamp("2024-06-30")) & (dates <= pd.Timestamp("2024-09-30"))
                for i, d in enumerate(dates):
                    if mask_q2[i]:
                        growths[i] = 1.5
            for d, g in zip(dates, growths):
                rows.append({"fund_code": code, "nav_date": d, "daily_growth_pct": g})
        df = pd.DataFrame(rows)

        sig = C3LowTurnoverMomentumSignal(df)

        q1 = pd.Timestamp("2024-06-30")
        _ = sig.generate_signal(q1)
        sel1 = set(sig.last_selected_satellites or [])

        q2 = pd.Timestamp("2024-09-30")
        _ = sig.generate_signal(q2)
        audit = sig.last_signal_audit

        sel2 = set(sig.last_selected_satellites or [])
        if sel1 == sel2:
            assert audit["action"] == "TRIGGER_REBALANCE"
            assert "DRIFT_THRESHOLD_TRIGGER" in audit["reason"]
            assert audit["max_drift_deviation_pct_points"] >= 8.0

    def test_drift_audit_field_records_decision_value(self):
        pool = list(SATELLITE_POOL)
        dates = pd.bdate_range("2018-01-04", "2026-07-30", freq="B")
        rows = []
        for code in list(CORE_WEIGHTS.keys()) + pool:
            growths = [0.04] * len(dates)
            for d, g in zip(dates, growths):
                rows.append({"fund_code": code, "nav_date": d, "daily_growth_pct": g})
        df = pd.DataFrame(rows)

        sig = C3LowTurnoverMomentumSignal(df)

        q1 = pd.Timestamp("2024-06-30")
        _ = sig.generate_signal(q1)

        q2 = pd.Timestamp("2024-09-30")
        _ = sig.generate_signal(q2)
        audit = sig.last_signal_audit

        assert "max_drift_deviation_pct_points" in audit
        assert isinstance(audit["max_drift_deviation_pct_points"], float)


class TestNoLookAheadBias:
    def test_future_data_does_not_affect_past_decision(self):
        df = _make_growth_df_with_trends(list(SATELLITE_POOL), trends={c: 0.10 for c in SATELLITE_POOL})

        sig1 = C3LowTurnoverMomentumSignal(df)
        date = pd.Timestamp("2024-06-30")
        t1 = sig1.generate_signal(date)

        df2 = df.copy()
        mask = df2["nav_date"] > pd.Timestamp("2024-07-01")
        df2.loc[mask, "daily_growth_pct"] += 5.0

        sig2 = C3LowTurnoverMomentumSignal(df2)
        t2 = sig2.generate_signal(date)

        assert t1 == t2


class TestWeightSum:
    def test_weights_sum_to_one(self):
        df = _make_growth_df_with_trends(list(SATELLITE_POOL), trends={c: 0.10 for c in SATELLITE_POOL})
        sig = C3LowTurnoverMomentumSignal(df)
        date = pd.Timestamp("2024-06-30")
        target = sig.generate_signal(date)

        total = sum(target.values())
        assert abs(total - 1.0) < 1e-9


class TestInitialBuild:
    def test_initial_build_selects_top2(self):
        df = _make_growth_df_with_trends(list(SATELLITE_POOL), trends={c: 0.12 for c in SATELLITE_POOL})
        sig = C3LowTurnoverMomentumSignal(df)
        date = pd.Timestamp("2024-06-30")
        _ = sig.generate_signal(date)
        audit = sig.last_signal_audit

        assert audit["action"] == "TRIGGER_REBALANCE"
        assert audit["reason"] == "INITIAL_BUILD"


class TestFirstBuildRequired:
    def test_first_call_is_always_rebalance(self):
        df = _make_growth_df_with_trends(list(SATELLITE_POOL))
        sig = C3LowTurnoverMomentumSignal(df)
        date = pd.Timestamp("2024-06-30")
        _ = sig.generate_signal(date)
        audit = sig.last_signal_audit
        assert audit["action"] == "TRIGGER_REBALANCE"


class TestQuarterEndObservation:
    def test_signals_generated_at_quarter_ends(self):
        df = _make_growth_df_with_trends(list(SATELLITE_POOL))
        sig = C3LowTurnoverMomentumSignal(df)

        for quarter_end in ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]:
            date = pd.Timestamp(quarter_end)
            sig.generate_signal(date)

        assert len(sig.audit_rows) == 4


class TestEmptySignalAction:
    def test_no_eligible_satellites_keeps_core(self):
        df = _make_growth_df_with_trends(list(SATELLITE_POOL), trends={c: -0.1 for c in SATELLITE_POOL})
        sig = C3LowTurnoverMomentumSignal(df)
        date = pd.Timestamp("2024-06-30")
        target = sig.generate_signal(date)

        sat_selected = [c for c in SATELLITE_POOL if target.get(c, 0) > 0]
        assert len(sat_selected) == 0
        core_keys = set(CORE_WEIGHTS.keys())
        assert all(k in target for k in core_keys)


class TestReturnType:
    def test_generate_signal_returns_dict(self):
        df = _make_growth_df_with_trends(list(SATELLITE_POOL))
        sig = C3LowTurnoverMomentumSignal(df)
        result = sig.generate_signal(pd.Timestamp("2024-06-30"))
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, float)


class TestNoQuarterEndFunction:
    def test_no_broken_quarter_end(self):
        from otf_rotation import c3_low_turnover_momentum as mod
        assert not hasattr(mod, "_quarter_end"), "Broken _quarter_end should be removed"
