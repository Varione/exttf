"""Tests for Phase A holdings-based explanatory attribution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from otf_rotation.strategy_attribution import (
    CASH_CODE,
    EPS,
    TOTAL_CODE,
    aggregate_attribution,
    compute_daily_attribution,
    validate_frozen_inputs,
)


def _frames():
    dates = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"])
    account = pd.DataFrame({
        "date": dates, "equity": [100.0, 101.0, 102.0],
        "gross_return": [0.0, 0.01, 0.0099009900990099],
        "daily_return": [0.0, 0.009, 0.0089009900990099],
        "transaction_cost": [0.0, 0.001, 0.001],
        "total_fee_amount": [0.0, 0.1, 0.1],
    })
    weights = pd.DataFrame({"date": dates, "000001": [0.0, 0.5, 0.6], "000002": [0.0, 0.0, 0.0]})
    returns = pd.DataFrame({
        "date": dates, "fund_code": ["000001", "000001", "000001"],
        "fund_return": [0.0, 0.02, -0.01], "nav_available": [True, True, True],
    })
    meta = pd.DataFrame({"fund_code": ["000001", "000002"], "fund_name": ["A", "B"], "asset_class": ["equity", "bond"]})
    fees = pd.DataFrame({"date": dates, "daily_total_fee_amount": [0.0, 0.1, 0.1], "total_fee_delta": [0.0, 0.0, 0.0]})
    return account, weights, returns, meta, fees


def test_uses_lagged_weights_not_current_weights():
    account, weights, returns, meta, fees = _frames()
    funds, totals = compute_daily_attribution(account, weights, returns, meta, fees, "TEST", {"000001"}, set())
    row = funds[(funds.date == "2021-01-05") & (funds.fund_code == "000001")].iloc[0]
    assert row.lagged_weight == 0.0
    assert row.actual_weight == 0.5
    assert row.fund_contribution == 0.0
    row2 = funds[(funds.date == "2021-01-06") & (funds.fund_code == "000001")].iloc[0]
    assert row2.lagged_weight == 0.5
    assert row2.fund_contribution == pytest.approx(-0.005)


def test_daily_accounting_identity_and_fee_reconciliation():
    account, weights, returns, meta, fees = _frames()
    funds, totals = compute_daily_attribution(account, weights, returns, meta, fees, "TEST", {"000001"}, set())
    assert (totals.accounting_error.abs() <= EPS).all()
    assert totals.accounting_residual.sum() == pytest.approx(0.0249009900990099)
    assert (totals.net_reconciliation_error.abs() <= EPS).all()


def test_future_return_change_does_not_change_prior_attribution():
    account, weights, returns, meta, fees = _frames()
    a, _ = compute_daily_attribution(account, weights, returns, meta, fees, "TEST", {"000001"}, set())
    changed = returns.copy()
    changed.loc[changed.date == pd.Timestamp("2021-01-06"), "fund_return"] = 9.0
    b, _ = compute_daily_attribution(account, weights, changed, meta, fees, "TEST", {"000001"}, set())
    assert a[a.date < "2021-01-06"].fund_contribution.tolist() == b[b.date < "2021-01-06"].fund_contribution.tolist()


def test_aggregate_contains_cash_and_core_satellite():
    account, weights, returns, meta, fees = _frames()
    funds, totals = compute_daily_attribution(account, weights, returns, meta, fees, "TEST", {"000001"}, set())
    fund, sleeve, period = aggregate_attribution(funds, totals, meta, "TEST")
    assert "core" in set(fund.sleeve)
    assert CASH_CODE in set(fund.fund_code)
    assert "full" in set(sleeve.period)
    full = period[period.period == "full"]
    assert abs(float(full.net_contribution_approx.sum()) - float(fund.net_contribution_approx.sum())) < 1e-12


def test_hash_gate_blocks_mismatch(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "input_hashes.json").write_text(json.dumps({"db_sha256": "a", "rules_sha256": "b", "mapping_sha256": "c", "config_sha256": "d"}))
    from otf_rotation.strategy_attribution import AttributionSpec
    spec = AttributionSpec("A", "A", bundle, frozenset(), frozenset(), tmp_path / "missing.db")
    result = validate_frozen_inputs([spec], tmp_path)
    assert result["passed"] is False
    assert any("missing_input_file" in item for item in result["mismatches"])


def test_event_overlap_check_is_structurally_supported():
    # This test documents the interval contract used by the runner: an event
    # interval ends before the next event's start.
    starts = pd.to_datetime(["2021-01-05", "2021-02-05"])
    ends = pd.to_datetime(["2021-02-04", "2021-03-01"])
    assert bool((starts[1:].to_numpy() <= ends[:-1].to_numpy()).any()) is False
