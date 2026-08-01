"""P1-2: NAV availability lag model tests and future-data invariance acceptance.

Every rotation signal must ignore NAV rows dated after the conservative
availability cutoff at each signal date.  Modifying data published after the
signal date must never change the historical signal, the product selection or
the covariance estimate.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from otf_rotation.nav_availability import (
    MODEL_ID,
    available_as_of,
    availability_facts,
    frame_as_of,
    lag_for,
    series_as_of,
)
from otf_rotation.core_satellite_momentum import (
    CORE_WEIGHTS,
    SATELLITE_POOL,
    CoreSatelliteMomentumSignal,
)
from otf_rotation.c3_low_turnover_momentum import (
    C3LowTurnoverMomentumSignal,
    SATELLITE_POOL as C3_SATELLITE_POOL,
)
from otf_rotation.candidate_strategies import B2LTSignal, D1Signal

QDII = "050025"
DOMESTIC = "160706"


def _dates(n: int = 260, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _growth_frame(
    dates: pd.DatetimeIndex, rates: dict[str, float], unit_nav: float = 1.0
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for code, rate in rates.items():
        for date in dates:
            rows.append(
                {
                    "fund_code": code,
                    "nav_date": date,
                    "daily_growth_pct": rate,
                    "unit_nav": unit_nav,
                }
            )
    return pd.DataFrame(rows)


def _nav_frame(
    dates: pd.DatetimeIndex,
    codes: list[str],
    values: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for code in codes:
        for i, date in enumerate(dates):
            value = (values or {}).get(code, [1.0] * len(dates))[i]
            rows.append({"fund_code": code, "nav_date": date, "unit_nav": value})
    return pd.DataFrame(rows)


class TestLagModel:
    def test_domestic_lag_is_one(self):
        assert lag_for(DOMESTIC) == 1

    def test_qdii_lag_is_two(self):
        assert lag_for("050025") == 2
        assert lag_for("000071") == 2

    def test_code_normalisation(self):
        assert lag_for("50025") == 2
        assert lag_for(" 160706 ") == 1

    def test_custom_qdii_set_overrides_defaults(self):
        assert lag_for("050025", qdii_codes=["050025"]) == 2
        assert lag_for("000071", qdii_codes=["050025"]) == 1

    def test_facts_payload_describes_model(self):
        facts = availability_facts()
        assert facts["model_id"] == MODEL_ID
        assert facts["conservative_availability_lag_applied"] is True
        assert facts["publication_timestamp_available"] is False
        assert "050025" in facts["qdii_codes"]


class TestAvailableAsOf:
    def test_t_plus_one_cutoff(self):
        dates = _dates(10)
        assert available_as_of(dates, dates[-1], 1) == dates[-2]

    def test_t_plus_two_cutoff(self):
        dates = _dates(10)
        assert available_as_of(dates, dates[-1], 2) == dates[-3]

    def test_mid_history_cutoff(self):
        dates = _dates(10)
        assert available_as_of(dates, dates[5], 1) == dates[4]

    def test_lag_beyond_history_falls_back_to_first_day(self):
        dates = _dates(3)
        assert available_as_of(dates, dates[2], 2) == dates[0]

    def test_non_trading_signal_day_uses_prior_trading_day(self):
        dates = _dates(10)
        saturday = pd.Timestamp("2020-01-11")
        assert available_as_of(dates, saturday, 1) == dates[6]

    def test_frame_as_of_filters_future_rows(self):
        dates = _dates(6)
        frame = _growth_frame(dates, {DOMESTIC: 0.01})
        cutoff = available_as_of(dates, dates[-1], 1)
        filtered = frame_as_of(frame, DOMESTIC, cutoff)
        assert filtered["nav_date"].max() <= cutoff
        assert len(filtered) == 5

    def test_series_as_of_stops_at_cutoff(self):
        dates = _dates(6)
        index = pd.DataFrame(
            {DOMESTIC: np.arange(1.0, 7.0)}, index=dates
        )
        cutoff = available_as_of(dates, dates[-1], 1)
        series = series_as_of(index, DOMESTIC, cutoff)
        assert series.index[-1] == cutoff
        assert len(series) == 5


class TestFutureDataDoesNotChangeHistoricalSignals:
    """P1-2 acceptance: NAV published after the signal date must not move it."""

    def test_c1_signal_ignores_future_navs(self):
        dates = _dates(260)
        rates = {**{c: 0.01 for c in CORE_WEIGHTS}, **{c: 0.05 for c in SATELLITE_POOL}}
        frame = _growth_frame(dates, rates)
        signal_date = dates[-1]
        before = CoreSatelliteMomentumSignal(
            frame, trading_dates=dates, qdii_codes=[QDII]
        )(signal_date)

        future = _growth_frame(
            pd.bdate_range(signal_date + pd.Timedelta(days=1), periods=5),
            {**{c: 50.0 for c in CORE_WEIGHTS}, **{c: 50.0 for c in SATELLITE_POOL}},
        )
        after = CoreSatelliteMomentumSignal(
            pd.concat([frame, future], ignore_index=True),
            trading_dates=dates,
            qdii_codes=[QDII],
        )(signal_date)
        assert before == pytest.approx(after)

    def test_c3_signal_ignores_future_navs(self):
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2018-01-04", "2026-07-30", freq="B")
        codes = list(C3_SATELLITE_POOL)
        rows: list[dict[str, object]] = []
        for code in codes:
            growths = rng.normal(0.05, 1.0, size=len(dates))
            for d, g in zip(dates, growths):
                rows.append({"fund_code": code, "nav_date": d, "daily_growth_pct": g})
        frame = pd.DataFrame(rows)
        signal_date = pd.Timestamp("2024-06-30")

        before = C3LowTurnoverMomentumSignal(
            frame, trading_dates=dates, qdii_codes=[QDII]
        ).generate_signal(signal_date)

        future_dates = pd.bdate_range(
            signal_date + pd.Timedelta(days=1), periods=5
        )
        future_rows: list[dict[str, object]] = []
        for code in codes:
            for d in future_dates:
                future_rows.append(
                    {"fund_code": code, "nav_date": d, "daily_growth_pct": 50.0}
                )
        combined = pd.concat([frame, pd.DataFrame(future_rows)], ignore_index=True)
        after = C3LowTurnoverMomentumSignal(
            combined, trading_dates=dates, qdii_codes=[QDII]
        ).generate_signal(signal_date)
        assert before == pytest.approx(after)

    def test_d1_signal_ignores_future_navs(self):
        dates = _dates(205)
        rows = _nav_frame(dates, ["000001", "260102"])
        rules = pd.DataFrame(
            [
                {"fund_code": c, "rule_status": "DISTRIBUTOR_VERIFIED",
                 "effective_from": "", "effective_to": ""}
                for c in ["000001", "260102"]
            ]
        )
        mapping = pd.DataFrame(
            [
                {"fund_code": c, "mapping_confidence": "HIGH",
                 "review_status": "APPROVED", "effective_from": "", "effective_to": ""}
                for c in ["000001", "260102"]
            ]
        )
        products = {
            "000001": {"asset_class": "domestic_equity", "budget": 0.90, "trend_filter": True},
            "260102": {"asset_class": "cash", "budget": 0.10, "trend_filter": False},
        }
        budgets = {"domestic_equity": 0.90, "cash": 0.10}
        signal_date = dates[-1]

        def build(frames: list[pd.DataFrame]) -> D1Signal:
            return D1Signal(
                pd.concat(frames, ignore_index=True), rules, mapping, products, budgets,
                fallback_fund="260102", ma_days=200, threshold=0.05,
                trading_dates=dates, qdii_codes=[QDII],
            )

        before = build([rows])(signal_date)

        future_rows = _nav_frame(
            pd.bdate_range(signal_date + pd.Timedelta(days=1), periods=3),
            ["000001", "260102"],
            {"000001": [100.0] * 3, "260102": [100.0] * 3},
        )
        after = build([rows, future_rows])(signal_date)
        assert before == pytest.approx(after)

    def test_b2lt_signal_ignores_future_navs(self):
        dates = _dates(60)
        codes = list(B2LTSignal.assets)
        frame = _nav_frame(dates, codes)
        signal_date = dates[-1]

        before = B2LTSignal(
            frame, threshold=0.05, trading_dates=dates, qdii_codes=[QDII]
        )(signal_date)

        future_rows = _nav_frame(
            pd.bdate_range(signal_date + pd.Timedelta(days=1), periods=3),
            codes,
            {c: [100.0] * 3 for c in codes},
        )
        combined = pd.concat([frame, future_rows], ignore_index=True)
        after = B2LTSignal(
            combined, threshold=0.05, trading_dates=dates, qdii_codes=[QDII]
        )(signal_date)
        assert before == pytest.approx(after)
