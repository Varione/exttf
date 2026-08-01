"""Frozen C2 low-turnover core/satellite signal.

The C2 study intentionally differs from C1 in three ways: it observes only
twice per year, retains an eligible incumbent satellite, and does not apply
continuous volatility scaling.  All signal inputs are restricted by a
conservative product-specific availability lag before they are evaluated.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from otf_rotation.core_satellite_momentum import (
    _natural_drift_from_index,
    _normalise_code,
    _normalise_growth_frame,
    build_total_return_index,
)


C2_CORE_WEIGHTS: dict[str, float] = {
    "001512": 0.25,
    "000148": 0.20,
    "000218": 0.15,
    "260102": 0.15,
}
C2_SATELLITE_POOL: tuple[str, ...] = (
    "160706",
    "000008",
    "007466",
    "050021",
    "050025",
    "000071",
)
C2_QDII_CODES: frozenset[str] = frozenset({"050025", "000071"})
C2_DOMESTIC_LAG = 1
C2_QDII_LAG = 2
C2_MIN_OBSERVATIONS = 253
C2_SHORT_DAYS = 126
C2_LONG_DAYS = 252
C2_MA_DAYS = 200
C2_MAX_SATELLITES = 3
C2_SATELLITE_BUDGET = 0.25
C2_SLOT_WEIGHT = C2_SATELLITE_BUDGET / C2_MAX_SATELLITES
C2_DRIFT_THRESHOLD = 0.075


def _as_date(value: object) -> pd.Timestamp:
    date = pd.Timestamp(value)
    if pd.isna(date):
        raise ValueError("c2_invalid_date")
    return date.normalize()


def _select_with_retention(
    evaluations: Mapping[str, Mapping[str, Any]],
    previous: list[str] | tuple[str, ...] | None,
    maximum: int = C2_MAX_SATELLITES,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Keep eligible incumbents first, then fill vacant slots by rank."""
    eligible = [
        code
        for code, row in evaluations.items()
        if bool(row.get("eligible"))
        and row.get("momentum_score") is not None
        and np.isfinite(float(row["momentum_score"]))
    ]
    ranked = sorted(
        eligible,
        key=lambda code: (-float(evaluations[code]["momentum_score"]), str(code)),
    )
    ranks = {code: index + 1 for index, code in enumerate(ranked)}
    previous_set = {str(code) for code in (previous or [])}
    selected: list[str] = []
    audit: dict[str, dict[str, Any]] = {}
    for code in ranked:
        retained = code in previous_set
        if retained and len(selected) < maximum:
            selected.append(code)
        audit[code] = {
            "rank": ranks[code],
            "eligible": True,
            "retained_existing_while_eligible": retained,
            "selected": False,
        }
    for code in ranked:
        if len(selected) >= maximum:
            break
        if code not in selected:
            selected.append(code)
    for code, row in evaluations.items():
        audit.setdefault(
            code,
            {
                "rank": ranks.get(code),
                "eligible": bool(row.get("eligible")),
                "retained_existing_while_eligible": False,
            },
        )
        audit[code]["selected"] = code in selected
    return selected, audit


class LowTurnoverCoreSatelliteSignal:
    """C2 semiannual signal with product-specific conservative availability."""

    def __init__(
        self,
        growth_df: pd.DataFrame,
        *,
        trading_dates: pd.DatetimeIndex | list[pd.Timestamp],
        core_weights: Mapping[str, float] = C2_CORE_WEIGHTS,
        satellite_pool: tuple[str, ...] = C2_SATELLITE_POOL,
        qdii_codes: frozenset[str] = C2_QDII_CODES,
        min_observations: int = C2_MIN_OBSERVATIONS,
    ) -> None:
        self.growth_df = _normalise_growth_frame(growth_df)
        self.total_return_index = build_total_return_index(self.growth_df)
        self.trading_dates = pd.DatetimeIndex(pd.to_datetime(trading_dates)).sort_values().unique()
        if self.trading_dates.empty:
            raise ValueError("c2_empty_trading_calendar")
        self.core_weights = {
            _normalise_code(code): float(weight) for code, weight in core_weights.items()
        }
        self.satellite_pool = tuple(_normalise_code(code) for code in satellite_pool)
        self.qdii_codes = frozenset(_normalise_code(code) for code in qdii_codes)
        self.min_observations = int(min_observations)
        self.reset()

    def reset(self) -> None:
        self.last_accepted_target: dict[str, float] | None = None
        self.last_reference_index: dict[str, float] | None = None
        self.last_selected_satellites: list[str] | None = None
        self.last_accepted_date: pd.Timestamp | None = None
        self.last_signal_audit: dict[str, Any] = {}
        self.audit_rows: list[dict[str, Any]] = []

    def _lag_for(self, code: str) -> int:
        return C2_QDII_LAG if _normalise_code(code) in self.qdii_codes else C2_DOMESTIC_LAG

    def available_date(self, code: str, signal_date: pd.Timestamp) -> pd.Timestamp:
        """Return the last China trading day conservatively available for code."""
        date = _as_date(signal_date)
        position = int(self.trading_dates.searchsorted(date, side="right") - 1)
        position = max(0, position - self._lag_for(code))
        return pd.Timestamp(self.trading_dates[position])

    def _series_as_of(self, code: str, signal_date: pd.Timestamp) -> tuple[pd.Series, pd.Timestamp]:
        as_of = self.available_date(code, signal_date)
        if code not in self.total_return_index.columns:
            return pd.Series(dtype=float), as_of
        series = self.total_return_index.loc[:as_of, code].dropna()
        return series, as_of

    def _evaluate(self, code: str, signal_date: pd.Timestamp) -> dict[str, Any]:
        code = _normalise_code(code)
        series, as_of = self._series_as_of(code, signal_date)
        observations = int(
            self.growth_df[
                (self.growth_df["fund_code"] == code)
                & (self.growth_df["nav_date"] <= as_of)
            ].shape[0]
        )
        row: dict[str, Any] = {
            "fund_code": code,
            "signal_date": _as_date(signal_date),
            "available_as_of": as_of,
            "availability_lag_trading_days": self._lag_for(code),
            "published_observations": observations,
            "latest_published_date": series.index.max() if not series.empty else None,
            "latest_total_return_index": None,
            "total_return_126": None,
            "total_return_252": None,
            "ma200": None,
            "momentum_score": None,
            "trend_condition": False,
            "momentum_condition": False,
            "eligible": False,
            "reason": "",
        }
        if observations < self.min_observations or len(series) < self.min_observations:
            row["reason"] = f"INSUFFICIENT_AVAILABLE_OBSERVATIONS_{self.min_observations}"
            return row
        current = float(series.iloc[-1])
        ret126 = current / float(series.iloc[-(C2_SHORT_DAYS + 1)]) - 1.0
        ret252 = current / float(series.iloc[-(C2_LONG_DAYS + 1)]) - 1.0
        ma200 = float(series.tail(C2_MA_DAYS).mean())
        score = 0.5 * ret126 + 0.5 * ret252
        trend = current >= ma200 - 1e-15
        positive = score > 0.0
        reasons: list[str] = []
        if not trend:
            reasons.append("BELOW_MA200")
        if not positive:
            reasons.append("MOMENTUM_NON_POSITIVE")
        row.update(
            {
                "latest_total_return_index": current,
                "total_return_126": ret126,
                "total_return_252": ret252,
                "ma200": ma200,
                "momentum_score": score,
                "trend_condition": trend,
                "momentum_condition": positive,
                "eligible": bool(trend and positive),
                "reason": ";".join(reasons) if reasons else "ELIGIBLE",
            }
        )
        return row

    def _current_indices(self, codes: list[str], signal_date: pd.Timestamp) -> dict[str, float]:
        values: dict[str, float] = {}
        for code in codes:
            series, _ = self._series_as_of(code, signal_date)
            if not series.empty and np.isfinite(float(series.iloc[-1])):
                values[code] = float(series.iloc[-1])
        return values

    def _proposed_target(
        self, signal_date: pd.Timestamp, evaluations: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, float], dict[str, Any]]:
        selected, selection_audit = _select_with_retention(
            evaluations, self.last_selected_satellites, C2_MAX_SATELLITES
        )
        # The frozen C2 rule does not continuously scale risk.  Any unused
        # satellite slots are explicitly sent to the cash fund.
        target = dict(self.core_weights)
        for code in selected:
            target[code] = C2_SLOT_WEIGHT
        used_satellite = C2_SLOT_WEIGHT * len(selected)
        cash_transfer = max(0.0, C2_SATELLITE_BUDGET - used_satellite)
        target["260102"] = target.get("260102", 0.0) + cash_transfer
        return target, {
            "selected_satellites": selected,
            "selection_audit": selection_audit,
            "base_satellite_weights": {code: C2_SLOT_WEIGHT for code in selected},
            "usable_satellite_weights": {code: C2_SLOT_WEIGHT for code in selected},
            "cash_transfer": cash_transfer,
            "cash_transfer_reasons": {
                code: "UNUSED_SATELLITE_SLOT_TO_CASH"
                for code in self.satellite_pool
                if code not in selected
            },
            "covariance": {"coverage_passed": False, "reason": "C2_VOL_SCALING_DISABLED"},
            "covariance_coverage": 0,
            "covariance_start": None,
            "covariance_end": None,
            "predicted_volatility_pct": None,
            "satellite_scale": 1.0 if selected else 0.0,
            "scale_audit": {"bisection_applied": False, "reason": "C2_VOL_SCALING_DISABLED"},
            "target_weights": target,
        }

    def __call__(self, date: pd.Timestamp) -> dict[str, float]:
        signal_date = _as_date(date)
        previous = list(self.last_selected_satellites or [])
        evaluations = {code: self._evaluate(code, signal_date) for code in self.satellite_pool}
        proposed_target, proposed = self._proposed_target(signal_date, evaluations)
        selected = list(proposed["selected_satellites"])
        selection_changed = self.last_selected_satellites is not None and set(selected) != set(previous)

        current_indices = self._current_indices(
            list(set(proposed_target) | set(self.last_accepted_target or {})), signal_date
        )
        drift: dict[str, float] = {}
        max_deviation = 0.0
        if self.last_accepted_target is None:
            rebalance = True
            decision = "FIRST_BUILD"
        elif not selected and not previous:
            rebalance = False
            decision = "NO_SIGNAL_KEEP_HOLDINGS"
        elif selection_changed:
            rebalance = True
            decision = "SATELLITE_SELECTION_SET_CHANGED"
        else:
            try:
                drift = _natural_drift_from_index(
                    self.last_accepted_target, self.last_reference_index or {}, current_indices
                )
                union = set(drift) | set(proposed_target)
                max_deviation = max(
                    abs(float(drift.get(code, 0.0)) - float(proposed_target.get(code, 0.0)))
                    for code in union
                ) if union else 0.0
            except ValueError:
                rebalance = False
                decision = "NO_REBALANCE_TOTAL_RETURN_INDEX_UNAVAILABLE"
            else:
                rebalance = max_deviation >= C2_DRIFT_THRESHOLD - 1e-12
                decision = "DRIFT_THRESHOLD_TRIGGER" if rebalance else "NO_REBALANCE_WITHIN_7_5PP_DRIFT"

        accepted_target = dict(proposed_target) if rebalance else {}
        if rebalance:
            self.last_accepted_target = dict(proposed_target)
            self.last_reference_index = {
                code: current_indices[code]
                for code in proposed_target
                if code in current_indices
            }
            self.last_selected_satellites = selected
            self.last_accepted_date = signal_date

        audit = {
            "strategy": "C2_LOW_TURNOVER_CORE_SATELLITE",
            "signal_date": signal_date,
            "rebalance": bool(rebalance),
            "decision": decision,
            "reason": decision,
            "selection_changed": bool(selection_changed),
            "previous_selected_satellites": previous,
            "selected_satellites": selected,
            "ranked_satellites": [
                code for code, row in sorted(
                    evaluations.items(),
                    key=lambda item: (
                        -float(item[1]["momentum_score"])
                        if item[1]["momentum_score"] is not None else np.inf,
                        item[0],
                    ),
                )
            ],
            "evaluations": list(evaluations.values()),
            "evaluations_by_code": evaluations,
            "buffer_audit": proposed["selection_audit"],
            "base_satellite_weights": proposed["base_satellite_weights"],
            "usable_satellite_weights": proposed["usable_satellite_weights"],
            "covariance": proposed["covariance"],
            "covariance_coverage": 0,
            "covariance_start": None,
            "covariance_end": None,
            "predicted_volatility_pct": None,
            "satellite_scale": proposed["satellite_scale"],
            "scale_audit": proposed["scale_audit"],
            "drift_weights": drift,
            "max_drift_deviation_pct_points": max_deviation * 100.0,
            "index_available_for_drift": bool(current_indices),
            "proposed_target_weights": proposed_target,
            "target_weights": accepted_target,
            "cash_transfer": proposed["cash_transfer"],
            "cash_transfer_reasons": proposed["cash_transfer_reasons"],
            "accepted_target_date": self.last_accepted_date,
            "parameters": {
                "min_available_observations": self.min_observations,
                "momentum_short_days": C2_SHORT_DAYS,
                "momentum_long_days": C2_LONG_DAYS,
                "trend_ma_days": C2_MA_DAYS,
                "maximum_satellites": C2_MAX_SATELLITES,
                "satellite_budget": C2_SATELLITE_BUDGET,
                "slot_weight": C2_SLOT_WEIGHT,
                "domestic_availability_lag": C2_DOMESTIC_LAG,
                "qdii_availability_lag": C2_QDII_LAG,
                "drift_threshold_pct_points": C2_DRIFT_THRESHOLD * 100.0,
                "continuous_volatility_scaling": False,
            },
        }
        self.last_signal_audit = audit
        self.audit_rows.append(dict(audit))
        return accepted_target
