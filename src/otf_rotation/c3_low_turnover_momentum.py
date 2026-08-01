"""Frozen C3 low-turnover core/satellite momentum signal.

Key differences from C1/C2:
- Quarterly cross-sectional ranking only.
- Existing satellites retained if trend_passing AND positive_momentum AND rank<=3.
- Minimum holding of 2 quarters; emergency exit if trend fails OR momentum<=0.
- Max 2 satellite slots at 15%% each; no volatility scaling.
- Unused satellite budget allocated to 001512/260102 at ratio 2:1.
- Rebalance triggers only when satellite selection set changes.
- Natural drift <8pp AND selection unchanged -> NO_TRADE (C2-style).
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


CORE_WEIGHTS: dict[str, float] = {
    "001512": 0.20,
    "000148": 0.15,
    "000218": 0.20,
    "260102": 0.15,
}
SATELLITE_POOL: tuple[str, ...] = (
    "160706",
    "000008",
    "007466",
    "050021",
    "050025",
    "000071",
)
SATELLITE_BUDGET = 0.30
MAX_SATELLITES = 2
SATELLITE_SLOT_WEIGHT = 0.15
MIN_PUBLISHED_OBSERVATIONS = 253
MOMENTUM_SHORT_DAYS = 126
MOMENTUM_LONG_DAYS = 252
TREND_MA_DAYS = 200
MIN_HOLDING_QUARTERS = 2
DRIFT_THRESHOLD = 0.08
FALLBACK_PRIMARY = "001512"
FALLBACK_SECONDARY = "260102"
FALLBACK_RATIO_PRIMARY = 2.0
FALLBACK_RATIO_SECONDARY = 1.0


def _quarters_held(selection_date: pd.Timestamp, current_date: pd.Timestamp) -> int:
    """Count completed calendar quarters between two dates."""
    if selection_date is None or current_date is None:
        return 0
    start_q = ((selection_date.month - 1) // 3 + 1) + (selection_date.year - 2000) * 4
    end_q = ((current_date.month - 1) // 3 + 1) + (current_date.year - 2000) * 4
    return max(0, int(end_q - start_q))


class C3LowTurnoverMomentumSignal:

    core_weights = dict(CORE_WEIGHTS)
    satellite_pool = SATELLITE_POOL

    def __init__(
        self,
        growth_df: pd.DataFrame,
        *,
        core_weights: Mapping[str, float] = CORE_WEIGHTS,
        satellite_pool: tuple[str, ...] = SATELLITE_POOL,
        min_observations: int = MIN_PUBLISHED_OBSERVATIONS,
        short_days: int = MOMENTUM_SHORT_DAYS,
        long_days: int = MOMENTUM_LONG_DAYS,
        ma_days: int = TREND_MA_DAYS,
        satellite_budget: float = SATELLITE_BUDGET,
        max_satellites: int = MAX_SATELLITES,
        satellite_slot_weight: float = SATELLITE_SLOT_WEIGHT,
        min_holding_quarters: int = MIN_HOLDING_QUARTERS,
        drift_threshold: float = DRIFT_THRESHOLD,
        trading_dates: pd.DatetimeIndex | None = None,
        qdii_codes: list[str] | None = None,
    ):
        self.growth_df = _normalise_growth_frame(growth_df)
        self.total_return_index = build_total_return_index(self.growth_df)
        self.trading_dates = trading_dates
        self.qdii_codes = list(qdii_codes or [])
        self.core_weights = {
            _normalise_code(code): float(weight) for code, weight in core_weights.items()
        }
        self.satellite_pool = tuple(_normalise_code(code) for code in satellite_pool)
        self.min_observations = int(min_observations)
        self.short_days = int(short_days)
        self.long_days = int(long_days)
        self.ma_days = int(ma_days)
        self.satellite_budget = float(satellite_budget)
        self.max_satellites = int(max_satellites)
        self.satellite_slot_weight = float(satellite_slot_weight)
        self.min_holding_quarters = int(min_holding_quarters)
        self.drift_threshold = float(drift_threshold)
        self.reset()

    def reset(self) -> None:
        self.last_accepted_target: dict[str, float] | None = None
        self.last_reference_index: dict[str, float] | None = None
        self.last_selected_satellites: list[str] | None = None
        self.last_accepted_date: pd.Timestamp | None = None
        self._selection_dates: dict[str, pd.Timestamp] = {}
        self.last_signal_audit: dict[str, Any] = {}
        self.audit_rows: list[dict[str, Any]] = []

    def evaluate_one(self, code: str, signal_date: pd.Timestamp) -> dict[str, Any]:
        code = _normalise_code(code)
        row: dict[str, Any] = {
            "fund_code": code,
            "signal_date": signal_date,
            "latest_total_return_index": None,
            "total_return_126": None,
            "total_return_252": None,
            "ma200": None,
            "momentum_score": None,
            "trend_condition": False,
            "momentum_condition": False,
            "eligible": False,
            "cross_sectional_rank": None,
            "reason": "",
        }
        if code not in self.total_return_index.columns:
            row["reason"] = "NOT_IN_TOTAL_RETURN_INDEX"
            return row

        cutoff = pd.Timestamp(signal_date)
        if self.trading_dates is not None:
            from otf_rotation.nav_availability import available_as_of, lag_for

            cutoff = available_as_of(
                self.trading_dates, pd.Timestamp(signal_date), lag_for(code, self.qdii_codes or [])
            )
        series = self.total_return_index.loc[:cutoff, code].dropna()
        if len(series) < self.min_observations:
            row["reason"] = f"INSUFFICIENT_PUBLISHED_OBSERVATIONS_{self.min_observations}"
            return row

        current = float(series.iloc[-1])
        ret126 = current / float(series.iloc[-(self.short_days + 1)]) - 1.0
        ret252 = current / float(series.iloc[-(self.long_days + 1)]) - 1.0
        ma200 = float(series.tail(self.ma_days).mean())
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

    def _cross_sectional_rankings(self, signal_date: pd.Timestamp) -> dict[str, dict[str, Any]]:
        evaluations: dict[str, dict[str, Any]] = {}
        eligible_codes: list[str] = []

        for code in self.satellite_pool:
            ev = self.evaluate_one(code, signal_date)
            evaluations[code] = ev
            if ev["eligible"] and np.isfinite(float(ev["momentum_score"])):
                eligible_codes.append(code)

        ranked = sorted(
            eligible_codes,
            key=lambda c: (-float(evaluations[c]["momentum_score"]), str(c)),
        )
        ranks = {code: idx + 1 for idx, code in enumerate(ranked)}

        for code, ev in evaluations.items():
            ev["cross_sectional_rank"] = ranks.get(code)
        return evaluations

    def _should_retain(self, code: str, ev: dict[str, Any], signal_date: pd.Timestamp) -> tuple[bool, str]:
        sel_date = self._selection_dates.get(code)
        quarters = _quarters_held(sel_date, signal_date) if sel_date else 0

        trend_pass = bool(ev.get("trend_condition"))
        pos_mom = bool(ev.get("momentum_condition"))
        rank_le_3 = ev.get("cross_sectional_rank") is not None and ev["cross_sectional_rank"] <= 3

        # Emergency exit: trend failure OR momentum<=0, regardless of holding period
        if not trend_pass:
            return False, "EMERGENCY_EXIT_BELOW_MA200"
        if not pos_mom:
            return False, "EMERGENCY_EXIT_MOMENTUM_NON_POSITIVE"

        # After min holding quarters: apply full retention rules
        if quarters >= self.min_holding_quarters:
            if trend_pass and pos_mom and rank_le_3:
                return True, "RETAINED_ALL_RULES_PASSED"
            reasons = []
            if not trend_pass:
                reasons.append("TREND_FAILED")
            if not pos_mom:
                reasons.append("MOMENTUM_NON_POSITIVE")
            if not rank_le_3:
                reasons.append(f"RANK_{ev.get('cross_sectional_rank')}")
            return False, "REMOVED_" + ";".join(reasons)

        # During min holding (trend+momentum passed above): protected from rank-only removal
        return True, "RETAINED_MIN_HOLDING_PROTECTED"

    def _select_satellites(self, evaluations: dict[str, dict[str, Any]], signal_date: pd.Timestamp) -> tuple[list[str], dict[str, dict[str, Any]]]:
        previous = list(self.last_selected_satellites or [])
        audit: dict[str, dict[str, Any]] = {}
        retained: set[str] = set()

        for code in previous:
            ev = evaluations.get(code)
            if ev is None:
                audit[code] = {"action": "REMOVED", "reason": "NO_EVALUATION"}
                continue

            keep, reason = self._should_retain(code, ev, signal_date)
            if keep:
                retained.add(code)
            audit[code] = {
                "action": "RETAINED" if keep else "REMOVED",
                "reason": reason,
                "quarters_held": _quarters_held(self._selection_dates.get(code), signal_date),
                "trend_passing": bool(ev.get("trend_condition")),
                "momentum_positive": bool(ev.get("momentum_condition")),
                "rank": ev.get("cross_sectional_rank"),
            }

        new_slots = max(0, self.max_satellites - len(retained))
        if new_slots > 0:
            eligible_ranked = sorted(
                [
                    c
                    for c in self.satellite_pool
                    if evaluations[c]["eligible"]
                    and np.isfinite(float(evaluations[c]["momentum_score"]))
                    and c not in retained
                ],
                key=lambda c: (-float(evaluations[c]["momentum_score"]), str(c)),
            )
            for code in eligible_ranked[:new_slots]:
                retained.add(code)
                audit[code] = {
                    "action": "SELECTED",
                    "reason": "NEW_FROM_RANK",
                    "rank": evaluations[code].get("cross_sectional_rank"),
                }

        selected = sorted(retained, key=lambda c: (evaluations.get(c, {}).get("cross_sectional_rank") or 10**9, str(c)))[: self.max_satellites]

        for code in self.satellite_pool:
            if code not in audit:
                ev = evaluations[code]
                audit[code] = {
                    "action": "NOT_SELECTED",
                    "rank": ev.get("cross_sectional_rank"),
                    "eligible": bool(ev.get("eligible")),
                }

        return selected, audit

    def _build_target_weights(self, selected_satellites: list[str]) -> tuple[dict[str, float], dict[str, Any]]:
        target = dict(self.core_weights)
        audit_info: dict[str, Any] = {
            "selected_satellites": selected_satellites,
            "satellite_count": len(selected_satellites),
        }

        used_satellite_budget = len(selected_satellites) * self.satellite_slot_weight
        for code in selected_satellites:
            target[code] = self.satellite_slot_weight

        unused = max(0.0, self.satellite_budget - used_satellite_budget)
        if unused > 1e-12:
            total_ratio = FALLBACK_RATIO_PRIMARY + FALLBACK_RATIO_SECONDARY
            target[FALLBACK_PRIMARY] = target.get(FALLBACK_PRIMARY, 0.0) + unused * FALLBACK_RATIO_PRIMARY / total_ratio
            target[FALLBACK_SECONDARY] = target.get(FALLBACK_SECONDARY, 0.0) + unused * FALLBACK_RATIO_SECONDARY / total_ratio

        audit_info["unused_satellite_budget"] = round(unused, 6)
        audit_info["fallback_to_001512"] = round(
            unused * FALLBACK_RATIO_PRIMARY / (FALLBACK_RATIO_PRIMARY + FALLBACK_RATIO_SECONDARY), 6
        )
        audit_info["fallback_to_260102"] = round(
            unused * FALLBACK_RATIO_SECONDARY / (FALLBACK_RATIO_PRIMARY + FALLBACK_RATIO_SECONDARY), 6
        )

        return target, audit_info

    def _current_indices(self, codes: list[str], signal_date: pd.Timestamp) -> dict[str, float]:
        """Return latest total-return index values at or before signal_date. No future data."""
        result: dict[str, float] = {}
        idx = self.total_return_index
        for code in codes:
            if code not in idx.columns:
                continue
            cutoff = pd.Timestamp(signal_date)
            if self.trading_dates is not None:
                from otf_rotation.nav_availability import available_as_of, lag_for

                cutoff = available_as_of(
                    self.trading_dates, pd.Timestamp(signal_date), lag_for(code, self.qdii_codes or [])
                )
            series = idx.loc[:cutoff, code].dropna()
            if not series.empty:
                result[code] = float(series.iloc[-1])
        return result

    def _compute_drift_deviation(
        self, proposed_target: dict[str, float], signal_date: pd.Timestamp
    ) -> tuple[float, dict[str, float], str]:
        """Compute max absolute deviation between natural-drift weights and proposed target.

        Returns (max_deviation, drift_weights, reason).
        """
        if self.last_accepted_target is None or self.last_reference_index is None:
            return 0.0, {}, "NO_PREVIOUS_TARGET"

        all_codes = list(set(proposed_target.keys()) | set(self.last_accepted_target.keys()))
        current_indices = self._current_indices(all_codes, signal_date)

        ref_keys = set(self.last_reference_index.keys())
        if not all(c in current_indices for c in ref_keys):
            missing = ref_keys - set(current_indices)
            return 0.0, {}, f"INDEX_UNAVAILABLE_{sorted(missing)}"

        try:
            drift = _natural_drift_from_index(
                self.last_accepted_target, self.last_reference_index, current_indices
            )
        except ValueError as e:
            return 0.0, {}, f"NATURAL_DRIFT_ERROR_{str(e)}"

        union = set(drift) | set(proposed_target)
        if not union:
            return 0.0, drift, "NO_UNION_CODES"

        max_dev = max(
            abs(float(drift.get(code, 0.0)) - float(proposed_target.get(code, 0.0)))
            for code in union
        )
        return max_dev, drift, "OK"

    def generate_signal(self, signal_date: pd.Timestamp) -> dict[str, float]:
        """Generate the C3 rebalance signal for a quarter-end date."""
        signal_date = pd.Timestamp(signal_date)
        evaluations = self._cross_sectional_rankings(signal_date)

        selected, selection_audit = self._select_satellites(evaluations, signal_date)
        proposed_target, weight_audit = self._build_target_weights(selected)

        # Decision logic
        action = "TRIGGER_REBALANCE"
        reason = "INITIAL_BUILD"
        drift_weights: dict[str, float] = {}
        max_drift_deviation = 0.0
        drift_reason = ""

        if self.last_accepted_target is not None:
            previous = list(self.last_selected_satellites or [])
            selection_changed = set(selected) != set(previous or [])

            if selection_changed:
                action = "TRIGGER_REBALANCE"
                reason = "SELECTION_SET_CHANGED"
            else:
                max_drift_deviation, drift_weights, drift_reason = self._compute_drift_deviation(
                    proposed_target, signal_date
                )
                if drift_reason != "OK":
                    action = "NO_TRADE"
                    reason = f"DRIFT_CHECK_UNAVAILABLE_{drift_reason}"
                elif max_drift_deviation >= self.drift_threshold - 1e-12:
                    action = "TRIGGER_REBALANCE"
                    reason = f"DRIFT_THRESHOLD_TRIGGER_{round(max_drift_deviation*100,1)}PP"
                else:
                    action = "NO_TRADE"
                    reason = f"SELECTION_UNCHANGED_DRIFT_BELOW_THRESHOLD_{round(max_drift_deviation*100,1)}PP"

        # Apply decision
        if action != "NO_TRADE":
            self.last_accepted_target = dict(proposed_target)
            self.last_selected_satellites = list(selected)
            self.last_accepted_date = signal_date

            # Update selection dates: remove exited codes, set new/reselected codes
            previous_set = set(self.last_selected_satellites or [])
            for code in list(self._selection_dates.keys()):
                if code not in previous_set:
                    del self._selection_dates[code]
            for code in selected:
                if code not in self._selection_dates:
                    self._selection_dates[code] = signal_date

            # Update reference index for future drift calculations
            all_codes = list(set(proposed_target.keys()))
            current_indices = self._current_indices(all_codes, signal_date)
            self.last_reference_index = {
                code: current_indices[code]
                for code in proposed_target
                if code in current_indices
            }

            accepted_target = dict(proposed_target)
        else:
            accepted_target = dict(self.last_accepted_target or proposed_target)

        audit = {
            "signal_date": signal_date,
            "action": action,
            "reason": reason,
            "evaluations": evaluations,
            "selection_audit": selection_audit,
            "weight_audit": weight_audit,
            "proposed_target_weights": proposed_target,
            "target_weights": accepted_target,
            "accepted_target_date": self.last_accepted_date,
            "max_drift_deviation_pct_points": round(max_drift_deviation * 100.0, 2),
            "drift_reason": drift_reason,
            "drift_weights": drift_weights,
            "index_available_for_drift": bool(drift_weights) or (drift_reason == "OK"),
            "parameters": {
                "min_published_observations": self.min_observations,
                "momentum_short_days": self.short_days,
                "momentum_long_days": self.long_days,
                "trend_ma_days": self.ma_days,
                "satellite_budget": self.satellite_budget,
                "max_satellites": self.max_satellites,
                "satellite_slot_weight": self.satellite_slot_weight,
                "min_holding_quarters": self.min_holding_quarters,
                "drift_threshold_pct_points": self.drift_threshold * 100.0,
            },
        }
        self.last_signal_audit = audit
        self.audit_rows.append(dict(audit))
        return accepted_target
