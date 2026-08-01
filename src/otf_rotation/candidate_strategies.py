"""Frozen B2-LT and D1 candidate signal modules.

The candidate signals are intentionally independent from the canonical B1/B2/
B3/S1 runner.  They emit an empty mapping when an observation does not call
for a rebalance; the candidate runner therefore omits that submission date
from the target table instead of translating it into a cash target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _normalise_code(value: object) -> str:
    text = str(value).strip()
    return text.zfill(6) if text.isdigit() else text


def _normalise_nav_frame(nav_df: pd.DataFrame) -> pd.DataFrame:
    required = {"fund_code", "nav_date", "unit_nav"}
    missing = required - set(nav_df.columns)
    if missing:
        raise ValueError(f"candidate_nav_missing_columns:{sorted(missing)}")
    frame = nav_df[["fund_code", "nav_date", "unit_nav"]].copy()
    frame["fund_code"] = frame["fund_code"].map(_normalise_code)
    frame["nav_date"] = pd.to_datetime(frame["nav_date"], errors="coerce")
    frame["unit_nav"] = pd.to_numeric(frame["unit_nav"], errors="coerce")
    frame = frame.dropna(subset=["fund_code", "nav_date", "unit_nav"])
    frame = frame[frame["unit_nav"] > 0]
    return frame.sort_values(["fund_code", "nav_date"]).drop_duplicates(
        ["fund_code", "nav_date"], keep="last"
    )


def latest_published_nav(
    nav_df: pd.DataFrame, fund_code: str, date: pd.Timestamp
) -> tuple[float | None, pd.Timestamp | None, int]:
    """Return the latest published NAV at or before ``date`` and its count."""
    frame = _normalise_nav_frame(nav_df)
    history = frame[
        (frame["fund_code"] == _normalise_code(fund_code))
        & (frame["nav_date"] <= pd.Timestamp(date))
    ]
    if history.empty:
        return None, None, 0
    row = history.iloc[-1]
    return float(row["unit_nav"]), pd.Timestamp(row["nav_date"]), int(len(history))


def natural_drift_weights(
    previous_target: Mapping[str, float],
    previous_navs: Mapping[str, float],
    current_navs: Mapping[str, float],
) -> dict[str, float]:
    """Calculate weights implied by the accepted target and NAV movement.

    No account value or future observation is needed: each accepted target is
    scaled by the ratio of current published NAV to the NAV recorded when the
    target was accepted, then normalised across the accepted holdings.
    """
    amounts: dict[str, float] = {}
    for code, weight in previous_target.items():
        previous_nav = float(previous_navs.get(code, np.nan))
        current_nav = float(current_navs.get(code, np.nan))
        if not np.isfinite(previous_nav) or previous_nav <= 0:
            raise ValueError(f"candidate_previous_nav_missing:{code}")
        if not np.isfinite(current_nav) or current_nav <= 0:
            raise ValueError(f"candidate_current_nav_missing:{code}")
        amounts[str(code)] = max(0.0, float(weight)) * current_nav / previous_nav
    total = sum(amounts.values())
    if total <= 0:
        return {}
    return {code: amount / total for code, amount in amounts.items()}


def _latest_navs(
    nav_df: pd.DataFrame, fund_codes: list[str], date: pd.Timestamp
) -> dict[str, float]:
    values: dict[str, float] = {}
    for code in fund_codes:
        nav, _, _ = latest_published_nav(nav_df, code, date)
        if nav is not None:
            values[_normalise_code(code)] = nav
    return values


@dataclass
class SignalAudit:
    signal_date: pd.Timestamp
    submit_date: str | None = None
    rebalance: bool = False
    decision: str = ""
    max_deviation_pct_points: float = 0.0
    drift_weights: dict[str, float] | None = None
    target_weights: dict[str, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "signal_date": self.signal_date,
            "submit_date": self.submit_date,
            "rebalance": self.rebalance,
            "decision": self.decision,
            "max_deviation_pct_points": self.max_deviation_pct_points,
            "drift_weights": dict(self.drift_weights or {}),
            "target_weights": dict(self.target_weights or {}),
        }


class B2LTSignal:
    """Quarter-end B2 static target with a frozen five-point trigger."""

    assets = ("160706", "000218", "001512", "260102")
    target = {code: 0.25 for code in assets}

    def __init__(self, nav_df: pd.DataFrame, threshold: float = 0.05):
        self.nav_df = _normalise_nav_frame(nav_df)
        self.threshold = float(threshold)
        self.audit_rows: list[dict[str, Any]] = []
        self.reset()

    def reset(self) -> None:
        self.last_accepted_target: dict[str, float] | None = None
        self.last_reference_navs: dict[str, float] | None = None
        self.last_accepted_date: pd.Timestamp | None = None
        self.last_signal_audit: dict[str, Any] = {}
        self.audit_rows = []

    def __call__(self, date: pd.Timestamp) -> dict[str, float]:
        signal_date = pd.Timestamp(date)
        current_navs = _latest_navs(self.nav_df, list(self.assets), signal_date)
        drift: dict[str, float] = {}
        max_deviation = 0.0
        if self.last_accepted_target is None:
            rebalance = True
            decision = "FIRST_BUILD"
        elif len(current_navs) != len(self.assets):
            rebalance = False
            decision = "NO_REBALANCE_NAV_UNAVAILABLE"
        else:
            try:
                drift = natural_drift_weights(
                    self.last_accepted_target,
                    self.last_reference_navs or {},
                    current_navs,
                )
                max_deviation = max(
                    abs(float(drift.get(code, 0.0)) - 0.25)
                    for code in self.assets
                )
            except ValueError:
                rebalance = False
                decision = "NO_REBALANCE_NAV_UNAVAILABLE"
            else:
                # The boundary is inclusive: exactly five percentage points
                # is a rebalance, not a skipped observation.
                rebalance = max_deviation >= self.threshold - 1e-12
                decision = (
                    "THRESHOLD_TRIGGER"
                    if rebalance
                    else "NO_REBALANCE_WITHIN_THRESHOLD"
                )

        target = dict(self.target) if rebalance else {}
        if rebalance:
            self.last_accepted_target = dict(self.target)
            self.last_reference_navs = dict(current_navs)
            self.last_accepted_date = signal_date
        audit = SignalAudit(
            signal_date=signal_date,
            rebalance=rebalance,
            decision=decision,
            max_deviation_pct_points=max_deviation * 100.0,
            drift_weights=drift,
            target_weights=target,
        ).as_dict()
        audit.update(
            {
                "strategy": "B2_LT",
                "observation_frequency": "quarter_end",
                "threshold_pct_points": self.threshold * 100.0,
                "reference_navs": dict(current_navs),
                "accepted_target_date": self.last_accepted_date,
            }
        )
        self.last_signal_audit = audit
        self.audit_rows.append(dict(audit))
        return target


def _frame_row(frame: pd.DataFrame | None, code: str) -> dict[str, Any] | None:
    if frame is None or frame.empty or "fund_code" not in frame:
        return None
    matches = frame[
        frame["fund_code"].map(_normalise_code) == _normalise_code(code)
    ]
    if matches.empty:
        return None
    return matches.iloc[-1].to_dict()


def _temporal_field_allows(row: Mapping[str, Any], date: pd.Timestamp) -> bool:
    effective_from = str(row.get("effective_from", "") or "").strip()
    effective_to = str(row.get("effective_to", "") or "").strip()
    if effective_from and pd.Timestamp(date) < pd.Timestamp(effective_from):
        return False
    if effective_to and pd.Timestamp(date) > pd.Timestamp(effective_to):
        return False
    return True


def product_history_facts(
    nav_df: pd.DataFrame,
    product_codes: list[str],
    oos_start: str | pd.Timestamp,
    ma_days: int = 200,
) -> dict[str, dict[str, Any]]:
    """Summarise PIT history without looking beyond the OOS start date."""
    frame = _normalise_nav_frame(nav_df)
    start = pd.Timestamp(oos_start)
    facts: dict[str, dict[str, Any]] = {}
    for code in product_codes:
        rows = frame[frame["fund_code"] == _normalise_code(code)]
        before = rows[rows["nav_date"] <= start]
        facts[_normalise_code(code)] = {
            "first_published_nav": (
                rows["nav_date"].min().strftime("%Y-%m-%d") if not rows.empty else None
            ),
            "published_rows_at_oos_start": int(len(before)),
            "ma200_available_at_oos_start": bool(len(before) >= ma_days),
        }
    return facts


class D1Signal:
    """Frozen multi-asset budget with equity-only 200-NAV-day trend filters."""

    def __init__(
        self,
        nav_df: pd.DataFrame,
        rules: pd.DataFrame | None,
        mapping: pd.DataFrame | None,
        products: Mapping[str, Mapping[str, Any]],
        asset_budgets: Mapping[str, float],
        fallback_fund: str = "260102",
        ma_days: int = 200,
        threshold: float = 0.05,
        rule_book: Any | None = None,
    ):
        self.nav_df = _normalise_nav_frame(nav_df)
        self.rules = rules
        self.mapping = mapping
        self.products = {
            _normalise_code(code): dict(value) for code, value in products.items()
        }
        self.asset_budgets = {str(key): float(value) for key, value in asset_budgets.items()}
        self.fallback_fund = _normalise_code(fallback_fund)
        self.ma_days = int(ma_days)
        self.threshold = float(threshold)
        self.rule_book = rule_book
        self.audit_rows: list[dict[str, Any]] = []
        self.reset()

    def reset(self) -> None:
        self.last_accepted_target: dict[str, float] | None = None
        self.last_reference_navs: dict[str, float] | None = None
        self.last_states: dict[str, str] | None = None
        self.last_accepted_date: pd.Timestamp | None = None
        self.last_signal_audit: dict[str, Any] = {}
        self.audit_rows = []

    def _evaluate_product(
        self, code: str, spec: Mapping[str, Any], signal_date: pd.Timestamp
    ) -> dict[str, Any]:
        code = _normalise_code(code)
        budget = float(spec["budget"])
        mapping_row = _frame_row(self.mapping, code)
        rule_row = _frame_row(self.rules, code)
        reasons: list[str] = []
        if mapping_row is None:
            reasons.append("MISSING_MAPPING")
        else:
            if str(mapping_row.get("mapping_confidence", "")).upper() != "HIGH":
                reasons.append("MAPPING_CONFIDENCE_NOT_HIGH")
            if str(mapping_row.get("review_status", "")).upper() != "APPROVED":
                reasons.append("MAPPING_NOT_APPROVED")
            if not _temporal_field_allows(mapping_row, signal_date):
                reasons.append("MAPPING_NOT_EFFECTIVE")
        if rule_row is None:
            reasons.append("MISSING_RULE")
        else:
            formal_statuses = {"OFFICIAL_VERIFIED", "DISTRIBUTOR_VERIFIED"}
            if str(rule_row.get("rule_status", "")).upper() not in formal_statuses:
                reasons.append("RULE_STATUS_NOT_FORMAL")
            if not _temporal_field_allows(rule_row, signal_date):
                reasons.append("RULE_NOT_EFFECTIVE")
        if self.rule_book is not None:
            allowed, reason = self.rule_book.is_rule_allowed(code, submit_date=signal_date)
            if not allowed:
                reasons.append(str(reason))

        product_rows = self.nav_df[
            (self.nav_df["fund_code"] == code)
            & (self.nav_df["nav_date"] <= signal_date)
        ]
        nav = None
        ma200 = None
        nav_date = None
        if product_rows.empty:
            reasons.append("PIT_NAV_MISSING")
        else:
            latest = product_rows.iloc[-1]
            nav = float(latest["unit_nav"])
            nav_date = pd.Timestamp(latest["nav_date"])
            if bool(spec.get("trend_filter", False)):
                if len(product_rows) < self.ma_days:
                    reasons.append("PIT_NAV_INSUFFICIENT_FOR_MA200")
                else:
                    ma200 = float(product_rows["unit_nav"].tail(self.ma_days).mean())

        eligible = not reasons
        trend_state = "NO_TREND"
        if bool(spec.get("trend_filter", False)):
            if not eligible:
                trend_state = reasons[-1]
            else:
                trend_state = "ABOVE_MA200" if nav >= float(ma200) else "BELOW_MA200"
                if trend_state == "BELOW_MA200":
                    eligible = False
                    reasons.append("TREND_BELOW_MA200")
        selected_fund = code if eligible else self.fallback_fund
        return {
            "fund_code": code,
            "asset_class": str(spec.get("asset_class", "")),
            "budget": budget,
            "trend_filter": bool(spec.get("trend_filter", False)),
            "nav": nav,
            "nav_date": nav_date,
            "ma200": ma200,
            "trend_state": trend_state,
            "eligible": bool(eligible),
            "reason": ";".join(reasons) if reasons else "ELIGIBLE",
            "selected_fund": selected_fund,
            "score": (nav / ma200 - 1.0) if nav is not None and ma200 else 0.0,
        }

    def _target_and_audits(
        self, signal_date: pd.Timestamp
    ) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, str], dict[str, float]]:
        # The cash sleeve is represented by the explicit 260102 product row
        # in ``products``.  Starting it here as well would double count the
        # frozen 10% cash budget before any fallback transfers are applied.
        target: dict[str, float] = {}
        evaluations: list[dict[str, Any]] = []
        states: dict[str, str] = {}
        current_navs: dict[str, float] = {self.fallback_fund: 1.0}
        for code, spec in self.products.items():
            evaluation = self._evaluate_product(code, spec, signal_date)
            evaluations.append(evaluation)
            states[code] = str(evaluation["trend_state"])
            if evaluation["nav"] is not None:
                current_navs[code] = float(evaluation["nav"])
            selected = str(evaluation["selected_fund"])
            target[selected] = target.get(selected, 0.0) + float(evaluation["budget"])
        target = {code: weight for code, weight in target.items() if weight > 1e-12}
        return target, evaluations, states, current_navs

    def __call__(self, date: pd.Timestamp) -> dict[str, float]:
        signal_date = pd.Timestamp(date)
        new_target, evaluations, states, current_navs = self._target_and_audits(signal_date)
        drift: dict[str, float] = {}
        max_deviation = 0.0
        state_changed = self.last_states is not None and states != self.last_states
        if self.last_accepted_target is None:
            rebalance = True
            decision = "FIRST_BUILD"
        else:
            if not state_changed:
                try:
                    drift = natural_drift_weights(
                        self.last_accepted_target,
                        self.last_reference_navs or {},
                        current_navs,
                    )
                    all_codes = set(drift) | set(new_target)
                    max_deviation = max(
                        abs(float(drift.get(code, 0.0)) - float(new_target.get(code, 0.0)))
                        for code in all_codes
                    ) if all_codes else 0.0
                except ValueError:
                    max_deviation = 0.0
                    decision = "NO_REBALANCE_DRIFT_NAV_UNAVAILABLE"
                    rebalance = False
                else:
                    rebalance = max_deviation >= self.threshold - 1e-12
                    decision = (
                        "DRIFT_THRESHOLD_TRIGGER"
                        if rebalance else "NO_REBALANCE_WITHIN_THRESHOLD"
                    )
            else:
                rebalance = True
                decision = "TREND_STATE_CHANGE"
        if rebalance:
            target = dict(new_target)
            self.last_accepted_target = dict(target)
            self.last_reference_navs = {
                code: float(current_navs.get(code, 1.0))
                for code in target
            }
            self.last_accepted_date = signal_date
        else:
            target = {}
        market_state = "TREND_MIXED"
        equity_states = [
            evaluation["trend_state"]
            for evaluation in evaluations
            if evaluation["trend_filter"]
        ]
        if equity_states and all(state == "ABOVE_MA200" for state in equity_states):
            market_state = "EQUITY_ABOVE_MA200"
        elif equity_states and all(state == "BELOW_MA200" for state in equity_states):
            market_state = "EQUITY_BELOW_MA200"

        audit = {
            "signal_date": signal_date,
            "rebalance": bool(rebalance),
            "decision": decision,
            "market_state": market_state,
            "state_changed": bool(state_changed),
            "max_deviation_pct_points": max_deviation * 100.0,
            "drift_weights": dict(drift),
            "target_weights": dict(target),
            "accepted_target_date": self.last_accepted_date,
            "states": dict(states),
            "state_scores": {item["fund_code"]: float(item["score"]) for item in evaluations},
            "evaluations": evaluations,
            "asset_budgets": dict(self.asset_budgets),
            "fund_weights": dict(new_target),
            "trend_ma_published_nav_days": self.ma_days,
            "threshold_pct_points": self.threshold * 100.0,
        }
        self.last_signal_audit = audit
        self.audit_rows.append(dict(audit))
        return target


def build_candidate_targets(
    signal_func: Any,
    signal_dates: list[pd.Timestamp],
    signal_map: Mapping[str, str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Build sparse target rows; empty signals mean no target row."""
    submit_for_signal = {str(signal): str(submit) for submit, signal in signal_map.items()}
    target_rows: dict[pd.Timestamp, dict[str, float]] = {}
    audits: list[dict[str, Any]] = []
    for signal_date in signal_dates:
        signal = pd.Timestamp(signal_date)
        submit = submit_for_signal.get(signal.strftime("%Y-%m-%d"))
        if submit is None:
            continue
        weights = signal_func(signal) or {}
        audit = getattr(signal_func, "last_signal_audit", None)
        if audit:
            row = dict(audit)
            row["submit_date"] = submit
            audits.append(row)
        if not weights:
            continue
        cleaned = {
            _normalise_code(code): float(value)
            for code, value in weights.items()
            if float(value) > 1e-12
        }
        if not cleaned:
            continue
        submit_date = pd.Timestamp(submit)
        if submit_date in target_rows:
            raise ValueError(f"candidate_duplicate_submit_date:{submit_date.date()}")
        target_rows[submit_date] = cleaned
    if not target_rows:
        return pd.DataFrame(), audits
    frame = pd.DataFrame.from_dict(target_rows, orient="index").fillna(0.0)
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    return frame.loc[:, (frame != 0).any(axis=0)], audits
