"""Frozen C1 core/satellite momentum signal and research helpers.

This module deliberately has no parameter-search surface.  The signal uses
published ``daily_growth_pct`` observations to construct a point-in-time total
return index.  Unit NAV is intentionally not used for momentum, covariance,
or natural-drift decisions.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np
import pandas as pd

from otf_trading_rules import ProductRuleBook


CORE_WEIGHTS: dict[str, float] = {
    "001512": 0.15,
    "000148": 0.10,
    "000218": 0.15,
    "260102": 0.10,
}
SATELLITE_POOL: tuple[str, ...] = (
    "160706",
    "000008",
    "007466",
    "050021",
    "050025",
    "000071",
)
SATELLITE_BASE_WEIGHT = 0.25
SATELLITE_BUDGET = 0.50
MIN_PUBLISHED_OBSERVATIONS = 253
MOMENTUM_SHORT_DAYS = 126
MOMENTUM_LONG_DAYS = 252
TREND_MA_DAYS = 200
COVARIANCE_DAYS = 60
VOLATILITY_TARGET = 0.10
DRIFT_THRESHOLD = 0.05
BUFFER_THRESHOLD = 0.03


def _normalise_code(value: object) -> str:
    text = str(value).strip()
    return text.zfill(6) if text.isdigit() else text


def _normalise_growth_frame(growth_df: pd.DataFrame) -> pd.DataFrame:
    required = {"fund_code", "nav_date", "daily_growth_pct"}
    missing = required - set(growth_df.columns)
    if missing:
        raise ValueError(f"c1_growth_missing_columns:{sorted(missing)}")
    frame = growth_df[["fund_code", "nav_date", "daily_growth_pct"]].copy()
    frame["fund_code"] = frame["fund_code"].map(_normalise_code)
    frame["nav_date"] = pd.to_datetime(frame["nav_date"], errors="coerce")
    frame["daily_growth_pct"] = pd.to_numeric(
        frame["daily_growth_pct"], errors="coerce"
    )
    frame = frame.dropna(subset=["fund_code", "nav_date", "daily_growth_pct"])
    if (frame["daily_growth_pct"] <= -100.0).any():
        raise ValueError("c1_daily_growth_must_be_above_minus_100_pct")
    return frame.sort_values(["fund_code", "nav_date"]).drop_duplicates(
        ["fund_code", "nav_date"], keep="last"
    )


def build_total_return_index(growth_df: pd.DataFrame) -> pd.DataFrame:
    """Build one total-return index per asset from published daily growth only."""
    frame = _normalise_growth_frame(growth_df)
    frame["total_return_index"] = frame.groupby("fund_code", sort=False)[
        "daily_growth_pct"
    ].transform(lambda values: (1.0 + values / 100.0).cumprod())
    return frame.pivot(
        index="nav_date", columns="fund_code", values="total_return_index"
    ).sort_index()


def _history_for_code(
    growth_df: pd.DataFrame, code: str, signal_date: pd.Timestamp
) -> pd.DataFrame:
    frame = _normalise_growth_frame(growth_df)
    return frame[
        (frame["fund_code"] == _normalise_code(code))
        & (frame["nav_date"] <= pd.Timestamp(signal_date))
    ].sort_values("nav_date")


def select_top2_with_buffer(
    evaluations: Mapping[str, Mapping[str, Any]],
    previous_selected: list[str] | tuple[str, ...] | None,
    buffer_pct_points: float = BUFFER_THRESHOLD * 100.0,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Select the top two eligible assets with the frozen 3pp retention rule."""
    eligible = []
    for code, row in evaluations.items():
        score = row.get("momentum_score")
        try:
            score_is_finite = np.isfinite(float(score))
        except (TypeError, ValueError):
            score_is_finite = False
        if bool(row.get("eligible")) and score_is_finite:
            eligible.append(code)
    ranked = sorted(
        eligible,
        key=lambda code: (-float(evaluations[code]["momentum_score"]), str(code)),
    )
    ranks = {code: index + 1 for index, code in enumerate(ranked)}
    if len(ranked) >= 2:
        second_score = float(evaluations[ranked[1]]["momentum_score"])
    elif ranked:
        second_score = float(evaluations[ranked[0]]["momentum_score"])
    else:
        second_score = None

    previous = [str(code) for code in (previous_selected or [])]
    retained: set[str] = set()
    audit: dict[str, dict[str, Any]] = {}
    for code, row in evaluations.items():
        score = row.get("momentum_score")
        is_eligible = code in eligible
        margin = (
            (float(score) - second_score) * 100.0
            if is_eligible and second_score is not None
            else None
        )
        retained_by_buffer = bool(
            is_eligible
            and code in previous
            and margin is not None
            and margin >= -float(buffer_pct_points) - 1e-12
        )
        if retained_by_buffer and len(retained) < 2:
            retained.add(code)
        audit[code] = {
            "rank": ranks.get(code),
            "eligible": is_eligible,
            "second_place_score": second_score,
            "buffer_margin_pct_points": margin,
            "retained_by_buffer": retained_by_buffer,
        }

    for code in ranked:
        if len(retained) >= 2:
            break
        retained.add(code)
    selected = sorted(retained, key=lambda code: ranks.get(code, 10**9))[:2]
    for code in evaluations:
        audit[code]["selected"] = code in selected
    return selected, audit


def estimate_total_return_covariance(
    growth_df: pd.DataFrame,
    codes: list[str] | tuple[str, ...],
    signal_date: pd.Timestamp,
    window: int = COVARIANCE_DAYS,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Estimate annual covariance from the latest common published returns."""
    return _estimate_total_return_covariance_normalized(
        _normalise_growth_frame(growth_df), codes, signal_date, window
    )


def _estimate_total_return_covariance_normalized(
    frame: pd.DataFrame,
    codes: list[str] | tuple[str, ...],
    signal_date: pd.Timestamp,
    window: int = COVARIANCE_DAYS,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Internal covariance path for an already-normalized growth frame."""
    selected_codes = [_normalise_code(code) for code in dict.fromkeys(codes)]
    limited = frame[
        (frame["nav_date"] <= pd.Timestamp(signal_date))
        & frame["fund_code"].isin(selected_codes)
    ]
    returns = limited.pivot(
        index="nav_date", columns="fund_code", values="daily_growth_pct"
    ).sort_index() / 100.0
    per_asset = {
        code: int(returns[code].dropna().tail(window).shape[0])
        if code in returns.columns else 0
        for code in selected_codes
    }
    if not selected_codes or any(per_asset[code] < window for code in selected_codes):
        return None, {
            "window_days": int(window),
            "signal_date": pd.Timestamp(signal_date),
            "codes": selected_codes,
            "per_asset_observations": per_asset,
            "common_observations": 0,
            "coverage_passed": False,
            "reason": "COVARIANCE_ASSET_HISTORY_INSUFFICIENT",
        }
    common = returns[selected_codes].dropna(how="any").tail(window)
    audit = {
        "window_days": int(window),
        "signal_date": pd.Timestamp(signal_date),
        "codes": selected_codes,
        "per_asset_observations": per_asset,
        "common_observations": int(len(common)),
        "common_start": common.index.min() if not common.empty else None,
        "common_end": common.index.max() if not common.empty else None,
        "coverage_passed": bool(len(common) >= window),
        "reason": "OK" if len(common) >= window else "COVARIANCE_COMMON_HISTORY_INSUFFICIENT",
    }
    if len(common) < window or len(selected_codes) == 1:
        return None, audit
    return common.cov(ddof=1) * 252.0, audit


def _portfolio_volatility(
    core_weights: Mapping[str, float],
    satellite_weights: Mapping[str, float],
    satellite_scale: float,
    satellite_budget: float,
    covariance: pd.DataFrame,
) -> float:
    weights = {str(code): float(weight) for code, weight in core_weights.items()}
    cash_code = "260102"
    scaled_satellite_total = 0.0
    for code, weight in satellite_weights.items():
        scaled = float(weight) * float(satellite_scale)
        weights[str(code)] = scaled
        scaled_satellite_total += scaled
    weights[cash_code] = weights.get(cash_code, 0.0) + max(
        0.0, float(satellite_budget) - scaled_satellite_total
    )
    codes = [code for code, weight in weights.items() if weight > 1e-14]
    if not codes or any(code not in covariance.index for code in codes):
        return float("nan")
    vector = np.array([weights[code] for code in codes], dtype=float)
    matrix = covariance.loc[codes, codes].to_numpy(dtype=float)
    variance = float(vector @ matrix @ vector)
    return float(np.sqrt(max(0.0, variance)))


def scale_satellite_weights_to_vol(
    core_weights: Mapping[str, float],
    satellite_weights: Mapping[str, float],
    covariance: pd.DataFrame,
    *,
    target_vol: float = VOLATILITY_TARGET,
    satellite_budget: float = SATELLITE_BUDGET,
) -> tuple[float, float, dict[str, Any]]:
    """Use the frozen [0,1] bisection scale for satellite exposure."""
    if not satellite_weights:
        return 0.0, _portfolio_volatility(
            core_weights, {}, 0.0, satellite_budget, covariance
        ), {
            "bisection_applied": False,
            "full_scale_volatility_pct": 0.0,
            "core_only_volatility_pct": 0.0,
            "reason": "NO_SELECTED_SATELLITES",
        }
    full_vol = _portfolio_volatility(
        core_weights, satellite_weights, 1.0, satellite_budget, covariance
    )
    core_vol = _portfolio_volatility(
        core_weights, satellite_weights, 0.0, satellite_budget, covariance
    )
    if not np.isfinite(full_vol) or not np.isfinite(core_vol):
        return 0.0, float("nan"), {
            "bisection_applied": False,
            "full_scale_volatility_pct": None,
            "core_only_volatility_pct": None,
            "reason": "COVARIANCE_NOT_FINITE",
        }
    if full_vol <= target_vol + 1e-14:
        scale = 1.0
        reason = "FULL_SATELLITE_SCALE_WITHIN_TARGET"
        applied = False
    elif core_vol > target_vol + 1e-14:
        scale = 0.0
        reason = "FIXED_CORE_VOLATILITY_ABOVE_TARGET"
        applied = True
    else:
        low, high = 0.0, 1.0
        for _ in range(80):
            mid = (low + high) / 2.0
            mid_vol = _portfolio_volatility(
                core_weights, satellite_weights, mid, satellite_budget, covariance
            )
            if mid_vol <= target_vol:
                low = mid
            else:
                high = mid
        scale = low
        reason = "BISECTION_TO_VOLATILITY_TARGET"
        applied = True
    predicted = _portfolio_volatility(
        core_weights, satellite_weights, scale, satellite_budget, covariance
    )
    return float(np.clip(scale, 0.0, 1.0)), float(predicted), {
        "bisection_applied": applied,
        "full_scale_volatility_pct": full_vol * 100.0,
        "core_only_volatility_pct": core_vol * 100.0,
        "target_volatility_pct": target_vol * 100.0,
        "reason": reason,
    }


def _natural_drift_from_index(
    previous_target: Mapping[str, float],
    previous_index: Mapping[str, float],
    current_index: Mapping[str, float],
) -> dict[str, float]:
    amounts: dict[str, float] = {}
    for code, weight in previous_target.items():
        old = float(previous_index.get(code, np.nan))
        current = float(current_index.get(code, np.nan))
        if not np.isfinite(old) or old <= 0 or not np.isfinite(current) or current <= 0:
            raise ValueError(f"c1_total_return_index_missing:{code}")
        amounts[str(code)] = max(0.0, float(weight)) * current / old
    total = sum(amounts.values())
    if total <= 0:
        return {}
    return {code: value / total for code, value in amounts.items()}


class CoreSatelliteMomentumSignal:
    """C1 signal with all research parameters fixed at construction time."""

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
        covariance_days: int = COVARIANCE_DAYS,
        vol_target: float = VOLATILITY_TARGET,
        satellite_budget: float = SATELLITE_BUDGET,
        buffer_threshold: float = BUFFER_THRESHOLD,
        drift_threshold: float = DRIFT_THRESHOLD,
    ):
        self.growth_df = _normalise_growth_frame(growth_df)
        self.total_return_index = build_total_return_index(self.growth_df)
        self.core_weights = {
            _normalise_code(code): float(weight) for code, weight in core_weights.items()
        }
        self.satellite_pool = tuple(_normalise_code(code) for code in satellite_pool)
        self.min_observations = int(min_observations)
        self.short_days = int(short_days)
        self.long_days = int(long_days)
        self.ma_days = int(ma_days)
        self.covariance_days = int(covariance_days)
        self.vol_target = float(vol_target)
        self.satellite_budget = float(satellite_budget)
        self.buffer_threshold = float(buffer_threshold)
        self.drift_threshold = float(drift_threshold)
        self.reset()

    def reset(self) -> None:
        self.last_accepted_target: dict[str, float] | None = None
        self.last_reference_index: dict[str, float] | None = None
        self.last_selected_satellites: list[str] | None = None
        self.last_accepted_date: pd.Timestamp | None = None
        self.last_signal_audit: dict[str, Any] = {}
        self.audit_rows: list[dict[str, Any]] = []

    @staticmethod
    def should_rebalance_for_drift(
        max_deviation: float, threshold: float = DRIFT_THRESHOLD
    ) -> bool:
        return float(max_deviation) >= float(threshold) - 1e-12

    def _evaluate(self, code: str, signal_date: pd.Timestamp) -> dict[str, Any]:
        history = self.growth_df[
            (self.growth_df["fund_code"] == _normalise_code(code))
            & (self.growth_df["nav_date"] <= pd.Timestamp(signal_date))
        ]
        observations = len(history)
        row: dict[str, Any] = {
            "fund_code": code,
            "published_observations": int(observations),
            "latest_published_date": history["nav_date"].max() if not history.empty else None,
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
        if observations < self.min_observations:
            row["reason"] = f"INSUFFICIENT_PUBLISHED_OBSERVATIONS_{self.min_observations}"
            return row
        series = self.total_return_index.loc[:pd.Timestamp(signal_date), code].dropna()
        if len(series) < self.min_observations:
            row["reason"] = f"INSUFFICIENT_TOTAL_RETURN_INDEX_{self.min_observations}"
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

    def _proposed_target(
        self, signal_date: pd.Timestamp, evaluations: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, float], dict[str, Any]]:
        previous = self.last_selected_satellites or []
        selected, buffer_audit = select_top2_with_buffer(
            evaluations, previous, self.buffer_threshold * 100.0
        )
        base_satellite = {
            code: SATELLITE_BASE_WEIGHT for code in selected
        }
        cash_reasons: dict[str, str] = {}
        covariance_audit: dict[str, Any] = {}
        usable_satellite = dict(base_satellite)
        covariance: pd.DataFrame | None = None
        predicted_vol = float("nan")
        scale = 0.0
        scale_audit: dict[str, Any] = {
            "bisection_applied": False,
            "reason": "NO_SELECTED_SATELLITES",
        }
        if selected:
            cov_codes = list(self.core_weights) + selected
            covariance, covariance_audit = _estimate_total_return_covariance_normalized(
                self.growth_df, cov_codes, signal_date, self.covariance_days
            )
            if covariance is None:
                for code in selected:
                    cash_reasons[code] = str(
                        covariance_audit.get("reason", "COVARIANCE_UNAVAILABLE")
                    )
                usable_satellite = {}
            else:
                scale, predicted_vol, scale_audit = scale_satellite_weights_to_vol(
                    self.core_weights,
                    usable_satellite,
                    covariance,
                    target_vol=self.vol_target,
                    satellite_budget=self.satellite_budget,
                )
        target = dict(self.core_weights)
        for code, base_weight in usable_satellite.items():
            target[code] = float(base_weight * scale)
        used_satellite = sum(target.get(code, 0.0) for code in selected)
        cash_transfer = max(0.0, self.satellite_budget - used_satellite)
        target["260102"] = target.get("260102", 0.0) + cash_transfer
        target = {
            code: float(weight) for code, weight in target.items() if weight > 1e-12
        }
        for code in selected:
            if code not in usable_satellite:
                cash_reasons.setdefault(code, "SATELLITE_BUDGET_TO_CASH")
        for code, evaluation in evaluations.items():
            if not evaluation.get("eligible") and "INSUFFICIENT" in str(evaluation.get("reason", "")):
                cash_reasons.setdefault(code, str(evaluation.get("reason")))
        proposed = {
            "selected_satellites": selected,
            "selection_audit": buffer_audit,
            "base_satellite_weights": base_satellite,
            "usable_satellite_weights": usable_satellite,
            "cash_transfer": cash_transfer,
            "cash_transfer_reasons": cash_reasons,
            "covariance": covariance_audit,
            "covariance_coverage": covariance_audit.get("common_observations", 0),
            "covariance_start": covariance_audit.get("common_start"),
            "covariance_end": covariance_audit.get("common_end"),
            "predicted_volatility_pct": (
                predicted_vol * 100.0 if np.isfinite(predicted_vol) else None
            ),
            "satellite_scale": scale,
            "scale_audit": scale_audit,
            "target_weights": target,
        }
        return target, proposed

    def __call__(self, date: pd.Timestamp) -> dict[str, float]:
        signal_date = pd.Timestamp(date)
        previous_selected = list(self.last_selected_satellites or [])
        evaluations = {
            code: self._evaluate(code, signal_date) for code in self.satellite_pool
        }
        proposed_target, proposed = self._proposed_target(signal_date, evaluations)
        selected = list(proposed["selected_satellites"])
        selection_changed = (
            self.last_selected_satellites is not None
            and set(selected) != set(previous_selected)
        )
        current_indices = {
            code: float(self.total_return_index.loc[:signal_date, code].dropna().iloc[-1])
            for code in set(proposed_target)
            if code in self.total_return_index.columns
            and not self.total_return_index.loc[:signal_date, code].dropna().empty
        }
        drift: dict[str, float] = {}
        max_deviation = 0.0
        index_available = True
        if self.last_accepted_target is None:
            rebalance = True
            decision = "FIRST_BUILD"
        elif selection_changed:
            rebalance = True
            decision = "SATELLITE_SELECTION_CHANGED"
        else:
            try:
                drift = _natural_drift_from_index(
                    self.last_accepted_target,
                    self.last_reference_index or {},
                    current_indices,
                )
                union = set(drift) | set(proposed_target)
                max_deviation = max(
                    abs(float(drift.get(code, 0.0)) - float(proposed_target.get(code, 0.0)))
                    for code in union
                ) if union else 0.0
            except ValueError:
                index_available = False
                rebalance = False
                decision = "NO_REBALANCE_TOTAL_RETURN_INDEX_UNAVAILABLE"
            else:
                rebalance = self.should_rebalance_for_drift(
                    max_deviation, self.drift_threshold
                )
                decision = (
                    "DRIFT_THRESHOLD_TRIGGER"
                    if rebalance else "NO_REBALANCE_WITHIN_5PP_DRIFT"
                )
        accepted_target = dict(proposed_target) if rebalance else {}
        if rebalance:
            self.last_accepted_target = dict(proposed_target)
            self.last_reference_index = {
                code: current_indices[code] for code in proposed_target if code in current_indices
            }
            self.last_selected_satellites = list(selected)
            self.last_accepted_date = signal_date
        audit = {
            "strategy": "C1_CORE_SATELLITE_MOMENTUM",
            "signal_date": signal_date,
            "rebalance": bool(rebalance),
            "decision": decision,
            "reason": decision,
            "selection_changed": bool(selection_changed),
            "previous_selected_satellites": previous_selected,
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
            "cash_transfer": proposed["cash_transfer"],
            "cash_transfer_reasons": proposed["cash_transfer_reasons"],
            "covariance": proposed["covariance"],
            "covariance_coverage": proposed["covariance_coverage"],
            "covariance_start": proposed["covariance_start"],
            "covariance_end": proposed["covariance_end"],
            "predicted_volatility_pct": proposed["predicted_volatility_pct"],
            "satellite_scale": proposed["satellite_scale"],
            "scale_audit": proposed["scale_audit"],
            "drift_weights": drift,
            "max_drift_deviation_pct_points": max_deviation * 100.0,
            "index_available_for_drift": index_available,
            "proposed_target_weights": proposed_target,
            "target_weights": accepted_target,
            "accepted_target_date": self.last_accepted_date,
            "parameters": {
                "min_published_observations": self.min_observations,
                "momentum_short_days": self.short_days,
                "momentum_long_days": self.long_days,
                "trend_ma_days": self.ma_days,
                "covariance_days": self.covariance_days,
                "volatility_target": self.vol_target,
                "satellite_budget": self.satellite_budget,
                "buffer_threshold_pct_points": self.buffer_threshold * 100.0,
                "drift_threshold_pct_points": self.drift_threshold * 100.0,
            },
        }
        self.last_signal_audit = audit
        self.audit_rows.append(dict(audit))
        return accepted_target


def clone_rule_book_with_delay_override(
    rule_book: ProductRuleBook,
    fund_codes: list[str] | tuple[str, ...] | set[str],
    *,
    increment: int = 1,
) -> ProductRuleBook:
    """Clone actual hit rules so delay stress cannot be masked by overrides."""
    codes = {_normalise_code(code) for code in fund_codes}
    updated = {}
    for code, rule in rule_book.rules.items():
        if code in codes:
            updated[code] = replace(
                rule,
                subscription_confirmation_days=rule.subscription_confirmation_days + increment,
                redemption_confirmation_days=rule.redemption_confirmation_days + increment,
                redemption_settlement_days=rule.redemption_settlement_days + increment,
            )
        else:
            updated[code] = rule
    return ProductRuleBook(
        rules=updated,
        fee_tiers=rule_book.fee_tiers,
        subscription_fee_tiers=rule_book.subscription_fee_tiers,
        events=rule_book.events,
        allowed_rule_statuses=set(rule_book.allowed_rule_statuses),
    )


def sustained_turnover_excluding_initial(
    daily: pd.DataFrame, orders: pd.DataFrame, initial_signal_date: str | pd.Timestamp
) -> dict[str, Any]:
    """Compute confirmed annual turnover after removing initial-build orders."""
    if orders.empty:
        return {
            "excluded_initial_order_count": 0,
            "annual_confirmed_turnover": {},
            "max_annual_confirmed_turnover": 0.0,
        }
    frame = orders.copy()
    frame["signal_date"] = pd.to_datetime(frame.get("signal_date"), errors="coerce")
    frame["confirmation_date"] = pd.to_datetime(
        frame.get("confirmation_date"), errors="coerce"
    )
    frame["filled_notional"] = pd.to_numeric(
        frame.get("filled_notional", 0.0), errors="coerce"
    ).fillna(0.0)
    initial = pd.Timestamp(initial_signal_date)
    initial_orders = frame[frame["signal_date"] == initial]
    remaining = frame[
        (frame["signal_date"] != initial)
        & frame["status"].astype(str).str.lower().isin({"confirmed", "settled"})
        & frame["confirmation_date"].notna()
    ].copy()
    equity = daily.copy()
    equity["date"] = pd.to_datetime(equity["date"], errors="coerce")
    equity["equity"] = pd.to_numeric(equity["equity"], errors="coerce")
    equity_by_date = equity.set_index("date")["equity"].to_dict()
    annual: dict[str, float] = {}
    for date, group in remaining.groupby("confirmation_date"):
        denominator = float(equity_by_date.get(date, np.nan))
        if not np.isfinite(denominator) or denominator <= 0:
            continue
        year = str(pd.Timestamp(date).year)
        annual[year] = annual.get(year, 0.0) + float(group["filled_notional"].sum()) / denominator
    annual = {year: round(value, 6) for year, value in sorted(annual.items())}
    return {
        "excluded_initial_order_count": int(len(initial_orders)),
        "annual_confirmed_turnover": annual,
        "max_annual_confirmed_turnover": round(max(annual.values()) if annual else 0.0, 6),
    }
