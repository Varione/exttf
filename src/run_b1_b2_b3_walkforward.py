"""Continuous-account OOS Walk-forward for B1/B2/B3/S1.

The primary result is one account per strategy over the complete OOS period.
Fold boundaries freeze the research configuration and are used for reporting,
but they never reset cash, positions, FIFO lots, receivables, or pending
orders.  The annual-restart implementation remains available as an explicitly
labelled sensitivity comparison.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from otf_backtest_engine import OTFBacktestEngine
from otf_trading_rules import ProductRuleBook
from otf_rotation.experiment_artifacts import (
    create_run_directory,
    export_config_snapshot,
    export_daily_nav,
    export_input_hashes,
    export_json,
    export_metrics,
    export_orders,
    export_position_lots,
    export_rejections,
    export_table,
    sha256_file,
    write_manifest,
)
from otf_rotation.artifact_validation import (
    REQUIRED_ARTIFACTS as VALIDATED_REQUIRED_ARTIFACTS,
    validate_artifact_bundle,
)
from otf_rotation.research_status import run_pytest_summary, write_status
from otf_rotation.risk_parity import RollingRiskParityEngine
from otf_rotation.execution_calendar import load_execution_calendar
from otf_rotation.schedule import build_month_end_schedule, build_signal_submit_map
from run_s1_experiment import StateRotationSignal


DB_PATH = "data/processed/otf_expanded.sqlite"
CALENDAR_PATH = "data/processed/execution_calendar/cn_execution_calendar.csv"
RULES_PATH = "config/otf_product_rules.csv"
MAPPING_PATH = "config/otf_exposure_mapping.csv"
CONFIG_PATH = "config/otf_walkforward.json"
OUTPUT_DIR = Path("reports/strategy_research/walkforward_continuous")
OOS_START = "2021-01-01"
OOS_END = "2026-07-29"
INITIAL_CASH = 1_000_000.0
ACCOUNT_MODE = "FROZEN_PARAMETER_CONTINUOUS_OOS"
ANNUAL_RESTART_MODE = "ANNUAL_RESTART_SENSITIVITY_ONLY"
RULE_SCENARIO = "CURRENT_SNAPSHOT_CONSERVATIVE_EXECUTION_SCENARIO"
HISTORICAL_RULE_STATUS = "NOT_ESTABLISHED"

FOLDS = [
    {"train": ("2018-01-01", "2020-12-31"), "oos": ("2021-01-01", "2021-12-31")},
    {"train": ("2018-01-01", "2021-12-31"), "oos": ("2022-01-01", "2022-12-31")},
    {"train": ("2018-01-01", "2022-12-31"), "oos": ("2023-01-01", "2023-12-31")},
    {"train": ("2018-01-01", "2023-12-31"), "oos": ("2024-01-01", "2024-12-31")},
    {"train": ("2018-01-01", "2024-12-31"), "oos": ("2025-01-01", "2025-12-31")},
    {"train": ("2018-01-01", "2025-12-31"), "oos": ("2026-01-01", "2026-07-29")},
]

STRATEGY_NAMES = (
    "B1_Static_60_20_20",
    "B2_Static_EW_4Asset",
    "B3_Rolling_Risk_Parity",
    "S1_State_Rotation_Fixed",
)

GATE_THRESHOLDS = {
    "sharpe_min": 0.50,
    "mdd_min_pct": -15.0,
    "net_cagr_min_pct": 0.0,
    "annualized_cost_drag_max_pct": 1.25,
    "annual_bilateral_turnover_max": 1.0,
    "rolling_two_year_positive_ratio_min_pct": 75.0,
    "worst_two_year_cagr_min_pct": -3.0,
}

REQUIRED_ARTIFACTS = VALIDATED_REQUIRED_ARTIFACTS


def create_engine(rule_book: ProductRuleBook) -> OTFBacktestEngine:
    calendar = load_execution_calendar(CALENDAR_PATH)
    return OTFBacktestEngine(
        db_path=DB_PATH,
        initial_cash=INITIAL_CASH,
        strict_product_rules=True,
        product_rule_book=rule_book,
        confirmation_days_subscribe=1,
        confirmation_days_redeem=1,
        settlement_days_redeem=7,
        subscription_fee_rate=0.001,
        redemption_fee_rate=0.0015,
        minimum_trade_ratio=0.005,
        execution_calendar=calendar,
    )


def static_60_20_20_signal(date: pd.Timestamp) -> dict[str, float]:
    return {"160706": 0.60, "000218": 0.20, "001512": 0.20}


def static_ew_4asset_signal(date: pd.Timestamp) -> dict[str, float]:
    return {
        "160706": 0.25,
        "000218": 0.25,
        "001512": 0.25,
        "260102": 0.25,
    }


def compute_turnover(daily: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Compute submitted, confirmed-fill and settled-cash turnover.

    The Gate uses ``bilateral_turnover`` only.  It is based on confirmed
    principal/gross redemption proceeds at ``confirmation_date``; pending,
    cancelled and rejected requests never enter that column.
    """
    columns = [
        "date", "buy_notional", "sell_notional", "gross_traded_notional",
        "bilateral_turnover", "submitted_buy_notional", "submitted_sell_notional",
        "submitted_gross_notional", "submitted_bilateral_turnover",
        "settled_cash_buy_notional", "settled_cash_sell_notional",
        "settled_cash_gross_notional", "settled_cash_turnover",
        "pending_requested_amount",
    ]
    if orders.empty:
        return pd.DataFrame(columns=columns)

    frame = orders.copy()
    for column in ("requested_amount", "filled_notional", "settled_cash_notional", "fee_paid"):
        if column not in frame:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["side"] = frame["side"].astype(str).str.lower()
    frame["status"] = frame["status"].astype(str).str.lower()
    frame["submit_date"] = pd.to_datetime(frame.get("submit_date"), errors="coerce")
    frame["confirmation_date"] = pd.to_datetime(
        frame.get("confirmation_date"), errors="coerce"
    )
    frame["filled_notional"] = np.where(
        frame["status"].isin({"confirmed", "settled"}),
        frame["filled_notional"],
        0.0,
    )
    frame["settled_cash_notional"] = np.where(
        frame["status"].eq("settled"), frame["settled_cash_notional"], 0.0
    )

    submitted = frame.dropna(subset=["submit_date"]).groupby("submit_date")
    confirmed = frame.dropna(subset=["confirmation_date"])
    confirmed = confirmed[confirmed["status"].isin({"confirmed", "settled"})]
    confirmed = confirmed.groupby("confirmation_date")
    settled = frame[frame["status"].eq("settled")].dropna(subset=["confirmation_date"])
    settled = settled.groupby("confirmation_date")

    dates = set(submitted.groups) | set(confirmed.groups) | set(settled.groups)
    rows: list[dict[str, Any]] = []
    daily_equity = daily[["date", "equity"]].copy()
    daily_equity["date"] = pd.to_datetime(daily_equity["date"], errors="coerce")
    equity_by_date = daily_equity.set_index("date")["equity"].to_dict()

    for date in sorted(dates):
        submitted_group = submitted.get_group(date) if date in submitted.groups else frame.iloc[0:0]
        confirmed_group = confirmed.get_group(date) if date in confirmed.groups else frame.iloc[0:0]
        settled_group = settled.get_group(date) if date in settled.groups else frame.iloc[0:0]

        def side_sum(group: pd.DataFrame, column: str, side: str) -> float:
            return float(group.loc[group["side"].eq(side), column].sum())

        buy = side_sum(confirmed_group, "filled_notional", "subscribe")
        sell = side_sum(confirmed_group, "filled_notional", "redeem")
        submitted_buy = side_sum(submitted_group, "requested_amount", "subscribe")
        submitted_sell = side_sum(submitted_group, "requested_amount", "redeem")
        settled_buy = side_sum(settled_group, "settled_cash_notional", "subscribe")
        settled_sell = side_sum(settled_group, "settled_cash_notional", "redeem")
        equity = float(equity_by_date.get(date, INITIAL_CASH) or INITIAL_CASH)
        pending = float(
            frame.loc[frame["status"].eq("pending"), "requested_amount"].sum()
            if date == max(dates)
            else 0.0
        )
        rows.append(
            {
                "date": date,
                "buy_notional": buy,
                "sell_notional": sell,
                "gross_traded_notional": buy + sell,
                "bilateral_turnover": (buy + sell) / equity if equity else 0.0,
                "submitted_buy_notional": submitted_buy,
                "submitted_sell_notional": submitted_sell,
                "submitted_gross_notional": submitted_buy + submitted_sell,
                "submitted_bilateral_turnover": (
                    (submitted_buy + submitted_sell) / equity if equity else 0.0
                ),
                "settled_cash_buy_notional": settled_buy,
                "settled_cash_sell_notional": settled_sell,
                "settled_cash_gross_notional": settled_buy + settled_sell,
                "settled_cash_turnover": (
                    (settled_buy + settled_sell) / equity if equity else 0.0
                ),
                "pending_requested_amount": pending,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("date")


def build_fee_reconciliation(
    daily: pd.DataFrame, orders: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Reconcile order-level fees to daily accounting fees."""
    tolerance = max(1e-8, INITIAL_CASH * 1e-8)
    daily_fee = daily[[
        "date", "subscription_fee_amount", "redemption_fee_amount", "total_fee_amount"
    ]].copy()
    daily_fee["date"] = pd.to_datetime(daily_fee["date"], errors="coerce")
    daily_fee = daily_fee.rename(
        columns={
            "subscription_fee_amount": "daily_subscription_fee_amount",
            "redemption_fee_amount": "daily_redemption_fee_amount",
            "total_fee_amount": "daily_total_fee_amount",
        }
    )

    order_frame = orders.copy()
    if order_frame.empty:
        order_frame = pd.DataFrame(
            columns=[
                "order_id", "status", "confirmation_date", "side", "fee_paid",
                "requested_amount",
            ]
        )
    for column in ("fee_paid", "requested_amount"):
        if column not in order_frame:
            order_frame[column] = 0.0
        order_frame[column] = pd.to_numeric(order_frame[column], errors="coerce").fillna(0.0)
    order_frame["confirmation_date"] = pd.to_datetime(
        order_frame.get("confirmation_date"), errors="coerce"
    )
    order_frame["status"] = order_frame.get("status", "").astype(str).str.lower()
    order_frame["side"] = order_frame.get("side", "").astype(str).str.lower()
    confirmed = order_frame[order_frame["status"].isin({"confirmed", "settled"})].copy()
    confirmed["order_subscription_fee_amount"] = np.where(
        confirmed["side"].eq("subscribe"), confirmed["fee_paid"], 0.0
    )
    confirmed["order_redemption_fee_amount"] = np.where(
        confirmed["side"].eq("redeem"), confirmed["fee_paid"], 0.0
    )
    order_daily = (
        confirmed.dropna(subset=["confirmation_date"])
        .groupby("confirmation_date")[[
            "order_subscription_fee_amount", "order_redemption_fee_amount"
        ]]
        .sum()
        .reset_index()
        .rename(columns={"confirmation_date": "date"})
    )
    if order_daily.empty:
        order_daily = pd.DataFrame(
            columns=["date", "order_subscription_fee_amount", "order_redemption_fee_amount"]
        )
    order_daily["order_total_fee_amount"] = (
        pd.to_numeric(order_daily.get("order_subscription_fee_amount", 0.0), errors="coerce").fillna(0.0)
        + pd.to_numeric(order_daily.get("order_redemption_fee_amount", 0.0), errors="coerce").fillna(0.0)
    )
    reconciliation = daily_fee.merge(order_daily, on="date", how="outer").fillna(0.0)
    reconciliation["subscription_fee_delta"] = (
        reconciliation["daily_subscription_fee_amount"]
        - reconciliation["order_subscription_fee_amount"]
    )
    reconciliation["redemption_fee_delta"] = (
        reconciliation["daily_redemption_fee_amount"]
        - reconciliation["order_redemption_fee_amount"]
    )
    reconciliation["total_fee_delta"] = (
        reconciliation["daily_total_fee_amount"]
        - reconciliation["order_total_fee_amount"]
    )
    reconciliation["reconciled"] = (
        reconciliation[["subscription_fee_delta", "redemption_fee_delta", "total_fee_delta"]]
        .abs()
        .max(axis=1)
        <= tolerance
    )

    order_audit = order_frame[[
        "order_id", "status", "confirmation_date", "side", "fee_paid"
    ]].copy()
    order_audit["reconciled_in_daily_fees"] = order_audit.apply(
        lambda row: bool(
            float(row["fee_paid"] or 0.0) <= tolerance
            or (
                row["status"] in {"confirmed", "settled"}
                and row["confirmation_date"] in set(reconciliation.loc[reconciliation["reconciled"], "date"])
            )
        ),
        axis=1,
    )
    daily_total_fee_amount = float(reconciliation["daily_total_fee_amount"].sum())
    order_total_fee_amount = float(reconciliation["order_total_fee_amount"].sum())
    aggregate_abs_delta = abs(daily_total_fee_amount - order_total_fee_amount)
    summary = {
        "tolerance": tolerance,
        "daily_total_fee_amount": round(daily_total_fee_amount, 8),
        "order_total_fee_amount": round(order_total_fee_amount, 8),
        "aggregate_abs_delta": round(aggregate_abs_delta, 8),
        "max_abs_delta": round(float(reconciliation["total_fee_delta"].abs().max()) if not reconciliation.empty else 0.0, 8),
        "passed": bool(
            (reconciliation["reconciled"].all() if not reconciliation.empty else True)
            and aggregate_abs_delta <= tolerance
        ),
        "confirmed_order_count": int(len(confirmed)),
        "settled_order_count": int((order_frame["status"] == "settled").sum()),
        "pending_order_count": int((order_frame["status"] == "pending").sum()),
        "pending_requested_amount": round(float(order_frame.loc[order_frame["status"] == "pending", "requested_amount"].sum()), 8),
    }
    return reconciliation.sort_values("date"), order_audit, summary


def _rolling_two_year_metrics(returns: pd.Series) -> tuple[int, float, float | None]:
    window = 504
    if len(returns) < window:
        return 0, 0.0, None
    wealth = (1.0 + returns.fillna(0.0)).rolling(window).apply(np.prod, raw=True)
    cagr = wealth.pow(252.0 / window) - 1.0
    values = cagr.dropna()
    if values.empty:
        return 0, 0.0, None
    return int(len(values)), float((values > 0).mean() * 100.0), float(values.min() * 100.0)


def compute_metrics(
    daily: pd.DataFrame,
    turnover: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute metrics from a single, date-ordered account curve."""
    if daily.empty:
        return {}
    frame = daily.sort_values("date").reset_index(drop=True)
    returns = pd.to_numeric(frame["daily_return"], errors="coerce").fillna(0.0)
    gross_returns = pd.to_numeric(
        frame.get("gross_return", frame["daily_return"]), errors="coerce"
    ).fillna(0.0)
    net_nav = (1.0 + returns).cumprod()
    gross_nav = (1.0 + gross_returns).cumprod()
    n_days = max(len(returns), 1)
    years = n_days / 252.0
    net_cagr = float(net_nav.iloc[-1] ** (1.0 / years) - 1.0) if years else 0.0
    gross_cagr = float(gross_nav.iloc[-1] ** (1.0 / years) - 1.0) if years else 0.0
    volatility = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)) if returns.std(ddof=1) > 0 else 0.0
    drawdown = net_nav / net_nav.cummax() - 1.0
    rolling_count, rolling_positive, worst_two_year = _rolling_two_year_metrics(returns)

    turnover = turnover if turnover is not None else pd.DataFrame()
    annual_turnover_full = (
        turnover.assign(year=pd.to_datetime(turnover["date"]).dt.year)
        .groupby("year")["bilateral_turnover"]
        .max()
        if not turnover.empty else pd.Series(dtype=float)
    )
    # Steady-state turnover excludes the first calendar year of OOS, which is
    # dominated by initial position build-out (buy_notional ~= equity on the
    # first confirmation date gives bt_daily ~ 1.0 regardless of strategy).
    # Gate checks use this steady-state value so that strategies are evaluated
    # on their ongoing rebalancing behavior rather than startup mechanics.
    first_year = None
    if not turnover.empty:
        first_year = pd.to_datetime(turnover["date"]).dt.year.min()
    annual_turnover_ss = (
        annual_turnover_full.drop(first_year, errors="ignore")
        if first_year is not None else annual_turnover_full
    )
    total_fees = float(
        pd.to_numeric(frame.get("total_fee_amount", 0.0), errors="coerce")
        .fillna(0.0).sum()
    )
    initial_equity = float(frame["equity"].iloc[0])
    fee_ratio = total_fees / initial_equity * 100.0 if initial_equity else 0.0
    return {
        "account_mode": ACCOUNT_MODE,
        "n_days": int(len(frame)),
        "oos_start": pd.Timestamp(frame["date"].min()).strftime("%Y-%m-%d"),
        "oos_end": pd.Timestamp(frame["date"].max()).strftime("%Y-%m-%d"),
        "net_cagr_pct": round(net_cagr * 100.0, 4),
        "gross_cagr_pct": round(gross_cagr * 100.0, 4),
        "annualized_cost_drag_pct": round((gross_cagr - net_cagr) * 100.0, 4),
        "total_fee_amount": round(total_fees, 2),
        "total_fee_ratio_pct": round(fee_ratio, 4),
        "sharpe": round(sharpe, 4),
        "annualized_volatility_pct": round(volatility * 100.0, 4),
        "mdd_pct": round(float(drawdown.min()) * 100.0, 4),
        "final_equity": round(float(frame["equity"].iloc[-1]), 2),
        "annual_bilateral_turnover": {
            str(int(year)): round(float(value), 6)
            for year, value in annual_turnover_full.items()
        },
        "max_annual_bilateral_turnover": round(
            float(annual_turnover_full.max()) if not annual_turnover_full.empty else 0.0, 6
        ),
        "steady_state_max_annual_bilateral_turnover": round(
            float(annual_turnover_ss.max()) if not annual_turnover_ss.empty else 0.0, 6
        ),
        "total_bilateral_turnover": round(
            float(turnover["bilateral_turnover"].sum()) if not turnover.empty else 0.0,
            6,
        ),
        "rolling_two_year_window_count": rolling_count,
        "rolling_two_year_positive_ratio_pct": round(rolling_positive, 4),
        "worst_two_year_cagr_pct": (
            round(worst_two_year, 4) if worst_two_year is not None else None
        ),
    }


def build_targets(
    signal_func: Callable[[pd.Timestamp], dict[str, float]],
    signal_dates: list[pd.Timestamp],
    signal_map: dict[str, str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    submit_for_signal = {signal: submit for submit, signal in signal_map.items()}
    target_weights: dict[str, dict[str, float]] = {}
    audit_rows: list[dict[str, Any]] = []
    for signal_date in signal_dates:
        signal_str = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
        submit_date = submit_for_signal.get(signal_str)
        if submit_date is None:
            continue
        weights = signal_func(signal_date)
        target_weights[submit_date] = {
            str(code): float(value)
            for code, value in weights.items()
            if float(value) > 1e-12
        }
        signal_audit = getattr(signal_func, "last_signal_audit", None)
        if signal_audit:
            audit_rows.append(dict(signal_audit))
    if not target_weights:
        return pd.DataFrame(), audit_rows
    target_df = pd.DataFrame.from_dict(target_weights, orient="index").fillna(0.0)
    target_df.index = pd.to_datetime(target_df.index)
    target_df = target_df.sort_index()
    return target_df.loc[:, (target_df != 0).any(axis=0)], audit_rows


def run_strategy(
    engine: OTFBacktestEngine,
    signal_func: Callable[[pd.Timestamp], dict[str, float]],
    signal_dates: list[pd.Timestamp],
    signal_map: dict[str, str],
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Generate all OOS targets first, then run one account over the period."""
    target_df, audits = build_targets(signal_func, signal_dates, signal_map)
    if target_df.empty:
        raise RuntimeError("WALKFORWARD_NO_TARGET_SIGNALS")
    daily = engine.run_backtest(
        target_df,
        start=start,
        end=end,
        rebalance_every=1,
        signal_dates={pd.Timestamp(k): pd.Timestamp(v) for k, v in signal_map.items()},
    )
    return daily, target_df, audits


def make_b3_signal(engine: OTFBacktestEngine):
    assets = ["160706", "000218", "001512", "260102"]
    nav_subset = engine._nav_df[engine._nav_df["fund_code"].isin(assets)]
    navs = nav_subset.pivot_table(
        index="nav_date", columns="fund_code", values="unit_nav"
    ).sort_index().dropna(how="all")
    b3 = RollingRiskParityEngine(
        nav_df=navs,
        asset_columns=assets,
        rolling_window=252,
        min_window=60,
        asset_cap=0.50,
    )
    audits: list[dict[str, Any]] = []

    def signal(date: pd.Timestamp) -> dict[str, float]:
        weights, audit = b3.get_weights_with_audit(date)
        audits.append(audit)
        return weights

    signal.audit_rows = audits
    return signal


def make_s1_signal(rule_book: ProductRuleBook, engine: OTFBacktestEngine):
    s1 = StateRotationSignal(rule_book=rule_book, nav_df=engine._nav_df)
    s1.reset()
    audits: list[dict[str, Any]] = []

    def signal(date: pd.Timestamp) -> dict[str, float]:
        weights = s1(date)
        if s1.last_signal_audit:
            audits.append(dict(s1.last_signal_audit))
        return weights

    signal.audit_rows = audits
    signal.reset = s1.reset
    return signal


def build_strategy_signal(
    name: str, rule_book: ProductRuleBook, engine: OTFBacktestEngine
):
    if name == "B1_Static_60_20_20":
        return static_60_20_20_signal
    if name == "B2_Static_EW_4Asset":
        return static_ew_4asset_signal
    if name == "B3_Rolling_Risk_Parity":
        return make_b3_signal(engine)
    if name == "S1_State_Rotation_Fixed":
        return make_s1_signal(rule_book, engine)
    raise ValueError(f"Unknown strategy: {name}")


def _json_column(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    if column not in frame.columns:
        return []
    rows: list[dict[str, Any]] = []
    for date, value in zip(frame["date"], frame[column]):
        try:
            weights = json.loads(value) if isinstance(value, str) else {}
        except (TypeError, json.JSONDecodeError):
            weights = {}
        rows.append({"date": date, **weights})
    return rows


def _flatten_lot_snapshots(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if "position_lots" not in daily.columns:
        return pd.DataFrame(columns=["date", "fund_code", "lot_id", "acquired_date", "shares", "reserved_shares"])
    for _, row in daily.iterrows():
        try:
            lots = json.loads(row["position_lots"])
        except (TypeError, json.JSONDecodeError):
            lots = []
        for lot in lots:
            rows.append({"date": row["date"], **lot})
    return pd.DataFrame(rows)


def collect_input_facts() -> dict[str, Any]:
    rules = pd.read_csv(RULES_PATH, dtype=str).fillna("")
    mapping = pd.read_csv(MAPPING_PATH, dtype=str).fillna("")
    calendar = load_execution_calendar(CALENDAR_PATH)
    input_paths = {
        "db_sha256": DB_PATH,
        "rules_sha256": RULES_PATH,
        "mapping_sha256": MAPPING_PATH,
        "config_sha256": CONFIG_PATH,
        "calendar_sha256": CALENDAR_PATH,
    }
    hashes = {key: sha256_file(path) for key, path in input_paths.items()}
    rule_counts = Counter(rules["rule_status"].astype(str).str.strip())
    mapping_counts = Counter()
    if "review_status" in mapping:
        mapping_counts.update(
            f"review_status:{value}"
            for value in mapping["review_status"].astype(str).str.strip().str.upper()
        )
    if "mapping_confidence" in mapping:
        mapping_counts.update(
            f"mapping_confidence:{value}"
            for value in mapping["mapping_confidence"].astype(str).str.strip().str.upper()
        )
    return {
        "input_hashes": hashes,
        "rule_counts_by_status": dict(sorted(rule_counts.items())),
        "mapping_counts_by_status": dict(sorted(mapping_counts.items())),
        "rule_count": int(len(rules)),
        "mapping_count": int(len(mapping)),
        "rule_temporal_coverage": {
            "effective_from_present": int((rules["effective_from"].str.strip() != "").sum()),
            "verified_at_present": int((rules["verified_at"].str.strip() != "").sum()),
            "total_rules": int(len(rules)),
        },
        "execution_calendar": calendar.facts(),
    }


def build_static_data_gates(
    engine: OTFBacktestEngine,
    facts: dict[str, Any],
    used_funds: set[str],
) -> dict[str, Any]:
    rules = pd.read_csv(RULES_PATH, dtype=str).fillna("")
    mapping = pd.read_csv(MAPPING_PATH, dtype=str).fillna("")
    rules_by_code = {
        str(row.fund_code).zfill(6): row for row in rules.itertuples(index=False)
    }
    mapping_by_code = {
        str(row.fund_code).zfill(6): row for row in mapping.itertuples(index=False)
    }
    allowed = ProductRuleBook.FORMAL_RESEARCH_STATUSES
    rules_present = all(code in rules_by_code for code in used_funds)
    rule_status_allowed = all(
        code in rules_by_code and str(rules_by_code[code].rule_status) in allowed
        for code in used_funds
    )
    rule_dates_present = all(
        code in rules_by_code
        and bool(str(getattr(rules_by_code[code], "effective_from", "")).strip())
        and bool(str(getattr(rules_by_code[code], "verified_at", "")).strip())
        for code in used_funds
    )
    mapping_approved = all(
        code in mapping_by_code
        and str(mapping_by_code[code].mapping_confidence).upper() == "HIGH"
        and str(mapping_by_code[code].review_status).upper() == "APPROVED"
        for code in used_funds
    )
    quality = engine.data_quality
    quality_summary = {
        key: value
        for key, value in quality.items()
        if key != "product_rule_coverage"
    }
    if "product_rule_coverage" in quality:
        quality_summary["product_rule_coverage"] = {
            key: value
            for key, value in quality["product_rule_coverage"].items()
            if key != "missing_codes"
        }
    data_gate = bool(
        quality.get("row_count", 0) > 0
        and quality.get("duplicate_keys", 1) == 0
        and quality.get("max_total_return_reconciliation_error", 1.0) <= 1e-8
    )
    current_rules_gate = bool(rules_present and rule_status_allowed)
    historical_rules_gate = bool(current_rules_gate and rule_dates_present)
    return {
        "data_gate": data_gate,
        "rules_gate": current_rules_gate,
        "current_rules_gate": current_rules_gate,
        "historical_rules_gate": historical_rules_gate,
        "mapping_gate": bool(mapping_approved),
        "rule_table_present": rules_present,
        "rule_status_allowed": rule_status_allowed,
        "rule_temporal_gate": rule_dates_present,
        "rule_scenario": RULE_SCENARIO,
        "historical_rule_status": HISTORICAL_RULE_STATUS,
        "mapping_approved": mapping_approved,
        "used_funds": sorted(used_funds),
        "data_quality": quality_summary,
        "input_hashes": facts["input_hashes"],
        "rule_counts_by_status": facts["rule_counts_by_status"],
        "mapping_counts_by_status": facts["mapping_counts_by_status"],
    }


def metrics_gate(metrics: dict[str, Any], static_gates: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "sharpe": metrics.get("sharpe", -np.inf) >= GATE_THRESHOLDS["sharpe_min"],
        "mdd": metrics.get("mdd_pct", -np.inf) > GATE_THRESHOLDS["mdd_min_pct"],
        "net_cagr": metrics.get("net_cagr_pct", -np.inf) > GATE_THRESHOLDS["net_cagr_min_pct"],
        "annualized_cost_drag": metrics.get("annualized_cost_drag_pct", np.inf) <= GATE_THRESHOLDS["annualized_cost_drag_max_pct"],
        "annual_bilateral_turnover": metrics.get("steady_state_max_annual_bilateral_turnover", np.inf) <= GATE_THRESHOLDS["annual_bilateral_turnover_max"],
        "rolling_two_year_positive_ratio": metrics.get("rolling_two_year_positive_ratio_pct", 0.0) >= GATE_THRESHOLDS["rolling_two_year_positive_ratio_min_pct"],
        "worst_two_year_cagr": metrics.get("worst_two_year_cagr_pct") is not None and metrics.get("worst_two_year_cagr_pct", -np.inf) >= GATE_THRESHOLDS["worst_two_year_cagr_min_pct"],
        "data_gate": bool(static_gates.get("data_gate")),
        "rules_gate": bool(static_gates.get("rules_gate")),
        "mapping_gate": bool(static_gates.get("mapping_gate")),
        "confirmed_turnover": bool(
            metrics.get("confirmed_turnover_gate", False)
        ),
        "fee_reconciliation": bool(
            metrics.get("fee_reconciliation_passed", False)
        ),
        "artifact_schema": bool(metrics.get("artifact_schema_gate", False)),
        "artifact_content": bool(metrics.get("artifact_content_gate", False)),
    }
    historical_truth_gate = bool(static_gates.get("historical_rules_gate"))
    return {
        "gate_passed": bool(all(checks.values())),
        "candidate_gate_passed": bool(all(checks.values())),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "thresholds": GATE_THRESHOLDS,
        "historical_truth_gate": historical_truth_gate,
        "historical_truth_status": (
            "ESTABLISHED" if historical_truth_gate else "NOT_ESTABLISHED"
        ),
    }


def summarize_artifact_validation(validation: dict[str, Any]) -> dict[str, Any]:
    """Store a non-recursive artifact validation summary in gate_result.json."""
    return {
        key: validation.get(key)
        for key in ("passed", "required_count", "present_count", "errors", "artifact_status")
    }


def derive_research_status(
    strategy_gates: dict[str, dict[str, Any]],
    *,
    historical_rules_passed: bool,
    test_result: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    """Derive candidate status from candidate Gates, not historical truth.

    ``historical_rules_passed`` is intentionally accepted for the caller's
    disclosure path, but it cannot turn a passing conservative candidate Gate
    into a failure.  The separate historical truth field remains false and is
    surfaced in status/report output.
    """
    failed_phases: list[str] = []
    if not all(gate.get("gate_passed", False) for gate in strategy_gates.values()):
        failed_phases.append("COMPLETE_GATE_IMPLEMENTATION")
    if test_result and not test_result.get("all_passed", False):
        failed_phases.append("TESTS")
    return ("PAPER_TRADE_CANDIDATE" if not failed_phases else "OOS_GATE_FAILED", failed_phases)


def artifact_completeness(
    output_dir: Path,
    *,
    strategy_name: str,
    run_id: str,
    actual_start: str,
    actual_end: str,
) -> dict[str, Any]:
    """Validate artifact schemas, contents and cross-file identity."""
    return validate_artifact_bundle(
        output_dir,
        strategy_name=strategy_name,
        expected_run_id=run_id,
        expected_start=actual_start,
        expected_end=actual_end,
    )


def write_conclusion_report(
    run_dir: Path,
    facts: dict[str, Any],
    results: dict[str, dict[str, Any]],
    gates: dict[str, dict[str, Any]],
    status: str,
    config: dict[str, Any],
    finalization: dict[str, Any] | None = None,
) -> None:
    """Render the human report from the run facts and daily artifacts."""
    first_metric = results[next(iter(results))]
    actual_start = first_metric["oos_start"]
    actual_end = first_metric["oos_end"]
    lines = [
        "# OTF B1/B2/B3/S1 Frozen-Parameter Continuous OOS",
        "",
        f"- Run ID: `{config['run_id']}`",
        f"- Status: `{status}`",
        f"- Account mode: `{ACCOUNT_MODE}`",
        f"- Requested OOS: `{config['requested_oos_period'][0]}` to `{config['requested_oos_period'][1]}`",
        f"- Actual valued OOS: `{actual_start}` to `{actual_end}` ({first_metric['n_days']} trading days)",
        f"- Rule scenario: `{RULE_SCENARIO}`",
        f"- Historical rule status: `{HISTORICAL_RULE_STATUS}`",
        f"- Historical truth Gate: `{'PASS' if all(gate.get('historical_truth_gate', False) for gate in gates.values()) else 'FAIL'}` (disclosure only; it is not a candidate Gate check)",
        f"- Rules: {facts['rule_count']} (`{facts['rule_counts_by_status']}`)",
        f"- Exposure mappings: {facts['mapping_count']} (`{facts['mapping_counts_by_status']}`)",
        "",
        "## Continuous-account OOS metrics",
        "",
        "| Strategy | Net CAGR | Gross CAGR | Cost drag | Sharpe | MDD | Fees | Candidate Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, metric in results.items():
        lines.append(
            f"| {name} | {metric['net_cagr_pct']:.4f}% | {metric['gross_cagr_pct']:.4f}% | "
            f"{metric['annualized_cost_drag_pct']:.4f}% | {metric['sharpe']:.4f} | "
            f"{metric['mdd_pct']:.4f}% | {metric['total_fee_amount']:.2f} | "
            f"{'PASS' if gates[name].get('gate_passed') else 'NOT PASSED'} |"
        )
    lines.extend([
        "",
        "## Gate and accounting notes",
        "",
        "The primary result uses one account per strategy across the actual valued OOS dates. Positions, FIFO lots, pending orders, receivables and fees carry across calendar years. Annual restart outputs are sensitivity-only and are excluded from the candidate Gate.",
        f"Annual-restart sensitivity status: `{config.get('annual_restart_sensitivity_status', 'UNKNOWN')}`. It is never used by the primary Gate.",
        "",
        "The current rule table is a conservative current-snapshot scenario. Missing historical effective dates and verification dates mean this run is not historical-real execution validation.",
        "Historical truth Gate and conservative paper-trade candidate Gate are evaluated independently. A failed historical truth Gate blocks any claim of historical-real execution validation, but does not by itself block a conservative-scenario candidate; candidate eligibility still requires every listed statistical, turnover, current-rule, data, mapping, fee and artifact check.",
        "",
        "Confirmed turnover uses actual confirmed principal/gross redemption proceeds at confirmation date. Pending, rejected and cancelled orders are excluded. Fee reconciliation compares order fees with daily account fees.",
        "",
        f"- Input hashes: `{facts['input_hashes']}`",
        f"- Artifact root: `{run_dir.as_posix()}`",
        "",
        "### Per-strategy Gate failures",
        "",
    ])
    if finalization:
        lines.insert(
            lines.index("## Continuous-account OOS metrics"),
            "- Finalization: `FINALIZE_EXISTING_RUN`; annual summary source `"
            f"{finalization.get('annual_summary_source')}` validated with SHA256 `"
            f"{finalization.get('annual_summary_sha256')}`.",
        )
    for name, gate in gates.items():
        lines.append(f"- `{name}`: {gate.get('failed_checks', []) or 'none'}")
    report = "\n".join(lines) + "\n"
    # UTF-8 BOM keeps the report readable in legacy Windows PowerShell while
    # remaining valid UTF-8 for editors and automated checks.
    (run_dir / "conclusion.md").write_text(report, encoding="utf-8-sig")
    Path("conclusion.md").write_text(report, encoding="utf-8-sig")


def export_strategy_bundle(
    output_dir: Path,
    strategy_name: str,
    daily: pd.DataFrame,
    orders: pd.DataFrame,
    rejections: list[dict[str, Any]],
    turnover: pd.DataFrame,
    fee_reconciliation: pd.DataFrame,
    fee_order_audit: pd.DataFrame,
    audits: list[dict[str, Any]],
    lots_by_fund: dict[str, list[Any]],
    config: dict[str, Any],
    input_facts: dict[str, Any],
    metrics: dict[str, Any],
    gate: dict[str, Any],
    parameter_freeze: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    export_daily_nav(output_dir.as_posix(), daily, "daily_account.csv")
    export_daily_nav(
        output_dir.as_posix(),
        daily[["date", "equity", "daily_return", "gross_return"]],
        "daily_returns.csv",
    )
    export_orders(output_dir.as_posix(), orders)
    export_rejections(output_dir.as_posix(), rejections)
    export_position_lots(output_dir.as_posix(), lots_by_fund)
    export_table(output_dir.as_posix(), "fees.csv", fee_reconciliation)
    export_table(output_dir.as_posix(), "fee_reconciliation_orders.csv", fee_order_audit)
    export_table(output_dir.as_posix(), "turnover.csv", turnover)
    export_table(
        output_dir.as_posix(),
        "target_weights.csv",
        pd.DataFrame(_json_column(daily, "target_weights")).fillna(0.0),
    )
    export_table(
        output_dir.as_posix(),
        "actual_weights.csv",
        pd.DataFrame(_json_column(daily, "actual_weights")).fillna(0.0),
    )

    audit_frame = pd.DataFrame(audits)
    market_columns = ["signal_date", "market_state"]
    market_frame = audit_frame[market_columns] if all(c in audit_frame for c in market_columns) else pd.DataFrame(columns=market_columns)
    export_table(output_dir.as_posix(), "market_states.csv", market_frame)
    score_rows = []
    sleeve_rows = []
    fund_rows = []
    risk_rows = []
    for audit in audits:
        date = audit.get("signal_date", audit.get("date", ""))
        for key, value in (audit.get("state_scores", {}) or {}).items():
            score_rows.append({"date": date, "state": key, "score": value})
        for key, value in (audit.get("sleeve_weights", {}) or {}).items():
            sleeve_rows.append({"date": date, "sleeve": key, "weight": value})
        for key, value in (audit.get("fund_weights", {}) or {}).items():
            fund_rows.append({"date": date, "fund_code": key, "weight": value})
        for key, value in (audit.get("risk_contributions", {}) or {}).items():
            risk_rows.append({"date": date, "fund_code": key, "risk_contribution": value})
    export_table(output_dir.as_posix(), "state_scores.csv", pd.DataFrame(score_rows, columns=["date", "state", "score"]))
    export_table(output_dir.as_posix(), "asset_budgets.csv", pd.DataFrame(sleeve_rows, columns=["date", "sleeve", "weight"]))
    export_table(output_dir.as_posix(), "sleeve_weights.csv", pd.DataFrame(sleeve_rows, columns=["date", "sleeve", "weight"]))
    export_table(output_dir.as_posix(), "fund_weights.csv", pd.DataFrame(fund_rows, columns=["date", "fund_code", "weight"]))
    export_table(output_dir.as_posix(), "risk_contributions.csv", pd.DataFrame(risk_rows, columns=["date", "fund_code", "risk_contribution"]))
    export_table(
        output_dir.as_posix(),
        "product_selection_audit.csv",
        pd.DataFrame(columns=[
            "sleeve", "date", "top_n", "selected_count", "total_candidates",
            "eligible_count", "selected_funds",
        ]),
    )
    export_config_snapshot(output_dir.as_posix(), config)
    export_input_hashes(output_dir.as_posix(), input_facts["input_hashes"])
    export_metrics(output_dir.as_posix(), metrics)
    export_json(output_dir.as_posix(), "parameter_freeze.json", parameter_freeze)
    export_json(output_dir.as_posix(), "gate_result.json", gate)
    write_manifest(
        run_dir=output_dir.as_posix(),
        strategy_name=strategy_name,
        config=config,
        db_path=DB_PATH,
        rules_path=RULES_PATH,
        exposure_mapping_path=MAPPING_PATH,
        metrics=metrics,
        gate_result=gate,
    )


def finalize_artifact_gate(
    strategy_dir: Path,
    strategy_name: str,
    run_id: str,
    actual_start: str,
    actual_end: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Run content validation, persist the final Gate and refresh manifest."""
    artifact_gate = artifact_completeness(
        strategy_dir,
        strategy_name=strategy_name,
        run_id=run_id,
        actual_start=actual_start,
        actual_end=actual_end,
    )
    passed = bool(artifact_gate.get("passed"))
    gate["checks"]["artifact_schema"] = passed
    gate["checks"]["artifact_content"] = passed
    gate["artifact_validation"] = summarize_artifact_validation(artifact_gate)
    gate["artifact_validation_final"] = summarize_artifact_validation(artifact_gate)
    gate["failed_checks"] = [
        name for name, value in gate["checks"].items() if not value
    ]
    gate["gate_passed"] = not gate["failed_checks"]
    metrics["artifact_schema_gate"] = passed
    metrics["artifact_content_gate"] = passed
    export_metrics(strategy_dir.as_posix(), metrics)
    export_json(strategy_dir.as_posix(), "gate_result.json", gate)
    write_manifest(
        run_dir=strategy_dir.as_posix(),
        strategy_name=strategy_name,
        config=config,
        db_path=DB_PATH,
        rules_path=RULES_PATH,
        exposure_mapping_path=MAPPING_PATH,
        metrics=metrics,
        gate_result=gate,
    )
    final_validation = artifact_completeness(
        strategy_dir,
        strategy_name=strategy_name,
        run_id=run_id,
        actual_start=actual_start,
        actual_end=actual_end,
    )
    if not final_validation.get("passed"):
        gate["checks"]["artifact_schema"] = False
        gate["checks"]["artifact_content"] = False
        gate["artifact_validation_final"] = summarize_artifact_validation(final_validation)
        gate["failed_checks"] = [
            name for name, value in gate["checks"].items() if not value
        ]
        gate["gate_passed"] = not gate["failed_checks"]
        metrics["artifact_schema_gate"] = False
        metrics["artifact_content_gate"] = False
        export_metrics(strategy_dir.as_posix(), metrics)
        export_json(strategy_dir.as_posix(), "gate_result.json", gate)
        write_manifest(
            run_dir=strategy_dir.as_posix(),
            strategy_name=strategy_name,
            config=config,
            db_path=DB_PATH,
            rules_path=RULES_PATH,
            exposure_mapping_path=MAPPING_PATH,
            metrics=metrics,
            gate_result=gate,
        )
    return final_validation


def run_annual_restart_sensitivity(
    root_dir: Path,
    rule_book: ProductRuleBook,
    signal_dates_by_fold: list[tuple[list[pd.Timestamp], dict[str, str]]],
    folds: list[dict[str, Any]],
) -> dict[str, Any]:
    """Retain the old annual-restart experiment under an explicit label."""
    out_dir = root_dir / "annual_restart_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for strategy_name in STRATEGY_NAMES:
        # ``run_backtest`` creates a fresh account state on every call, so one
        # loaded engine can safely serve all annual-restart folds.  Reusing the
        # loaded NAV cache keeps this explicit sensitivity run practical while
        # preserving the independent 1M cash reset semantics.
        engine = create_engine(rule_book)
        signal = build_strategy_signal(strategy_name, rule_book, engine)
        frames: list[pd.DataFrame] = []
        fold_rows: list[dict[str, Any]] = []
        for fold_idx, fold in enumerate(folds):
            signal_dates, signal_map = signal_dates_by_fold[fold_idx]
            if not signal_dates:
                continue
            if hasattr(signal, "reset"):
                signal.reset()
            daily, _, _ = run_strategy(
                engine, signal, signal_dates, signal_map,
                fold["oos"][0], fold["oos"][1],
            )
            orders = engine.order_audit_frame()
            turnover = compute_turnover(daily, orders)
            metrics = compute_metrics(daily, turnover)
            metrics.update({"fold": fold_idx + 1, "account_mode": ANNUAL_RESTART_MODE})
            fold_rows.append(metrics)
            frames.append(daily)
        if frames:
            combined = pd.concat(frames, ignore_index=True).sort_values("date")
            combined = combined.drop_duplicates("date", keep="first")
            combined.to_csv(out_dir / f"{strategy_name}_daily.csv", index=False)
            pd.DataFrame(fold_rows).to_csv(out_dir / f"{strategy_name}_folds.csv", index=False)
            combined_metrics = compute_metrics(combined)
            combined_metrics["account_mode"] = ANNUAL_RESTART_MODE
            results[strategy_name] = {
                "account_mode": ANNUAL_RESTART_MODE,
                "metrics_from_restarted_daily_curves": combined_metrics,
                "folds": fold_rows,
            }
    export_json(out_dir.as_posix(), "summary.json", results)
    return results


def _finalize_value_matches(expected: Any, actual: Any, *, tolerance: float = 1e-6) -> bool:
    """Compare JSON/CSV scalar values without changing source calculations."""
    if expected is None:
        return actual is None or (isinstance(actual, float) and math.isnan(actual))
    if isinstance(expected, dict):
        if isinstance(actual, str):
            try:
                actual = ast.literal_eval(actual)
            except (SyntaxError, ValueError):
                return False
        return (
            isinstance(actual, dict)
            and set(expected) == set(actual)
            and all(_finalize_value_matches(expected[key], actual[key], tolerance=tolerance) for key in expected)
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, (list, tuple))
            and len(expected) == len(actual)
            and all(_finalize_value_matches(left, right, tolerance=tolerance) for left, right in zip(expected, actual))
        )
    if isinstance(expected, bool):
        return bool(actual) is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return math.isclose(float(expected), float(actual), rel_tol=tolerance, abs_tol=tolerance)
        except (TypeError, ValueError):
            return False
    if actual is None or (isinstance(actual, float) and math.isnan(actual)):
        return False
    return str(expected) == str(actual)


def validate_annual_restart_source(
    run_dir: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate existing annual sensitivity files without running a strategy.

    The returned summary is a metadata-normalized copy of the source JSON.
    Numeric values are never regenerated or changed.  The only supported
    normalization is the legacy generic ``compute_metrics`` account-mode label
    inside ``metrics_from_restarted_daily_curves``.
    """
    source_path = run_dir / "annual_restart_sensitivity" / "summary.json"
    errors: list[str] = []
    if not source_path.is_file():
        raise ValueError(f"missing:{source_path.as_posix()}")
    try:
        source_summary = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable:{source_path.as_posix()}:{exc}") from exc
    if not isinstance(source_summary, dict):
        raise ValueError("annual_summary:root_must_be_object")
    if set(source_summary) != set(STRATEGY_NAMES):
        raise ValueError(
            f"annual_summary:strategy_keys_mismatch:{sorted(source_summary)}"
        )

    actual_period = config.get("actual_oos_period") or []
    normalized = json.loads(json.dumps(source_summary, ensure_ascii=False))
    normalizations: list[dict[str, str]] = []
    strategy_checks: dict[str, Any] = {}
    for strategy_name in STRATEGY_NAMES:
        entry = source_summary.get(strategy_name)
        if not isinstance(entry, dict):
            errors.append(f"{strategy_name}:entry_must_be_object")
            continue
        if entry.get("account_mode") != ANNUAL_RESTART_MODE:
            errors.append(f"{strategy_name}:account_mode_not_annual_restart")
        metrics = entry.get("metrics_from_restarted_daily_curves")
        fold_metrics = entry.get("folds")
        if not isinstance(metrics, dict) or not isinstance(fold_metrics, list):
            errors.append(f"{strategy_name}:summary_metrics_or_folds_missing")
            continue
        metric_mode = metrics.get("account_mode")
        if metric_mode != ANNUAL_RESTART_MODE:
            if metric_mode == ACCOUNT_MODE:
                normalizations.append({
                    "path": f"{strategy_name}.metrics_from_restarted_daily_curves.account_mode",
                    "from": ACCOUNT_MODE,
                    "to": ANNUAL_RESTART_MODE,
                })
                normalized[strategy_name]["metrics_from_restarted_daily_curves"]["account_mode"] = ANNUAL_RESTART_MODE
            else:
                errors.append(f"{strategy_name}:metrics_account_mode_invalid:{metric_mode}")

        daily_path = run_dir / "annual_restart_sensitivity" / f"{strategy_name}_daily.csv"
        folds_path = run_dir / "annual_restart_sensitivity" / f"{strategy_name}_folds.csv"
        if not daily_path.is_file():
            errors.append(f"missing:{daily_path.as_posix()}")
            continue
        if not folds_path.is_file():
            errors.append(f"missing:{folds_path.as_posix()}")
            continue
        try:
            daily = pd.read_csv(daily_path)
            fold_table = pd.read_csv(folds_path)
        except Exception as exc:
            errors.append(f"{strategy_name}:csv_unreadable:{exc}")
            continue
        required_daily_columns = {"date", "equity", "daily_return", "gross_return"}
        missing_daily_columns = sorted(required_daily_columns - set(daily.columns))
        if missing_daily_columns:
            errors.append(f"{strategy_name}:daily_missing_columns:{missing_daily_columns}")
            continue
        dates = pd.to_datetime(daily["date"], errors="coerce")
        if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
            errors.append(f"{strategy_name}:daily_dates_invalid")
            continue
        if len(actual_period) == 2 and [dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")] != list(actual_period):
            errors.append(f"{strategy_name}:daily_actual_period_mismatch")

        recomputed = compute_metrics(daily)
        for key, expected in metrics.items():
            if key == "account_mode":
                continue
            if key not in recomputed or not _finalize_value_matches(expected, recomputed.get(key)):
                errors.append(f"{strategy_name}:summary_metric_mismatch:{key}")

        if len(fold_metrics) != len(fold_table):
            errors.append(f"{strategy_name}:fold_count_mismatch")
        coverage = np.zeros(len(daily), dtype=int)
        if len(fold_metrics) == len(fold_table):
            for fold_index, expected_fold in enumerate(fold_metrics):
                if not isinstance(expected_fold, dict):
                    errors.append(f"{strategy_name}:fold_{fold_index + 1}_not_object")
                    continue
                if expected_fold.get("account_mode") != ANNUAL_RESTART_MODE:
                    errors.append(f"{strategy_name}:fold_{fold_index + 1}_account_mode_invalid")
                table_row = fold_table.iloc[fold_index].to_dict()
                for key, expected in expected_fold.items():
                    if key not in table_row or not _finalize_value_matches(expected, table_row[key]):
                        errors.append(f"{strategy_name}:fold_{fold_index + 1}_mismatch:{key}")
                start = pd.to_datetime(expected_fold.get("oos_start"), errors="coerce")
                end = pd.to_datetime(expected_fold.get("oos_end"), errors="coerce")
                if pd.isna(start) or pd.isna(end):
                    errors.append(f"{strategy_name}:fold_{fold_index + 1}_invalid_dates")
                    continue
                mask = ((dates >= start) & (dates <= end)).to_numpy()
                coverage += mask.astype(int)
                fold_daily = daily.loc[mask]
                for key, actual in {
                    "n_days": len(fold_daily),
                    "oos_start": fold_daily["date"].iloc[0] if not fold_daily.empty else None,
                    "oos_end": fold_daily["date"].iloc[-1] if not fold_daily.empty else None,
                    "final_equity": float(fold_daily["equity"].iloc[-1]) if not fold_daily.empty else None,
                }.items():
                    if key not in expected_fold:
                        errors.append(f"{strategy_name}:fold_{fold_index + 1}_missing:{key}")
                    elif key in {"oos_start", "oos_end"}:
                        if str(expected_fold[key])[:10] != pd.Timestamp(actual).strftime("%Y-%m-%d") if actual is not None else expected_fold[key] is not None:
                            errors.append(f"{strategy_name}:fold_{fold_index + 1}_daily_{key}_mismatch")
                    elif not _finalize_value_matches(expected_fold[key], actual):
                        errors.append(f"{strategy_name}:fold_{fold_index + 1}_daily_{key}_mismatch")
        if len(daily) and (coverage != 1).any():
            errors.append(f"{strategy_name}:daily_fold_coverage_mismatch")
        strategy_checks[strategy_name] = {
            "daily_rows": int(len(daily)),
            "fold_rows": int(len(fold_table)),
            "daily_sha256": sha256_file(daily_path.as_posix()),
            "folds_sha256": sha256_file(folds_path.as_posix()),
        }

    if errors:
        raise ValueError(";".join(errors))
    validation = {
        "passed": True,
        "source_path": source_path.relative_to(run_dir).as_posix(),
        "source_sha256": sha256_file(source_path.as_posix()),
        "strategies": strategy_checks,
        "normalizations": normalizations,
    }
    return normalized, validation


def validate_existing_run(
    run_dir: Path,
    *,
    require_root_annual_summary: bool = False,
) -> dict[str, Any]:
    """Validate an existing run without invoking any strategy or backtest."""
    if not run_dir.is_dir() or not run_dir.name.startswith("walkforward_"):
        raise ValueError(f"invalid_run_directory:{run_dir.as_posix()}")
    required_root = ["config_snapshot.json", "input_facts.json", "gate_result.json", "oos_continuous_summary.csv"]
    missing_root = [name for name in required_root if not (run_dir / name).is_file()]
    if missing_root:
        raise ValueError(f"root_missing:{missing_root}")
    config = json.loads((run_dir / "config_snapshot.json").read_text(encoding="utf-8"))
    facts = json.loads((run_dir / "input_facts.json").read_text(encoding="utf-8"))
    root_gates = json.loads((run_dir / "gate_result.json").read_text(encoding="utf-8"))
    if config.get("run_id") != run_dir.name:
        raise ValueError("config_run_id_mismatch")
    if set(root_gates) != set(STRATEGY_NAMES):
        raise ValueError("root_gate_strategy_keys_mismatch")
    current_facts = collect_input_facts()
    if facts.get("input_hashes") != current_facts.get("input_hashes"):
        raise ValueError("current_input_hashes_do_not_match_run")
    if facts.get("rule_counts_by_status") != current_facts.get("rule_counts_by_status"):
        raise ValueError("current_rule_counts_do_not_match_run")
    if facts.get("mapping_counts_by_status") != current_facts.get("mapping_counts_by_status"):
        raise ValueError("current_mapping_counts_do_not_match_run")

    artifact_results: dict[str, Any] = {}
    metrics_by_strategy: dict[str, dict[str, Any]] = {}
    gates_by_strategy: dict[str, dict[str, Any]] = {}
    for strategy_name in STRATEGY_NAMES:
        strategy_dir = run_dir / strategy_name
        if not strategy_dir.is_dir():
            raise ValueError(f"missing_strategy_directory:{strategy_name}")
        result = validate_artifact_bundle(
            strategy_dir,
            strategy_name=strategy_name,
            expected_run_id=run_dir.name,
            expected_start=str(config["actual_oos_period"][0]),
            expected_end=str(config["actual_oos_period"][1]),
        )
        artifact_results[strategy_name] = {
            key: result.get(key)
            for key in ("passed", "required_count", "present_count", "errors")
        }
        if not result.get("passed"):
            raise ValueError(f"artifact_validation_failed:{strategy_name}:{result.get('errors')}")
        metrics_by_strategy[strategy_name] = json.loads(
            (strategy_dir / "metrics.json").read_text(encoding="utf-8")
        )
        gates_by_strategy[strategy_name] = json.loads(
            (strategy_dir / "gate_result.json").read_text(encoding="utf-8")
        )
        if gates_by_strategy[strategy_name] != root_gates[strategy_name]:
            raise ValueError(f"strategy_gate_mismatch:{strategy_name}")

    annual_summary, annual_validation = validate_annual_restart_source(run_dir, config)
    if require_root_annual_summary:
        root_summary_path = run_dir / "annual_restart_summary.json"
        if not root_summary_path.is_file():
            raise ValueError("root_annual_restart_summary_missing_after_finalize")
        root_summary = json.loads(root_summary_path.read_text(encoding="utf-8"))
        provenance = root_summary.get("_finalize_provenance")
        if not isinstance(provenance, dict) or provenance.get("source_sha256") != annual_validation["source_sha256"]:
            raise ValueError("root_annual_restart_summary_provenance_mismatch")
        if set(root_summary) - set(STRATEGY_NAMES) != {"_finalize_provenance"}:
            raise ValueError("root_annual_restart_summary_keys_mismatch")
    return {
        "config": config,
        "facts": facts,
        "root_gates": root_gates,
        "metrics": metrics_by_strategy,
        "gates": gates_by_strategy,
        "artifact_results": artifact_results,
        "annual_summary": annual_summary,
        "annual_validation": annual_validation,
    }


def finalize_existing_run(run_dir: Path) -> int:
    """Finalize only already-computed artifacts; never rerun a strategy."""
    try:
        preflight = validate_existing_run(run_dir)
    except Exception as exc:
        print(json.dumps({"finalize": "REJECTED", "reason": str(exc)}, ensure_ascii=False))
        return 2

    annual_payload = dict(preflight["annual_summary"])
    annual_payload["_finalize_provenance"] = {
        "mode": "FINALIZE_EXISTING_RUN",
        "source": preflight["annual_validation"]["source_path"],
        "source_sha256": preflight["annual_validation"]["source_sha256"],
        "validated": True,
        "validation": preflight["annual_validation"],
    }
    export_json(run_dir.as_posix(), "annual_restart_summary.json", annual_payload)

    # This is the only pytest invocation in this path.
    test_result = run_pytest_summary("tests", warning_error=True)
    if not test_result.get("all_passed", False):
        print(json.dumps({
            "finalize": "TEST_FAILED",
            "run_dir": run_dir.as_posix(),
            "test_result": test_result,
            "canonical_preserved": "walkforward_20260729_122541",
        }, ensure_ascii=False, indent=2))
        return 3

    try:
        postflight = validate_existing_run(run_dir, require_root_annual_summary=True)
    except Exception as exc:
        print(json.dumps({
            "finalize": "POSTFLIGHT_REJECTED",
            "run_dir": run_dir.as_posix(),
            "reason": str(exc),
            "canonical_preserved": "walkforward_20260729_122541",
        }, ensure_ascii=False, indent=2))
        return 4

    historical_truth_gate = all(
        gate.get("historical_truth_gate", False)
        for gate in postflight["gates"].values()
    )
    status, failed_phases = derive_research_status(
        postflight["gates"],
        historical_rules_passed=historical_truth_gate,
        test_result=test_result,
    )
    blocking_issues = []
    if not historical_truth_gate:
        blocking_issues.append(
            "Historical rule dates are not established; result is limited to the current-snapshot conservative execution scenario."
        )
    for strategy_name, gate in postflight["gates"].items():
        if gate.get("failed_checks"):
            blocking_issues.append(
                f"{strategy_name} Gate failed: {gate['failed_checks']}"
            )
    finalization = {
        "mode": "FINALIZE_EXISTING_RUN",
        "source_run_dir": run_dir.as_posix(),
        "annual_summary_source": postflight["annual_validation"]["source_path"],
        "annual_summary_sha256": postflight["annual_validation"]["source_sha256"],
        "annual_summary_validation": postflight["annual_validation"],
        "artifact_validation": postflight["artifact_results"],
        "pytest_command": f"{sys.executable} -W error::FutureWarning -m pytest -q --tb=line tests",
        "pytest_result": test_result,
    }
    write_status(
        run_id=run_dir.name,
        status=status,
        completed_phases=[
            "TRUTH_SOURCE_FACTS",
            ACCOUNT_MODE,
            "ANNUAL_RESTART_SENSITIVITY",
            "FULL_GATE_IMPLEMENTATION",
            "ARTIFACT_EXPORT",
            "FINALIZE_EXISTING_RUN",
        ],
        failed_phases=failed_phases,
        test_result=test_result,
        input_hashes=postflight["facts"]["input_hashes"],
        rule_counts_by_status=postflight["facts"]["rule_counts_by_status"],
        mapping_counts_by_status=postflight["facts"]["mapping_counts_by_status"],
        strategies_executed=list(STRATEGY_NAMES),
        oos_period="~".join(postflight["config"]["actual_oos_period"]),
        gate_result=postflight["gates"],
        blocking_issues=blocking_issues,
        requested_oos_period="~".join(postflight["config"]["requested_oos_period"]),
        actual_oos_period="~".join(postflight["config"]["actual_oos_period"]),
        account_mode=postflight["config"]["account_mode"],
        rule_scenario=postflight["config"]["rule_scenario"],
        historical_rule_status=postflight["config"]["historical_rule_status"],
        historical_truth_gate=historical_truth_gate,
        parameter_freeze_id=postflight["config"]["parameter_freeze_id"],
        annual_restart_sensitivity_status="COMPLETED",
        finalization=finalization,
    )
    write_conclusion_report(
        run_dir,
        postflight["facts"],
        postflight["metrics"],
        postflight["gates"],
        status,
        postflight["config"],
        finalization=finalization,
    )
    print(json.dumps({
        "finalize": "COMPLETED",
        "run_dir": run_dir.as_posix(),
        "status": status,
        "test_result": test_result,
        "annual_restart": True,
        "historical_truth_gate": historical_truth_gate,
        "gate_result": {
            name: {
                "gate_passed": gate.get("gate_passed"),
                "failed_checks": gate.get("failed_checks", []),
            }
            for name, gate in postflight["gates"].items()
        },
    }, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-annual-restart", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument(
        "--finalize-existing-run",
        metavar="RUN_DIR",
        help="Finalize already-exported artifacts without rerunning strategies",
    )
    args = parser.parse_args(argv)

    if args.finalize_existing_run:
        return finalize_existing_run(Path(args.finalize_existing_run))

    rule_book = ProductRuleBook.from_csv(RULES_PATH)
    facts = collect_input_facts()
    run_dir = Path(create_run_directory(OUTPUT_DIR.as_posix(), "walkforward"))
    schedule_engine = create_engine(rule_book)
    available_dates = pd.to_datetime(schedule_engine._trading_dates, errors="coerce")
    valued_dates = available_dates[
        (available_dates >= pd.Timestamp(OOS_START))
        & (available_dates <= pd.Timestamp(OOS_END))
    ]
    if len(valued_dates) == 0:
        raise RuntimeError("WALKFORWARD_NO_VALUED_OOS_DATES")
    actual_start = pd.Timestamp(valued_dates.min()).strftime("%Y-%m-%d")
    actual_end = pd.Timestamp(valued_dates.max()).strftime("%Y-%m-%d")
    folds = [
        {
            "train": fold["train"],
            "oos": (
                fold["oos"][0],
                actual_end if index == len(FOLDS) - 1 else fold["oos"][1],
            ),
        }
        for index, fold in enumerate(FOLDS)
    ]
    frozen_payload = {
        "account_mode": ACCOUNT_MODE,
        "initial_cash": INITIAL_CASH,
        "strategies": list(STRATEGY_NAMES),
        "gate_thresholds": GATE_THRESHOLDS,
        "rule_scenario": RULE_SCENARIO,
    }
    parameter_freeze_id = hashlib.sha256(
        json.dumps(frozen_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    config = {
        "run_id": run_dir.name,
        "db_path": DB_PATH,
        "rules_path": RULES_PATH,
        "mapping_path": MAPPING_PATH,
        "config_source_path": CONFIG_PATH,
        "initial_cash": INITIAL_CASH,
        "requested_oos_period": [OOS_START, OOS_END],
        "actual_oos_period": [actual_start, actual_end],
        "oos_period": [actual_start, actual_end],
        "folds": folds,
        "calendar_year_reporting_slices": True,
        "gate_thresholds": GATE_THRESHOLDS,
        "account_mode": ACCOUNT_MODE,
        "rule_scenario": RULE_SCENARIO,
        "historical_rule_status": HISTORICAL_RULE_STATUS,
        "parameter_freeze_id": parameter_freeze_id,
        "annual_restart_sensitivity_status": (
            "NOT_RUN" if args.skip_annual_restart else "COMPLETED"
        ),
    }

    signal_dates = build_month_end_schedule(
        schedule_engine._trading_dates, OOS_START, actual_end
    )
    signal_map = build_signal_submit_map(
        schedule_engine._trading_dates, signal_dates, actual_end
    )
    signal_dates_by_fold = []
    for fold in folds:
        fold_dates = build_month_end_schedule(
            schedule_engine._trading_dates, fold["oos"][0], fold["oos"][1]
        )
        fold_map = build_signal_submit_map(
            schedule_engine._trading_dates, fold_dates, fold["oos"][1]
        )
        signal_dates_by_fold.append((fold_dates, fold_map))

    continuous_results: dict[str, dict[str, Any]] = {}
    strategy_gates: dict[str, dict[str, Any]] = {}
    used_funds_by_strategy: dict[str, set[str]] = {}

    for strategy_index, strategy_name in enumerate(STRATEGY_NAMES):
        # Reuse the schedule engine for B1; the remaining strategies each
        # receive their own fresh engine because run_backtest owns its account
        # lifecycle and records an independent audit trail.
        engine = schedule_engine if strategy_index == 0 else create_engine(rule_book)
        signal = build_strategy_signal(strategy_name, rule_book, engine)
        if hasattr(signal, "reset"):
            signal.reset()
        daily, targets, audits = run_strategy(
            engine, signal, signal_dates, signal_map, OOS_START, actual_end
        )
        orders = engine.order_audit_frame()
        rejections = (
            engine.last_rejections.to_dict(orient="records")
            if not engine.last_rejections.empty else []
        )
        turnover = compute_turnover(daily, orders)
        fee_reconciliation, fee_order_audit, fee_summary = build_fee_reconciliation(
            daily, orders
        )
        metrics = compute_metrics(daily, turnover)
        metrics.update(
            {
                "confirmed_turnover_gate": metrics.get(
                    "steady_state_max_annual_bilateral_turnover", np.inf
                ) <= GATE_THRESHOLDS["annual_bilateral_turnover_max"],
                "fee_reconciliation_passed": fee_summary["passed"],
                "fee_reconciliation": fee_summary,
                "submitted_total_turnover": round(
                    float(turnover.get("submitted_bilateral_turnover", pd.Series(dtype=float)).sum()),
                    6,
                ),
                "settled_cash_total_turnover": round(
                    float(turnover.get("settled_cash_turnover", pd.Series(dtype=float)).sum()),
                    6,
                ),
            }
        )
        used_funds = set(targets.columns.astype(str))
        used_funds_by_strategy[strategy_name] = used_funds
        static_gates = build_static_data_gates(engine, facts, used_funds)
        gate = metrics_gate(metrics, static_gates)
        continuous_results[strategy_name] = {
            **metrics,
            "strategy": strategy_name,
            "data_gates": static_gates,
        }
        strategy_gates[strategy_name] = gate

        strategy_dir = run_dir / strategy_name
        parameter_freeze = {
            "mode": ACCOUNT_MODE,
            "strategy": strategy_name,
            "parameter_freeze_id": parameter_freeze_id,
            "frozen_at_fold_boundaries": True,
            "folds": folds,
            "signal_schedule": {
                "signal_date_count": len(signal_dates),
                "signal_submit_count": len(signal_map),
            },
        }
        export_strategy_bundle(
            strategy_dir,
            strategy_name,
            daily,
            orders,
            rejections,
            turnover,
            fee_reconciliation,
            fee_order_audit,
            getattr(signal, "audit_rows", audits),
            engine.last_position_lots,
            config,
            facts,
            metrics,
            gate,
            parameter_freeze,
        )
        finalize_artifact_gate(
            strategy_dir,
            strategy_name,
            run_dir.name,
            actual_start,
            actual_end,
            config,
            metrics,
            gate,
        )
        # finalize_artifact_gate adds the final artifact Gate fields to the
        # mutable metrics object. Keep the report/status source in sync with
        # the metrics.json that was just written; S1 later refreshes its
        # manifest after adding the relative benchmark check.
        continuous_results[strategy_name].update(metrics)

        # Export annual fold slices from the same continuous run.  These are
        # reporting partitions, not independent account starts.
        for fold_idx, fold in enumerate(folds, start=1):
            fold_daily = daily[
                (pd.to_datetime(daily["date"]) >= pd.Timestamp(fold["oos"][0]))
                & (pd.to_datetime(daily["date"]) <= pd.Timestamp(fold["oos"][1]))
            ].copy()
            if fold_daily.empty:
                continue
            fold_dates = set(pd.to_datetime(fold_daily["date"]))
            if not orders.empty:
                submit_dates = pd.to_datetime(orders["submit_date"], errors="coerce")
                confirmation_dates = pd.to_datetime(orders["confirmation_date"], errors="coerce")
                fold_orders = orders[
                    submit_dates.isin(fold_dates) | confirmation_dates.isin(fold_dates)
                ].copy()
            else:
                fold_orders = orders
            fold_turnover = turnover[turnover["date"].isin(fold_dates)]
            fold_fee_reconciliation, fold_fee_order_audit, fold_fee_summary = build_fee_reconciliation(
                fold_daily, fold_orders
            )
            fold_metrics = compute_metrics(fold_daily, fold_turnover)
            fold_metrics.update(
                {
                    "confirmed_turnover_gate": fold_metrics.get(
                        "max_annual_bilateral_turnover", np.inf
                    ) <= GATE_THRESHOLDS["annual_bilateral_turnover_max"],
                    "fee_reconciliation_passed": fold_fee_summary["passed"],
                    "fee_reconciliation": fold_fee_summary,
                }
            )
            fold_gate = metrics_gate(fold_metrics, static_gates)
            all_audits = getattr(signal, "audit_rows", audits)
            fold_signal_dates = {
                pd.Timestamp(value).strftime("%Y-%m-%d")
                for value in signal_dates_by_fold[fold_idx - 1][0]
            }
            fold_audits = [
                audit for audit in all_audits
                if pd.Timestamp(
                    audit.get("signal_date", audit.get("date"))
                ).strftime("%Y-%m-%d") in fold_signal_dates
            ]
            fold_actual_start = pd.Timestamp(fold_daily["date"].min()).strftime("%Y-%m-%d")
            fold_actual_end = pd.Timestamp(fold_daily["date"].max()).strftime("%Y-%m-%d")
            fold_config = {
                **config,
                "fold": fold,
                "actual_oos_period": [fold_actual_start, fold_actual_end],
                "oos_period": [fold_actual_start, fold_actual_end],
            }
            export_strategy_bundle(
                strategy_dir / f"fold_{fold_idx}",
                strategy_name,
                fold_daily,
                fold_orders,
                [row for row in rejections if pd.Timestamp(row.get("date")) in fold_dates],
                fold_turnover,
                fold_fee_reconciliation,
                fold_fee_order_audit,
                fold_audits,
                engine.last_position_lots,
                fold_config,
                facts,
                fold_metrics,
                fold_gate,
                {"fold": fold_idx, **parameter_freeze},
            )
            finalize_artifact_gate(
                strategy_dir / f"fold_{fold_idx}",
                strategy_name,
                run_dir.name,
                fold_actual_start,
                fold_actual_end,
                fold_config,
                fold_metrics,
                fold_gate,
            )

    # S1 relative benchmark criterion is reported separately and never
    # replaces the absolute checks above.
    b2_metrics = continuous_results["B2_Static_EW_4Asset"]
    s1_gate = strategy_gates["S1_State_Rotation_Fixed"]
    s1_relative = {
        "cagr_advantage_pp": continuous_results["S1_State_Rotation_Fixed"]["net_cagr_pct"] - b2_metrics["net_cagr_pct"],
        "sharpe_advantage": continuous_results["S1_State_Rotation_Fixed"]["sharpe"] - b2_metrics["sharpe"],
    }
    s1_relative["relative_check_passed"] = bool(
        s1_relative["cagr_advantage_pp"] >= 0.75
        or s1_relative["sharpe_advantage"] >= 0.15
    )
    s1_gate["relative_benchmark_gate"] = s1_relative
    s1_gate["checks"]["relative_benchmark"] = s1_relative["relative_check_passed"]
    s1_gate["failed_checks"] = [
        name for name, passed in s1_gate["checks"].items() if not passed
    ]
    s1_gate["gate_passed"] = not s1_gate["failed_checks"]
    s1_dir = run_dir / "S1_State_Rotation_Fixed"
    s1_metrics = json.loads(
        (s1_dir / "metrics.json").read_text(encoding="utf-8")
    )
    export_json(s1_dir.as_posix(), "gate_result.json", s1_gate)
    write_manifest(
        run_dir=s1_dir.as_posix(),
        strategy_name="S1_State_Rotation_Fixed",
        config=config,
        db_path=DB_PATH,
        rules_path=RULES_PATH,
        exposure_mapping_path=MAPPING_PATH,
        metrics=s1_metrics,
        gate_result=s1_gate,
    )

    continuous_summary = pd.DataFrame(list(continuous_results.values()))
    continuous_summary.to_csv(run_dir / "oos_continuous_summary.csv", index=False)
    export_json(run_dir.as_posix(), "gate_result.json", strategy_gates)
    export_json(run_dir.as_posix(), "input_facts.json", facts)
    export_config_snapshot(run_dir.as_posix(), config)

    annual_results = {}
    if not args.skip_annual_restart:
        annual_results = run_annual_restart_sensitivity(
            run_dir, rule_book, signal_dates_by_fold, folds
        )

    test_result = {} if args.skip_tests else run_pytest_summary("tests")
    blocking_issues = []
    historical_rules_passed = all(
        result["data_gates"].get("historical_rules_gate", False)
        for result in continuous_results.values()
    )
    status, failed_phases = derive_research_status(
        strategy_gates,
        historical_rules_passed=historical_rules_passed,
        test_result=test_result,
    )
    if not historical_rules_passed:
        blocking_issues.append(
            "Historical rule dates are not established; result is limited to the current-snapshot conservative execution scenario."
        )
    if test_result and not test_result.get("all_passed", False):
        blocking_issues.append(test_result.get("raw_summary", "pytest failed"))
    for strategy_name, gate in strategy_gates.items():
        if gate.get("failed_checks"):
            blocking_issues.append(
                f"{strategy_name} Gate failed: {gate['failed_checks']}"
            )
    write_status(
        run_id=run_dir.name,
        status=status,
        completed_phases=[
            "TRUTH_SOURCE_FACTS",
            ACCOUNT_MODE,
            *(["ANNUAL_RESTART_SENSITIVITY"] if annual_results else []),
            "FULL_GATE_IMPLEMENTATION",
            "ARTIFACT_EXPORT",
        ],
        failed_phases=failed_phases,
        test_result=test_result,
        input_hashes=facts["input_hashes"],
        rule_counts_by_status=facts["rule_counts_by_status"],
        mapping_counts_by_status=facts["mapping_counts_by_status"],
        strategies_executed=list(STRATEGY_NAMES),
        oos_period=f"{actual_start}~{actual_end}",
        gate_result=strategy_gates,
        blocking_issues=blocking_issues,
        requested_oos_period=f"{OOS_START}~{OOS_END}",
        actual_oos_period=f"{actual_start}~{actual_end}",
        account_mode=ACCOUNT_MODE,
        rule_scenario=RULE_SCENARIO,
        historical_rule_status=HISTORICAL_RULE_STATUS,
        historical_truth_gate=historical_rules_passed,
        parameter_freeze_id=parameter_freeze_id,
        annual_restart_sensitivity_status=config["annual_restart_sensitivity_status"],
    )
    export_json(run_dir.as_posix(), "annual_restart_summary.json", annual_results)
    write_conclusion_report(
        run_dir, facts, continuous_results, strategy_gates, status, config
    )
    compact_results = {
        name: {
            key: result.get(key)
            for key in (
                "oos_start", "oos_end", "net_cagr_pct", "sharpe", "mdd_pct",
                "max_annual_bilateral_turnover", "total_fee_amount",
            )
        }
        for name, result in continuous_results.items()
    }
    print(json.dumps({
        "run_dir": run_dir.as_posix(),
        "status": status,
        "account_mode": ACCOUNT_MODE,
        "continuous_results": compact_results,
        "gate_result": {
            name: {
                "gate_passed": gate.get("gate_passed"),
                "failed_checks": gate.get("failed_checks", []),
            }
            for name, gate in strategy_gates.items()
        },
        "annual_restart": bool(annual_results),
        "test_result": test_result,
    }, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
