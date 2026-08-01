"""Explanatory attribution for frozen OTF strategy account bundles.

This module deliberately does not run a strategy.  It consumes the account,
weight, order, fee and turnover artifacts already produced by a frozen run and
reconciles their realised returns against the fund NAV growth series.

The important timing convention is explicit: the contribution on valuation
date ``t`` uses the actual portfolio weights from the previous valuation date.
Current-date weights are never used to explain current-date returns.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


EPS = 1e-10
ECONOMIC_RESIDUAL_WARN = 0.0025
TOTAL_CODE = "__TOTAL__"
CASH_CODE = "__CASH__"
ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_code(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _date_frame(frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    result = frame.copy()
    result[column] = pd.to_datetime(result[column], errors="coerce")
    result = result.dropna(subset=[column]).sort_values(column).reset_index(drop=True)
    return result


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return default if not math.isfinite(value) else value


@dataclass(frozen=True)
class AttributionSpec:
    key: str
    strategy: str
    bundle: Path
    core_codes: frozenset[str]
    satellite_codes: frozenset[str]
    db_path: Path


def load_spec(key: str, bundle: str | Path, db_path: str | Path) -> AttributionSpec:
    bundle = Path(bundle)
    config = _json_load(bundle / "config_snapshot.json")
    core = {canonical_code(k) for k in config.get("core_weights", {})}
    satellites = {canonical_code(k) for k in config.get("satellite_pool", [])}
    strategy = str(config.get("strategy", ""))
    if not strategy:
        strategy = _json_load(bundle / "metrics.json").get("strategy", key)
    return AttributionSpec(
        key=key,
        strategy=strategy,
        bundle=bundle,
        core_codes=frozenset(core),
        satellite_codes=frozenset(satellites),
        db_path=Path(db_path),
    )


def validate_frozen_inputs(specs: Iterable[AttributionSpec], root: Path = ROOT) -> dict[str, Any]:
    """Validate the frozen database/rules/mapping identity gate.

    Config hashes are intentionally allowed to differ between C1/C2.  The
    data, rules and exposure mapping hashes must be identical, and must also
    match the bytes currently present in the workspace.
    """

    specs = list(specs)
    if not specs:
        raise ValueError("at least one attribution input is required")
    input_hashes: dict[str, dict[str, str]] = {}
    for spec in specs:
        hashes = _json_load(spec.bundle / "input_hashes.json")
        hashes = {
            "db_sha256": hashes.get("db_sha256", ""),
            "rules_sha256": hashes.get("rules_sha256", ""),
            "mapping_sha256": hashes.get("mapping_sha256", hashes.get("exposure_mapping_sha256", "")),
            "config_sha256": hashes.get("config_sha256", ""),
        }
        input_hashes[spec.key] = hashes
    required = ("db_sha256", "rules_sha256", "mapping_sha256")
    reference = input_hashes[specs[0].key]
    mismatch: list[str] = []
    for spec in specs[1:]:
        for name in required:
            if input_hashes[spec.key].get(name) != reference.get(name):
                mismatch.append(f"{spec.key}:{name}:does_not_match_first")
    actual_paths = {
        "db_sha256": specs[0].db_path if specs[0].db_path.is_absolute() else root / specs[0].db_path,
        "rules_sha256": root / "config/otf_product_rules.csv",
        "mapping_sha256": root / "config/otf_exposure_mapping.csv",
    }
    actual_hashes: dict[str, str] = {}
    for name, path in actual_paths.items():
        if not path.exists():
            mismatch.append(f"missing_input_file:{path}")
            actual_hashes[name] = ""
        else:
            actual_hashes[name] = sha256_file(path)
            if actual_hashes[name] != reference.get(name):
                mismatch.append(f"workspace_hash_mismatch:{name}")
    return {
        "passed": not mismatch,
        "required_hashes": list(required),
        "bundle_input_hashes": input_hashes,
        "workspace_hashes": actual_hashes,
        "mismatches": mismatch,
    }


def load_fund_metadata(db_path: str | Path, codes: Iterable[str]) -> pd.DataFrame:
    codes = sorted({canonical_code(code) for code in codes if canonical_code(code)})
    if not codes:
        return pd.DataFrame(columns=["fund_code", "fund_name", "asset_class", "fund_type", "underlying_name"])
    placeholders = ",".join("?" for _ in codes)
    with sqlite3.connect(str(db_path)) as connection:
        query = f"""
            SELECT fund_code, fund_name, asset_class, fund_type, underlying_name,
                   share_class, fund_family, etf_symbol, etf_name
            FROM otf_fund_catalog WHERE fund_code IN ({placeholders})
        """
        metadata = pd.read_sql_query(query, connection, params=codes)
    if metadata.empty:
        return pd.DataFrame({"fund_code": codes, "fund_name": codes, "asset_class": "unknown"})
    metadata["fund_code"] = metadata["fund_code"].map(canonical_code)
    for column in ("fund_name", "asset_class", "fund_type", "underlying_name", "share_class", "fund_family", "etf_symbol", "etf_name"):
        if column not in metadata:
            metadata[column] = ""
    metadata["asset_class"] = metadata["asset_class"].fillna("unknown").replace("", "unknown")
    return metadata.drop_duplicates("fund_code")


def load_fund_returns(db_path: str | Path, codes: Iterable[str]) -> pd.DataFrame:
    codes = sorted({canonical_code(code) for code in codes if canonical_code(code)})
    if not codes:
        return pd.DataFrame(columns=["date", "fund_code", "daily_growth_pct"])
    placeholders = ",".join("?" for _ in codes)
    with sqlite3.connect(str(db_path)) as connection:
        query = f"""
            SELECT fund_code, nav_date AS date, daily_growth_pct,
                   cumulative_nav_imputed, total_return_factor
            FROM otf_fund_nav WHERE fund_code IN ({placeholders})
            ORDER BY nav_date, fund_code
        """
        returns = pd.read_sql_query(query, connection, params=codes)
    if returns.empty:
        return pd.DataFrame(columns=["date", "fund_code", "fund_return", "nav_available"])
    returns["fund_code"] = returns["fund_code"].map(canonical_code)
    returns["date"] = pd.to_datetime(returns["date"], errors="coerce")
    returns["daily_growth_pct"] = pd.to_numeric(returns["daily_growth_pct"], errors="coerce")
    returns["fund_return"] = returns["daily_growth_pct"] / 100.0
    returns["nav_available"] = returns["fund_return"].notna()
    return returns[["date", "fund_code", "fund_return", "nav_available", "cumulative_nav_imputed", "total_return_factor"]]


def _period_label(date: pd.Timestamp) -> str:
    return str(int(date.year))


def _period_group_label(value: Any) -> str:
    year = int(str(value)[:4])
    if 2021 <= year <= 2023:
        return "2021_2023"
    if 2024 <= year <= 2026:
        return "2024_2026"
    return str(value)


def _group_contribution(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby(group_columns, dropna=False, sort=True)
    result = grouped.agg(
        gross_contribution=("fund_contribution", "sum"),
        fee_drag=("allocated_fee_drag", "sum"),
        holding_days=("held", "sum"),
        average_actual_weight=("actual_weight", "mean"),
        average_lagged_weight=("lagged_weight", "mean"),
        positive_gross_contribution=("positive_contribution", "sum"),
        negative_gross_contribution=("negative_contribution", "sum"),
        missing_nav_weight_days=("missing_nav_weight", "sum"),
    ).reset_index()
    result["net_contribution_approx"] = result["gross_contribution"] - result["fee_drag"]
    result["positive_gross_pnl_ratio"] = np.where(
        result["positive_gross_contribution"].sum() > 0,
        result["positive_gross_contribution"] / result["positive_gross_contribution"].sum(),
        np.nan,
    )
    return result


def compute_daily_attribution(
    daily_account: pd.DataFrame,
    actual_weights: pd.DataFrame,
    fund_returns: pd.DataFrame,
    fund_metadata: pd.DataFrame,
    fees: pd.DataFrame | None = None,
    strategy: str = "",
    core_codes: Iterable[str] = (),
    satellite_codes: Iterable[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return long daily fund rows and one total row per valuation date."""

    account = _date_frame(daily_account)
    weights = _date_frame(actual_weights)
    account["gross_return"] = pd.to_numeric(account["gross_return"], errors="coerce").fillna(0.0)
    account["net_return"] = pd.to_numeric(account.get("daily_return", 0.0), errors="coerce").fillna(0.0)
    account["transaction_cost"] = pd.to_numeric(account.get("transaction_cost", 0.0), errors="coerce").fillna(0.0)
    account["total_fee_amount"] = pd.to_numeric(account.get("total_fee_amount", 0.0), errors="coerce").fillna(0.0)
    account["prior_equity"] = account["equity"].shift(1).fillna(account["equity"].iloc[0])
    account["prior_equity"] = pd.to_numeric(account["prior_equity"], errors="coerce").replace(0.0, np.nan).ffill().fillna(1.0)
    account["period"] = account["date"].map(_period_label)
    if fees is not None and not fees.empty:
        fee_frame = _date_frame(fees).rename(columns={"daily_total_fee_amount": "fee_file_amount"})
        if "fee_file_amount" in fee_frame:
            account = account.merge(fee_frame[["date", "fee_file_amount"]], on="date", how="left")
    if "fee_file_amount" not in account:
        account["fee_file_amount"] = account["total_fee_amount"]
    account["fee_file_amount"] = pd.to_numeric(account["fee_file_amount"], errors="coerce").fillna(0.0)

    weight_cols = [c for c in weights.columns if c != "date"]
    weight_rename = {column: canonical_code(column) for column in weight_cols}
    weights = weights.rename(columns=weight_rename)
    weight_cols = sorted({canonical_code(c) for c in weight_cols if canonical_code(c)})
    for code in weight_cols:
        weights[code] = pd.to_numeric(weights[code], errors="coerce").fillna(0.0)
    weights = weights.set_index("date").reindex(account["date"]).ffill().fillna(0.0)
    lagged = weights.shift(1).fillna(0.0)
    current = weights

    ret = fund_returns.copy()
    if not ret.empty:
        ret["date"] = pd.to_datetime(ret["date"], errors="coerce")
        ret["fund_code"] = ret["fund_code"].map(canonical_code)
        ret = ret.drop_duplicates(["date", "fund_code"], keep="last")
        ret_wide = ret.pivot(index="date", columns="fund_code", values="fund_return")
        avail_wide = ret.pivot(index="date", columns="fund_code", values="nav_available")
    else:
        ret_wide = pd.DataFrame(index=account["date"])
        avail_wide = pd.DataFrame(index=account["date"])
    ret_wide = ret_wide.reindex(account["date"])
    avail_wide = avail_wide.reindex(account["date"])
    metadata = fund_metadata.copy()
    if not metadata.empty:
        metadata["fund_code"] = metadata["fund_code"].map(canonical_code)
    metadata_by_code = metadata.set_index("fund_code").to_dict("index") if not metadata.empty else {}
    core = {canonical_code(c) for c in core_codes}
    satellite = {canonical_code(c) for c in satellite_codes}

    rows: list[dict[str, Any]] = []
    totals: list[dict[str, Any]] = []
    for pos, row in account.iterrows():
        date = row["date"]
        gross = float(row["gross_return"])
        net = float(row["net_return"])
        fee_drag = float(row["transaction_cost"])
        prior_weights = lagged.iloc[pos] if pos < len(lagged) else pd.Series(dtype=float)
        current_weights = current.iloc[pos] if pos < len(current) else pd.Series(dtype=float)
        codes = sorted(set(weight_cols) | set(ret_wide.columns))
        fund_sum = 0.0
        for code in codes:
            weight = _safe_float(prior_weights.get(code, 0.0))
            actual_weight = _safe_float(current_weights.get(code, 0.0))
            value = ret_wide.iloc[pos].get(code, np.nan) if pos < len(ret_wide) else np.nan
            available = bool(avail_wide.iloc[pos].get(code, False)) if pos < len(avail_wide) else False
            contribution = weight * _safe_float(value) if available and math.isfinite(_safe_float(value, np.nan)) else 0.0
            missing_weight = weight if weight > 0 and not available else 0.0
            fund_sum += contribution
            meta = metadata_by_code.get(code, {})
            role = "core" if code in core else ("satellite" if code in satellite else "other")
            rows.append({
                "strategy": strategy,
                "date": date.date().isoformat(),
                "period": row["period"],
                "fund_code": code,
                "fund_name": meta.get("fund_name", code),
                "asset_class": meta.get("asset_class", "unknown") or "unknown",
                "sleeve": role,
                "row_type": "FUND",
                "lagged_weight": weight,
                "actual_weight": actual_weight,
                "fund_return": _safe_float(value, np.nan),
                "nav_available": available,
                "fund_contribution": contribution,
                "allocated_fee_drag": fee_drag * weight,
                "held": int(weight > 0),
                "positive_contribution": max(contribution, 0.0),
                "negative_contribution": min(contribution, 0.0),
                "missing_nav_weight": missing_weight,
                "gross_return": gross,
                "net_return": net,
                "cash_contribution": np.nan,
                "accounting_residual": np.nan,
                "accounting_error": np.nan,
                "net_reconciliation_error": np.nan,
            })
        cash_weight = 1.0 - float(prior_weights.sum())
        cash_return = 0.0
        cash_contribution = cash_weight * cash_return
        residual = gross - fund_sum - cash_contribution
        missing_total_weight = 0.0
        if pos < len(avail_wide):
            for code in codes:
                if not bool(avail_wide.iloc[pos].get(code, False)):
                    missing_total_weight += _safe_float(prior_weights.get(code, 0.0))
        fee_amount_drag = float(row["fee_file_amount"]) / float(row["prior_equity"])
        net_error = net - (gross - fee_drag)
        account_fee_reconciliation_error = float(row["total_fee_amount"]) - float(row["fee_file_amount"])
        totals.append({
            "strategy": strategy,
            "date": date.date().isoformat(),
            "period": row["period"],
            "fund_code": TOTAL_CODE,
            "fund_name": "TOTAL",
            "asset_class": "total",
            "sleeve": "total",
            "row_type": "TOTAL",
            "lagged_weight": float(prior_weights.sum()),
            "actual_weight": float(current_weights.sum()),
            "fund_return": np.nan,
            "nav_available": True,
            "fund_contribution": fund_sum,
            "allocated_fee_drag": fee_drag,
            "held": int(prior_weights.sum() > 0),
            "positive_contribution": np.nan,
            "negative_contribution": np.nan,
            "missing_nav_weight": missing_total_weight,
            "gross_return": gross,
            "net_return": net,
            "cash_weight": cash_weight,
            "cash_return": cash_return,
            "cash_contribution": cash_contribution,
            "accounting_residual": residual,
            "accounting_error": gross - (fund_sum + cash_contribution + residual),
            "fee_amount": float(row["fee_file_amount"]),
            "account_total_fee_amount": float(row["total_fee_amount"]),
            "account_fee_reconciliation_error": account_fee_reconciliation_error,
            "fee_amount_drag": fee_amount_drag,
            "cost_drag": fee_drag,
            "net_reconciliation_error": net_error,
            "prior_equity": float(row["prior_equity"]),
            "equity": float(row["equity"]),
        })
    fund_rows = pd.DataFrame(rows)
    total_rows = pd.DataFrame(totals)
    if fund_rows.empty:
        fund_rows = pd.DataFrame(columns=["strategy", "date", "fund_code", "fund_contribution"])
    return fund_rows, total_rows


def aggregate_attribution(
    fund_rows: pd.DataFrame,
    total_rows: pd.DataFrame,
    metadata: pd.DataFrame,
    strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if fund_rows.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    sleeve_rows = fund_rows.copy()
    cash_rows = total_rows.copy()
    cash_rows["fund_code"] = CASH_CODE
    cash_rows["fund_name"] = "Cash / unsettled"
    cash_rows["asset_class"] = "cash"
    cash_rows["sleeve"] = "cash"
    cash_rows["fund_contribution"] = cash_rows["cash_contribution"].fillna(0.0)
    cash_rows["allocated_fee_drag"] = cash_rows["cost_drag"] * cash_rows["cash_weight"]
    cash_rows["actual_weight"] = 1.0 - cash_rows["actual_weight"]
    cash_rows["lagged_weight"] = cash_rows["cash_weight"]
    cash_rows["held"] = (cash_rows["cash_weight"] > 0).astype(int)
    cash_rows["positive_contribution"] = cash_rows["fund_contribution"].clip(lower=0.0)
    cash_rows["negative_contribution"] = cash_rows["fund_contribution"].clip(upper=0.0)
    cash_rows["missing_nav_weight"] = 0.0
    cash_rows["period"] = cash_rows["period"]
    all_rows = pd.concat([fund_rows, cash_rows[fund_rows.columns]], ignore_index=True, sort=False)
    fund = _group_contribution(all_rows, ["fund_code", "fund_name", "asset_class", "sleeve"])
    fund.insert(0, "strategy", strategy)
    annual_fund = _group_contribution(all_rows, ["period", "fund_code", "fund_name", "asset_class", "sleeve"])
    annual_fund.insert(0, "strategy", strategy)
    grouped_rows = all_rows.copy()
    grouped_rows["period"] = grouped_rows["period"].map(_period_group_label)
    grouped_fund = _group_contribution(grouped_rows, ["period", "fund_code", "fund_name", "asset_class", "sleeve"])
    grouped_fund.insert(0, "strategy", strategy)
    full_fund = _group_contribution(all_rows.assign(period="full"), ["period", "fund_code", "fund_name", "asset_class", "sleeve"])
    full_fund.insert(0, "strategy", strategy)
    period_fund = pd.concat([annual_fund, grouped_fund, full_fund], ignore_index=True, sort=False)
    # The total row's allocated fee is the whole cost drag; for sleeve/fund
    # attribution it is allocated by prior actual exposure, with cash retaining
    # its proportional share.
    annual_sleeve = _group_contribution(all_rows, ["period", "sleeve"])
    annual_sleeve.insert(0, "strategy", strategy)
    grouped_sleeve = _group_contribution(grouped_rows, ["period", "sleeve"])
    grouped_sleeve.insert(0, "strategy", strategy)
    full_sleeve = _group_contribution(all_rows.assign(period="full"), ["period", "sleeve"])
    full_sleeve.insert(0, "strategy", strategy)
    sleeve = pd.concat([full_sleeve, annual_sleeve, grouped_sleeve], ignore_index=True)
    return fund, sleeve, period_fund


def _account_interval(account: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp | None = None) -> pd.DataFrame:
    result = account[(account["date"] >= start)].copy()
    if end is not None:
        result = result[result["date"] <= end]
    return result


def _compound(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return float(np.prod(1.0 + values.to_numpy(dtype=float)) - 1.0)


def build_event_attribution(
    spec: AttributionSpec,
    b2_account: pd.DataFrame,
) -> pd.DataFrame:
    market_path = spec.bundle / "market_states.csv"
    states = _read_csv(market_path) if market_path.exists() else pd.DataFrame()
    account = _date_frame(_read_csv(spec.bundle / "daily_account.csv"))
    account["gross_return"] = pd.to_numeric(account["gross_return"], errors="coerce").fillna(0.0)
    account["net_return"] = pd.to_numeric(account["daily_return"], errors="coerce").fillna(0.0)
    account["total_fee_amount"] = pd.to_numeric(account["total_fee_amount"], errors="coerce").fillna(0.0)
    account["transaction_cost"] = pd.to_numeric(account["transaction_cost"], errors="coerce").fillna(0.0)
    b2_account = _date_frame(b2_account)
    if "net_return" not in b2_account:
        b2_account["net_return"] = pd.to_numeric(b2_account.get("daily_return", 0.0), errors="coerce").fillna(0.0)
    b2_account["gross_return"] = pd.to_numeric(b2_account["gross_return"], errors="coerce").fillna(0.0)
    orders = _read_csv(spec.bundle / "orders.csv") if (spec.bundle / "orders.csv").exists() else pd.DataFrame()
    if not orders.empty:
        for column in ("signal_date", "submit_date", "confirmation_date", "redemption_arrival_date"):
            if column in orders:
                orders[column] = pd.to_datetime(orders[column], errors="coerce")
    if states.empty or "signal_date" not in states:
        # Static baselines do not emit market_states.csv, but their accepted
        # order batches are still valid rebalance/build events for attribution.
        if orders.empty or "signal_date" not in orders:
            return pd.DataFrame()
        signal_dates = sorted(orders["signal_date"].dropna().unique())
        states = pd.DataFrame({
            "signal_date": signal_dates,
            "decision": "ORDER_BATCH",
            "market_state": "STATIC_ORDER_BATCH",
            "rebalance_bool": True,
            "accepted_bool": True,
        })
    else:
        states["signal_date"] = pd.to_datetime(states["signal_date"], errors="coerce")
        states["rebalance_bool"] = states.get("rebalance", True).map(_bool) if "rebalance" in states else True
        if "accepted_target_weights" in states:
            states["accepted_bool"] = states["accepted_target_weights"].fillna("").astype(str).str.len() > 2
        else:
            states["accepted_bool"] = True
        states = states[states["signal_date"].notna() & states["rebalance_bool"] & states["accepted_bool"]].copy()
    turnover = _date_frame(_read_csv(spec.bundle / "turnover.csv")) if (spec.bundle / "turnover.csv").exists() else pd.DataFrame()
    if not turnover.empty:
        for column in ("bilateral_turnover", "submitted_bilateral_turnover", "settled_cash_turnover"):
            if column in turnover:
                turnover[column] = pd.to_numeric(turnover[column], errors="coerce").fillna(0.0)
    states = states.sort_values("signal_date").reset_index(drop=True)
    starts: list[pd.Timestamp] = []
    for _, event in states.iterrows():
        signal = event["signal_date"]
        event_orders = orders[orders["signal_date"] == signal] if not orders.empty and "signal_date" in orders else pd.DataFrame()
        dates = []
        if not event_orders.empty:
            for column in ("confirmation_date", "redemption_arrival_date"):
                if column in event_orders:
                    dates.extend(event_orders[column].dropna().tolist())
        if dates:
            ready = max(dates)
        else:
            ready = signal
        after = account.loc[account["date"] >= ready, "date"]
        starts.append(after.iloc[0] if not after.empty else pd.NaT)
    rows: list[dict[str, Any]] = []
    for idx, (_, event) in enumerate(states.iterrows()):
        start = starts[idx]
        if pd.isna(start):
            continue
        next_starts = [x for x in starts[idx + 1:] if not pd.isna(x) and x > start]
        end = (min(next_starts) - pd.Timedelta(days=1)) if next_starts else account["date"].max()
        interval = _account_interval(account, start, end)
        signal = event["signal_date"]
        event_orders = orders[orders["signal_date"] == signal] if not orders.empty and "signal_date" in orders else pd.DataFrame()
        order_submit = event_orders["submit_date"].min() if not event_orders.empty else pd.NaT
        turnover_interval = turnover[(turnover["date"] >= signal) & (turnover["date"] <= end)] if not turnover.empty else pd.DataFrame()
        turnover_value = float(turnover_interval.get("submitted_bilateral_turnover", pd.Series(dtype=float)).sum()) if not turnover_interval.empty else 0.0
        if turnover_value == 0.0 and not turnover_interval.empty:
            turnover_value = float(turnover_interval.get("bilateral_turnover", pd.Series(dtype=float)).sum())
        b2 = _account_interval(b2_account, start, end)
        selected = str(event.get("selected_satellites", ""))
        rows.append({
            "strategy": spec.strategy,
            "event_number": idx + 1,
            "signal_date": signal.date().isoformat(),
            "submit_date": order_submit.date().isoformat() if pd.notna(order_submit) else "",
            "interval_start": start.date().isoformat(),
            "interval_end": end.date().isoformat(),
            "decision": event.get("decision", ""),
            "rebalance_reason": event.get("decision", event.get("rebalance_reason", "")),
            "market_state": event.get("market_state", ""),
            "selected_satellites": selected,
            "interval_days": int(len(interval)),
            "gross_return": _compound(interval["gross_return"]),
            "net_return": _compound(interval["net_return"]),
            "fee_amount": float(interval["total_fee_amount"].sum()),
            "cost_drag_sum": float(interval["transaction_cost"].sum()),
            "turnover": turnover_value,
            "b2_gross_return": _compound(b2["gross_return"]) if not b2.empty else np.nan,
            "b2_net_return": _compound(b2["net_return"]) if not b2.empty else np.nan,
            "relative_b2_gross_return": _compound(interval["gross_return"]) - _compound(b2["gross_return"]) if not b2.empty else np.nan,
            "relative_b2_net_return": _compound(interval["net_return"]) - _compound(b2["net_return"]) if not b2.empty else np.nan,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["gross_rank_best"] = result["gross_return"].rank(method="min", ascending=False).astype(int)
        result["gross_rank_worst"] = result["gross_return"].rank(method="min", ascending=True).astype(int)
    return result


def build_relative_attribution(
    results: Mapping[str, dict[str, Any]],
    b2_key: str = "B2",
) -> pd.DataFrame:
    if b2_key not in results:
        return pd.DataFrame()
    b2 = results[b2_key]
    b2_total = b2["total_rows"]
    rows: list[dict[str, Any]] = []
    for key, result in results.items():
        if key == b2_key:
            continue
        total = result["total_rows"]
        merged = total[["date", "gross_return", "net_return", "cost_drag", "accounting_residual", "cash_contribution"]].merge(
            b2_total[["date", "gross_return", "net_return", "cost_drag", "accounting_residual", "cash_contribution"]],
            on="date", suffixes=("", "_b2"), how="inner",
        )
        strategy_fund = float(result["total_rows"]["fund_contribution"].sum())
        b2_fund = float(b2_total["fund_contribution"].sum())
        fund_delta = strategy_fund - b2_fund
        cash_delta = float(total["cash_contribution"].sum() - b2_total["cash_contribution"].sum())
        residual_delta = float(total["accounting_residual"].sum() - b2_total["accounting_residual"].sum())
        fee_delta = float(total["cost_drag"].sum() - b2_total["cost_drag"].sum())
        gross_additive = float(total["gross_return"].sum() - b2_total["gross_return"].sum())
        net_additive = float(total["net_return"].sum() - b2_total["net_return"].sum())
        rows.append({
            "strategy": result["spec"].strategy,
            "benchmark": b2["spec"].strategy,
            "comparison_key": key,
            "start_date": str(total["date"].min()),
            "end_date": str(total["date"].max()),
            "gross_total_return": _compound(total["gross_return"]),
            "b2_gross_total_return": _compound(b2_total["gross_return"]),
            "gross_total_return_difference_compounded": _compound(total["gross_return"]) - _compound(b2_total["gross_return"]),
            "net_total_return": _compound(total["net_return"]),
            "b2_net_total_return": _compound(b2_total["net_return"]),
            "net_total_return_difference_compounded": _compound(total["net_return"]) - _compound(b2_total["net_return"]),
            "strategy_net_cagr_pct": float(result["metrics"].get("net_cagr_pct", np.nan)),
            "b2_net_cagr_pct": float(b2["metrics"].get("net_cagr_pct", np.nan)),
            "annualized_cagr_difference_pp": float(result["metrics"].get("net_cagr_pct", np.nan) - b2["metrics"].get("net_cagr_pct", np.nan)),
            "fund_exposure_contribution_difference_additive": fund_delta,
            "cash_contribution_difference_additive": cash_delta,
            "fee_drag_difference_additive": fee_delta,
            "residual_difference_additive": residual_delta,
            "gross_return_difference_additive": gross_additive,
            "net_return_difference_additive": net_additive,
            "explanatory_method": "holdings-based explanatory attribution; not causal Brinson alpha",
            "same_dates": bool(len(merged) == len(total) == len(b2_total)),
        })
    return pd.DataFrame(rows)


def build_status(
    results: Mapping[str, dict[str, Any]],
    hash_gate: dict[str, Any],
    event_frames: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    if not hash_gate["passed"]:
        errors.append("INPUT_HASH_GATE_FAILED")
    accounting_max = 0.0
    net_max = 0.0
    economic: dict[str, float] = {}
    fee_recon: dict[str, float] = {}
    account_fee_recon: dict[str, float] = {}
    future_data_passed = True
    for key, result in results.items():
        total = result["total_rows"]
        accounting_max = max(accounting_max, float(total["accounting_error"].abs().max()))
        net_max = max(net_max, float(total["net_reconciliation_error"].abs().max()))
        residual_abs_sum = float(total["accounting_residual"].abs().sum())
        economic[key] = residual_abs_sum
        if residual_abs_sum > ECONOMIC_RESIDUAL_WARN:
            warnings.append(f"{key}:ECONOMIC_RESIDUAL_ABS_SUM_GT_0.25PP:{residual_abs_sum:.10f}")
        fee_diff = float(result["fees"]["total_fee_delta"].abs().sum()) if "total_fee_delta" in result["fees"] else 0.0
        fee_recon[key] = fee_diff
        if fee_diff > 0.01:
            errors.append(f"{key}:FEE_RECONCILIATION_FAILED:{fee_diff:.10f}")
        account_fee_diff = float(total["account_fee_reconciliation_error"].abs().max())
        account_fee_recon[key] = account_fee_diff
        if account_fee_diff > 1e-10:
            errors.append(f"{key}:ACCOUNT_FEE_FILE_RECONCILIATION_FAILED:{account_fee_diff:.12g}")
        # Fund returns are joined by exact date; no future return can enter a
        # row because the contribution source is always previous-date weights.
        if bool((result["daily_fund_rows"]["lagged_weight"] < -EPS).any()):
            errors.append(f"{key}:NEGATIVE_LAGGED_WEIGHT")
    overlap_errors: list[str] = []
    for key, frame in event_frames.items():
        if frame.empty:
            continue
        starts = pd.to_datetime(frame["interval_start"])
        ends = pd.to_datetime(frame["interval_end"])
        if bool((starts.iloc[1:].to_numpy() <= ends.iloc[:-1].to_numpy()).any()):
            overlap_errors.append(key)
    if overlap_errors:
        errors.append("EVENT_INTERVAL_OVERLAP:" + ",".join(overlap_errors))
    if accounting_max > EPS:
        errors.append(f"ACCOUNTING_IDENTITY_FAILED:{accounting_max:.12g}")
    if net_max > 1e-10:
        warnings.append(f"NET_RECONCILIATION_ERROR_GT_1E-10:{net_max:.12g}")
    state = "FAILED" if errors else ("COMPLETE_WITH_WARNINGS" if warnings else "COMPLETE")
    return {
        "state": state,
        "scope": "explanatory_attribution_of_existing_frozen_runs",
        "not_new_oos": True,
        "not_candidate_selection": True,
        "input_hash_gate": hash_gate,
        "accounting_identity_gate": {"passed": accounting_max <= EPS, "max_abs_daily_error": accounting_max, "tolerance": EPS},
        "net_reconciliation_gate": {"passed": net_max <= 1e-10, "max_abs_error": net_max, "tolerance": 1e-10},
        "fee_reconciliation_gate": {"passed": not any("FEE_RECONCILIATION_FAILED" in item or "ACCOUNT_FEE_FILE_RECONCILIATION_FAILED" in item for item in errors), "max_aggregate_abs_delta": max(fee_recon.values(), default=0.0), "max_account_vs_fee_file_abs_delta": max(account_fee_recon.values(), default=0.0)},
        "future_data_timing_gate": {"passed": future_data_passed, "rule": "date_t uses actual_weights from the immediately previous valuation date"},
        "event_interval_gate": {"passed": not overlap_errors, "overlap_strategies": overlap_errors},
        "economic_residual_abs_sum": economic,
        "economic_residual_warn_threshold": ECONOMIC_RESIDUAL_WARN,
        "warnings": warnings,
        "errors": errors,
    }


def build_input_facts(specs: Iterable[AttributionSpec], hash_gate: dict[str, Any]) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "scope": "Phase A frozen-run holdings-based explanatory attribution",
        "historical_truth": "inherited from source runs; not re-established here",
        "new_oos": False,
        "strategy_candidate": False,
        "weight_timing": "previous valuation date actual_weights",
        "fund_return_source": "otf_fund_nav.daily_growth_pct / 100",
        "cash_return_assumption": 0.0,
        "hash_gate": hash_gate,
        "strategies": {},
    }
    for spec in specs:
        files = {}
        for name in ("daily_account.csv", "actual_weights.csv", "orders.csv", "fees.csv", "turnover.csv", "market_states.csv", "input_hashes.json", "metrics.json", "config_snapshot.json"):
            path = spec.bundle / name
            files[name] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} if path.exists() else {"path": str(path), "missing": True}
        facts["strategies"][spec.key] = {
            "strategy": spec.strategy,
            "bundle": str(spec.bundle),
            "core_codes": sorted(spec.core_codes),
            "satellite_codes": sorted(spec.satellite_codes),
            "source_files": files,
            "db_path": str(spec.db_path),
        }
    return facts


def run_attribution(
    specs: list[AttributionSpec],
    output_dir: str | Path,
    root: Path = ROOT,
) -> dict[str, Any]:
    hash_gate = validate_frozen_inputs(specs, root=root)
    if not hash_gate["passed"]:
        raise RuntimeError("frozen input hash gate failed: " + "; ".join(hash_gate["mismatches"]))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    results: dict[str, dict[str, Any]] = {}
    for spec in specs:
        bundle = spec.bundle
        daily = _read_csv(bundle / "daily_account.csv")
        weights = _read_csv(bundle / "actual_weights.csv")
        orders = _read_csv(bundle / "orders.csv")
        fees = _read_csv(bundle / "fees.csv")
        turnover = _read_csv(bundle / "turnover.csv")
        codes = [c for c in weights.columns if c != "date"]
        metadata = load_fund_metadata(spec.db_path, codes)
        returns = load_fund_returns(spec.db_path, codes)
        fund_rows, total_rows = compute_daily_attribution(
            daily, weights, returns, metadata, fees=fees, strategy=spec.strategy,
            core_codes=spec.core_codes, satellite_codes=spec.satellite_codes,
        )
        fund_attr, sleeve_attr, period_attr = aggregate_attribution(fund_rows, total_rows, metadata, spec.strategy)
        metrics = _json_load(bundle / "metrics.json")
        results[spec.key] = {
            "spec": spec, "metrics": metrics, "daily_fund_rows": fund_rows,
            "total_rows": total_rows, "fund_attribution": fund_attr,
            "sleeve_attribution": sleeve_attr, "period_attribution": period_attr,
            "fees": fees, "orders": orders, "turnover": turnover,
        }
    b2_account = _date_frame(_read_csv(next(s.bundle for s in specs if s.key == "B2") / "daily_account.csv"))
    event_frames: dict[str, pd.DataFrame] = {}
    for key, result in results.items():
        event_frames[key] = build_event_attribution(result["spec"], b2_account)
    relative = build_relative_attribution(results)
    status = build_status(results, hash_gate, event_frames)
    facts = build_input_facts(specs, hash_gate)

    daily_all = pd.concat([pd.concat([r["daily_fund_rows"], r["total_rows"]], ignore_index=True, sort=False) for r in results.values()], ignore_index=True, sort=False)
    fund_all = pd.concat([r["fund_attribution"] for r in results.values()], ignore_index=True, sort=False)
    sleeve_all = pd.concat([r["sleeve_attribution"] for r in results.values()], ignore_index=True, sort=False)
    period_all = pd.concat([r["period_attribution"] for r in results.values()], ignore_index=True, sort=False)
    asset_class_all = (
        period_all.groupby(["strategy", "period", "asset_class", "sleeve"], dropna=False, sort=True)
        .agg(
            gross_contribution=("gross_contribution", "sum"),
            fee_drag=("fee_drag", "sum"),
            holding_days=("holding_days", "sum"),
            average_actual_weight=("average_actual_weight", "sum"),
            average_lagged_weight=("average_lagged_weight", "sum"),
            positive_gross_contribution=("positive_gross_contribution", "sum"),
            negative_gross_contribution=("negative_gross_contribution", "sum"),
            missing_nav_weight_days=("missing_nav_weight_days", "sum"),
            net_contribution_approx=("net_contribution_approx", "sum"),
        )
        .reset_index()
    )
    asset_denominator = asset_class_all.groupby("strategy")["positive_gross_contribution"].transform("sum")
    asset_class_all["positive_gross_pnl_ratio"] = np.where(
        asset_denominator > 0,
        asset_class_all["positive_gross_contribution"] / asset_denominator,
        np.nan,
    )
    event_all = pd.concat([frame for frame in event_frames.values() if not frame.empty], ignore_index=True, sort=False) if any(not frame.empty for frame in event_frames.values()) else pd.DataFrame()
    daily_all["date"] = daily_all["date"].astype(str)
    for name, frame in (("daily_attribution.csv", daily_all), ("fund_attribution.csv", fund_all), ("sleeve_attribution.csv", sleeve_all), ("period_attribution.csv", period_all), ("asset_class_attribution.csv", asset_class_all), ("event_attribution.csv", event_all), ("relative_attribution.csv", relative)):
        frame.to_csv(output / name, index=False, encoding="utf-8-sig")

    concentration_rows: list[dict[str, Any]] = []
    for key, result in results.items():
        fund = result["fund_attribution"].copy()
        positive = fund[fund["positive_gross_contribution"] > 0].sort_values("positive_gross_contribution", ascending=False)
        denom = float(positive["positive_gross_contribution"].sum())
        period = result["period_attribution"].copy()
        p2426 = period[period["period"] == "2024_2026"]
        full = period[period["period"] == "full"]
        core_gross = float(fund.loc[fund["sleeve"] == "core", "gross_contribution"].sum())
        satellite_gross = float(fund.loc[fund["sleeve"] == "satellite", "gross_contribution"].sum())
        concentration_rows.append({
            "strategy": result["spec"].strategy,
            "top1_positive_gross_pnl_ratio": float(positive.iloc[0]["positive_gross_contribution"] / denom) if denom > 0 and len(positive) else np.nan,
            "top3_positive_gross_pnl_ratio": float(positive.head(3)["positive_gross_contribution"].sum() / denom) if denom > 0 else np.nan,
            "2024_2026_positive_gross_pnl_share_of_full_positive": float(p2426["positive_gross_contribution"].sum() / denom) if denom > 0 else np.nan,
            "2024_2026_net_contribution_share_of_full_net": float(p2426["net_contribution_approx"].sum() / full["net_contribution_approx"].sum()) if not full.empty and abs(float(full["net_contribution_approx"].sum())) > EPS else np.nan,
            "core_gross_contribution": core_gross,
            "satellite_gross_contribution": satellite_gross,
            "core_net_contribution_approx": float(fund.loc[fund["sleeve"] == "core", "net_contribution_approx"].sum()),
            "satellite_net_contribution_approx": float(fund.loc[fund["sleeve"] == "satellite", "net_contribution_approx"].sum()),
        })
    concentration = pd.DataFrame(concentration_rows)
    concentration.to_csv(output / "concentration.csv", index=False, encoding="utf-8-sig")
    facts["concentration"] = concentration.to_dict("records")

    reconciliation: dict[str, Any] = {"strategies": {}}
    for key, result in results.items():
        total = result["total_rows"]
        reconciliation["strategies"][key] = {
            "daily_rows": int(len(total)),
            "full_period_gross_account_return_additive": float(total["gross_return"].sum()),
            "full_period_fund_contribution_additive": float(total["fund_contribution"].sum()),
            "full_period_cash_contribution_additive": float(total["cash_contribution"].sum()),
            "full_period_residual_additive": float(total["accounting_residual"].sum()),
            "full_period_accounting_error": float(total["accounting_error"].sum()),
            "max_abs_daily_accounting_error": float(total["accounting_error"].abs().max()),
            "economic_residual_abs_sum": float(total["accounting_residual"].abs().sum()),
            "fee_amount": float(total["fee_amount"].sum()),
            "account_total_fee_amount": float(total["account_total_fee_amount"].sum()),
            "account_vs_fee_file_fee_delta_abs_max": float(total["account_fee_reconciliation_error"].abs().max()),
            "cost_drag_sum": float(total["cost_drag"].sum()),
            "fee_file_total_delta_abs_sum": float(result["fees"]["total_fee_delta"].abs().sum()),
            "max_abs_net_reconciliation_error": float(total["net_reconciliation_error"].abs().max()),
        }
    (output / "input_facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (output / "reconciliation.json").write_text(json.dumps(reconciliation, ensure_ascii=False, indent=2), encoding="utf-8")
    status["concentration_file"] = str(output / "concentration.csv")
    status["event_counts"] = {key: int(len(frame)) for key, frame in event_frames.items()}
    (output / "attribution_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    conclusion_lines = [
        "# Phase A 收益归因结论",
        "",
        "本文件是对既有 C1/C2/B2/B2-LT 冻结账户的 holdings-based explanatory attribution，不是新 OOS、候选筛选或因果 Alpha 证明。",
        "",
        f"状态：**{status['state']}**。输入哈希闸门：{hash_gate['passed']}；会计恒等式闸门：{status['accounting_identity_gate']['passed']}。",
        "",
        "## 口径",
        "",
        "日期 t 的基金贡献严格使用 t-1 估值日 actual_weights × t 日 otf_fund_nav.daily_growth_pct/100；现金收益单列并按 0 收益处理。未匹配 NAV、订单确认/份额变化和账户计价差异进入 residual，未被隐藏。费用按账户 transaction_cost 做净收益勾稽，并同时核对 fees.csv。",
        "",
        "## 证据结论",
        "",
    ]
    for row in concentration_rows:
        conclusion_lines.append(
            f"- {row['strategy']}：Top1/Top3 正毛收益贡献集中度为 "
            f"{row['top1_positive_gross_pnl_ratio']:.2%}/{row['top3_positive_gross_pnl_ratio']:.2%}；"
            f"2024-2026 占全期正毛收益 {row['2024_2026_positive_gross_pnl_share_of_full_positive']:.2%}；"
            f"核心/卫星毛贡献 {row['core_gross_contribution']:.6f}/{row['satellite_gross_contribution']:.6f}。"
        )
    for key, result in results.items():
        total = result["total_rows"]
        years = total.groupby(total["date"].str[:4])["gross_return"].sum().to_dict()
        conclusion_lines.append(f"- {result['spec'].strategy} 年度 gross_return 加总：{json.dumps(years, ensure_ascii=False)}。")
        events = event_frames.get(key, pd.DataFrame())
        if not events.empty:
            best = events.loc[events["gross_return"].idxmax()]
            worst = events.loc[events["gross_return"].idxmin()]
            conclusion_lines.append(
                f"- {result['spec'].strategy} 事件区间最好/最差："
                f"{best['signal_date']}→{best['interval_end']} gross {best['gross_return']:.4%}；"
                f"{worst['signal_date']}→{worst['interval_end']} gross {worst['gross_return']:.4%}。"
            )
    conclusion_lines.extend([
        "",
        "## 审计回答",
        "",
        "- C1 的优势应结合 Top1/Top3 集中度和 2024-2026 正毛收益占比阅读；这只能证明收益来源的集中程度，不能证明因果 Alpha。",
        "- C2 的低收益来源应以核心/卫星和年度、事件表中的实际贡献分解判断；本报告不将防守暴露、空置卫星、择时或费用在证据不足时强行归因为单一原因。",
        "- 可进入后续研究的只是已被归因表支持的结构性观察；本报告不新增、不优化任何 C3 参数，也不把归因结果标记为候选策略。",
    ])
    if status["warnings"]:
        conclusion_lines.extend(["", "## WARN", ""] + [f"- {item}" for item in status["warnings"]])
    conclusion_lines.extend([
        "",
        "## 局限",
        "",
        "1. 这是持仓解释归因，不是严格 Brinson 分解，不能将暴露贡献称为因果 Alpha。",
        "2. 账户源 run 没有完整的历史 NAV publication timestamp；本报告继承该限制。",
        "3. residual 反映了现有账户估值、订单确认、缺失 NAV 和计价口径之间的差异；经济 residual 超过阈值时仅 WARN，不强行归因。",
    ])
    (output / "attribution_conclusion.md").write_text("\n".join(conclusion_lines) + "\n", encoding="utf-8")

    manifest: dict[str, Any] = {
        "artifact_schema_version": "phase_a_attribution_v1",
        "scope": "existing_frozen_runs_only",
        "run_id": output.name,
        "input_hashes": hash_gate,
        "artifacts": {},
    }
    for path in sorted(output.iterdir()):
        if path.name == "manifest.json":
            continue
        if path.is_file():
            rows = None
            columns = None
            if path.suffix.lower() == ".csv":
                frame = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
                columns = list(frame.columns)
                rows = sum(1 for _ in path.open("r", encoding="utf-8-sig")) - 1
            manifest["artifacts"][path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size, "rows": rows, "columns": columns}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output_dir": str(output), "status": status, "manifest": manifest, "results": results}
