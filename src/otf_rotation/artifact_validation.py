"""Schema and content validation for strategy research artifacts.

The validator is intentionally independent from the runner so it can be used
by both the Gate and end-to-end tests.  A file that merely exists is not a
valid research artifact: it must be readable, have the expected columns and
contain internally consistent dates, identifiers and reconciliation fields.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_ARTIFACTS = (
    "manifest.json",
    "config_snapshot.json",
    "input_hashes.json",
    "daily_account.csv",
    "daily_returns.csv",
    "target_weights.csv",
    "actual_weights.csv",
    "market_states.csv",
    "state_scores.csv",
    "asset_budgets.csv",
    "sleeve_weights.csv",
    "fund_weights.csv",
    "product_selection_audit.csv",
    "orders.csv",
    "order_rejections.csv",
    "position_lots.csv",
    "fees.csv",
    "fee_reconciliation_orders.csv",
    "turnover.csv",
    "risk_contributions.csv",
    "metrics.json",
    "gate_result.json",
    "parameter_freeze.json",
)


CSV_SCHEMAS: dict[str, tuple[str, ...]] = {
    "daily_account.csv": (
        "date", "equity", "daily_return", "gross_return", "total_fee_amount"
    ),
    "daily_returns.csv": ("date", "equity", "daily_return", "gross_return"),
    "target_weights.csv": ("date",),
    "actual_weights.csv": ("date",),
    "market_states.csv": ("signal_date", "market_state"),
    "state_scores.csv": ("date", "state", "score"),
    "asset_budgets.csv": ("date", "sleeve", "weight"),
    "sleeve_weights.csv": ("date", "sleeve", "weight"),
    "fund_weights.csv": ("date", "fund_code", "weight"),
    "product_selection_audit.csv": (
        "sleeve", "date", "top_n", "selected_count", "total_candidates",
        "eligible_count", "selected_funds",
    ),
    "orders.csv": (
        "order_id", "fund_code", "side", "status", "signal_date", "submit_date",
        "confirmation_date", "redemption_arrival_date", "requested_amount",
        "cash_frozen", "confirmed_nav", "shares_confirmed", "filled_notional",
        "settled_cash_notional", "fee_paid", "effective_fee_rate",
    ),
    "order_rejections.csv": ("fund_code", "side", "date", "reason"),
    "position_lots.csv": (
        "fund_code", "lot_id", "acquired_date", "shares", "reserved_shares"
    ),
    "fees.csv": (
        "date", "daily_subscription_fee_amount", "daily_redemption_fee_amount",
        "daily_total_fee_amount", "order_subscription_fee_amount",
        "order_redemption_fee_amount", "order_total_fee_amount",
        "subscription_fee_delta", "redemption_fee_delta", "total_fee_delta",
        "reconciled",
    ),
    "fee_reconciliation_orders.csv": (
        "order_id", "status", "confirmation_date", "side", "fee_paid",
        "reconciled_in_daily_fees",
    ),
    "turnover.csv": (
        "date", "buy_notional", "sell_notional", "gross_traded_notional",
        "bilateral_turnover", "submitted_buy_notional", "submitted_sell_notional",
        "submitted_gross_notional", "submitted_bilateral_turnover",
        "settled_cash_buy_notional", "settled_cash_sell_notional",
        "settled_cash_gross_notional", "settled_cash_turnover",
        "pending_requested_amount",
    ),
    "risk_contributions.csv": ("date", "fund_code", "risk_contribution"),
}

# The schema is the single source of truth for tabular artifact validation.
# Keep the historical CSV_SCHEMAS mapping above as a small compatibility
# surface for callers that only need the required columns.
CSV_ARTIFACT_SPECS: dict[str, dict[str, Any]] = {
    name: {"columns": columns, "date_columns": (), "numeric_columns": ()}
    for name, columns in CSV_SCHEMAS.items()
}
CSV_ARTIFACT_SPECS.update(
    {
        "daily_account.csv": {
            "columns": CSV_SCHEMAS["daily_account.csv"],
            "date_columns": ("date", "signal_date"),
            "required_date_columns": ("date",),
            "numeric_columns": (
                "equity", "daily_return", "gross_return", "total_fee_amount",
            ),
            "nonempty": True,
        },
        "daily_returns.csv": {
            "columns": CSV_SCHEMAS["daily_returns.csv"],
            "date_columns": ("date",),
            "required_date_columns": ("date",),
            "numeric_columns": ("equity", "daily_return", "gross_return"),
            "nonempty": True,
        },
        "target_weights.csv": {
            "columns": CSV_SCHEMAS["target_weights.csv"],
            "date_columns": ("date",),
            "required_date_columns": ("date",),
            "numeric_columns": (),
            "weight_table": True,
            "nonempty": True,
        },
        "actual_weights.csv": {
            "columns": CSV_SCHEMAS["actual_weights.csv"],
            "date_columns": ("date",),
            "required_date_columns": ("date",),
            "numeric_columns": (),
            "weight_table": True,
            "nonempty": True,
        },
        "market_states.csv": {
            "columns": CSV_SCHEMAS["market_states.csv"],
            "date_columns": ("signal_date",),
            "required_date_columns": ("signal_date",),
            "nonempty": True,
        },
        "state_scores.csv": {
            "columns": CSV_SCHEMAS["state_scores.csv"],
            "date_columns": ("date",),
            "required_date_columns": ("date",),
            "numeric_columns": ("score",),
            "nonempty": True,
        },
        "asset_budgets.csv": {
            "columns": CSV_SCHEMAS["asset_budgets.csv"],
            "date_columns": ("date",),
            "required_date_columns": ("date",),
            "numeric_columns": ("weight",),
            "nonempty": True,
        },
        "sleeve_weights.csv": {
            "columns": CSV_SCHEMAS["sleeve_weights.csv"],
            "date_columns": ("date",),
            "required_date_columns": ("date",),
            "numeric_columns": ("weight",),
            "nonempty": True,
        },
        "fund_weights.csv": {
            "columns": CSV_SCHEMAS["fund_weights.csv"],
            "date_columns": ("date",),
            "required_date_columns": ("date",),
            "numeric_columns": ("weight",),
            "nonempty": True,
        },
        "product_selection_audit.csv": {
            "columns": CSV_SCHEMAS["product_selection_audit.csv"],
            "date_columns": ("date",),
            "required_date_columns": ("date",),
            "numeric_columns": (
                "top_n", "selected_count", "total_candidates", "eligible_count",
            ),
        },
        "orders.csv": {
            "columns": CSV_SCHEMAS["orders.csv"],
            "date_columns": (
                "signal_date", "submit_date", "confirmation_date",
                "redemption_arrival_date",
            ),
            "required_date_columns": ("signal_date", "submit_date"),
            "numeric_columns": (
                "requested_amount", "cash_frozen", "confirmed_nav",
                "shares_confirmed", "filled_notional", "settled_cash_notional",
                "fee_paid", "effective_fee_rate",
            ),
        },
        "order_rejections.csv": {
            "columns": CSV_SCHEMAS["order_rejections.csv"],
            "date_columns": ("date",),
            "required_date_columns": ("date",),
        },
        "position_lots.csv": {
            "columns": CSV_SCHEMAS["position_lots.csv"],
            "date_columns": ("acquired_date",),
            "required_date_columns": ("acquired_date",),
            "numeric_columns": ("shares", "reserved_shares"),
        },
        "fees.csv": {
            "columns": CSV_SCHEMAS["fees.csv"],
            "date_columns": ("date",),
            "required_date_columns": ("date",),
            "numeric_columns": (
                "daily_subscription_fee_amount", "daily_redemption_fee_amount",
                "daily_total_fee_amount", "order_subscription_fee_amount",
                "order_redemption_fee_amount", "order_total_fee_amount",
                "subscription_fee_delta", "redemption_fee_delta", "total_fee_delta",
            ),
            "nonempty": True,
        },
        "fee_reconciliation_orders.csv": {
            "columns": CSV_SCHEMAS["fee_reconciliation_orders.csv"],
            "date_columns": ("confirmation_date",),
            "numeric_columns": ("fee_paid",),
        },
        "turnover.csv": {
            "columns": CSV_SCHEMAS["turnover.csv"],
            "date_columns": ("date",),
            "required_date_columns": ("date",),
            "numeric_columns": (
                "buy_notional", "sell_notional", "gross_traded_notional",
                "bilateral_turnover", "submitted_buy_notional",
                "submitted_sell_notional", "submitted_gross_notional",
                "submitted_bilateral_turnover", "settled_cash_buy_notional",
                "settled_cash_sell_notional", "settled_cash_gross_notional",
                "settled_cash_turnover", "pending_requested_amount",
            ),
            "nonempty": True,
        },
        "risk_contributions.csv": {
            "columns": CSV_SCHEMAS["risk_contributions.csv"],
            "date_columns": ("date",),
            "required_date_columns": ("date",),
            "numeric_columns": ("risk_contribution",),
            "nonempty": True,
        },
    }
)


def artifact_status_for_strategy(strategy_name: str, filename: str) -> str:
    """Return the schema-defined status for one strategy artifact."""
    if filename == "market_states.csv":
        # market_states is REQUIRED for all core/satellite strategies including C3
        return "REQUIRED" if strategy_name.startswith(("S1_", "D1_", "C1_", "C2_", "C3_")) else "NOT_APPLICABLE"
    if filename in {"state_scores.csv", "asset_budgets.csv", "sleeve_weights.csv", "fund_weights.csv"}:
        # These are only REQUIRED for strategies that generate state/sleeve/fund breakdowns (not C3)
        return "REQUIRED" if strategy_name.startswith(("S1_", "D1_", "C1_", "C2_")) else "NOT_APPLICABLE"
    if filename == "risk_contributions.csv":
        return "REQUIRED" if strategy_name.startswith(("B3_", "C1_", "C2_")) else "NOT_APPLICABLE"
    if filename == "product_selection_audit.csv":
        return "REQUIRED" if strategy_name.startswith(("S2_", "D1_", "C1_", "C2_")) else "NOT_APPLICABLE"
    return "REQUIRED"


CORE_NONEMPTY_ARTIFACTS = {
    "daily_account.csv", "daily_returns.csv", "target_weights.csv",
    "actual_weights.csv", "orders.csv", "fees.csv", "turnover.csv",
}


JSON_SCHEMAS: dict[str, tuple[str, ...]] = {
    "config_snapshot.json": ("run_id", "account_mode", "requested_oos_period", "actual_oos_period"),
    "input_hashes.json": (),
    "metrics.json": (
        "account_mode", "n_days", "oos_start", "oos_end", "net_cagr_pct",
        "sharpe", "mdd_pct", "total_fee_amount", "max_annual_bilateral_turnover",
        "fee_reconciliation_passed",
    ),
    "gate_result.json": ("gate_passed", "checks", "failed_checks"),
    "parameter_freeze.json": ("mode", "strategy", "parameter_freeze_id"),
    "manifest.json": ("run_id", "strategy_name", "input_hashes", "artifacts"),
}


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            return None, "JSON root must be an object"
        return value, None
    except Exception as exc:  # pragma: no cover - error text is part of the audit
        return None, f"unreadable JSON: {exc}"


def _read_csv(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    try:
        return pd.read_csv(path, encoding="utf-8-sig"), None
    except Exception as exc:  # pragma: no cover - error text is part of the audit
        return None, f"unreadable CSV: {exc}"


def _nonempty(value: Any) -> bool:
    return value is not None and str(value).strip() != "" and str(value).lower() != "nan"


def _is_hash(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(value or "")))


def _conditional_status(strategy_name: str, filename: str) -> str:
    """Backward-compatible alias for the schema applicability function."""
    return artifact_status_for_strategy(strategy_name, filename)


def _json_weights_valid(frame: pd.DataFrame, column: str) -> list[str]:
    errors: list[str] = []
    if column not in frame:
        return errors
    for index, raw in frame[column].items():
        try:
            value = json.loads(raw) if isinstance(raw, str) else {}
            if not isinstance(value, dict):
                errors.append(f"{column}[{index}] is not an object")
                continue
            numeric = {str(key): float(weight) for key, weight in value.items()}
            if any(weight < -1e-10 for weight in numeric.values()):
                errors.append(f"{column}[{index}] contains negative weight")
            if sum(numeric.values()) > 1.0 + 1e-6:
                errors.append(f"{column}[{index}] sums above 1")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{column}[{index}] invalid JSON: {exc}")
    return errors


def _validate_numeric_columns(
    frame: pd.DataFrame,
    filename: str,
    columns: tuple[str, ...],
) -> list[str]:
    errors: list[str] = []
    for column in columns:
        if column not in frame:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted[frame[column].notna()].isna().any():
            errors.append(f"{filename}:non_numeric_{column}")
        if (converted.dropna() < -1e-12).any() and column not in {
            "daily_return", "gross_return", "score",
            "total_fee_delta", "subscription_fee_delta", "redemption_fee_delta",
        }:
            errors.append(f"{filename}:negative_{column}")
    return errors


def _validate_dates(
    frame: pd.DataFrame,
    filename: str,
    date_columns: tuple[str, ...],
    required_date_columns: tuple[str, ...],
) -> list[str]:
    errors: list[str] = []
    for column in date_columns:
        if column not in frame:
            continue
        parsed = pd.to_datetime(frame[column], errors="coerce")
        if column in required_date_columns and parsed.isna().any():
            errors.append(f"{filename}:invalid_dates:{column}")
        elif column not in required_date_columns and parsed[frame[column].notna()].isna().any():
            errors.append(f"{filename}:invalid_nullable_dates:{column}")
    return errors


def _inventory_entry(path: Path, status: str) -> dict[str, Any]:
    """Build the on-disk inventory entry used by both export and validation."""
    info: dict[str, Any] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "status": status,
        "generated_at": pd.Timestamp.fromtimestamp(
            path.stat().st_mtime, tz="UTC"
        ).isoformat(),
    }
    if path.suffix.lower() == ".csv":
        frame, error = _read_csv(path)
        if error:
            info["parse_error"] = error
        else:
            assert frame is not None
            info.update({"rows": int(len(frame)), "columns": list(frame.columns)})
    elif path.suffix.lower() == ".json":
        payload, error = _read_json(path)
        if error:
            info["parse_error"] = error
        else:
            assert payload is not None
            info["keys"] = sorted(payload)
    return info


def validate_artifact_bundle(
    output_dir: str | Path,
    *,
    strategy_name: str | None = None,
    expected_run_id: str | None = None,
    expected_start: str | None = None,
    expected_end: str | None = None,
) -> dict[str, Any]:
    """Validate a strategy artifact directory and return an auditable result.

    The validator deliberately checks the files on disk and then compares the
    complete result with the manifest inventory.  This makes a modified
    column, stale row count, wrong applicability status, or an unlisted file a
    content failure rather than a mere existence success.
    """
    directory = Path(output_dir)
    errors: list[str] = []
    artifact_status: dict[str, str] = {}
    inventory: dict[str, dict[str, Any]] = {}

    for filename in REQUIRED_ARTIFACTS:
        path = directory / filename
        status = artifact_status_for_strategy(strategy_name or "", filename)
        artifact_status[filename] = status
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"missing_or_empty:{filename}")
            continue

        info = _inventory_entry(path, status)
        inventory[filename] = info
        if "parse_error" in info:
            errors.append(f"{filename}:{info['parse_error']}")
            continue

        if filename.endswith(".csv"):
            frame, error = _read_csv(path)
            if error:
                errors.append(f"{filename}:{error}")
                continue
            assert frame is not None
            spec = CSV_ARTIFACT_SPECS[filename]
            missing_columns = [
                column for column in spec["columns"] if column not in frame.columns
            ]
            if missing_columns:
                errors.append(f"{filename}:missing_columns={missing_columns}")
            if status == "NOT_APPLICABLE":
                if not frame.empty:
                    errors.append(f"{filename}:must_be_empty_for_{strategy_name}")
                # Header-only is the only valid representation of N/A.
                if list(frame.columns) != list(spec["columns"]):
                    errors.append(f"{filename}:schema_columns_mismatch_for_not_applicable")
                continue
            if bool(spec.get("nonempty")) and frame.empty:
                errors.append(f"{filename}:required_nonempty_for_{strategy_name}")
            errors.extend(
                _validate_dates(
                    frame,
                    filename,
                    tuple(spec.get("date_columns", ())),
                    tuple(spec.get("required_date_columns", ())),
                )
            )
            errors.extend(
                _validate_numeric_columns(
                    frame, filename, tuple(spec.get("numeric_columns", ()))
                )
            )
            if filename in {"daily_account.csv", "daily_returns.csv"} and "date" in frame:
                dates = pd.to_datetime(frame["date"], errors="coerce")
                if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
                    errors.append(f"{filename}:invalid_or_nonmonotonic_dates")
                if expected_start and not frame.empty and dates.min().strftime("%Y-%m-%d") != expected_start:
                    errors.append(f"{filename}:start_date_mismatch")
                if expected_end and not frame.empty and dates.max().strftime("%Y-%m-%d") != expected_end:
                    errors.append(f"{filename}:end_date_mismatch")
            if filename == "daily_account.csv" and not frame.empty:
                errors.extend(_json_weights_valid(frame, "target_weights"))
                errors.extend(_json_weights_valid(frame, "actual_weights"))
                for column in ("equity", "daily_return", "gross_return", "total_fee_amount"):
                    if column in frame and pd.to_numeric(frame[column], errors="coerce").isna().any():
                        errors.append(f"{filename}:non_numeric_{column}")
            if bool(spec.get("weight_table")) and not frame.empty:
                forbidden = {"cash_mgt", "MONEY_MARKET", "CD_NCD"}
                if forbidden.intersection(map(str, frame.columns)):
                    errors.append(f"{filename}:sleeve_key_leaked_into_fund_weights")
                value_columns = [column for column in frame.columns if column != "date"]
                values = frame[value_columns].apply(pd.to_numeric, errors="coerce")
                if values.isna().any().any():
                    errors.append(f"{filename}:non_numeric_weight")
                if (values < -1e-12).any().any():
                    errors.append(f"{filename}:negative_weight")
                if (values.sum(axis=1) > 1.0 + 1e-6).any():
                    errors.append(f"{filename}:weights_sum_above_one")
            if filename == "fund_weights.csv" and not frame.empty:
                sleeve_keys = {"cash_mgt", "MONEY_MARKET", "CD_NCD"}
                if "fund_code" in frame and frame["fund_code"].astype(str).isin(sleeve_keys).any():
                    errors.append("fund_weights.csv:sleeve_key_leaked_into_fund_codes")
            if filename in {"sleeve_weights.csv", "asset_budgets.csv"} and not frame.empty:
                if "sleeve" in frame and frame["sleeve"].astype(str).isin({"160706", "000218", "001512", "260102"}).any():
                    errors.append(f"{filename}:fund_key_leaked_into_sleeves")
            if filename == "fees.csv" and not frame.empty:
                if "reconciled" in frame:
                    bool_values = frame["reconciled"].astype(str).str.strip().str.lower()
                    if not bool_values.isin({"true", "false", "1", "0"}).all():
                        errors.append("fees.csv:reconciled_not_boolean")
                    elif not bool_values.isin({"true", "1"}).all():
                        errors.append("fees.csv:fee_reconciliation_failed")
            if filename == "turnover.csv" and not frame.empty:
                for column in ("bilateral_turnover", "submitted_bilateral_turnover", "settled_cash_turnover"):
                    if column in frame and (pd.to_numeric(frame[column], errors="coerce") < -1e-12).any():
                        errors.append(f"turnover.csv:negative_{column}")
        else:
            payload, error = _read_json(path)
            if error:
                errors.append(f"{filename}:{error}")
                continue
            assert payload is not None
            missing_keys = [key for key in JSON_SCHEMAS[filename] if key not in payload]
            if missing_keys:
                errors.append(f"{filename}:missing_keys={missing_keys}")
            if filename == "input_hashes.json":
                if not payload or any(not _is_hash(value) for value in payload.values()):
                    errors.append("input_hashes.json:all_input_hashes_must_be_sha256")
            if filename == "parameter_freeze.json" and payload.get("mode") != "FROZEN_PARAMETER_CONTINUOUS_OOS":
                errors.append("parameter_freeze.json:mode_not_frozen_continuous_oos")
            if filename == "gate_result.json":
                checks = payload.get("checks")
                failed = payload.get("failed_checks")
                if not isinstance(checks, dict) or not checks:
                    errors.append("gate_result.json:checks_missing_or_empty")
                elif not all(isinstance(value, bool) for value in checks.values()):
                    errors.append("gate_result.json:checks_must_be_boolean")
                if not isinstance(failed, list):
                    errors.append("gate_result.json:failed_checks_not_list")
                elif isinstance(checks, dict):
                    expected_failed = [name for name, value in checks.items() if not value]
                    if failed != expected_failed:
                        errors.append("gate_result.json:failed_checks_inconsistent")
                    if payload.get("gate_passed") != (not failed):
                        errors.append("gate_result.json:gate_passed_inconsistent")
            if filename == "manifest.json":
                manifest_strategy = payload.get("strategy_name")
                if strategy_name and manifest_strategy != strategy_name:
                    errors.append("manifest.json:strategy_name_mismatch")
                if expected_run_id and payload.get("run_id") != expected_run_id:
                    errors.append("manifest.json:run_id_mismatch")
                manifest_hashes = payload.get("input_hashes", {})
                if not isinstance(manifest_hashes, dict) or any(not _is_hash(value) for value in manifest_hashes.values()):
                    errors.append("manifest.json:input_hashes_invalid")
        # JSON inventory needs its own entry after successful parsing too.
        if filename not in inventory:
            inventory[filename] = info

    # Include any extra files so the manifest cannot silently omit them.
    if directory.exists():
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.name not in inventory:
                inventory[path.name] = _inventory_entry(
                    path,
                    artifact_status_for_strategy(strategy_name or "", path.name),
                )
                if path.name != "manifest.json":
                    errors.append(f"untracked_artifact:{path.name}")

    # Cross-file identity, dates, metrics, gate and manifest inventory checks.
    config, config_error = _read_json(directory / "config_snapshot.json")
    manifest, manifest_error = _read_json(directory / "manifest.json")
    metrics, metrics_error = _read_json(directory / "metrics.json")
    gate, gate_error = _read_json(directory / "gate_result.json")
    input_hashes, input_hashes_error = _read_json(directory / "input_hashes.json")
    errors.extend(
        error for error in (
            config_error, manifest_error, metrics_error, gate_error, input_hashes_error
        ) if error
    )
    if config and manifest:
        if config.get("run_id") != manifest.get("run_id"):
            errors.append("run_id_mismatch_between_config_and_manifest")
        if expected_run_id and config.get("run_id") != expected_run_id:
            errors.append("config_snapshot:run_id_mismatch")
        if manifest.get("config_snapshot") != config:
            errors.append("manifest.json:config_snapshot_mismatch")
    if config and metrics:
        actual = config.get("actual_oos_period") or []
        if len(actual) == 2 and [metrics.get("oos_start"), metrics.get("oos_end")] != actual:
            errors.append("metrics_actual_oos_period_mismatch")
        if metrics.get("account_mode") != config.get("account_mode"):
            errors.append("metrics_account_mode_mismatch")
    if metrics and gate:
        checks = gate.get("checks", {})
        for key in ("artifact_schema", "artifact_content"):
            if key in checks and f"{key}_gate" in metrics and metrics.get(f"{key}_gate") != checks[key]:
                errors.append(f"{key}_gate_mismatch_between_metrics_and_gate")
    if input_hashes and manifest:
        manifest_hashes = manifest.get("input_hashes", {})
        for key in ("db_sha256", "rules_sha256", "mapping_sha256", "config_sha256"):
            if input_hashes.get(key) != manifest_hashes.get(
                "exposure_mapping_sha256" if key == "mapping_sha256" else key
            ):
                errors.append(f"input_hashes_mismatch:{key}")
    if manifest and metrics and manifest.get("metrics") != metrics:
        errors.append("manifest.json:metrics_mismatch")
    if manifest and gate and manifest.get("gate_result") != gate:
        errors.append("manifest.json:gate_result_mismatch")
    if manifest:
        manifest_inventory = manifest.get("artifacts", {})
        actual_inventory = {
            name: info for name, info in inventory.items() if name != "manifest.json"
        }
        if not isinstance(manifest_inventory, dict):
            errors.append("manifest.json:artifact_inventory_missing")
        else:
            if set(manifest_inventory) != set(actual_inventory):
                errors.append("manifest.json:artifact_inventory_set_mismatch")
            for filename, info in actual_inventory.items():
                listed = manifest_inventory.get(filename)
                if not isinstance(listed, dict):
                    errors.append(f"manifest.json:artifact_not_listed:{filename}")
                    continue
                for field in ("sha256", "bytes", "status"):
                    if listed.get(field) != info.get(field):
                        errors.append(f"manifest.json:artifact_{field}_mismatch:{filename}")
                for field in ("rows", "columns", "keys"):
                    if field in info and listed.get(field) != info.get(field):
                        errors.append(f"manifest.json:artifact_{field}_mismatch:{filename}")
                if not _nonempty(listed.get("generated_at")):
                    errors.append(f"manifest.json:artifact_generated_at_missing:{filename}")
                expected_status = artifact_status_for_strategy(
                    str(manifest.get("strategy_name") or strategy_name or ""), filename
                )
                if listed.get("status") != expected_status:
                    errors.append(f"manifest.json:artifact_status_mismatch:{filename}")

    # The two primary daily tables must describe the same valued account.
    daily_account, _ = _read_csv(directory / "daily_account.csv")
    daily_returns, _ = _read_csv(directory / "daily_returns.csv")
    if (
        daily_account is not None and daily_returns is not None
        and "date" in daily_account and "date" in daily_returns
    ):
        account_dates = pd.to_datetime(daily_account["date"], errors="coerce")
        return_dates = pd.to_datetime(daily_returns["date"], errors="coerce")
        if list(account_dates) != list(return_dates):
            errors.append("daily_account_and_returns_date_mismatch")
        if metrics and metrics.get("n_days") != len(daily_account):
            errors.append("metrics_n_days_mismatch")

    return {
        "passed": not errors,
        "required_count": len(REQUIRED_ARTIFACTS),
        "present_count": sum(name in inventory for name in REQUIRED_ARTIFACTS),
        "errors": errors,
        "artifact_status": artifact_status,
        "inventory": inventory,
    }
