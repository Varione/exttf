"""Retrospective rolling validation for fixed OTC strategy outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_REPORT_DIR = Path("reports/mapped_otf_strategy/experiments")
GATE_THRESHOLDS = {
    "positive_cagr_ratio_min": 0.70,
    "median_sharpe_min": 0.40,
    "worst_max_drawdown_min": -0.25,
    "worst_cagr_min": -0.05,
}


def period_metrics(frame: pd.DataFrame) -> dict[str, float]:
    returns = pd.to_numeric(frame["daily_return"], errors="coerce").dropna()
    if len(returns) < 2:
        raise ValueError("INSUFFICIENT_RETURN_OBSERVATIONS")
    years = len(returns) / 252.0
    wealth = float((1.0 + returns).prod())
    cagr = wealth ** (1.0 / years) - 1.0
    volatility = float(returns.std(ddof=1) * np.sqrt(252))
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
        if returns.std(ddof=1) > 0
        else 0.0
    )
    nav = (1.0 + returns).cumprod()
    max_drawdown = float((nav / nav.cummax() - 1.0).min())
    return {
        "observations": int(len(returns)),
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def rolling_windows(
    daily: pd.DataFrame,
    *,
    years: int = 2,
    minimum_observations: int = 252,
) -> pd.DataFrame:
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    first_year = int(frame["date"].dt.year.min())
    last_date = frame["date"].max()
    rows: list[dict] = []
    for start_year in range(first_year, int(last_date.year) + 1):
        start = pd.Timestamp(f"{start_year}-01-01")
        end = start + pd.DateOffset(years=years) - pd.Timedelta(days=1)
        if end > last_date:
            continue
        sample = frame.loc[frame["date"].between(start, end)]
        if len(sample) < minimum_observations:
            continue
        rows.append(
            {
                "window": f"{start.date()}_{end.date()}",
                "start": start,
                "end": end,
                **period_metrics(sample),
            }
        )
    return pd.DataFrame(rows)


def evaluate_stability(windows: pd.DataFrame) -> dict:
    if windows.empty:
        raise ValueError("NO_VALID_ROLLING_WINDOWS")
    positive_ratio = float(windows["cagr"].gt(0).mean())
    median_sharpe = float(windows["sharpe"].median())
    worst_drawdown = float(windows["max_drawdown"].min())
    worst_cagr = float(windows["cagr"].min())
    checks = {
        "positive_cagr_ratio": positive_ratio
        >= GATE_THRESHOLDS["positive_cagr_ratio_min"],
        "median_sharpe": median_sharpe
        >= GATE_THRESHOLDS["median_sharpe_min"],
        "worst_max_drawdown": worst_drawdown
        >= GATE_THRESHOLDS["worst_max_drawdown_min"],
        "worst_cagr": worst_cagr >= GATE_THRESHOLDS["worst_cagr_min"],
    }
    return {
        "rolling_window_count": int(len(windows)),
        "positive_cagr_ratio": positive_ratio,
        "median_sharpe": median_sharpe,
        "worst_max_drawdown": worst_drawdown,
        "worst_cagr": worst_cagr,
        "checks": checks,
        "stability_gate_passed": bool(all(checks.values())),
        "validation_status": "RETROSPECTIVE_NOT_PRISTINE_OOS",
    }


def validate_report_directory(report_dir: Path = DEFAULT_REPORT_DIR) -> pd.DataFrame:
    summaries: list[dict] = []
    all_windows: list[pd.DataFrame] = []
    experiment_summary = pd.read_csv(report_dir / "summary.csv")
    for variant in experiment_summary["variant"]:
        path = report_dir / f"daily_{variant}.csv"
        daily = pd.read_csv(path, parse_dates=["date"])
        windows = rolling_windows(daily)
        result = evaluate_stability(windows)
        summaries.append({"variant": variant, **result})
        windows.insert(0, "variant", variant)
        all_windows.append(windows)
    summary = pd.DataFrame(summaries)
    flat_summary = summary.drop(columns=["checks"]).copy()
    for check_name in GATE_THRESHOLDS:
        key = check_name.removesuffix("_min")
        flat_summary[f"check_{key}"] = summary["checks"].map(
            lambda checks: checks[key]
        )
    flat_summary.to_csv(report_dir / "rolling_stability_summary.csv", index=False)
    pd.concat(all_windows, ignore_index=True).to_csv(
        report_dir / "rolling_two_year_windows.csv", index=False
    )
    payload = {
        "thresholds": GATE_THRESHOLDS,
        "validation_status": "RETROSPECTIVE_NOT_PRISTINE_OOS",
        "strategies": summaries,
    }
    (report_dir / "rolling_stability_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return flat_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    result = validate_report_directory(args.report_dir)
    columns = [
        "variant", "rolling_window_count", "positive_cagr_ratio",
        "median_sharpe", "worst_max_drawdown", "worst_cagr",
        "stability_gate_passed",
    ]
    print(result[columns].to_string(index=False))


if __name__ == "__main__":
    main()
