"""Walk-forward research framework with strict OOS isolation."""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest_engine import BacktestEngine


def _expand_dates(
    start_date: str,
    end_date: str,
    all_dates: pd.DatetimeIndex,
    train_years: int,
    val_years: int,
    test_years: int,
    roll_months: int,
) -> list[dict[str, str]]:
    """Generate walk-forward window boundaries anchored to trading days."""
    windows: list[dict[str, str]] = []
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    min_test_days = int(0.5 * 365.25)

    while True:
        train_start = current
        train_end_idx = _find_date_approx(all_dates, train_start, train_years)
        if train_end_idx is None or train_end_idx < 1:
            break
        train_end = all_dates[train_end_idx]

        val_end_idx = _find_date_approx(all_dates, train_end, val_years)
        if val_end_idx is None or val_end_idx <= train_end_idx:
            break
        val_start_idx = max(train_end_idx + 1, 0)
        val_start = all_dates[val_start_idx]
        val_end = all_dates[val_end_idx]

        test_end_idx = _find_date_approx(all_dates, val_end, test_years)
        if test_end_idx is None or test_end_idx <= val_end_idx:
            break
        test_start_idx = val_end_idx + 1
        if test_start_idx >= len(all_dates):
            break

        # Clamp test_end to end_date
        raw_test_end = all_dates[min(test_end_idx, len(all_dates) - 1)]
        test_end_actual = min(raw_test_end, end)

        # Skip windows with too few trading days in test
        test_span_days = (test_end_actual - all_dates[test_start_idx]).days
        if test_span_days < min_test_days:
            break

        test_start = all_dates[test_start_idx]

        windows.append(
            {
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "val_start": val_start.strftime("%Y-%m-%d"),
                "val_end": val_end.strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": test_end_actual.strftime("%Y-%m-%d"),
            }
        )

        if test_end_actual >= end:
            break

        roll_days = int(roll_months * 30.44)
        current = test_end_actual + timedelta(days=roll_days)
        if current > end:
            break

    return windows


def _find_date_approx(
    all_dates: pd.DatetimeIndex, anchor: pd.Timestamp, years: int
) -> int | None:
    """Find the index closest to anchor + years in trading days."""
    target = anchor + timedelta(days=int(years * 365.25))
    diffs = np.abs((all_dates - target).days)
    idx = int(diffs.argmin())
    if all_dates[idx] < anchor:
        return None
    return idx


def run_walk_forward(
    strategy_names: list[str],
    start_date: str = "2018-01-01",
    end_date: str = "2026-07-17",
    train_years: int = 3,
    val_years: int = 1,
    test_years: int = 1,
    roll_months: int = 6,
    *,
    factor_path: str = "data/processed/factors_all_repaired.csv",
    regime_path: str = "data/processed/regime_predictions.csv",
    db_path: str = "data/processed/etf.sqlite",
    data_mode: str = "etf",
    price_mode: str = "total_return_proxy",
    fee_rate_per_side: float = 0.0003,
    slippage_rate_per_side: float = 0.0002,
    n_hold: int = 20,
    max_weight: float = 0.05,
    signal_to_return_lag: int = 2,
    rebalance_every: int = 5,
    respect_regime: bool = True,
) -> pd.DataFrame:
    """Walk-forward backtest with strict OOS isolation.

    Each window uses train for parameter selection/validation,
    val for final parameter choice, and test for single-read evaluation.
    Here we evaluate each strategy on the test window using fixed parameters.
    """
    engine = BacktestEngine(
        factor_path=factor_path,
        regime_path=regime_path,
        db_path=db_path,
        data_mode=data_mode,
        price_mode=price_mode,
        fee_rate_per_side=fee_rate_per_side,
        slippage_rate_per_side=slippage_rate_per_side,
        require_pit=True,
        require_full_pit=False,
    )

    windows = _expand_dates(
        start_date, end_date, engine.dates,
        train_years, val_years, test_years, roll_months,
    )

    if not windows:
        print("No walk-forward windows generated; adjusting date ranges.")
        return pd.DataFrame()

    print(f"Walk-forward: {len(windows)} windows generated")
    for i, w in enumerate(windows):
        print(f"  Window {i}: train={w['train_start']}~{w['train_end']}, "
              f"val={w['val_start']}~{w['val_end']}, "
              f"test={w['test_start']}~{w['test_end']}")

    rows: list[dict] = []
    for window_id, w in enumerate(windows):
        # Run test window for each strategy
        for strategy_name in strategy_names:
            try:
                daily = engine.run_backtest(
                    strategy_name,
                    start=w["test_start"],
                    end=w["test_end"],
                    n_hold=n_hold,
                    max_weight=max_weight,
                    signal_to_return_lag=signal_to_return_lag,
                    rebalance_every=rebalance_every,
                    respect_regime=respect_regime,
                )
                if daily.empty:
                    continue

                metrics = engine.calculate_metrics(
                    daily["return"],
                    daily["turnover"],
                    daily["transaction_cost"],
                    daily["gross_return"],
                    daily["exposure"],
                    daily.get("cash_weight", None),
                    dates=pd.to_datetime(daily["date"]),
                )

                # Compute B0 benchmark for excess return
                bench_daily = engine.run_backtest(
                    "B0_BuyHold_EW",
                    start=w["test_start"],
                    end=w["test_end"],
                    n_hold=n_hold,
                    max_weight=max_weight,
                    signal_to_return_lag=signal_to_return_lag,
                    rebalance_every=rebalance_every,
                    respect_regime=False,
                )
                if not bench_daily.empty and strategy_name != "B0_BuyHold_EW":
                    bench_metrics = engine.calculate_metrics(
                        daily["return"],
                        benchmark_returns=bench_daily["return"],
                        dates=pd.to_datetime(daily["date"]),
                    )
                    metrics.update(bench_metrics)

                row = {
                    "window_id": window_id,
                    "train_start": w["train_start"],
                    "train_end": w["train_end"],
                    "val_start": w["val_start"],
                    "val_end": w["val_end"],
                    "test_start": w["test_start"],
                    "test_end": w["test_end"],
                    "strategy": strategy_name,
                }
                row.update(metrics)
                rows.append(row)
            except Exception as e:
                print(f"  Error: window={window_id} strategy={strategy_name}: {e}")
                continue

    result = pd.DataFrame(rows)
    return result


def run_walk_forward_with_benchmarks(
    strategy_names: list[str],
    start_date: str = "2018-01-01",
    end_date: str = "2026-07-17",
    train_years: int = 3,
    val_years: int = 1,
    test_years: int = 1,
    roll_months: int = 6,
    **kwargs,
) -> pd.DataFrame:
    """Run walk-forward including B0 benchmark in strategy list."""
    all_names = list(set(strategy_names + ["B0_BuyHold_EW"]))
    return run_walk_forward(
        all_names,
        start_date=start_date,
        end_date=end_date,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        roll_months=roll_months,
        **kwargs,
    )


def summarize_walk_forward(
    df: pd.DataFrame,
    strategies: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize walk-forward results by strategy (averaged across windows)."""
    if df.empty:
        return pd.DataFrame()

    strat_list = strategies if strategies else df["strategy"].unique().tolist()
    summary_rows: list[dict] = []

    for s in strat_list:
        subset = df[df["strategy"] == s]
        if subset.empty:
            continue

        row = {"strategy": s, "n_windows": len(subset)}

        # Average numeric metrics across windows
        numeric_cols = subset.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ("window_id",):
                continue
            row[f"{col}_mean"] = float(subset[col].mean())
            row[f"{col}_std"] = float(subset[col].std())
            row[f"{col}_min"] = float(subset[col].min())
            row[f"{col}_max"] = float(subset[col].max())

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


if __name__ == "__main__":
    import json

    strategies = [
        "S04_VolTarget_Trend",
        "S01_CS_Momentum",
        "S12_Low_Volatility",
        "S13_Quality",
        "B0_BuyHold_EW",
        "B1_Momentum_Agnostic",
        "B2_Cash",
        "B3_Index_Proxy",
    ]

    print("=" * 80)
    print("Phase 3: Walk-Forward Research Framework")
    print("=" * 80)

    results = run_walk_forward(
        strategy_names=strategies,
        start_date="2018-01-01",
        end_date="2026-07-17",
        train_years=3,
        val_years=1,
        test_years=1,
        roll_months=6,
    )

    if not results.empty:
        output_dir = Path("reports/strategy_research")
        output_dir.mkdir(parents=True, exist_ok=True)

        results.to_csv(output_dir / "walk_forward_results.csv", index=False)
        print(f"\nWalk-forward results saved to {output_dir / 'walk_forward_results.csv'}")

        summary = summarize_walk_forward(results, strategies)
        if not summary.empty:
            summary.to_csv(output_dir / "walk_forward_summary.csv", index=False)
            print("\n--- Strategy Ranking (by average Sharpe) ---")
            sharpe_col = "Sharpe_mean"
            if sharpe_col in summary.columns:
                ranked = summary.sort_values(sharpe_col, ascending=False)
                display_cols = [
                    c for c in [
                        "strategy", "n_windows",
                        "CAGR%_mean", "Sharpe_mean", "Sortino_mean",
                        "Max_Drawdown%_mean", "Calmar_mean",
                        "Exposure%_mean", "Turnover%_mean",
                    ]
                    if c in ranked.columns
                ]
                print(ranked[display_cols].to_string(index=False))

        # Full phase 3 report
        full_report = output_dir / "phase3_full_report.csv"
        results.to_csv(full_report, index=False)
        print(f"\nFull Phase 3 report saved to {full_report}")
