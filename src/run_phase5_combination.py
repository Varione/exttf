"""Phase 5.3: Strategy Combination Walk-Forward Backtest."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest_engine import BacktestEngine
from strategy_combiner import (
    EqualRiskContribution,
    VolatilityScaling,
    RegimeConditionedWeights,
    combine_strategy_returns,
)


def expand_dates(
    start_date: str, end_date: str, all_dates: pd.DatetimeIndex,
    train_years: int = 3, val_years: int = 1, test_years: int = 1, roll_months: int = 6,
) -> list[dict[str, str]]:
    from datetime import timedelta

    def find_date_approx(dates, anchor, years):
        target = anchor + timedelta(days=int(years * 365.25))
        diffs = np.abs((dates - target).days)
        idx = int(diffs.argmin())
        if dates[idx] < anchor:
            return None
        return idx

    windows = []
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    min_test_days = int(0.5 * 365.25)

    while True:
        train_start = current
        train_end_idx = find_date_approx(all_dates, train_start, train_years)
        if train_end_idx is None or train_end_idx < 1:
            break
        train_end = all_dates[train_end_idx]

        val_end_idx = find_date_approx(all_dates, train_end, val_years)
        if val_end_idx is None or val_end_idx <= train_end_idx:
            break
        val_start_idx = max(train_end_idx + 1, 0)
        val_start = all_dates[val_start_idx]
        val_end = all_dates[val_end_idx]

        test_end_idx = find_date_approx(all_dates, val_end, test_years)
        if test_end_idx is None or test_end_idx <= val_end_idx:
            break
        test_start_idx = val_end_idx + 1
        if test_start_idx >= len(all_dates):
            break

        raw_test_end = all_dates[min(test_end_idx, len(all_dates) - 1)]
        test_end_actual = min(raw_test_end, end)

        test_span_days = (test_end_actual - all_dates[test_start_idx]).days
        if test_span_days < min_test_days:
            break

        test_start = all_dates[test_start_idx]

        windows.append({
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "val_start": val_start.strftime("%Y-%m-%d"),
            "val_end": val_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": test_end_actual.strftime("%Y-%m-%d"),
        })

        if test_end_actual >= end:
            break

        roll_days = int(roll_months * 30.44)
        current = test_end_actual + timedelta(days=roll_days)
        if current > end:
            break

    return windows


def get_strategy_daily_returns(
    engine: BacktestEngine,
    strategy_name: str,
    start: str,
    end: str,
) -> pd.Series:
    """Get daily returns for a strategy."""
    daily = engine.run_backtest(
        strategy_name, start=start, end=end,
        n_hold=20, max_weight=0.05,
        signal_to_return_lag=2, rebalance_every=5,
        respect_regime=False,
    )
    if daily.empty:
        return pd.Series(dtype=float)
    return daily.set_index(pd.to_datetime(daily["date"]))["return"]


def main():
    print("=" * 80)
    print("Phase 5.3: Strategy Combination Research")
    print("=" * 80)

    engine = BacktestEngine()
    windows = expand_dates(
        "2018-01-01", "2026-07-17", engine.dates,
        train_years=3, val_years=1, test_years=1, roll_months=6,
    )
    print(f"\nWalk-forward windows: {len(windows)}")

    # Strategies to combine
    strat_names = ["S04_VolTarget_Trend", "S01_CS_Momentum", "S23_Oversold_Long"]

    # Combination methods
    combiners = {
        "ERC": EqualRiskContribution(),
        "VolScaling": VolatilityScaling(),
        "RegimeConditioned": RegimeConditionedWeights(),
    }

    rows: list[dict] = []

    for method_name, combiner in combiners.items():
        print(f"\n--- Method: {method_name} ---")

        for wi, w in enumerate(windows):
            # Get individual strategy returns for train+val period (for weight estimation)
            tv_returns = {}
            for sname in strat_names:
                ret = get_strategy_daily_returns(
                    engine, sname,
                    start=w["train_start"], end=w["val_end"],
                )
                if not ret.empty:
                    tv_returns[sname] = ret

            if len(tv_returns) < 2:
                print(f"  Window {wi}: insufficient strategy returns on train+val")
                continue

            # Align to common index
            common_idx = list(tv_returns.values())[0].index
            for sname, ret in tv_returns.items():
                common_idx = common_idx.intersection(ret.index)
            tv_aligned = pd.DataFrame({s: r.loc[common_idx] for s, r in tv_returns.items()})

            # Compute weights on train+val
            if method_name == "RegimeConditioned":
                regime_df = engine.regime_preds
                regime_probs = regime_df.set_index("date").loc[common_idx] if "date" in engine.regime_preds.columns else None
                if regime_probs is not None and len(regime_probs) > 0:
                    weights = combiner.get_weights(tv_aligned, regime_probs)
                else:
                    weights = {s: 1.0 / len(strat_names) for s in strat_names}
            else:
                weights = combiner.get_weights(tv_aligned)

            print(f"  Window {wi}: weights={{{', '.join(f'{k}={v:.3f}' for k, v in weights.items())}}}")

            # Get test period returns and combine
            test_returns = {}
            for sname in strat_names:
                ret = get_strategy_daily_returns(
                    engine, sname,
                    start=w["test_start"], end=w["test_end"],
                )
                if not ret.empty:
                    test_returns[sname] = ret

            if len(test_returns) < 2:
                continue

            # Combine returns
            combined_ret = combine_strategy_returns(test_returns, weights)
            if combined_ret.empty:
                continue

            # Calculate metrics for combined strategy
            metrics = engine.calculate_metrics(
                combined_ret,
                dates=combined_ret.index,
            )

            # B0 benchmark
            bench_ret = get_strategy_daily_returns(
                engine, "B0_BuyHold_EW",
                start=w["test_start"], end=w["test_end"],
            )
            if not bench_ret.empty:
                common_test = combined_ret.index.intersection(bench_ret.index)
                if len(common_test) > 10:
                    bm = engine.calculate_metrics(
                        combined_ret.loc[common_test],
                        benchmark_returns=bench_ret.loc[common_test],
                        dates=common_test,
                    )
                    metrics.update(bm)

            row = {
                "window_id": wi,
                "method": method_name,
                "weights": str(weights),
            }
            for k in ["CAGR%", "Sharpe", "Sortino", "Max_Drawdown%",
                       "Calmar", "Annualized_Volatility%"]:
                row[k] = metrics.get(k, np.nan)
            row["excess_vs_b0%"] = metrics.get("Excess_Return%", np.nan)
            row["n_strategies"] = len(weights)
            rows.append(row)

    result_df = pd.DataFrame(rows)
    out_dir = Path("reports/strategy_research")
    out_path = out_dir / "strategy_combination_results.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # Average across windows per method
    if not result_df.empty:
        avg = result_df.groupby("method")[["CAGR%", "Sharpe", "Max_Drawdown%", "excess_vs_b0%"]].mean()
        print("\n--- Average Metrics by Combination Method ---")
        print(avg.to_string())

        # Compare with individual strategies
        print("\n--- Individual Strategy Baseline (from walk_forward_summary.csv) ---")
        try:
            wf_summary = pd.read_csv(out_dir / "walk_forward_summary.csv")
            for sname in strat_names + ["B0_BuyHold_EW"]:
                row = wf_summary[wf_summary["strategy"] == sname]
                if not row.empty:
                    sharpe = row["Sharpe_mean"].values[0] if "Sharpe_mean" in row.columns else np.nan
                    cagr = row["CAGR%_mean"].values[0] if "CAGR%_mean" in row.columns else np.nan
                    print(f"  {sname}: CAGR={cagr:.2f}%, Sharpe={sharpe:.4f}")
        except Exception as e:
            print(f"  Could not load walk_forward_summary.csv: {e}")

    return result_df


if __name__ == "__main__":
    main()
