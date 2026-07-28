"""Phase 5.2: S04 Signal Decomposition Analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest_engine import BacktestEngine
from strategy_library import Strategy


class S04_TrendOnly(Strategy):
    """S04 with only trend filter (no vol targeting)."""
    name = "S04_Trend_Only"
    regime = 0
    description = "Trend filter only: mom_20 * (ma_dist > 0), equal weight"
    literature = "Decomposition"
    positive_only = True

    def __init__(self, factor_df: pd.DataFrame, *, trend_lookback: int = 60):
        super().__init__(factor_df)
        self._trend_col = f"ma_dist_{trend_lookback}"

    def compute_signal(self, date: pd.Timestamp) -> pd.Series:
        day_data = self._get_day_data(date)
        if len(day_data) < max(20, getattr(self, '_trend_lookback', 60)):
            return pd.Series(dtype=float)

        ma_dist = day_data.set_index("symbol")[self._trend_col]
        trend_filter = (ma_dist > 0).astype(float)
        mom = day_data.set_index("symbol")["mom_20"]

        signal = mom * trend_filter
        return signal


class S04_VolTargetOnly(Strategy):
    """S04 with only vol targeting (no trend filter)."""
    name = "S04_VolTarget_Only"
    regime = 0
    description = "Vol targeting only: mom_20 / real_vol, no trend filter"
    literature = "Decomposition"
    positive_only = True

    def __init__(self, factor_df: pd.DataFrame, *, vol_lookback: int = 10):
        super().__init__(factor_df)
        self._vol_col = f"real_vol_{vol_lookback}"

    def compute_signal(self, date: pd.Timestamp) -> pd.Series:
        day_data = self._get_day_data(date)
        if len(day_data) < 20:
            return pd.Series(dtype=float)

        mom = day_data.set_index("symbol")["mom_20"]
        vol = day_data.set_index("symbol")[self._vol_col]
        vol_safe = vol.clip(lower=1e-8)

        signal = mom / vol_safe
        return signal


def run_decomposition(
    engine: BacktestEngine,
    strategy_name: str,
    start: str,
    end: str,
    n_hold: int = 20,
    max_weight: float = 0.05,
    rebalance_every: int = 5,
) -> dict | None:
    """Run a backtest using engine's standard method."""
    daily = engine.run_backtest(
        strategy_name, start=start, end=end,
        n_hold=n_hold, max_weight=max_weight,
        signal_to_return_lag=2, rebalance_every=rebalance_every,
        respect_regime=False,
    )
    if daily.empty:
        return None

    metrics = engine.calculate_metrics(
        daily["return"],
        daily["turnover"],
        daily["transaction_cost"],
        daily["gross_return"],
        daily["exposure"],
        daily.get("cash_weight", None),
        dates=pd.to_datetime(daily["date"]),
    )

    # B0 benchmark for excess
    bench_daily = engine.run_backtest(
        "B0_BuyHold_EW", start=start, end=end,
        n_hold=n_hold, max_weight=max_weight,
        signal_to_return_lag=2, rebalance_every=rebalance_every,
        respect_regime=False,
    )
    if not bench_daily.empty:
        bench_metrics = engine.calculate_metrics(
            daily["return"],
            benchmark_returns=bench_daily["return"],
            dates=pd.to_datetime(daily["date"]),
        )
        metrics.update(bench_metrics)

    return metrics


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


def main():
    print("=" * 80)
    print("Phase 5.2: S04 Signal Decomposition Analysis")
    print("=" * 80)

    engine = BacktestEngine()
    windows = expand_dates(
        "2018-01-01", "2026-07-17", engine.dates,
        train_years=3, val_years=1, test_years=1, roll_months=6,
    )
    print(f"\nWalk-forward windows: {len(windows)}")

    # Components to test
    components = [
        ("S04_Trend_Only", "Trend filter only (no vol targeting)"),
        ("S04_VolTarget_Only", "Vol targeting only (no trend filter)"),
        ("S04_VolTarget_Trend", "Full S04 (trend + vol targeting)"),
        ("B0_BuyHold_EW", "Buy and hold benchmark"),
    ]

    # Register decomposition strategies
    from strategy_library import STRATEGIES
    STRATEGIES["S04_Trend_Only"] = S04_TrendOnly
    STRATEGIES["S04_VolTarget_Only"] = S04_VolTargetOnly

    # Initialize strategies with engine factors
    strat_instances = {
        "S04_Trend_Only": S04_TrendOnly(engine.factors, trend_lookback=60),
        "S04_VolTarget_Only": S04_VolTargetOnly(engine.factors, vol_lookback=10),
    }

    rows: list[dict] = []
    for comp_name, desc in components:
        print(f"\n--- Testing: {comp_name} ({desc}) ---")
        for wi, w in enumerate(windows):
            if comp_name in ("S04_Trend_Only", "S04_VolTarget_Only"):
                # Run custom backtest for decomposition strategies
                strat = strat_instances[comp_name]
                daily_rows = []
                live_weights = {}
                last_rebal_idx = None

                test_dates = engine.dates[(engine.dates >= w["test_start"]) & (engine.dates <= w["test_end"])]
                for date in test_dates:
                    full_idx = engine._date_to_index[date]
                    signal_idx = full_idx - 2
                    signal_date = engine.dates[signal_idx] if signal_idx >= 0 else pd.NaT

                    should_rebal = (
                        signal_idx >= 0
                        and (last_rebal_idx is None or full_idx - last_rebal_idx >= 5)
                    )

                    target_w = dict(live_weights)
                    if should_rebal:
                        target_w = strat.get_positions(signal_date, n_hold=20, max_weight=0.05)
                        last_rebal_idx = full_idx

                    traded = BacktestEngine._gross_traded_notional(live_weights, target_w)
                    tc = traded * (engine.fee_rate_per_side + engine.slippage_rate_per_side)
                    daily_rets = engine.rets_wide.loc[date]
                    exposure = float(sum(target_w.values()))
                    cw = max(0.0, 1.0 - exposure)

                    gross_ret = float(
                        sum(wt * float(daily_rets.get(s, 0.0)) for s, wt in target_w.items())
                        + cw * engine.cash_daily_return
                    )
                    net_ret = gross_ret - tc
                    live_weights = BacktestEngine._drift_weights(target_w, daily_rets, gross_ret)

                    daily_rows.append({
                        "date": date, "return": net_ret, "gross_return": gross_ret,
                        "transaction_cost": tc, "turnover": traded / 2.0,
                        "exposure": exposure, "cash_weight": cw,
                    })

                daily = pd.DataFrame(daily_rows)
                if daily.empty:
                    continue

                metrics = engine.calculate_metrics(
                    daily["return"], daily["turnover"], daily["transaction_cost"],
                    daily["gross_return"], daily["exposure"], daily.get("cash_weight", None),
                    dates=pd.to_datetime(daily["date"]),
                )

                # B0 benchmark
                bench = engine.run_backtest(
                    "B0_BuyHold_EW", start=w["test_start"], end=w["test_end"],
                    n_hold=20, max_weight=0.05, signal_to_return_lag=2,
                    rebalance_every=5, respect_regime=False,
                )
                if not bench.empty and comp_name != "B0_BuyHold_EW":
                    bm = engine.calculate_metrics(
                        daily["return"], benchmark_returns=bench["return"],
                        dates=pd.to_datetime(daily["date"]),
                    )
                    metrics.update(bm)

            else:
                metrics = run_decomposition(
                    engine, comp_name,
                    start=w["test_start"], end=w["test_end"],
                    n_hold=20, max_weight=0.05, rebalance_every=5,
                )
                if not metrics:
                    continue

            row = {
                "window_id": wi,
                "component": comp_name,
                "description": desc,
            }
            for k in ["CAGR%", "Sharpe", "Sortino", "Max_Drawdown%",
                       "Calmar", "Exposure%", "Turnover%", "Transaction_Cost_Drag%"]:
                row[k] = metrics.get(k, np.nan)
            row["excess_vs_b0%"] = metrics.get("Excess_Return%", np.nan)
            rows.append(row)

    result_df = pd.DataFrame(rows)
    out_dir = Path("reports/strategy_research")
    out_path = out_dir / "s04_component_decomposition.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # Average across windows
    if not result_df.empty:
        avg = result_df.groupby("component")[["CAGR%", "Sharpe", "Max_Drawdown%", "excess_vs_b0%"]].mean()
        print("\n--- Average Metrics by Component ---")
        print(avg.to_string())

        # Determine which component contributes most
        trend_row = avg.loc["S04_Trend_Only"] if "S04_Trend_Only" in avg.index else None
        vol_row = avg.loc["S04_VolTarget_Only"] if "S04_VolTarget_Only" in avg.index else None
        full_row = avg.loc["S04_VolTarget_Trend"] if "S04_VolTarget_Trend" in avg.index else None

        if trend_row is not None and vol_row is not None:
            trend_sharpe = trend_row["Sharpe"]
            vol_sharpe = vol_row["Sharpe"]
            full_sharpe = full_row["Sharpe"] if full_row is not None else np.nan
            print(f"\n--- Decomposition Conclusion ---")
            print(f"Trend only Sharpe: {trend_sharpe:.4f}")
            print(f"Vol target only Sharpe: {vol_sharpe:.4f}")
            print(f"Full S04 Sharpe: {full_sharpe:.4f}")

            best_single = max(trend_sharpe, vol_sharpe)
            if full_sharpe > best_single + 0.05:
                print("Synergy: Combined meaningfully better than best single component")
            elif full_sharpe < best_single - 0.05:
                dominant = "Trend" if trend_sharpe >= vol_sharpe else "Vol targeting"
                print(f"Dominant component: {dominant} (full combo underperforms due to interaction effects)")
            else:
                dominant = "Trend" if trend_sharpe >= vol_sharpe else "Vol targeting"
                print(f"Primary contributor: {dominant} (similar performance across variants)")

    return result_df


if __name__ == "__main__":
    main()
