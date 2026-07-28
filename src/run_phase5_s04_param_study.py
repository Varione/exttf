"""Phase 5.1: S04 VolTargetTrend parameter sensitivity analysis (walk-forward)."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest_engine import BacktestEngine
from strategy_library import S04_VolTargetTrend, Strategy


class S04_Configurable(S04_VolTargetTrend):
    """S04 with configurable trend_lookback, vol_lookback, target_vol_annual."""

    name = "S04_Config"

    def __init__(self, factor_df: pd.DataFrame, *,
                 trend_lookback: int = 60,
                 vol_lookback: int = 10,
                 target_vol_annual: float = 0.10):
        super().__init__(factor_df)
        self._trend_lookback = trend_lookback
        self._vol_lookback = vol_lookback
        self._trend_col = f"ma_dist_{trend_lookback}"
        self._vol_col = f"real_vol_{vol_lookback}"
        self._target_vol_annual = target_vol_annual

    def compute_signal(self, date: pd.Timestamp) -> pd.Series:
        day_data = self._get_day_data(date)
        min_rows = max(20, self._trend_lookback)
        if len(day_data) < min_rows:
            return pd.Series(dtype=float)

        ma_dist = day_data.set_index("symbol")[self._trend_col]
        trend_filter = (ma_dist > 0).astype(float)

        mom = day_data.set_index("symbol")["mom_20"]
        vol = day_data.set_index("symbol")[self._vol_col]
        vol_safe = vol.clip(lower=1e-8)

        signal = (mom / vol_safe) * trend_filter
        return signal

    def get_positions(self, date: pd.Timestamp, n_hold: int = 20,
                      max_weight: float = 0.05,
                      target_vol_annual: float | None = None) -> dict[str, float]:
        tv = target_vol_annual if target_vol_annual is not None else self._target_vol_annual
        signals = self.compute_signal(date)
        signals = signals.replace([np.inf, -np.inf], np.nan).dropna()
        signals = signals[signals > 0].sort_values(ascending=False)
        if signals.empty or n_hold <= 0:
            return {}

        top_n = signals.head(min(n_hold, len(signals)))
        day_data = self._get_day_data(date)
        if len(day_data) == 0:
            return {}

        vol_series = day_data.set_index("symbol")[self._vol_col]
        selected_vols = [vol_series[sym] for sym in top_n.index if sym in vol_series.index]
        if not selected_vols:
            return {}

        n = len(selected_vols)
        rho = 0.3
        port_vol_daily = self._estimate_portfolio_vol_daily(selected_vols, rho=rho)
        if not np.isfinite(port_vol_daily) or port_vol_daily <= 0:
            return {}

        target_vol_daily = tv / np.sqrt(252)
        scale = min(1.0, target_vol_daily / port_vol_daily)
        weight = min(max_weight, scale / n)
        return {sym: weight for sym in top_n.index}


def run_single_backtest(
    engine: BacktestEngine,
    trend_lb: int,
    vol_lb: int,
    target_vol: float,
    rebalance_every: int,
    start: str,
    end: str,
    b0_returns: pd.Series | None = None,
    n_hold: int = 20,
    max_weight: float = 0.05,
) -> dict | None:
    """Run a single S04 backtest with given parameters."""
    strat = S04_Configurable(
        engine.factors,
        trend_lookback=trend_lb,
        vol_lookback=vol_lb,
        target_vol_annual=target_vol,
    )

    test_dates = engine.dates[(engine.dates >= start) & (engine.dates <= end)]
    if test_dates.empty:
        return None

    rows: list[dict] = []
    live_weights: dict[str, float] = {}
    last_rebalance_index: int | None = None

    for date in test_dates:
        full_idx = engine._date_to_index[date]
        signal_idx = full_idx - 2  # signal_to_return_lag=2
        signal_date = engine.dates[signal_idx] if signal_idx >= 0 else pd.NaT

        should_rebalance = (
            signal_idx >= 0
            and (last_rebalance_index is None or full_idx - last_rebalance_index >= rebalance_every)
        )

        target_weights = dict(live_weights)
        if should_rebalance:
            target_weights = strat.get_positions(
                signal_date, n_hold=n_hold, max_weight=max_weight
            )
            last_rebalance_index = full_idx

        traded_notional = BacktestEngine._gross_traded_notional(live_weights, target_weights)
        transaction_cost = traded_notional * (engine.fee_rate_per_side + engine.slippage_rate_per_side)

        daily_rets = engine.rets_wide.loc[date]
        exposure = float(sum(target_weights.values()))
        cash_weight = max(0.0, 1.0 - exposure)

        gross_return = float(
            sum(w * float(daily_rets.get(s, 0.0)) for s, w in target_weights.items())
            + cash_weight * engine.cash_daily_return
        )
        net_return = gross_return - transaction_cost

        live_weights = BacktestEngine._drift_weights(target_weights, daily_rets, gross_return)

        rows.append({
            "date": date,
            "return": net_return,
            "gross_return": gross_return,
            "transaction_cost": transaction_cost,
            "turnover": traded_notional / 2.0,
            "exposure": exposure,
            "cash_weight": cash_weight,
        })

    daily = pd.DataFrame(rows)
    if daily.empty:
        return None

    daily_dates = pd.to_datetime(daily["date"])
    daily_indexed = daily.set_index(daily_dates)

    metrics = engine.calculate_metrics(
        daily_indexed["return"],
        daily_indexed["turnover"],
        daily_indexed["transaction_cost"],
        daily_indexed["gross_return"],
        daily_indexed["exposure"],
        daily_indexed.get("cash_weight", None),
        dates=daily_dates,
    )

    # B0 benchmark for excess (use cached returns if provided)
    if b0_returns is not None and not b0_returns.empty:
        bench_metrics = engine.calculate_metrics(
            daily_indexed["return"],
            benchmark_returns=b0_returns,
            dates=daily_dates,
        )
        metrics.update(bench_metrics)

    return metrics


def expand_dates(
    start_date: str, end_date: str, all_dates: pd.DatetimeIndex,
    train_years: int = 3, val_years: int = 1, test_years: int = 1, roll_months: int = 6,
) -> list[dict[str, str]]:
    """Generate walk-forward windows (copied from walk_forward.py)."""
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
    print("Phase 5.1: S04 Parameter Sensitivity Analysis")
    print("=" * 80)

    engine = BacktestEngine()
    windows = expand_dates(
        "2018-01-01", "2026-07-17", engine.dates,
        train_years=3, val_years=1, test_years=1, roll_months=6,
    )
    print(f"\nWalk-forward windows: {len(windows)}")
    for i, w in enumerate(windows):
        print(f"  W{i}: test={w['test_start']}~{w['test_end']}")

    # Parameter grid
    param_grid = {
        "trend_lb": [20, 60, 120],
        "vol_lb": [10, 20, 60],
        "target_vol": [0.05, 0.10, 0.15],
        "rebalance": [3, 5, 10],
    }
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    print(f"\nParameter combinations: {len(combos)}")

    # Pre-compute B0 benchmark returns for all periods
    print("\n--- Precomputing B0 benchmark returns ---")
    b0_cache: dict[str, pd.Series] = {}

    # B0 for train+val period
    tv_start = windows[0]["train_start"]
    tv_end = windows[min(1, len(windows) - 1)]["val_end"] if len(windows) > 1 else windows[0]["val_end"]
    b0_tv_daily = engine.run_backtest(
        "B0_BuyHold_EW", start=tv_start, end=tv_end,
        n_hold=20, max_weight=0.05, signal_to_return_lag=2,
        rebalance_every=5, respect_regime=False,
    )
    if not b0_tv_daily.empty:
        b0_cache["tv"] = b0_tv_daily.set_index(pd.to_datetime(b0_tv_daily["date"]))["return"]

    # B0 for each test window
    for wi, w in enumerate(windows):
        b0_test_daily = engine.run_backtest(
            "B0_BuyHold_EW", start=w["test_start"], end=w["test_end"],
            n_hold=20, max_weight=0.05, signal_to_return_lag=2,
            rebalance_every=5, respect_regime=False,
        )
        if not b0_test_daily.empty:
            b0_cache[f"test_{wi}"] = b0_test_daily.set_index(pd.to_datetime(b0_test_daily["date"]))["return"]

    # Step 1: Run each combo on train+val for parameter selection
    print("\n--- Step 1: Parameter selection on train+val ---")
    tv_sharpe = {}  # combo_idx -> avg Sharpe on train+val
    for ci, vals in enumerate(combos):
        p = dict(zip(keys, vals))
        try:
            metrics = run_single_backtest(
                engine, p["trend_lb"], p["vol_lb"], p["target_vol"], p["rebalance"],
                start=tv_start, end=tv_end,
                b0_returns=b0_cache.get("tv"),
            )
            if metrics and "Sharpe" in metrics:
                tv_sharpe[ci] = metrics["Sharpe"]
            else:
                tv_sharpe[ci] = -999.0
        except Exception as e:
            print(f"    Combo {ci}: error {e}")
            tv_sharpe[ci] = -999.0

        if (ci + 1) % 20 == 0:
            print(f"    Evaluated {ci + 1}/{len(combos)} combos on train+val")

    # Select best combo on train+val
    best_ci = max(tv_sharpe, key=tv_sharpe.get)  # type: ignore[arg-type]
    best_vals = combos[best_ci]
    best_params = dict(zip(keys, best_vals))
    print(f"\nBest params (train+val Sharpe={tv_sharpe[best_ci]:.4f}): {best_params}")

    # Step 2: Run ALL combos on test windows (single read)
    print("\n--- Step 2: Running all combos on OOS test windows ---")
    rows: list[dict] = []
    for ci, vals in enumerate(combos):
        p = dict(zip(keys, vals))
        for wi, w in enumerate(windows):
            try:
                b0_key = f"test_{wi}"
                metrics = run_single_backtest(
                    engine, p["trend_lb"], p["vol_lb"], p["target_vol"], p["rebalance"],
                    start=w["test_start"], end=w["test_end"],
                    b0_returns=b0_cache.get(b0_key),
                )
                if not metrics:
                    continue

                row = {
                    "window_id": wi,
                    "trend_lb": p["trend_lb"],
                    "vol_lb": p["vol_lb"],
                    "target_vol": p["target_vol"],
                    "rebalance": p["rebalance"],
                    "n_hold": 20,
                }

                # Key metrics
                for k in ["CAGR%", "Sharpe", "Max_Drawdown%"]:
                    row[f"{k}"] = metrics.get(k, np.nan)

                # Excess vs B0
                row["excess_vs_b0%"] = metrics.get("Excess_Return%", np.nan)
                rows.append(row)
            except Exception as e:
                print(f"    Error combo={ci} window={wi}: {e}")

        if (ci + 1) % 10 == 0:
            print(f"    Combo {ci + 1}/{len(combos)} complete")

    result_df = pd.DataFrame(rows)
    out_dir = Path("reports/strategy_research")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "s04_parameter_study.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # Show top 10 by Sharpe
    if "Sharpe" in result_df.columns:
        ranked = result_df.sort_values("Sharpe", ascending=False).head(10)
        print("\n--- Top 10 windows by Sharpe ---")
        display_cols = ["window_id", "trend_lb", "vol_lb", "target_vol", "rebalance",
                        "CAGR%", "Sharpe", "Max_Drawdown%", "excess_vs_b0%"]
        avail = [c for c in display_cols if c in ranked.columns]
        print(ranked[avail].to_string(index=False))

    # Average across windows per combo
    print("\n--- Average metrics per parameter combo (across test windows) ---")
    avg_cols = ["trend_lb", "vol_lb", "target_vol", "rebalance"]
    metric_cols = ["CAGR%", "Sharpe", "Max_Drawdown%", "excess_vs_b0%"]
    avail_metrics = [c for c in metric_cols if c in result_df.columns]

    if avail_metrics:
        grouped = result_df.groupby(avg_cols)[avail_metrics].mean().reset_index()
        grouped = grouped.sort_values("Sharpe", ascending=False)
        print(grouped.to_string(index=False))

        # Save best combo info
        best_row = grouped.iloc[0]
        print(f"\nBest OOS combo: {best_row.to_dict()}")

    return result_df, best_params


if __name__ == "__main__":
    main()
