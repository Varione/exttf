"""B7: 低频场外轮动策略 walk-forward 研究

实现月频多周期动量 + 趋势过滤 + 债券避险的 OTF 策略。
使用滚动 walk-forward 验证 OOS 表现。
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from otf_backtest_engine import OTFBacktestEngine, OTFStrategySignal
from otf_trading_rules import ProductRuleBook

OUTPUT_DIR = Path("reports/strategy_research/otf_walkforward")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def multi_period_momentum(
    returns_wide: pd.DataFrame,
    date: pd.Timestamp,
    lookbacks: list[int] = None,
    trend_filter_days: int = 200,
    n_hold: int = 5,
) -> dict[str, float]:
    """Multi-period momentum with trend filter.

    Scores each fund by average rank across multiple lookback windows.
    Only selects from funds above their 200-day trend filter.
    """
    if lookbacks is None:
        lookbacks = [60, 120, 252]

    if date not in returns_wide.index:
        return {}

    idx = returns_wide.index.get_loc(date)
    min_lookback = max(lookbacks)
    if idx < min_lookback:
        return {}

    # Trend filter: only funds above 200-day SMA
    trend_window = returns_wide.iloc[idx - trend_filter_days : idx]
    if len(trend_window) < trend_filter_days // 2:
        return {}
    trend_series = (1.0 + trend_window).prod()

    # Multi-period momentum
    scores = pd.Series(0.0, index=returns_wide.columns)
    for lb in lookbacks:
        if idx < lb:
            continue
        window = returns_wide.iloc[idx - lb : idx]
        cum_ret = (1.0 + window).prod() - 1.0
        # Rank across funds (lower rank = better)
        ranks = cum_ret.rank(ascending=False, na_option="bottom")
        scores += ranks

    # Average rank
    scores = scores / len(lookbacks)

    # Apply trend filter
    above_trend = trend_series > 1.0
    scores = scores[above_trend].dropna()

    if len(scores) == 0:
        return {}

    # Select top N
    selected = scores.nsmallest(n_hold)
    weight = 1.0 / len(selected)
    return {sym: weight for sym in selected.index}


def trend_filtered_bond_bailout(
    date: pd.Timestamp,
    equity_momentum_signal: dict[str, float],
    bond_funds: list[str],
    short_bond_funds: list[str],
    equity_trend_ok: bool,
) -> dict[str, float]:
    """If equity trend fails, bail out to bonds/short bonds."""
    if equity_trend_ok and equity_momentum_signal:
        return equity_momentum_signal

    # Allocate to short bonds + bonds
    result = {}
    candidates = short_bond_funds + bond_funds
    for f in candidates:
        result[f] = 1.0 / len(candidates) if candidates else 0.0
    return result


def evaluate_oos(results: list[dict]) -> pd.DataFrame:
    """Aggregate walk-forward OOS results into performance summary."""
    df = pd.DataFrame(results)
    if df.empty:
        return df

    combined_returns = pd.Series(dtype=float)
    for r in results:
        rets = r.get("daily_returns", pd.Series(dtype=float))
        combined_returns = pd.concat([combined_returns, rets])

    if combined_returns.empty:
        return df

    wealth = (1.0 + combined_returns).cumprod()
    total_ret = wealth.iloc[-1] - 1.0
    n = len(combined_returns)
    ann_ret = wealth.iloc[-1] ** (252.0 / n) - 1.0 if n > 0 else 0.0
    ann_std = combined_returns.std() * np.sqrt(252) if n > 1 else 0.0
    sharpe = ann_ret / ann_std if ann_std > 0 else 0.0
    dd = wealth / wealth.cummax() - 1.0
    mdd = dd.min()

    metrics = {
        "OOS_Total_Return_pct": round(float(total_ret) * 100, 2),
        "OOS_CAGR_pct": round(float(ann_ret) * 100, 2),
        "OOS_Sharpe": round(float(sharpe), 3),
        "OOS_Max_Drawdown_pct": round(float(mdd) * 100, 2),
        "OOS_Volatility_pct": round(float(ann_std) * 100, 2),
        "OOS_Calmar": round(float(ann_ret / abs(mdd)), 3) if mdd < 0 else 0.0,
        "OOS_n_days": n,
        "OOS_Win_Rate_pct": round(float((combined_returns > 0).mean()) * 100, 1),
    }
    return pd.Series(metrics)


def main():
    print("=" * 60)
    print("B7: 低频场外轮动策略 walk-forward 研究")
    print("=" * 60)

    # Load product rules
    rule_book = ProductRuleBook.from_csv()
    print(f"Product rules loaded: {len(rule_book.rules)} funds")

    engine = OTFBacktestEngine(
        db_path="data/processed/otf.sqlite",
        product_rule_book=rule_book,
        confirmation_days_subscribe=1,
        confirmation_days_redeem=1,
        settlement_days_redeem=7,
        subscription_fee_rate=0.001,
        redemption_fee_rate=0.0015,
        initial_cash=1_000_000.0,
    )

    # Identify fund categories
    catalog = engine._catalog_df
    all_funds = engine.available_fund_codes
    print(f"Available funds: {len(all_funds)}")

    equity_funds = [f for f in all_funds if catalog.loc[catalog["fund_code"] == f, "asset_class"].values[0]
                    in ("宽基股票", "红利", "行业")]
    bond_funds = [f for f in all_funds if catalog.loc[catalog["fund_code"] == f, "asset_class"].values[0]
                  in ("债券",)]
    short_bond_funds = [f for f in all_funds if catalog.loc[catalog["fund_code"] == f, "asset_class"].values[0]
                       in ("债券短久期",)]
    gold_funds = [f for f in all_funds if catalog.loc[catalog["fund_code"] == f, "asset_class"].values[0]
                  in ("黄金",)]
    qdii_funds = [f for f in all_funds if catalog.loc[catalog["fund_code"] == f, "asset_class"].values[0]
                  in ("QDII港股", "纳指100", "标普500")]

    print(f"  Equity: {len(equity_funds)}, Bond: {len(bond_funds)}, ShortBond: {len(short_bond_funds)}")
    print(f"  Gold: {len(gold_funds)}, QDII: {len(qdii_funds)}")

    # Build returns
    signal = OTFStrategySignal(engine)
    returns_wide = signal.build_returns_wide()
    print(f"Returns shape: {returns_wide.shape}")

    # Walk-forward windows (3-year train, 2-year test, rolling annually)
    all_dates = returns_wide.index
    backtest_start = pd.Timestamp("2018-01-01")
    backtest_end = pd.Timestamp("2026-07-17")

    windows = []
    current = backtest_start
    while current < backtest_end:
        train_start = current
        train_end = current + pd.DateOffset(years=3)
        test_start = train_end
        test_end = min(test_start + pd.DateOffset(years=2), backtest_end)

        if (test_end - train_start).days < 3 * 365:
            break

        windows.append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })
        current = current + pd.DateOffset(years=1)

    print(f"\nWalk-forward windows: {len(windows)}")

    param_combos = [
        {"name": "MPM_60_120_252_T200", "lookbacks": [60, 120, 252], "trend_days": 200, "n_hold": 5},
        {"name": "MPM_120_252_T200", "lookbacks": [120, 252], "trend_days": 200, "n_hold": 5},
        {"name": "MPM_60_120_T200", "lookbacks": [60, 120], "trend_days": 200, "n_hold": 5},
        {"name": "MPM_60_120_252_T120", "lookbacks": [60, 120, 252], "trend_days": 120, "n_hold": 5},
    ]

    # Track cumulative OOS returns for each param set
    param_oos_returns: dict[str, list[pd.Series]] = {p["name"]: [] for p in param_combos}

    # Pre-compute trading dates for signal→submit mapping
    trading_dates = all_dates

    for wi, w in enumerate(windows):
        print(f"\n--- Window {wi+1}/{len(windows)}: "
              f"Train {w['train_start'].date()}~{w['train_end'].date()}, "
              f"Test {w['test_start'].date()}~{w['test_end'].date()}")
        test_dates = all_dates[(all_dates >= w["test_start"]) & (all_dates <= w["test_end"])]
        if len(test_dates) == 0:
            continue

        # Build signal→submit mapping: signal on date, order on next trading day
        signal_map: dict[pd.Timestamp, pd.Timestamp] = {}
        for sd in test_dates:
            sub_candidates = trading_dates[trading_dates > sd]
            if len(sub_candidates) > 0 and sub_candidates[0] <= w["test_end"]:
                signal_map[sub_candidates[0]] = sd

        for param in param_combos:
            name = param["name"]
            target_rows = []
            param_signal_map: dict[pd.Timestamp, pd.Timestamp] = {}

            for sub_date, sd in signal_map.items():
                mom = multi_period_momentum(
                    returns_wide, sd,
                    lookbacks=param["lookbacks"],
                    trend_filter_days=param["trend_days"],
                    n_hold=param["n_hold"],
                )
                row: dict[str, object] = {"date": sub_date}
                for f in engine.available_fund_codes:
                    row[f] = mom.get(f, 0.0)
                target_rows.append(row)
                param_signal_map[sub_date] = sd

            targets_df = pd.DataFrame(target_rows)
            if targets_df.empty:
                continue
            targets_df = targets_df.set_index("date")

            wf_engine = OTFBacktestEngine(
                db_path="data/processed/otf.sqlite",
                product_rule_book=rule_book,
                confirmation_days_subscribe=1,
                confirmation_days_redeem=1,
                settlement_days_redeem=7,
                subscription_fee_rate=0.001,
                redemption_fee_rate=0.0015,
                initial_cash=1_000_000.0,
                minimum_trade_ratio=0.01,
                strict_product_rules=True,
            )
            daily = wf_engine.run_backtest(
                targets_df,
                start=str(w["test_start"].date()),
                end=str(w["test_end"].date()),
                rebalance_every=21,
                signal_dates=param_signal_map,
            )

            param_oos_returns[name].append(pd.Series(
                daily["daily_return"].values, index=pd.to_datetime(daily["date"])
            ))

        # Equal weight benchmark with signal→submit separation
        ew_rows = []
        ew_signal_map: dict[pd.Timestamp, pd.Timestamp] = {}
        for sub_date, sd in signal_map.items():
            ew_weight = 1.0 / max(len(engine.available_fund_codes), 1)
            row: dict[str, object] = {"date": sub_date}
            for f in engine.available_fund_codes:
                row[f] = ew_weight
            ew_rows.append(row)
            ew_signal_map[sub_date] = sd

        ew_targets = pd.DataFrame(ew_rows).set_index("date") if ew_rows else pd.DataFrame()

        wf_engine_ew = OTFBacktestEngine(
            db_path="data/processed/otf.sqlite",
            product_rule_book=rule_book,
            confirmation_days_subscribe=1,
            confirmation_days_redeem=1,
            settlement_days_redeem=7,
            subscription_fee_rate=0.001,
            redemption_fee_rate=0.0015,
            initial_cash=1_000_000.0,
            minimum_trade_ratio=0.01,
            strict_product_rules=True,
        )
        ew_daily = wf_engine_ew.run_backtest(
            ew_targets,
            start=str(w["test_start"].date()),
            end=str(w["test_end"].date()),
            rebalance_every=21,
            signal_dates=ew_signal_map,
        )
        if "EW_OOS" not in param_oos_returns:
            param_oos_returns["EW_OOS"] = []
        param_oos_returns["EW_OOS"].append(pd.Series(
            ew_daily["daily_return"].values, index=pd.to_datetime(ew_daily["date"])
        ))

    # Aggregate OOS results
    print("\n\n" + "=" * 60)
    print("OOS PERFORMANCE SUMMARY")
    print("=" * 60)

    all_results = []
    for name, returns_list in param_oos_returns.items():
        if not returns_list:
            continue
        combined = pd.concat(returns_list).sort_index()
        # Deduplicate on index (keep first)
        combined = combined[~combined.index.duplicated(keep="first")]

        wealth = (1.0 + combined).cumprod()
        total_ret = wealth.iloc[-1] - 1.0
        n = len(combined)
        ann_ret = wealth.iloc[-1] ** (252.0 / n) - 1.0 if n > 1 else 0.0
        ann_std = combined.std() * np.sqrt(252) if n > 1 else 0.0
        sharpe = ann_ret / ann_std if ann_std > 0 else 0.0
        dd = wealth / wealth.cummax() - 1.0
        mdd = dd.min()
        calmar = ann_ret / abs(mdd) if mdd < 0 else 0.0
        wr = (combined > 0).mean()

        result = {
            "strategy": name,
            "CAGR_pct": round(float(ann_ret) * 100, 2),
            "Total_Return_pct": round(float(total_ret) * 100, 2),
            "Sharpe": round(float(sharpe), 3),
            "Max_Drawdown_pct": round(float(mdd) * 100, 2),
            "Volatility_pct": round(float(ann_std) * 100, 2),
            "Calmar": round(float(calmar), 3),
            "Win_Rate_pct": round(float(wr) * 100, 1),
            "n_days": n,
        }
        all_results.append(result)
        print(f"\n{name}:")
        print(f"  CAGR={result['CAGR_pct']}%, Sharpe={result['Sharpe']}, "
              f"MDD={result['Max_Drawdown_pct']}%, Calmar={result['Calmar']}, "
              f"WinRate={result['Win_Rate_pct']}%")

    # Save results
    results_df = pd.DataFrame(all_results)
    csv_path = OUTPUT_DIR / "otf_walkforward_summary.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    json_path = OUTPUT_DIR / "otf_walkforward_manifest.json"
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "param_combos": param_combos,
        "windows": [
            {"train_start": str(w["train_start"].date()),
             "train_end": str(w["train_end"].date()),
             "test_start": str(w["test_start"].date()),
             "test_end": str(w["test_end"].date())}
            for w in windows
        ],
        "results": all_results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Manifest saved: {json_path}")

    print("\n" + "=" * 60)
    print("B7 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
