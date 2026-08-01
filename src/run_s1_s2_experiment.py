"""
S1 + S2 Experiment: Compare State Rotation with Fixed vs Dynamic Products.

Per planning.md P0-1, S1 and S2 are strictly isolated:
- S1 (StateRotationSignalS1): MarketState + AssetBudget + FIXED verified products.
  Never creates or calls ProductSelector.
- S2 (StateRotationSignalS2): MarketState + AssetBudget + DYNAMIC ProductSelector.
  Never falls back to fixed products; budget transfers to fallback cash sleeves.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import logging
import pandas as pd
import numpy as np

from otf_rotation.market_state import MarketStateEngine
from otf_rotation.asset_budget import AssetBudgetEngine, ExposureSelector
from otf_rotation.product_selector import ProductSelector
from otf_rotation.schedule import build_month_end_schedule, build_signal_submit_map
from otf_rotation.risk_parity import RollingRiskParityEngine
from otf_backtest_engine import OTFBacktestEngine
from otf_trading_rules import ProductRuleBook

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DB_PATH = "data/processed/otf_expanded.sqlite"
RULES_PATH = "config/otf_product_rules.csv"
START = "2018-01-01"
END = "2026-07-17"
INITIAL_CASH = 1_000_000.0


def create_rule_book() -> ProductRuleBook:
    return ProductRuleBook.from_csv(RULES_PATH)


def create_engine(rule_book: ProductRuleBook) -> OTFBacktestEngine:
    return OTFBacktestEngine(
        db_path=DB_PATH,
        product_rule_book=rule_book,
        initial_cash=INITIAL_CASH,
        confirmation_days_subscribe=1,
        confirmation_days_redeem=1,
        settlement_days_redeem=7,
        subscription_fee_rate=0.001,
        redemption_fee_rate=0.0015,
        strict_product_rules=True,
        minimum_trade_ratio=0.005,
    )


# ================================================================
# §11.3: Signal -> submit schedule
# ================================================================

def build_schedule(
    engine: OTFBacktestEngine,
    start: str,
    end: str,
) -> tuple[list[pd.Timestamp], dict[str, str]]:
    """Build month-end signal schedule using shared scheduler per P1-1."""
    trading_dates = engine._trading_dates
    signal_dates = build_month_end_schedule(trading_dates, start, end)
    signal_map = build_signal_submit_map(trading_dates, signal_dates, end)
    return signal_dates, signal_map


# ================================================================
# Signal Functions
# ================================================================

def static_60_20_20_signal(date: pd.Timestamp) -> dict[str, float]:
    return {"160706": 0.60, "000218": 0.20, "001512": 0.20}


def static_risk_parity_signal(date: pd.Timestamp) -> dict[str, float]:
    return {"160706": 0.25, "000218": 0.25, "001512": 0.25, "260102": 0.25}


class StateRotationSignalS1:
    """S1: MarketState + AssetBudget + FIXED verified products.

    Never creates or calls ProductSelector. Reads only from fixed_product_map.
    """

    def __init__(self, rule_book: ProductRuleBook | None = None):
        self._mse = MarketStateEngine()
        self._budget = AssetBudgetEngine()
        # S1 selector has NO product_selector injected — strict isolation
        self._selector = ExposureSelector(
            self._mse, self._budget,
            product_selector=None,
            top_n_funds=1,
        )
        self._last_state: str | None = None

    def __call__(self, signal_date: pd.Timestamp) -> dict[str, float]:
        state, scores, _ = self._mse.get_state(signal_date, prev_state=self._last_state)
        self._last_state = state

        fund_weights = self._selector.map_to_fixed_products(signal_date, state)
        if len(fund_weights) == 0:
            return {"260102": 1.0}

        total = fund_weights.sum()
        if total > 1.0:
            fund_weights = fund_weights / total
        elif total < 1.0:
            fund_weights["260102"] = fund_weights.get("260102", 0.0) + (1.0 - total)

        return fund_weights.to_dict()


class StateRotationSignalS2:
    """S2: MarketState + AssetBudget + DYNAMIC ProductSelector.

    Never falls back to fixed products when no candidates are found;
    budget transfers to fallback cash/short-term sleeves instead.
    """

    def __init__(self, rule_book: ProductRuleBook | None = None):
        self._mse = MarketStateEngine()
        self._budget = AssetBudgetEngine()
        self._product_selector = ProductSelector(
            db_path=DB_PATH,
            rules_path=RULES_PATH,
            allow_auto_rules=False,
            rule_book=rule_book,
        )
        self._selector = ExposureSelector(
            self._mse, self._budget,
            product_selector=self._product_selector,
            top_n_funds=2,
        )
        self._last_state: str | None = None

    def __call__(self, signal_date: pd.Timestamp) -> dict[str, float]:
        state, scores, _ = self._mse.get_state(signal_date, prev_state=self._last_state)
        self._last_state = state

        fund_weights = self._selector.map_to_dynamic_products(
            signal_date, state, fallback_sleeves=["MONEY_MARKET"]
        )
        if len(fund_weights) == 0:
            return {"260102": 1.0}

        total = fund_weights.sum()
        if total > 1.0:
            fund_weights = fund_weights / total
        elif total < 1.0:
            fund_weights["260102"] = fund_weights.get("260102", 0.0) + (1.0 - total)

        return fund_weights.to_dict()


# ================================================================
# Run
# ================================================================

def run_strategy(
    name: str,
    signal_func,
    engine: OTFBacktestEngine,
    signal_dates: list[pd.Timestamp],
    signal_map: dict[str, str],
) -> pd.DataFrame:
    """Run strategy with signal->submit separation."""
    print(f"  Running {name} ...")

    target_weights: dict[str, dict[str, float]] = {}

    submit_for_signal: dict[str, str] = {}
    for sub_str, sig_str in signal_map.items():
        submit_for_signal[sig_str] = sub_str

    for sd in signal_dates:
        sd_str = sd.strftime("%Y-%m-%d")
        w = signal_func(sd)
        submit_date = submit_for_signal.get(sd_str)
        if submit_date is None:
            continue
        target_weights[submit_date] = w

    target_df = pd.DataFrame.from_dict(target_weights, orient="index")
    target_df = target_df.fillna(0.0).sort_index()
    target_df = target_df.loc[:, (target_df != 0).any(axis=0)]

    daily = engine.run_backtest(
        target_df,
        start=START,
        end=END,
        rebalance_every=1,
        signal_dates={pd.Timestamp(k): pd.Timestamp(v) for k, v in signal_map.items()},
    )
    return daily


def compute_metrics(daily: pd.DataFrame) -> dict:
    """Compute performance, cost and risk metrics."""
    if len(daily) == 0:
        return {
            "net_cagr_pct": 0, "gross_cagr_pct": 0, "sharpe": 0,
            "mdd_pct": 0, "calmar": 0, "win_rate_pct": 0,
            "total_cost_pct": 0, "annualized_cost_drag_pct": 0,
            "final_equity": 0, "total_fees": 0,
        }

    eq = daily["equity"].values.astype(np.float64)
    rets = daily["daily_return"].values.astype(np.float64)
    gross_rets = daily["gross_return"].values.astype(np.float64)

    total_days = len(rets)
    years = total_days / 252

    net_cagr = (eq[-1] / eq[0]) ** (1 / max(years, 0.01)) - 1 if eq[0] > 0 else 0.0

    gross_eq = INITIAL_CASH * np.cumprod(1 + gross_rets)
    gross_cagr = (gross_eq[-1] / gross_eq[0]) ** (1 / max(years, 0.01)) - 1 if gross_eq[0] > 0 else 0.0

    cost_drag = gross_cagr - net_cagr

    sharpe = np.mean(rets) / max(np.std(rets, ddof=1), 1e-8) * np.sqrt(252)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    mdd = float(np.min(dd))
    calmar = net_cagr / max(abs(mdd), 0.01)
    win_rate = np.mean(rets > 0)

    prior_eq = np.roll(eq, 1)
    prior_eq[0] = INITIAL_CASH
    cost_col = [c for c in daily.columns if "cost" in c.lower() or "fee" in c.lower()]
    cost_ratios = daily[cost_col[0]].values.astype(np.float64) if cost_col else np.zeros(len(eq))
    fee_amounts = cost_ratios * prior_eq
    total_fees = float(fee_amounts.sum())

    total_cost_ratio = total_fees / eq[-1] if eq[-1] > 0 else 0.0

    return {
        "net_cagr_pct": round(net_cagr * 100, 2),
        "gross_cagr_pct": round(gross_cagr * 100, 2),
        "annualized_cost_drag_pct": round(cost_drag * 100, 2),
        "sharpe": round(sharpe, 3),
        "mdd_pct": round(mdd * 100, 2),
        "calmar": round(calmar, 3),
        "win_rate_pct": round(win_rate * 100, 1),
        "total_cost_ratio_pct": round(total_cost_ratio * 100, 4),
        "final_equity": round(eq[-1], 2),
        "total_fees": round(total_fees, 2),
    }


def main():
    print("=" * 60)
    print("S1+S2 Experiment: State Rotation vs Benchmarks")
    print("=" * 60)

    rule_book = create_rule_book()
    engine = create_engine(rule_book)
    print(f"Engine: {len(engine.product_rule_book.rules)} rules, {len(engine._nav_df)} NAV records")
    print(f"strict_product_rules={engine.strict_product_rules}")

    signal_dates, signal_map = build_schedule(engine, START, END)
    print(f"Signal dates: {len(signal_dates)} ({signal_dates[0].date()} ~ {signal_dates[-1].date()})")
    print(f"Signal->submit mapping: {len(signal_map)} entries")
    print()

    # B3 Rolling Risk Parity using same 4 assets as B2
    b2_assets = ["160706", "000218", "001512", "260102"]
    nav_subset = engine._nav_df[engine._nav_df["fund_code"].isin(b2_assets)]
    asset_navs = nav_subset.pivot_table(
        index="nav_date", columns="fund_code", values="unit_nav"
    ).sort_index().dropna(how="all")
    b3_engine = RollingRiskParityEngine(
        nav_df=asset_navs,
        asset_columns=b2_assets,
        rolling_window=252,
        min_window=60,
        asset_cap=0.50,
    )

    strategies = [
        ("B1_Static_60_20_20", lambda d: static_60_20_20_signal(d)),
        ("B2_Static_EW_4Asset", lambda d: static_risk_parity_signal(d)),
        ("B3_Rolling_Risk_Parity", b3_engine.get_weights),
    ]

    # S1: fixed products — never uses ProductSelector
    s1_signal = StateRotationSignalS1(rule_book=rule_book)
    strategies.append(("S1_State_Rotation_Fixed", s1_signal))

    # S2: dynamic products — never falls back to fixed products
    s2_signal = StateRotationSignalS2(rule_book=rule_book)
    strategies.append(("S2_State_Rotation_Dynamic", s2_signal))

    results = []
    for name, sig_func in strategies:
        daily = run_strategy(name, sig_func, engine, signal_dates, signal_map)
        metrics = compute_metrics(daily)
        metrics["strategy"] = name
        results.append(metrics)

        print(f"  {name}: NetCAGR={metrics['net_cagr_pct']}% GrossCAGR={metrics['gross_cagr_pct']}% "
              f"CostDrag={metrics['annualized_cost_drag_pct']}% Sharpe={metrics['sharpe']}, "
              f"MDD={metrics['mdd_pct']}% Calmar={metrics['calmar']}, "
              f"Fees={metrics.get('total_fees', 0):.2f}")

    print()
    print("=" * 110)
    print("RESULTS SUMMARY")
    print("=" * 110)
    hdr = (f"{'Strategy':35s} {'NetCAGR%':>9s} {'GrossCAGR%':>10s} {'CostDrag%':>9s} "
           f"{'Sharpe':>7s} {'MDD%':>7s} {'Calmar':>7s} {'Fees':>10s}")
    print(hdr)
    print("-" * 110)
    for r in sorted(results, key=lambda x: -x["net_cagr_pct"]):
        print(f"{r['strategy']:35s} {r['net_cagr_pct']:8.2f}% {r['gross_cagr_pct']:9.2f}% "
              f"{r['annualized_cost_drag_pct']:8.2f}% {r['sharpe']:6.3f} {r['mdd_pct']:6.2f}% "
              f"{r['calmar']:6.3f} {r.get('total_fees', 0):9.2f}")

    output_dir = Path("reports/strategy_research/s1_s2_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_dir / "s1_s2_summary.csv", index=False)

    manifest = {
        "experiment": "S1_S2_Comparison",
        "period": f"{START}~{END}",
        "strategies": [r["strategy"] for r in results],
        "signal_dates": len(signal_dates),
        "signal_map_entries": len(signal_map),
        "rules_count": len(rule_book.rules),
        "status": "completed",
    }
    with open(output_dir / "s1_s2_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
