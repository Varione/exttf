"""
S1 Experiment: State Rotation + Fixed Representative Products

Fixes:
- §11.3: Signal dates (month-end) → submit dates (next trading day) with signal_dates map
- §11.2: ProductSelector requires date, funds filtered by inception/NAV availability
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
from otf_rotation.schedule import build_month_end_schedule, build_signal_submit_map
from otf_rotation.product_selector import ProductSelector
from otf_rotation.risk_parity import RollingRiskParityEngine
from otf_rotation.experiment_artifacts import (
    create_run_directory, write_manifest, export_daily_nav, export_orders,
    export_rejections, export_metrics, sha256_file,
)
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
# §11.3: Signal → submit schedule (shared per P1-1)
# ================================================================

def build_schedule(
    engine: OTFBacktestEngine,
    start: str,
    end: str,
) -> tuple[list[pd.Timestamp], dict[str, str]]:
    """Build month-end signal schedule using shared scheduler.

    Returns:
        signal_dates: list of signal observation dates (last trading day of each month)
        signal_map: dict[submit_date_str -> signal_date_str]

    Per P1-1: no pseudo month-end at backtest start; first signal is first natural month-end.
    """
    trading_dates = engine._trading_dates
    signal_dates = build_month_end_schedule(trading_dates, start, end)
    signal_map = build_signal_submit_map(trading_dates, signal_dates, end)
    return signal_dates, signal_map


# ================================================================
# Strategy Signals
# ================================================================

def static_60_20_20_signal(date: pd.Timestamp) -> dict[str, float]:
    return {"160706": 0.60, "000218": 0.20, "001512": 0.20}


def static_risk_parity_signal(date: pd.Timestamp) -> dict[str, float]:
    return {"160706": 0.25, "000218": 0.25, "001512": 0.25, "260102": 0.25}


class StateRotationSignal:
    """S1: MSE → AssetBudget → ExposureSelector with risk control wiring (P1-A).

    Wires prev_weights, vol_scale_weights, and MSE reset per planning.md P1-A.
    """

    def __init__(
        self,
        rule_book: ProductRuleBook | None = None,
        nav_df: pd.DataFrame | None = None,
    ):
        self._mse = MarketStateEngine()
        self._budget = AssetBudgetEngine()
        # S1: never creates or calls ProductSelector (P0-1)
        self._selector = ExposureSelector(
            self._mse, self._budget,
            product_selector=None,
            top_n_funds=2,
        )
        self._last_state: str | None = None
        self._prev_sleeve_weights: dict[str, float] | None = None
        self._prev_fund_weights: dict[str, float] | None = None
        self._nav_df = nav_df
        self.last_signal_audit: dict[str, object] = {}

    def reset(self):
        """Reset for independent experiment (P1-A)."""
        self._mse.reset()
        self._selector.reset()
        self._last_state = None
        self._prev_sleeve_weights = None
        self._prev_fund_weights = None
        self.last_signal_audit = {}

    def __call__(self, signal_date: pd.Timestamp) -> dict[str, float]:
        state, scores, _ = self._mse.get_state(signal_date, prev_state=self._last_state)
        self._last_state = state

        # Keep the rebalance band and volatility scaling in sleeve space.
        # Only map to fund codes after all sleeve-level risk controls have
        # completed; otherwise ``cash_mgt`` can become an invalid fund code
        # and previous fund-code keys cannot match sleeve names.
        sleeve_weights = self._selector.select_weights(
            signal_date,
            state,
            prev_weights=self._selector.previous_sleeve_weights,
        )
        if len(sleeve_weights) == 0:
            return {"260102": 1.0}

        total = sleeve_weights.sum()
        if total > 1.0:
            sleeve_weights = sleeve_weights / total
        elif total < 1.0:
            cash_sleeve = self._budget.get_class_sleeves("cash_mgt")[0]
            sleeve_weights[cash_sleeve] = (
                sleeve_weights.get(cash_sleeve, 0.0) + (1.0 - total)
            )

        sleeve_result = sleeve_weights.to_dict()
        # ``vol_scale_weights`` accepts one cash bucket. Collapse all cash
        # management sleeves before scaling so CD_NCD is not treated as risky
        # and the scaled weights cannot exceed 100%.
        cash_sleeves = self._budget.get_class_sleeves("cash_mgt")
        cash_sleeve = cash_sleeves[0]
        cash_total = sum(sleeve_result.pop(key, 0.0) for key in cash_sleeves)
        sleeve_result[cash_sleeve] = cash_total
        raw_fund_weights = self._selector.map_sleeves_to_fixed_products(
            sleeve_result
        )

        # P1-A: apply the 9% target at the sleeve layer, then map to funds.
        scaled_sleeves = self._budget.vol_scale_weights(
            weights=sleeve_result,
            estimated_port_vol=self._estimate_portfolio_vol(
                signal_date, raw_fund_weights.to_dict()
            ),
            cash_class=self._budget.get_class_sleeves("cash_mgt")[0],
        )
        # Enforce the portfolio invariant after every risk-control operation.
        # This is deliberately an assertion-by-construction, not a truncation
        # in the backtest engine: any residual is assigned to the valid cash
        # sleeve and the exact sleeve target is what gets mapped to funds.
        scaled_sleeves = {
            key: max(0.0, float(value))
            for key, value in scaled_sleeves.items()
            if float(value) > 1e-12
        }
        scaled_total = sum(scaled_sleeves.values())
        if scaled_total > 1.0 + 1e-12:
            scaled_sleeves = {
                key: value / scaled_total
                for key, value in scaled_sleeves.items()
            }
        elif scaled_total < 1.0 - 1e-12:
            cash_sleeve = self._budget.get_class_sleeves("cash_mgt")[0]
            scaled_sleeves[cash_sleeve] = scaled_sleeves.get(cash_sleeve, 0.0) + (1.0 - scaled_total)
        self._selector.set_previous_sleeve_weights(scaled_sleeves)
        self._prev_sleeve_weights = dict(scaled_sleeves)
        scaled_result = self._selector.map_sleeves_to_fixed_products(scaled_sleeves)
        result = scaled_result.to_dict()

        self._prev_fund_weights = result
        self.last_signal_audit = {
            "signal_date": str(pd.Timestamp(signal_date).date()),
            "market_state": state,
            "state_scores": scores,
            "sleeve_weights": dict(scaled_sleeves),
            "fund_weights": dict(result),
            "estimated_portfolio_vol": self._estimate_portfolio_vol(
                signal_date, raw_fund_weights.to_dict()
            ),
            "target_vol": self._budget.target_vol,
        }
        return result

    def _estimate_portfolio_vol(
        self, signal_date: pd.Timestamp, fund_weights: dict[str, float]
    ) -> float:
        """Estimate annualized portfolio volatility from historical NAV returns.

        Uses selected funds over last 60 trading days before signal_date.
        Returns 0.09 (target) if insufficient data, so no scaling occurs.
        """
        try:
            nav_df = self._nav_df
            if nav_df is None:
                return 0.09

            funds = [f for f in fund_weights.keys() if fund_weights[f] > 0]
            if not funds:
                return 0.09

            subset = nav_df[nav_df["fund_code"].isin(funds)]
            subset = subset[subset["nav_date"] <= signal_date]
            pivoted = subset.pivot_table(index="nav_date", columns="fund_code", values="unit_nav")
            if len(pivoted) < 30:
                return 0.09

            returns = pivoted.pct_change(fill_method=None).dropna()
            if len(returns) < 20:
                return 0.09

            weights_arr = np.array([fund_weights.get(f, 0.0) for f in returns.columns])
            port_returns = (returns * weights_arr).sum(axis=1)
            annualized_vol = port_returns.std(ddof=1) * np.sqrt(252)
            return float(annualized_vol)
        except Exception:
            return 0.09


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
    """Run strategy with signal→submit separation."""
    print(f"  Running {name} ...")

    target_weights: dict[str, dict[str, float]] = {}

    # Build submit_date → submit_date reverse map: for each signal_date,
    # find the submit_date it maps to.
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
    """Compute performance, cost and risk metrics.

    Gross CAGR is reconstructed from daily gross_return to measure
    strategy-level performance before fees. Cost drag = gross - net CAGR.
    """
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

    # Net CAGR
    net_cagr = (eq[-1] / eq[0]) ** (1 / max(years, 0.01)) - 1 if eq[0] > 0 else 0.0

    # Gross CAGR (reconstruct gross equity from gross_return)
    gross_eq = INITIAL_CASH * np.cumprod(1 + gross_rets)
    gross_cagr = (gross_eq[-1] / gross_eq[0]) ** (1 / max(years, 0.01)) - 1 if gross_eq[0] > 0 else 0.0

    # Annualized cost drag = gross - net CAGR
    cost_drag = gross_cagr - net_cagr

    # Sharpe, MDD, Calmar (from net returns)
    sharpe = np.mean(rets) / max(np.std(rets, ddof=1), 1e-8) * np.sqrt(252)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    mdd = float(np.min(dd))
    calmar = net_cagr / max(abs(mdd), 0.01)
    win_rate = np.mean(rets > 0)

    # Cumulative fee amount
    # transaction_cost = daily_fees / prior_end_equity
    # Reconstruct daily fee amount from cost ratio * prior equity
    prior_eq = np.roll(eq, 1)
    prior_eq[0] = INITIAL_CASH
    cost_col = [c for c in daily.columns if "cost" in c.lower() or "fee" in c.lower()]
    cost_ratios = daily[cost_col[0]].values.astype(np.float64) if cost_col else np.zeros(len(eq))
    fee_amounts = cost_ratios * prior_eq
    total_fees = float(fee_amounts.sum())

    # Total cost ratio = cumulative fees / final equity
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
    print("S1 Experiment: State Rotation vs Benchmarks")
    print("=" * 60)

    rule_book = create_rule_book()
    engine = create_engine(rule_book)
    print(f"Engine: {len(engine.product_rule_book.rules)} rules, {len(engine._nav_df)} NAV records")
    print(f"strict_product_rules={engine.strict_product_rules}")

    # Build signal schedule (shared per P1-1)
    signal_dates, signal_map = build_schedule(engine, START, END)
    print(f"Signal dates: {len(signal_dates)} ({signal_dates[0].date()} ~ {signal_dates[-1].date()})")
    print(f"Signal→submit mapping: {len(signal_map)} entries")
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

    s1_signal = StateRotationSignal(rule_book=rule_book, nav_df=engine._nav_df)
    strategies.append(("S1_State_Rotation_Fixed", s1_signal))

    # P1-A: reset MSE before each independent strategy run
    for name, sig_func in strategies:
        if hasattr(sig_func, "reset"):
            sig_func.reset()

    # Base output directory for this experiment batch
    base_output_dir = Path("reports/strategy_research/s1_experiment")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    all_run_dirs: list[str] = []

    for name, sig_func in strategies:
        daily = run_strategy(name, sig_func, engine, signal_dates, signal_map)
        metrics = compute_metrics(daily)
        metrics["strategy"] = name
        results.append(metrics)

        print(f"  {name}: NetCAGR={metrics['net_cagr_pct']}% GrossCAGR={metrics['gross_cagr_pct']}% "
              f"CostDrag={metrics['annualized_cost_drag_pct']}% Sharpe={metrics['sharpe']}, "
              f"MDD={metrics['mdd_pct']}% Calmar={metrics['calmar']}, "
              f"Fees={metrics.get('total_fees', 0):.2f}")

        # Export full artifacts for this strategy (P0-C)
        run_dir = create_run_directory(str(base_output_dir), f"run_{name}")
        all_run_dirs.append(run_dir)

        export_daily_nav(run_dir, daily, "daily_account.csv")

        returns_df = daily[["date", "daily_return", "gross_return"]].copy()
        returns_df.rename(columns={"daily_return": "net_return"}, inplace=True)
        export_daily_nav(run_dir, returns_df, "daily_returns.csv")

        orders_df = engine.order_audit_frame()
        export_orders(run_dir, orders_df)

        rejections = engine.last_rejections.to_dict(orient="records") if hasattr(engine, "last_rejections") and len(engine.last_rejections) > 0 else []
        export_rejections(run_dir, rejections)

        export_metrics(run_dir, metrics)

        write_manifest(
            run_dir=run_dir,
            strategy_name=name,
            config={"start": START, "end": END, "initial_cash": INITIAL_CASH},
            db_path=DB_PATH,
            rules_path=RULES_PATH,
            exposure_mapping_path="config/otf_exposure_mapping.csv",
            metrics=metrics,
        )

        print(f"    Artifacts: {run_dir}")

    # Summary
    print()
    print("=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    hdr = (f"{'Strategy':30s} {'NetCAGR%':>9s} {'GrossCAGR%':>10s} {'CostDrag%':>9s} "
           f"{'Sharpe':>7s} {'MDD%':>7s} {'Calmar':>7s} {'Fees':>10s}")
    print(hdr)
    print("-" * 90)
    for r in sorted(results, key=lambda x: -x["net_cagr_pct"]):
        print(f"{r['strategy']:30s} {r['net_cagr_pct']:8.2f}% {r['gross_cagr_pct']:9.2f}% "
              f"{r['annualized_cost_drag_pct']:8.2f}% {r['sharpe']:6.3f} {r['mdd_pct']:6.2f}% "
              f"{r['calmar']:6.3f} {r.get('total_fees', 0):9.2f}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(base_output_dir / "s1_summary.csv", index=False)

    manifest = {
        "experiment": "S1_State_Rotation_vs_Benchmarks",
        "period": f"{START}~{END}",
        "strategies": [r["strategy"] for r in results],
        "signal_dates": len(signal_dates),
        "signal_map_entries": len(signal_map),
        "run_directories": all_run_dirs,
        "input_hashes": {
            "db_sha256": sha256_file(DB_PATH),
            "rules_sha256": sha256_file(RULES_PATH),
        },
        "status": "completed",
    }
    with open(base_output_dir / "s1_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nResults saved to {base_output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
