"""Phase 2.2: Transaction cost sensitivity analysis.

Runs S04, S01, B0, B1 under three cost scenarios:
  Low:    fee 0.01% + slippage 0.01%  (total 2bp per side)
  Base:   fee 0.03% + slippage 0.02%  (total 5bp per side)
  Stress: fee 0.05% + slippage 0.05%  (total 10bp per side)

Outputs reports/strategy_research/cost_sensitivity.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtest_engine import BacktestEngine

DB_PATH = "data/processed/etf.sqlite"
FACTOR_CSV = "data/processed/factors_all_repaired.csv"
REGIME_CSV = "data/processed/regime_predictions.csv"

STRATEGIES_TO_TEST = [
    "S04_VolTarget_Trend",
    "S01_CS_Momentum",
    "B0_BuyHold_EW",
    "B1_Momentum_Agnostic",
]

COST_SCENARIOS = {
    "Low": {"fee_rate_per_side": 0.0001, "slippage_rate_per_side": 0.0001},
    "Base": {"fee_rate_per_side": 0.0003, "slippage_rate_per_side": 0.0002},
    "Stress": {"fee_rate_per_side": 0.0005, "slippage_rate_per_side": 0.0005},
}


def run_sensitivity():
    rows: list[dict] = []

    for scenario_name, cost_params in COST_SCENARIOS.items():
        print(f"\n=== Cost Scenario: {scenario_name} ===")
        print(f"  fee={cost_params['fee_rate_per_side']:.4%}, slippage={cost_params['slippage_rate_per_side']:.4%}")

        engine = BacktestEngine(
            factor_path=FACTOR_CSV,
            regime_path=REGIME_CSV,
            db_path=DB_PATH,
            data_mode="etf",
            price_mode="total_return_proxy",
            fee_rate_per_side=cost_params["fee_rate_per_side"],
            slippage_rate_per_side=cost_params["slippage_rate_per_side"],
            require_pit=True,
        )

        for strategy_name in STRATEGIES_TO_TEST:
            print(f"  Running {strategy_name}...")
            daily = engine.run_backtest(
                strategy_name,
                start="2018-01-01",
                end="2026-07-17",
                n_hold=20,
                max_weight=0.05,
                signal_to_return_lag=2,
                rebalance_every=5,
            )

            metrics = engine.calculate_metrics(
                daily["return"],
                turnover=daily["turnover"],
                transaction_cost=daily["transaction_cost"],
                gross_returns=daily["gross_return"],
                exposures=daily["exposure"],
            )

            row = {
                "strategy": strategy_name,
                "cost_scenario": scenario_name,
                "ann_return%": round(metrics.get("ann_return", 0.0), 4),
                "sharpe": round(metrics.get("sharpe", 0.0), 4),
                "max_dd%": round(metrics.get("max_drawdown", 0.0), 4),
                "annualized_cost_drag%": round(
                    metrics.get("annualized_cost_drag", 0.0), 4
                ),
                "total_turnover": round(metrics.get("total_turnover", 0.0), 2),
                "avg_exposure%": round(daily["exposure"].mean() * 100, 2),
            }
            rows.append(row)
            print(f"    ann_return={row['ann_return%']:.2f}%, sharpe={row['sharpe']:.3f}, "
                  f"max_dd={row['max_dd%']:.2f}%, cost_drag={row['annualized_cost_drag%']:.2f}%")

    df = pd.DataFrame(rows)
    output_dir = Path("reports/strategy_research")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cost_sensitivity.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    run_sensitivity()
