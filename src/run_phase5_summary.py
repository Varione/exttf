"""Phase 5.4: Summary Report."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    print("=" * 80)
    print("Phase 5.4: Summary Report")
    print("=" * 80)

    out_dir = Path("reports/strategy_research")
    summary_rows: list[dict] = []

    # 1. S04 Parameter Study Results
    param_path = out_dir / "s04_parameter_study.csv"
    if param_path.exists():
        param_df = pd.read_csv(param_path)
        avg = param_df.groupby(["trend_lb", "vol_lb", "target_vol", "rebalance"])[
            ["CAGR%", "Sharpe", "Max_Drawdown%", "excess_vs_b0%"]
        ].mean().reset_index()
        best = avg.sort_values("Sharpe", ascending=False).iloc[0]

        summary_rows.append({
            "section": "S04_Parameter_Study",
            "metric": "best_trend_lb",
            "value": int(best["trend_lb"]),
            "detail": f"Sharpe={best['Sharpe']:.4f}",
        })
        summary_rows.append({
            "section": "S04_Parameter_Study",
            "metric": "best_vol_lb",
            "value": int(best["vol_lb"]),
            "detail": "",
        })
        summary_rows.append({
            "section": "S04_Parameter_Study",
            "metric": "best_target_vol",
            "value": float(best["target_vol"]),
            "detail": "",
        })
        summary_rows.append({
            "section": "S04_Parameter_Study",
            "metric": "best_rebalance",
            "value": int(best["rebalance"]),
            "detail": "",
        })
        summary_rows.append({
            "section": "S04_Parameter_Study",
            "metric": "best_ann_return%",
            "value": float(best["CAGR%"]),
            "detail": "",
        })
        summary_rows.append({
            "section": "S04_Parameter_Study",
            "metric": "best_sharpe",
            "value": float(best["Sharpe"]),
            "detail": "",
        })
        summary_rows.append({
            "section": "S04_Parameter_Study",
            "metric": "best_max_dd%",
            "value": float(best["Max_Drawdown%"]),
            "detail": "",
        })
        summary_rows.append({
            "section": "S04_Parameter_Study",
            "metric": "best_excess_vs_b0%",
            "value": float(best["excess_vs_b0%"]),
            "detail": "",
        })

        # Worst for comparison
        worst = avg.sort_values("Sharpe", ascending=True).iloc[0]
        summary_rows.append({
            "section": "S04_Parameter_Study",
            "metric": "worst_sharpe",
            "value": float(worst["Sharpe"]),
            "detail": f"params={worst['trend_lb']},{worst['vol_lb']},{worst['target_vol']},{worst['rebalance']}",
        })

        print(f"S04 Best params: trend_lb={int(best['trend_lb'])}, vol_lb={int(best['vol_lb'])}, "
              f"target_vol={best['target_vol']:.2f}, rebalance={int(best['rebalance'])}")
        print(f"  Sharpe={best['Sharpe']:.4f}, CAGR%={best['CAGR%']:.2f}, MaxDD%={best['Max_Drawdown%']:.2f}")
    else:
        print("WARNING: s04_parameter_study.csv not found")

    # 2. S04 Component Decomposition
    decomp_path = out_dir / "s04_component_decomposition.csv"
    if decomp_path.exists():
        decomp_df = pd.read_csv(decomp_path)
        avg_comp = decomp_df.groupby("component")[["CAGR%", "Sharpe", "Max_Drawdown%"]].mean()

        for comp in avg_comp.index:
            row_data = avg_comp.loc[comp]
            summary_rows.append({
                "section": "S04_Decomposition",
                "metric": comp,
                "value": float(row_data["Sharpe"]),
                "detail": f"CAGR%={row_data['CAGR%']:.2f}, MaxDD%={row_data['Max_Drawdown%']:.2f}",
            })

        # Determine dominant component
        trend_row = avg_comp.loc["S04_Trend_Only"] if "S04_Trend_Only" in avg_comp.index else None
        vol_row = avg_comp.loc["S04_VolTarget_Only"] if "S04_VolTarget_Only" in avg_comp.index else None
        full_row = avg_comp.loc["S04_VolTarget_Trend"] if "S04_VolTarget_Trend" in avg_comp.index else None
        trend_sharpe = trend_row["Sharpe"] if trend_row is not None else np.nan
        vol_sharpe = vol_row["Sharpe"] if vol_row is not None else np.nan
        full_sharpe = full_row["Sharpe"] if full_row is not None else np.nan

        if trend_row is not None and vol_row is not None:
            best_single = max(trend_sharpe, vol_sharpe)
            dominant = "Trend" if trend_sharpe >= vol_sharpe else "Vol_Targeting"
            synergy = full_sharpe - best_single if not np.isnan(full_sharpe) else np.nan

            summary_rows.append({
                "section": "S04_Decomposition",
                "metric": "dominant_component",
                "value": 0,
                "detail": dominant,
            })
            summary_rows.append({
                "section": "S04_Decomposition",
                "metric": "synergy_sharpe",
                "value": float(synergy) if not np.isnan(synergy) else 0.0,
                "detail": "positive=complementary, negative=dominant single component",
            })

        print(f"\nDecomposition: Trend Sharpe={trend_sharpe:.4f}, Vol Sharpe={vol_sharpe:.4f}, Full={full_sharpe:.4f}")
    else:
        print("WARNING: s04_component_decomposition.csv not found")

    # 3. Strategy Combination Results
    comb_path = out_dir / "strategy_combination_results.csv"
    if comb_path.exists():
        comb_df = pd.read_csv(comb_path)
        avg_comb = comb_df.groupby("method")[["CAGR%", "Sharpe", "Max_Drawdown%", "excess_vs_b0%"]].mean()

        for method in avg_comb.index:
            row_data = avg_comb.loc[method]
            summary_rows.append({
                "section": "Strategy_Combination",
                "metric": method,
                "value": float(row_data["Sharpe"]),
                "detail": f"CAGR%={row_data['CAGR%']:.2f}, MaxDD%={row_data['Max_Drawdown%']:.2f}, excess%={row_data['excess_vs_b0%']:.2f}",
            })

        # Best combination method
        best_method = avg_comb.sort_values("Sharpe", ascending=False).index[0]
        summary_rows.append({
            "section": "Strategy_Combination",
            "metric": "best_method",
            "value": 0,
            "detail": best_method,
        })

        print(f"\nCombination: Best method = {best_method}")
        print(avg_comb.to_string())
    else:
        print("WARNING: strategy_combination_results.csv not found")

    # 4. Write summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "phase5_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")

    return summary_df


if __name__ == "__main__":
    main()
