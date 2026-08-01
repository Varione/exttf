"""P1-1d: Order-level rule version coverage report for frozen runs.

For every order in the frozen baseline runs, resolve the dated rule version
covering its submit_date and report coverage per strategy, per product and
overall.  Versions only cover dates inside explicit effective windows, so
orders submitted before the snapshot verification date are honestly reported
as UNCOVERED.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from otf_trading_rules import ProductRuleBook  # noqa: E402

FROZEN_RUNS = {
    "WF_B1": "reports/strategy_research/walkforward_continuous/walkforward_20260801_121725/B1_Static_60_20_20/orders.csv",
    "WF_B2": "reports/strategy_research/walkforward_continuous/walkforward_20260801_121725/B2_Static_EW_4Asset/orders.csv",
    "WF_B3": "reports/strategy_research/walkforward_continuous/walkforward_20260801_121725/B3_Rolling_Risk_Parity/orders.csv",
    "WF_S1": "reports/strategy_research/walkforward_continuous/walkforward_20260801_121725/S1_State_Rotation_Fixed/orders.csv",
    "C1": "reports/strategy_research/core_satellite/core_satellite_20260801_114514/C1_CORE_SATELLITE_MOMENTUM/orders.csv",
    "C2": "reports/strategy_research/core_satellite_low_turnover/low_turnover_core_satellite_20260801_115521/C2_LOW_TURNOVER_CORE_SATELLITE/orders.csv",
    "C3": "reports/strategy_research/c3_low_turnover_momentum/c3_low_turnover_momentum_20260801_120449/C3_LOW_TURNOVER_MOMENTUM/orders.csv",
    "M20_D1": "reports/strategy_research/candidate_strategies/candidate_20260801_121109/D1_MultiAsset_Trend_Defensive/orders.csv",
    "M20_B2LT": "reports/strategy_research/candidate_strategies/candidate_20260801_121109/B2_LT_Static_EW_4Asset/orders.csv",
}

OUT_PATH = Path("reports/historical_truth/order_rule_version_coverage.json")


def main() -> int:
    book = ProductRuleBook.from_csv()
    per_strategy: dict[str, dict[str, object]] = {}
    per_product: dict[str, dict[str, object]] = {}
    total_orders = 0
    total_covered = 0
    uncovered_samples: list[dict[str, str]] = []

    for name, path in FROZEN_RUNS.items():
        df = pd.read_csv(path)
        covered = 0
        orders = 0
        for _, row in df.iterrows():
            code = str(row["fund_code"]).zfill(6)
            submit_date = pd.Timestamp(row["submit_date"])
            version = book.rule_version_for(code, submit_date)
            orders += 1
            covered += 1 if version else 0
            per_product.setdefault(
                code, {"orders": 0, "covered": 0, "version_ids": set()}
            )
            per_product[code]["orders"] += 1
            if version:
                per_product[code]["covered"] += 1
                per_product[code]["version_ids"].add(version.rule_version_id)
            elif len(uncovered_samples) < 200:
                uncovered_samples.append(
                    {"strategy": name, "fund_code": code, "submit_date": str(submit_date.date())}
                )
        per_strategy[name] = {
            "orders": orders,
            "covered": covered,
            "uncovered": orders - covered,
            "coverage_ratio": round(covered / orders, 6) if orders else 1.0,
        }
        total_orders += orders
        total_covered += covered

    report = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "rule_versions_path": "config/otf_rule_versions.csv",
        "total_orders": total_orders,
        "total_covered": total_covered,
        "total_uncovered": total_orders - total_covered,
        "overall_coverage_ratio": round(total_covered / total_orders, 6),
        "historical_rule_status": "NOT_ESTABLISHED",
        "per_strategy": per_strategy,
        "per_product": {
            code: {
                "orders": info["orders"],
                "covered": info["covered"],
                "uncovered": info["orders"] - info["covered"],
                "coverage_ratio": round(info["covered"] / info["orders"], 6)
                if info["orders"]
                else 1.0,
                "version_ids": sorted(info["version_ids"]),
            }
            for code, info in per_product.items()
        },
        "uncovered_sample": uncovered_samples[:20],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"total orders={total_orders} covered={total_covered} "
          f"ratio={report['overall_coverage_ratio']}")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
