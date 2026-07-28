"""Pre-specified all-OTC allocation with diversified defensive sleeves."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from mapped_otf_strategy import MappedETFSignalBuilder, OTF_RESEARCH_DB
from otf_backtest_engine import OTFBacktestEngine
from strategy_validation import evaluate_stability, rolling_windows


OUTPUT = Path("reports/all_otf_allocation")
START = "2018-01-01"
END = "2026-07-17"

# Economic weights were specified before observing portfolio results.  Multiple
# money funds remain in the database for source robustness, but only one is
# used to avoid duplicating the same exposure.
DEFENSIVE_SLEEVES = {
    "260102": 0.15,  # money
    "006663": 0.25,  # ultra-short bond
    "001512": 0.25,  # 3-5Y government bond index
    "005839": 0.20,  # 1-3Y policy-bank bond index
    "000148": 0.15,  # high-grade credit
}
ZERO_FEE_RESEARCH_ASSUMPTION = {
    "260102": 0.0,
    "040003": 0.0,
    "217004": 0.0,
    "050003": 0.0,
    "202301": 0.0,
    "006663": 0.0,
}


def load_nav_dates(db_path: str | Path) -> dict[str, pd.DatetimeIndex]:
    with sqlite3.connect(db_path) as connection:
        nav = pd.read_sql_query(
            "SELECT fund_code,nav_date FROM otf_fund_nav ORDER BY fund_code,nav_date",
            connection,
        )
    nav["fund_code"] = nav["fund_code"].astype(str).str.zfill(6)
    nav["nav_date"] = pd.to_datetime(nav["nav_date"])
    return {
        code: pd.DatetimeIndex(group["nav_date"])
        for code, group in nav.groupby("fund_code")
    }


def add_defensive_residual(
    core_targets: pd.DataFrame,
    signal_dates: dict[pd.Timestamp, pd.Timestamp],
    nav_dates: dict[str, pd.DatetimeIndex],
    sleeves: dict[str, float] = DEFENSIVE_SLEEVES,
    minimum_history: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Allocate unused risk budget using only funds known before each signal."""
    rows: list[dict] = []
    audits: list[dict] = []
    for submit_date, core_row in core_targets.iterrows():
        signal_date = pd.Timestamp(signal_dates[submit_date])
        core = core_row[core_row > 1e-12].to_dict()
        core_exposure = min(1.0, float(sum(core.values())))
        eligible = {
            code: weight
            for code, weight in sleeves.items()
            if code in nav_dates
            and int(nav_dates[code].searchsorted(signal_date, side="left"))
            >= minimum_history
        }
        normalizer = sum(eligible.values())
        residual = max(0.0, 1.0 - core_exposure)
        defensive = (
            {code: residual * weight / normalizer for code, weight in eligible.items()}
            if normalizer > 0
            else {}
        )
        combined = dict(core)
        for code, weight in defensive.items():
            combined[code] = combined.get(code, 0.0) + weight
        rows.append({"date": submit_date, **combined})
        audits.append(
            {
                "signal_date": signal_date,
                "submit_date": submit_date,
                "core_exposure": core_exposure,
                "defensive_exposure": sum(defensive.values()),
                "eligible_defensive_funds": ";".join(sorted(eligible)),
                "used_nav_strictly_before_signal": True,
            }
        )
    targets = pd.DataFrame(rows).fillna(0.0).set_index("date")
    return targets.reindex(columns=sorted(targets.columns), fill_value=0.0), pd.DataFrame(audits)


def make_engine(db_path: str = OTF_RESEARCH_DB, cost_multiplier: float = 1.0) -> OTFBacktestEngine:
    zero = {code: rate * cost_multiplier for code, rate in ZERO_FEE_RESEARCH_ASSUMPTION.items()}
    return OTFBacktestEngine(
        db_path,
        confirmation_days_subscribe=1,
        confirmation_days_redeem=1,
        settlement_days_redeem=1,
        subscription_fee_rate=0.001 * cost_multiplier,
        redemption_fee_rate=0.0015 * cost_multiplier,
        minimum_trade_ratio=0.005,
        fund_subscription_fee_rates=zero,
        fund_redemption_fee_rates=zero,
    )


def run(db_path: str = OTF_RESEARCH_DB) -> pd.DataFrame:
    engine = make_engine(db_path)
    builder = MappedETFSignalBuilder(otf_db=db_path)
    core_targets, signal_dates, core_audit = builder.build_core_sleeve_targets(
        engine._trading_dates, start=START, end=END
    )
    nav_dates = load_nav_dates(db_path)
    diversified, defensive_audit = add_defensive_residual(
        core_targets, signal_dates, nav_dates
    )
    variants = {
        "Core_Cash_Residual": core_targets,
        "Core_ShortBond_Residual": builder.build_core_sleeve_targets(
            engine._trading_dates,
            start=START,
            end=END,
            defensive_fund_code="006663",
        )[0],
        "Core_Diversified_Defensive": diversified,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    stability_payload: list[dict] = []
    for name, targets in variants.items():
        daily = engine.run_backtest(
            targets,
            start=START,
            end=END,
            rebalance_every=1,
            signal_dates=signal_dates,
        )
        metrics = engine.calculate_metrics(daily)
        windows = rolling_windows(daily)
        stability = evaluate_stability(windows)
        rows.append({"variant": name, **metrics, **{k: v for k, v in stability.items() if k != "checks"}})
        stability_payload.append({"variant": name, **stability})
        daily.to_csv(OUTPUT / f"daily_{name}.csv", index=False)
        windows.to_csv(OUTPUT / f"rolling_{name}.csv", index=False)

    # One transparent fee stress: double all non-zero assumed transaction fees.
    stress_engine = make_engine(db_path, cost_multiplier=2.0)
    stress_daily = stress_engine.run_backtest(
        diversified,
        start=START,
        end=END,
        rebalance_every=1,
        signal_dates=signal_dates,
    )
    stress_metrics = stress_engine.calculate_metrics(stress_daily)
    rows.append({"variant": "Core_Diversified_Defensive_2xFees", **stress_metrics})
    stress_daily.to_csv(OUTPUT / "daily_Core_Diversified_Defensive_2xFees.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    core_audit.to_csv(OUTPUT / "core_signal_audit.csv", index=False)
    defensive_audit.to_csv(OUTPUT / "defensive_allocation_audit.csv", index=False)
    (OUTPUT / "validation.json").write_text(
        json.dumps(
            {
                "pit_status": "PIT_PARTIAL",
                "validation_status": "RETROSPECTIVE_NOT_PRISTINE_OOS",
                "strategy_pool": "438 mapped ETF feeders plus direct OTC defensive sleeves",
                "money_fund_duplicates_used_for_strategy": False,
                "interbank_cd_enabled": False,
                "interbank_cd_exclusion_reason": "seven-day holding rule not modeled",
                "fee_assumptions_are_product_verified": False,
                "stability": stability_payload,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    print(run().to_string(index=False))
