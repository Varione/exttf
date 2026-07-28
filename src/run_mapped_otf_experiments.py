"""Run pre-specified robustness variants for mapped OTC execution."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mapped_otf_strategy import MappedETFSignalBuilder, OTF_DB
from otf_backtest_engine import OTFBacktestEngine


OUTPUT = Path("reports/mapped_otf_strategy/experiments")
START = "2018-01-01"
END = "2026-07-17"


def make_engine() -> OTFBacktestEngine:
    return OTFBacktestEngine(
        OTF_DB,
        confirmation_days_subscribe=1,
        confirmation_days_redeem=1,
        settlement_days_redeem=1,
        subscription_fee_rate=0.001,
        redemption_fee_rate=0.0015,
    )


def run() -> pd.DataFrame:
    engine = make_engine()
    builder = MappedETFSignalBuilder()
    variants = [
        ("M20_Top10_Trend", 20, 10, True),
        ("M10_Top10_Trend", 10, 10, True),
        ("M40_Top10_Trend", 40, 10, True),
        ("M20_Top5_Trend", 20, 5, True),
        ("M20_Top10_NoTrend", 20, 10, False),
    ]
    rows: list[dict] = []
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, frequency, n_hold, use_trend in variants:
        targets, signal_dates, audit = builder.build_targets(
            engine._trading_dates,
            start=START,
            end=END,
            rebalance_every=frequency,
            n_hold=n_hold,
            use_trend=use_trend,
        )
        daily = engine.run_backtest(
            targets,
            start=START,
            end=END,
            rebalance_every=1,
            signal_dates=signal_dates,
        )
        metrics = engine.calculate_metrics(daily)
        metrics.update(
            {
                "variant": name,
                "rebalance_every": frequency,
                "n_hold": n_hold,
                "use_trend": use_trend,
                "signal_count": len(audit),
                "average_selected": float(audit["selected_count"].mean()),
                "pit_status": "PIT_PARTIAL",
            }
        )
        rows.append(metrics)
        daily.to_csv(OUTPUT / f"daily_{name}.csv", index=False)
        audit.to_csv(OUTPUT / f"signals_{name}.csv", index=False)

    # Static mapped-universe benchmark, restricted to funds with at least
    # 252 published NAVs by the first order date.
    first_order = engine._trading_dates[engine._trading_dates > pd.Timestamp(START)][0]
    eligible = [
        row.fund_code
        for row in builder.mapping.itertuples(index=False)
        if builder._fund_has_history(row.fund_code, first_order, minimum=252)
    ]
    weight = 1.0 / len(eligible)
    target = pd.DataFrame(
        [{"date": first_order, **{code: weight for code in eligible}}]
    ).set_index("date")
    daily = engine.run_backtest(
        target,
        start=START,
        end=END,
        rebalance_every=1,
        signal_dates={first_order: pd.Timestamp(START)},
    )
    metrics = engine.calculate_metrics(daily)
    metrics.update(
        {
            "variant": "Mapped_EW_BuyHold",
            "rebalance_every": 0,
            "n_hold": len(eligible),
            "use_trend": False,
            "signal_count": 1,
            "average_selected": len(eligible),
            "pit_status": "PIT_PARTIAL",
        }
    )
    rows.append(metrics)
    daily.to_csv(OUTPUT / "daily_Mapped_EW_BuyHold.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    (OUTPUT / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    result = run()
    columns = [
        "variant", "CAGR%", "Annualized_Volatility%", "Max_Drawdown%",
        "Sharpe", "Calmar", "Annualized_Cost_Drag%", "Exposure%",
    ]
    print(result[columns].to_string(index=False))
