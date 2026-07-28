"""Run pre-specified robustness variants for mapped OTC execution."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mapped_otf_strategy import (
    MappedETFSignalBuilder,
    OTF_DB,
    OTF_RESEARCH_DB,
)
from otf_backtest_engine import OTFBacktestEngine


OUTPUT = Path("reports/mapped_otf_strategy/experiments")
START = "2018-01-01"
END = "2026-07-17"
EXPERIMENT_DB = OTF_RESEARCH_DB if Path(OTF_RESEARCH_DB).exists() else OTF_DB


def make_engine() -> OTFBacktestEngine:
    return OTFBacktestEngine(
        EXPERIMENT_DB,
        confirmation_days_subscribe=1,
        confirmation_days_redeem=1,
        settlement_days_redeem=1,
        subscription_fee_rate=0.001,
        redemption_fee_rate=0.0015,
        minimum_trade_ratio=0.005,
        fund_subscription_fee_rates={"006663": 0.0},
        fund_redemption_fee_rates={"006663": 0.0},
    )


def run() -> pd.DataFrame:
    engine = make_engine()
    builder = MappedETFSignalBuilder(otf_db=EXPERIMENT_DB)
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

    robust_targets, robust_signal_dates, robust_audit = builder.build_robust_targets(
        engine._trading_dates,
        start=START,
        end=END,
    )
    robust_daily = engine.run_backtest(
        robust_targets,
        start=START,
        end=END,
        rebalance_every=1,
        signal_dates=robust_signal_dates,
    )
    robust_metrics = engine.calculate_metrics(robust_daily)
    robust_metrics.update(
        {
            "variant": "Robust_Monthly_Ensemble",
            "rebalance_every": 0,
            "n_hold": 8,
            "use_trend": True,
            "signal_count": len(robust_audit),
            "average_selected": float(robust_audit["selected_count"].mean()),
            "pit_status": "PIT_PARTIAL",
        }
    )
    rows.append(robust_metrics)
    robust_daily.to_csv(
        OUTPUT / "daily_Robust_Monthly_Ensemble.csv", index=False
    )
    robust_audit.to_csv(
        OUTPUT / "signals_Robust_Monthly_Ensemble.csv", index=False
    )

    diversified_targets, diversified_signal_dates, diversified_audit = (
        builder.build_targets(
            engine._trading_dates,
            start=START,
            end=END,
            rebalance_every=20,
            n_hold=10,
            max_per_class=3,
            use_trend=True,
            deduplicate_exposure=True,
            max_pair_correlation=0.90,
            target_volatility=0.12,
        )
    )
    diversified_daily = engine.run_backtest(
        diversified_targets,
        start=START,
        end=END,
        rebalance_every=1,
        signal_dates=diversified_signal_dates,
    )
    diversified_metrics = engine.calculate_metrics(diversified_daily)
    diversified_metrics.update(
        {
            "variant": "Diversified_M20_Vol12",
            "rebalance_every": 20,
            "n_hold": 10,
            "use_trend": True,
            "signal_count": len(diversified_audit),
            "average_selected": float(
                diversified_audit["selected_count"].mean()
            ),
            "pit_status": "PIT_PARTIAL",
        }
    )
    rows.append(diversified_metrics)
    diversified_daily.to_csv(
        OUTPUT / "daily_Diversified_M20_Vol12.csv", index=False
    )
    diversified_audit.to_csv(
        OUTPUT / "signals_Diversified_M20_Vol12.csv", index=False
    )

    core_targets, core_signal_dates, core_audit = builder.build_core_sleeve_targets(
        engine._trading_dates,
        start=START,
        end=END,
    )
    core_daily = engine.run_backtest(
        core_targets,
        start=START,
        end=END,
        rebalance_every=1,
        signal_dates=core_signal_dates,
    )
    core_metrics = engine.calculate_metrics(core_daily)
    core_metrics.update(
        {
            "variant": "CoreSleeve_Monthly_Vol10",
            "rebalance_every": 0,
            "n_hold": int(core_audit["selected_count"].max()),
            "use_trend": True,
            "signal_count": len(core_audit),
            "average_selected": float(core_audit["selected_count"].mean()),
            "pit_status": "PIT_PARTIAL",
        }
    )
    rows.append(core_metrics)
    core_daily.to_csv(
        OUTPUT / "daily_CoreSleeve_Monthly_Vol10.csv", index=False
    )
    core_audit.to_csv(
        OUTPUT / "signals_CoreSleeve_Monthly_Vol10.csv", index=False
    )

    defensive_code = "006663" if "006663" in engine.available_fund_codes else None
    core_bond_targets, core_bond_signal_dates, core_bond_audit = (
        builder.build_core_sleeve_targets(
            engine._trading_dates,
            start=START,
            end=END,
            defensive_fund_code=defensive_code,
        )
    )
    if (
        defensive_code is not None
        and core_bond_targets.index.to_series().diff().dropna().dt.days.min() < 7
    ):
        raise RuntimeError("DEFENSIVE_C_SHARE_HOLDING_PERIOD_UNDER_7_DAYS")
    core_bond_daily = engine.run_backtest(
        core_bond_targets,
        start=START,
        end=END,
        rebalance_every=1,
        signal_dates=core_bond_signal_dates,
    )
    core_bond_metrics = engine.calculate_metrics(core_bond_daily)
    core_bond_metrics.update(
        {
            "variant": "CoreSleeve_Monthly_Vol10_ShortBond",
            "rebalance_every": 0,
            "n_hold": int(core_bond_audit["selected_count"].max()) + 1,
            "use_trend": True,
            "signal_count": len(core_bond_audit),
            "average_selected": float(
                core_bond_targets.gt(1e-12).sum(axis=1).mean()
            ),
            "pit_status": "PIT_PARTIAL",
        }
    )
    rows.append(core_bond_metrics)
    core_bond_daily.to_csv(
        OUTPUT / "daily_CoreSleeve_Monthly_Vol10_ShortBond.csv", index=False
    )
    core_bond_audit.to_csv(
        OUTPUT / "signals_CoreSleeve_Monthly_Vol10_ShortBond.csv", index=False
    )

    # A single pre-specified portfolio-of-strategies candidate: retain the
    # return-seeking baseline while limiting its concentration with a 40%
    # strategic core sleeve.  Targets are combined only on dates when either
    # component issues a new executable instruction.
    active_targets, active_signal_dates, _ = builder.build_targets(
        engine._trading_dates,
        start=START,
        end=END,
        rebalance_every=20,
        n_hold=10,
        use_trend=True,
    )
    blend_dates = active_targets.index.union(core_targets.index).sort_values()
    active_aligned = active_targets.reindex(blend_dates).ffill().fillna(0.0)
    core_aligned = core_targets.reindex(blend_dates).ffill().fillna(0.0)
    blend_columns = active_aligned.columns.union(core_aligned.columns)
    blend_targets = (
        0.60 * active_aligned.reindex(columns=blend_columns, fill_value=0.0)
        + 0.40 * core_aligned.reindex(columns=blend_columns, fill_value=0.0)
    )
    blend_signal_dates = {
        date: max(
            active_signal_dates.get(date, pd.Timestamp.min),
            core_signal_dates.get(date, pd.Timestamp.min),
        )
        for date in blend_dates
    }
    blend_daily = engine.run_backtest(
        blend_targets,
        start=START,
        end=END,
        rebalance_every=1,
        signal_dates=blend_signal_dates,
    )
    blend_metrics = engine.calculate_metrics(blend_daily)
    blend_metrics.update(
        {
            "variant": "Blend60Active40Core",
            "rebalance_every": 0,
            "n_hold": 0,
            "use_trend": True,
            "signal_count": len(blend_targets),
            "average_selected": float(
                blend_targets.gt(1e-12).sum(axis=1).mean()
            ),
            "pit_status": "PIT_PARTIAL",
        }
    )
    rows.append(blend_metrics)
    blend_daily.to_csv(
        OUTPUT / "daily_Blend60Active40Core.csv", index=False
    )
    blend_targets.to_csv(
        OUTPUT / "signals_Blend60Active40Core.csv", index=True
    )

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

    windows = [
        ("2018_2021", pd.Timestamp("2018-01-01"), pd.Timestamp("2021-12-31")),
        ("2022_2024", pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31")),
        ("2025_2026", pd.Timestamp("2025-01-01"), pd.Timestamp(END)),
    ]
    window_rows: list[dict] = []
    for variant in summary["variant"]:
        daily_path = OUTPUT / f"daily_{variant}.csv"
        daily_frame = pd.read_csv(daily_path, parse_dates=["date"])
        for window_name, window_start, window_end in windows:
            sample = daily_frame.loc[
                daily_frame["date"].between(window_start, window_end)
            ].copy()
            metrics = engine.calculate_metrics(sample)
            metrics.update({"variant": variant, "window": window_name})
            window_rows.append(metrics)
    pd.DataFrame(window_rows).to_csv(
        OUTPUT / "window_summary.csv", index=False
    )
    return summary


if __name__ == "__main__":
    result = run()
    columns = [
        "variant", "CAGR%", "Annualized_Volatility%", "Max_Drawdown%",
        "Sharpe", "Calmar", "Annualized_Cost_Drag%", "Exposure%",
    ]
    print(result[columns].to_string(index=False))
