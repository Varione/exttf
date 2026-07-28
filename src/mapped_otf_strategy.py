"""ETF-signal strategy executed through mapped OTC ETF feeder funds."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import load_price_series
from otf_backtest_engine import OTFBacktestEngine


ETF_DB = "data/processed/etf.sqlite"
OTF_DB = "data/processed/otf_mapped.sqlite"
REPORT_DIR = Path("reports/mapped_otf_strategy")


class MappedETFSignalBuilder:
    def __init__(self, etf_db: str = ETF_DB, otf_db: str = OTF_DB):
        self.etf_db = etf_db
        self.otf_db = otf_db
        self.mapping, self.nav_dates = self._load_mapping()
        prices = load_price_series(
            etf_db, data_mode="etf", price_mode="total_return_proxy"
        )
        symbols = set(self.mapping["etf_symbol"])
        prices = prices.loc[
            prices["symbol"].isin(symbols), ["date", "symbol", "price"]
        ]
        self.prices = prices.pivot(
            index="date", columns="symbol", values="price"
        ).sort_index()
        self.returns = self.prices.pct_change(fill_method=None)

    def _load_mapping(self) -> tuple[pd.DataFrame, dict[str, pd.DatetimeIndex]]:
        with sqlite3.connect(self.otf_db) as conn:
            catalog = pd.read_sql_query(
                """
                SELECT c.*, COUNT(n.nav_date) AS nav_rows,
                       MIN(n.nav_date) AS first_nav, MAX(n.nav_date) AS last_nav
                FROM otf_fund_catalog c
                JOIN otf_fund_nav n ON n.fund_code = c.fund_code
                GROUP BY c.fund_code
                """,
                conn,
            )
            nav = pd.read_sql_query(
                "SELECT fund_code, nav_date FROM otf_fund_nav "
                "ORDER BY fund_code, nav_date",
                conn,
            )
        catalog["fund_code"] = catalog["fund_code"].astype(str).str.zfill(6)
        catalog["etf_symbol"] = catalog["etf_symbol"].astype(str).str.zfill(6)
        catalog["first_nav"] = pd.to_datetime(catalog["first_nav"])
        catalog = catalog.sort_values(
            ["etf_symbol", "nav_rows", "first_nav", "fund_code"],
            ascending=[True, False, True, True],
        ).drop_duplicates("etf_symbol", keep="first")
        nav["fund_code"] = nav["fund_code"].astype(str).str.zfill(6)
        nav["nav_date"] = pd.to_datetime(nav["nav_date"])
        nav_dates = {
            code: pd.DatetimeIndex(group["nav_date"].sort_values())
            for code, group in nav.groupby("fund_code")
        }
        return catalog.reset_index(drop=True), nav_dates

    def _fund_has_history(
        self, fund_code: str, date: pd.Timestamp, minimum: int = 60
    ) -> bool:
        dates = self.nav_dates.get(fund_code)
        return (
            dates is not None
            and int(dates.searchsorted(date, side="right")) >= minimum
        )

    @staticmethod
    def _inverse_vol_weights(
        volatility: pd.Series, max_weight: float = 0.20
    ) -> dict[str, float]:
        inv = 1.0 / volatility.clip(lower=1e-6)
        raw = inv / inv.sum()
        # Residual stays in cash when the cap binds.
        return raw.clip(upper=max_weight).to_dict()

    def build_targets(
        self,
        otf_dates: pd.DatetimeIndex,
        *,
        start: str = "2018-01-01",
        end: str = "2026-07-17",
        rebalance_every: int = 20,
        n_hold: int = 10,
        max_per_class: int = 3,
        use_trend: bool = True,
    ) -> tuple[pd.DataFrame, dict[pd.Timestamp, pd.Timestamp], pd.DataFrame]:
        etf_dates = self.prices.index[
            (self.prices.index >= pd.Timestamp(start))
            & (self.prices.index <= pd.Timestamp(end))
        ]
        scheduled = etf_dates[::rebalance_every]
        mapping_by_symbol = self.mapping.set_index("etf_symbol")
        targets: list[dict] = []
        audits: list[dict] = []
        signal_dates: dict[pd.Timestamp, pd.Timestamp] = {}

        for signal_date in scheduled:
            location = self.prices.index.get_loc(signal_date)
            if location < 252:
                continue
            future_otf = otf_dates[otf_dates > signal_date]
            if future_otf.empty:
                continue
            submit_date = future_otf[0]
            current = self.prices.iloc[location]
            mom60 = current / self.prices.iloc[location - 60] - 1.0
            mom120 = current / self.prices.iloc[location - 120] - 1.0
            mom252 = current / self.prices.iloc[location - 252] - 1.0
            trend = current > self.prices.iloc[location - 199 : location + 1].mean()
            volatility = (
                self.returns.iloc[location - 59 : location + 1].std()
                * np.sqrt(252)
            )
            score = (
                0.50 * mom60 + 0.30 * mom120 + 0.20 * mom252
            ) / volatility
            candidates = pd.DataFrame(
                {"score": score, "volatility": volatility, "trend": trend}
            ).dropna()
            positive = candidates["score"].gt(0)
            candidates = candidates.loc[
                positive & candidates["trend"]
                if use_trend
                else positive
            ]
            candidates = candidates.join(
                mapping_by_symbol[
                    ["fund_code", "fund_name", "asset_class", "mapping_score"]
                ],
                how="inner",
            )
            candidates = candidates.loc[
                [
                    self._fund_has_history(code, submit_date)
                    for code in candidates["fund_code"]
                ]
            ].sort_values("score", ascending=False)

            selected_symbols: list[str] = []
            class_counts: dict[str, int] = {}
            for symbol, row in candidates.iterrows():
                asset_class = str(row["asset_class"])
                if class_counts.get(asset_class, 0) >= max_per_class:
                    continue
                selected_symbols.append(symbol)
                class_counts[asset_class] = class_counts.get(asset_class, 0) + 1
                if len(selected_symbols) >= n_hold:
                    break
            selected = candidates.loc[selected_symbols]
            weights = (
                self._inverse_vol_weights(selected["volatility"])
                if not selected.empty
                else {}
            )
            target = {
                selected.loc[symbol, "fund_code"]: float(weight)
                for symbol, weight in weights.items()
            }
            targets.append({"date": submit_date, **target})
            signal_dates[submit_date] = signal_date
            audits.append(
                {
                    "signal_date": signal_date,
                    "submit_date": submit_date,
                    "candidate_count": len(candidates),
                    "selected_count": len(selected),
                    "target_exposure": sum(target.values()),
                    "selected_etfs": ";".join(selected_symbols),
                    "selected_funds": ";".join(target),
                }
            )

        target_frame = pd.DataFrame(targets).fillna(0.0).set_index("date")
        target_frame = target_frame.reindex(
            columns=sorted(target_frame.columns), fill_value=0.0
        )
        return target_frame, signal_dates, pd.DataFrame(audits)


def run_backtest(
    *,
    start: str = "2018-01-01",
    end: str = "2026-07-17",
    rebalance_every: int = 20,
    n_hold: int = 10,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    engine = OTFBacktestEngine(
        OTF_DB,
        confirmation_days_subscribe=1,
        confirmation_days_redeem=1,
        settlement_days_redeem=1,
        subscription_fee_rate=0.001,
        redemption_fee_rate=0.0015,
    )
    builder = MappedETFSignalBuilder()
    targets, signal_dates, audit = builder.build_targets(
        engine._trading_dates,
        start=start,
        end=end,
        rebalance_every=rebalance_every,
        n_hold=n_hold,
    )
    daily = engine.run_backtest(
        targets,
        start=start,
        end=end,
        rebalance_every=1,
        signal_dates=signal_dates,
    )
    metrics = engine.calculate_metrics(daily)
    metrics.update(
        {
            "strategy": "ETF_Signal_MultiPeriod_Trend_OTF_Execution",
            "mapped_fund_count": len(builder.mapping),
            "mapped_etf_count": builder.mapping["etf_symbol"].nunique(),
            "rebalance_every_etf_days": rebalance_every,
            "n_hold": n_hold,
            "signal_to_submit_lag": "next OTF publication date",
            "pit_status": "PIT_PARTIAL",
        }
    )
    return daily, metrics, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-07-17")
    parser.add_argument("--rebalance-every", type=int, default=20)
    parser.add_argument("--n-hold", type=int, default=10)
    args = parser.parse_args()
    daily, metrics, audit = run_backtest(
        start=args.start,
        end=args.end,
        rebalance_every=args.rebalance_every,
        n_hold=args.n_hold,
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(REPORT_DIR / "daily.csv", index=False)
    audit.to_csv(REPORT_DIR / "signals.csv", index=False)
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
