"""ETF-signal strategy executed through mapped OTC ETF feeder funds."""

from __future__ import annotations

import argparse
import json
import re
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

    @staticmethod
    def _exposure_group(row: pd.Series) -> str:
        """Build an issuer-agnostic key for economically similar exposures."""
        text = str(row.get("underlying_name", "")) or str(row.get("etf_name", ""))
        text = re.sub(
            r"ETF|交易型开放式指数证券投资基金|联接基金|发起式|指数|基金",
            "",
            text,
            flags=re.I,
        )
        text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", text).upper()
        aliases = (
            ("黄金", "COMMODITY_GOLD"),
            ("白银", "COMMODITY_SILVER"),
            ("纳斯达克", "OVERSEAS_NASDAQ"),
            ("纳指", "OVERSEAS_NASDAQ"),
            ("标普500", "OVERSEAS_SP500"),
            ("恒生科技", "OVERSEAS_HSTECH"),
            ("恒生", "OVERSEAS_HSI"),
            ("沪深300", "CN_CSI300"),
            ("中证500", "CN_CSI500"),
            ("中证1000", "CN_CSI1000"),
            ("创业板", "CN_CHINEXT"),
            ("科创50", "CN_STAR50"),
        )
        for token, group in aliases:
            if token in text:
                return group
        return text or str(row.name)

    @staticmethod
    def _portfolio_volatility(
        returns: pd.DataFrame, weights: pd.Series
    ) -> float:
        aligned = returns.reindex(columns=weights.index).dropna(how="all")
        covariance = aligned.cov(min_periods=max(20, len(aligned) // 2)) * 252
        covariance = covariance.fillna(0.0)
        value = float(weights.to_numpy() @ covariance.to_numpy() @ weights.to_numpy())
        return float(np.sqrt(max(0.0, value)))

    def build_robust_targets(
        self,
        otf_dates: pd.DatetimeIndex,
        *,
        start: str = "2018-01-01",
        end: str = "2026-07-17",
        n_hold: int = 8,
        max_per_class: int = 2,
        max_pair_correlation: float = 0.85,
        target_volatility: float = 0.10,
        max_weight: float = 0.20,
    ) -> tuple[pd.DataFrame, dict[pd.Timestamp, pd.Timestamp], pd.DataFrame]:
        """Monthly ensemble momentum with exposure and correlation controls."""
        eligible_dates = self.prices.index[
            (self.prices.index >= pd.Timestamp(start))
            & (self.prices.index <= pd.Timestamp(end))
        ]
        scheduled = (
            pd.Series(eligible_dates, index=eligible_dates)
            .groupby(eligible_dates.to_period("M"))
            .max()
            .tolist()
        )
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
            ret63 = current / self.prices.iloc[location - 63] - 1.0
            ret126 = current / self.prices.iloc[location - 126] - 1.0
            ret252 = current / self.prices.iloc[location - 252] - 1.0
            trend = current > self.prices.iloc[location - 199 : location + 1].mean()
            volatility = (
                self.returns.iloc[location - 62 : location + 1].std()
                * np.sqrt(252)
            )
            positive_horizons = (
                ret63.gt(0).astype(int)
                + ret126.gt(0).astype(int)
                + ret252.gt(0).astype(int)
            )
            score = pd.concat(
                [
                    ret63.rank(pct=True),
                    ret126.rank(pct=True),
                    ret252.rank(pct=True),
                ],
                axis=1,
            ).mean(axis=1)
            candidates = pd.DataFrame(
                {
                    "score": score,
                    "volatility": volatility,
                    "trend": trend,
                    "positive_horizons": positive_horizons,
                }
            ).dropna()
            candidates = candidates.loc[
                candidates["trend"] & candidates["positive_horizons"].ge(2)
            ].join(
                mapping_by_symbol[
                    [
                        "fund_code", "fund_name", "etf_name", "underlying_name",
                        "asset_class", "mapping_score",
                    ]
                ],
                how="inner",
            )
            candidates = candidates.loc[
                [
                    self._fund_has_history(code, submit_date)
                    for code in candidates["fund_code"]
                ]
            ].sort_values("score", ascending=False)
            candidates["exposure_group"] = candidates.apply(
                self._exposure_group, axis=1
            )

            correlation = self.returns.iloc[
                location - 125 : location + 1
            ].corr(min_periods=60)
            selected_symbols: list[str] = []
            selected_groups: set[str] = set()
            class_counts: dict[str, int] = {}
            correlation_rejections = 0
            duplicate_rejections = 0
            for symbol, row in candidates.iterrows():
                asset_class = str(row["asset_class"])
                group = str(row["exposure_group"])
                if group in selected_groups:
                    duplicate_rejections += 1
                    continue
                if class_counts.get(asset_class, 0) >= max_per_class:
                    continue
                if selected_symbols:
                    pairwise = correlation.loc[symbol, selected_symbols].abs().dropna()
                    if not pairwise.empty and float(pairwise.max()) >= max_pair_correlation:
                        correlation_rejections += 1
                        continue
                selected_symbols.append(symbol)
                selected_groups.add(group)
                class_counts[asset_class] = class_counts.get(asset_class, 0) + 1
                if len(selected_symbols) >= n_hold:
                    break

            selected = candidates.loc[selected_symbols]
            if selected.empty:
                target: dict[str, float] = {}
                ex_ante_volatility = 0.0
            else:
                base = 1.0 / selected["volatility"].clip(lower=1e-6)
                base = (base / base.sum()).clip(upper=max_weight)
                trailing = self.returns.iloc[location - 62 : location + 1]
                ex_ante_volatility = self._portfolio_volatility(trailing, base)
                scale = (
                    min(1.0, target_volatility / ex_ante_volatility)
                    if ex_ante_volatility > 0
                    else 0.0
                )
                weights = (base * scale).clip(upper=max_weight)
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
                    "ex_ante_volatility": ex_ante_volatility,
                    "duplicate_rejections": duplicate_rejections,
                    "correlation_rejections": correlation_rejections,
                    "selected_etfs": ";".join(selected_symbols),
                    "selected_funds": ";".join(target),
                }
            )

        target_frame = pd.DataFrame(targets).fillna(0.0).set_index("date")
        target_frame = target_frame.reindex(
            columns=sorted(target_frame.columns), fill_value=0.0
        )
        return target_frame, signal_dates, pd.DataFrame(audits)

    def _select_core_sleeves(self) -> pd.DataFrame:
        """Choose oldest mapped signal proxy for each pre-defined asset sleeve."""
        mapping = self.mapping.copy()
        text = (
            mapping["etf_name"].fillna("")
            + mapping["underlying_name"].fillna("")
        )
        definitions = {
            "CSI300": text.str.contains("沪深300") & ~text.str.contains("红利"),
            "CSI500": text.str.contains("中证500"),
            "CSI1000": text.str.contains("中证1000"),
            "CHINEXT": text.str.contains("创业板") & ~text.str.contains("红利"),
            "DIVIDEND": text.str.contains("红利")
            & mapping["asset_class"].eq("equity_dividend"),
            "GOLD": text.str.contains("黄金"),
            "NASDAQ": text.str.contains("纳斯达克|纳指", regex=True),
            "SP500": text.str.contains("标普500"),
            "HANGSENG": text.str.contains("恒生")
            & ~text.str.contains("科技|红利|消费|医药"),
        }
        rows: list[pd.Series] = []
        for sleeve, mask in definitions.items():
            candidates = mapping.loc[mask].copy()
            if candidates.empty:
                continue
            candidates["first_signal_date"] = candidates["etf_symbol"].map(
                lambda symbol: self.prices[symbol].first_valid_index()
                if symbol in self.prices
                else pd.NaT
            )
            candidates = candidates.dropna(subset=["first_signal_date"]).sort_values(
                ["first_signal_date", "etf_symbol", "fund_code"]
            )
            if candidates.empty:
                continue
            chosen = candidates.iloc[0].copy()
            chosen["sleeve"] = sleeve
            rows.append(chosen)
        return pd.DataFrame(rows).reset_index(drop=True)

    def build_core_sleeve_targets(
        self,
        otf_dates: pd.DatetimeIndex,
        *,
        start: str = "2018-01-01",
        end: str = "2026-07-17",
        target_volatility: float = 0.10,
        max_weight: float = 0.25,
    ) -> tuple[pd.DataFrame, dict[pd.Timestamp, pd.Timestamp], pd.DataFrame]:
        """Strategic core sleeves with absolute momentum and monthly trading."""
        sleeves = self._select_core_sleeves().set_index("etf_symbol")
        eligible_dates = self.prices.index[
            (self.prices.index >= pd.Timestamp(start))
            & (self.prices.index <= pd.Timestamp(end))
        ]
        scheduled = (
            pd.Series(eligible_dates, index=eligible_dates)
            .groupby(eligible_dates.to_period("M"))
            .max()
            .tolist()
        )
        targets: list[dict] = []
        audits: list[dict] = []
        signal_dates: dict[pd.Timestamp, pd.Timestamp] = {}
        symbols = sleeves.index.intersection(self.prices.columns)

        for signal_date in scheduled:
            location = self.prices.index.get_loc(signal_date)
            if location < 252:
                continue
            future_otf = otf_dates[otf_dates > signal_date]
            if future_otf.empty:
                continue
            submit_date = future_otf[0]
            current = self.prices.loc[signal_date, symbols]
            ret126 = current / self.prices.iloc[location - 126][symbols] - 1.0
            ret252 = current / self.prices.iloc[location - 252][symbols] - 1.0
            trend = current > self.prices.iloc[
                location - 199 : location + 1
            ][symbols].mean()
            volatility = self.returns.iloc[
                location - 62 : location + 1
            ][symbols].std() * np.sqrt(252)
            score = 0.5 * ret126 + 0.5 * ret252
            candidates = pd.DataFrame(
                {"score": score, "volatility": volatility, "trend": trend}
            ).dropna()
            candidates = candidates.loc[candidates["trend"] & candidates["score"].gt(0)]
            candidates = candidates.join(
                sleeves[["fund_code", "fund_name", "sleeve"]], how="inner"
            )
            candidates = candidates.loc[
                [
                    self._fund_has_history(code, submit_date)
                    for code in candidates["fund_code"]
                ]
            ]
            if candidates.empty:
                target: dict[str, float] = {}
                ex_ante_volatility = 0.0
            else:
                base = 1.0 / candidates["volatility"].clip(lower=1e-6)
                base = (base / base.sum()).clip(upper=max_weight)
                trailing = self.returns.iloc[location - 62 : location + 1]
                ex_ante_volatility = self._portfolio_volatility(trailing, base)
                scale = (
                    min(1.0, target_volatility / ex_ante_volatility)
                    if ex_ante_volatility > 0
                    else 0.0
                )
                weights = (base * scale).clip(upper=max_weight)
                target = {
                    candidates.loc[symbol, "fund_code"]: float(weight)
                    for symbol, weight in weights.items()
                }
            targets.append({"date": submit_date, **target})
            signal_dates[submit_date] = signal_date
            audits.append(
                {
                    "signal_date": signal_date,
                    "submit_date": submit_date,
                    "selected_count": len(candidates),
                    "target_exposure": sum(target.values()),
                    "ex_ante_volatility": ex_ante_volatility,
                    "selected_etfs": ";".join(candidates.index),
                    "selected_sleeves": ";".join(candidates["sleeve"]),
                    "selected_funds": ";".join(target),
                }
            )

        target_frame = pd.DataFrame(targets).fillna(0.0).set_index("date")
        target_frame = target_frame.reindex(
            columns=sorted(target_frame.columns), fill_value=0.0
        )
        return target_frame, signal_dates, pd.DataFrame(audits)

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
        deduplicate_exposure: bool = False,
        max_pair_correlation: float | None = None,
        target_volatility: float | None = None,
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
                    [
                        "fund_code", "fund_name", "etf_name", "underlying_name",
                        "asset_class", "mapping_score",
                    ]
                ],
                how="inner",
            )
            candidates = candidates.loc[
                [
                    self._fund_has_history(code, submit_date)
                    for code in candidates["fund_code"]
                ]
            ].sort_values("score", ascending=False)

            if deduplicate_exposure:
                candidates["exposure_group"] = candidates.apply(
                    self._exposure_group, axis=1
                )
            correlation = (
                self.returns.iloc[location - 125 : location + 1].corr(
                    min_periods=60
                )
                if max_pair_correlation is not None
                else None
            )

            selected_symbols: list[str] = []
            selected_groups: set[str] = set()
            class_counts: dict[str, int] = {}
            duplicate_rejections = 0
            correlation_rejections = 0
            for symbol, row in candidates.iterrows():
                asset_class = str(row["asset_class"])
                if deduplicate_exposure:
                    group = str(row["exposure_group"])
                    if group in selected_groups:
                        duplicate_rejections += 1
                        continue
                if class_counts.get(asset_class, 0) >= max_per_class:
                    continue
                if correlation is not None and selected_symbols:
                    pairwise = correlation.loc[symbol, selected_symbols].abs().dropna()
                    if (
                        not pairwise.empty
                        and float(pairwise.max()) >= float(max_pair_correlation)
                    ):
                        correlation_rejections += 1
                        continue
                selected_symbols.append(symbol)
                if deduplicate_exposure:
                    selected_groups.add(str(row["exposure_group"]))
                class_counts[asset_class] = class_counts.get(asset_class, 0) + 1
                if len(selected_symbols) >= n_hold:
                    break
            selected = candidates.loc[selected_symbols]
            ex_ante_volatility = 0.0
            if selected.empty:
                weights: dict[str, float] = {}
            else:
                weights = self._inverse_vol_weights(selected["volatility"])
                if target_volatility is not None:
                    weight_series = pd.Series(weights)
                    trailing = self.returns.iloc[location - 59 : location + 1]
                    ex_ante_volatility = self._portfolio_volatility(
                        trailing, weight_series
                    )
                    scale = (
                        min(1.0, target_volatility / ex_ante_volatility)
                        if ex_ante_volatility > 0
                        else 0.0
                    )
                    weights = (weight_series * scale).to_dict()
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
                    "ex_ante_volatility": ex_ante_volatility,
                    "duplicate_rejections": duplicate_rejections,
                    "correlation_rejections": correlation_rejections,
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
