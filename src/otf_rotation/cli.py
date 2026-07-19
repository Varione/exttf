from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import run_backtest, summary
from .adjustments import build_total_return_proxy, fetch_all_adjustments
from .data import fetch_all_etf_history, fetch_etf_prices, load_price_csv, read_universe
from .processing import build_database, build_research_universe, build_universe_snapshot
from .research import run_research_report


def _add_common_backtest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--defensive", default="BOND")


def main() -> None:
    parser = argparse.ArgumentParser(description="场外 ETF 联接基金轮动研究工具")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch-etf", help="用 AKShare 抓取场内 ETF 信号价格")
    fetch.add_argument("--config", default="config/universe.csv")
    fetch.add_argument("--start-date", default="20180101")
    fetch.add_argument("--end-date", default="20500101")
    fetch.add_argument("--output", default="data/raw/signal_prices.csv")

    fetch_all = sub.add_parser("fetch-all-etf", help="抓取当前可发现的全部 ETF 历史日线")
    fetch_all.add_argument("--start-date", default="20000101")
    fetch_all.add_argument("--end-date", default="20500101")
    fetch_all.add_argument("--output-dir", default="data/raw/all_etf")
    fetch_all.add_argument("--workers", type=int, default=4)
    fetch_all.add_argument("--retries", type=int, default=2)
    fetch_all.add_argument("--sleep-seconds", type=float, default=0.2)
    fetch_all.add_argument("--no-resume", action="store_true")
    fetch_all.add_argument("--refresh-catalog", action="store_true")

    process_all = sub.add_parser("process-all-etf", help="将全量 ETF 文件整理为 SQLite 并生成质量报告")
    process_all.add_argument("--raw-dir", default="data/raw/all_etf")
    process_all.add_argument("--database", default="data/processed/etf.sqlite")
    process_all.add_argument("--quality-csv", default="reports/all_etf_quality.csv")
    process_all.add_argument("--summary-json", default="reports/all_etf_quality_summary.json")

    snapshot = sub.add_parser("universe-snapshot", help="按质量和流动性条件生成可交易池快照")
    snapshot.add_argument("--quality-csv", default="reports/all_etf_quality.csv")
    snapshot.add_argument("--output", default="reports/strategy_universe_snapshot.csv")
    snapshot.add_argument("--min-history-rows", type=int, default=252)
    snapshot.add_argument("--max-stale-days", type=int, default=30)
    snapshot.add_argument("--min-median-amount-60d", type=float, default=10_000_000)

    small_universe = sub.add_parser("make-research-universe", help="生成小型分层研究池")
    small_universe.add_argument("--quality-csv", default="reports/all_etf_quality.csv")
    small_universe.add_argument("--output", default="config/research_universe_top200.csv")
    small_universe.add_argument("--per-category", type=int, default=50)
    small_universe.add_argument("--cash-count", type=int, default=5)

    research = sub.add_parser("research-all-etf", help="在全量 ETF 数据上运行动态宇宙策略研究")
    research.add_argument("--database", default="data/processed/etf.sqlite")
    research.add_argument("--output-dir", default="reports/core_strategy")
    research.add_argument("--min-history-rows", type=int, default=252)
    research.add_argument("--max-stale-days", type=int, default=30)
    research.add_argument("--min-median-amount-60d", type=float, default=10_000_000)
    research.add_argument("--top-n", type=int, default=10)
    research.add_argument("--max-per-category", type=int, default=3)
    research.add_argument("--single-cap", type=float, default=0.30)
    research.add_argument("--category-cap", type=float, default=0.40)
    research.add_argument("--cost-bps", type=float, default=15.0)
    research.add_argument("--price-mode", choices=["raw", "total-return"], default="total-return")
    research.add_argument("--universe-file", default="config/research_universe_top200.csv")
    research.add_argument("--start-date", default="2018-01-01")
    research.add_argument("--end-date", default="2026-07-17")
    research.add_argument("--trend-window", type=int, default=200)
    research.add_argument("--breadth-threshold", type=float, default=0.0)
    research.add_argument("--target-vol", type=float, default=0.12)
    research.add_argument("--stop-drawdown", type=float, default=0.0)
    research.add_argument("--stop-reentry-breadth", type=float, default=0.50)

    adjustments = sub.add_parser("fetch-adjustments", help="抓取新浪 ETF 分红/拆分因子")
    adjustments.add_argument("--catalog", default="data/raw/all_etf/etf_catalog.csv")
    adjustments.add_argument("--output-dir", default="data/raw/all_etf/adjustments")
    adjustments.add_argument("--workers", type=int, default=4)
    adjustments.add_argument("--retries", type=int, default=2)
    adjustments.add_argument("--sleep-seconds", type=float, default=0.2)
    adjustments.add_argument("--no-resume", action="store_true")

    proxy = sub.add_parser("build-total-return", help="用累计现金分红构建总回报价格代理")
    proxy.add_argument("--raw-dir", default="data/raw/all_etf")
    proxy.add_argument("--output-dir", default="data/processed/total_return")
    proxy.add_argument("--database", default="data/processed/etf.sqlite")

    backtest = sub.add_parser("backtest", help="运行周频轮动回测")
    backtest.add_argument("--prices", default="data/raw/signal_prices.csv")
    backtest.add_argument("--output-dir", default="reports")
    _add_common_backtest_args(backtest)

    demo = sub.add_parser("demo", help="生成模拟数据并跑通完整回测")
    demo.add_argument("--output-dir", default="reports")
    _add_common_backtest_args(demo)

    args = parser.parse_args()
    if args.command == "fetch-etf":
        universe = read_universe(args.config)
        prices = fetch_etf_prices(universe, args.start_date, args.end_date, args.output)
        print(f"已保存 {prices.shape[0]} 个交易日、{prices.shape[1]} 个资产: {args.output}")
        return

    if args.command == "fetch-all-etf":
        stats = fetch_all_etf_history(
            args.start_date,
            args.end_date,
            args.output_dir,
            workers=args.workers,
            retries=args.retries,
            sleep_seconds=args.sleep_seconds,
            resume=not args.no_resume,
            refresh_catalog=args.refresh_catalog,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    if args.command == "process-all-etf":
        result = build_database(args.raw_dir, args.database, args.quality_csv, args.summary_json)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "universe-snapshot":
        result = build_universe_snapshot(
            args.quality_csv,
            args.output,
            args.min_history_rows,
            args.max_stale_days,
            args.min_median_amount_60d,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "make-research-universe":
        result = build_research_universe(args.quality_csv, args.output, args.per_category, args.cash_count)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "research-all-etf":
        symbols = None
        if args.universe_file:
            universe = pd.read_csv(args.universe_file, dtype=str)
            symbols = universe["symbol"].astype(str).str.zfill(6).tolist()
        result = run_research_report(
            args.database,
            args.output_dir,
            price_mode=args.price_mode,
            symbols=symbols,
            start_date=args.start_date or None,
            end_date=args.end_date or None,
            min_history_rows=args.min_history_rows,
            max_stale_days=args.max_stale_days,
            min_median_amount_60d=args.min_median_amount_60d,
            top_n=args.top_n,
            max_per_category=args.max_per_category,
            single_cap=args.single_cap,
            category_cap=args.category_cap,
            cost_bps=args.cost_bps,
            trend_window=args.trend_window,
            breadth_threshold=args.breadth_threshold,
            target_vol=args.target_vol,
            stop_drawdown=args.stop_drawdown,
            stop_reentry_breadth=args.stop_reentry_breadth,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "fetch-adjustments":
        result = fetch_all_adjustments(
            args.catalog, args.output_dir, args.workers, args.retries, args.sleep_seconds, not args.no_resume
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "build-total-return":
        result = build_total_return_proxy(args.raw_dir, args.output_dir, args.database)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "demo":
        dates = pd.bdate_range("2018-01-01", "2026-07-17")
        rng = np.random.default_rng(42)
        assets = ["CSI300", "CSI500", "GOLD", "BOND"]
        shocks = rng.normal(0.00015, 0.012, (len(dates), len(assets)))
        shocks[:, 0] += 0.00010
        shocks[:, 3] += 0.00008
        prices = pd.DataFrame((1 + shocks).cumprod(axis=0) * 100, index=dates, columns=assets)
    else:
        prices = load_price_csv(args.prices)

    result = run_backtest(prices, args.defensive, args.top_n, args.cost_bps)
    result.daily.to_csv(output_dir / "daily.csv", index_label="date")
    result.orders.to_csv(output_dir / "orders.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary(result), f, ensure_ascii=False, indent=2)
    print(json.dumps(summary(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
