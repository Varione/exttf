"""Build etf_lifecycle table and generate summary report."""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = "data/processed/etf.sqlite"


def build_lifecycle():
    conn = sqlite3.connect(DB_PATH)

    # Step 1: Aggregate etf_daily per symbol
    daily_agg = pd.read_sql_query(
        """
        SELECT
            symbol,
            MIN(date) AS list_date,
            MAX(date) AS last_date,
            COUNT(*) AS total_trading_days
        FROM etf_daily
        GROUP BY symbol
        """,
        conn,
    )

    # Step 2: Get catalog info
    catalog = pd.read_sql_query(
        "SELECT symbol, name FROM etf_catalog",
        conn,
    )

    # Step 3: Get quality info
    quality = pd.read_sql_query(
        """
        SELECT
            symbol,
            asset_class,
            rows_raw,
            rows_valid,
            duplicate_dates,
            null_close,
            non_positive_close,
            span_days,
            median_amount_60d,
            zero_volume_ratio_60d
        FROM etf_quality
        """,
        conn,
    )

    conn.close()

    # Merge on symbol (daily_agg already has list_date, last_date, total_trading_days)
    merged = daily_agg.merge(catalog, on="symbol", how="left")
    merged = merged.merge(quality, on="symbol", how="left")

    # Compute derived fields
    merged["list_date"] = pd.to_datetime(merged["list_date"])
    merged["last_date"] = pd.to_datetime(merged["last_date"])
    merged["first_year"] = merged["list_date"].dt.year

    cutoff = pd.Timestamp("2026-07-17")
    merged["is_active"] = merged["last_date"] >= cutoff

    # Data quality score: rows_valid / rows_raw (1.0 = perfect)
    merged["data_quality_score"] = (
        pd.to_numeric(merged["rows_valid"], errors="coerce")
        / pd.to_numeric(merged["rows_raw"], errors="coerce").replace(0, 1)
    )
    merged["data_quality_score"] = merged["data_quality_score"].clip(upper=1.0).round(4)
    # This builder starts from the current catalog and observed quote range.
    # It cannot prove coverage of products that disappeared before collection.
    merged["lifecycle_source"] = "observed_price_range"
    merged["source_independent"] = 0
    merged["universe_scope"] = "current_snapshot_only"

    # Fill missing name
    merged["name"] = merged["name"].fillna(merged["symbol"])

    # Fill missing asset_class
    merged["asset_class"] = merged["asset_class"].fillna("unknown")

    # Select final columns
    lifecycle = merged[
        [
            "symbol",
            "name",
            "asset_class",
            "list_date",
            "last_date",
            "is_active",
            "total_trading_days",
            "first_year",
            "data_quality_score",
            "lifecycle_source",
            "source_independent",
            "universe_scope",
        ]
    ].copy()

    lifecycle = lifecycle.sort_values("symbol").reset_index(drop=True)

    # Write to SQLite
    conn = sqlite3.connect(DB_PATH)
    lifecycle.to_sql("etf_lifecycle", conn, if_exists="replace", index=False)

    # Verify
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM etf_lifecycle")
    count = cursor.fetchone()[0]
    print(f"etf_lifecycle written: {count} rows")
    conn.close()

    return lifecycle


def generate_report(lifecycle: pd.DataFrame):
    out_dir = Path("reports/data_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Basic stats
    total = len(lifecycle)
    n_active = int(lifecycle["is_active"].sum())
    n_inactive = total - n_active

    stats = pd.DataFrame({
        "metric": [
            "total_etfs",
            "active_count",
            "inactive_count",
            "active_pct",
            "median_list_year",
            "min_list_date",
            "max_last_date",
            "median_trading_days",
            "median_quality_score",
        ],
        "value": [
            total,
            n_active,
            n_inactive,
            round(n_active / total * 100, 2),
            lifecycle["first_year"].median(),
            lifecycle["list_date"].min(),
            lifecycle["last_date"].max(),
            lifecycle["total_trading_days"].median(),
            lifecycle["data_quality_score"].median(),
        ],
    })

    stats_path = out_dir / "etf_lifecycle_summary.csv"
    stats.to_csv(stats_path, index=False)
    print(f"Summary report: {stats_path}")

    # Year distribution
    year_dist = (
        lifecycle.groupby("first_year")
        .size()
        .reset_index()
        .rename(columns={0: "count"})
        .sort_values("first_year")
    )
    year_path = out_dir / "etf_lifecycle_by_year.csv"
    year_dist.to_csv(year_path, index=False)
    print(f"Year distribution: {year_path}")

    # Quality score distribution
    quality_bins = [0.0, 0.5, 0.7, 0.8, 0.9, 1.0]
    quality_labels = ["<0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    lifecycle["quality_bin"] = pd.cut(
        lifecycle["data_quality_score"],
        bins=quality_bins,
        labels=quality_labels,
        include_lowest=True,
    )
    quality_dist = (
        lifecycle.groupby("quality_bin", observed=True)
        .size()
        .reset_index()
        .rename(columns={0: "count"})
    )
    quality_path = out_dir / "etf_lifecycle_quality_dist.csv"
    quality_dist.to_csv(quality_path, index=False)
    print(f"Quality distribution: {quality_path}")

    # Asset class distribution
    ac_dist = (
        lifecycle.groupby("asset_class", observed=True)
        .size()
        .reset_index()
        .rename(columns={0: "count"})
        .sort_values("count", ascending=False)
    )
    ac_path = out_dir / "etf_lifecycle_asset_class.csv"
    ac_dist.to_csv(ac_path, index=False)
    print(f"Asset class distribution: {ac_path}")

    return stats, year_dist, quality_dist, ac_dist


def print_summary(lifecycle: pd.DataFrame, stats, year_dist, quality_dist, ac_dist):
    print("\n=== ETF Lifecycle Summary ===")
    for _, row in stats.iterrows():
        print(f"  {row['metric']}: {row['value']}")

    print(f"\n=== Listing Year Distribution (top 10) ===")
    for _, row in year_dist.head(10).iterrows():
        print(f"  {int(row['first_year'])}: {row['count']}")

    print(f"\n=== Quality Score Distribution ===")
    for _, row in quality_dist.iterrows():
        print(f"  {row['quality_bin']}: {row['count']}")

    print(f"\n=== Asset Class Distribution (top 10) ===")
    for _, row in ac_dist.head(10).iterrows():
        print(f"  {row['asset_class']}: {row['count']}")


if __name__ == "__main__":
    lifecycle = build_lifecycle()
    stats, year_dist, quality_dist, ac_dist = generate_report(lifecycle)
    print_summary(lifecycle, stats, year_dist, quality_dist, ac_dist)
