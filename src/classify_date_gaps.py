import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings("ignore")

DB_PATH = r"D:/etf/data/processed/etf.sqlite"
PROCESSED_DIR = r"D:/etf/data/processed"
REPORTS_DIR = r"D:/etf/reports/data_validation"

os.makedirs(REPORTS_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)

print("=" * 60)
print("Phase 5.3: Date Gap Classification (v2)")
print("=" * 60)

# ============================================================
# Step 1: Infer Trading Calendar (bimodal method)
# ============================================================
print("\n[Step 1] Inferring trading calendar...")

daily_df = pd.read_sql_query("SELECT symbol, date FROM etf_daily", conn)
daily_df["date"] = pd.to_datetime(daily_df["date"])

dates_per_day = daily_df.groupby("date").size()
total_symbols = daily_df["symbol"].nunique()
print(f"  Total symbols: {total_symbols}")
print(f"  Date range: {dates_per_day.index.min()} ~ {dates_per_day.index.max()}")

# Bimodal approach: separate dates into clusters by count
# Non-trading days have 0 or very few ETFs
# Trading days have many ETFs (even early years had 5+)
weekday_mask = dates_per_day.index.dayofweek < 5
weekend_dates = dates_per_day.index[~weekday_mask]
weekday_dates = dates_per_day.index[weekday_mask]

weekday_counts = dates_per_day[weekday_mask]
print(f"\n  Weekday dates: {len(weekday_dates)}")
print(f"  Weekend dates: {len(weekend_dates)}")
print(f"  Weekday counts: min={weekday_counts.min()}, median={weekday_counts.median()}")

# Among weekdays, find the gap between "holiday" and "trading day"
# Holidays on weekdays typically have 0 ETFs (no trading)
# Use K-means style: split at the valley
zero_count_weekdays = weekday_counts[weekday_counts == 0].index.tolist()
low_count_weekdays = weekday_counts[(weekday_counts > 0) & (weekday_counts < 10)].index.tolist()
high_count_weekdays = weekday_counts[weekday_counts >= 10].index.tolist()

print(f"\n  Weekday with 0 ETFs (holidays): {len(zero_count_weekdays)}")
print(f"  Weekday with 1-9 ETFs: {len(low_count_weekdays)}")
print(f"  Weekday with 10+ ETFs: {len(high_count_weekdays)}")

# Trading days = weekdays with >= 10 ETFs
trading_days = sorted(high_count_weekdays)
non_trading_days = sorted(list(weekend_dates) + zero_count_weekdays + low_count_weekdays)

print(f"\n  Trading days: {len(trading_days)} (~{len(trading_days)/21:.0f}/year)")
print(f"  Non-trading days: {len(non_trading_days)}")

# Verify by year
years = pd.Series(trading_days).dt.year.value_counts().sort_index()
print(f"\n  Trading days per year:")
for y, c in years.items():
    print(f"    {y}: {c}")

trading_calendar = pd.DataFrame({
    "date": pd.date_range(daily_df["date"].min().date(), daily_df["date"].max().date()),
})
trading_calendar["is_trading_day"] = trading_calendar["date"].isin(trading_days).astype(int)

cal_path = os.path.join(PROCESSED_DIR, "trading_calendar.csv")
trading_calendar.to_csv(cal_path, index=False, encoding="utf-8-sig")
print(f"\n  Saved: {cal_path}")

# ============================================================
# Step 2: Load ETF quality info
# ============================================================
print("\n[Step 2] Loading ETF lifecycle info...")

quality_df = pd.read_sql_query("SELECT symbol, first_date, last_date FROM etf_quality", conn)
quality_df["first_date"] = pd.to_datetime(quality_df["first_date"])
quality_df["last_date"] = pd.to_datetime(quality_df["last_date"])
quality_map = quality_df.set_index("symbol")[["first_date", "last_date"]].to_dict("index")

# ============================================================
# Step 3: Analyze date gaps per ETF
# ============================================================
print("\n[Step 3] Analyzing date gaps per ETF...")

trading_set = set(trading_days)
symbols = daily_df["symbol"].unique()
total = len(symbols)

sym_dates_map = daily_df.groupby("symbol")["date"].apply(set).to_dict()

all_gaps = []
count = 0

for symbol in symbols:
    count += 1
    if count % 300 == 0:
        print(f"  Processing {count}/{total}...")

    sym_dates = sorted(sym_dates_map[symbol])
    sym_dates_set = set(sym_dates)

    info = quality_map.get(symbol, {})
    first_date = pd.Timestamp(info["first_date"]) if info else pd.Timestamp(sym_dates[0])
    last_date = pd.Timestamp(info["last_date"]) if info else pd.Timestamp(sym_dates[-1])

    expected_trading = sorted([d for d in trading_set
                               if first_date <= pd.Timestamp(d) <= last_date])
    missing_in_range = [d for d in expected_trading if d not in sym_dates_set]

    if not missing_in_range:
        continue

    consecutive_start = missing_in_range[0]
    consecutive_end = missing_in_range[0]

    for i in range(1, len(missing_in_range)):
        prev_date = pd.Timestamp(missing_in_range[i - 1])
        curr_date = pd.Timestamp(missing_in_range[i])

        if (curr_date - prev_date).days <= 2:
            consecutive_end = curr_date
        else:
            trading_days_in_gap = sum(1 for d in missing_in_range
                                       if consecutive_start <= d <= consecutive_end)
            all_gaps.append({
                "symbol": symbol,
                "gap_start": consecutive_start,
                "gap_end": consecutive_end,
                "gap_days": (consecutive_end - consecutive_start).days + 1,
                "trading_days_missing": trading_days_in_gap,
                "gap_type": "UNKNOWN",
                "is_suspicious": False,
            })
            consecutive_start = curr_date
            consecutive_end = curr_date

    trading_days_in_gap = sum(1 for d in missing_in_range
                               if consecutive_start <= d <= consecutive_end)
    all_gaps.append({
        "symbol": symbol,
        "gap_start": consecutive_start,
        "gap_end": consecutive_end,
        "gap_days": (consecutive_end - consecutive_start).days + 1,
        "trading_days_missing": trading_days_in_gap,
        "gap_type": "UNKNOWN",
        "is_suspicious": False,
    })

print(f"  Total gaps found: {len(all_gaps)}")

# ============================================================
# Step 4: Classify each gap
# ============================================================
print("\n[Step 4] Classifying gaps...")

dates_with_any_data = set(dates_per_day.index)

def classify_gap(gap, symbol):
    gap_start = pd.Timestamp(gap["gap_start"])
    gap_end = pd.Timestamp(gap["gap_end"])
    info = quality_map.get(symbol, {})
    first_date = pd.Timestamp(info["first_date"]) if info else gap_start
    last_date = pd.Timestamp(info["last_date"]) if info else gap_end

    if gap_end < first_date:
        return "PRE_LISTING"
    if gap_start > last_date:
        return "POST_DELISTING"

    mid_point = gap_start + (gap_end - gap_start) / 2
    check_dates = [gap_start, mid_point, gap_end]

    other_has_data = any(d in dates_with_any_data for d in check_dates)

    if other_has_data:
        return "SUSPENSION"
    else:
        return "DATA_MISSING"

for gap in all_gaps:
    gap["gap_type"] = classify_gap(gap, gap["symbol"])
    gap["is_suspicious"] = gap["gap_days"] > 30

gaps_df = pd.DataFrame(all_gaps)

if len(gaps_df) > 0:
    gaps_df["gap_start"] = gaps_df["gap_start"].dt.strftime("%Y-%m-%d")
    gaps_df["gap_end"] = gaps_df["gap_end"].dt.strftime("%Y-%m-%d")

gaps_path = os.path.join(PROCESSED_DIR, "date_gaps.csv")
gaps_df.to_csv(gaps_path, index=False, encoding="utf-8-sig")
print(f"  Saved: {gaps_path}")

# ============================================================
# Step 5: Summary report
# ============================================================
print("\n[Step 5] Generating summary report...")

summary_data = []

if len(gaps_df) > 0:
    type_counts = gaps_df.groupby("gap_type").agg(
        count=("symbol", "count"),
        total_gap_days=("gap_days", "sum"),
        avg_gap_days=("gap_days", "mean"),
        max_gap_days=("gap_days", "max"),
        min_gap_days=("gap_days", "min"),
        total_trading_days_missing=("trading_days_missing", "sum"),
    ).reset_index()
    type_counts.columns = ["gap_type", "count", "total_gap_days", "avg_gap_days",
                           "max_gap_days", "min_gap_days", "total_trading_days_missing"]

    suspicious_df = gaps_df[gaps_df["is_suspicious"]]

    backtest_start = pd.Timestamp("2018-01-01")
    backtest_end = pd.Timestamp("2026-07-17")

    in_backtest = gaps_df[
        (pd.to_datetime(gaps_df["gap_start"]) <= backtest_end) &
        (pd.to_datetime(gaps_df["gap_end"]) >= backtest_start)
    ]

    for _, row in type_counts.iterrows():
        bt_gaps = len(in_backtest[in_backtest["gap_type"] == row["gap_type"]])
        summary_data.append({
            "gap_type": row["gap_type"],
            "count": int(row["count"]),
            "total_gap_days": int(row["total_gap_days"]),
            "avg_gap_days": round(row["avg_gap_days"], 1),
            "max_gap_days": int(row["max_gap_days"]),
            "min_gap_days": int(row["min_gap_days"]),
            "total_trading_days_missing": int(row["total_trading_days_missing"]),
            "in_backtest_window": bt_gaps,
        })

    summary_df = pd.DataFrame(summary_data)
else:
    suspicious_df = pd.DataFrame()
    in_backtest = pd.DataFrame()
    summary_df = pd.DataFrame(columns=["gap_type", "count", "total_gap_days", "avg_gap_days",
                                        "max_gap_days", "min_gap_days", "total_trading_days_missing",
                                        "in_backtest_window"])

summary_path = os.path.join(REPORTS_DIR, "date_gap_summary.csv")
summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
print(f"  Saved: {summary_path}")

# ============================================================
# Print Results
# ============================================================
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

print(f"\nTrading calendar: {len(trading_days)} trading days")

if len(gaps_df) > 0:
    print(f"\nTotal gaps: {len(gaps_df)} across {gaps_df['symbol'].nunique()} ETFs")
    print(f"\nGap type distribution:")
    for _, row in summary_df.iterrows():
        print(f"  {row['gap_type']}:")
        print(f"    count={row['count']}, total_days={row['total_gap_days']}, "
              f"avg={row['avg_gap_days']}d, max={row['max_gap_days']}d")
        print(f"    trading_days_missing={row['total_trading_days_missing']}, "
              f"in_backtest={row['in_backtest_window']}")

    print(f"\nSuspicious gaps (>30 days): {len(suspicious_df)}")
    if len(suspicious_df) > 0:
        print("\nSuspicious ETFs (top 30):")
        for _, row in suspicious_df.head(30).iterrows():
            print(f"  {row['symbol']}: {row['gap_start']} -> {row['gap_end']} "
                  f"({int(row['gap_days'])}d, {row['gap_type']})")

    if len(in_backtest) > 0:
        affected_symbols_bt = in_backtest["symbol"].nunique()
        print(f"\nBacktest impact (2018-01-01 ~ 2026-07-17):")
        print(f"  Gaps overlapping: {len(in_backtest)}")
        print(f"  Unique ETFs affected: {affected_symbols_bt}/{total_symbols}")
        in_bt_types = in_backtest.groupby("gap_type").size()
        print("  By type:")
        for gtype, cnt in in_bt_types.items():
            print(f"    {gtype}: {cnt}")

        # Impact assessment
        bt_trading_days = len([d for d in trading_days
                               if backtest_start <= pd.Timestamp(d) <= backtest_end])
        total_missing_bt = in_backtest["trading_days_missing"].sum() if "trading_days_missing" in in_backtest.columns else 0
        print(f"\n  Backtest trading days: {bt_trading_days}")
        print(f"  Total missing trading days in backtest: {total_missing_bt}")
        print(f"  Coverage loss: {total_missing_bt / (bt_trading_days * affected_symbols_bt) * 100:.2f}%")
else:
    print("\nNo gaps found.")

conn.close()
print("\nDone.")
