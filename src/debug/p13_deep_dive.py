"""Deep dive: understand discrepancy patterns in total_return_proxy validation."""
import sqlite3
import numpy as np
import pandas as pd

DB_PATH = "data/processed/etf.sqlite"

with sqlite3.connect(DB_PATH) as conn:
    df = pd.read_sql_query(
        "SELECT pm.symbol, pm.date, pm.total_return_proxy, pm.raw_close,"
        " pm.cumulative_dividend, pm.split_factor, pm.factor_f,"
        " er.reference_value, er.unit_nav, er.cumulative_nav,"
        " er.reference_source, er.source_independent, er.official_or_exchange"
        " FROM etf_daily_price_modes pm"
        " JOIN etf_daily_external_reference er ON pm.symbol=er.symbol AND pm.date=er.date"
        " ORDER BY pm.symbol, pm.date",
        conn,
    )

df["date"] = pd.to_datetime(df["date"])
for col in ["total_return_proxy", "raw_close", "unit_nav", "cumulative_nav", "reference_value"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ==================== CHECK: reference_value vs unit_nav ====================
print("=== Checking if reference_value equals unit_nav per symbol ===")
for symbol in sorted(df["symbol"].unique()):
    sub = df[df["symbol"] == symbol]
    diff = (sub["reference_value"] - sub["unit_nav"]).abs()
    max_diff = diff.max()
    mean_diff = diff.mean()
    same = "SAME" if max_diff < 0.001 else f"DIFFER (max={max_diff:.4f}, mean={mean_diff:.4f})"
    print(f"  {symbol}: {same}")

# ==================== DEEP DIVE: Top discrepancy symbols ====================
print("\n=== DEEP DIVE: 159518 (worst MAE) ===")
sub = df[df["symbol"] == "159518"].copy()
sub["proxy_ret"] = sub["total_return_proxy"].pct_change()
sub["ref_ret"] = sub["reference_value"].pct_change()
sub["raw_ret"] = sub["raw_close"].pct_change()

# Show period info
print(f"Period: {sub['date'].min()} to {sub['date'].max()}")
print(f"Proxy range: {sub['total_return_proxy'].min():.4f} - {sub['total_return_proxy'].max():.4f}")
print(f"Ref range:   {sub['reference_value'].min():.4f} - {sub['reference_value'].max():.4f}")
print(f"Raw range:   {sub['raw_close'].min():.4f} - {sub['raw_close'].max():.4f}")

# Check April 2025
apr = sub[sub["date"].between("2025-04-01", "2025-04-15")]
print(f"\nApril 2025 data for 159518:")
print(apr[["date", "total_return_proxy", "raw_close", "reference_value", "unit_nav", "proxy_ret", "ref_ret"]].to_string(index=False))

# ==================== DEEP DIVE: 510880 (worst cumulative bias) ====================
print("\n=== DEEP DIVE: 510880 (worst cumulative bias) ===")
sub2 = df[df["symbol"] == "510880"].copy()
sub2["proxy_ret"] = sub2["total_return_proxy"].pct_change()
sub2["ref_ret"] = sub2["reference_value"].pct_change()

print(f"Period: {sub2['date'].min()} to {sub2['date'].max()}")
print(f"Proxy range: {sub2['total_return_proxy'].min():.4f} - {sub2['total_return_proxy'].max():.4f}")
print(f"Ref range:   {sub2['reference_value'].min():.4f} - {sub2['reference_value'].max():.4f}")

# Check if proxy and ref have different starting levels
print(f"\nFirst 5 rows:")
print(sub2.head()[["date", "total_return_proxy", "raw_close", "reference_value", "unit_nav"]].to_string(index=False))

# Cumulative returns by year
sub2["year"] = sub2["date"].dt.year
for year in sorted(sub2["year"].unique()):
    yr = sub2[sub2["year"] == year]
    p_cum = (1 + yr["proxy_ret"]).prod() - 1
    r_cum = (1 + yr["ref_ret"]).prod() - 1
    print(f"  {year}: proxy={p_cum*100:.2f}%, ref={r_cum*100:.2f}%, drift={((1+p_cum)/(1+r_cum)-1)*100:.2f}%")

# ==================== CHECK: raw_close vs unit_nav alignment ====================
print("\n=== Checking raw_close vs unit_nav alignment ===")
for symbol in sorted(df["symbol"].unique()):
    sub = df[df["symbol"] == symbol]
    ratio = sub["raw_close"] / sub["unit_nav"]
    print(f"  {symbol}: ratio range [{ratio.min():.4f}, {ratio.max():.4f}], median={ratio.median():.4f}")

# ==================== ALTERNATIVE: Compare proxy returns vs raw_close returns ====================
print("\n=== Alternative: proxy_return vs raw_close_return ===")
df["raw_ret"] = df.groupby("symbol")["raw_close"].pct_change()
df_alt = df.dropna(subset=["proxy_ret" if "proxy_ret" in df.columns else "total_return_proxy", "raw_ret"]).copy()

# Recompute proxy_ret properly
df["proxy_ret"] = df.groupby("symbol")["total_return_proxy"].pct_change()
df_alt = df.dropna(subset=["proxy_ret", "raw_ret"]).copy()

alt_results = []
for symbol in sorted(df_alt["symbol"].unique()):
    sub = df_alt[df_alt["symbol"] == symbol]
    diff = (sub["proxy_ret"] - sub["raw_ret"]).abs()
    mae = diff.mean() * 10000  # bps
    rmse = np.sqrt(((sub["proxy_ret"] - sub["raw_ret"]) ** 2).mean()) * 10000
    n_div = (diff > 0.01).sum()  # days with >1% difference (likely dividend)
    alt_results.append({
        "symbol": symbol,
        "mae_proxy_vs_raw_bps": mae,
        "rmse_proxy_vs_raw_bps": rmse,
        "n_div_days": n_div,
        "total_days": len(sub),
    })

alt_df = pd.DataFrame(alt_results)
print(alt_df.to_string(index=False))
print(f"\nMedian MAE proxy vs raw: {alt_df['mae_proxy_vs_raw_bps'].median():.2f} bps")
