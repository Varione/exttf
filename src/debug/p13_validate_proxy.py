"""P1-3: total_return_proxy validation against external references.

Compares proxy daily returns vs reference daily returns.
Uses reference_value as primary reference (total-return when available, unit_nav otherwise).
Uses cumulative_nav where available for secondary comparison.
"""
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

DB_PATH = "data/processed/etf.sqlite"
REPORT_DIR = Path("reports/data_validation")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== LOAD DATA ====================
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
df["year"] = df["date"].dt.year
for col in ["total_return_proxy", "raw_close", "unit_nav", "cumulative_nav", "reference_value"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ==================== COMPUTE RETURNS ====================
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
df["proxy_return"] = df.groupby("symbol")["total_return_proxy"].pct_change()
df["ref_value_return"] = df.groupby("symbol")["reference_value"].pct_change()
df["unit_nav_return"] = df.groupby("symbol")["unit_nav"].pct_change()

# cumulative_nav return (only for symbols that have it)
df["cum_nav_return"] = df.groupby("symbol")["cumulative_nav"].pct_change()

# Drop first row per symbol
df_valid = df.dropna(subset=["proxy_return", "ref_value_return"]).copy()

# Filter extreme returns (>50% in a day is clearly a split/error)
mask_ok = (
    df_valid["proxy_return"].abs().le(0.50)
    & df_valid["ref_value_return"].abs().le(0.50)
)
df_clean = df_valid[mask_ok].copy()

print(f"Total joined rows: {len(df)}")
print(f"After return calc + filter: {len(df_clean)} ({len(df_clean)/len(df)*100:.1f}%)")
print(f"Symbols with data: {df_clean['symbol'].nunique()}")

# ==================== PER-SYMBOL ERROR METRICS ====================
results_ref = []  # reference_value comparison
results_cum = []  # cumulative_nav comparison (subset)

for symbol in sorted(df_clean["symbol"].unique()):
    sub = df_clean[df_clean["symbol"] == symbol].copy()
    
    # --- Primary: proxy vs reference_value ---
    diff = sub["proxy_return"] - sub["ref_value_return"]
    mae = diff.abs().mean()
    rmse = np.sqrt((diff ** 2).mean())
    max_single = diff.abs().max()
    
    # Cumulative returns over the period
    proxy_cum = (1 + sub["proxy_return"]).prod() - 1
    ref_cum = (1 + sub["ref_value_return"]).prod() - 1
    cum_bias = ((1 + proxy_cum) / (1 + ref_cum) - 1) * 100 if ref_cum > -1 else float("nan")
    
    period_str = f"{sub['date'].min().strftime('%Y-%m-%d')} to {sub['date'].max().strftime('%Y-%m-%d')}"
    sources = ";".join(sub["reference_source"].unique())
    is_official = int(sub["official_or_exchange"].iloc[0])
    
    results_ref.append({
        "symbol": symbol,
        "period": period_str,
        "n_obs": len(sub),
        "mae_daily_return_bps": mae * 10000,       # basis points
        "rmse_daily_return_bps": rmse * 10000,
        "cumulative_bias_pct": cum_bias,
        "max_single_day_error_bps": max_single * 10000,
        "proxy_cumulative_return_pct": proxy_cum * 100,
        "ref_cumulative_return_pct": ref_cum * 100,
        "reference_source": sources,
        "is_official": is_official,
    })
    
    # --- Secondary: proxy vs cumulative_nav (if available) ---
    sub_cum = sub.dropna(subset=["cum_nav_return"]).copy()
    if len(sub_cum) >= 20:
        diff_cum = sub_cum["proxy_return"] - sub_cum["cum_nav_return"]
        # Filter extreme
        sub_cum_ok = sub_cum[diff_cum.abs().le(0.50)]
        if len(sub_cum_ok) >= 10:
            mae_c = abs(diff_cum.mean())
            rmse_c = np.sqrt((sub_cum_ok["proxy_return"] - sub_cum_ok["cum_nav_return"]) ** 2).mean()
            max_s_c = (sub_cum_ok["proxy_return"] - sub_cum_ok["cum_nav_return"]).abs().max()
            
            proxy_c = (1 + sub_cum_ok["proxy_return"]).prod() - 1
            ref_c = (1 + sub_cum_ok["cum_nav_return"]).prod() - 1
            cum_b_c = ((1 + proxy_c) / (1 + ref_c) - 1) * 100 if ref_c > -1 else float("nan")
            
            results_cum.append({
                "symbol": symbol,
                "period": f"{sub_cum_ok['date'].min().strftime('%Y-%m-%d')} to {sub_cum_ok['date'].max().strftime('%Y-%m-%d')}",
                "n_obs": len(sub_cum_ok),
                "mae_daily_return_bps": mae_c * 10000,
                "rmse_daily_return_bps": rmse_c * 10000,
                "cumulative_bias_pct": cum_b_c,
                "max_single_day_error_bps": max_s_c * 10000,
            })

ref_df = pd.DataFrame(results_ref)
cum_df = pd.DataFrame(results_cum) if results_cum else pd.DataFrame()

print("\n" + "="*80)
print("PRIMARY: proxy vs reference_value (all 20 symbols)")
print("="*80)
cols_show = ["symbol", "n_obs", "mae_daily_return_bps", "rmse_daily_return_bps",
             "cumulative_bias_pct", "max_single_day_error_bps", "is_official"]
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
print(ref_df[cols_show].to_string(index=False))

if not cum_df.empty:
    print("\n" + "="*80)
    print("SECONDARY: proxy vs cumulative_nav (symbols with NAV data)")
    print("="*80)
    print(cum_df.to_string(index=False))

# ==================== SUMMARY STATISTICS ====================
print("\n" + "="*80)
print("SUMMARY STATISTICS (reference_value comparison)")
print("="*80)
print(f"MAE median:       {ref_df['mae_daily_return_bps'].median():.2f} bps ({ref_df['mae_daily_return_bps'].median()/100:.3f}%)")
print(f"MAE max:          {ref_df['mae_daily_return_bps'].max():.2f} bps ({ref_df.loc[ref_df['mae_daily_return_bps'].idxmax(), 'symbol']})")
print(f"RMSE median:      {ref_df['rmse_daily_return_bps'].median():.2f} bps ({ref_df['rmse_daily_return_bps'].median()/100:.3f}%)")
print(f"RMSE max:         {ref_df['rmse_daily_return_bps'].max():.2f} bps ({ref_df.loc[ref_df['rmse_daily_return_bps'].idxmax(), 'symbol']})")
print(f"Cumulative bias median:     {ref_df['cumulative_bias_pct'].median():.2f}%")
print(f"Cumulative bias max abs:    {ref_df['cumulative_bias_pct'].abs().max():.2f}% ({ref_df.loc[ref_df['cumulative_bias_pct'].abs().idxmax(), 'symbol']})")
print(f"Max single day error median:{ref_df['max_single_day_error_bps'].median():.2f} bps")

# Pass/fail count (MAE < 10bps = 0.1% threshold)
n_pass = (ref_df["mae_daily_return_bps"] < 10).sum()
print(f"\nPass (MAE < 10bps = 0.1%): {n_pass}/{len(ref_df)} symbols")
fail_symbols = ref_df[ref_df["mae_daily_return_bps"] >= 10][["symbol", "mae_daily_return_bps"]]
if len(fail_symbols) > 0:
    print("FAILING symbols:")
    print(fail_symbols.to_string(index=False))

# ==================== YEARLY DRIFT ANALYSIS ====================
drift_results = []
for symbol in sorted(df_clean["symbol"].unique()):
    sub = df_clean[df_clean["symbol"] == symbol].copy()
    for year in sorted(sub["year"].unique()):
        yr = sub[sub["year"] == year].copy()
        proxy_yr = (1 + yr["proxy_return"]).prod() - 1
        ref_yr = (1 + yr["ref_value_return"]).prod() - 1
        drift = ((1 + proxy_yr) / (1 + ref_yr) - 1) * 100 if ref_yr > -1 else float("nan")
        drift_results.append({
            "symbol": symbol,
            "year": year,
            "proxy_yearly_return_pct": proxy_yr * 100,
            "ref_yearly_return_pct": ref_yr * 100,
            "drift_pct": drift,
        })

drift_df = pd.DataFrame(drift_results)

# Check for time trend in drift (does error grow over time?)
print("\n" + "="*80)
print("YEARLY DRIFT ANALYSIS")
print("="*80)
trend_results = []
for symbol in sorted(df_clean["symbol"].unique()):
    sym_d = drift_df[drift_df["symbol"] == symbol][["year", "drift_pct"]].dropna()
    if len(sym_d) >= 3:
        corr = sym_d["year"].corr(sym_d["drift_pct"])
        max_abs = sym_d["drift_pct"].abs().max()
        trend_results.append({
            "symbol": symbol,
            "n_years": len(sym_d),
            "drift_year_corr": corr,
            "max_abs_drift_pct": max_abs,
        })

trend_df = pd.DataFrame(trend_results)
print(trend_df.to_string(index=False))

# Symbols with significant time trend (|corr| > 0.7)
bad_trend = trend_df[trend_df["drift_year_corr"].abs() > 0.7]
if len(bad_trend) > 0:
    print("\nWARNING: Symbols with significant drift trend over time:")
    print(bad_trend.to_string(index=False))

# ==================== DIVIDEND/SPLIT DATE CHECK ====================
print("\n" + "="*80)
print("DIVIDEND/SPLIT DATE CHECK")
print("="*80)

# Find dates where proxy_return and ref_return diverge significantly (>2%)
diverge = df_clean.copy()
diverge["diff_abs"] = (diverge["proxy_return"] - diverge["ref_value_return"]).abs()
big_diff = diverge[diverge["diff_abs"] > 0.02].copy()
print(f"Days with |proxy_ret - ref_ret| > 2%: {len(big_diff)}")

# Group by symbol and show top discrepancies
if len(big_diff) > 0:
    big_diff_sorted = big_diff.sort_values("diff_abs", ascending=False)
    print("\nTop 30 largest discrepancies:")
    print(big_diff_sorted[["symbol", "date", "proxy_return", "ref_value_return", "diff_abs"]]
          .head(30).to_string(index=False))
    
    # Check if these coincide with dividend/split events
    for _, row in big_diff_sorted.head(10).iterrows():
        sym = row["symbol"]
        d = row["date"]
        # Check cumulative_dividend change
        div_mask = (df["symbol"] == sym) & (df["date"] == d)
        if div_mask.any():
            div_info = df[div_mask]["cumulative_dividend"].iloc[0]
            prev_div = df[(df["symbol"] == sym) & (df["date"] < d)]["cumulative_dividend"].iloc[-1] if len(df[(df["symbol"] == sym) & (df["date"] < d)]) > 0 else 0
            split_info = df[div_mask]["split_factor"].iloc[0]
            print(f"  {sym} {d.date()}: div_change={div_info-prev_div:.4f}, split_factor={split_info}")

# ==================== SAVE REPORTS ====================
report_path = REPORT_DIR / "price_mode_validation.csv"
ref_df.to_csv(report_path, index=False, encoding="utf-8")
print(f"\nReport saved: {report_path}")

drift_report_path = REPORT_DIR / "yearly_drift.csv"
drift_df.to_csv(drift_report_path, index=False, encoding="utf-8")
print(f"Drift report saved: {drift_report_path}")

if not cum_df.empty:
    cum_report_path = REPORT_DIR / "cumulative_nav_validation.csv"
    cum_df.to_csv(cum_report_path, index=False, encoding="utf-8")
    print(f"Cumulative NAV report saved: {cum_report_path}")

trend_report_path = REPORT_DIR / "drift_trend.csv"
trend_df.to_csv(trend_report_path, index=False, encoding="utf-8")
print(f"Trend report saved: {trend_report_path}")

# ==================== CONCLUSION ====================
print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

mae_median_pct = ref_df["mae_daily_return_bps"].median() / 100
if mae_median_pct < 0.1:
    print("PASS: MAE median < 0.1% threshold")
else:
    print(f"FAIL: MAE median {mae_median_pct:.3f}% > 0.1% threshold")

# Check cumulative bias
if ref_df["cumulative_bias_pct"].abs().max() < 5:
    print("PASS: Cumulative bias within 5% for all symbols")
else:
    worst = ref_df.loc[ref_df["cumulative_bias_pct"].abs().idxmax()]
    print(f"WARNING: {worst['symbol']} has cumulative bias of {worst['cumulative_bias_pct']:.2f}%")

# Check drift trend
if len(bad_trend) > 0:
    print(f"WARNING: {len(bad_trend)} symbols show significant drift trend over time")
else:
    print("PASS: No significant drift trend detected")

# Final recommendation
n_pass_strict = (ref_df["mae_daily_return_bps"] < 10).sum()
n_pass_loose = (ref_df["mae_daily_return_bps"] < 20).sum()
print(f"\nRecommendation:")
print(f"  - Strict pass (MAE<10bps): {n_pass_strict}/20 symbols")
print(f"  - Loose pass (MAE<20bps):  {n_pass_loose}/20 symbols")
if n_pass_strict >= 15:
    print("  -> total_return_proxy is SUITABLE for backtesting (majority pass strict)")
elif n_pass_loose >= 15:
    print("  -> total_return_proxy is ACCEPTABLE with caveats (majority pass loose)")
else:
    print("  -> total_return_proxy needs IMPROVEMENT before full deployment")
