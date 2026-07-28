"""P1-3 FINAL: total_return_proxy validation - comprehensive analysis.

KEY FINDING: total_return_proxy == raw_close for all observed data points.
This means the proxy does NOT adjust for dividends despite its name.
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
        " pm.cumulative_dividend, pm.split_factor, pm.factor_f, pm.hfq_reference,"
        " er.reference_value, er.unit_nav, er.cumulative_nav,"
        " er.reference_source, er.source_independent, er.official_or_exchange"
        " FROM etf_daily_price_modes pm"
        " JOIN etf_daily_external_reference er ON pm.symbol=er.symbol AND pm.date=er.date"
        " ORDER BY pm.symbol, pm.date",
        conn,
    )

df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
for col in ["total_return_proxy", "raw_close", "unit_nav", "cumulative_nav",
             "reference_value", "cumulative_dividend", "split_factor", "factor_f"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ==================== CRITICAL CHECK: proxy == raw_close? ====================
df["proxy_raw_diff"] = (df["total_return_proxy"] - df["raw_close"]).abs()
max_diff = df.groupby("symbol")["proxy_raw_diff"].max()
print("=== MAX |total_return_proxy - raw_close| per symbol ===")
for s, v in max_diff.items():
    status = "IDENTICAL" if v < 0.0001 else f"DIFFERS ({v:.4f})"
    print(f"  {s}: {status}")

# ==================== COMPUTE RETURNS ====================
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
df["proxy_return"] = df.groupby("symbol")["total_return_proxy"].pct_change()
df["raw_return"] = df.groupby("symbol")["raw_close"].pct_change()
df["ref_return"] = df.groupby("symbol")["reference_value"].pct_change()
df["cum_nav_return"] = df.groupby("symbol")["cumulative_nav"].pct_change()

df_valid = df.dropna(subset=["proxy_return", "ref_return"]).copy()
mask_ok = (
    df_valid["proxy_return"].abs().le(0.50)
    & df_valid["ref_return"].abs().le(0.50)
)
df_clean = df_valid[mask_ok].copy()

# ==================== PER-SYMBOL ERROR METRICS ====================
results = []
for symbol in sorted(df_clean["symbol"].unique()):
    sub = df_clean[df_clean["symbol"] == symbol].copy()
    
    # proxy vs reference_value (daily returns)
    diff = sub["proxy_return"] - sub["ref_return"]
    mae = diff.abs().mean() * 10000  # bps
    rmse = np.sqrt((diff ** 2).mean()) * 10000
    max_single = diff.abs().max() * 10000
    
    # Cumulative returns
    proxy_cum = (1 + sub["proxy_return"]).prod() - 1
    ref_cum = (1 + sub["ref_return"]).prod() - 1
    cum_bias = ((1 + proxy_cum) / (1 + ref_cum) - 1) * 100 if ref_cum > -1 else float("nan")
    
    # Check if proxy == raw for this symbol
    proxy_raw_diff = (sub["total_return_proxy"] - sub["raw_close"]).abs().max()
    
    # Max cumulative dividend during period
    max_cum_div = sub["cumulative_dividend"].max()
    
    period_str = f"{sub['date'].min().strftime('%Y-%m-%d')} to {sub['date'].max().strftime('%Y-%m-%d')}"
    sources = ";".join(sub["reference_source"].unique())
    is_official = int(sub["official_or_exchange"].iloc[0])
    
    # Check if reference_value == unit_nav (i.e., NOT total-return adjusted)
    ref_unit_diff = (sub["reference_value"] - sub["unit_nav"]).abs().max()
    ref_is_total_return = ref_unit_diff > 0.01
    
    results.append({
        "symbol": symbol,
        "period": period_str,
        "n_obs": len(sub),
        "mae_daily_return_bps": mae,
        "rmse_daily_return_bps": rmse,
        "cumulative_bias_pct": cum_bias,
        "max_single_day_error_bps": max_single,
        "proxy_cumulative_return_pct": proxy_cum * 100,
        "ref_cumulative_return_pct": ref_cum * 100,
        "reference_source": sources,
        "is_official": is_official,
        "proxy_equals_raw": int(proxy_raw_diff < 0.0001),
        "max_cumulative_dividend": max_cum_div,
        "ref_is_total_return": int(ref_is_total_return),
    })

res_df = pd.DataFrame(results)

print("\n" + "="*90)
print("PER-SYMBOL ERROR METRICS (proxy vs reference_value daily returns)")
print("="*90)
cols = ["symbol", "mae_daily_return_bps", "rmse_daily_return_bps", 
        "cumulative_bias_pct", "max_single_day_error_bps",
        "proxy_equals_raw", "max_cumulative_dividend", "ref_is_total_return"]
print(res_df[cols].to_string(index=False))

# ==================== SUMMARY ====================
print("\n" + "="*90)
print("SUMMARY STATISTICS")
print("="*90)
print(f"MAE median:       {res_df['mae_daily_return_bps'].median():.2f} bps ({res_df['mae_daily_return_bps'].median()/100:.3f}%)")
print(f"MAE max:          {res_df['mae_daily_return_bps'].max():.2f} bps ({res_df.loc[res_df['mae_daily_return_bps'].idxmax(), 'symbol']})")
print(f"RMSE median:      {res_df['rmse_daily_return_bps'].median():.2f} bps ({res_df['rmse_daily_return_bps'].median()/100:.3f}%)")
print(f"RMSE max:         {res_df['rmse_daily_return_bps'].max():.2f} bps ({res_df.loc[res_df['rmse_daily_return_bps'].idxmax(), 'symbol']})")
print(f"Cumulative bias median:     {res_df['cumulative_bias_pct'].median():.2f}%")
print(f"Cumulative bias max abs:    {res_df['cumulative_bias_pct'].abs().max():.2f}% ({res_df.loc[res_df['cumulative_bias_pct'].abs().idxmax(), 'symbol']})")

# ==================== YEARLY DRIFT ====================
drift_results = []
for symbol in sorted(df_clean["symbol"].unique()):
    sub = df_clean[df_clean["symbol"] == symbol].copy()
    for year in sorted(sub["year"].unique()):
        yr = sub[sub["year"] == year].copy()
        proxy_yr = (1 + yr["proxy_return"]).prod() - 1
        ref_yr = (1 + yr["ref_return"]).prod() - 1
        drift = ((1 + proxy_yr) / (1 + ref_yr) - 1) * 100 if ref_yr > -1 else float("nan")
        drift_results.append({
            "symbol": symbol,
            "year": year,
            "proxy_yearly_return_pct": proxy_yr * 100,
            "ref_yearly_return_pct": ref_yr * 100,
            "drift_pct": drift,
        })

drift_df = pd.DataFrame(drift_results)

# Trend analysis
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

# ==================== DIVIDEND DATE ANALYSIS ====================
# Find days where cumulative_dividend changes (dividend distribution date)
df_clean["prev_cum_div"] = df_clean.groupby("symbol")["cumulative_dividend"].shift(1)
df_clean["div_event"] = (df_clean["cumulative_dividend"] - df_clean["prev_cum_div"]).abs() > 0.001

div_days = df_clean[df_clean["div_event"]].copy()
print(f"\n=== Dividend distribution days: {len(div_days)} total ===")

# On dividend days, check proxy vs reference return behavior
if len(div_days) > 0:
    div_results = []
    for symbol in sorted(div_days["symbol"].unique()):
        sym_div = div_days[div_days["symbol"] == symbol]
        avg_proxy_ret = sym_div["proxy_return"].mean() * 100
        avg_ref_ret = sym_div["ref_return"].mean() * 100
        avg_diff = (sym_div["proxy_return"] - sym_div["ref_return"]).abs().mean() * 100
        div_results.append({
            "symbol": symbol,
            "n_div_days": len(sym_div),
            "avg_proxy_ret_on_div_pct": avg_proxy_ret,
            "avg_ref_ret_on_div_pct": avg_ref_ret,
            "avg_diff_on_div_pct": avg_diff,
        })
    
    div_df = pd.DataFrame(div_results)
    print(div_df.to_string(index=False))

# ==================== SAVE REPORTS ====================
report_path = REPORT_DIR / "price_mode_validation.csv"
res_df.to_csv(report_path, index=False, encoding="utf-8")

drift_report_path = REPORT_DIR / "yearly_drift.csv"
drift_df.to_csv(drift_report_path, index=False, encoding="utf-8")

trend_report_path = REPORT_DIR / "drift_trend.csv"
trend_df.to_csv(trend_report_path, index=False, encoding="utf-8")

print(f"\nReports saved:")
print(f"  {report_path}")
print(f"  {drift_report_path}")
print(f"  {trend_report_path}")

# ==================== FINAL CONCLUSION ====================
print("\n" + "="*90)
print("FINAL CONCLUSION")
print("="*90)

all_identical = res_df["proxy_equals_raw"].all()
print(f"\n1. CRITICAL: total_return_proxy == raw_close for ALL symbols: {all_identical}")
if all_identical:
    print("   -> The proxy does NOT adjust for dividends despite its name")
    print("   -> It is effectively a price-return series, not a total-return series")

# Check which references are total-return vs price-return
n_ref_total = res_df["ref_is_total_return"].sum()
print(f"\n2. Reference types:")
print(f"   - Total-return adjusted (reference_value != unit_nav): {n_ref_total}/20 symbols")
print(f"   - Price-return only (reference_value == unit_nav): {20 - n_ref_total}/20 symbols")

# Implications
print(f"\n3. Error interpretation:")
print(f"   For symbols where ref_is_total_return=1, errors capture both:")
print(f"   (a) proxy NOT adjusting for dividends (systematic undercount)")
print(f"   (b) Any data quality issues")
print(f"   For symbols where ref_is_total_return=0, errors capture only data quality issues")

# 510880 special case
r510880 = res_df[res_df["symbol"] == "510880"].iloc[0]
print(f"\n4. 510880 (bond ETF, official reference):")
print(f"   - Cumulative bias: {r510880['cumulative_bias_pct']:.1f}%")
print(f"   - Max cumulative dividend: {r510880['max_cumulative_dividend']:.4f}")
print(f"   - ref_is_total_return: {r510880['ref_is_total_return']}")
if r510880["ref_is_total_return"] == 0:
    print(f"   -> Bias is EXPECTED: proxy (price-return) vs reference (price-return)")
    print(f"      Both are price-return, so bias should be small if data is consistent")
else:
    print(f"   -> Bias is EXPECTED: proxy ignores dividends, reference includes them")

# Recommendation
print(f"\n5. Recommendation:")
if all_identical:
    print("   total_return_proxy CANNOT be used as a total-return series.")
    print("   It needs to be rebuilt to incorporate cumulative_dividend adjustments.")
    print("   Until then, treat it as raw_close (price-return only).")
    
    # Check if it's acceptable for price-return analysis
    n_pass_loose = (res_df["mae_daily_return_bps"] < 20).sum()
    print(f"\n   For price-return backtesting (ignoring dividends):")
    print(f"   - {n_pass_loose}/20 symbols have MAE < 20bps (acceptable)")
    
    # Symbols to exclude
    bad = res_df[res_df["mae_daily_return_bps"] > 50][["symbol", "mae_daily_return_bps"]]
    if len(bad) > 0:
        print(f"\n   Symbols with excessive error (>50bps MAE) - EXCLUDE:")
        for _, r in bad.iterrows():
            print(f"     {r['symbol']}: MAE={r['mae_daily_return_bps']:.1f} bps")
