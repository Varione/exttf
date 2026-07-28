"""Fast correlation: compute factors for 50 ETFs directly, then correlate."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import sqlite3
import pandas as pd
import numpy as np
import time
from factor_definitions import FACTORS, get_all_categories, get_factors_by_category

t0 = time.time()

# Load top 50 ETFs
print("Loading top 50 ETFs...")
conn = sqlite3.connect("data/processed/etf.sqlite")
query = """
    SELECT d.symbol, d.date, d.open, d.high, d.low, d.close, d.volume, d.amount
    FROM etf_daily d
    INNER JOIN (
        SELECT symbol FROM etf_quality
        WHERE rows_valid >= 250 AND median_amount_60d > 1e6
        ORDER BY rows_valid DESC
        LIMIT 50
    ) q ON d.symbol = q.symbol
"""
df = pd.read_sql_query(query, conn)
df["date"] = pd.to_datetime(df["date"])
conn.close()

n_etfs = df["symbol"].nunique()
print(f"Loaded {len(df)} rows, {n_etfs} ETFs")

# Compute all factors per ETF, collect into pivot tables
factor_cols = [f.name for f in FACTORS]
pivot_data = {col: {} for col in factor_cols}

for symbol, group in df.groupby("symbol"):
    group = group.sort_values("date").reset_index(drop=True)
    for f in FACTORS:
        try:
            result = f.compute(group)
            pivot_data[f.name][symbol] = result
        except Exception:
            pass

print(f"Factors computed in {time.time()-t0:.1f}s")

# Build stacked arrays
t1 = time.time()
stacked = {}
for col in factor_cols:
    parts = []
    dates_list = []
    syms_list = []
    for symbol, series in pivot_data[col].items():
        # Get dates for this symbol
        sym_df = df[df["symbol"] == symbol]
        for i, val in enumerate(series.values):
            if np.isfinite(val):
                parts.append(val)
                dates_list.append(sym_df["date"].iloc[i])
                syms_list.append(symbol)
    stacked[col] = pd.Series(parts, pd.MultiIndex.from_arrays([dates_list, syms_list]))

# Find common (date, symbol) pairs
all_idx = set(stacked[factor_cols[0]].index)
for col in factor_cols[1:]:
    all_idx &= set(stacked[col].index)
common = sorted(all_idx)
print(f"Common obs: {len(common)} ({time.time()-t1:.1f}s)")

# Correlation matrix
n = len(factor_cols)
corr_arr = np.zeros((n, n))
for i in range(n):
    s1 = stacked[factor_cols[i]].loc[common].values
    for j in range(i, n):
        s2 = stacked[factor_cols[j]].loc[common].values
        mask = np.isfinite(s1) & np.isfinite(s2)
        if mask.sum() > 30:
            r = np.corrcoef(s1[mask], s2[mask])[0, 1]
            corr_arr[i, j] = r if not np.isnan(r) else 0.0
            corr_arr[j, i] = corr_arr[i, j]
    if (i+1) % 25 == 0:
        print(f"  Corr: {i+1}/{n} ({time.time()-t1:.1f}s)")

corr_df = pd.DataFrame(corr_arr, index=factor_cols, columns=factor_cols)
corr_df.to_csv("data/processed/factor_correlation.csv")
print(f"Correlation saved in {time.time()-t1:.1f}s")

# Helpers
def _cat(name):
    for f in FACTORS:
        if f.name == name:
            return f.category
    return "?"

idx_map = {f: i for i, f in enumerate(factor_cols)}

# Select uncorrelated
for thresh in [0.3, 0.5, 0.7]:
    sel = [factor_cols[0]]
    for f in factor_cols[1:]:
        fi = idx_map[f]
        if all(abs(corr_arr[fi, idx_map[s]]) < threshold for s in sel):
            sel.append(f)
    cats = {}
    for f in sel:
        c = _cat(f)
        cats[c] = cats.get(c, 0) + 1
    print(f"\n=== {len(sel)} factors (threshold={thresh}) ===")
    for c, cnt in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  [{c}] {cnt}")
    for f in sel:
        print(f"    {f}")

# High correlation pairs
print("\n=== Top 30 High Correlation (>0.8) ===")
hc = []
for i in range(n):
    for j in range(i+1, n):
        r = abs(corr_arr[i, j])
        if r > 0.8:
            hc.append((factor_cols[i], factor_cols[j], r))
hc.sort(key=lambda x: x[2], reverse=True)
for c1, c2, r in hc[:30]:
    print(f"  {c1:30s} vs {c2:30s}: {r:.4f}")

# Category summary
print("\n=== Factors per category ===")
for cat in get_all_categories():
    print(f"  {cat}: {len(get_factors_by_category(cat))} factors")
print(f"\nTotal: {len(FACTORS)} factors")

# Quality report
print("\n=== Factor Quality (top 30 cs_std) ===")
quality = []
for col in factor_cols:
    s = stacked[col]
    valid = s.notna() & (s != 0) & np.isfinite(s)
    coverage = valid.mean()
    # CS std by date
    by_date = s.loc[valid].groupby(level=0).std()
    cs_std = by_date.mean()
    quality.append((col, _cat(col), coverage, cs_std))

quality.sort(key=lambda x: x[3], reverse=True)
for col, cat, cov, std in quality[:30]:
    print(f"  {col:30s} [{cat:15s}] cov={cov:.2%} cs_std={std:.6f}")

print(f"\nTotal time: {time.time()-t0:.1f}s")
