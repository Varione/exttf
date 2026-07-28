"""Efficient correlation using vectorized panel building."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import sqlite3
import pandas as pd
import numpy as np
import time
from factor_definitions import FACTORS, get_all_categories, get_factors_by_category

t0 = time.time()
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
conn.close()
df["date"] = pd.to_datetime(df["date"])

factor_cols = [f.name for f in FACTORS]
n = len(factor_cols)
symbols = sorted(df["symbol"].unique())
dates_np = np.sort(np.unique(df["date"].values))
sym_idx = {s: i for i, s in enumerate(symbols)}
date_idx = {d: i for i, d in enumerate(dates_np)}

print(f"Loaded {len(df)} rows, {len(symbols)} ETFs, {len(dates_np)} dates, {n} factors ({time.time()-t0:.1f}s)")

# Build panel arrays directly
print("Computing factors into panel...")
panel = {col: np.zeros((len(dates_np), len(symbols))) for col in factor_cols}

total_filled = 0
for symbol, group in df.groupby("symbol"):
    g = group.sort_values("date").reset_index(drop=True)
    si = sym_idx[symbol]
    # Map each row to date index using searchsorted (fast vectorized)
    dates_g = g["date"].values  # numpy datetime64, already sorted
    row_date_idx = np.searchsorted(dates_np, dates_g)
    for f in FACTORS:
        try:
            vals = f.compute(g).values
            valid = np.isfinite(vals)
            panel[f.name][row_date_idx[valid], si] = vals[valid]
            total_filled += valid.sum()
        except Exception:
            pass

print(f"Panel built in {time.time()-t0:.1f}s, filled={total_filled}")

# Verify data quality
nz = sum((panel[c] != 0).sum() for c in factor_cols)
total = len(dates_np) * len(symbols) * n
print(f"Non-zero entries: {nz}/{total} ({nz/total*100:.1f}%)")

# Correlation matrix
t1 = time.time()
print("Computing correlation matrix...")
corr_arr = np.zeros((n, n))
for i in range(n):
    a1 = panel[factor_cols[i]].ravel()
    for j in range(i, n):
        a2 = panel[factor_cols[j]].ravel()
        mask = np.isfinite(a1) & np.isfinite(a2) & ((a1 != 0) | (a2 != 0))
        if mask.sum() > 30:
            r = np.corrcoef(a1[mask], a2[mask])[0, 1]
            corr_arr[i, j] = r if not np.isnan(r) else 0.0
            corr_arr[j, i] = corr_arr[i, j]
    if (i+1) % 25 == 0:
        print(f"  {i+1}/{n} ({time.time()-t1:.1f}s)")

corr_df = pd.DataFrame(corr_arr, index=factor_cols, columns=factor_cols)
corr_df.to_csv("data/processed/factor_correlation.csv")
print(f"Correlation saved in {time.time()-t1:.1f}s")

# Analysis
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
        if all(abs(corr_arr[fi, idx_map[s]]) < thresh for s in sel):
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
if not hc:
    print("  (none above 0.8)")
else:
    for c1, c2, r in hc[:30]:
        print(f"  {c1:30s} vs {c2:30s}: {r:.4f}")

# Category summary
print("\n=== Factors per category ===")
for cat in get_all_categories():
    print(f"  {cat}: {len(get_factors_by_category(cat))} factors")
print(f"Total: {len(FACTORS)} factors")

# Quality report
print("\n=== Factor Quality (top 30 cs_std) ===")
quality = []
for col in factor_cols:
    arr = panel[col]
    valid = arr != 0
    coverage = valid.mean()
    date_stds = []
    for di in range(len(dates_np)):
        row = arr[di]
        if (row != 0).sum() > 5:
            date_stds.append(np.std(row[row != 0]))
    cs_std_mean = np.mean(date_stds) if date_stds else 0
    quality.append((col, _cat(col), coverage, cs_std_mean))

quality.sort(key=lambda x: x[3], reverse=True)
for col, cat, cov, std in quality[:30]:
    print(f"  {col:30s} [{cat:15s}] cov={cov:.2%} cs_std={std:.6f}")

print(f"\nTotal time: {time.time()-t0:.1f}s")
