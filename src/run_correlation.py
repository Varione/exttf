"""Fast correlation analysis for 124 factors."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import time
from factor_definitions import FACTORS


def _get_category(name):
    for f in FACTORS:
        if f.name == name:
            return f.category
    return "?"


t0 = time.time()
csv_path = "data/processed/factors_all_repaired.csv"

# Read only essential columns
print("Loading data...")
cols_needed = ["symbol", "date"] + [f.name for f in FACTORS]
df = pd.read_csv(csv_path, usecols=cols_needed, parse_dates=["date"])
print(f"Loaded {len(df)} rows, {len(cols_needed)} cols in {time.time()-t0:.1f}s")

factor_cols = [f.name for f in FACTORS]

# Sample top 50 ETFs for correlation
symbol_counts = df["symbol"].value_counts()
top_symbols = symbol_counts.head(50).index
sampled = df[df["symbol"].isin(top_symbols)]
print(f"Sampled {len(sampled)} rows from 50 ETFs")

# Stack cross-sectional data
t1 = time.time()
factor_data = {}
for col in factor_cols:
    pvt = sampled.pivot(index="date", columns="symbol", values=col)
    factor_data[col] = pvt.stack()

# Find common index
all_idx = set(factor_data[factor_cols[0]].index)
for col in factor_cols[1:]:
    all_idx &= set(factor_data[col].index)
common = sorted(all_idx)
print(f"Common observations: {len(common)} ({time.time()-t1:.1f}s)")

# Compute correlation matrix
n = len(factor_cols)
corr_arr = np.zeros((n, n))
for i in range(n):
    s1 = factor_data[factor_cols[i]].loc[common].values
    for j in range(i, n):
        s2 = factor_data[factor_cols[j]].loc[common].values
        mask = np.isfinite(s1) & np.isfinite(s2)
        if mask.sum() > 30:
            r = np.corrcoef(s1[mask], s2[mask])[0, 1]
            if np.isnan(r):
                r = 0.0
        else:
            r = 0.0
        corr_arr[i, j] = r
        corr_arr[j, i] = r
    if (i + 1) % 25 == 0:
        print(f"  Correlation: {i+1}/{n} ({time.time()-t0:.1f}s)")

corr_df = pd.DataFrame(corr_arr, index=factor_cols, columns=factor_cols)
corr_df.to_csv("data/processed/factor_correlation.csv")
print(f"Correlation matrix saved in {time.time()-t1:.1f}s")

# Select uncorrelated factors
def select_uncorrelated(threshold):
    selected = [factor_cols[0]]
    for i, f in enumerate(factor_cols[1:]):
        idx = i + 1
        if all(abs(corr_arr[idx, factor_cols.index(s)]) < threshold for s in selected):
            selected.append(f)
    return selected

for thresh in [0.3, 0.5, 0.7]:
    selected = select_uncorrelated(thresh)
    cats = {}
    for f in selected:
        cat = _get_category(f)
        cats[cat] = cats.get(cat, 0) + 1
    print(f"\n=== {len(selected)} factors (threshold={thresh}) ===")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  [{cat}] {count}")
    for f in selected:
        print(f"    {f}")

# High correlation pairs
print("\n=== Top 30 High Correlation Pairs (>0.8) ===")
high_corr = []
for i in range(n):
    for j in range(i+1, n):
        r = abs(corr_arr[i, j])
        if r > 0.8:
            high_corr.append((factor_cols[i], factor_cols[j], r))
high_corr.sort(key=lambda x: x[2], reverse=True)
for c1, c2, r in high_corr[:30]:
    print(f"  {c1:30s} vs {c2:30s}: {r:.4f}")

# Category summary
from factor_definitions import get_all_categories, get_factors_by_category
print("\n=== Factors per category ===")
for cat in get_all_categories():
    factors = get_factors_by_category(cat)
    print(f"  {cat}: {len(factors)} factors")
print(f"\nTotal: {len(FACTORS)} factors")

# Quick quality report
print("\n=== Factor Quality (coverage + cs_std, top 30) ===")
quality_rows = []
for col in factor_cols:
    vals = df[col]
    valid = (vals != 0) & vals.notna() & np.isfinite(vals)
    coverage = valid.mean()
    cs_std = df.loc[valid].groupby("date")[col].std().mean()
    quality_rows.append((col, _get_category(col), coverage, cs_std))

quality_rows.sort(key=lambda x: x[3], reverse=True)
for col, cat, cov, std in quality_rows[:30]:
    print(f"  {col:30s} [{cat:15s}] coverage={cov:.2%} cs_std={std:.6f}")

print(f"\nTotal time: {time.time()-t0:.1f}s")
