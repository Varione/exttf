"""Analyze saved correlation matrix."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from factor_definitions import FACTORS, get_all_categories, get_factors_by_category

corr_df = pd.read_csv("data/processed/factor_correlation.csv", index_col=0)
factor_cols = corr_df.columns.tolist()
n = len(factor_cols)
corr_arr = corr_df.values

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
for c1, c2, r in hc[:30]:
    print(f"  {c1:30s} vs {c2:30s}: {r:.4f}")

# Category summary
print("\n=== Factors per category ===")
for cat in get_all_categories():
    print(f"  {cat}: {len(get_factors_by_category(cat))} factors")
print(f"Total: {len(FACTORS)} factors")
