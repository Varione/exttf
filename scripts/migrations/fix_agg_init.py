import sys
src = 'src/otf_backtest_engine.py'
with open(src, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the aggregation logic bug
old_agg = '''        from collections import defaultdict
        adj_agg: dict[tuple[str, int], float] = defaultdict(float)
        dist_agg: dict[tuple[str, int], float] = defaultdict(float)
        for _, row in self._nav_df.iterrows():
            source_date = pd.Timestamp(row.nav_date)
            # First execution date >= source_date
            pos = self._trading_dates.searchsorted(source_date, side="left")
            if pos >= len(self._trading_dates):
                continue  # source after last execution date; ignore
            exec_idx = pos
            adj = float(row.share_adjustment_factor) if pd.notna(row.share_adjustment_factor) else 1.0
            dist = float(row.distribution_per_share) if pd.notna(row.distribution_per_share) else 0.0
            key = (row.fund_code, exec_idx)
            if adj != 1.0:
                adj_agg[key] *= adj if adj_agg[key] == 0.0 else adj_agg[key] * adj
            if dist > 0.0:
                dist_agg[key] += dist'''

new_agg = '''        from collections import defaultdict
        # Use dict with explicit product tracking; start at 1.0 for multiplication identity.
        adj_agg: dict[tuple[str, int], float] = {}
        dist_agg: dict[tuple[str, int], float] = defaultdict(float)
        for _, row in self._nav_df.iterrows():
            source_date = pd.Timestamp(row.nav_date)
            # First execution date >= source_date
            pos = self._trading_dates.searchsorted(source_date, side="left")
            if pos >= len(self._trading_dates):
                continue  # source after last execution date; ignore
            exec_idx = pos
            adj = float(row.share_adjustment_factor) if pd.notna(row.share_adjustment_factor) else 1.0
            dist = float(row.distribution_per_share) if pd.notna(row.distribution_per_share) else 0.0
            key = (row.fund_code, exec_idx)
            if adj != 1.0:
                adj_agg[key] = adj if key not in adj_agg else adj_agg[key] * adj
            if dist > 0.0:
                dist_agg[key] += dist'''

content = content.replace(old_agg, new_agg)

with open(src, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed aggregation init logic. New length:", len(content))