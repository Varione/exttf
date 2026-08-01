import sys
src = 'src/otf_backtest_engine.py'
with open(src, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the buggy aggregation with correct searchsorted-based algorithm
old_block = '''        # Aggregate adjustment factors and distributions across holiday gaps.
        # For each fund, multiply all share_adjustment_factors between two
        # consecutive execution dates into the later execution date so that
        # NAV changes on non-execution dates (QDII holidays) are captured.
        nav_df_sorted = self._nav_df.sort_values(["fund_code", "nav_date"])
        exec_index_set = set(self._date_to_index.values())
        for fund_code, fund_rows in nav_df_sorted.groupby("fund_code"):
            pending_adj_product = 1.0
            pending_dist_sum = 0.0
            last_exec_idx = None
            for _, row in fund_rows.iterrows():
                date_idx = self._date_to_index.get(row.nav_date)
                adj = float(row.share_adjustment_factor) if pd.notna(row.share_adjustment_factor) else 1.0
                dist = float(row.distribution_per_share) if pd.notna(row.distribution_per_share) else 0.0

                if date_idx is not None:
                    # This NAV date is also an execution date; flush any pending factors.
                    if last_exec_idx is not None and pending_adj_product != 1.0:
                        self._share_adjustment_lookup[(fund_code, date_idx)] = (
                            self._share_adjustment_lookup.get((fund_code, date_idx), 1.0)
                            * pending_adj_product
                        )
                    if last_exec_idx is not None and pending_dist_sum > 1e-12:
                        self._distribution_lookup[(fund_code, date_idx)] = (
                            self._distribution_lookup.get((fund_code, date_idx), 0.0)
                            + pending_dist_sum
                        )
                    # Now record this execution-date's own factors.
                    if adj != 1.0:
                        self._share_adjustment_lookup[(fund_code, date_idx)] = (
                            self._share_adjustment_lookup.get((fund_code, date_idx), 1.0)
                            * adj
                        )
                    if dist > 0.0:
                        self._distribution_lookup[(fund_code, date_idx)] = (
                            self._distribution_lookup.get((fund_code, date_idx), 0.0)
                            + dist
                        )
                    last_exec_idx = date_idx
                    pending_adj_product = 1.0
                    pending_dist_sum = 0.0
                else:
                    # NAV-only date (holiday); accumulate factors for next execution date.
                    pending_adj_product *= adj
                    pending_dist_sum += dist'''

new_block = '''        # Aggregate adjustment factors and distributions across holiday gaps.
        # For each fund, for each source NAV row, map to the first execution date
        # >= that source date (searchsorted left).  All factors mapping to the same
        # (fund, exec_idx) are multiplied together.  This does NOT depend on the
        # fund having a raw NAV row on the target execution date.
        from collections import defaultdict
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
                dist_agg[key] += dist
        # Write aggregated factors into lookup tables
        for key, product in adj_agg.items():
            if product != 1.0:
                self._share_adjustment_lookup[key] = (
                    self._share_adjustment_lookup.get(key, 1.0) * product
                )
        for key, total in dist_agg.items():
            if total > 1e-12:
                self._distribution_lookup[key] = (
                    self._distribution_lookup.get(key, 0.0) + total
                )'''

content = content.replace(old_block, new_block)

with open(src, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed share_adjustment aggregation with searchsorted. New length:", len(content))