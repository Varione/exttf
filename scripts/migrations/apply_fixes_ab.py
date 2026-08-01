import sys
src = 'src/otf_backtest_engine.py'
with open(src, 'r', encoding='utf-8') as f:
    content = f.read()

# A1: Change C2_NOT_EXECUTION_DATE to OTF_NOT_EXECUTION_DATE
content = content.replace('C2_NOT_EXECUTION_DATE:', 'OTF_NOT_EXECUTION_DATE:')

# A2: Add target_weights validation in run_backtest after numeric conversion
old_validation = '''        target_weights = numeric_targets

        if start is None:'''

new_validation = '''        target_weights = numeric_targets

        # Validate that any date with non-NaN targets is an execution date.
        if self.execution_calendar is not None and not target_weights.empty:
            exec_set = set(self._trading_dates)
            bad_dates = []
            for idx_date in target_weights.index:
                if idx_date not in exec_set and target_weights.loc[idx_date].notna().any():
                    bad_dates.append(idx_date.strftime("%Y-%m-%d"))
            if bad_dates:
                raise RuntimeError(f"OTF_TARGET_NOT_EXECUTION_DATE:{','.join(sorted(bad_dates))}")

        if start is None:'''

content = content.replace(old_validation, new_validation)

# B: Fix share_adjustment_lookup to aggregate across holiday gaps
old_adj_lookup = '''        for row in self._nav_df.itertuples(index=False):
            date_idx = self._date_to_index.get(row.nav_date)
            if date_idx is not None and pd.notna(row.distribution_per_share):
                self._distribution_lookup[(row.fund_code, date_idx)] = max(
                    0.0, float(row.distribution_per_share)
                )
            if date_idx is not None and pd.notna(row.share_adjustment_factor):
                self._share_adjustment_lookup[(row.fund_code, date_idx)] = float(
                    row.share_adjustment_factor
                )'''

new_adj_lookup = '''        # Aggregate adjustment factors and distributions across holiday gaps.
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

content = content.replace(old_adj_lookup, new_adj_lookup)

with open(src, 'w', encoding='utf-8') as f:
    f.write(content)

print("Engine patch A+B applied. New length:", len(content))