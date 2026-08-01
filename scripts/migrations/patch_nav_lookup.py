src = 'src/otf_backtest_engine.py'
with open(src, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix NAV lookup to use all NAV dates for ffill before reindexing to execution dates
old_nav_lookup = '''        nav_pivot = self._nav_df.pivot_table(
            index="fund_code", columns="nav_date", values="unit_nav"
        )
        self._nav_lookup = {}
        self._valuation_nav_lookup = {}
        self._distribution_lookup = {}
        self._share_adjustment_lookup = {}
        for fund_code in nav_pivot.index:
            series = nav_pivot.loc[fund_code]
            valuation_series = series.reindex(self._trading_dates).ffill()
            for idx, date in enumerate(self._trading_dates):
                if date in series.index and not np.isnan(series[date]):
                    self._nav_lookup[(fund_code, idx)] = float(series[date])
                valuation_nav = valuation_series.iloc[idx]
                if not np.isnan(valuation_nav):
                    self._valuation_nav_lookup[(fund_code, idx)] = float(
                        valuation_nav
                    )'''

new_nav_lookup = '''        nav_pivot = self._nav_df.pivot_table(
            index="fund_code", columns="nav_date", values="unit_nav"
        )
        self._nav_lookup = {}
        self._valuation_nav_lookup = {}
        self._distribution_lookup = {}
        self._share_adjustment_lookup = {}
        for fund_code in nav_pivot.index:
            series = nav_pivot.loc[fund_code]
            # Use all valuation source dates for ffill so QDII NAV changes on
            # non-execution dates are captured, then reindex to execution dates.
            full_series = series.reindex(self._valuation_source_dates).ffill()
            valuation_series = full_series.reindex(self._trading_dates)
            for idx, date in enumerate(self._trading_dates):
                if date in series.index and not np.isnan(series[date]):
                    self._nav_lookup[(fund_code, idx)] = float(series[date])
                valuation_nav = valuation_series.iloc[idx]
                if not np.isnan(valuation_nav):
                    self._valuation_nav_lookup[(fund_code, idx)] = float(
                        valuation_nav
                    )'''

content = content.replace(old_nav_lookup, new_nav_lookup)

with open(src, 'w', encoding='utf-8') as f:
    f.write(content)

print("NAV lookup fix applied. New length:", len(content))