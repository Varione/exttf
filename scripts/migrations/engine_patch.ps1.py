import sys
src = 'src/otf_backtest_engine.py'
with open(src, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    'from otf_trading_rules import ProductRuleBook\n',
    'from otf_trading_rules import ProductRuleBook\nfrom otf_rotation.execution_calendar import ExecutionCalendar\n'
)

old_init_sig = '''        channel: str = "ALL",
    ):'''
new_init_sig = '''        channel: str = "ALL",
        execution_calendar: ExecutionCalendar | None = None,
    ):'''
content = content.replace(old_init_sig, new_init_sig)

old_channel = 'self.channel = channel\n'
new_channel = '''self.channel = channel
        self.execution_calendar = execution_calendar
        if execution_calendar is not None and not isinstance(execution_calendar, ExecutionCalendar):
            raise ValueError("execution_calendar must be an ExecutionCalendar instance")

'''
content = content.replace(old_channel, new_channel)

old_trading_dates = '''        all_dates = self._nav_df["nav_date"].drop_duplicates().sort_values()
        self._trading_dates = pd.DatetimeIndex(all_dates)
        self._date_to_index = {
            date: idx for idx, date in enumerate(self._trading_dates)
        }'''

new_trading_dates = '''        all_nav_dates = self._nav_df["nav_date"].drop_duplicates().sort_values()
        self._valuation_source_dates = pd.DatetimeIndex(all_nav_dates)
        if self.execution_calendar is not None:
            self._trading_dates = pd.DatetimeIndex(self.execution_calendar.dates)
        else:
            self._trading_dates = pd.DatetimeIndex(all_nav_dates)
        self._date_to_index = {
            date: idx for idx, date in enumerate(self._trading_dates)
        }'''
content = content.replace(old_trading_dates, new_trading_dates)

old_submit_check = '''        submit_idx = self._date_to_index.get(submit_date)
        if submit_idx is None:
            return None'''

new_submit_check = '''        submit_idx = self._date_to_index.get(submit_date)
        if submit_idx is None:
            if self.execution_calendar is not None:
                raise RuntimeError(f"C2_NOT_EXECUTION_DATE:{submit_date.date()}")
            return None'''
content = content.replace(old_submit_check, new_submit_check)

with open(src, 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch applied. New length:", len(content))