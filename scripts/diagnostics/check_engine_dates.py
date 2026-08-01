# coding: utf-8
import sys, pandas as pd
sys.path.insert(0,'src')
from otf_backtest_engine import OTFBacktestEngine
e = OTFBacktestEngine(db_path='data/processed/otf_expanded.sqlite')
dates_to_check = ['2022-03-05','2022-03-20','2022-12-31','2023-01-02','2023-05-02','2023-12-31','2024-06-30']
vals = set(e._trading_dates.strftime('%Y-%m-%d'))
print('Engine trading dates count:', len(e._trading_dates))
for d in dates_to_check:
    print(d + ':', 'PRESENT' if d in vals else 'ABSENT')