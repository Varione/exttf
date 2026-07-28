import sys, os
sys.path.insert(0, 'src')

import pandas as pd
from strategy_library import S01_CrossSectionalMomentum, B0_BuyHoldEW

# Load data
factors = pd.read_csv('data/processed/factors_all_repaired.csv', parse_dates=['date'])
regime = pd.read_csv('data/processed/regime_predictions.csv', parse_dates=['date'])

# Test on a specific date
test_date = pd.Timestamp('2018-01-02')
current_regime = regime[regime['date'] == test_date]['regime'].iloc[0]
print(f'Test date: {test_date.date()}, Regime: {current_regime}')

# Test strategy
s01 = S01_CrossSectionalMomentum(factors)
signal = s01.compute_signal(test_date)
print(f'\nS01 signal length: {len(signal)}')
print(f'Signal head:\n{signal.head(10)}')

positions = s01.get_positions(test_date, n_hold=20, max_weight=0.05)
print(f'\nPositions: {len(positions)}')
print(f'Position symbols: {list(positions.keys())[:5]}...')

# Check if these symbols exist in price data
import sqlite3
conn = sqlite3.connect('data/processed/etf.sqlite')
prices = pd.read_sql(f"SELECT DISTINCT symbol FROM etf_daily WHERE symbol IN ({','.join(['?']*len(positions))})", conn, params=list(positions.keys()))
conn.close()
print(f'Found in prices: {len(prices)} symbols')

# Test B0
b0 = B0_BuyHoldEW(factors)
b0_signal = b0.compute_signal(test_date)
print(f'\nB0 signal length: {len(b0_signal)}')
b0_positions = b0.get_positions(test_date, n_hold=20, max_weight=0.05)
print(f'B0 positions: {len(b0_positions)}, symbols: {list(b0_positions.keys())[:5]}...')
