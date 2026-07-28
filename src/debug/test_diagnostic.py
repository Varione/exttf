import sys, os
sys.path.insert(0, 'src')

print("[STAGE] DIAGNOSTIC", flush=True)
import numpy as np
import pandas as pd
from strategy_library import S01_CrossSectionalMomentum as S01_Momentum

# Load factors
factors = pd.read_csv("data/processed/factors_all_repaired.csv", parse_dates=["date"])
factors["symbol"] = factors["symbol"].astype(str)

print("[RESULT] factor dates: %d unique, range %s ~ %s" % (
    len(factors["date"].unique()),
    factors["date"].min().strftime("%Y-%m-%d"),
    factors["date"].max().strftime("%Y-%m-%d")
), flush=True)

# Check factor date format
sample_date = factors["date"].iloc[0]
print("[RESULT] sample factor date: %s (type: %s)" % (
    sample_date, type(sample_date).__name__
), flush=True)
print("[RESULT] sample factor date_str: %s" % sample_date.strftime("%Y-%m-%d"), flush=True)

# Test strategy with a known date
strat = S01_Momentum(factors)

# Try querying a date that should exist in factors
test_date_str = "2020-06-30"
day_data = strat._get_day_data(pd.Timestamp(test_date_str))
print("[RESULT] _get_day_data('%s'): %d rows" % (test_date_str, len(day_data)), flush=True)

# Check if the date exists in the lookup dict
if test_date_str in strat._factor_by_date:
    print("[CHECK] Date found in _factor_by_date: PASS", flush=True)
else:
    print("[CHECK] Date NOT found in _factor_by_date: FAIL", flush=True)
    # Check what dates are around it
    keys = sorted(strat._factor_by_date.keys())
    idx = next((i for i, k in enumerate(keys) if k >= test_date_str), len(keys))
    print("[RESULT] Nearby dates: %s" % str(keys[max(0,idx-3):idx+3]), flush=True)

# Test get_positions
positions = strat.get_positions(pd.Timestamp(test_date_str), n_hold=20, max_weight=0.05)
print("[RESULT] get_positions('%s'): %d positions" % (test_date_str, len(positions)), flush=True)

# Now test with RL environment dates
from rl_environment import StrategyWeightEnv
env = StrategyWeightEnv(start='2020-01-01', end='2021-12-31', lookback=5, n_hold=20, max_weight=0.05)

# Check date alignment
env_dates_str = [d.strftime("%Y-%m-%d") for d in env.dates[:5]]
factor_dates_str = sorted(strat._factor_by_date.keys())[:5]
print("[RESULT] First 5 env dates: %s" % str(env_dates_str), flush=True)
print("[RESULT] First 5 factor dates: %s" % str(factor_dates_str), flush=True)

# Check overlap
env_date_set = set(d.strftime("%Y-%m-%d") for d in env.dates)
factor_date_set = set(strat._factor_by_date.keys())
overlap = env_date_set & factor_date_set
print("[RESULT] Env dates: %d, Factor dates: %d, Overlap: %d (%.1f%%)" % (
    len(env_date_set), len(factor_date_set), len(overlap), len(overlap)/len(env_date_set)*100
), flush=True)

# Check strategy_returns
non_zero = (env.strategy_returns.values != 0).sum(axis=1)
total_days = len(env.dates)
days_with_any = np.sum(non_zero > 0)
print("[RESULT] Strategy returns: %d/%d days have non-zero" % (days_with_any, total_days), flush=True)

# Check a specific strategy
s = env.strategy_names[0]
non_zero_s = (env.strategy_returns[s].values != 0).sum()
print("[RESULT] %s: %d days non-zero" % (s, non_zero_s), flush=True)

print("\n[STAGE] COMPLETE", flush=True)
