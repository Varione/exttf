import sys, os
sys.path.insert(0, 'src')

print("[STAGE] INIT", flush=True)
print("=" * 60, flush=True)
print("ETF Project - Basic Functionality Test", flush=True)
print("=" * 60, flush=True)

# Test 1: Backtest Engine (short period)
print("\n[STAGE] TEST_1_BACKTEST_ENGINE", flush=True)
from rl_environment import StrategyWeightEnv
import numpy as np
import pandas as pd

env = StrategyWeightEnv(start='2020-01-01', end='2021-12-31', lookback=5, n_hold=20, max_weight=0.05)
non_zero = (env.strategy_returns.values != 0).sum(axis=1)
total_days = len(env.dates)
result1 = np.sum(non_zero > 0)
print("[RESULT] test_1_backtest: %d/%d days have non-zero returns" % (result1, total_days), flush=True)
print("[CHECK] test_1_backtest %s (%.1f%%)" % ("PASS" if result1 > total_days * 0.8 else "FAIL", result1/total_days*100), flush=True)

# Test 2: Strategy Library dictionary optimization
print("\n[STAGE] TEST_2_STRATEGY_LIBRARY", flush=True)
from strategy_library import S01_CrossSectionalMomentum as S01_Momentum, S13_QualityFactor as S13_Quality
strat = S01_Momentum(env.factors)
positions = strat.get_positions(pd.Timestamp('2020-06-30'), n_hold=20, max_weight=0.05)
print("[RESULT] test_2_strategy: %d positions selected" % len(positions), flush=True)
print("[CHECK] test_2_strategy %s" % ("PASS" if len(positions) > 0 else "FAIL"), flush=True)

# Test 3: RL Agent continuous action space
print("\n[STAGE] TEST_3_RL_AGENT", flush=True)
from rl_agent import PPOAgent
agent = PPOAgent(state_dim=env.state_dim, action_dim=env.n_strategies)
state = env.reset()
weights, _, _, _, _ = agent.select_action(state, temperature=1.0)
print("[RESULT] test_3_rl: sum=%.3f min=%.3f max=%.3f" % (weights.sum(), weights.min(), weights.max()), flush=True)
print("[CHECK] test_3_rl %s" % ("PASS" if abs(weights.sum() - 1.0) < 0.01 else "FAIL"), flush=True)

# Test 4: No hard-coded strategy blocking
print("\n[STAGE] TEST_4_STRATEGY_COUNT", flush=True)
print("[RESULT] test_4_strategies: %d strategies (expected 11)" % env.n_strategies, flush=True)
print("[CHECK] test_4_strategies %s" % ("PASS" if env.n_strategies == 11 else "FAIL"), flush=True)

# Test 5: Reward function simplified
print("\n[STAGE] TEST_5_REWARD_FUNCTION", flush=True)
state = env.reset()
weights_eq = np.ones(env.n_strategies) / env.n_strategies
ns, reward, done, info = env.step(weights_eq)
print("[RESULT] test_5_reward: reward=%.4f return=%.6f" % (reward, info['return']), flush=True)
print("[CHECK] test_5_reward %s" % ("PASS" if not np.isnan(reward) else "FAIL"), flush=True)

# Test 6: Backtest returns reasonable
print("\n[STAGE] TEST_6_RETURN_DISTRIBUTION", flush=True)
rets = env.strategy_returns.dropna()
for col in rets.columns[:5]:
    r = rets[col].dropna()
    print("[RESULT] %s: mean=%.5f std=%.5f" % (col, r.mean(), r.std()), flush=True)

print("\n" + "=" * 60, flush=True)
print("[STAGE] COMPLETE", flush=True)
print("All tests completed!", flush=True)
print("=" * 60, flush=True)
