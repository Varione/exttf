"""Training script v2 for RL adaptive strategy weight allocation."""

import sys, os, time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))

from rl_environment import StrategyWeightEnv
from rl_agent import PPOAgent


def train(train_start="2018-01-01", train_end="2021-12-31", n_epochs=50):
    print("=" * 60)
    print("RL Adaptive Strategy Weight Allocation v2")
    print("=" * 60)
    
    env = StrategyWeightEnv(
        start=train_start, end=train_end,
        lookback=5, n_hold=20, max_weight=0.05,
    )
    
    print(f"State dim: {env.state_dim}, Strategies: {env.n_strategies}")
    print(f"Trading days: {len(env.dates)}")
    
    agent = PPOAgent(
        state_dim=env.state_dim,
        action_dim=env.n_strategies,
        lr=1e-3, gamma=0.99, clip_epsilon=0.2,
        entropy_coef=0.01, value_coef=0.5,
    )
    
    steps_per_epoch = len(env.dates) - env.lookback
    temperature = 1.0
    
    best_reward = -np.inf
    all_rewards = []
    
    for epoch in range(n_epochs):
        state = env.reset()
        epoch_rewards = []
        
        for _ in range(steps_per_epoch):
            weights, action, log_prob, value, entropy = agent.select_action(state, temperature)
            next_state, reward, done, info = env.step(weights)
            
            agent.store(state, action, reward, log_prob, value, done)
            epoch_rewards.append(reward)
            
            state = next_state
            if done:
                break
        
        result = agent.update()
        
        if (epoch + 1) % 10 == 0:
            avg_reward = np.mean(epoch_rewards)
            all_rewards.append(avg_reward)
            
            policy_loss, value_loss, entropy = result if result else (0, 0, 0)
            
            print(f"Epoch {epoch+1:4d}/{n_epochs} | "
                  f"Reward: {avg_reward:.4f} | "
                  f"CumReturn: {info['cum_20d']*100:.2f}% | "
                  f"PL: {policy_loss:.4f} | VL: {value_loss:.6f} | "
                  f"Ent: {entropy:.4f}")
            
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save("models/rl_agent_best.pth")
        
        temperature *= 0.998
    
    agent.save("models/rl_agent_final.pth")
    
    print(f"\nTraining complete. Best reward: {best_reward:.4f}")
    
    # Plot training curve
    pd.Series(all_rewards).to_csv("data/processed/rl_training_curve.csv", index=False)
    
    return env, agent


def evaluate(env, agent, temperature=0.5):
    print("\n" + "=" * 60)
    print("Out-of-Sample Evaluation")
    print("=" * 60)
    
    state = env.reset()
    
    test_returns = []
    weight_history = []
    regime_history = []
    n_steps = len(env.dates) - env.current_idx
    
    for _ in range(n_steps):
        weights, action, _, _, _ = agent.select_action(state, temperature=temperature)
        next_state, reward, done, info = env.step(weights)
        
        test_returns.append(info["return"])
        weight_history.append(info["weights"].copy())
        
        date_str = env.dates[env.current_idx - 1].strftime("%Y-%m-%d")
        regime_history.append(env.regime_map.get(date_str, [0.33, 0.33, 0.34]))
        
        state = next_state
        if done:
            break
    
    ret_series = pd.Series(test_returns)
    cum = (1 + ret_series).cumprod() - 1
    
    print(f"\nTest days: {len(ret_series)}")
    print(f"Total return: {cum.iloc[-1]*100:.2f}%")
    ann_ret = (1 + cum.iloc[-1]) ** (252/len(ret_series)) - 1
    print(f"Annualized return: {ann_ret*100:.2f}%")
    sharpe = ret_series.mean() / (ret_series.std() + 1e-8) * np.sqrt(252)
    print(f"Sharpe ratio: {sharpe:.4f}")
    
    wealth = (1 + ret_series).cumprod()
    dd = wealth / wealth.cummax() - 1
    print(f"Max drawdown: {dd.min()*100:.2f}%")
    print(f"Win rate: {(ret_series > 0).mean()*100:.1f}%")
    
    # Weights by regime
    w_df = pd.DataFrame(weight_history, columns=env.strategy_names)
    r_df = pd.DataFrame(regime_history, columns=["prob_r0", "prob_r1", "prob_r2"])
    r_df["regime"] = r_df[["prob_r0", "prob_r1", "prob_r2"]].idxmax(axis=1).str.replace("prob_r", "")
    
    print("\nAverage weights by regime (top 3):")
    for regime in ["0", "1", "2"]:
        mask = r_df["regime"] == regime
        if mask.sum() > 0:
            top = w_df[mask].mean().nlargest(3)
            print(f"  Regime {regime} ({mask.sum()} days):")
            for s, w in top.items():
                print(f"    {s}: {w:.4f}")
    
    # Save
    result_df = pd.DataFrame({
        "return": test_returns,
        "cum_return": cum.values,
    })
    result_df.to_csv("data/processed/rl_backtest_results.csv", index=False)
    w_df.to_csv("data/processed/rl_weight_history.csv", index=False)
    
    # Compare with benchmarks
    bench = pd.read_csv("data/processed/backtest_results_repaired.csv")
    print("\nComparison with individual strategies:")
    print(bench[["strategy", "ann_return", "sharpe", "max_drawdown"]].to_string(index=False))
    print(f"\nRL Agent: ann_ret={ann_ret*100:.2f}%, sharpe={sharpe:.4f}, max_dd={dd.min()*100:.2f}%")
    
    return ret_series, w_df


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    _, agent = train(train_start="2018-01-01", train_end="2021-12-31")
    test_env = StrategyWeightEnv(
        start="2022-01-01", end="2026-07-17",
        lookback=5, n_hold=20, max_weight=0.05,
    )
    evaluate(test_env, agent)
