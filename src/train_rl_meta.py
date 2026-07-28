"""Walk-forward RL Meta Allocator training and evaluation.

Trains a PPO agent to allocate across strategies with regime awareness.
Uses strict walk-forward: Train 2018-2021, Val 2022-2024, Test 2025-2026.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))

from rl_environment import StrategyWeightEnv
from rl_agent import PPOAgent


def train_meta_allocator(
    train_start: str = "2018-01-01",
    train_end: str = "2021-12-31",
    val_start: str = "2022-01-01",
    val_end: str = "2024-12-31",
    test_start: str = "2025-01-01",
    test_end: str = "2026-07-17",
    n_epochs: int = 50,
    regime_path: str = "data/processed/regime_predictions_LSTM.csv",
):
    """Train RL meta allocator with walk-forward validation."""
    print("=" * 60)
    print("RL Meta Allocator Training (Walk-Forward)")
    print("=" * 60)

    # Training environment
    train_env = StrategyWeightEnv(
        start=train_start,
        end=train_end,
        lookback=5,
        n_hold=20,
        max_weight=0.05,
        respect_regime=True,
        max_single_weight=0.5,
        regime_path=regime_path,
    )

    print(f"State dim: {train_env.state_dim}")
    print(f"Strategies: {train_env.n_strategies}")
    print(f"Trading days (train): {len(train_env.dates)}")

    agent = PPOAgent(
        state_dim=train_env.state_dim,
        action_dim=train_env.n_strategies,
        lr=1e-3,
        gamma=0.99,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
    )

    steps_per_epoch = len(train_env.dates) - train_env.lookback
    temperature = 1.0
    best_reward = -np.inf
    all_rewards = []

    for epoch in range(n_epochs):
        state = train_env.reset()
        epoch_rewards = []

        for _ in range(steps_per_epoch):
            weights, action, log_prob, value, entropy = agent.select_action(
                state, temperature
            )
            next_state, reward, done, info = train_env.step(weights)

            agent.store(state, action, reward, log_prob, value, done)
            epoch_rewards.append(reward)

            state = next_state
            if done:
                break

        result = agent.update()

        if (epoch + 1) % 10 == 0:
            avg_reward = np.mean(epoch_rewards)
            all_rewards.append(avg_reward)

            policy_loss, value_loss, entropy_val = result if result else (0, 0, 0)

            print(
                f"Epoch {epoch+1:4d}/{n_epochs} | "
                f"Reward: {avg_reward:.4f} | "
                f"Cum20d: {info['cum_20d']*100:.2f}% | "
                f"DD: {info['drawdown']*100:.2f}% | "
                f"PL: {policy_loss:.4f} | VL: {value_loss:.6f} | "
                f"Ent: {entropy_val:.4f}"
            )

            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save("models/rl_meta_allocator_best.pth")

        temperature *= 0.998

    agent.save("models/rl_meta_allocator_final.pth")
    print(f"\nTraining complete. Best reward: {best_reward:.4f}")

    # Save training curve
    pd.Series(all_rewards).to_csv(
        "data/processed/rl_meta_training_curve.csv", index=False
    )

    # Validation evaluation
    print("\n" + "=" * 60)
    print("Validation Period Evaluation")
    print("=" * 60)
    val_metrics = evaluate_period(
        agent,
        start=val_start,
        end=val_end,
        period_name="Validation",
        regime_path=regime_path,
        temperature=0.5,
    )

    return agent, val_metrics


def evaluate_period(
    agent,
    start: str,
    end: str,
    period_name: str = "Test",
    regime_path: str = "data/processed/regime_predictions_LSTM.csv",
    temperature: float = 0.5,
) -> dict:
    """Evaluate RL agent on a specific period."""
    env = StrategyWeightEnv(
        start=start,
        end=end,
        lookback=5,
        n_hold=20,
        max_weight=0.05,
        respect_regime=True,
        max_single_weight=0.5,
        regime_path=regime_path,
    )

    state = env.reset()
    test_returns = []
    weight_history = []
    regime_history = []
    n_steps = len(env.dates) - env.current_idx

    for _ in range(n_steps):
        weights, action, _, _, _ = agent.select_action(state, temperature=temperature)
        next_state, reward, done, info = env.step(weights)

        test_returns.append(info["return"])
        weight_history.append(info["effective_weights"].copy())

        date_str = env.dates[env.current_idx - 1].strftime("%Y-%m-%d")
        regime_history.append(env.regime_map.get(date_str, [0.33, 0.33, 0.34]))

        state = next_state
        if done:
            break

    ret_series = pd.Series(test_returns)
    cum = (1 + ret_series).cumprod() - 1

    ann_ret = (1 + cum.iloc[-1]) ** (252 / len(ret_series)) - 1 if len(ret_series) > 0 else 0
    sharpe = ret_series.mean() / (ret_series.std() + 1e-8) * np.sqrt(252) if len(ret_series) > 1 else 0

    wealth = (1 + ret_series).cumprod()
    dd = wealth / wealth.cummax() - 1
    max_dd = float(dd.min()) if len(dd) > 0 else 0

    metrics = {
        "period": period_name,
        "n_days": len(ret_series),
        "total_return%": float(cum.iloc[-1] * 100) if len(cum) > 0 else 0,
        "ann_return%": float(ann_ret * 100),
        "sharpe": float(sharpe),
        "max_drawdown%": float(max_dd * 100),
        "win_rate%": float((ret_series > 0).mean() * 100) if len(ret_series) > 0 else 0,
        "volatility%": float(ret_series.std() * np.sqrt(252) * 100) if len(ret_series) > 1 else 0,
    }

    print(f"\n{period_name} Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save results
    result_df = pd.DataFrame({
        "return": test_returns,
        "cum_return": cum.values if len(cum) > 0 else [],
    })
    result_df.to_csv(f"data/processed/rl_meta_{period_name.lower()}_results.csv", index=False)

    w_df = pd.DataFrame(weight_history, columns=env.strategy_names)
    w_df.to_csv(f"data/processed/rl_meta_{period_name.lower()}_weights.csv", index=False)

    # Weights by regime analysis
    r_df = pd.DataFrame(regime_history, columns=["prob_r0", "prob_r1", "prob_r2"])
    r_df["regime"] = r_df[["prob_r0", "prob_r1", "prob_r2"]].idxmax(axis=1).str.replace("prob_r", "")

    print(f"\nAverage weights by regime (top 3):")
    for regime in ["0", "1", "2"]:
        mask = r_df["regime"] == regime
        if mask.sum() > 0:
            top = w_df[mask].mean().nlargest(3)
            print(f"  Regime {regime} ({mask.sum()} days):")
            for s, w in top.items():
                print(f"    {s}: {w:.4f}")

    return metrics


def run_comparison(
    agent_path: str = "models/rl_meta_allocator_best.pth",
    test_start: str = "2025-01-01",
    test_end: str = "2026-07-17",
    regime_path: str = "data/processed/regime_predictions_LSTM.csv",
):
    """Compare RL meta allocator against baselines."""
    print("\n" + "=" * 60)
    print("RL vs Baseline Comparison (OOS Test)")
    print("=" * 60)

    # Load trained agent
    if not os.path.exists(agent_path):
        print(f"Agent not found at {agent_path}, training first...")
        agent, _ = train_meta_allocator()
    else:
        temp_env = StrategyWeightEnv(
            start=test_start, end=test_end,
            respect_regime=True, regime_path=regime_path,
        )
        agent = PPOAgent(
            state_dim=temp_env.state_dim,
            action_dim=temp_env.n_strategies,
        )
        agent.load(agent_path)

    # RL evaluation
    rl_metrics = evaluate_period(
        agent,
        start=test_start,
        end=test_end,
        period_name="RL_Meta",
        regime_path=regime_path,
        temperature=0.3,
    )

    # Baselines: equal weight and regime-weighted
    from backtest_engine import BacktestEngine
    engine = BacktestEngine(
        regime_path=regime_path,
        data_mode="etf",
        price_mode="total_return_proxy",
    )

    from strategy_library import STRATEGIES
    strat_returns = {}
    for name in STRATEGIES:
        daily = engine.run_backtest(
            name,
            start=test_start,
            end=test_end,
            n_hold=20,
            max_weight=0.05,
            respect_regime=True,
            use_regime_probabilities=True,
        )
        if not daily.empty:
            strat_returns[name] = daily.set_index("date")["return"]

    combined_df = pd.DataFrame(strat_returns).fillna(0.0)

    # Equal weight
    ew_returns = combined_df.mean(axis=1)
    ew_metrics = engine.calculate_metrics(ew_returns)
    ew_summary = {
        "period": "Equal_Weight",
        "ann_return%": ew_metrics.get("CAGR%", 0),
        "sharpe": ew_metrics.get("Sharpe", 0),
        "max_drawdown%": ew_metrics.get("Max_Drawdown%", 0),
        "volatility%": ew_metrics.get("Annualized_Volatility%", 0),
        "win_rate%": ew_metrics.get("Daily_Win_Rate%", 0),
    }

    # Regime-weighted
    from regime_predictor import RegimePredictor
    predictor = RegimePredictor(model_type="LSTM")
    probs_df = predictor.predict_probabilities()

    from strategy_combiner import RegimeConditionedWeights
    rcw = RegimeConditionedWeights()
    prob_input = probs_df.set_index("date") if "date" in probs_df.columns else probs_df
    rw_weights = rcw.get_weights(combined_df, prob_input)

    rw_returns = combined_df.mul(
        pd.Series(rw_weights).reindex(combined_df.columns, fill_value=0)
    ).sum(axis=1)
    rw_metrics_calc = engine.calculate_metrics(rw_returns)
    rw_summary = {
        "period": "Regime_Weighted",
        "ann_return%": rw_metrics_calc.get("CAGR%", 0),
        "sharpe": rw_metrics_calc.get("Sharpe", 0),
        "max_drawdown%": rw_metrics_calc.get("Max_Drawdown%", 0),
        "volatility%": rw_metrics_calc.get("Annualized_Volatility%", 0),
        "win_rate%": rw_metrics_calc.get("Daily_Win_Rate%", 0),
    }

    # Print comparison table
    comparison = pd.DataFrame([ew_summary, rw_summary, rl_metrics])
    display_cols = ["period", "ann_return%", "sharpe", "max_drawdown%", "volatility%"]
    print("\nComparison Table:")
    print(comparison[display_cols].to_string(index=False))

    # Gate E check
    print("\n" + "=" * 60)
    print("Gate E Check: RL Incremental Value")
    print("=" * 60)

    improvements = []
    if rl_metrics["ann_return%"] > max(ew_summary["ann_return%"], rw_summary["ann_return%"]):
        improvements.append("return")
    if rl_metrics["max_drawdown%"] < min(ew_summary["max_drawdown%"], rw_summary["max_drawdown%"]):
        improvements.append("drawdown")
    if rl_metrics["volatility%"] < max(ew_summary["volatility%"], rw_summary["volatility%"]):
        improvements.append("volatility")
    if rl_metrics["sharpe"] > max(ew_summary["sharpe"], rw_summary["sharpe"]):
        improvements.append("sharpe")

    if improvements:
        print(f"RL shows improvement in: {', '.join(improvements)}")
        print("Gate E: PASS - RL has incremental value")
    else:
        print("RL does not show clear improvement over baselines")
        print("Gate E: FAIL - RL does not justify added complexity")

    comparison.to_csv("data/processed/rl_meta_comparison.csv", index=False)
    return comparison


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    t0 = time.time()

    agent, val_metrics = train_meta_allocator(
        train_start="2018-01-01",
        train_end="2021-12-31",
        val_start="2022-01-01",
        val_end="2024-12-31",
        test_start="2025-01-01",
        test_end="2026-07-17",
        n_epochs=30,
    )

    comparison = run_comparison(
        agent_path="models/rl_meta_allocator_best.pth",
        test_start="2025-01-01",
        test_end="2026-07-17",
    )

    print(f"\nTotal time: {time.time()-t0:.1f}s")
