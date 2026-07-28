# Phase 6 + Phase 7 Task Plan

## Goal
Implement probability-based regime prediction (Phase 6) and RL meta allocator (Phase 7), with strict walk-forward validation and Gate E checks.

## Current State Analysis

### Existing Components
- **Regime detection:** KMeans hard labels in `detect_regimes.py`, saved as `regime_labels.csv` + `regime_model_cluster.joblib`
- **Classification models:** MLP, LSTM, Transformer trained in `train_classifier.py`, saved as `.pt` files in `data/processed/`
  - LSTM is best (Macro-F1=0.833 per task description)
  - Models output logits via `predict_proba()` which applies softmax
- **Backtest engine:** `backtest_engine.py` supports discrete regime via `regime_map` dict
- **Strategy combiner:** `strategy_combiner.py` has `RegimeConditionedWeights` class (already supports prob DataFrame)
- **RL environment:** `rl_environment.py` uses `respect_regime=False`, regime probs as 3-dim state input
- **RL agent:** PPO in `rl_agent.py`, trained via `train_rl.py`

### Key Data Shapes
- Classification data: X shape (4885, 1040), lookback=20, n_factors=52
- Regime labels: 4948 dates, 3 clusters
- Strategy returns: Pre-computed per-strategy in RL environment

---

## Phase 6: Probability Regime Predictor

### 6.1 Create `src/regime_predictor.py`
- `RegimePredictor` class loads trained model + normalization params
- `predict_probabilities(factor_df)` - computes daily factors, feeds to model, returns softmax probabilities
- `get_regime_weighted_strategy_weights(probs, strategy_regimes)` - weighted combination formula
- Support all 3 models (MLP, LSTM, Transformer) via model_type parameter
- Fallback: KMeans distance-based probability if model unavailable

### 6.2 Modify `backtest_engine.py`
- Add `use_regime_probabilities: bool = False` parameter
- When enabled, compute strategy weights as weighted sum over regime probabilities
- Track and output `regime_transition_count` for turnover comparison
- Modify `run_backtest` to support probability-weighted regime logic

### 6.3 Model Comparison Script
- Create `src/run_regime_comparison.py`
- Compare 4 methods: Cluster baseline, MLP probs, LSTM probs, Transformer probs
- Metrics: OOS Sharpe, Turnover reduction, Drawdown reduction, State persistence
- Output: CSV comparison table + summary

---

## Phase 7: RL Meta Allocator

### 7.1 Modify `rl_environment.py`
- **Critical fix:** Change `respect_regime=True` so strategies only generate returns in their designated regime
- Enhance state space: regime probs (3) + strategy metrics (returns, vol, drawdown, correlation)
- Action space: strategy weights with constraints (sum <= 1, w >= 0, max single weight 0.5)
- Reward function: risk_adjusted_return - turnover_penalty - drawdown_penalty - concentration_penalty

### 7.2 Create Walk-Forward RL Training
- Create `src/train_rl_meta.py`
- Walk-forward: Train 2018-2021, Val 2022-2024, Test 2025-2026
- Constraints: max single strategy weight 0.5, tunable turnover penalty
- Output: `models/rl_meta_allocator.pth`

### 7.3 RL vs Baseline Comparison
- Create `src/run_rl_comparison.py`
- Compare: Equal weight, Regime-conditioned (Phase 6), RL meta allocator
- Gate E check: RL must show improvement in at least one of return/drawdown/volatility/turnover

---

## Implementation Order

1. **regime_predictor.py** - Core probability prediction logic
2. **backtest_engine.py modifications** - Probability regime support
3. **run_regime_comparison.py** - Phase 6 validation
4. **rl_environment.py modifications** - Meta allocator environment
5. **train_rl_meta.py** - Walk-forward RL training
6. **run_rl_comparison.py** - Phase 7 validation + Gate E check
7. **Tests** - pytest tests for new components

## Files to Create
- `src/regime_predictor.py` (new)
- `src/run_regime_comparison.py` (new)
- `src/train_rl_meta.py` (new)
- `src/run_rl_comparison.py` (new)
- `tests/test_regime_predictor.py` (new)
- `tests/test_rl_meta.py` (new)

## Files to Modify
- `src/backtest_engine.py` - Add probability regime support
- `src/rl_environment.py` - Fix respect_regime, enhance state/reward
