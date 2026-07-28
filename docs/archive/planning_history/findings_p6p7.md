# Phase 6 + Phase 7 Findings

## Codebase Analysis

### Regime Model Architecture
- Models saved with: model_state, mean, std, factor_cols, n_classes, n_factors, lookback, test_acc, macro_f1
- Normalization: mean/std computed from training period only (prevents leakage)
- Input shape per sample: (lookback=20, n_factors=52)
- Models: MLP (flat), LSTM (sequential), Transformer (attention)

### Current Regime Flow
1. `detect_regimes.py` computes daily cross-sectional factor means
2. KMeans clustering on training period only
3. Labels saved as discrete integers in `regime_labels.csv`
4. `backtest_engine.py` loads regime predictions, maps date -> regime int
5. Strategy execution gated by `respect_regime` flag

### RL Environment Current State
- `respect_regime=False` - all strategies always active (semantic issue)
- Regime probs used as state input but NOT for gating strategy returns
- Reward: simple `port_return * 100.0` (no risk adjustment)
- Weight constraint: clip to [0, 0.4] per strategy

### Strategy-Regime Mapping
- Regime 0 (bull/trend): S01, S02, S03, S04
- Regime 1 (range-bound): S11, S12, S13
- Regime 2 (bear/crisis): S21, S22, S23
- Regime -1 (agnostic): B0, B1, B2, B3

## Design Decisions

### Probability Method Choice
- LSTM models output logits -> softmax for probability
- This is natural and well-calibrated (Macro-F1=0.833)
- No need for distance-based approximation unless model unavailable

### RL Reward Design
- Current reward too simple: `port_return * 100`
- New reward needs: Sharpe proxy + turnover penalty + drawdown floor + concentration penalty
- Formula: `sharpe_proxy - lambda_t * turnover - lambda_dd * max_dd - lambda_c * concentration`

### Walk-Forward RL Training
- Must recompute strategy returns for each walk-forward window (to prevent leakage)
- Train env uses train period, validate on val period, test on test period
- Single final model saved from best validation performance
