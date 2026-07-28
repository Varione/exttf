# P1 Strategy and Backtest Improvements Plan

## Goal
Fix modeling gaps discovered during P0 audit, then re-run backtest on complete data.

## P1-1: Fix Win Rate Calculation
**Status:** pending → Subagent: general-5090d-9b
**Problem:** `calculate_metrics` uses `(returns > 0).mean()` which counts cash days (return≈0) as losses
**Fix:** Add `active_win_rate` metric that only counts days with exposure > 0.1

## P1-2: S04 Covariance-Based Vol Targeting
**Status:** pending → Subagent: general-5090
**Problem:** S04 uses component average daily volatility, not portfolio volatility
**Fix:** Implement covariance matrix estimation (rolling 60-day) and true portfolio vol targeting

## P1-3: Cross-Sectional Standardization + Sector Cap
**Status:** pending → Subagent: general-5090
**Problem:** S13/S21 mix raw factors without cross-sectional z-score; Top-N can concentrate on same sector
**Fix:** Add cross-sectional z-score normalization in signal computation; add asset class cap using etf_quality table

## Execution Order
```
P1-1 (win rate fix) ──┐
                       ├──> P1-6: Re-run backtest + verification
P1-2 (S04 cov) ───────┤
P1-3 (z-score + cap) ─┘
```

P1-1 and P1-2 can run in parallel (independent files).
P1-3 depends on understanding etf_quality schema (quick check first).
All three must complete before Phase 6 re-run.

## Files to Modify
- `src/backtest_engine.py` — P1-1: win_rate calculation
- `src/strategy_library.py` — P1-2: S04 vol targeting, P1-3: z-score + sector cap
- `src/data_loader.py` — P1-3: may need asset_class lookup function
