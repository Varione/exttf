# Unified ETF Strategy Experiment Plan

## Goal
Build and run a reproducible unified ETF strategy experiment chain with data gates, factor rebuild, backtesting, and comprehensive reporting.

## Phases

### Phase 1: Environment Setup
- [x] Explore codebase structure
- [x] Verify canonical DB (data/processed/etf.sqlite): 1,386,449 rows, 1,549 symbols
- [x] Compare root DB vs canonical DB differences
- [x] Confirm factor artifact is stale (1,384,712 rows, no manifest)

### Phase 2: Create Unified Runner and Config
- [ ] Create `config/unified_experiment.json` with fixed parameters
- [ ] Create `src/run_unified_experiment.py` with:
  - Canonical DB lock (SHA256 fingerprint)
  - Data gate checks (data_mode, price_mode, require_pit)
  - Factor rebuild trigger
  - Regime/label rebuild
  - Backtest execution for fixed strategies
  - Metrics computation (full period + OOS windows)
  - Report output with manifest

### Phase 3: Factor Rebuild
- [ ] Run `compute_all_factors` with resume_from_checkpoint=false
- [ ] Validate: 1,386,449 rows, 1,549 symbols, 131 columns
- [ ] Verify manifest completed=true

### Phase 4: Regime and Classification Rebuild
- [ ] Rebuild regime labels/predictions from fresh factors
- [ ] Ensure no look-ahead bias

### Phase 5: Backtest Execution
- [ ] Run fixed strategies: S04_VolTarget_Trend, S23_Oversold_Long, S01_CS_Momentum, B0_BuyHold_EW, B1_Momentum_Agnostic
- [ ] Fixed params: n_hold=20, max_weight=0.05, signal_to_return_lag=2, rebalance_every=5

### Phase 6: Reporting
- [ ] Generate reports/unified_experiment/<run_id>/ with all outputs
- [ ] experiment_manifest.json, summary.csv, per-strategy daily CSVs, run.log

### Phase 7: Testing
- [ ] Create tests/test_unified_experiment.py
- [ ] Run pytest verification

## Key Constraints
- Canonical DB ONLY: data/processed/etf.sqlite
- Data gate fail-closed: data_mode=etf, price_mode=total_return_proxy, require_pit=true
- No git reset/checkout/clean
- No commit/push
- Do not overwrite stale backtest_results files
