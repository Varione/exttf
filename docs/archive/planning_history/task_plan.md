# Task Plan - ETF Strategy Project

## Phase 0: Data Chain Reconstruction (P0)

### Phase 0-A: Core Data Artifacts (COMPLETE)
- Canonical database rebuilt: data/processed/etf.sqlite (1,549 symbols, 1,386,449 rows)
- Factor artifact regenerated: factors_all_repaired.csv (1,386,449 rows, 131 columns)
- Manifest verified: completed=true, validation_errors=[]
- Regime predictions rebuilt: regime_predictions.csv (4,948 days)
- Classification dataset constructed: classification_data.npz (4,885 samples, 1,040 features)

### Phase 0-B: Document and Status Synchronization (COMPLETE)
- progress.md rewritten with verified data sources
- findings.md cleaned of stale statistics
- task_plan.md updated to reflect real completion status
- Data consistency verified across 4 sources

### Phase 0-C: Old Script Isolation (PENDING)
- Identify all scripts bypassing new backtest rules
- Quarantine or update deprecated scripts
- Ensure all backtests use total_return_proxy price mode

## Phase 1: Modeling Fixes (COMPLETE)

### P1.1 Win Rate Fix (COMPLETE)
- Fixed look-ahead bias in signal calculation
- S04_VolTarget_Trend win rate corrected

### P1.2 Covariance Volatility Fix (COMPLETE)
- Fixed dimension mismatch in covariance-based volatility
- Ensured cross-sectional standardization consistency

### P1.3 Cross-Sectional Standardization (COMPLETE)
- Unified z-score normalization across all factors
- Fixed rolling window boundary conditions

### P1.4 Asset Class Weight Limits (COMPLETE)
- max_weight=0.05 per ETF enforced
- Single ETF position cap applied

### P1.5 Backtest Results (COMPLETE)
- 6 strategies tested: S04, S23, B0, S01, S12, S13
- Unified experiment run: 20260727_163752
- PIT status: PIT_PARTIAL

## Phase 2: External Reference Expansion (PENDING)
- Current coverage: 20 symbols (1.3%)
- Target: Expand independent reference data coverage
- Priority: High-volume ETFs first

## Phase 3: Regime Model Enhancement (PENDING)
- Current: 4,948 days of predictions
- Target: Improve regime classification accuracy
- Explore: Alternative clustering methods

## Phase 4: Data Quality Remediation (PENDING)
- 1,542 symbols have date continuity gaps
- Average gap: 3.38 days, max gap: 175 days
- Target: Fill gaps using forward-fill or interpolation

## Phase 5: Strategy Optimization (PENDING)
- Current best Sharpe: S12_Low_Volatility (0.539)
- Current best return: B0_BuyHold_EW (40.95%)
- Target: Improve risk-adjusted returns across strategies

## Phase 6: Robustness Testing (PENDING)
- Walk-forward analysis
- Parameter sensitivity analysis
- Regime-specific performance decomposition

## Phase 7: Production Pipeline (PENDING)
- Automated data refresh pipeline
- Scheduled backtest re-runs
- Monitoring and alerting

## Phase 8: Final Deliverables (PENDING)
- Comprehensive strategy report
- Production-ready codebase
- Documentation and API

---

## Data Artifacts Summary

| Artifact | Path | Stats |
|----------|------|-------|
| Canonical DB | data/processed/etf.sqlite | 1,549 symbols, 1,386,449 rows |
| Factor CSV | data/processed/factors_all_repaired.csv | 1,386,449 rows, 131 cols |
| Manifest | data/processed/factors_all_repaired.csv.manifest.json | completed=true |
| Regime | data/processed/regime_predictions.csv | 4,948 days (2006-2026) |
| Classification | data/processed/classification_data.npz | 4,885 samples |
| Backtest P1 | data/processed/backtest_results_p1.csv | 6 strategies |
| Unified Exp | reports/unified_experiment/20260727_163752/ | 5 strategies, gate passed |

## Fingerprint Reference

- DB SHA256: 3441d394e7cc8cb0f7b5f45d45e3a5329c208ee9203c2c2e95e0c679f5a46d8d
- Factor schema hash: dd4fa3c6616a54b8
- Latest run ID: 20260727_163752
- PIT status: PIT_PARTIAL
