# Progress Log

## Session 2026-07-27

### Phase 0-A: Data Chain Reconstruction (COMPLETE)

Date: 2026-07-27
Status: Complete

#### Execution Summary
- Canonical database rebuilt from 1,549 CSV files
- Factor artifact regenerated with full repair pipeline
- Regime predictions rebuilt from scratch
- Classification dataset constructed

#### Key Results
**Database (data/processed/etf.sqlite)**
- Symbols: 1,549
- Total rows: 1,386,449
- Date range: 2005-02-23 ~ 2026-07-17
- SHA256 fingerprint: 3441d394e7cc8cb0f7b5f45d45e3a5329c208ee9203c2c2e95e0c679f5a46d8d

**Factor Artifact (data/processed/factors_all_repaired.csv)**
- Rows: 1,386,449 | Symbols: 1,549 | Columns: 131
- Manifest completed: true
- Schema hash: dd4fa3c6616a54b8
- Price mode: total_return_proxy
- Validation errors: 0

**Regime Predictions (data/processed/regime_predictions.csv)**
- Days: 4,948
- Date range: 2006-03-10 ~ 2026-07-17

**Classification Dataset (data/processed/classification_data.npz)**
- Samples: 4,885
- Features: 1,040 (X shape 4885x1040)
- Factor columns: 52
- Date range: 2006-04-03 ~ end

### Phase 0-B: Document and Status Synchronization (IN PROGRESS)

Date: 2026-07-27
Status: In progress

#### Actions Taken
1. Verified all data artifacts against manifest
2. Rewritten progress.md with verified data
3. Updated findings.md with current facts
4. Updated task_plan.md with real completion status

### Phase 0-C: Old Script Isolation (IN PROGRESS)

Date: 2026-07-27
Status: Pending

---

## Phase 1: Modeling Fixes (COMPLETE)

Date: 2026-07-27
Status: Complete

#### Key Results
**Backtest Results (data/processed/backtest_results_p1.csv)**
- Strategies tested: 6
- Period: 2018-01-01 ~ 2026-07-17

| Strategy | Total Return | Sharpe | Max DD |
|----------|-------------|--------|--------|
| S04_VolTarget_Trend | 20.38% | 0.389 | -11.06% |
| S23_Oversold_Long | -6.60% | -0.355 | -9.97% |
| B0_BuyHold_EW | 40.95% | 0.396 | -25.13% |
| S01_CS_Momentum | 15.84% | 0.329 | -12.35% |
| S12_Low_Volatility | 15.22% | 0.539 | -6.52% |
| S13_Quality | 24.18% | 0.282 | -20.56% |

**Unified Experiment (reports/unified_experiment/20260727_163752/)**
- Run ID: 20260727_163752
- Gate passed: true
- Strategies: S04, S23, S01, B0, B1 (5 strategies)
- PIT status: PIT_PARTIAL
- OOS windows: 2018-2021, 2022-2024, 2025-2026

### Phase 4: Data Quality Audit (COMPLETE)

Date: 2026-07-27
Status: Complete

#### Key Results
1. Validation status: All PASS (1,386,449 rows)
2. External reference coverage: 20 symbols (31,175 rows, 1.3% coverage)
3. Price mode distribution: total_return_proxy 100%
4. Date continuity gaps:
   - Symbols with gaps: 1,542
   - Max gap: 175 days
   - Avg gap: 3.38 days
