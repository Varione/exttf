# Phase 5 Task Plan: Strategy Research

## Goal
Deepen S04 strategy research and study strategy combinations using walk-forward framework.

## Phases

### Phase 5.1: S04 Parameter Sensitivity Analysis [COMPLETE]
- [x] Create configurable S04 strategy class
- [x] Run parameter grid search in walk-forward framework (81 combos)
- [x] Output `reports/strategy_research/s04_parameter_study.csv`

### Phase 5.2: S04 Signal Decomposition [COMPLETE]
- [x] Create trend-only variant
- [x] Create vol-targeting-only variant
- [x] Run all 3 variants + baseline B0
- [x] Output `reports/strategy_research/s04_component_decomposition.csv`

### Phase 5.3: Strategy Combination Research [COMPLETE]
- [x] Create `src/strategy_combiner.py` with ERC, VolScaling, RegimeConditioned
- [x] Run combination walk-forward backtest
- [x] Output `reports/strategy_research/strategy_combination_results.csv`

### Phase 5.4: Summary Report [COMPLETE]
- [x] Aggregate all results
- [x] Output `reports/strategy_research/phase5_summary.csv`

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| S04_Configurable missing _trend_lookback attr | Added to __init__ | Fixed |
| excess_vs_b0% showing 0.0 for all combos | Integer-indexed returns vs datetime-indexed B0 caused NaN alignment | Set datetime index on custom backtest DataFrame |
| avg_comp.loc.get() AttributeError | .loc has no .get method | Used direct .loc[] with conditional check |
