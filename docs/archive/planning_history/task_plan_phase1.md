# Phase 1 Plan: Data Credibility and PIT Improvement

## Goal
Improve PIT from PARTIAL to FULL by adding ETF lifecycle data, classifying date gaps, and validating total_return_proxy.

## Execution Order
```
P1-1 (ETF lifecycle) ──> Gate B check
P1-2 (Gap classification) ──> P1-3 (Price validation)
```

## P1-1: ETF Lifecycle Database
**Status:** pending → Subagent: general-5090
**Problem:** Current PIT only checks observation count and turnover, not lifecycle events.
**Fix:** Create etf_lifecycle table with list_date, delist_date, status from etf_catalog + etf_daily analysis.

## P1-2: Date Gap Classification
**Status:** pending → Subagent: general-5090
**Problem:** 1,542 symbols have gaps but they're not classified (suspension vs pre-listing vs data error).
**Fix:** Classify each gap into categories using lifecycle data + exchange calendar.

## P1-3: Total Return Proxy Validation
**Status:** pending → Subagent: general-5090
**Problem:** total_return_proxy is used for all backtests but only 20 symbols have external reference.
**Fix:** Build validation report comparing proxy vs reference for available symbols.

## Constraints
- Python: D:\miniconda\envs\agents\python.exe
- Work directory: D:\etf
- Canonical DB: data/processed/etf.sqlite (do not modify)
- All changes must be reversible via git
