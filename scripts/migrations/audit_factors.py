"""Phase 4: Factor quality audit, de-redundancy, and core factor set selection."""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from factor_definitions import FACTORS, get_all_categories

warnings.filterwarnings("ignore")

FACTOR_CSV = "data/processed/factors_all_repaired.csv"
REGIME_CSV = "data/processed/regime_predictions.csv"
OUTPUT_DIR = "reports/factor_research"
BACKTEST_START = "2018-01-01"
BACKTEST_END = "2026-07-17"


def get_factor_category(name: str) -> str:
    for f in FACTORS:
        if f.name == name:
            return f.category
    return "unknown"


def fast_rank_corr(fv: np.ndarray, rv: np.ndarray) -> tuple:
    """Compute both Pearson and rank correlation. Returns (ic, ric) or (nan, nan)."""
    n = len(fv)
    if n < 30:
        return np.nan, np.nan
    
    # Rank using argsort of argsort
    rf = np.empty(n, dtype=float)
    rr = np.empty(n, dtype=float)
    ord_f = np.argsort(fv)
    ord_r = np.argsort(rv)
    rf[ord_f] = np.arange(n, dtype=float)
    rr[ord_r] = np.arange(n, dtype=float)
    
    # Pearson IC
    mf = fv.mean()
    mr = rv.mean()
    cov_pr = np.sum((fv - mf) * (rv - mr))
    sf = np.sqrt(np.sum((fv - mf) ** 2))
    sr = np.sqrt(np.sum((rv - mr) ** 2))
    ic = cov_pr / (sf * sr) if sf > 0 and sr > 0 else np.nan
    
    # Rank IC
    mrf = rf.mean()
    mrr = rr.mean()
    cov_rk = np.sum((rf - mrf) * (rr - mrr))
    sfr = np.sqrt(np.sum((rf - mrf) ** 2))
    srr = np.sqrt(np.sum((rr - mrr) ** 2))
    ric = cov_rk / (sfr * srr) if sfr > 0 and srr > 0 else np.nan
    
    return ic, ric


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ============================================================
    #  Step 1: Load data
    # ============================================================
    print("=" * 70)
    print("Step 1: Loading data")
    print("=" * 70)

    factors = pd.read_csv(FACTOR_CSV, parse_dates=["date"])
    factors["symbol"] = factors["symbol"].astype(str).str.zfill(6)
    factor_cols = [f.name for f in FACTORS if f.name in factors.columns]
    print(f"  Factors: {len(factor_cols)}, Rows: {len(factors):,}")

    regimes = pd.read_csv(REGIME_CSV, parse_dates=["date"])
    regime_map = dict(zip(regimes["date"].dt.strftime("%Y-%m-%d"), regimes["regime"]))

    # Load prices for forward returns
    from data_loader import load_price_series
    prices = load_price_series("data/processed/etf.sqlite", data_mode="etf", price_mode="total_return_proxy")
    prices["symbol"] = prices["symbol"].astype(str).str.zfill(6)
    prices["date"] = pd.to_datetime(prices["date"])

    # Forward returns in long format
    prices = prices.sort_values(["symbol", "date"])
    prices["daily_ret"] = prices.groupby("symbol")["price"].pct_change().fillna(0.0)
    prices["fwd_1"] = prices.groupby("symbol")["daily_ret"].shift(-1).fillna(0.0)
    prices["fwd_5"] = prices.groupby("symbol")["daily_ret"].shift(-5).fillna(0.0)
    prices["fwd_20"] = prices.groupby("symbol")["daily_ret"].shift(-20).fillna(0.0)

    # Merge
    factors = factors.merge(
        prices[["symbol", "date", "fwd_1", "fwd_5", "fwd_20"]],
        on=["symbol", "date"],
        how="left",
    )
    factors[["fwd_1", "fwd_5", "fwd_20"]] = factors[["fwd_1", "fwd_5", "fwd_20"]].fillna(0.0)
    factors["date_str"] = factors["date"].dt.strftime("%Y-%m-%d")
    factors["regime"] = factors["date_str"].map(regime_map).fillna(-1).astype(int)

    # Pre-group by date (as dict for fast lookup)
    print("  Pre-grouping by date...")
    date_groups = {}
    for _, grp in factors.groupby("date", sort=True, observed=False):
        date_groups[grp["date"].iloc[0]] = grp
    all_dates = sorted(date_groups.keys())
    print(f"  Dates: {len(all_dates)}, Merged rows: {len(factors):,}")

    # Pre-convert factor columns to numeric
    for col in factor_cols:
        factors[col] = pd.to_numeric(factors[col], errors="coerce")

    # ============================================================
    #  Step 2: Factor quality audit
    # ============================================================
    audit_path = os.path.join(OUTPUT_DIR, "factor_quality_audit.csv")
    if os.path.exists(audit_path):
        print(f"\n  Audit file exists, reloading from {audit_path}")
        audit_df = pd.read_csv(audit_path)
        print(f"  Loaded {len(audit_df)} factor audits")
        skip_audit = True
    else:
        skip_audit = False

    if not skip_audit:
        print("\n" + "=" * 70)
        print("Step 2: Factor quality audit")
        print("=" * 70)

        reports = []

        for idx, col in enumerate(factor_cols):
            if (idx + 1) % 20 == 0:
                print(f"  {idx + 1}/{len(factor_cols)}: {col}", end=" ", flush=True)

            # Coverage
            col_vals = factors[col]
            valid = col_vals.notna() & (col_vals != 0)
            coverage = float(valid.mean())

            # Cross-sectional dispersion (sampled for speed)
            cs_std_sum = 0.0
            cs_std_count = 0
            for d in all_dates[::3]:
                grp = date_groups[d]
                fv = grp[col].values
                vm = np.isfinite(fv) & (fv != 0)
                if vm.sum() > 5:
                    cs_std_sum += np.std(fv[vm], ddof=1)
                    cs_std_count += 1
            cs_dispersion = cs_std_sum / cs_std_count if cs_std_count > 0 else 0.0

            # IC computation per date
            ic_list = []
            ric_list = []
            for d in all_dates:
                grp = date_groups[d]
                fv = grp[col].values
                rv = grp["fwd_1"].values
                joint = np.isfinite(fv) & np.isfinite(rv) & (fv != 0)
                if joint.sum() < 30:
                    continue
                ic, ric = fast_rank_corr(fv[joint], rv[joint])
                if np.isfinite(ic):
                    ic_list.append(ic)
                if np.isfinite(ric):
                    ric_list.append(ric)

            ic_mean = float(np.mean(ic_list)) if ic_list else 0.0
            ic_std = float(np.std(ic_list, ddof=1)) if len(ic_list) > 1 else 0.0
            icir = ic_mean / ic_std if ic_std > 1e-10 else 0.0
            rank_ic_mean = float(np.mean(ric_list)) if ric_list else 0.0
            rank_ic_std = float(np.std(ric_list, ddof=1)) if len(ric_list) > 1 else 0.0
            rank_icir = rank_ic_mean / rank_ic_std if rank_ic_std > 1e-10 else 0.0

            # Turnover (sampled)
            rank_changes = []
            prev_ranks = None
            for d in all_dates[::max(1, len(all_dates) // 200)]:
                grp = date_groups[d]
                fv = grp[col].values
                vm = np.isfinite(fv) & (fv != 0)
                if vm.sum() < 20:
                    continue
                ranks = pd.Series(fv[vm]).rank().values
                if prev_ranks is not None:
                    ml = min(len(ranks), len(prev_ranks))
                    rank_changes.append(float(np.mean(np.abs(ranks[:ml] - prev_ranks[:ml]))))
                prev_ranks = ranks
            turnover = float(np.mean(rank_changes)) if rank_changes else 0.0

            # Decay (sampled dates for speed)
            decay_5p = 0.0
            decay_20p = 0.0
            ric_5_list = []
            ric_20_list = []
            for d in all_dates[::max(1, len(all_dates) // 100)]:
                grp = date_groups[d]
                fv = grp[col].values

                rv5 = grp["fwd_5"].values
                j5 = np.isfinite(fv) & np.isfinite(rv5) & (fv != 0)
                if j5.sum() >= 30:
                    _, r5 = fast_rank_corr(fv[j5], rv5[j5])
                    if np.isfinite(r5):
                        ric_5_list.append(r5)

                rv20 = grp["fwd_20"].values
                j20 = np.isfinite(fv) & np.isfinite(rv20) & (fv != 0)
                if j20.sum() >= 30:
                    _, r20 = fast_rank_corr(fv[j20], rv20[j20])
                    if np.isfinite(r20):
                        ric_20_list.append(r20)

            decay_5p = float(np.mean(ric_5_list)) if ric_5_list else 0.0
            decay_20p = float(np.mean(ric_20_list)) if ric_20_list else 0.0

            # Regime stability (sampled)
            ic_by_regime = {}
            for reg in [0, 1, 2]:
                ric_reg = []
                for d in all_dates[::max(1, len(all_dates) // 150)]:
                    grp = date_groups[d]
                    reg_mask = grp["regime"] == reg
                    if reg_mask.sum() < 20:
                        continue
                    fv = grp.loc[reg_mask, col].values
                    rv = grp.loc[reg_mask, "fwd_1"].values
                    joint = np.isfinite(fv) & np.isfinite(rv) & (fv != 0)
                    if joint.sum() < 20:
                        continue
                    _, r = fast_rank_corr(fv[joint], rv[joint])
                    if np.isfinite(r):
                        ric_reg.append(r)
                ic_by_regime[reg] = float(np.mean(ric_reg)) if ric_reg else 0.0

            reports.append({
                "factor": col,
                "category": get_factor_category(col),
                "coverage": round(coverage, 4),
                "missing_ratio": round(1.0 - coverage, 4),
                "cs_dispersion": round(cs_dispersion, 6),
                "ic_mean": round(ic_mean, 6),
                "ic_std": round(ic_std, 6),
                "icir": round(icir, 4),
                "rank_ic_mean": round(rank_ic_mean, 6),
                "rank_ic_std": round(rank_ic_std, 6),
                "rank_icir": round(rank_icir, 4),
                "turnover": round(turnover, 4),
                "decay_1p": round(rank_ic_mean, 6),
                "decay_5p": round(decay_5p, 6),
                "decay_20p": round(decay_20p, 6),
                "ic_regime_0": round(ic_by_regime[0], 6),
                "ic_regime_1": round(ic_by_regime[1], 6),
                "ic_regime_2": round(ic_by_regime[2], 6),
            })
            print(f"rank_icir={rank_icir:.4f}")

        audit_df = pd.DataFrame(reports)
        audit_df.to_csv(audit_path, index=False)
        print(f"\n  Audit saved: {len(audit_df)} factors -> {audit_path}")

    top20 = audit_df.nlargest(20, "rank_icir")[["factor", "category", "coverage",
            "cs_dispersion", "ic_mean", "rank_ic_mean", "icir", "rank_icir",
            "turnover", "decay_1p", "decay_5p", "decay_20p"]]
    print("\n  === Top 20 factors by Rank ICIR ===")
    print(top20.to_string(index=False))

    # ============================================================
    #  Step 3: De-redundancy
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 3: De-redundancy and core factor selection")
    print("=" * 70)

    cat_to_factors = {}
    for col in factor_cols:
        cat = get_factor_category(col)
        cat_to_factors.setdefault(cat, []).append(col)

    for cat, flist in sorted(cat_to_factors.items()):
        print(f"  [{cat}] {len(flist)} factors")

    # --- 3.1: Within-group correlation filtering ---
    print("\n  --- Correlation filtering (threshold=0.8) ---")

    sample_rows = factors.sample(min(50000, len(factors)), random_state=42)
    cs_data = {}
    for col in factor_cols:
        v = sample_rows[col].dropna().values
        if len(v) > 100 and np.isfinite(v).all():
            cs_data[col] = v

    removed_corr_count = 0
    remaining_by_cat = {}

    for cat, flist in cat_to_factors.items():
        available = [f for f in flist if f in cs_data]
        if not available:
            remaining_by_cat[cat] = []
            continue

        icir_map = {}
        for f in available:
            row = audit_df[audit_df["factor"] == f]
            icir_map[f] = abs(row["rank_icir"].iloc[0]) if len(row) > 0 else 0.0

        available_sorted = sorted(available, key=lambda x: icir_map[x], reverse=True)
        selected = [available_sorted[0]]

        for f in available_sorted[1:]:
            is_redundant = False
            for s in selected:
                v1 = cs_data.get(f, [])
                v2 = cs_data.get(s, [])
                ml = min(len(v1), len(v2))
                if ml < 50:
                    continue
                r = np.corrcoef(v1[:ml], v2[:ml])[0, 1]
                if np.isnan(r):
                    r = 0.0
                if abs(r) > 0.8:
                    is_redundant = True
                    removed_corr_count += 1
                    break
            if not is_redundant:
                selected.append(f)

        remaining_by_cat[cat] = selected

    n_after_corr = sum(len(v) for v in remaining_by_cat.values())
    print(f"  {len(factor_cols)} -> {n_after_corr} (removed {removed_corr_count} correlated)")

    # --- 3.2: IC stability screening ---
    print("\n  --- IC stability screening ---")

    all_selected = []
    removed_unstable = []

    for cat, flist in remaining_by_cat.items():
        for f in flist:
            row = audit_df[audit_df["factor"] == f]
            if len(row) == 0:
                continue
            r = row.iloc[0]

            ic_vals = [r["ic_regime_0"], r["ic_regime_1"], r["ic_regime_2"]]
            ic_vals = [v for v in ic_vals if abs(v) > 1e-8]

            if len(ic_vals) >= 2:
                signs = np.sign(ic_vals)
                if not (np.all(signs == signs[0]) or np.all(signs == -signs[0])):
                    removed_unstable.append(f)
                    continue

            if abs(r["rank_icir"]) < 0.01:
                removed_unstable.append(f)
                continue

            all_selected.append(f)

    print(f"  {n_after_corr} -> {len(all_selected)} (removed {len(removed_unstable)} unstable)")

    # --- 3.3: OOS stability screening ---
    print("\n  --- OOS stability screening ---")

    def compute_ic_window(factor_col, start, end, max_dates=80):
        mask = (factors["date_str"] >= start) & (factors["date_str"] <= end)
        dates_in_w = sorted(factors.loc[mask, "date"].unique())
        if len(dates_in_w) < 10:
            return 0.0
        sampled = dates_in_w[::max(1, len(dates_in_w) // max_dates)]

        ic_vals = []
        for d in sampled:
            grp = date_groups.get(d)
            if grp is None or len(grp) < 30:
                continue
            fv = grp[factor_col].values
            rv = grp["fwd_1"].values
            joint = np.isfinite(fv) & np.isfinite(rv) & (fv != 0)
            if joint.sum() < 30:
                continue
            _, ric = fast_rank_corr(fv[joint], rv[joint])
            if np.isfinite(ric):
                ic_vals.append(ric)

        return float(np.mean(ic_vals)) if ic_vals else 0.0

    oos_stable = []
    removed_oos = []

    for f in all_selected:
        ic_train = compute_ic_window(f, "2018-01-01", "2022-12-31")
        ic_test = compute_ic_window(f, "2023-01-01", BACKTEST_END)

        if ic_train == 0.0 or ic_test == 0.0:
            removed_oos.append(f)
            continue

        if np.sign(ic_train) == np.sign(ic_test):
            oos_stable.append(f)
        else:
            removed_oos.append(f)

    n_after_oos = len(oos_stable)
    print(f"  {len(all_selected)} -> {n_after_oos} (removed {len(removed_oos)} OOS-unstable)")

    # --- Ensure S04 required factors are included (unconditional) ---
    s04_required = {"mom_20", "real_vol_10", "ma_dist_60"}
    for req in s04_required:
        if req not in oos_stable:
            oos_stable.append(req)
            print(f"  Added S04-required factor: {req}")

    # --- Cap at 15, protecting S04-required factors ---
    if len(oos_stable) > 15:
        protected = [f for f in oos_stable if f in s04_required]
        others = [f for f in oos_stable if f not in s04_required]
        others = sorted(others, key=lambda f: abs(
            audit_df.loc[audit_df["factor"] == f, "rank_icir"].iloc[0]
        ), reverse=True)[:max(0, 15 - len(protected))]
        oos_stable = protected + others
        print(f"  Capped at {len(oos_stable)} factors ({len(protected)} S04-protected)")

    if len(oos_stable) < 5:
        extra = [f for f in all_selected if f not in oos_stable]
        extra = sorted(extra, key=lambda f: abs(
            audit_df.loc[audit_df["factor"] == f, "rank_icir"].iloc[0]
        ), reverse=True)
        oos_stable.extend(extra[:max(0, 5 - len(oos_stable))])
        print(f"  Relaxed filter to reach {len(oos_stable)} factors")

    core_factors = oos_stable
    print(f"\n  === CORE FACTOR SET: {len(core_factors)} factors ===")

    # Save core factor set
    core_df = audit_df[audit_df["factor"].isin(core_factors)].copy()
    for f in core_factors:
        mask_f = core_df["factor"] == f
        ic_train = compute_ic_window(f, "2018-01-01", "2022-12-31")
        ic_test = compute_ic_window(f, "2023-01-01", BACKTEST_END)
        core_df.loc[mask_f, "ic_train"] = round(ic_train, 6)
        core_df.loc[mask_f, "ic_test"] = round(ic_test, 6)

    core_output = core_df[["factor", "category", "coverage", "cs_dispersion",
                            "rank_ic_mean", "rank_icir", "turnover",
                            "decay_1p", "decay_5p", "decay_20p",
                            "ic_regime_0", "ic_regime_1", "ic_regime_2"]].copy()
    core_output = core_output.sort_values("rank_icir", ascending=False)
    core_path = os.path.join(OUTPUT_DIR, "core_factor_set.csv")
    core_output.to_csv(core_path, index=False)
    print(f"\n  Saved to {core_path}")
    print(core_output[["factor", "category", "rank_icir"]].to_string(index=False))

    print(f"\n  === Reduction: {len(factor_cols)} -> {n_after_corr} -> {len(all_selected)} -> {len(core_factors)} ===")

    # ============================================================
    #  Step 4: S04 backtest comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 4: S04 backtest comparison (full vs core)")
    print("=" * 70)

    from backtest_engine import BacktestEngine

    factors_full = pd.read_csv(FACTOR_CSV, parse_dates=["date"])
    factors_full["symbol"] = factors_full["symbol"].astype(str).str.zfill(6)

    print("\n  --- S04 with ALL factors ---")
    engine_full = BacktestEngine(
        factors_df=factors_full,
        fee_rate_per_side=0.0003,
        slippage_rate_per_side=0.0002,
        require_pit=True,
        require_full_pit=False,
    )

    daily_full = engine_full.run_backtest(
        "S04_VolTarget_Trend",
        start=BACKTEST_START,
        end=BACKTEST_END,
        n_hold=20,
        max_weight=0.05,
        signal_to_return_lag=2,
        rebalance_every=5,
        respect_regime=True,
    )

    metrics_full = engine_full.calculate_metrics(
        daily_full["return"],
        daily_full["turnover"],
        daily_full["transaction_cost"],
        daily_full["gross_return"],
        daily_full["exposure"],
        daily_full.get("cash_weight", None),
        dates=pd.to_datetime(daily_full["date"]),
    )

    # Core factor backtest
    meta_cols = ["symbol", "date", "pit_eligible", "pit_observations",
                 "pit_median_amount_60d", "reference_verified", "price_mode"]
    core_col_list = meta_cols + [f for f in core_factors if f in factors_full.columns]

    print(f"\n  --- S04 with CORE ({len(core_factors)}) factors ---")
    factors_core = factors_full[core_col_list].copy()

    engine_core = BacktestEngine(
        factors_df=factors_core,
        fee_rate_per_side=0.0003,
        slippage_rate_per_side=0.0002,
        require_pit=True,
        require_full_pit=False,
    )

    daily_core = engine_core.run_backtest(
        "S04_VolTarget_Trend",
        start=BACKTEST_START,
        end=BACKTEST_END,
        n_hold=20,
        max_weight=0.05,
        signal_to_return_lag=2,
        rebalance_every=5,
        respect_regime=True,
    )

    metrics_core = engine_core.calculate_metrics(
        daily_core["return"],
        daily_core["turnover"],
        daily_core["transaction_cost"],
        daily_core["gross_return"],
        daily_core["exposure"],
        daily_core.get("cash_weight", None),
        dates=pd.to_datetime(daily_core["date"]),
    )

    # Comparison table
    comparison = pd.DataFrame([
        {
            "setup": "Full (all factors)",
            "n_factors": len(factor_cols),
            "CAGR%": round(metrics_full.get("CAGR%", 0), 2),
            "Sharpe": round(metrics_full.get("Sharpe", 0), 4),
            "Max_DD%": round(metrics_full.get("Max_Drawdown%", 0), 2),
            "Sortino": round(metrics_full.get("Sortino", 0), 4),
            "Calmar": round(metrics_full.get("Calmar", 0), 4),
            "Turnover%": round(metrics_full.get("Turnover%", 0), 2),
            "Cost_Drag%": round(metrics_full.get("Transaction_Cost_Drag%", 0), 2),
        },
        {
            "setup": f"Core ({len(core_factors)} factors)",
            "n_factors": len(core_factors),
            "CAGR%": round(metrics_core.get("CAGR%", 0), 2),
            "Sharpe": round(metrics_core.get("Sharpe", 0), 4),
            "Max_DD%": round(metrics_core.get("Max_Drawdown%", 0), 2),
            "Sortino": round(metrics_core.get("Sortino", 0), 4),
            "Calmar": round(metrics_core.get("Calmar", 0), 4),
            "Turnover%": round(metrics_core.get("Turnover%", 0), 2),
            "Cost_Drag%": round(metrics_core.get("Transaction_Cost_Drag%", 0), 2),
        },
    ])

    print("\n  === Full vs Core Backtest Comparison (S04) ===")
    print(comparison.to_string(index=False))

    # OOS windows
    oos_windows = [
        {"name": "OOS_2018_2021", "start": "2018-01-01", "end": "2021-12-31"},
        {"name": "OOS_2022_2024", "start": "2022-01-01", "end": "2024-12-31"},
        {"name": "OOS_2025_2026", "start": "2025-01-01", "end": "2026-07-17"},
    ]

    oos_results = []
    for w in oos_windows:
        mf = engine_full.calculate_metrics(
            daily_full.loc[(daily_full["date"] >= w["start"]) & (daily_full["date"] <= w["end"]), "return"]
        )
        mc = engine_core.calculate_metrics(
            daily_core.loc[(daily_core["date"] >= w["start"]) & (daily_core["date"] <= w["end"]), "return"]
        )
        oos_results.append({
            "window": w["name"],
            "full_cagr%": round(mf.get("CAGR%", 0), 2),
            "full_sharpe": round(mf.get("Sharpe", 0), 4),
            "full_dd%": round(mf.get("Max_Drawdown%", 0), 2),
            "core_cagr%": round(mc.get("CAGR%", 0), 2),
            "core_sharpe": round(mc.get("Sharpe", 0), 4),
            "core_dd%": round(mc.get("Max_Drawdown%", 0), 2),
        })

    if oos_results:
        print("\n  === OOS Window Comparison ===")
        print(pd.DataFrame(oos_results).to_string(index=False))

    # Save all results
    comparison.to_csv(os.path.join(OUTPUT_DIR, "core_vs_full_comparison.csv"), index=False)
    daily_full.to_csv(os.path.join(OUTPUT_DIR, "s04_full_daily.csv"), index=False)
    daily_core.to_csv(os.path.join(OUTPUT_DIR, "s04_core_daily.csv"), index=False)
    if oos_results:
        pd.DataFrame(oos_results).to_csv(os.path.join(OUTPUT_DIR, "core_vs_full_oos.csv"), index=False)

    sharpe_diff = abs(metrics_full.get("Sharpe", 0) - metrics_core.get("Sharpe", 0))
    cagr_diff = abs(metrics_full.get("CAGR%", 0) - metrics_core.get("CAGR%", 0))
    print(f"\n  === Information Loss Assessment ===")
    print(f"  Sharpe diff: {sharpe_diff:.4f}, CAGR diff: {cagr_diff:.2f}%")

    if sharpe_diff < 0.1 and cagr_diff < 1.0:
        print("  Result: Core set preserves information (minimal loss)")
    elif sharpe_diff < 0.3 and cagr_diff < 3.0:
        print("  Result: Acceptable information loss")
    else:
        print("  WARNING: Significant information loss detected")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 4 complete in {elapsed:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
