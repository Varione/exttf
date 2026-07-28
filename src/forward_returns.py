"""Compute forward returns by regime using factor data directly."""

import numpy as np
import pandas as pd

# Load regime predictions
pred_df = pd.read_csv("data/processed/regime_predictions.csv", parse_dates=["date"])
print(f"Regime predictions: {pred_df.shape[0]} days")
print(f"Period: {pred_df['date'].min().date()} ~ {pred_df['date'].max().date()}")
print()

# Load factor data (contains all we need)
factors = pd.read_csv("data/processed/factors_all_repaired.csv", parse_dates=["date"])
top_etfs = factors["symbol"].value_counts().head(100).index.tolist()
factors = factors[factors["symbol"].isin(top_etfs)]

# Merge regime labels
pred_df["date_str"] = pred_df["date"].dt.strftime("%Y-%m-%d")
factors["date_str"] = factors["date"].astype(str).str[:10]
merged = factors.merge(pred_df[["date_str", "regime"]], on="date_str", how="inner")
print(f"Merged data: {merged.shape[0]} rows ({merged['symbol'].nunique()} ETFs)")
print()

# Use momentum factors as return proxies
# mom_5 = 5-day return, mom_20 = 20-day return, etc.
# These are already computed in the factor definitions

print("=" * 70)
print("FORWARD RETURN ANALYSIS BY REGIME (using momentum as proxy)")
print("=" * 70)

for regime_id in [0, 1, 2]:
    r_data = merged[merged["regime"] == regime_id]
    n_days = r_data["date"].nunique()
    n_etfs = r_data["symbol"].nunique()

    print(f"\nRegime {regime_id}: {n_days} trading days, {n_etfs} ETFs")
    print(f"  {'Horizon':<10} {'Mean%':>8} {'Median%':>9} {'Std%':>7} {'PosRate':>8}")

    # mom_5 ~ 5-day return, mom_20 ~ 20-day return
    for col, label in [("mom_5", "5d"), ("mom_20", "20d"), ("mom_60", "60d")]:
        vals = r_data[col].dropna()
        if len(vals) > 0:
            mean_ret = vals.mean() * 100
            med_ret = vals.median() * 100
            std_ret = vals.std() * 100
            pos_rate = (vals > 0).mean() * 100
            print(f"  {label:<10} {mean_ret:>+7.3f} {med_ret:>+8.3f} {std_ret:>+6.3f} {pos_rate:>7.1f}%")

# Annualized returns by regime
print(f"\n{'='*60}")
print("ANNUALIZED RETURNS BY REGIME")
print(f"{'='*60}\n")

for regime_id in [0, 1, 2]:
    r_data = merged[merged["regime"] == regime_id]
    daily_mom = r_data.groupby("date")["mom_5"].mean().dropna()
    if len(daily_mom) > 20:
        # mom_5 is already a 5-day return, so annualize differently
        ann_ret = daily_mom.mean() * (252/5) * 100
        ann_vol = daily_mom.std() * np.sqrt(252/5) * 100
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        n_days = len(daily_mom)
        print(f"Regime {regime_id}: {n_days} days | AnnRet={ann_ret:+.2f}% AnnVol={ann_vol:.2f}% Sharpe={sharpe:.2f}")

# Strategy-specific analysis
print(f"\n{'='*60}")
print("STRATEGY-SPECIFIC ANALYSIS")
print(f"{'='*60}\n")

for regime_id in [0, 1, 2]:
    r_dates = pred_df[pred_df["regime"] == regime_id]["date_str"].tolist()
    r_factors = merged[merged["regime"] == regime_id]

    if len(r_factors) < 100:
        continue

    print(f"Regime {regime_id}:")

    if regime_id == 0:
        # Mean reversion: buy low ibias (price below MA)
        signals = r_factors.groupby("symbol").agg({
            "ibias_20": "mean", "mr_speed_20": "mean", "mom_5": "mean"
        }).dropna()
        bottom10 = signals["ibias_20"].nsmallest(10)
        print(f"  [Mean Reversion] Oversold ETFs (bottom 10 by ibias_20):")
        for sym, val in bottom10.items():
            fwd = r_factors[(r_factors["symbol"] == sym) & (r_factors["mom_5"].notna())]["mom_5"]
            avg_fwd = fwd.mean() * 100 if len(fwd) > 0 else 0
            print(f"    {sym}: ibias={val:.4f} avg_fwd_5d={avg_fwd:+.2f}%")

    elif regime_id == 1:
        # Momentum: buy high mom_20
        signals = r_factors.groupby("symbol").agg({
            "mom_20": "mean", "win_rate_20": "mean", "slope_20": "mean"
        }).dropna()
        top10 = signals["mom_20"].nlargest(10)
        print(f"  [Momentum] Strong momentum ETFs (top 10 by mom_20):")
        for sym, val in top10.items():
            fwd = r_factors[(r_factors["symbol"] == sym) & (r_factors["mom_5"].notna())]["mom_5"]
            avg_fwd = fwd.mean() * 100 if len(fwd) > 0 else 0
            print(f"    {sym}: mom_20={val:.4f} avg_fwd_5d={avg_fwd:+.2f}%")

    else:
        # Defensive: low vol + high omega
        signals = r_factors.groupby("symbol").agg({
            "real_vol_10": "mean", "omega_20": "mean", "mom_5": "mean"
        }).dropna()
        signals = signals[signals["real_vol_10"] > 0].copy()
        signals["def_score"] = -signals["real_vol_10"] + (signals["omega_20"] - signals["omega_20"].mean()) / (signals["omega_20"].std() + 1e-8)
        top10 = signals["def_score"].nlargest(10)
        print(f"  [Defensive] Best defensive ETFs (top 10 by def_score):")
        for sym, val in top10.items():
            fwd = r_factors[(r_factors["symbol"] == sym) & (r_factors["mom_5"].notna())]["mom_5"]
            avg_fwd = fwd.mean() * 100 if len(fwd) > 0 else 0
            print(f"    {sym}: score={val:.3f} vol={signals.loc[sym, 'real_vol_10']:.4f} omega={signals.loc[sym, 'omega_20']:.2f} avg_fwd_5d={avg_fwd:+.2f}%")

    print()
