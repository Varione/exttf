"""Market regime detection - training set only to prevent data leakage."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from factor_definitions import FACTORS


def load_factors(csv_path: str = "data/processed/factors_all_repaired.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return df


def get_selected_factors(threshold: float = 0.5) -> list[str]:
    corr = pd.read_csv("data/processed/factor_correlation.csv", index_col=0)
    cols = corr.columns.tolist()
    idx_map = {f: i for i, f in enumerate(cols)}

    sel = [cols[0]]
    for f in cols[1:]:
        fi = idx_map[f]
        if all(abs(corr.values[fi, idx_map[s]]) < threshold for s in sel):
            sel.append(f)
    return sel


def compute_daily_factors(
    factors_df: pd.DataFrame, factor_cols: list[str], top_n: int = 100
) -> pd.DataFrame:
    """Compute daily cross-sectional means from the PIT-eligible universe."""
    sampled = factors_df
    if "pit_eligible" in sampled.columns:
        eligible = sampled["pit_eligible"]
        if eligible.dtype != bool:
            eligible = eligible.astype(str).str.lower().isin({"1", "true", "yes"})
        sampled = sampled.loc[eligible.fillna(False)]
    daily = sampled.groupby("date")[factor_cols].mean().sort_index()
    # Rolling window to smooth
    daily = daily.rolling(20, min_periods=5).mean()
    return daily.dropna()


def detect_regimes_train_only(
    daily_means: pd.DataFrame,
    train_end: str = "2018-01-01",
    n_clusters: int = 3,
):
    """Cluster using ONLY training period data, then label all dates."""
    # Split by date
    train_mask = daily_means.index < train_end
    X_train = daily_means.loc[train_mask]
    X_all = daily_means

    print(f"Training period for clustering: {X_train.index[0].date()} ~ {X_train.index[-1].date()} ({len(X_train)} days)")
    print(f"Full period: {X_all.index[0].date()} ~ {X_all.index[-1].date()} ({len(X_all)} days)")

    # Scale using training stats only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # K-means on training data only
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_labels = kmeans.fit_predict(X_train_scaled)

    print(f"\nTraining regime distribution:")
    for r in sorted(np.unique(train_labels)):
        cnt = (train_labels == r).sum()
        print(f"  Regime {r}: {cnt} days ({cnt/len(train_labels)*100:.1f}%)")

    # Label ALL dates using training kmeans
    X_all_scaled = scaler.transform(X_all)
    all_labels = kmeans.predict(X_all_scaled)

    regime_series = pd.Series(all_labels, index=X_all.index, name="regime")

    print(f"\nFull period regime distribution:")
    for r in sorted(np.unique(all_labels)):
        cnt = (all_labels == r).sum()
        print(f"  Regime {r}: {cnt} days ({cnt/len(all_labels)*100:.1f}%)")

    return regime_series, daily_means, scaler, kmeans


def characterize_regimes(
    regime_series: pd.Series,
    daily_means: pd.DataFrame,
    factor_cols: list[str],
):
    chars = []
    for regime_id in sorted(regime_series.unique()):
        mask = regime_series == regime_id
        means = daily_means[mask].mean()
        chars.append({
            "regime": regime_id,
            "days": mask.sum(),
            **{f"mean_{c}": means[c] for c in factor_cols},
        })
    return pd.DataFrame(chars)


def assign_regime_to_date(date: pd.Timestamp, regime_series: pd.Series) -> int | None:
    """Return the latest regime known on or before ``date``; never peek ahead."""
    idx = regime_series.index
    if date < idx[0] or date > idx[-1]:
        return None
    pos = int(np.searchsorted(idx, date, side="right")) - 1
    return int(regime_series.iloc[pos])


if __name__ == "__main__":
    import time
    t0 = time.time()

    df = load_factors()
    factor_cols = get_selected_factors(0.5)
    print(f"Factors: {len(factor_cols)}\n")

    daily_means = compute_daily_factors(df, factor_cols, top_n=100)
    regime_series, _, scaler, kmeans = detect_regimes_train_only(daily_means, train_end="2018-01-01", n_clusters=3)

    # Characterize each regime
    chars = characterize_regimes(regime_series, daily_means, factor_cols)
    key_factors = {
        'momentum': ['mom_5', 'mom_20', 'mom_60', 'mom_120'],
        'volatility': ['real_vol_10', 'park_vol_10', 'tr_vol_20', 'vol_ratio_5_20'],
        'trend': ['r2_20', 'r2_60', 'slope_20', 'hull_dist_20'],
        'volume': ['vol_ma_5', 'vol_trend_20', 'amount_vol_ratio', 'vp_corr_20'],
        'risk': ['mdd_60', 'omega_20', 'down_capture_20', 'tail_ratio_60'],
        'distribution': ['skew_20', 'kurt_20', 'autocorr_20'],
        'trend_str': ['adx_proxy_14', 'choppiness_14', 'trend_str_10'],
    }

    print(f"\n=== 各 Regime 特征 ===\n")
    for regime_id in sorted(regime_series.unique()):
        mask = regime_series == regime_id
        means = daily_means[mask].mean()
        print(f"Regime {regime_id}: {mask.sum()} days ({mask.sum()/len(daily_means)*100:.1f}%)")
        for cat, factors in key_factors.items():
            vals = [(f, means[f]) for f in factors if f in means.index]
            if vals:
                print(f"  [{cat}]", end=' ')
                for f, v in sorted(vals, key=lambda x: -abs(x[1]))[:3]:
                    print(f"{f}={v:.4f}", end=' ')
                print()
        print()

    # Save regime labels
    regime_series.to_csv("data/processed/regime_labels.csv")

    # Save scaler and kmeans for later use
    import joblib
    joblib.dump((scaler, kmeans), "data/processed/regime_model_cluster.joblib")

    print(f"Saved regime labels and cluster model ({time.time()-t0:.1f}s)")
