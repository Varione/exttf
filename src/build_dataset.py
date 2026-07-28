"""Classification dataset builder: factor features -> next-day regime."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from detect_regimes import (
    load_factors, get_selected_factors, compute_daily_factors, detect_regimes_train_only, assign_regime_to_date
)


def build_classification_dataset(
    factors_df: pd.DataFrame,
    factor_cols: list[str],
    regime_series: pd.Series,
    lookback: int = 5,
    forward: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build supervised classification dataset.

    For each date, use cross-sectional mean of factors over [date-lookback+1, date]
    to predict the regime at date+forward.
    """
    sampled = factors_df
    if "pit_eligible" in sampled.columns:
        eligible = sampled["pit_eligible"]
        if eligible.dtype != bool:
            eligible = eligible.astype(str).str.lower().isin({"1", "true", "yes"})
        sampled = sampled.loc[eligible.fillna(False)]

    daily = sampled.groupby("date")[factor_cols].mean()
    daily = daily.sort_index()

    features = []
    labels = []
    dates = []

    n = len(daily)
    for i in range(lookback, n - forward):
        date = daily.index[i]

        feat_window = daily.iloc[i - lookback + 1: i + 1].values
        if np.any(np.isnan(feat_window)) or np.any(np.isinf(feat_window)):
            continue

        # ``forward`` is measured in observed trading rows, not calendar days.
        future_date = daily.index[i + forward]
        regime = regime_series.get(future_date)
        if regime is None or pd.isna(regime):
            continue

        features.append(feat_window.flatten())
        labels.append(regime)
        dates.append(date)

    X = np.array(features, dtype=np.float64)
    y = np.array(labels, dtype=np.int64)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y, np.array(dates)


def split_time_series(
    X: np.ndarray, y: np.ndarray, dates: np.ndarray,
    train_end: str = "2018-01-01",
    val_end: str = "2022-01-01",
) -> dict:
    """Split into train/val/test by date."""
    dates_dt = pd.to_datetime(dates)
    train_mask = dates_dt < train_end
    val_mask = (dates_dt >= train_end) & (dates_dt < val_end)
    test_mask = dates_dt >= val_end

    splits = {
        "train": (X[train_mask], y[train_mask]),
        "val": (X[val_mask], y[val_mask]),
        "test": (X[test_mask], y[test_mask]),
    }
    for name, (xs, ys) in splits.items():
        print(f"  {name}: {xs.shape[0]} samples, class dist: {dict(zip(*np.unique(ys, return_counts=True)))}")
    return splits


if __name__ == "__main__":
    import time
    t0 = time.time()

    df = load_factors()
    factor_cols = get_selected_factors(0.5)
    print(f"Factors: {len(factor_cols)}\n")

    daily_means = compute_daily_factors(df, factor_cols, top_n=100)
    regime_series, _, scaler, kmeans = detect_regimes_train_only(daily_means, train_end="2018-01-01", n_clusters=3)

    print("\n=== Building classification dataset ===\n")
    LOOKBACK = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    X, y, dates = build_classification_dataset(df, factor_cols, regime_series, lookback=LOOKBACK, forward=1)

    print("\n=== Time series split ===\n")
    splits = split_time_series(X, y, dates, train_end="2018-01-01", val_end="2022-01-01")

    # Save dataset
    np.savez(
        "data/processed/classification_data.npz",
        X=X, y=y, dates=dates,
        factor_cols=factor_cols,
    )
    print(f"\nSaved to data/processed/classification_data.npz ({time.time()-t0:.1f}s)")
