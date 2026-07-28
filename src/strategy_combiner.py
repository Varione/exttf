"""Strategy combination methods for portfolio allocation."""

from __future__ import annotations

import numpy as np
import pandas as pd


class EqualRiskContribution:
    """Equal Risk Contribution (ERC) / Hierarchical Risk Parity.

    Each strategy contributes equally to the portfolio volatility.
    Uses a simplified ERC approach based on marginal risk contributions.
    """

    def get_weights(self, strategy_returns: pd.DataFrame, *, n_iter: int = 100) -> dict[str, float]:
        """Compute ERC weights from strategy return series.

        Args:
            strategy_returns: DataFrame with strategy names as columns,
                             daily returns as values.
            n_iter: Number of optimization iterations.

        Returns:
            Dictionary mapping strategy name to weight (sums to 1.0).
        """
        strategies = list(strategy_returns.columns)
        n = len(strategies)
        if n == 0:
            return {}
        if n == 1:
            return {strategies[0]: 1.0}

        cov = strategy_returns.cov().values
        vols = np.sqrt(np.diag(cov))

        # Skip zero-vol strategies
        valid = vols > 1e-10
        if not valid.all():
            valid_strats = [s for s, v in zip(strategies, valid)]
            valid_cov = cov[np.ix_(valid, valid)]
            valid_vols = np.sqrt(np.diag(valid_cov))
        else:
            valid_strats = strategies
            valid_cov = cov
            valid_vols = vols
            n = len(valid_strats)

        if n == 1:
            return {valid_strats[0]: 1.0}

        # Simplified ERC: inverse volatility weighting as initial guess,
        # then iterate to equalize risk contributions
        w = 1.0 / valid_vols
        w = w / w.sum()

        for _ in range(n_iter):
            # Portfolio variance
            port_var = float(w @ valid_cov @ w)
            if port_var < 1e-20:
                break

            # Marginal risk contribution
            mrc = valid_cov @ w  # d(port_var)/dw
            # Risk contribution
            rc = w * mrc
            # Target: equal risk contribution = port_var / n
            target_rc = port_var / n

            # Update weights proportional to target_rc / mrc
            w_new = np.zeros(n)
            for i in range(n):
                if mrc[i] > 1e-20:
                    w_new[i] = target_rc / mrc[i]
                else:
                    w_new[i] = 1.0 / n

            w_new = np.maximum(w_new, 1e-8)
            w = w_new / w_new.sum()

        return {s: float(w[i]) for i, s in enumerate(valid_strats)}


class VolatilityScaling:
    """Volatility Scaling: allocate weights inversely proportional to volatility.

    Scales each strategy's weight by 1/volatility, then optionally
    scales the total portfolio to a target volatility.
    """

    def get_weights(
        self,
        strategy_returns: pd.DataFrame,
        *,
        target_vol: float = 0.10,
    ) -> dict[str, float]:
        """Compute volatility-scaled weights.

        Args:
            strategy_returns: DataFrame with strategy daily returns.
            target_vol: Target annualized portfolio volatility (unused in
                       basic inverse-vol weighting; used for scaling).

        Returns:
            Dictionary mapping strategy name to weight (sums to 1.0).
        """
        strategies = list(strategy_returns.columns)
        if not strategies:
            return {}

        vols = strategy_returns.std().values
        valid = vols > 1e-10

        if not valid.any():
            # All zero vol -> equal weight
            return {s: 1.0 / len(strategies) for s in strategies}

        inv_vol = np.zeros(len(strategies))
        inv_vol[valid] = 1.0 / vols[valid]
        inv_vol[~valid] = 0.0

        w = inv_vol / inv_vol.sum()
        return {s: float(wi) for s, wi in zip(strategies, w)}


class RegimeConditionedWeights:
    """Regime-conditioned strategy weights.

    Allocates weights based on regime probabilities and strategy
    historical performance in each regime.
    """

    def get_weights(
        self,
        strategy_returns: pd.DataFrame,
        regime_probs: pd.Series | pd.DataFrame,
        *,
        lookback: int = 252,
    ) -> dict[str, float]:
        """Compute regime-conditioned weights.

        Args:
            strategy_returns: DataFrame with strategy daily returns.
            regime_probs: Series (regime labels) or DataFrame with columns
                         prob_r0, prob_r1, prob_r2 (regime probabilities).
            lookback: Number of days to use for performance estimation.

        Returns:
            Dictionary mapping strategy name to weight.
        """
        strategies = list(strategy_returns.columns)
        if not strategies:
            return {}

        n_strat = len(strategies)
        recent = strategy_returns.tail(lookback)

        # If regime_probs is a DataFrame with prob_r0, prob_r1, prob_r2
        if isinstance(regime_probs, pd.DataFrame):
            prob_cols = [c for c in ["prob_r0", "prob_r1", "prob_r2"] if c in regime_probs.columns]
            if prob_cols:
                probs = regime_probs[prob_cols].tail(lookback)
                # Align indices
                common_idx = recent.index.intersection(probs.index)
                if len(common_idx) < 10:
                    return {s: 1.0 / n_strat for s in strategies}

                recent = recent.loc[common_idx]
                probs = probs.loc[common_idx]

                # Compute average regime probabilities
                avg_probs = probs.mean().values

                # Compute strategy performance conditioned on each regime
                # Use rolling regime assignment (argmax of probs)
                regime_assignments = probs.values.argmax(axis=1)
                n_regimes = len(prob_cols)

                # Weight by: sum over regimes of P(regime) * strategy_return_in_regime
                perf_weighted = np.zeros(n_strat)
                for r in range(n_regimes):
                    mask = regime_assignments == r
                    if mask.sum() > 0:
                        reg_returns = recent.values[mask]
                        reg_mean = reg_returns.mean(axis=0)
                        perf_weighted += avg_probs[r] * reg_mean

                # Softmax-like weighting
                perf_safe = perf_weighted - perf_weighted.min() + 1e-8
                w = perf_safe / perf_safe.sum()
                return {s: float(wi) for s, wi in zip(strategies, w)}

        # If regime_probs is a Series (discrete regime labels)
        if isinstance(regime_probs, pd.Series):
            recent_regimes = regime_probs.tail(lookback)
            common_idx = recent.index.intersection(recent_regimes.index)
            if len(common_idx) < 10:
                return {s: 1.0 / n_strat for s in strategies}

            recent = recent.loc[common_idx]
            recent_regimes = recent_regimes.loc[common_idx]

            # Current regime
            current_regime = recent_regimes.iloc[-1]

            # Performance in current regime
            mask = recent_regimes == current_regime
            if mask.sum() > 0:
                reg_returns = recent.values[mask.values]
                perf = reg_returns.mean(axis=0)
                perf_safe = perf - perf.min() + 1e-8
                w = perf_safe / perf_safe.sum()
                return {s: float(wi) for s, wi in zip(strategies, w)}

        # Fallback: equal weight
        return {s: 1.0 / n_strat for s in strategies}


def combine_strategy_returns(
    daily_returns: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.Series:
    """Combine multiple strategy daily returns using given weights.

    Args:
        daily_returns: Dict mapping strategy name to daily return Series.
        weights: Dict mapping strategy name to weight.

    Returns:
        Combined daily return Series.
    """
    common_idx = None
    series_list = []
    for name, ret in daily_returns.items():
        w = weights.get(name, 0.0)
        if w <= 0:
            continue
        s = ret * w
        if common_idx is None:
            common_idx = s.index
        else:
            common_idx = common_idx.intersection(s.index)
        series_list.append(s)

    if not series_list:
        return pd.Series(dtype=float)

    combined = sum(series_list)  # type: ignore[assignment]
    if common_idx is not None:
        combined = combined.loc[common_idx]
    return combined


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500)
    test_returns = pd.DataFrame({
        "S01": np.random.randn(500) * 0.01 + 0.0003,
        "S04": np.random.randn(500) * 0.008 + 0.0002,
        "S23": np.random.randn(500) * 0.012 + 0.0001,
    }, index=dates)

    erc = EqualRiskContribution()
    print("ERC weights:", erc.get_weights(test_returns))

    vs = VolatilityScaling()
    print("Vol scaling weights:", vs.get_weights(test_returns))

    rcw = RegimeConditionedWeights()
    regime_df = pd.DataFrame({
        "prob_r0": np.random.rand(500) * 0.4 + 0.1,
        "prob_r1": np.random.rand(500) * 0.3 + 0.1,
        "prob_r2": np.random.rand(500) * 0.2 + 0.1,
    }, index=dates)
    regime_df = regime_df / regime_df.sum(axis=1)
    print("Regime-conditioned weights:", rcw.get_weights(test_returns, regime_df))
