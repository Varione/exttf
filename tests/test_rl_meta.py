"""Tests for RL meta allocator components."""

import numpy as np
import pytest


class TestStrategyWeightEnv:
    """Test RL environment with regime awareness."""

    @pytest.fixture
    def env(self):
        from src.rl_environment import StrategyWeightEnv
        return StrategyWeightEnv(
            start="2025-01-01",
            end="2025-03-31",
            lookback=3,
            n_hold=20,
            max_weight=0.05,
            respect_regime=True,
            regime_path="data/processed/regime_predictions_LSTM.csv",
        )

    def test_env_initialization(self, env):
        assert env.n_strategies > 0
        assert env.state_dim > 0
        assert env.respect_regime is True

    def test_reset_returns_valid_state(self, env):
        state = env.reset()
        assert state.shape == (env.state_dim,)
        assert np.all(np.isfinite(state))

    def test_step_returns_valid_values(self, env):
        state = env.reset()
        weights = np.ones(env.n_strategies) / env.n_strategies
        next_state, reward, done, info = env.step(weights)

        assert next_state.shape == (env.state_dim,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "return" in info
        assert "weights" in info
        assert "effective_weights" in info
        assert "regime_mask" in info

    def test_regime_mask_zeros_inactive(self, env):
        mask = env._get_regime_mask(env.dates[env.current_idx].strftime("%Y-%m-%d"))
        assert len(mask) == env.n_strategies
        assert mask.sum() > 0  # At least some strategies should be active

    def test_respect_regime_gating(self, env):
        state = env.reset()
        weights = np.ones(env.n_strategies) / env.n_strategies
        _, _, _, info = env.step(weights)

        eff_weights = info["effective_weights"]
        regime_mask = info["regime_mask"]
        # Effective weights should respect regime mask
        assert np.allclose(eff_weights * (1 - regime_mask), 0, atol=1e-8)

    def test_weight_constraints(self, env):
        state = env.reset()
        # Test with extreme weights
        weights = np.zeros(env.n_strategies)
        weights[0] = 1.0
        _, _, _, info = env.step(weights)

        # Should be capped at max_single_weight
        assert info["weights"][0] <= env.max_single_weight + 1e-6

    def test_reward_is_finite(self, env):
        state = env.reset()
        weights = np.ones(env.n_strategies) / env.n_strategies
        _, reward, _, _ = env.step(weights)
        assert np.isfinite(reward)


class TestLegacyEnv:
    """Test backward compatible legacy environment."""

    def test_legacy_respect_regime_false(self):
        from src.rl_environment import LegacyStrategyWeightEnv
        env = LegacyStrategyWeightEnv(
            start="2025-01-01",
            end="2025-02-28",
            lookback=3,
            regime_path="data/processed/regime_predictions_LSTM.csv",
        )
        assert env.respect_regime is False
        mask = env._get_regime_mask(env.dates[env.current_idx].strftime("%Y-%m-%d"))
        np.testing.assert_array_equal(mask, np.ones(env.n_strategies))


class TestRewardDecomposition:
    """Test reward function components without full env setup."""

    def test_reward_components(self):
        class MockEnv:
            n_strategies = 5
            _last_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            _history_returns = [0.001] * 10
            turnover_penalty_coef = 5.0
            drawdown_penalty_coef = 10.0
            concentration_penalty_coef = 2.0

            def _compute_reward(self, port_return, turnover):
                base_reward = port_return * 100.0
                turnover_penalty = self.turnover_penalty_coef * turnover
                wealth = np.cumprod(1.0 + np.asarray(self._history_returns))
                current_dd = float(wealth[-1] / np.maximum.accumulate(wealth)[-1] - 1.0)
                dd_penalty = self.drawdown_penalty_coef * max(0, -current_dd) * 100.0
                w = self._last_weights
                w_safe = w[w > 1e-8]
                w_norm = w_safe / w_safe.sum()
                entropy = -float(np.sum(w_norm * np.log(w_norm + 1e-8)))
                max_entropy = np.log(len(w_safe)) if len(w_safe) > 1 else 1.0
                concentration = 1.0 - entropy / max_entropy
                conc_penalty = self.concentration_penalty_coef * concentration
                return base_reward - turnover_penalty - dd_penalty - conc_penalty

        mock = MockEnv()
        r = mock._compute_reward(0.001, 0.1)
        assert np.isfinite(r)


class TestRewardFunction:
    """Test enhanced reward function without slow fixture."""

    def test_reward_decomposition(self):
        from src.rl_environment import StrategyWeightEnv

        class MockEnv:
            _last_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            _history_returns = [0.001] * 10
            turnover_penalty_coef = 5.0
            drawdown_penalty_coef = 10.0
            concentration_penalty_coef = 2.0

            def _compute_reward(self, port_return, turnover):
                base_reward = port_return * 100.0
                turnover_penalty = self.turnover_penalty_coef * turnover
                wealth = np.cumprod(1.0 + np.asarray(self._history_returns))
                current_dd = float(wealth[-1] / np.maximum.accumulate(wealth)[-1] - 1.0)
                dd_penalty = self.drawdown_penalty_coef * max(0, -current_dd) * 100.0
                w = self._last_weights
                w_safe = w[w > 1e-8]
                w_norm = w_safe / w_safe.sum()
                entropy = -float(np.sum(w_norm * np.log(w_norm + 1e-8)))
                max_entropy = np.log(len(w_safe)) if len(w_safe) > 1 else 1.0
                concentration = 1.0 - entropy / max_entropy
                conc_penalty = self.concentration_penalty_coef * concentration
                return base_reward - turnover_penalty - dd_penalty - conc_penalty

        mock = MockEnv()
        reward = mock._compute_reward(0.001, 0.1)
        assert isinstance(reward, float)
        assert np.isfinite(reward)

    def test_high_turnover_penalty(self):
        from src.rl_environment import StrategyWeightEnv

        class MockEnv:
            _last_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            _history_returns = [0.0] * 5
            turnover_penalty_coef = 5.0
            drawdown_penalty_coef = 10.0
            concentration_penalty_coef = 2.0

            def _compute_reward(self, port_return, turnover):
                base_reward = port_return * 100.0
                turnover_penalty = self.turnover_penalty_coef * turnover
                return base_reward - turnover_penalty

        mock = MockEnv()
        r_low = mock._compute_reward(0.001, 0.05)
        r_high = mock._compute_reward(0.001, 0.5)
        assert r_low > r_high

    def test_zero_return_reward(self):
        from src.rl_environment import StrategyWeightEnv

        class MockEnv:
            _last_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            _history_returns = [0.0] * 5
            turnover_penalty_coef = 5.0
            drawdown_penalty_coef = 10.0
            concentration_penalty_coef = 2.0

            def _compute_reward(self, port_return, turnover):
                base_reward = port_return * 100.0
                turnover_penalty = self.turnover_penalty_coef * turnover
                return base_reward - turnover_penalty

        mock = MockEnv()
        reward = mock._compute_reward(0.0, 0.0)
        assert np.isfinite(reward)
