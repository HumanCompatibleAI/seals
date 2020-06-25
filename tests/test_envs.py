"""Smoke tests for all environments."""

import gym
import numpy as np
import pytest

import seals  # noqa: F401 required for env registration
from seals import base_envs
from seals.testing import envs

ENV_NAMES = [
    env_spec.id
    for env_spec in gym.envs.registration.registry.all()
    if env_spec.id.startswith(f"{seals.GYM_ID_PREFIX}/")
]

DETERMINISTIC_ENVS = []


env = pytest.fixture(envs.make_env_fixture(skip_fn=pytest.skip))


@pytest.mark.parametrize("env_name", ENV_NAMES)
class TestEnvs:
    """Battery of simple tests for environments."""

    def test_seed(self, env: gym.Env, env_name: str):
        """Tests environment seeding."""
        envs.test_seed(env, env_name, DETERMINISTIC_ENVS)

    def test_premature_step(self, env: gym.Env):
        """Tests if step() before reset() raises error."""
        envs.test_premature_step(env, skip_fn=pytest.skip, raises_fn=pytest.raises)

    def test_rollout_schema(self, env: gym.Env):
        """Tests if environments have correct types on `step()` and `reset()`."""
        envs.test_rollout_schema(env)


def test_base_envs():
    """Test parts of base_envs not covered elsewhere."""

    class NewEnv(base_envs.TabularModelEnv):
        def __init__(self):
            nS = 3
            nA = 2
            transition_matrix = np.random.rand(nS, nA, nS)
            transition_matrix /= transition_matrix.sum(axis=2)[:, :, None]
            reward_matrix = np.random.rand(nS)
            super().__init__(
                transition_matrix=transition_matrix, reward_matrix=reward_matrix,
            )

    env = NewEnv()

    assert np.all(np.eye(3) == env.feature_matrix)

    envs.test_premature_step(env, skip_fn=pytest.skip, raises_fn=pytest.raises)

    env.reset()
    assert env.n_actions_taken == 0
    env.step(env.action_space.sample())
    assert env.n_actions_taken == 1
    env.step(env.action_space.sample())
    assert env.n_actions_taken == 2
