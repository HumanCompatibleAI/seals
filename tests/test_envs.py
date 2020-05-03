"""Smoke tests for all environments."""

import gym
import pytest

import benchmark_environments  # noqa: F401
from benchmark_environments.testing import envs

ENV_NAMES = [
    env_spec.id
    for env_spec in gym.envs.registration.registry.all()
    if env_spec.id.startswith(f"{benchmark_environments.GYM_ID_PREFIX}/")
]

DETERMINISTIC_ENVS = []


env = pytest.fixture(envs.make_env_fixture(skip_fn=pytest.skip))


@pytest.mark.parametrize("env_name", ENV_NAMES)
class TestEnvs:
    """Battery of simple tests for environments."""

    def test_seed(self, env, env_name):
        """Tests environment seeding."""
        envs.test_seed(env, env_name, DETERMINISTIC_ENVS)

    def test_premature_step(self, env):
        """Tests if step() before reset() raises error."""
        envs.test_premature_step(env, skip_fn=pytest.skip, raises_fn=pytest.raises)

    def test_rollout_schema(self, env):
        """Tests if environments have correct types on `step()` and `reset()`."""
        envs.test_rollout_schema(env)
