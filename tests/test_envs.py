"""Smoke tests for all environments."""

from typing import List

import gym
import pytest

import seals  # noqa: F401 required for env registration
from seals.testing import envs

ENV_NAMES: List[str] = [
    env_spec.id
    for env_spec in gym.envs.registration.registry.all()
    if env_spec.id.startswith(f"{seals.GYM_ID_PREFIX}/")
]

DETERMINISTIC_ENVS: List[str] = [
    "seals/EarlyTermPos-v0",
    "seals/EarlyTermNeg-v0",
    "seals/Branching-v0",
    "seals/InitShiftTrain-v0",
    "seals/InitShiftTest-v0",
]


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
