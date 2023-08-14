"""Test RL on MuJoCo adapted environments."""

from typing import Tuple

import gymnasium as gym
import pytest
import stable_baselines3
from stable_baselines3.common import evaluation

import seals  # noqa: F401 Import required for env registration


def _eval_env(
    env_name: str,
    total_timesteps: int,
) -> Tuple[float, float]:  # pragma: no cover
    """Train PPO2 for `total_timesteps` on `env_name` and evaluate returns."""
    env = gym.make(env_name)
    model = stable_baselines3.PPO("MlpPolicy", env)
    model.learn(total_timesteps=total_timesteps)
    res = evaluation.evaluate_policy(model, env)
    assert isinstance(res[0], float)
    return res


# SOMEDAY(adam): tests are flaky and consistently fail in some environments
# Unclear if they even should pass in some cases.
# See discussion in GH#6 and GH#40.
@pytest.mark.expensive
@pytest.mark.parametrize(
    "env_base",
    ["HalfCheetah", "Ant", "Hopper", "Humanoid", "Swimmer", "Walker2d"],
)
def test_fixed_env_model_as_good_as_gym_env_model(env_base: str):  # pragma: no cover
    """Compare original and modified MuJoCo v3 envs."""
    train_timesteps = 200000

    gym_reward, _ = _eval_env(f"{env_base}-v4", total_timesteps=train_timesteps)
    fixed_reward, _ = _eval_env(
        f"seals/{env_base}-v1",
        total_timesteps=train_timesteps,
    )

    epsilon = 0.1
    sign = 1 if gym_reward > 0 else -1
    assert (1 - sign * epsilon) * gym_reward <= fixed_reward
