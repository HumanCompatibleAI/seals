"""Test RL on MuJoCo adapted environments."""

from typing import Tuple

import gym
import pytest
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import MlpPolicy

import seals  # noqa: F401 Import required for env registration


def _eval_env(
    env_name: str, total_timesteps: int,
) -> Tuple[float, int]:  # pragma: no cover
    """Train PPO2 for `total_timesteps` on `env_name` and evaluate returns."""
    env = gym.make(env_name)
    model = PPO2(MlpPolicy, env)
    model.learn(total_timesteps=total_timesteps)
    return evaluate_policy(model, env)


@pytest.mark.expensive
@pytest.mark.parametrize(
    "env_base", ["HalfCheetah", "Ant", "Hopper", "Humanoid", "Swimmer", "Walker2d"],
)
def test_fixed_env_model_as_good_as_gym_env_model(env_base: str):  # pragma: no cover
    """Compare original and modified MuJoCo v3 envs."""
    train_timesteps = 200000

    gym_reward, _ = _eval_env(f"{env_base}-v3", total_timesteps=train_timesteps)
    fixed_reward, _ = _eval_env(
        f"seals/{env_base}-v0", total_timesteps=train_timesteps,
    )

    epsilon = 0.1
    sign = 1 if gym_reward > 0 else -1
    assert (1 - sign * epsilon) * gym_reward <= fixed_reward
