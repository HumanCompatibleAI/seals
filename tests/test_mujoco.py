"""Test MuJoCo adaptations."""

from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import MlpPolicy

import gym
from gym.wrappers.time_limit import TimeLimit
import benchmark_environments.mujoco
import pytest


@pytest.mark.expensive
@pytest.mark.parametrize("env_base",
                         ['HalfCheetah', 'Ant', 'Hopper', 'Humanoid',
                          'Swimmer', 'Walker2d'])
def test_fixed_env_model_as_good_as_gym_env_model(env_base):
    """Compare original and modified MuJoCo v3 envs."""
    train_timesteps = 200000

    gym_reward, _ = _eval_env(f'{env_base}-v3',
                              total_timesteps=train_timesteps)
    fixed_reward, _ = _eval_env(f'benchmark_environments/{env_base}-v0',
                                total_timesteps=train_timesteps)

    epsilon = 0.1
    sign = 1 if gym_reward > 0 else -1
    assert (1 - sign * epsilon) * gym_reward <= fixed_reward


def _eval_env(env_name, total_timesteps):
    env = gym.make(env_name)
    model = PPO2(MlpPolicy, env)
    model.learn(total_timesteps=total_timesteps)
    return evaluate_policy(model, env)
