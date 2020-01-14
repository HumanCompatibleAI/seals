"""Test MuJoCo adaptations."""

from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import gym
from gym.wrappers.time_limit import TimeLimit
import benchmark_environments

def test_fixed_env_model_as_good_as_gym_env_model():
    train_timesteps = 200000
    num_eval_episodes = 50
    verbose = 0

    gym_env = gym.make('HalfCheetah-v3')
    fixed_env = gym.make('benchmark_environments/HalfCheetah-v0')

    gym_model = PPO2(MlpPolicy, gym_env, verbose=verbose)
    gym_model.learn(total_timesteps=train_timesteps)

    fixed_model = PPO2(MlpPolicy, fixed_env, verbose=verbose)
    fixed_model.learn(total_timesteps=train_timesteps)

    gym_env_eval = TimeLimit(gym_env, fixed_env.spec.max_episode_steps)
    gym_reward, _ = evaluate_policy(gym_model, gym_env_eval)
    fixed_reward, _ = evaluate_policy(fixed_model, fixed_env)

    epsilon = 0.1
    sign = 1 if gym_reward > 0 else -1
    assert (1 - sign * epsilon) * gym_reward <= fixed_reward
