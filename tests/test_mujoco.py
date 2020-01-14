"""Test MuJoCo adaptations."""

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import gym
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

    max_timesteps_per_episode = fixed_env.spec.max_episode_steps

    gym_reward = _eval_model(gym_model, gym_env, num_episodes=num_eval_episodes,
            max_timesteps_per_episode=max_timesteps_per_episode)
    fixed_reward = _eval_model(fixed_model, fixed_env, num_episodes=num_eval_episodes,
            max_timesteps_per_episode=max_timesteps_per_episode)

    epsilon = 0.1
    sign = 1 if gym_reward > 0 else -1
    assert (1 - sign * epsilon) * gym_reward <= fixed_reward

def _eval_model(model, env, *, num_episodes, max_timesteps_per_episode):
    total_re = 0

    for episode in range(num_episodes):
        ob = env.reset()
        step = 0
        done = False
        while not done and step < max_timesteps_per_episode:
            action, _ = model.predict(ob)
            new_ob, re, done, _ = env.step(action)

            total_re += re

            ob = new_ob
            step += 1

    return total_re / num_episodes
