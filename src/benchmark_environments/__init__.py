"""Benchmark environments for reward modeling and imitation."""

import gym

def get_gym_v3_max_episode_steps(env_base):
    return gym.envs.registry.env_specs[f'{env_base}-v3'].max_episode_steps

for env_base in ['HalfCheetah', 'Ant', 'Hopper', 'Humanoid', 'Swimmer', 'Walker2d']:
    gym.register(
        id=f'benchmark_environments/{env_base}-v0',
        entry_point=f'benchmark_environments.mujoco:{env_base}Env',
        max_episode_steps=get_gym_v3_max_episode_steps(env_base)
    )

__version__ = "0.01"
