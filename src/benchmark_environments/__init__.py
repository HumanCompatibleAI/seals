"""Benchmark environments for reward modeling and imitation."""

import gym


def _get_gym_v3_max_episode_steps(env_base_):
    return gym.envs.registry.env_specs[f'{env_base_}-v3'].max_episode_steps

for env_base in ['HalfCheetah', 'Ant', 'Hopper', 'Humanoid', 'Swimmer', 'Walker2d']:
    gym.register(
        id=f'benchmark_environments/{env_base}-v0',
        entry_point=f'benchmark_environments.mujoco:{env_base}Env',
        max_episode_steps=_get_gym_v3_max_episode_steps(env_base),
    )

__version__ = "0.01"
