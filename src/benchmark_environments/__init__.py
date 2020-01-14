"""Benchmark environments for reward modeling and imitation."""

from gym.envs.registration import register

__version__ = "0.01"

for env_base in ['HalfCheetah', 'Ant', 'Hopper', 'Humanoid', 'Swimmer', 'Walker2d']:
    register(
        id=f'benchmark_environments/{env_base}-v0',
        entry_point=f'benchmark_environments.mujoco:{env_base}Env',
        max_episode_steps=200
    )
