"""Benchmark environments for reward modeling and imitation."""

from gym.envs.registration import register

__version__ = "0.01"

register(
    id='benchmark_environments/HalfCheetah-v0',
    entry_point='benchmark_environments.mujoco:HalfCheetahEnv',
    max_episode_steps=200
)
