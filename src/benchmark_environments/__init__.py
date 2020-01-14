from gym.envs.registration import register

__version__ = "0.01"

register(
    id='FixedHalfCheetah-v0',
    entry_point='benchmark_environments.mujoco:HalfCheetahEnv'
)
