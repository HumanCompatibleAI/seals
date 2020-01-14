import gym
from gym.envs.mujoco import half_cheetah_v3

class HalfCheetahEnv(half_cheetah_v3.HalfCheetahEnv):
    def __init__(self, *args, time_horizon=200, **kwargs):
        self._time_horizon = time_horizon
        self._step_count = 0
        kwargs['exclude_current_positions_from_observation'] = False
        super().__init__(*args, **kwargs)

    @property
    def done(self):
        return self._step_count >= self._time_horizon

    def step(self, action):
        self._step_count += 1
        ob, re, _, info = super().step(action)
        return ob, re, self.done, info

    def reset(self):
        self._step_count = 0
        return super().reset()
