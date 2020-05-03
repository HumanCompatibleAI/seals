"""Miscellaneous utilities."""

from typing import Optional

import gym
import numpy as np


class AutoResetWrapper(gym.Wrapper):
    """Hides done=True and auto-resets at the end of each episode."""

    def step(self, action):
        """When done=True, returns done=False instead and automatically resets.

        When an automatic reset happens, the observation from reset is returned,
        and the overridden observation is stored in
        `info["terminal_observation"]`.
        """
        obs, rew, done, info = self.env.step(action)
        if done:
            info["terminal_observation"] = obs
            obs = self.env.reset()
        return obs, rew, False, info


class AbsorbAfterDoneWrapper(gym.Wrapper):
    """Transition into absorbing state instead of episode termination.

    When the environment being wrapped returns `done=True`, we return an absorbing
    observation. This wrapper always returns `done=False`.

    A convenient way to add absorbing states to environments like MountainCar.
    """

    def __init__(
        self,
        env: gym.Env,
        absorb_reward: float = 0.0,
        absorb_obs: Optional[np.ndarray] = None,
    ):
        """Initialize AbsorbAfterDoneWrapper.

        Args:
          env: The wrapped Env.
          absorb_reward: The reward returned at the absorb state.
          absorb_obs: The observation returned at the absorb state. If None, then
            repeat the final observation before absorb.
        """
        super().__init__(env)
        self.absorb_reward = absorb_reward
        self.absorb_obs_default = absorb_obs
        self.absorb_obs_this_episode = None
        self.at_absorb_state = None

    def reset(self, *args, **kwargs):
        """Reset the environment."""
        self.at_absorb_state = False
        self.absorb_obs_this_episode = None
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        """Advance the environment by one step.

        This wrapped `step()` always returns done=False.

        After the first done is returned by the underlying Env, we enter an artificial
        absorb state.

        In this artificial absorb state, we stop calling
        `self.env.step(action)` (i.e. the `action` argument is entirely ignored) and
        we return fixed values for obs, rew, done, and info. The values of `obs` and
        `rew` depend on initialization arguments. `info` is always an empty dictionary.
        """
        if not self.at_absorb_state:
            inner_obs, inner_rew, done, inner_info = self.env.step(action)
            if done:
                # Initialize the artificial absorb state, which we will repeatedly use
                # starting on the next call to `step()`.
                self.at_absorb_state = True

                if self.absorb_obs_default is None:
                    self.absorb_obs_this_episode = inner_obs
                else:
                    self.absorb_obs_this_episode = self.absorb_obs_default
            obs, rew, info = inner_obs, inner_rew, inner_info
        else:
            assert self.absorb_obs_this_episode is not None
            assert self.absorb_reward is not None
            obs = self.absorb_obs_this_episode
            rew = self.absorb_reward
            info = {}

        return obs, rew, False, info


def make_env_no_wrappers(env_name: str, **kwargs) -> gym.Env:
    """Gym sometimes wraps envs in TimeLimit before returning from gym.make().

    This helper method builds directly from spec to avoid this wrapper.
    """
    return gym.envs.registry.env_specs[env_name].make(**kwargs)


def get_gym_max_episode_steps(env_name: str) -> Optional[int]:
    """Get the `max_episode_steps` attribute associated with a gym Spec."""
    return gym.envs.registry.env_specs[env_name].max_episode_steps
