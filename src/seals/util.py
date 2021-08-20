"""Miscellaneous utilities."""

from typing import Optional, Tuple

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


class ObsCastWrapper(gym.Wrapper):
    """Cast observations to specified dtype.

    Some external environments return observations of a different type than the
    declared observation space. Where possible, this should be fixed upstream,
    but casting can be a viable workaround -- especially when the returned
    observations are higher resolution than the observation space.
    """

    def __init__(self, env: gym.Env, dtype: np.dtype):
        """Builds ObsCastWrapper.

        Args:
            env: the environment to wrap.
            dtype: the dtype to cast observations to.
        """
        super().__init__(env)
        self.dtype = dtype

    def reset(self):
        """Returns reset observation, cast to self.dtype."""
        return super().reset().astype(self.dtype)

    def step(self, action):
        """Returns (obs, rew, done, info) with obs cast to self.dtype."""
        obs, rew, done, info = super().step(action)
        return obs.astype(self.dtype), rew, done, info


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


def sample_distribution(
    p: np.ndarray,
    random: np.random.RandomState,
) -> int:
    """Samples an integer with probabilities given by p."""
    return random.choice(np.arange(len(p)), p=p)


def one_hot_encoding(pos: int, size: int) -> np.ndarray:
    """Returns a 1-D hot encoding of a given position and size."""
    return np.eye(size)[pos]


def grid_transition_fn(
    state: np.ndarray,
    action: int,
    x_bounds: Tuple[float, float] = (-np.inf, np.inf),
    y_bounds: Tuple[float, float] = (-np.inf, np.inf),
):
    """Returns transition of a deterministic gridworld.

    Agent is bounded in the region limited by x_bounds and y_bounds,
    ends inclusive.

    (0, 0) is interpreted to be top-left corner.

    Actions:
    0: Right
    1: Down
    2: Left
    3: Up
    4: Stay put
    """
    dirs = [
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
        (0, 0),
    ]

    x, y = state
    dx, dy = dirs[action]

    next_x = np.clip(x + dx, *x_bounds)
    next_y = np.clip(y + dy, *y_bounds)
    next_state = np.array([next_x, next_y], dtype=state.dtype)

    return next_state
