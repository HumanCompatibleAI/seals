"""Miscellaneous utilities."""

from dataclasses import dataclass
from typing import Any, Generic, List, Optional, Sequence, SupportsFloat, Tuple, Union, Dict

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType


class AutoResetWrapper(
    gym.Wrapper, Generic[WrapperObsType, WrapperActType, ObsType, ActType]
):
    """Hides done=True and auto-resets at the end of each episode.

    Depending on the flag 'discard_terminal_observation', either discards the terminal
    observation or pads with an additional 'reset transition'. The former is the default
    behavior.
    In the latter case, the action taken during the 'reset transition' will not have an
    effect, the reward will be constant (set by the wrapper argument `reset_reward`,
    which has default value 0.0), and info an empty dictionary.
    """

    def __init__(self, env, discard_terminal_observation=True, reset_reward=0.0):
        """Builds the wrapper.

        Args:
            env: The environment to wrap.
            discard_terminal_observation: Defaults to True. If True, the terminal
                observation is discarded and the environment is reset immediately. The
                returned observation will then be the start of the next episode. The
                overridden observation is stored in `info["terminal_observation"]`.
                If False, the terminal observation is returned and the environment is
                reset in the next step.
            reset_reward: The reward to return for the reset transition. Defaults to
                0.0.
        """
        super().__init__(env)
        self.discard_terminal_observation = discard_terminal_observation
        self.reset_reward = reset_reward
        self.previous_done = False  # Whether the previous step returned done=True.

    def step(
        self, action: WrapperActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """When done=True, returns done=False, then reset depending on flag.

        Depending on whether we are discarding the terminal observation,
        either resets the environment and discards,
        or returns the terminal observation, and then uses the next step to reset the
        environment, after which steps will be performed as normal.
        """
        if self.discard_terminal_observation:
            return self._step_discard(action)
        else:
            return self._step_pad(action)

    def _step_pad(
        self, action: WrapperActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """When done=True, return done=False instead and return the terminal obs.

        The agent will then usually be asked to perform an action based on
        the terminal observation. In the next step, this final action will be ignored
        to instead reset the environment and return the initial observation of the new
        episode.

        Some potential caveats:
        - The underlying environment will perform fewer steps than the wrapped
          environment.
        - The number of steps the agent performs and the number of steps recorded in the
          underlying environment will not match, which could cause issues if these are
          assumed to be the same.
        """
        if self.previous_done:
            self.previous_done = False
            reset_obs, reset_info_dict = self.env.reset()
            info = {"reset_info_dict": reset_info_dict}
            # This transition will only reset the environment, the action is ignored.
            return reset_obs, self.reset_reward, False, False, info

        obs, rew, terminated, truncated, info = self.env.step(action)
        if terminated:
            self.previous_done = True
        return obs, rew, False, truncated, info

    def _step_discard(
        self, action: WrapperActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """When done=True, returns done=False instead and automatically resets.

        When an automatic reset happens, the observation from reset is returned,
        and the overridden observation is stored in
        `info["terminal_observation"]`.
        """
        obs, rew, terminated, truncated, info = self.env.step(action)
        if terminated:
            info["terminal_observation"] = obs
            obs, reset_info_dict = self.env.reset()
            info["reset_info_dict"] = reset_info_dict
        return obs, rew, False, truncated, info


@dataclass
class BoxRegion:
    """A rectangular region dataclass used by MaskScoreWrapper."""

    x: Tuple
    y: Tuple


MaskedRegionSpecifier = List[BoxRegion]


class MaskScoreWrapper(
    gym.Wrapper[npt.NDArray, ActType, npt.NDArray, ActType],
    Generic[ActType],
):
    """Mask a list of box-shaped regions in the observation to hide reward info.

    Intended for environments whose observations are raw pixels (like Atari
    environments). Used to mask regions of the observation that include information
    that could be used to infer the reward, like the game score or enemy ship count.
    """

    def __init__(
        self,
        env: gym.Env,
        score_regions: MaskedRegionSpecifier,
        fill_value: Union[float, Sequence[float]] = 0,
    ):
        """Builds MaskScoreWrapper.

        Args:
            env: The environment to wrap.
            score_regions: A list of box-shaped regions to mask, each denoted by
                a dictionary `{"x": (x0, x1), "y": (y0, y1)}`, where `x0 < x1`
                and `y0 < y1`.
            fill_value: The fill_value for the masked region. By default is black.
                Can support RGB colors by being a sequence of values [r, g, b].

        Raises:
            ValueError: If a score region does not conform to the spec.
        """
        super().__init__(env)
        self.fill_value = np.array(fill_value, env.observation_space.dtype)

        if env.observation_space.shape is None:
            raise ValueError("Observation space must have a shape.")
        self.mask = np.ones(env.observation_space.shape, dtype=bool)
        for r in score_regions:
            if r.x[0] >= r.x[1] or r.y[0] >= r.y[1]:
                raise ValueError('Invalid region: "x" and "y" must be increasing.')
            self.mask[r.x[0] : r.x[1], r.y[0] : r.y[1]] = 0

    def _mask_obs(self, obs) -> npt.NDArray:
        return np.where(self.mask, obs, self.fill_value)

    def step(
        self, action: ActType
    ) -> Tuple[npt.NDArray, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Returns (obs, rew, terminated, truncated, info) with masked obs."""
        obs, rew, terminated, truncated, info = self.env.step(action)
        return self._mask_obs(obs), rew, terminated, truncated, info

    def reset(self, **kwargs):
        """Returns masked reset observation."""
        obs = self.env.reset(**kwargs)
        return self._mask_obs(obs)


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

    def reset(self, seed=None):
        """Returns reset observation, cast to self.dtype."""
        obs, info = super().reset(seed=seed)
        return obs.astype(self.dtype), info

    def step(self, action):
        """Returns (obs, rew, terminated, truncated, info) with obs cast to self.dtype."""
        obs, rew, terminated, truncated, info = super().step(action)
        return obs.astype(self.dtype), rew, terminated, truncated, info


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
            inner_obs, inner_rew, terminated, truncated, inner_info = self.env.step(
                action
            )
            if terminated:
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
            truncated = False

        return obs, rew, False, truncated, info


def make_env_no_wrappers(env_name: str, **kwargs) -> gym.Env:
    """Gym sometimes wraps envs in TimeLimit before returning from gymnasium.make().

    This helper method builds directly from spec to avoid this wrapper.
    """
    return gym.spec(env_name).make(**kwargs)


def get_gym_max_episode_steps(env_name: str) -> Optional[int]:
    """Get the `max_episode_steps` attribute associated with a gym Spec."""
    return gym.spec(env_name).max_episode_steps


def sample_distribution(
    p: np.ndarray,
    random: np.random.Generator,
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
