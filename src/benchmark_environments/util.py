"""Miscellaneous utilities."""

import functools
from typing import Callable, Optional

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


class FixedRewardAfterDoneWrapper(gym.Wrapper):
    """After the inner environment returns done=True, modify the returned reward.

    Always returns done=False so that the episode can continue until reset().
    """

    def __init__(self, env: gym.Env, end_reward: float = 0.0):
        """Initialize wrapper.

        Args:
            env: The wrapped Env.
            end_reward: The reward returned after the inner environment returns
                done=True from `step()`.
        """
        super().__init__(env)
        self.saw_done = None
        self.end_reward = end_reward

    def reset(self, *args, **kwargs):
        """Reset the environment."""
        self.saw_done = False
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        """Advance the environment by one step.

        This wrapped `step()` always returns done=False.
        """
        obs, rew, done, info = self.env.step(action)
        if self.saw_done:
            rew = self.end_reward
        self.saw_done = self.saw_done or done
        return obs, rew, False, info


def make_env_no_wrappers(env_name: str, **kwargs) -> gym.Env:
    """Gym sometimes wraps envs in TimeLimit before returning from gym.make().

    This helper method builds directly from step to avoid this wrapper.
    """
    return gym.envs.registry.env_specs[env_name].make(**kwargs)


def get_gym_max_episode_steps(env_name: str) -> Optional[int]:
    """Get the `max_episode_steps` attribute associated with a gym Spec."""
    return gym.envs.registry.env_specs[env_name].max_episode_steps


def _gym_register_as_decorator(
    func: Callable[..., gym.Env],
    id_base: str,
    *,
    module_name: str,
    id_prefix: str = "",
    **register_kwargs,
) -> Callable[..., gym.Env]:
    """Decorator variant of `gym.register` with autogenerated `id` and `entry_point`.

    `**register_kwargs` is passed through directly to `gym.register`. This function
    takes optional arguments `reward_threshold`, `nondeterministic`,
    `max_episode_steps`, and `kwargs`, documented below.

    Args:
        func: Callable that returns an env. This Callable should be defined in the
            module with name `module_name`, because `func.__name__` is used to
            automatically construct the `entry_point` argument to `gym.register`.
        id_base: The Gym-registered `id` is `id_prefix + id_base`.
        module_name: The name of the module where the decorator will be applied.
            Usually the user will want to use the global variable `__name__`.
        id_prefix: The Gym-registered `id` is `id_prefix + id_base`.

        reward_threshold (Optional[int]): The reward threshold before the task is
            considered solved.
        nondeterministic (bool): Whether this environment is non-deterministic even
            after seeding.
        max_episode_steps (Optional[int]): The maximum number of steps that an episode
            can consist of.
        kwargs (dict): Kwargs to pass into `func`. (Note that this is not a `**`
            argument. There is in fact a dict parameter called `kwargs` in
            `gym.envs.registration.EnvSpec`.)
    """
    entry_point = f"{module_name}:{func.__name__}"
    gym.register(id=id_prefix + id_base, entry_point=entry_point, **register_kwargs)
    return func


def curried_gym_register_as_decorator(module_name: str) -> Callable[[str], Callable]:
    """Two-staged curry that builds a easy-to-use `gym.register` decorator.

    See `benchmark_environments.classic` for example usage.

    This function and its return value build a `functools.partial` instance of
    `_gym_register_as_decorator`.

    In the first curry stage the `module_name` argument of `_gym_register_as_decorator`
    is filled. In the second stage, the `id_base` arguments, and optionally other
    `**kwargs` are filled.

    Returns:
        The second curry function which accepts the `id_base` argument, and which
        finally returns the decorator.
    """

    def build_decorator(id_base: str, *args, **kwargs) -> Callable:
        return functools.partial(
            _gym_register_as_decorator,
            *args,
            id_base=id_base,
            module_name=module_name,
            id_prefix="benchmark_environments/",
            **kwargs,
        )

    return build_decorator
