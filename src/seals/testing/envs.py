"""Helper methods for tests of custom Gym environments.

This is used in our test suite in `tests/test_envs.py`. It is also used in sister
projects such as `imitation`, and may be useful in other codebases.
"""

import re
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import gym
from gym.envs.mujoco import mujoco_env
import numpy as np

Step = Tuple[Any, Optional[float], bool, Mapping[str, Any]]
Rollout = Sequence[Step]
"""A sequence of 4-tuples (obs, rew, done, info) as returned by `get_rollout`."""


def make_env_fixture(
    skip_fn: Callable[[str], None],
) -> Callable[[str], Iterator[gym.Env]]:
    """Creates a fixture function, calling `skip_fn` when dependencies are missing.

    For example, in `pytest`, one would use::

        env = pytest.fixture(make_env_fixture(skip_fn=pytest.skip))

    Then any method with an `env` parameter will receive the created environment, with
    the `env_name` parameter automatically passed to the fixture.

    In `unittest`, one would use::

        def skip_fn(msg):
            raise unittest.SkipTest(msg)

        make_env = contextlib.contextmanager(make_env_fixture(skip_fn=skip_fn))

    And then call `with make_env(env_name) as env:` to create environments.

    Args:
        skip_fn: the function called when a dependency is missing to skip the test.

    Returns:
        A method to create Gym environments given their name.
    """

    def f(env_name: str) -> Iterator[gym.Env]:
        """Create environment `env_name`.

        Args:
            env_name: The name of the environment in the Gym registry.

        Yields:
            The created environment.

        Raises:
            gym.error.DependencyNotInstalled: if a dependency is missing
                other than MuJoCo (for MuJoCo, the test is instead skipped).
        """
        env = None
        try:
            env = gym.make(env_name)
            yield env
        except gym.error.DependencyNotInstalled as e:  # pragma: no cover
            if e.args[0].find("mujoco_py") != -1:
                skip_fn("Requires `mujoco_py`, which isn't installed.")
            else:
                raise
        finally:
            if env is not None:
                env.close()

    return f


def matches_list(env_name: str, patterns: Iterable[str]) -> bool:
    """Returns True if `env_name` matches any of the patterns in `patterns`."""
    return any(re.match(env_pattern, env_name) for env_pattern in patterns)


def get_rollout(env: gym.Env, actions: Iterable[Any]) -> Rollout:
    """Performs sequence of actions `actions` in `env`.

    Args:
      env: the environment to rollout in.
      actions: the actions to perform.

    Returns:
      A sequence of 4-tuples (obs, rew, done, info).
    """
    ret: List[Step] = [(env.reset(), None, False, {})]
    for act in actions:
        ret.append(env.step(act))
    return ret


def assert_equal_rollout(rollout_a: Rollout, rollout_b: Rollout) -> None:
    """Checks rollouts for equality.

    Raises:
        AssertionError if they are not equal.
    """
    for step_a, step_b in zip(rollout_a, rollout_b):
        ob_a, rew_a, done_a, info_a = step_a
        ob_b, rew_b, done_b, info_b = step_b
        np.testing.assert_equal(ob_a, ob_b)
        assert rew_a == rew_b
        assert done_a == done_b
        np.testing.assert_equal(info_a, info_b)


def has_same_observations(rollout_a: Rollout, rollout_b: Rollout) -> bool:
    """True if `rollout_a` and `rollout_b` have the same observations."""
    obs_list_a = [step[0] for step in rollout_a]
    obs_list_b = [step[0] for step in rollout_b]
    return all(np.all(obs_a == obs_b) for obs_a, obs_b in zip(obs_list_a, obs_list_b))


def test_seed(env: gym.Env, env_name: str, deterministic_envs: Iterable[str]) -> None:
    """Tests environment seeding.

    If non-deterministic, different seeds should produce different transitions.
    If deterministic, should be invariant to seed.

    Raises:
        AssertionError if test fails.
    """
    env.action_space.seed(0)
    actions = [env.action_space.sample() for _ in range(10)]

    # With the same seed, should always get the same result
    seeds = env.seed(42)
    assert isinstance(seeds, list)
    assert len(seeds) > 0
    rollout_a = get_rollout(env, actions)

    env.seed(42)
    rollout_b = get_rollout(env, actions)

    assert_equal_rollout(rollout_a, rollout_b)

    # For most non-deterministic environments, if we try enough seeds we should
    # eventually get a different result. For deterministic environments, all
    # seeds should produce the same starting state.
    def different_seeds_same_rollout(seed1, seed2):
        new_actions = [env.action_space.sample() for _ in range(10)]
        env.seed(seed1)
        new_rollout_1 = get_rollout(env, new_actions)
        env.seed(seed2)
        new_rollout_2 = get_rollout(env, new_actions)
        return has_same_observations(new_rollout_1, new_rollout_2)

    is_deterministic = matches_list(env_name, deterministic_envs)
    same_obs = all(different_seeds_same_rollout(seed, seed + 1) for seed in range(100))
    assert same_obs == is_deterministic


def _check_obs(obs: np.ndarray, obs_space: gym.Space) -> None:
    """Check obs is consistent with obs_space."""
    if obs_space.shape:
        assert obs.shape == obs_space.shape, f"obs.shape: {obs.shape} doesn't match obs_space.shape: {obs_space.shape}."
        assert obs.dtype == obs_space.dtype, f"obs.dtype: {obs.dtype} doesn't match obs_space.dtype: {obs_space.dtype}."
    assert obs in obs_space


def _sample_and_check(env: gym.Env, obs_space: gym.Space) -> bool:
    """Sample from env and check return value is of valid type."""
    act = env.action_space.sample()
    obs, rew, done, info = env.step(act)
    _check_obs(obs, obs_space)
    assert isinstance(rew, float), f"rew is type `{type(rew)}` instead of type `float`."
    assert isinstance(done, bool), f"done is type `{type(done)}` instead of type `bool`."
    assert isinstance(info, dict), f"info is type `{type(info)}` instead of type `dict`."
    return done


def test_rollout_schema(
    env: gym.Env,
    steps_after_done: int = 10,
    max_steps: int = 10000,
) -> None:
    """Check custom environments have correct types on `step` and `reset`.

    Args:
        env: The environment to test.
        steps_after_done: The number of steps to take after `done` is True, the nominal
            episode termination. This is an abuse of the Gym API, but we would like the
            environments to handle this case gracefully.
        max_steps: Test fails if we do not get `done` after this many timesteps.

    Raises:
        AssertionError if test fails.
    """
    obs_space = env.observation_space
    obs = env.reset()
    _check_obs(obs, obs_space)

    for _ in range(max_steps):
        done = _sample_and_check(env, obs_space)
        if done:
            break

    assert done is True, "did not get to end of episode"

    for _ in range(steps_after_done):
        _sample_and_check(env, obs_space)


def test_premature_step(env: gym.Env, skip_fn, raises_fn) -> None:
    """Test that you must call reset() before calling step().

    Example usage in pytest:
        test_premature_step(env, skip_fn=pytest.skip, raises_fn=pytest.raises)

    Args:
        env: The environment to test.
        skip_fn: called when the environment is incompatible with the test.
        raises_fn: Context manager to check exception is thrown.

    Raises:
        AssertionError if test fails.
    """
    if hasattr(env, "sim") and hasattr(env, "model"):  # pragma: no cover
        # We can't use isinstance since importing mujoco_py will fail on
        # machines without MuJoCo installed
        skip_fn("MuJoCo environments cannot perform this check.")

    act = env.action_space.sample()
    with raises_fn(Exception):  # need to call env.reset() first
        env.step(act)


def test_render(env: gym.Env, raises_fn) -> None:
    """Test that render() supports the modes declared.

    Example usage in pytest:
        test_render(env, raises_fn=pytest.raises)

    Args:
        env: The environment to test.
        raises_fn: Context manager to check NotImplementedError is thrown when
            environment metadata indicates modes are supported.

    Raises:
        AssertionError: if test fails. This occurs if:
            (a) `env.render(mode=mode)` fails for any mode declared supported
            in `env.metadata["render.modes"]`; (b) env.render() *succeeds* when
            `env.metadata["render.modes"]` is empty; (c) `env.render(mode="rgb_array")`
            returns different values at the same time step.
    """
    env.reset()  # make sure environment is in consistent state

    render_modes = env.metadata["render.modes"]
    if not render_modes:
        # No modes supported -- render() should fail.
        with raises_fn(NotImplementedError):
            env.render()
    else:
        for mode in render_modes:
            env.render(mode=mode)

        # WARNING(adam): there seems to be a memory leak with Gym 0.17.3
        # & MuJoCoPy 1.50.1.68. `MujocoEnv.close()` does not call `finish()`
        # on the viewer (commented out) so the resources are not released.
        # For now this is OK, but may bite if we end up testing a lot of
        # MuJoCo environments.
        is_mujoco = isinstance(env.unwrapped, mujoco_env.MujocoEnv)
        if "rgb_array" in render_modes and not is_mujoco:
            # Render should not change without calling `step()`.
            # MuJoCo rendering fails this check, ignore -- not much we can do.
            resa = env.render(mode="rgb_array")
            resb = env.render(mode="rgb_array")
            assert np.allclose(resa, resb)


class CountingEnv(gym.Env):
    """At timestep `t` of each episode, has `t == obs == reward / 10`.

    Episodes finish after `episode_length` calls to `step()`, or equivalently
    `episode_length` actions. For example, if we have `episode_length=5`,
    then an episode has the following observations and rewards:

    ```
    obs = [0, 1, 2, 3, 4, 5]
    rews = [10, 20, 30, 40, 50]
    ```
    """

    def __init__(self, episode_length: int = 5):
        """Initialize a CountingEnv.

        Params:
            episode_length: The number of actions before each episode ends.
        """
        assert episode_length >= 1
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=())
        self.action_space = gym.spaces.Box(low=0, high=np.inf, shape=())
        self.episode_length = episode_length
        self.timestep = None

    def reset(self):
        """Reset method for CountingEnv."""
        t, self.timestep = 0, 1
        return np.array(t, dtype=self.observation_space.dtype)

    def step(self, action):
        """Step method for CountingEnv."""
        if self.timestep is None:  # pragma: no cover
            raise RuntimeError("Need to reset before first step().")
        if np.array(action) not in self.action_space:  # pragma: no cover
            raise ValueError(f"Invalid action {action}")
        if self.timestep > self.episode_length:  # pragma: no cover
            raise ValueError("Should reset env. Episode is over.")

        t, self.timestep = self.timestep, self.timestep + 1
        obs = np.array(t, dtype=self.observation_space.dtype)
        rew = t * 10.0
        done = t == self.episode_length
        return obs, rew, done, {}
