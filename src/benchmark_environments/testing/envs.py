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
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import gym
import numpy as np

Rollout = Sequence[Tuple[Any, Optional[float], bool, Mapping[str, Any]]]


def make_env_fixture(
    skip_fn: Callable[[str], None]
) -> Callable[[str], Iterator[gym.Env]]:
    """Creates a fixture function, calling `skip_fn` when dependencies are missing.

    For example, in `pytest`, one would use:
        env = pytest.fixture(make_env_fixture(skip_fn=pytest.skip))
    Then any method with an `env` parameter will receive the created environment, with
    the `env_name` parameter automatically passed to the fixture.

    In `unittest`, one would use:
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
    ret = [(env.reset(), None, False, {})]
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

    # For non-deterministic environments, if we try enough seeds we should
    # eventually get a different result. For deterministic environments, all
    # seeds should produce the same starting state.
    def new_seed_equals_orig_rollout(seed):
        env.seed(seed)
        new_rollout = get_rollout(env, actions)
        return has_same_observations(new_rollout, rollout_a)

    is_deterministic = matches_list(env_name, deterministic_envs)
    same_obs = all(new_seed_equals_orig_rollout(seed) for seed in range(20))
    assert same_obs == is_deterministic


def test_rollout_schema(env: gym.Env, num_steps: int = 4) -> None:
    """Check custom environments have correct types on `step` and `reset`.

    Raises:
        AssertionError if test fails.
    """
    obs_space = env.observation_space
    obs = env.reset()
    assert obs in obs_space

    for _ in range(num_steps):
        act = env.action_space.sample()
        obs, rew, done, info = env.step(act)
        assert obs in obs_space
        assert isinstance(rew, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)


def test_premature_step(env: gym.Env, skip_fn, raises_fn) -> None:
    """Test that you must call reset() before calling step().

    Example usage in pytest:
        test_premature_step(env, skip_fn=pytest.skip, exception_fn=pytest.raises)

    Args:
        env: The environment to test.
        skip_fn: called when the environment is incompatible with the test.
        exception_fn: Context manager to check exception is thrown.

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
