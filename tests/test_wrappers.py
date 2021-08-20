"""Tests for wrapper classes."""

import numpy as np
import pytest

from seals import util
from seals.testing import envs


def test_auto_reset_wrapper(episode_length=3, n_steps=100, n_manual_reset=2):
    """Check that AutoResetWrapper returns correct values from step and reset.

    Also check that calls to .reset() do not interfere with automatic resets.
    """
    env = util.AutoResetWrapper(envs.CountingEnv(episode_length=episode_length))

    for _ in range(n_manual_reset):
        obs = env.reset()
        assert obs == 0

        for t in range(1, n_steps + 1):
            act = env.action_space.sample()
            obs, rew, done, info = env.step(act)
            expected_obs = t % episode_length

            assert obs == expected_obs
            assert done is False

            if expected_obs == 0:  # End of episode
                assert info.get("terminal_observation", None) == episode_length
                assert rew == episode_length * 10
            else:
                assert "terminal_observation" not in info
                assert rew == expected_obs * 10


def test_absorb_repeat_custom_state(
    absorb_reward=-4,
    absorb_obs=-3.0,
    episode_length=6,
    n_steps=100,
    n_manual_reset=3,
):
    """Check that AbsorbAfterDoneWrapper returns custom state and reward."""
    env = envs.CountingEnv(episode_length=episode_length)
    env = util.AbsorbAfterDoneWrapper(
        env,
        absorb_reward=absorb_reward,
        absorb_obs=absorb_obs,
    )

    for r in range(n_manual_reset):
        env.reset()
        for t in range(1, n_steps + 1):
            act = env.action_space.sample()
            obs, rew, done, _ = env.step(act)
            assert done is False
            if t > episode_length:
                expected_obs = absorb_obs
                expected_rew = absorb_reward
            else:
                expected_obs = t
                expected_rew = t * 10.0
            assert obs == expected_obs
            assert rew == expected_rew


def test_absorb_repeat_final_state(episode_length=6, n_steps=100, n_manual_reset=3):
    """Check that AbsorbAfterDoneWrapper can repeat final state."""
    env = envs.CountingEnv(episode_length=episode_length)
    env = util.AbsorbAfterDoneWrapper(env, absorb_reward=-1, absorb_obs=None)

    for _ in range(n_manual_reset):
        env.reset()
        for t in range(1, n_steps + 1):
            act = env.action_space.sample()
            obs, rew, done, _ = env.step(act)
            assert done is False
            if t > episode_length:
                expected_obs = episode_length
                expected_rew = -1
            else:
                expected_obs = t
                expected_rew = t * 10.0
            assert obs == expected_obs
            assert rew == expected_rew


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
def test_obs_cast(dtype: np.dtype, episode_length: int = 5):
    """Check obs_cast observations are of specified dtype and not mangled.

    Test uses CountingEnv with small integers, which can be represented in
    all the specified dtypes without any loss of precision.
    """
    env = envs.CountingEnv(episode_length=episode_length)
    env = util.ObsCastWrapper(env, dtype)

    obs = env.reset()
    assert obs.dtype == dtype
    assert obs == 0
    for t in range(1, episode_length + 1):
        act = env.action_space.sample()
        obs, rew, done, _ = env.step(act)
        assert done == (t == episode_length)
        assert obs.dtype == dtype
        assert obs == t
        assert rew == t * 10.0
