"""Tests for wrapper classes."""

import numpy as np
import pytest

from seals import util
from seals.testing import envs


def test_auto_reset_wrapper_pad(episode_length=3, n_steps=100, n_manual_reset=2):
    """Check that AutoResetWrapper returns correct values from step and reset.

    AutoResetWrapper that pads trajectory with an extra transition containing the
    terminal observations.
    Also check that calls to .reset() do not interfere with automatic resets.
    Due to the padding, the number of steps counted inside the environment and the
    number of steps performed outside the environment, i.e., the number of actions
    performed, will differ. This test checks that this difference is consistent.
    """
    env = util.AutoResetWrapper(
        envs.CountingEnv(episode_length=episode_length),
        discard_terminal_observation=False,
    )

    for _ in range(n_manual_reset):
        obs, info = env.reset()
        assert obs == 0

        # We count the number of episodes, so we can sanity check the padding.
        num_episodes = 0
        next_episode_end = episode_length
        for t in range(1, n_steps + 1):
            act = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step(act)

            # AutoResetWrapper overrides all terminated and truncated signals.
            assert terminated is False
            assert truncated is False

            if t == next_episode_end:
                # Unlike the AutoResetWrapper that discards terminal observations,
                # here the final observation is returned directly, and is not stored
                # in the info dict.
                # Due to padding, for every episode the final observation is offset from
                # the outer step by one.
                assert obs == (t - num_episodes) / (num_episodes + 1)
                assert rew == episode_length * 10
            if t == next_episode_end + 1:
                num_episodes += 1
                # Because the final step returned the final observation, the initial
                # obs of the next episode is returned in this additional step.
                assert obs == 0
                # Consequently, the next episode end is one step later, so it is
                # episode_length steps from now.
                next_episode_end = t + episode_length

                # Reward of the 'reset transition' is fixed to be 0.
                assert rew == 0

                # Sanity check padding. Padding should be 1 for each past episode.
                assert (
                    next_episode_end
                    == (num_episodes + 1) * episode_length + num_episodes
                )


def test_auto_reset_wrapper_discard(episode_length=3, n_steps=100, n_manual_reset=2):
    """Check that AutoResetWrapper returns correct values from step and reset.

    Tests for AutoResetWrapper that discards terminal observations.
    Also check that calls to .reset() do not interfere with automatic resets.
    """
    env = util.AutoResetWrapper(
        envs.CountingEnv(episode_length=episode_length),
        discard_terminal_observation=True,
    )

    for _ in range(n_manual_reset):
        obs, info = env.reset()
        assert obs == 0

        for t in range(1, n_steps + 1):
            act = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step(act)
            expected_obs = t % episode_length

            assert obs == expected_obs
            assert terminated is False
            assert truncated is False

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
            obs, rew, terminated, truncated, _ = env.step(act)
            assert terminated is False
            assert truncated is False
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
            obs, rew, terminated, truncated, _ = env.step(act)
            assert terminated is False
            assert truncated is False
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
    env = util.ObsCastWrapper(
        envs.CountingEnv(episode_length=episode_length),
        dtype,
    )

    obs, _ = env.reset()
    assert obs.dtype == dtype
    assert obs == 0
    for t in range(1, episode_length + 1):
        act = env.action_space.sample()
        obs, rew, terminated, truncated, _ = env.step(act)
        assert terminated == (t == episode_length)
        assert truncated is False
        assert obs.dtype == dtype
        assert obs == t
        assert rew == t * 10.0
