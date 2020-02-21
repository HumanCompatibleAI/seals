"""Tests for wrapper classes."""

import pytest

from benchmark_environments import util
from benchmark_environments.testing import envs


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


@pytest.mark.parametrize("episode_end_reward", [-10, 0, 3.3])
def test_ep_fixed_reward_after_done_wrapper(
    episode_end_reward, episode_length=3, n_steps=100,
):
    """Check that EpisodeEndRewardWrapper returns the correct rewards."""
    env = util.FixedRewardAfterDoneWrapper(
        envs.CountingEnv(episode_length=episode_length), episode_end_reward,
    )
    env.reset()

    for t in range(1, n_steps + 1):
        act = env.action_space.sample()
        _, rew, done, _ = env.step(act)
        assert done is False
        if t > episode_length:
            expected_rew = episode_end_reward
        else:
            expected_rew = t * 10.0
        assert rew == expected_rew


def test_absorb_repeat_custom_state(
    absorb_reward=-4, absorb_obs=-3.0, episode_length=6, n_steps=100, n_manual_reset=3,
):
    """Check that AbsorbAfterDoneWrapper returns custom state."""
    env = envs.CountingEnv(episode_length=episode_length)
    env = util.AbsorbAfterDoneWrapper(
        env, absorb_reward=absorb_reward, absorb_obs=absorb_obs,
    )

    obs_list = []

    for r in range(n_manual_reset):
        env.reset()
        for t in range(1, n_steps + 1):
            act = env.action_space.sample()
            obs, rew, done, _ = env.step(act)
            obs_list.append(obs)  # DEBUG
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
