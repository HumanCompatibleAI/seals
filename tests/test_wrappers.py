"""Tests for wrapper classes."""

from benchmark_environments.testing.envs import CountingEnv
from benchmark_environments.util import AutoResetWrapper


def test_auto_reset_wrapper(episode_length=3, n_steps=100, n_manual_reset=2):
    """Check that AutoResetWrapper returns correct values from step and reset.

    Also check that calls to .reset() do not interfere with automatic resets.
    """
    env = AutoResetWrapper(CountingEnv(episode_length=episode_length))

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
