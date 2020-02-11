from benchmark_environments.testing.envs import CountingEnv
from benchmark_environments.util import AutoResetWrapper


def test_auto_reset_wrapper(episode_length=3, n_steps=100):
    env = AutoResetWrapper(CountingEnv(episode_length=episode_length))

    obs = env.reset()
    assert obs == 0

    for t in range(1, n_steps + 1):
        act = env.action_space.sample()
        obs, rew, done, info = env.step(act)
        expected_obs = t % episode_length

        assert obs == expected_obs
        assert done is False

        # End of episode
        if expected_obs == 0:
            assert info.get("terminal_observation", None) == episode_length
            assert rew == episode_length * 10
        else:
            assert "terminal_obs" not in info
            assert rew == expected_obs * 10
