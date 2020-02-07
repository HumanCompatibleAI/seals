"""General testing for all envs."""


import re

import gym
import numpy as np
import pytest

import benchmark_environments.mujoco  # noqa: F401 Import required for env registration

ENV_NAMES = [
    env_spec.id
    for env_spec in gym.envs.registration.registry.all()
    if env_spec.id.startswith("benchmark_environments/")
]

DETERMINISTIC_ENVS = []


def make_env_fixture(skip_fn):
    def f(env_name):
        env = None
        try:
            env = gym.make(env_name)
            yield env
        except gym.error.DependencyNotInstalled as err:  # pragma: no cover
            if err.args[0].find("mujoco_py") != -1:
                skip_fn("Requires `mujoco_py`, which isn't installed.")
            else:
                raise
        finally:
            if env is not None:
                env.close()

    return f


env = pytest.fixture(make_env_fixture(skip_fn=pytest.skip))


@pytest.mark.parametrize("env_name", ENV_NAMES)
class TestEnvs:
    """Battery of simple tests for environments."""

    def test_seed(self, env, env_name):
        """Tests environment seeding.

    If non-deterministic, different seeds should produce different transitions.
    If deterministic, should be invariant to seed.
    """

        def get_rollout(env, actions):
            step_results = [(env.reset(), None, None, None)]
            for act in actions:
                step_results.append(env.step(act))
            return step_results

        def assert_equal_rollout(rollout_a, rollout_b):
            for step_a, step_b in zip(rollout_a, rollout_b):
                ob_a, rew_a, done_a, info_a = step_a
                ob_b, rew_b, done_b, info_b = step_b
                np.testing.assert_equal(ob_a, ob_b)
                assert rew_a == rew_b
                assert done_a == done_b
                np.testing.assert_equal(info_a, info_b)

        def has_same_observations(first_rollout, second_rollout):
            first_obs_list = [step[0] for step in first_rollout]
            second_obs_list = [step[0] for step in second_rollout]
            return all(
                np.all(first_obs == second_obs)
                for first_obs, second_obs in zip(first_obs_list, second_obs_list)
            )

        is_deterministic = any(
            re.match(env_pattern, env_name) for env_pattern in DETERMINISTIC_ENVS
        )

        env.action_space.seed(0)
        actions = [env.action_space.sample() for _ in range(10)]

        # With the same seed, should always get the same result
        seeds = env.seed(42)
        assert isinstance(seeds, list)
        assert len(seeds) > 0
        rollout_a = get_rollout(env, actions)

        env.seed(42)
        rollout_b = get_rollout(env, actions)

        if is_deterministic:
            assert_equal_rollout(rollout_a, rollout_b)

        # For non-deterministic environments, if we try enough seeds we should
        # eventually get a different result. For deterministic environments, all
        # seeds will produce the same starting state.
        def seeded_rollout_equals_rollout_a(seed):
            env.seed(seed)
            new_rollout = get_rollout(env, actions)
            return has_same_observations(new_rollout, rollout_a)

        same_obs = all(seeded_rollout_equals_rollout_a(seed) for seed in range(20))
        assert same_obs == is_deterministic

    def test_premature_step_required(self, env):
        """Test that you must call reset() before calling step()."""
        if hasattr(env, "sim") and hasattr(env, "model"):  # pragma: no cover
            # We can't use isinstance since importing mujoco_py will fail on
            # machines without MuJoCo installed
            pytest.skip("MuJoCo environments cannot perform this check.")

        act = env.action_space.sample()
        with pytest.raises(Exception):  # need to call env.reset() first
            env.step(act)

    def test_model_based_env(self, env):
        """Smoke test for each of the ModelBasedEnv methods with type checks."""
        if not hasattr(env, "state_space"):  # pragma: no cover
            pytest.skip("This test is only for subclasses of ModelBasedEnv.")

        state = env.initial_state()
        assert env.state_space.contains(state)

        action = env.action_space.sample()
        new_state = env.transition(state, action)
        assert env.state_space.contains(new_state)

        reward = env.reward(state, action, new_state)
        assert isinstance(reward, float)

        done = env.terminal(state, 0)
        assert isinstance(done, bool)

        obs = env.obs_from_state(state)
        assert env.observation_space.contains(obs)
        next_obs = env.obs_from_state(new_state)
        assert env.observation_space.contains(next_obs)

    def test_rollout_schema(self, env):
        """Check custom environments have correct types on `step` and `reset`."""
        obs_space = env.observation_space
        obs = env.reset()
        assert obs in obs_space

        for _ in range(4):
            act = env.action_space.sample()
            obs, rew, done, info = env.step(act)
            assert obs in obs_space
            assert isinstance(rew, float)
            assert isinstance(done, bool)
            assert isinstance(info, dict)
