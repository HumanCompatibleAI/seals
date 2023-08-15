"""Test the base_envs module.

Note base_envs is also tested indirectly via smoke tests in `test_envs`,
so the tests in this file focus on features unique to classes in `base_envs`.
"""

import gymnasium as gym
import numpy as np
import pytest

from seals import base_envs
from seals.testing import envs


class NewEnv(base_envs.TabularModelMDP):
    """Test the TabularModelMDP class."""

    def __init__(self):
        """Build environment."""
        np.random.seed(0)
        nS = 3
        nA = 2
        transition_matrix = np.random.random((nS, nA, nS))
        transition_matrix /= transition_matrix.sum(axis=2)[:, :, None]
        reward_matrix = np.random.random((nS,))
        super().__init__(
            transition_matrix=transition_matrix,
            reward_matrix=reward_matrix,
        )


def test_base_envs():
    """Test parts of base_envs not covered elsewhere."""
    env = NewEnv()

    assert np.all(np.eye(3) == env.feature_matrix)

    envs.test_premature_step(env, skip_fn=pytest.skip, raises_fn=pytest.raises)

    env.reset(seed=0)
    assert env.n_actions_taken == 0
    env.step(env.action_space.sample())
    assert env.n_actions_taken == 1
    env.step(env.action_space.sample())
    assert env.n_actions_taken == 2

    new_state = env.state_space.sample()
    env.state = new_state
    assert env.state == new_state

    bad_state = "not a state"
    with pytest.raises(ValueError, match=r".*not in.*"):
        env.state = bad_state  # type: ignore

    with pytest.raises(NotImplementedError, match=r"Options not supported.*"):
        env.reset(options={"option": "value"})


def test_tabular_env_validation():
    """Test input validation for base_envs.TabularModelEnv."""
    with pytest.raises(ValueError, match=r"Malformed transition_matrix.*"):
        base_envs.TabularModelMDP(
            transition_matrix=np.zeros((3, 1, 4)),
            reward_matrix=np.zeros((3,)),
        )
    with pytest.raises(ValueError, match=r"initial_state_dist has multiple.*"):
        base_envs.TabularModelMDP(
            transition_matrix=np.zeros((3, 1, 3)),
            reward_matrix=np.zeros((3,)),
            initial_state_dist=np.zeros((3, 4)),
        )
    with pytest.raises(ValueError, match=r"transition_matrix and initial_state_dist.*"):
        base_envs.TabularModelMDP(
            transition_matrix=np.zeros((3, 1, 3)),
            reward_matrix=np.zeros((3,)),
            initial_state_dist=np.zeros((2)),
        )
    with pytest.raises(ValueError, match=r"transition_matrix and reward_matrix.*"):
        base_envs.TabularModelMDP(
            transition_matrix=np.zeros((4, 1, 4)),
            reward_matrix=np.zeros((3,)),
        )
    with pytest.raises(ValueError, match=r"transition_matrix and observation_matrix.*"):
        base_envs.TabularModelPOMDP(
            transition_matrix=np.zeros((3, 1, 3)),
            reward_matrix=np.zeros((3,)),
            observation_matrix=np.zeros((4, 3)),
        )

    env = base_envs.TabularModelMDP(
        transition_matrix=np.zeros((3, 1, 3)),
        reward_matrix=np.zeros((3,)),
    )
    env.reset(seed=0)
    with pytest.raises(ValueError, match=r".*not in.*"):
        env.step(4)


def test_expose_pomdp_state_wrapper():
    """Test the ExposePOMDPStateWrapper class."""
    env = NewEnv()
    wrapped_env = base_envs.ExposePOMDPStateWrapper(env)

    assert wrapped_env.observation_space == env.state_space
    state, _ = wrapped_env.reset(seed=0)
    assert state == env.state
    assert state in env.state_space

    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = wrapped_env.step(action)
    assert next_state == env.state
    assert next_state in env.state_space


def test_tabular_pompd_obs_space_int():
    """Test the TabularModelPOMDP class with an integer observation space."""
    env = base_envs.TabularModelPOMDP(
        transition_matrix=np.zeros(
            (3, 1, 3),
        ),
        reward_matrix=np.zeros((3,)),
        observation_matrix=np.zeros((3, 3), dtype=np.int64),
    )
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.dtype == np.int64
