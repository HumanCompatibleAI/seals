"""Test the base_envs module.

Note base_envs is also tested indirectly via smoke tests in `test_envs`,
so the tests in this file focus on features unique to classes in `base_envs`.
"""

import numpy as np
import pytest

from seals import base_envs
from seals.testing import envs


def test_base_envs():
    """Test parts of base_envs not covered elsewhere."""

    class NewEnv(base_envs.TabularModelMDP):
        def __init__(self):
            nS = 3
            nA = 2
            transition_matrix = np.random.rand(nS, nA, nS)
            transition_matrix /= transition_matrix.sum(axis=2)[:, :, None]
            reward_matrix = np.random.rand(nS)
            super().__init__(
                transition_matrix=transition_matrix,
                reward_matrix=reward_matrix,
            )

    env = NewEnv()

    assert np.all(np.eye(3) == env.feature_matrix)

    envs.test_premature_step(env, skip_fn=pytest.skip, raises_fn=pytest.raises)

    env.reset()
    assert env.n_actions_taken == 0
    env.step(env.action_space.sample())
    assert env.n_actions_taken == 1
    env.step(env.action_space.sample())
    assert env.n_actions_taken == 2


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

    env = base_envs.TabularModelMDP(
        transition_matrix=np.zeros((3, 1, 3)),
        reward_matrix=np.zeros((3,)),
    )
    env.reset()
    with pytest.raises(ValueError, match=r".*not in.*"):
        env.step(4)
