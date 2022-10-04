"""Test the `diagnostics.*` environments."""

import numpy as np
import pytest

from seals.diagnostics import cliff_world, init_shift, random_trans


def test_init_shift_validation():
    """Test input validation for init_shift.InitShiftEnv."""
    for invalid_state in [-1, 7, 8, 100]:
        with pytest.raises(ValueError, match=r"Initial state.*"):
            init_shift.InitShiftEnv(initial_state=invalid_state)


def test_cliff_world_draw_value_vec():
    """Smoke test for cliff_world.CliffWorldEnv.draw_value_vec()."""
    env = cliff_world.CliffWorldEnv(
        width=7,
        height=4,
        horizon=9,
        use_xy_obs=False,
    )
    D = np.zeros(env.state_dim)
    env.draw_value_vec(D)


def test_random_transition_env_init():
    """Test that RandomTransitionEnv initializes correctly."""
    random_trans.RandomTransitionEnv(
        n_states=3,
        n_actions=2,
        branch_factor=3,
        horizon=10,
        random_obs=False,
    )
    random_trans.RandomTransitionEnv(
        n_states=3,
        n_actions=2,
        branch_factor=3,
        horizon=10,
        random_obs=True,
    )
    random_trans.RandomTransitionEnv(
        n_states=3,
        n_actions=2,
        branch_factor=3,
        horizon=10,
        random_obs=True,
        obs_dim=10,
    )
    with pytest.raises(ValueError, match="obs_dim must be None if random_obs is False"):
        random_trans.RandomTransitionEnv(
            n_states=3,
            n_actions=2,
            branch_factor=3,
            horizon=10,
            random_obs=False,
            obs_dim=3,
        )


def test_make_random_matrices_no_explicit_rng():
    """Test that random matrix maker static methods work without an explicit RNG."""
    random_trans.RandomTransitionEnv.make_random_trans_mat(3, 2, 3)
    random_trans.RandomTransitionEnv.make_random_state_dist(3, 3)
    random_trans.RandomTransitionEnv.make_obs_mat(3, True, 3)
    with pytest.raises(ValueError, match="obs_dim must be set if random_obs is True"):
        random_trans.RandomTransitionEnv.make_obs_mat(3, True)
    random_trans.RandomTransitionEnv.make_obs_mat(3, False)
    with pytest.raises(ValueError, match="obs_dim must be None if random_obs is False"):
        random_trans.RandomTransitionEnv.make_obs_mat(3, False, 3)
