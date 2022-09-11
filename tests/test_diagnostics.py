"""Test the `diagnostics.*` environments."""

import numpy as np
import pytest

from seals.diagnostics import cliff_world, init_shift


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
