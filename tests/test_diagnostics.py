"""Test the `diagnostics.*` environments."""

import pytest

from seals.diagnostics import init_shift


def test_init_shift_validation():
    """Test input validation for init_shift.InitShiftEnv."""
    for invalid_state in [-1, 7, 8, 100]:
        with pytest.raises(ValueError, match=r"Initial state.*"):
            init_shift.InitShiftEnv(initial_state=invalid_state)
