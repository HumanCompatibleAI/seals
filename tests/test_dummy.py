"""Dummy file to make CI not complain. Delete once we have real tests!"""

import pytest


@pytest.mark.parametrize("i", range(20))
def test_dummy(i):
    """Dummy test to make CI not complain."""
    # TODO(adam): remove me once we have real tests!
    assert True
