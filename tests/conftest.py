"""Configuration for pytest."""

import pytest

pytest.register_assert_rewrite("seals.testing")


def pytest_addoption(parser):
    """Add --expensive option."""
    parser.addoption(
        "--expensive",
        action="store_true",
        dest="expensive",
        default=False,
        help="enable expensive tests",
    )


def pytest_collection_modifyitems(config, items):
    """Make expensive tests be skipped without an --expensive flag."""
    if config.getoption("--expensive"):  # pragma: no cover
        return
    skip_expensive = pytest.mark.skip(reason="needs --expensive option to run")
    for item in items:
        if "expensive" in item.keywords:
            item.add_marker(skip_expensive)
