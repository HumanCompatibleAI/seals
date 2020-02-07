"""Configuration for pytest."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--expensive",
        action="store_true",
        dest="expensive",
        default=False,
        help="enable expensive tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--expensive"):
        return
    skip_expensive = pytest.mark.skip(reason="needs --expensive option to run")
    for item in items:
        if "expensive" in item.keywords:
            item.add_marker(skip_expensive)
