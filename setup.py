"""setup.py for seals project."""

from typing import TYPE_CHECKING

from setuptools import find_packages, setup  # type:ignore

if TYPE_CHECKING:
    from setuptools_scm.version import ScmVersion


def get_version(version: "ScmVersion") -> str:
    """Generates the version string for the package.

    This function replaces the default version format used by setuptools_scm
    to allow development builds to be versioned using the git commit hash
    instead of the number of commits since the last release, which leads to
    duplicate version identifiers when using multiple branches
    (see https://github.com/HumanCompatibleAI/imitation/issues/500).
    The version has the following format:
    {version}[.dev{build}]
    where build is the shortened commit hash converted to base 10.

    Args:
        version: The version object given by setuptools_scm, calculated
            from the git repository.

    Returns:
        The formatted version string to use for the package.
    """
    # We import setuptools_scm here because it is only installed after the module
    # is loaded and the setup function is called.
    from setuptools_scm import version as scm_version

    if version.node:
        # By default node corresponds to the short commit hash when using git,
        # plus a "g" prefix. We remove the "g" prefix from the commit hash which
        # is added by setuptools_scm by default ("g" for git vs. mercurial etc.)
        # because letters are not valid for version identifiers in PEP 440.
        # We also convert from hexadecimal to base 10 for the same reason.
        version.node = str(int(version.node.lstrip("g"), 16))
    if version.exact:
        # an exact version is when the current commit is tagged with a version.
        return version.format_with("{tag}")
    else:
        # the current commit is not tagged with a version, so we guess
        # what the "next" version will be (this can be disabled but is the
        # default behavior of setuptools_scm so it has been left in).
        return version.format_next_version(
            scm_version.guess_next_version,
            fmt="{guessed}.dev{node}",
        )


def get_local_version(version: "ScmVersion", time_format="%Y%m%d") -> str:
    """Generates the local version string for the package.

    By default, when commits are made on top of a release version, setuptools_scm
    sets the version to be {version}.dev{distance}+{node} where {distance} is the number
    of commits since the last release and {node} is the short commit hash.
    This function replaces the default version format used by setuptools_scm
    so that committed changes away from a release version are not considered
    local versions but dev versions instead (by using the format
    {version}.dev{node} instead. This is so that we can push test releases
    to TestPyPI (it does not accept local versions).
    Local versions are still present if there are uncommitted changes (if the tree
    is dirty), in which case the current date is added to the version.

    Args:
        version: The version object given by setuptools_scm, calculated
            from the git repository.
        time_format: The format to use for the date.

    Returns:
        The formatted local version string to use for the package.
    """
    return version.format_choice(
        "",
        "+d{time:{time_format}}",
        time_format=time_format,
    )


def get_readme() -> str:
    """Retrieve content from README."""
    with open("README.md", "r") as f:
        return f.read()


ATARI_REQUIRE = [
    "opencv-python",
    "ale-py~=0.8.1",
    "pillow",
    "autorom[accept-rom-license]~=0.4.2",
    "shimmy[atari] >=0.1.0,<1.0",
]
TESTS_REQUIRE = [
    "black",
    "coverage~=4.5.4",
    "codecov",
    "codespell",
    "darglint>=1.5.6",
    "flake8",
    "flake8-blind-except",
    "flake8-builtins",
    "flake8-commas",
    "flake8-debugger",
    "flake8-docstrings",
    "flake8-isort",
    "isort",
    "matplotlib",
    "mypy",
    "pydocstyle",
    "pytest",
    "pytest-cov",
    "pytest-xdist",
    "pytype",
    "stable-baselines3>=0.9.0",
    "setuptools_scm~=7.0.5",
    "gymnasium[classic-control,mujoco]",
    *ATARI_REQUIRE,
]
DOCS_REQUIRE = [
    "sphinx",
    "sphinx-autodoc-typehints>=1.21.5",
    "sphinx-rtd-theme",
]


setup(
    name="seals",
    use_scm_version={"local_scheme": get_local_version, "version_scheme": get_version},
    description="Suite of Environments for Algorithms that Learn Specifications",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    author="Center for Human-Compatible AI",
    python_requires=">=3.8.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"seals": ["py.typed"]},
    install_requires=["gymnasium", "numpy"],
    tests_require=TESTS_REQUIRE,
    extras_require={
        # recommended packages for development
        "dev": ["ipdb", "jupyter", *TESTS_REQUIRE, *DOCS_REQUIRE],
        "docs": DOCS_REQUIRE,
        "test": TESTS_REQUIRE,
        "mujoco": ["gymnasium[mujoco]"],
        "atari": ATARI_REQUIRE,
    },
    url="https://github.com/HumanCompatibleAI/benchmark-environments",
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
