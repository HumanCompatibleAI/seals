"""setup.py for benchmark_environments project."""

from setuptools import find_packages, setup
import src.benchmark_environments  # pytype: disable=import-error

TF_VERSION = ">=1.15.0,<2.0"
TESTS_REQUIRE = [
    # remove pin once https://github.com/nedbat/coveragepy/issues/881 fixed
    "black",
    "coverage==4.5.4",
    "codecov",
    "codespell",
    "isort",
    "flake8",
    "flake8-blind-except",
    "flake8-builtins",
    "flake8-debugger",
    "flake8-docstrings",
    "flake8-isort",
    "pydocstyle",
    "pytest",
    "pytest-cov",
    "pytest-shard",
    "pytest-xdist",
    "pytype",
    "sphinx",
    "sphinxcontrib-napoleon",
    "stable-baselines>=2.8.0",
    "tensorflow>=1.8.0,<2.0.0",
]

setup(
    name="benchmark_environments",
    version=src.benchmark_environments.__version__,
    description=("Implementation of modern IRL and imitation learning algorithms."),
    author="Center for Human-Compatible AI and Google",
    python_requires=">=3.7.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={
        "benchmark_environments": ["py.typed", "envs/examples/airl_envs/assets/*.xml"],
    },
    install_requires=["gym[mujoco]"],
    tests_require=TESTS_REQUIRE,
    extras_require={
        # recommended packages for development
        "dev": ["ipdb", "jupyter", *TESTS_REQUIRE],
        "test": TESTS_REQUIRE,
    },
    url="https://github.com/HumanCompatibleAI/benchmark-environments",
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)
