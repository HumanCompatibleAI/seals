[![CircleCI](https://circleci.com/gh/HumanCompatibleAI/seals.svg?style=svg)](https://circleci.com/gh/HumanCompatibleAI/seals) [![codecov](https://codecov.io/gh/HumanCompatibleAI/seals/branch/master/graph/badge.svg)](https://codecov.io/gh/HumanCompatibleAI/seals)

**Status**: alpha, pre-release.

The [CHAI](https://humancompatible.ai/) Suite of Environments for Algorithms that Learn 
Specifications (SEALS) is a toolkit for evaluating specification learning algorithms, such as
reward or imitation learning. The environments are compatible with [Gym](https://github.com/openai/gym),
but are designed to test algorithms that learn from user data, without requiring a procedurally
specified reward function.

This is currently a work-in progress. While you are welcome to use the repository, we may make
breaking changes at any time without prior notice. We intend for it to eventually contain:

  - A suite of diagnostic tasks for reward learning.
  - Wrappers around common RL benchmark environments that help to avoid common pitfalls in
    benchmarking (e.g. by masking visible score counters in Gym Atari tasks).
  - New challenge tasks for specification learning algorithms.
 
You may also be interested in our sister project [imitation](https://github.com/humancompatibleai/imitation/),
providing implementations of a variety of imitation and reward learning algorithms.

# Quickstart

To install the latest release from PyPI, run:
 
```bash
pip install seals
```

All SEALS environments are available in the Gym registry. Simply import it and then use as you
would with your usual RL or specification learning algroithm:

```python
import gym
import seals

env = gym.make('seals/CartPole-v0')
```

We make releases periodically, but if you wish to use the latest versino of the code, you can
install directly from Git master:

```bash
pip install git+https://github.com/HumanCompatibleAI/seals.git
```

# Contributing

For development, clone the source code and create a virtual environment for this project:

```bash
git clone git@github.com:HumanCompatibleAI/seals.git
cd seals
./ci/build_venv.sh
pip install -e .[dev]  # install extra tools useful for development
```

## Code style

We follow a PEP8 code style with line length 88, and typically follow the [Google Code Style Guide](http://google.github.io/styleguide/pyguide.html),
but defer to PEP8 where they conflict. We use the `black` autoformatter to avoid arguing over formatting.
Docstrings follow the Google docstring convention defined [here](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings),
with an extensive example in the [Sphinx docs](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

All PRs must pass linting via the `ci/code_checks.sh` script. It is convenient to install this as a commit hook:

```bash
ln -s ../../ci/code_checks.sh .git/hooks/pre-commit
```

## Tests

We use [pytest](https://docs.pytest.org/en/latest/) for unit tests
and [codecov](http://codecov.io/) for code coverage.
We also use [pytype](https://github.com/google/pytype) for type checking.

## Workflow

Trivial changes (e.g. typo fixes) may be made directly by maintainers. Any non-trivial changes
must be proposed in a PR and approved by at least one maintainer. PRs must pass the continuous 
integration tests (CircleCI linting, type checking, unit tests and CodeCov) to be merged.

It is often helpful to open an issue before proposing a PR, to allow for discussion of the design
before coding commences.
