[![CircleCI](https://circleci.com/gh/HumanCompatibleAI/benchmark-environments.svg?style=svg)](https://circleci.com/gh/HumanCompatibleAI/benchmark-environments) [![codecov](https://codecov.io/gh/HumanCompatibleAI/benchmark-environments/branch/master/graph/badge.svg)](https://codecov.io/gh/HumanCompatibleAI/benchmark-environments) 

**Status**: alpha, pre-release.

benchmark-environments is a suite of benchmarks for imitation-learning and
reward-learning algorithms. It is currently a work-in-progress, but we intend for it to eventually
contain a suite of diagnostic tasks for reward-learning, wrappers around common RL benchmark
environments that help to avoid common pitfalls in benchmarking (e.g. by masking visible score
counters in Gym Atari tasks), and new challenge tasks for imitation- and reward-learning. This
benchmark suite is a complement to our  [https://github.com/humancompatibleai/imitation/](imitation)
package of baseline algorithms for imitation learning.

# Usage

To install the latest release from PyPI, run:
 
```bash
pip install benchmark-environments
```

To install from Git master:

```
pip install git+https://github.com/HumanCompatibleAI/benchmark-environments.git
```

# Contributing

For development, clone the source code and create a virtual environment for this project:

 ```
git clone git@github.com:HumanCompatibleAI/benchmark-environments.git
cd benchmark-environments
./ci/build_venv.sh
pip install -e .[dev]  # install extra tools useful for development
```

## Code style

We follow a PEP8 code style with line width 88, and typically follow the [Google Code Style Guide](http://google.github.io/styleguide/pyguide.html),
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
