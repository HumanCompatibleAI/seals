"""Test `seals.util`."""

import collections

import gymnasium as gym
import numpy as np
import pytest

from seals import GYM_ATARI_ENV_SPECS, util


def test_mask_score_wrapper_enforces_spec():
    """Test that MaskScoreWrapper enforces the spec."""
    atari_env = gym.make(GYM_ATARI_ENV_SPECS[0].id)
    desired_error_message = 'Invalid region: "x" and "y" must be increasing.'
    with pytest.raises(ValueError, match=desired_error_message):
        util.MaskScoreWrapper(atari_env, [util.BoxRegion(x=(0, 1), y=(1, 0))])
    with pytest.raises(ValueError, match=desired_error_message):
        util.MaskScoreWrapper(atari_env, [util.BoxRegion(x=(1, 0), y=(0, 1))])


def test_sample_distribution():
    """Test util.sample_distribution."""
    distr_size = 5
    distr = np.random.random((distr_size,))
    distr /= distr.sum()

    n_samples = 1000
    rng = np.random.default_rng(0)
    sample_count = collections.Counter(
        util.sample_distribution(distr, rng) for _ in range(n_samples)
    )

    empirical_distr = np.array([sample_count[i] for i in range(distr_size)]) / n_samples

    # Empirical distribution matches real distribution
    l1_err = np.sum(np.abs(empirical_distr - distr))
    assert l1_err < 0.1

    # Same seed gives same samples
    assert all(
        util.sample_distribution(distr, random=np.random.default_rng(seed))
        == util.sample_distribution(distr, random=np.random.default_rng(seed))
        for seed in range(20)
    )


def test_one_hot_encoding():
    """Test util.one_hot_encoding."""
    Case = collections.namedtuple("Case", ["pos", "size", "encoding"])

    cases = [
        Case(pos=0, size=1, encoding=np.array([1.0])),
        Case(pos=1, size=5, encoding=np.array([0.0, 1.0, 0.0, 0.0, 0.0])),
        Case(pos=3, size=4, encoding=np.array([0.0, 0.0, 0.0, 1.0])),
        Case(pos=2, size=3, encoding=np.array([0.0, 0.0, 1.0])),
        Case(pos=2, size=6, encoding=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])),
    ]

    assert all(
        np.all(util.one_hot_encoding(pos, size) == encoding)
        for pos, size, encoding in cases
    )
