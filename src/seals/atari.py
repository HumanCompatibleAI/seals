"""Adaptation of Atari environments for specification learning algorithms."""

import gym

from seals.util import AutoResetWrapper


def fixed_length_atari(atari_env_id):
    """Fixed-length variant of a given atari environment."""
    return AutoResetWrapper(gym.make(atari_env_id))
