"""Adaptation of Atari environments for specification learning algorithms."""

from typing import Iterable

import gym

from seals.util import AutoResetWrapper, get_gym_max_episode_steps


def fixed_length_atari(atari_env_id: str) -> gym.Env:
    """Fixed-length variant of a given Atari environment."""
    return AutoResetWrapper(gym.make(atari_env_id))


def _not_ram_or_det(env_id: str) -> bool:
    """Checks a gym Atari environment isn't deterministic or using RAM observations."""
    slash_separated = env_id.split("/")
    # environment name should look like "ALE/Amidar-v5" or "Amidar-ramNoFrameskip-v4"
    assert len(slash_separated) in (1, 2)
    after_slash = slash_separated[-1]
    hyphen_separated = after_slash.split("-")
    assert len(hyphen_separated) > 1
    not_ram = not ("ram" in hyphen_separated[1])
    not_deterministic = not ("Deterministic" in env_id)
    return not_ram and not_deterministic


def _supported_atari_env(gym_spec: gym.envs.registration.EnvSpec) -> bool:
    """Checks if a gym Atari environment is one of the ones we will support."""
    is_atari = gym_spec.entry_point == "gym.envs.atari:AtariEnv"
    v5_and_plain = gym_spec.id.endswith("-v5") and not ("NoFrameskip" in gym_spec.id)
    v4_and_no_frameskip = gym_spec.id.endswith("-v4") and "NoFrameskip" in gym_spec.id
    return (
        is_atari
        and _not_ram_or_det(gym_spec.id)
        and (v5_and_plain or v4_and_no_frameskip)
    )


def _seals_name(gym_spec: gym.envs.registration.EnvSpec) -> str:
    """Makes a Gym ID for an Atari environment in the seals namespace."""
    slash_separated = gym_spec.id.split("/")
    return "seals/" + slash_separated[-1]


def register_atari_envs(
    gym_atari_env_specs: Iterable[gym.envs.registration.EnvSpec],
) -> None:
    """Register wrapped gym Atari environments."""
    for gym_spec in gym_atari_env_specs:
        gym.register(
            id=_seals_name(gym_spec),
            entry_point="seals.atari:fixed_length_atari",
            max_episode_steps=get_gym_max_episode_steps(gym_spec.id),
            kwargs=dict(atari_env_id=gym_spec.id),
        )
