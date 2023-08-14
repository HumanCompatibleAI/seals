"""Adaptation of Atari environments for specification learning algorithms."""

from typing import Dict, Iterable, Optional

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec

from seals.util import (
    AutoResetWrapper,
    BoxRegion,
    MaskedRegionSpecifier,
    MaskScoreWrapper,
    get_gym_max_episode_steps,
)

SCORE_REGIONS: Dict[str, MaskedRegionSpecifier] = {
    "BeamRider": [
        BoxRegion(x=(5, 20), y=(45, 120)),
        BoxRegion(x=(28, 40), y=(15, 40)),
    ],
    "Breakout": [BoxRegion(x=(0, 16), y=(35, 80))],
    "Enduro": [
        BoxRegion(x=(163, 173), y=(55, 110)),
        BoxRegion(x=(177, 188), y=(68, 107)),
    ],
    "Pong": [BoxRegion(x=(0, 24), y=(0, 160))],
    "Qbert": [BoxRegion(x=(6, 15), y=(33, 71))],
    "Seaquest": [BoxRegion(x=(7, 19), y=(80, 110))],
    "SpaceInvaders": [BoxRegion(x=(10, 20), y=(0, 160))],
}


def _get_score_region(atari_env_id: str) -> Optional[MaskedRegionSpecifier]:
    basename = atari_env_id.split("/")[-1].split("-")[0]
    basename = basename.replace("NoFrameskip", "")
    return SCORE_REGIONS.get(basename)


def make_atari_env(atari_env_id: str, masked: bool, *args, **kwargs) -> gym.Env:
    """Fixed-length, optionally masked-score variant of a given Atari environment."""
    env: gym.Env = AutoResetWrapper(gym.make(atari_env_id, *args, **kwargs))

    if masked:
        score_region = _get_score_region(atari_env_id)
        if score_region is None:
            raise ValueError(
                "Requested environment does not yet support masking. "
                "See https://github.com/HumanCompatibleAI/seals/issues/61.",
            )
        env = MaskScoreWrapper(env, score_region)

    return env


def _not_ram_or_det(env_id: str) -> bool:
    """Checks a gym Atari environment isn't deterministic or using RAM observations."""
    slash_separated = env_id.split("/")
    # environment name should look like "ALE/Amidar-v5" or "Amidar-ramNoFrameskip-v4"
    assert len(slash_separated) in (1, 2)
    after_slash = slash_separated[-1]
    hyphen_separated = after_slash.split("-")
    assert len(hyphen_separated) > 1
    not_ram = "ram" not in hyphen_separated[1]
    not_deterministic = "Deterministic" not in env_id
    return not_ram and not_deterministic


def _supported_atari_env(gym_spec: EnvSpec) -> bool:
    """Checks if a gym Atari environment is one of the ones we will support."""
    is_atari = gym_spec.entry_point == "shimmy.atari_env:AtariEnv"
    v5_and_plain = gym_spec.id.endswith("-v5") and "NoFrameskip" not in gym_spec.id
    v4_and_no_frameskip = gym_spec.id.endswith("-v4") and "NoFrameskip" in gym_spec.id
    return (
        is_atari
        and _not_ram_or_det(gym_spec.id)
        and (v5_and_plain or v4_and_no_frameskip)
    )


def _seals_name(gym_spec: EnvSpec, masked: bool) -> str:
    """Makes a Gym ID for an Atari environment in the seals namespace."""
    slash_separated = gym_spec.id.split("/")
    name = "seals/" + slash_separated[-1]

    if not masked:
        last_hyphen_idx = name.rfind("-v")
        name = name[:last_hyphen_idx] + "-Unmasked" + name[last_hyphen_idx:]
    return name


def register_atari_envs(
    gym_atari_env_specs: Iterable[EnvSpec],
) -> None:
    """Register masked and unmasked wrapped gym Atari environments."""

    def register_gym(masked):
        gym.register(
            id=_seals_name(gym_spec, masked=masked),
            entry_point="seals.atari:make_atari_env",
            max_episode_steps=get_gym_max_episode_steps(gym_spec.id),
            kwargs=dict(atari_env_id=gym_spec.id, masked=masked),
        )

    for gym_spec in gym_atari_env_specs:
        register_gym(masked=False)
        if _get_score_region(gym_spec.id) is not None:
            register_gym(masked=True)
