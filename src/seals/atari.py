"""Adaptation of Atari environments for specification learning algorithms."""

from typing import Dict, Iterable, List, Optional

import gym

from seals.util import AutoResetWrapper, MaskScoreWrapper, get_gym_max_episode_steps

SCORE_REGIONS: Dict[str, List[Dict[str, int]]] = {
    "BeamRider": [
        dict(x0=5, x1=20, y0=45, y1=120),
        dict(x0=28, x1=40, y0=15, y1=40),
    ],
    "Breakout": [dict(x0=0, x1=16, y0=35, y1=80)],
    "Enduro": [
        dict(x0=163, x1=173, y0=55, y1=110),
        dict(x0=177, x1=188, y0=68, y1=107),
    ],
    "Pong": [dict(x0=0, x1=24, y0=0, y1=160)],
    "Qbert": [dict(x0=6, x1=15, y0=33, y1=71)],
    "Seaquest": [dict(x0=7, x1=19, y0=80, y1=110)],
    "SpaceInvaders": [dict(x0=10, x1=20, y0=0, y1=160)],
}


def _get_score_region(atari_env_id: str) -> Optional[List[Dict[str, int]]]:
    basename = atari_env_id.split("/")[-1].split("-")[0]
    return SCORE_REGIONS.get(basename)


def make_atari_env(atari_env_id: str) -> gym.Env:
    """Fixed-length, masked-score variant of a given Atari environment."""
    score_region = _get_score_region(atari_env_id)
    if score_region is None:
        raise ValueError(
            "Requested environment not supported. "
            + "See https://github.com/HumanCompatibleAI/seals/issues/61.",
        )

    env = MaskScoreWrapper(gym.make(atari_env_id), score_region)
    return AutoResetWrapper(env)


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
    score_regions_available = _get_score_region(gym_spec.id) is not None
    return (
        is_atari
        and _not_ram_or_det(gym_spec.id)
        and (v5_and_plain or v4_and_no_frameskip)
        and score_regions_available
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
            entry_point="seals.atari:make_atari_env",
            max_episode_steps=get_gym_max_episode_steps(gym_spec.id),
            kwargs=dict(atari_env_id=gym_spec.id),
        )
