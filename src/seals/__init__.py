"""Benchmark environments for reward modeling and imitation."""

import gym

from seals import util
import seals.diagnostics  # noqa: F401
from seals.version import VERSION as __version__  # noqa: F401

# Classic control

gym.register(
    id="seals/CartPole-v0",
    entry_point="seals.classic_control:FixedHorizonCartPole",
    max_episode_steps=500,
)

gym.register(
    id="seals/MountainCar-v0",
    entry_point="seals.classic_control:mountain_car",
    max_episode_steps=200,
)

# MuJoCo

for env_base in ["Ant", "HalfCheetah", "Hopper", "Humanoid", "Swimmer", "Walker2d"]:
    gym.register(
        id=f"seals/{env_base}-v0",
        entry_point=f"seals.mujoco:{env_base}Env",
        max_episode_steps=util.get_gym_max_episode_steps(f"{env_base}-v3"),
    )

# Atari


def _not_ram_or_det(env_id):
    slash_separated = env_id.split("/")
    # environment name should look like "ALE/Amidar-v5" or "Amidar-ramNoFrameskip-v4"
    assert len(slash_separated) in [1, 2]
    after_slash = slash_separated[-1]
    hyphen_separated = after_slash.split("-")
    assert len(hyphen_separated) > 1
    not_ram = not ("ram" in hyphen_separated[1])
    not_deterministic = not ("Deterministic" in env_id)
    return not_ram and not_deterministic


def _supported_atari_env(gym_spec):
    is_atari = gym_spec.entry_point == "gym.envs.atari:AtariEnv"
    v5_and_plain = gym_spec.id.endswith("-v5") and not ("NoFrameskip" in gym_spec.id)
    v4_and_no_frameskip = gym_spec.id.endswith("-v4") and "NoFrameskip" in gym_spec.id
    return (
        is_atari
        and _not_ram_or_det(gym_spec.id)
        and (v5_and_plain or v4_and_no_frameskip)
    )


# not a filter so that it doesn't update during the for loop below.
GYM_ATARI_ENV_SPECS = [
    gym_spec for gym_spec in gym.envs.registry.all() if _supported_atari_env(gym_spec)
]


def _seals_name(gym_spec):
    slash_separated = gym_spec.id.split("/")
    return "seals/" + slash_separated[-1]


for gym_spec in GYM_ATARI_ENV_SPECS:
    gym.register(
        id=_seals_name(gym_spec),
        entry_point="seals.atari:fixed_length_atari",
        max_episode_steps=util.get_gym_max_episode_steps(gym_spec.id),
        kwargs=dict(atari_env_id=gym_spec.id),
    )
