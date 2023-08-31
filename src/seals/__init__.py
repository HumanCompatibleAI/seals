"""Benchmark environments for reward modeling and imitation."""

from importlib import metadata

import gymnasium as gym

from seals import atari, util
import seals.diagnostics  # noqa: F401

try:
    __version__ = metadata.version("seals")
except metadata.PackageNotFoundError:  # pragma: no cover
    # package is not installed
    pass

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
        id=f"seals/{env_base}-v1",
        entry_point=f"seals.mujoco:{env_base}Env",
        max_episode_steps=util.get_gym_max_episode_steps(f"{env_base}-v4"),
    )

# Atari

GYM_ATARI_ENV_SPECS = list(filter(atari._supported_atari_env, gym.registry.values()))
atari.register_atari_envs(GYM_ATARI_ENV_SPECS)
