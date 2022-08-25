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

ATARI_ENV_NAMES = [
    "Adventure",
    "AirRaid",
    "Alien",
    "Amidar",
    "Assault",
    "Asterix",
    "Asteroids",
    "Atlantis",
    "BankHeist",
    "BattleZone",
    "BeamRider",
    "Berzerk",
    "Bowling",
    "Boxing",
    "Breakout",
    "Carnival",
    "Centipede",
    "ChopperCommand",
    "CrazyClimber",
    "Defender",
    "DemonAttack",
    "DoubleDunk",
    "ElevatorAction",
    "Enduro",
    "FishingDerby",
    "Freeway",
    "Frostbite",
    "Gopher",
    "Gravitar",
    "Hero",
    "IceHockey",
    "Jamesbond",
    "JourneyEscape",
    "Kangaroo",
    "Krull",
    "KungFuMaster",
    "MontezumaRevenge",
    "MsPacman",
    "NameThisGame",
    "Phoenix",
    "Pitfall",
    "Pong",
    "Pooyan",
    "PrivateEye",
    "Qbert",
    "Riverraid",
    "RoadRunner",
    "Robotank",
    "Seaquest",
    "Skiing",
    "Solaris",
    "SpaceInvaders",
    "StarGunner",
    "Tennis",
    "TimePilot",
    "Tutankham",
    "UpNDown",
    "Venture",
    "VideoPinball",
    "WizardOfWor",
    "YarsRevenge",
    "Zaxxon",
]

for env_name in ATARI_ENV_NAMES:
    for frameskip in [True, False]:
        seals_name = "seals/" + env_name + ("-v5" if frameskip else "NoFrameskip-v4")
        func_name = env_name.lower() + ("_v5" if frameskip else "_noframeskip")
        gym_name = (
            ("ALE/" + env_name + "-v5") if frameskip else (env_name + "NoFrameskip-v4")
        )
        gym.register(
            id=seals_name,
            entry_point=f"seals.atari:{func_name}",
            max_episode_steps=util.get_gym_max_episode_steps(gym_name),
        )
