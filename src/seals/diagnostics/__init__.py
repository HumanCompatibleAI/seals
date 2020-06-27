"""Simple diagnostic environments."""

import gym

gym.register(
    id="seals/RiskyPath-v0",
    entry_point="seals.diagnostics.risky_path:RiskyPathEnv",
    max_episode_steps=5,
)

gym.register(
    id="seals/Sort-v0",
    entry_point="seals.diagnostics.sort:SortEnv",
    max_episode_steps=6,
)
