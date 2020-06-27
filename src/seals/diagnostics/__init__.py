"""Simple diagnostic environments."""

import gym

gym.register(
    id="seals/RiskyPath-v0",
    entry_point="seals.diagnostics.risky_path:RiskyPathEnv",
    max_episode_steps=5,
)

gym.register(
    id="seals/LargestSum-v0",
    entry_point="seals.diagnostics.largest_sum:LargestSumEnv",
    max_episode_steps=1,
)
