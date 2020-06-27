"""Simple diagnostic environments."""

import gym

gym.register(
    id="seals/RiskyPath-v0",
    entry_point="seals.diagnostics.risky_path:RiskyPathEnv",
    max_episode_steps=5,
)

gym.register(
    id="seals/EarlyTermPos-v0",
    entry_point="seals.diagnostics.early_term:EarlyTermPosEnv",
    max_episode_steps=10,
)

gym.register(
    id="seals/EarlyTermNeg-v0",
    entry_point="seals.diagnostics.early_term:EarlyTermNegEnv",
    max_episode_steps=10,
)
