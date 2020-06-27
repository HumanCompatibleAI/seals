"""Simple diagnostic environments."""

import gym

gym.register(
    id="seals/RiskyPath-v0",
    entry_point="seals.diagnostics.risky_path:RiskyPathEnv",
    max_episode_steps=5,
)

gym.register(
    id="seals/NoisyObs-v0",
    entry_point="seals.diagnostics.noisy_obs:NoisyObsEnv",
    max_episode_steps=15,
)
