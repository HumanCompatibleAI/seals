"""Simple diagnostic environments."""

import gym

gym.register(
    id="seals/RiskyPath-v0",
    entry_point="seals.diagnostics.risky_path:RiskyPathEnv",
    max_episode_steps=5,
)

gym.register(
    id="seals/InitShiftLearner-v0",
    entry_point="seals.diagnostics.init_shift:InitShiftLearnerEnv",
    max_episode_steps=3,
)

gym.register(
    id="seals/InitShiftExpert-v0",
    entry_point="seals.diagnostics.init_shift:InitShiftExpertEnv",
    max_episode_steps=3,
)
