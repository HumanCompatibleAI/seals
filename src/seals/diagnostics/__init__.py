"""Simple diagnostic environments."""

import gym

gym.register(
    id="seals/RiskyPath-v0",
    entry_point="seals.diagnostics.risky_path:RiskyPathEnv",
    max_episode_steps=5,
)

gym.register(
    id="seals/ProcGoal-v0",
    entry_point="seals.diagnostics.proc_goal:ProcGoalEnv",
    max_episode_steps=20,
)
