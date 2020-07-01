"""Simple diagnostic environments."""

import gym

gym.register(
    id="seals/Branching-v0",
    entry_point="seals.diagnostics.branching:BranchingEnv",
    max_episode_steps=11,
)

gym.register(
    id="seals/EarlyTermNeg-v0",
    entry_point="seals.diagnostics.early_term:EarlyTermNegEnv",
    max_episode_steps=10,
)

gym.register(
    id="seals/EarlyTermPos-v0",
    entry_point="seals.diagnostics.early_term:EarlyTermPosEnv",
    max_episode_steps=10,
)

gym.register(
    id="seals/InitShiftTrain-v0",
    entry_point="seals.diagnostics.init_shift:InitShiftTrainEnv",
    max_episode_steps=3,
)

gym.register(
    id="seals/InitShiftTest-v0",
    entry_point="seals.diagnostics.init_shift:InitShiftTestEnv",
    max_episode_steps=3,
)

gym.register(
    id="seals/LargestSum-v0",
    entry_point="seals.diagnostics.largest_sum:LargestSumEnv",
    max_episode_steps=1,
)

gym.register(
    id="seals/NoisyObs-v0",
    entry_point="seals.diagnostics.noisy_obs:NoisyObsEnv",
    max_episode_steps=15,
)

gym.register(
    id="seals/Parabola-v0",
    entry_point="seals.diagnostics.parabola:ParabolaEnv",
    max_episode_steps=20,
)

gym.register(
    id="seals/ProcGoal-v0",
    entry_point="seals.diagnostics.proc_goal:ProcGoalEnv",
    max_episode_steps=20,
)

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
