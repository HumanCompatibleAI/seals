"""Simple diagnostic environments."""

import gym

envs_info = [
    ("Branching-v0", "branching:BranchingEnv", 11),
    ("EarlyTermNeg-v0", "early_term:EarlyTermNegEnv", 10),
    ("EarlyTermPos-v0", "early_term:EarlyTermPosEnv", 10),
    ("InitShiftTrain-v0", "init_shift:InitShiftTrainEnv", 3),
    ("InitShiftTest-v0", "init_shift:InitShiftTestEnv", 3),
    ("LargestSum-v0", "largest_sum:LargestSumEnv", 1),
    ("NoisyObs-v0", "noisy_obs:NoisyObsEnv", 15),
    ("Parabola-v0", "parabola:ParabolaEnv", 20),
    ("ProcGoal-v0", "proc_goal:ProcGoalEnv", 20),
    ("RiskyPath-v0", "risky_path:RiskyPathEnv", 5),
    ("Sort-v0", "sort:SortEnv", 6),
]

for name, source, horizon in envs_info:
    gym.register(
        id=f"seals/{name}",
        entry_point=f"seals.diagnostics.{source}",
        max_episode_steps=horizon,
    )
