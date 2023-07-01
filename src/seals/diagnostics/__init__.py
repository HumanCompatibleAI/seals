"""Simple diagnostic environments."""

import gymnasium as gym

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


def register_cliff_world(suffix, kwargs):
    """Register a CliffWorld with the given suffix and keyword arguments."""
    gym.register(
        f"seals/CliffWorld{suffix}-v0",
        entry_point="seals.diagnostics.cliff_world:CliffWorldEnv",
        kwargs=kwargs,
    )


def register_all_cliff_worlds():
    """Register all CliffWorld environments."""
    for width, height, horizon in [(7, 4, 9), (15, 6, 18), (100, 20, 110)]:
        for use_xy in [False, True]:
            use_xy_str = "XY" if use_xy else ""
            register_cliff_world(
                f"{width}x{height}{use_xy_str}",
                kwargs={
                    "width": width,
                    "height": height,
                    "use_xy_obs": use_xy,
                    "horizon": horizon,
                },
            )


register_all_cliff_worlds()

# These parameter choices are somewhat arbitrary.
# We anticipate most users will want to construct RandomTransitionEnv directly.
gym.register(
    "seals/Random-v0",
    entry_point="seals.diagnostics.random_trans:RandomTransitionEnv",
    kwargs={
        "n_states": 16,
        "n_actions": 3,
        "branch_factor": 2,
        "horizon": 20,
        "random_obs": True,
        "obs_dim": 5,
        "generator_seed": 42,
    },
)
