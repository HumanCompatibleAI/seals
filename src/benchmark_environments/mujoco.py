"""Adaptation of MuJoCo environments for IRL.
"""

import gym
from gym.envs.mujoco import half_cheetah_v3, ant_v3, hopper_v3, humanoid_v3, swimmer_v3, walker2d_v3

def include_position_in_observation(cls):
    old_init = cls.__init__

    def new_init(*args, **kwargs):
        kwargs['exclude_current_positions_from_observation'] = False
        old_init(*args, **kwargs)

    cls.__init__ = new_init
    return cls

def no_early_termination(cls):
    cls.done = False
    return cls

@include_position_in_observation
@no_early_termination
class HalfCheetahEnv(half_cheetah_v3.HalfCheetahEnv):
    pass

@include_position_in_observation
@no_early_termination
class AntEnv(ant_v3.AntEnv):
    pass

@include_position_in_observation
@no_early_termination
class HopperEnv(hopper_v3.HopperEnv):
    pass

@include_position_in_observation
@no_early_termination
class HumanoidEnv(humanoid_v3.HumanoidEnv):
    pass

@include_position_in_observation
@no_early_termination
class SwimmerEnv(swimmer_v3.SwimmerEnv):
    pass

@include_position_in_observation
@no_early_termination
class Walker2dEnv(walker2d_v3.Walker2dEnv):
    pass
