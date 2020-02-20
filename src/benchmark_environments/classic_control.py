"""Adaptation of classic Gym environments for IRL."""

import gym.wrappers

from benchmark_environments import util

register = util.curried_gym_register_as_decorator(__name__)


@register("CartPole-v0")
def cart_pole():
    """Fixed-length variant of CartPole-v1.

    If the agent fails (IE pole falls over or CartPole exits the screen),
    all rewards become 0 for the duration of the episode.

    Done is always returned on timestep 500 only.
    """
    env = util.make_env_no_wrappers("CartPole-v1")
    env = util.FixedRewardAfterDoneWrapper(env)
    env = gym.wrappers.TimeLimit(env, 500)
    return env


@register("MountainCar-v0")
def mountain_car():
    """Fixed-length variant of MountainCar-v0.

    In the event of early episode completion (IE, the car reaches the
    goal), we enter an absorbing state that repeats the final observation
    and returns reward 0.
    Done is always returned on timestep 200 only.
    """
    env = util.make_env_no_wrappers("MountainCar-v0")
    env = util.AbsorbAfterDoneWrapper(env)
    env = gym.wrappers.TimeLimit(env, 200)
    return env
