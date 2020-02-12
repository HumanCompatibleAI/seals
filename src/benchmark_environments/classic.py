"""Adaptation of classic Gym environments for IRL."""
from gym.wrappers import TimeLimit

from benchmark_environments import util

register = util.curried_gym_register_as_decorator(__name__)


@register("CartPole-v0")
def cart_pole():
    """Fixed-length variant of CartPole-v0."""
    env = util.make_env_no_wrappers("CartPole-v0")
    env = util.EpisodeEndRewardWrapper(env, -1)
    env = util.AutoResetWrapper(env)

    max_steps = util.get_gym_max_episode_steps("CartPole-v0")
    if max_steps is not None:
        env = TimeLimit(env, max_steps)
    return env


@register("MountainCar-v0")
def mountain_car():
    """Fixed-length variant of MountainCar-v0."""
    env = util.make_env_no_wrappers("MountainCar-v0")
    env = util.EpisodeEndRewardWrapper(env, 1)
    env = util.AutoResetWrapper(env)

    max_steps = util.get_gym_max_episode_steps("CartPole-v0")
    if max_steps is not None:
        env = TimeLimit(env, max_steps)
    return env
