"""Adaptation of classic Gym environments for specification learning algorithms."""

import warnings

import numpy as np
from gymnasium import spaces
from gymnasium.envs import classic_control


from seals import util


class FixedHorizonCartPole(classic_control.CartPoleEnv):
    """Fixed-length variant of CartPole-v1.

    Reward is 1.0 whenever the CartPole is an "ok" state (i.e., the pole is upright
    and the cart is on the screen). Otherwise, reward is 0.0.

    Terminated is always False.
    By default, this environment is wrapped in 'TimeLimit' with max steps 500,
    Truncation is handled by that.
    """

    def __init__(self):
        """Builds FixedHorizonCartPole, modifying observation_space from gym parent."""
        super().__init__()

        high = [
            np.finfo(np.float32).max,  # x axis
            np.finfo(np.float32).max,  # x velocity
            np.pi,  # theta in radians
            np.finfo(np.float32).max,  # theta velocity
        ]
        high = np.array(high)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def reset(self, seed=None, options={}):
        """Reset for FixedHorizonCartPole."""
        observation, info = super().reset(seed=seed, options=options)
        return observation.astype(np.float32), info

    def step(self, action):
        """Step function for FixedHorizonCartPole."""
        with warnings.catch_warnings():
            # Filter out CartPoleEnv warning for calling step() beyond done=True.
            warnings.filterwarnings("ignore", ".*You are calling.*")
            super().step(action)

        self.state = list(self.state)
        x, _, theta, _ = self.state

        # Normalize theta to [-pi, pi] range.
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        self.state[2] = theta

        state_ok = bool(
            abs(x) < self.x_threshold and abs(theta) < self.theta_threshold_radians,
        )

        rew = 1.0 if state_ok else 0.0
        return np.array(self.state, dtype=np.float32), rew, False, False, {}


def mountain_car():
    """Fixed-length variant of MountainCar-v0.

    In the event of early episode completion (i.e., the car reaches the
    goal), we enter an absorbing state that repeats the final observation
    and returns reward 0.

    Done is always returned on timestep 200 only.
    """
    env = util.make_env_no_wrappers("MountainCar-v0")
    env = util.ObsCastWrapper(env, dtype=np.float32)
    env = util.AbsorbAfterDoneWrapper(env)
    return env
