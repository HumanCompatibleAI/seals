"""Environment testing for generalization in continuous spaces."""

from gymnasium import spaces
import numpy as np

from seals import base_envs


class ParabolaEnv(base_envs.ResettableMDP):
    """Environment to mimic parabola curves.

    This environment tests algorithms' ability to learn in continuous
    action spaces, a challenge for Q-learning methods in particular.
    The goal is to mimic the path of a parabola p(x) = A*x**2 + B*x +
    C, where A, B and C are constants sampled uniformly from [-1, 1]
    at the start of the episode.
    """

    def __init__(self, x_step: float = 0.05, bounds: float = 5):
        """Construct environment.

        Args:
            x_step: x position difference between timesteps.
            bounds: limits coordinates, useful for keeping rewards in
                a small bounded range.
        """
        super().__init__()
        self._x_step = x_step
        self._bounds = bounds

        state_high = np.array([bounds, bounds, 1.0, 1.0, 1.0])
        state_low = (-1) * state_high

        self.state_space = spaces.Box(low=state_low, high=state_high)
        self.action_space = spaces.Box(low=(-2) * bounds, high=2 * bounds, shape=())

    def terminal(self, state: int, n_actions_taken: int) -> bool:
        """Always returns False."""
        return False

    def initial_state(self) -> np.ndarray:
        """Get state by sampling a random parabola."""
        a, b, c = -1 + 2 * self.np_random.random((3,))
        x, y = 0, c
        return np.array([x, y, a, b, c], dtype=self.state_space.dtype)

    def reward(self, state: np.ndarray, action: int, new_state: np.ndarray) -> float:
        """Negative squared vertical distance from parabola."""
        x, y, a, b, c = state
        target_y = a * x**2 + b * x + c
        return (-1) * (y - target_y) ** 2

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        """Update x according to x_step and y according to action."""
        x, y, a, b, c = state
        next_x = np.clip(x + self._x_step, -self._bounds, self._bounds)
        next_y = np.clip(y + action, -self._bounds, self._bounds)
        return np.array([next_x, next_y, a, b, c], dtype=self.state_space.dtype)
