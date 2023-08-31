"""Large gridworld with random agent and goal position."""

from gymnasium import spaces
import numpy as np

from seals import base_envs, util


class ProcGoalEnv(base_envs.ResettableMDP):
    """Large gridworld with random agent and goal position.

    In this task, the agent starts at a random position in a large
    grid, and must navigate to a goal randomly placed in a
    neighborhood around the agent.  The observation is a 4-dimensional
    vector containing the (x,y) coordinates of the agent and the goal.
    The reward at each timestep is the negative Manhattan distance
    between the two positions.  With a large enough grid, generalizing
    is necessary to achieve good performance, since most initial
    states will be unseen.
    """

    def __init__(self, bounds: int = 100, distance: int = 10):
        """Constructs environment.

        Args:
            bounds: the absolute values of the coordinates of the initial agent
                position are bounded by `bounds`. Increasing the value might make
                generalization harder.
            distance: initial distance between agent and goal.
        """
        super().__init__()
        self._bounds = bounds
        self._distance = distance
        self.state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
        self.action_space = spaces.Discrete(5)

    def terminal(self, state: np.ndarray, n_actions_taken: int) -> bool:
        """Always returns False."""
        return False

    def initial_state(self) -> np.ndarray:
        """Samples random agent position and random goal."""
        pos = self.np_random.integers(low=-self._bounds, high=self._bounds, size=(2,))

        x_dist = self.np_random.integers(self._distance)
        y_dist = self._distance - x_dist
        random_signs = 2 * self.np_random.integers(2, size=2) - 1
        goal = pos + random_signs * (x_dist, y_dist)

        return np.concatenate([pos, goal]).astype(self.observation_space.dtype)

    def reward(self, state: np.ndarray, action: int, new_state: np.ndarray) -> float:
        """Negative L1 distance to goal."""
        return (-1) * np.sum(np.abs(state[2:] - state[:2]))

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        """Returns next state according to grid."""
        pos, goal = state[:2], state[2:]
        next_pos = util.grid_transition_fn(pos, action)
        return np.concatenate([next_pos, goal])
